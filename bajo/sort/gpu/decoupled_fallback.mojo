"""Radix sort using Decoupled Fallback.

from Smith, Levien and Owens (SPAA 2025)
https://github.com/b0nes164/Decoupled-Fallback-Paper
"""
from std.atomic import Atomic, Ordering
from max.gpu import (
    thread_idx,
    block_idx,
    lane_id,
    WARP_SIZE,
)
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.memory import AddressSpace
from max.gpu.primitives import warp
from max.gpu.primitives import block
from max.gpu.sync import barrier
from std.math import ceildiv, max
from std.memory import stack_allocation
from std.sys.info import bit_width_of

from .utils import circular_shift, warp_level_multi_split
from .radix_common import (
    init_radix_workspace,
    global_histogram,
    scan_local_histogram,
    scatter_radix_tile,
    enqueue_radix_passes,
)
from .validation import require_sort_capacity

comptime ordering = Ordering.RELAXED

# Two state bits distinguish missing, reduction, and inclusive descriptors.
# Each word carries 16 count bits, retaining the full UInt32 range.
comptime SPLIT_READY = UInt32(0x40000000)
comptime SPLIT_INCLUSIVE = UInt32(0x80000000)
comptime SPLIT_STATE_MASK = UInt32(0xC0000000)
comptime SPLIT_VALUE_MASK = UInt32(0xFFFF)
comptime FALLBACK_SPIN_LIMIT = 4


@always_inline
def publish_split_histogram[
    BLOCK_SIZE: Int, STATE: UInt32
](
    pass_hist: Pointer[UInt32, MutAnyOrigin],
    histogram: Pointer[UInt32, _, address_space=AddressSpace.SHARED],
    row: Int,
):
    # Adjacent lanes publish the halves with the same atomic instruction.
    # Do not replace this with two sequential stores by one lane: a reader
    # seeing a partial inclusive update must wait for the matching half.
    comptime assert WARP_SIZE >= 2 and WARP_SIZE % 2 == 0
    for word in range(thread_idx.x, 512, BLOCK_SIZE):
        var count = histogram[unsafe_offset=word >> 1]
        var encoded = (
            (count >> UInt32(16 * (word & 1))) & SPLIT_VALUE_MASK
        ) | STATE
        var destination = pass_hist.unsafe_offset(row * 512 + word)
        comptime if STATE == SPLIT_READY:
            # Install only a missing reduction. All helpers compute the same
            # immutable value; a late owner must preserve an inclusive state.
            var expected = UInt32(0)
            _ = Atomic.compare_exchange[
                success_ordering=Ordering.RELAXED,
                failure_ordering=Ordering.RELAXED,
            ](destination, expected, encoded)
        else:
            # Only the owner promotes to inclusive. One paired atomic store
            # avoids a CAS retry leaving half a prefix unpublished when that
            # owner's subgroup is subsequently starved.
            comptime assert STATE == SPLIT_INCLUSIVE
            Atomic.store[ordering=Ordering.RELAXED](destination, encoded)


@always_inline
def fallback_lookback[
    local_origin: MutOrigin,
    scratch_origin: MutOrigin,
    //,
    keys_dtype: DType,
    BLOCK_SIZE: Int,
    KEYS_PER_THREAD: Int,
    SUPPRESS_PUBLICATION: Bool = False,
](
    keys: Pointer[Scalar[keys_dtype], MutAnyOrigin],
    pass_hist: Pointer[UInt32, MutAnyOrigin],
    local_offsets: Pointer[
        UInt32, local_origin, address_space=AddressSpace.SHARED
    ],
    scratch: Pointer[UInt32, scratch_origin, address_space=AddressSpace.SHARED],
    local_reduction: UInt32,
    partition_index: Int,
    radix_shift: UInt32,
):
    comptime RADIX = 256
    comptime PART_SIZE = BLOCK_SIZE * KEYS_PER_THREAD
    comptime assert BLOCK_SIZE >= RADIX and BLOCK_SIZE % WARP_SIZE == 0
    var tid = thread_idx.x
    var previous = UInt32(0)
    var lookback = partition_index
    var finished = tid >= RADIX

    while True:
        var missing = UInt32.MAX
        if not finished:
            var spins = 0
            while not finished and spins < FALLBACK_SPIN_LIMIT:
                var address = pass_hist.unsafe_offset(
                    (lookback * RADIX + tid) * 2
                )
                var low = Atomic.load[ordering=Ordering.RELAXED](address)
                var high = Atomic.load[ordering=Ordering.RELAXED](
                    address.unsafe_offset(1)
                )
                var state = low & SPLIT_STATE_MASK
                if state != 0 and state == (high & SPLIT_STATE_MASK):
                    previous += (low & SPLIT_VALUE_MASK) | (
                        (high & SPLIT_VALUE_MASK) << 16
                    )
                    if state == SPLIT_INCLUSIVE:
                        finished = True
                    else:
                        lookback -= 1
                    spins = 0
                else:
                    spins += 1
            if not finished:
                missing = UInt32(lookback)

        # All lanes agree on one blocking predecessor. Finished digit lanes
        # still participate in every collective and every fallback histogram.
        var chosen = block.min[block_size=BLOCK_SIZE, broadcast=True](missing)
        if chosen == UInt32.MAX:
            break
        debug_assert["safe", _use_compiler_assume=True](chosen > 0)
        for digit in range(tid, RADIX, BLOCK_SIZE):
            scratch[unsafe_offset=digit] = 0
        barrier()

        # A predecessor is always a full tile: the partial tail can never be
        # before another tile. Read only immutable pass input, never output.
        var start = (Int(chosen) - 1) * PART_SIZE
        var warp_start = (tid / WARP_SIZE) * WARP_SIZE * KEYS_PER_THREAD
        var source = start + warp_start + lane_id()
        var fallback_keys = Array[Scalar[keys_dtype], KEYS_PER_THREAD](
            fill_with=lambda (i: Int) -> Scalar[keys_dtype]: keys[
                unsafe_offset=source + i * WARP_SIZE
            ]
        )
        # Share the existing portable ballot-based histogram reduction. The
        # returned ranks are unused; equal digits need one atomic per warp.
        _ = warp_level_multi_split[keys_dtype, 8, KEYS_PER_THREAD](
            fallback_keys, lane_id(), Scalar[keys_dtype](radix_shift), scratch
        )
        barrier()
        publish_split_histogram[BLOCK_SIZE, SPLIT_READY](
            pass_hist, scratch, Int(chosen)
        )
        barrier()
        # Re-read the winning state. An inclusive update racing the fallback
        # must be consumed as a prefix, not counted as a local reduction.

    if tid < RADIX:
        scratch[unsafe_offset=tid] = previous + local_reduction
        local_offsets[unsafe_offset=tid] = (
            previous - local_offsets[unsafe_offset=tid]
        )
    barrier()
    comptime if not SUPPRESS_PUBLICATION:
        publish_split_histogram[BLOCK_SIZE, SPLIT_INCLUSIVE](
            pass_hist, scratch, partition_index + 1
        )
    barrier()


def _validate_size(size: Int, capacity: Int, key_count: Int) raises:
    if size < 0 or size > Int(Int32.MAX):
        raise "Decoupled Fallback size must fit a nonnegative Int32"
    if size > capacity or size > key_count:
        raise "Decoupled Fallback input exceeds the buffer capacity"


def init_fallback_workspace[
    NUM_PASSES: Int, BLOCK_SIZE: Int
](
    global_hist: Pointer[UInt32, MutAnyOrigin],
    pass_hist: Pointer[UInt32, MutAnyOrigin],
    index: Pointer[UInt32, MutAnyOrigin],
    pass_hist_size: Int64,
):
    init_radix_workspace[NUM_PASSES, BLOCK_SIZE, .int64](
        global_hist, pass_hist, index, pass_hist_size
    )


def scan_global_fallback(
    global_hist: Pointer[UInt32, MutAnyOrigin],
    pass_hist: Pointer[UInt32, MutAnyOrigin],
    hist_blocks: Int32,
):
    var tid = thread_idx.x
    var value = global_hist[unsafe_offset=tid + block_idx.x * 256]
    var prefix = block.prefix_sum[block_size=256, exclusive=True](value)
    var offset = 2 * (tid + block_idx.x * 256 * Int(hist_blocks))
    pass_hist[unsafe_offset=offset] = (
        prefix & SPLIT_VALUE_MASK
    ) | SPLIT_INCLUSIVE
    pass_hist[unsafe_offset=offset + 1] = (prefix >> 16) | SPLIT_INCLUSIVE


def digit_binning[
    keys_dtype: DType,
    vals_dtype: DType,
    *,
    BITS_PER_PASS: Int,
    BLOCK_SIZE: Int,
    KEYS_PER_THREAD: Int,
    HAVE_PAYLOAD: Bool,
    _TEST_SUPPRESS_PUBLICATION: Bool = False,
](
    keys_current: Pointer[Scalar[keys_dtype], MutAnyOrigin],
    keys_alternate: Pointer[Scalar[keys_dtype], MutAnyOrigin],
    vals_current_opt: Optional[Pointer[Scalar[vals_dtype], MutAnyOrigin]],
    vals_alternate_opt: Optional[Pointer[Scalar[vals_dtype], MutAnyOrigin]],
    pass_hist: Pointer[UInt32, MutAnyOrigin],
    index: Pointer[UInt32, MutAnyOrigin],
    size: Int32,
    radix_shift: UInt32,
):
    var size_int = Int(size)
    comptime RADIX = 2**BITS_PER_PASS
    comptime assert BITS_PER_PASS == 8 and KEYS_PER_THREAD > 0
    comptime RADIX_MASK = Scalar[keys_dtype](RADIX - 1)
    comptime NUM_WARPS = BLOCK_SIZE / WARP_SIZE
    comptime WARP_PART_SIZE = WARP_SIZE * KEYS_PER_THREAD
    comptime PART_SIZE = NUM_WARPS * WARP_PART_SIZE
    comptime TOTAL_WARP_HISTS_SIZE = NUM_WARPS * RADIX
    comptime BIN_PART_SIZE = BLOCK_SIZE * KEYS_PER_THREAD

    # Histogram, key staging, and payload staging reuse the same storage.
    # Preserve the full scalar width instead of narrowing through UInt32.
    comptime STORAGE_BYTES = max(
        TOTAL_WARP_HISTS_SIZE * 4,
        max(
            BIN_PART_SIZE * (bit_width_of[keys_dtype]() / 8),
            BIN_PART_SIZE * (bit_width_of[vals_dtype]() / 8),
        ),
    )
    var s_storage = stack_allocation[
        ceildiv(STORAGE_BYTES, 8), UInt64, address_space=AddressSpace.SHARED
    ]()
    var s_warp_histograms = s_storage.unsafe_bitcast[UInt32]()
    var s_keys = s_storage.unsafe_bitcast[Scalar[keys_dtype]]()
    var s_values = s_storage.unsafe_bitcast[Scalar[vals_dtype]]()
    var s_local_histogram = stack_allocation[
        RADIX, UInt32, address_space=AddressSpace.SHARED
    ]()

    var s_fallback_histogram = stack_allocation[
        RADIX, UInt32, address_space=AddressSpace.SHARED
    ]()
    var tile_reduction = UInt32(0)

    var tid = thread_idx.x
    var lid = lane_id()
    var wid = tid / WARP_SIZE

    var s_warp_hist_ptr = s_warp_histograms.unsafe_offset(wid << BITS_PER_PASS)

    for i in range(tid, TOTAL_WARP_HISTS_SIZE, BLOCK_SIZE):
        s_warp_histograms[unsafe_offset=i] = 0

    # atomically assign partition tiles
    if tid == 0:
        var idx_offset = Int(radix_shift >> 3)
        # Offset scratch is unused until the ranked histogram is reduced.
        # A key-staging slot may still belong to the histogram for small tiles.
        s_local_histogram[unsafe_offset=0] = Atomic.fetch_add[
            ordering=ordering
        ](index.unsafe_offset(idx_offset), 1)
    barrier()

    var partition_index = Int(s_local_histogram[unsafe_offset=0])

    # load keys
    var BIN_SUB_PART_START = wid * WARP_PART_SIZE
    var BIN_PART_START = partition_index * PART_SIZE
    var t_base = lid + BIN_SUB_PART_START + BIN_PART_START
    var full_partition = BIN_PART_START + PART_SIZE <= size_int
    var keys = Array[Scalar[keys_dtype], KEYS_PER_THREAD](
        fill_with=lambda (i: Int) -> Scalar[keys_dtype]: keys_current[
            unsafe_offset=t_base + i * WARP_SIZE
        ] if full_partition
        or t_base + i * WARP_SIZE < size_int else Scalar[keys_dtype].MAX
    )

    # Warp-Level Multi-Split (WLMS)
    var offsets = warp_level_multi_split[
        keys_dtype, BITS_PER_PASS, KEYS_PER_THREAD
    ](keys, lid, Scalar[keys_dtype](radix_shift), s_warp_hist_ptr)
    barrier()

    # Exclusive prefix sum up the warp histograms
    if tid < RADIX:
        var reduction = s_warp_histograms[unsafe_offset=tid]
        for i in range(tid + RADIX, TOTAL_WARP_HISTS_SIZE, RADIX):
            reduction += s_warp_histograms[unsafe_offset=i]
            s_warp_histograms[unsafe_offset=i] = (
                reduction - s_warp_histograms[unsafe_offset=i]
            )

        tile_reduction = reduction
        s_fallback_histogram[unsafe_offset=tid] = reduction

        var sum = warp.prefix_sum[exclusive=False](reduction)
        var shifted = circular_shift(sum)
        s_local_histogram[unsafe_offset=tid] = shifted
    barrier()

    comptime if not _TEST_SUPPRESS_PUBLICATION:
        publish_split_histogram[BLOCK_SIZE, SPLIT_READY](
            pass_hist, s_fallback_histogram, partition_index + 1
        )
        barrier()

    scan_local_histogram[BITS_PER_PASS](s_local_histogram)

    # Update offsets
    comptime for i in range(KEYS_PER_THREAD):
        var t2 = Int((keys[i] >> Scalar[keys_dtype](radix_shift)) & RADIX_MASK)
        offsets[i] += s_local_histogram[unsafe_offset=t2]
        if wid > 0:
            offsets[i] += s_warp_histograms[unsafe_offset=wid * RADIX + t2]
    barrier()

    # Scatter keys into shared memory
    comptime for i in range(KEYS_PER_THREAD):
        s_keys[unsafe_offset=Int(offsets[i])] = keys[i]

    # Cooperatively recover missing predecessor histograms before scattering.
    fallback_lookback[
        keys_dtype, BLOCK_SIZE, KEYS_PER_THREAD, _TEST_SUPPRESS_PUBLICATION
    ](
        keys_current,
        pass_hist,
        s_local_histogram,
        s_fallback_histogram,
        tile_reduction,
        partition_index,
        radix_shift,
    )

    scatter_radix_tile[
        keys_dtype,
        vals_dtype,
        BITS_PER_PASS=BITS_PER_PASS,
        BLOCK_SIZE=BLOCK_SIZE,
        KEYS_PER_THREAD=KEYS_PER_THREAD,
        HAVE_PAYLOAD=HAVE_PAYLOAD,
    ](
        keys_alternate,
        vals_current_opt,
        vals_alternate_opt,
        s_keys,
        s_values,
        s_local_histogram,
        offsets,
        partition_index,
        size,
        radix_shift,
    )


struct DecoupledFallbackWorkspace[
    keys_dtype: DType,
    vals_dtype: DType,
    *,
    BLOCK_SIZE: Int,
    KEYS_PER_THREAD: Int,
]:
    var keys_alternate: DeviceBuffer[Self.keys_dtype]
    var vals_alternate: DeviceBuffer[Self.vals_dtype]
    var global_hist: DeviceBuffer[.uint32]
    var pass_hist: DeviceBuffer[.uint32]
    var index: DeviceBuffer[.uint32]

    def __init__(out self, ctx: DeviceContext, size: Int) raises:
        comptime assert (
            Self.keys_dtype == DType.uint32 or Self.keys_dtype == DType.uint64
        )
        comptime assert (
            Self.BLOCK_SIZE >= 256
            and Self.BLOCK_SIZE % WARP_SIZE == 0
            and Self.KEYS_PER_THREAD > 0
        )
        if size < 0 or size > Int(Int32.MAX):
            raise "Decoupled Fallback size must fit a nonnegative Int32"
        comptime RADIX = 256
        comptime BIN_PART_SIZE = Self.BLOCK_SIZE * Self.KEYS_PER_THREAD
        comptime NUM_PASSES = bit_width_of[Self.keys_dtype]() / 8

        var binning_blocks = ceildiv(size, BIN_PART_SIZE)
        var hist_blocks = binning_blocks + 1

        self.keys_alternate = ctx.enqueue_create_buffer[Self.keys_dtype](size)
        self.vals_alternate = ctx.enqueue_create_buffer[Self.vals_dtype](size)
        self.global_hist = ctx.enqueue_create_buffer[.uint32](
            RADIX * NUM_PASSES
        )
        self.pass_hist = ctx.enqueue_create_buffer[.uint32](
            hist_blocks * RADIX * NUM_PASSES * 2
        )
        self.index = ctx.enqueue_create_buffer[.uint32](NUM_PASSES)

    def validate(self, size: Int) raises:
        _validate_size(size, len(self.keys_alternate), len(self.keys_alternate))
        comptime passes = bit_width_of[Self.keys_dtype]() / 8
        var rows = ceildiv(size, Self.BLOCK_SIZE * Self.KEYS_PER_THREAD) + 1
        require_sort_capacity(
            256 * passes, len(self.global_hist), "global histogram"
        )
        require_sort_capacity(
            rows * 256 * passes * 2, len(self.pass_hist), "pass histogram"
        )
        require_sort_capacity(passes, len(self.index), "partition index")


def decoupled_fallback_radix_sort_keys[
    keys_dtype: DType,
    *,
    KEYS_PER_THREAD: Int,
    BINNING_TPB: Int,
    G_HIST_TPB: Int = 256,
    G_HIST_ITEMS_PER_THREAD: Int = 32,
    VEC_WIDTH: SIMDLength = 4,
](
    ctx: DeviceContext,
    mut workspace: DecoupledFallbackWorkspace[
        keys_dtype,
        keys_dtype,
        BLOCK_SIZE=BINNING_TPB,
        KEYS_PER_THREAD=KEYS_PER_THREAD,
    ],
    mut keys: DeviceBuffer[keys_dtype],
    size: Int,
) raises:
    _validate_size(size, len(workspace.keys_alternate), len(keys))
    if size <= 1:
        return
    workspace.validate(size)
    enqueue_radix_passes[
        keys_dtype,
        keys_dtype,
        BLOCK_SIZE=BINNING_TPB,
        KEYS_PER_THREAD=KEYS_PER_THREAD,
        HAVE_PAYLOAD=False,
        HISTOGRAM_WORDS=2,
        G_HIST_TPB=G_HIST_TPB,
        G_HIST_ITEMS_PER_THREAD=G_HIST_ITEMS_PER_THREAD,
        VEC_WIDTH=VEC_WIDTH,
        scan_fn=scan_global_fallback,
        binning_fn=digit_binning[
            keys_dtype,
            keys_dtype,
            BITS_PER_PASS=8,
            BLOCK_SIZE=BINNING_TPB,
            KEYS_PER_THREAD=KEYS_PER_THREAD,
            HAVE_PAYLOAD=False,
        ],
    ](
        ctx,
        keys.unsafe_ptr(),
        workspace.keys_alternate.unsafe_ptr(),
        Optional[Pointer[Scalar[keys_dtype], MutAnyOrigin]](),
        Optional[Pointer[Scalar[keys_dtype], MutAnyOrigin]](),
        workspace.global_hist.unsafe_ptr(),
        workspace.pass_hist.unsafe_ptr(),
        workspace.index.unsafe_ptr(),
        size,
    )


def decoupled_fallback_radix_sort_pairs[
    keys_dtype: DType,
    vals_dtype: DType,
    *,
    KEYS_PER_THREAD: Int,
    BINNING_TPB: Int,
    G_HIST_TPB: Int = 256,
    G_HIST_ITEMS_PER_THREAD: Int = 32,
    VEC_WIDTH: SIMDLength = 4,
](
    ctx: DeviceContext,
    mut workspace: DecoupledFallbackWorkspace[
        keys_dtype,
        vals_dtype,
        BLOCK_SIZE=BINNING_TPB,
        KEYS_PER_THREAD=KEYS_PER_THREAD,
    ],
    mut keys: DeviceBuffer[keys_dtype],
    mut values: DeviceBuffer[vals_dtype],
    size: Int,
) raises:
    _validate_size(
        size, len(workspace.keys_alternate), min(len(keys), len(values))
    )
    if size <= 1:
        return
    workspace.validate(size)
    require_sort_capacity(size, len(workspace.vals_alternate), "value scratch")
    enqueue_radix_passes[
        keys_dtype,
        vals_dtype,
        BLOCK_SIZE=BINNING_TPB,
        KEYS_PER_THREAD=KEYS_PER_THREAD,
        HAVE_PAYLOAD=True,
        HISTOGRAM_WORDS=2,
        G_HIST_TPB=G_HIST_TPB,
        G_HIST_ITEMS_PER_THREAD=G_HIST_ITEMS_PER_THREAD,
        VEC_WIDTH=VEC_WIDTH,
        scan_fn=scan_global_fallback,
        binning_fn=digit_binning[
            keys_dtype,
            vals_dtype,
            BITS_PER_PASS=8,
            BLOCK_SIZE=BINNING_TPB,
            KEYS_PER_THREAD=KEYS_PER_THREAD,
            HAVE_PAYLOAD=True,
        ],
    ](
        ctx,
        keys.unsafe_ptr(),
        workspace.keys_alternate.unsafe_ptr(),
        Optional(values.unsafe_ptr()),
        Optional(workspace.vals_alternate.unsafe_ptr()),
        workspace.global_hist.unsafe_ptr(),
        workspace.pass_hist.unsafe_ptr(),
        workspace.index.unsafe_ptr(),
        size,
    )


def decoupled_fallback_radix_sort_pairs[
    keys_dtype: DType,
    vals_dtype: DType,
    *,
    BLOCK_SIZE: Int = 512,
    KEYS_PER_THREAD: Int = (
        15 if bit_width_of[keys_dtype]() <= 32
        and bit_width_of[vals_dtype]() <= 32 else 9
    ),
    G_HIST_TPB: Int = 256,
    G_HIST_ITEMS_PER_THREAD: Int = 32,
    VEC_WIDTH: SIMDLength = 4,
](
    ctx: DeviceContext,
    mut keys: DeviceBuffer[keys_dtype],
    mut values: DeviceBuffer[vals_dtype],
    size: Int,
) raises:
    _validate_size(
        size, min(len(keys), len(values)), min(len(keys), len(values))
    )
    if size <= 1:
        return
    var workspace = DecoupledFallbackWorkspace[
        keys_dtype,
        vals_dtype,
        BLOCK_SIZE=BLOCK_SIZE,
        KEYS_PER_THREAD=KEYS_PER_THREAD,
    ](ctx, size)
    decoupled_fallback_radix_sort_pairs[
        BINNING_TPB=BLOCK_SIZE,
        KEYS_PER_THREAD=KEYS_PER_THREAD,
        G_HIST_TPB=G_HIST_TPB,
        G_HIST_ITEMS_PER_THREAD=G_HIST_ITEMS_PER_THREAD,
        VEC_WIDTH=VEC_WIDTH,
    ](ctx, workspace, keys, values, size)
