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
    scan_local_histogram,
    scatter_radix_tile,
    enqueue_radix_passes,
)
from .validation import (
    validate_sort_size,
    require_sort_capacity,
    validate_onesweep_size,
)

comptime ordering = Ordering.RELAXED

# Flags for Decoupled Look-back
comptime FLAG_NOT_READY = 0
comptime FLAG_REDUCTION = 1
comptime FLAG_INCLUSIVE = 2
comptime FLAG_MASK = 3


def init_onesweep_workspace[
    NUM_PASSES: Int, BLOCK_SIZE: Int
](
    global_hist: Pointer[UInt32, MutAnyOrigin],
    pass_hist: Pointer[UInt32, MutAnyOrigin],
    index: Pointer[UInt32, MutAnyOrigin],
    pass_hist_size: Int32,
):
    init_radix_workspace[NUM_PASSES, BLOCK_SIZE, .int32](
        global_hist, pass_hist, index, pass_hist_size
    )


def scan_global(
    global_hist: Pointer[UInt32, MutAnyOrigin],
    pass_hist: Pointer[UInt32, MutAnyOrigin],
    hist_blocks: Int32,
):
    var hist_blocks_int = Int(hist_blocks)
    comptime RADIX = 256
    var tid = thread_idx.x
    var bid = block_idx.x

    var val = global_hist[unsafe_offset=tid + bid * RADIX]
    var out_val = block.prefix_sum[
        block_size=RADIX,
        exclusive=True,
    ](val)
    out_val = (out_val << 2) | FLAG_INCLUSIVE
    pass_hist[unsafe_offset=tid + bid * RADIX * hist_blocks_int] = out_val


def digit_binning[
    keys_dtype: DType,
    vals_dtype: DType,
    *,
    BITS_PER_PASS: Int,
    BLOCK_SIZE: Int,
    KEYS_PER_THREAD: Int,
    HAVE_PAYLOAD: Bool,
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

    var tid = thread_idx.x
    var lid = lane_id()
    var wid = tid / WARP_SIZE

    var s_warp_hist_ptr = s_warp_histograms.unsafe_offset(wid << BITS_PER_PASS)

    for i in range(tid, TOTAL_WARP_HISTS_SIZE, BLOCK_SIZE):
        s_warp_histograms[unsafe_offset=i] = 0

    # atomically assign partition tiles
    if tid == 0:
        var idx_offset = Int(radix_shift >> 3)
        s_warp_histograms[unsafe_offset=BIN_PART_SIZE - 1] = Atomic.fetch_add[
            ordering=ordering
        ](index.unsafe_offset(idx_offset), 1)
    barrier()

    var partition_index = Int(
        s_warp_histograms[unsafe_offset=BIN_PART_SIZE - 1]
    )

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

        _ = Atomic.fetch_add[ordering=ordering](
            pass_hist.unsafe_offset(tid + (partition_index + 1) * RADIX),
            FLAG_REDUCTION | (reduction << 2),
        )

        var sum = warp.prefix_sum[exclusive=False](reduction)
        var shifted = circular_shift(sum)
        s_local_histogram[unsafe_offset=tid] = shifted
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

    # Decoupled Look-back (Spin Loop)
    if tid < RADIX:
        var reduction: UInt32 = 0
        var k = partition_index
        while True:
            var flag_payload = pass_hist.unsafe_offset(
                tid + k * RADIX
            ).unsafe_load[volatile=True]()
            var flag_mask = flag_payload & FLAG_MASK
            if flag_mask == FLAG_INCLUSIVE:
                reduction += flag_payload >> 2
                _ = Atomic.fetch_add[ordering=ordering](
                    pass_hist.unsafe_offset(
                        tid + (partition_index + 1) * RADIX
                    ),
                    FLAG_REDUCTION | (reduction << 2),
                )
                s_local_histogram[unsafe_offset=tid] = (
                    reduction - s_local_histogram[unsafe_offset=tid]
                )
                break
            if flag_mask == FLAG_REDUCTION:
                reduction += flag_payload >> 2
                k -= 1
    barrier()

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


struct OneSweepWorkspace[
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
        comptime RADIX = 256
        comptime BIN_PART_SIZE = Self.BLOCK_SIZE * Self.KEYS_PER_THREAD
        comptime NUM_PASSES = bit_width_of[Self.keys_dtype]() / 8

        comptime assert Self.keys_dtype == .uint32 or Self.keys_dtype == .uint64
        comptime assert (
            Self.BLOCK_SIZE >= 256 and Self.BLOCK_SIZE % WARP_SIZE == 0
        )
        comptime assert Self.KEYS_PER_THREAD > 0
        var histogram_size = validate_onesweep_size[BIN_PART_SIZE, NUM_PASSES](
            size
        )

        self.keys_alternate = ctx.enqueue_create_buffer[Self.keys_dtype](size)
        self.vals_alternate = ctx.enqueue_create_buffer[Self.vals_dtype](size)
        self.global_hist = ctx.enqueue_create_buffer[.uint32](
            RADIX * NUM_PASSES
        )
        self.pass_hist = ctx.enqueue_create_buffer[.uint32](histogram_size)
        self.index = ctx.enqueue_create_buffer[.uint32](NUM_PASSES)

    def validate(self, size: Int) raises:
        """Check retained scratch, including its publicly mutable side buffers.
        """
        comptime passes = bit_width_of[Self.keys_dtype]() / 8
        var histogram_size = validate_onesweep_size[
            Self.BLOCK_SIZE * Self.KEYS_PER_THREAD, passes
        ](size)
        require_sort_capacity(size, len(self.keys_alternate), "key scratch")
        require_sort_capacity(
            256 * passes, len(self.global_hist), "global histogram"
        )
        require_sort_capacity(
            histogram_size, len(self.pass_hist), "pass histogram"
        )
        require_sort_capacity(passes, len(self.index), "partition index")


def onesweep_radix_sort_keys[
    keys_dtype: DType, *, KEYS_PER_THREAD: Int, BINNING_TPB: Int
](
    ctx: DeviceContext,
    mut workspace: OneSweepWorkspace[
        keys_dtype,
        keys_dtype,
        BLOCK_SIZE=BINNING_TPB,
        KEYS_PER_THREAD=KEYS_PER_THREAD,
    ],
    mut keys: DeviceBuffer[keys_dtype],
    size: Int,
) raises:
    validate_sort_size(size)
    require_sort_capacity(size, len(keys), "key input")
    if size <= 1:
        return
    workspace.validate(size)
    enqueue_radix_passes[
        keys_dtype,
        keys_dtype,
        BLOCK_SIZE=BINNING_TPB,
        KEYS_PER_THREAD=KEYS_PER_THREAD,
        HAVE_PAYLOAD=False,
        HISTOGRAM_WORDS=1,
        G_HIST_TPB=128,
        G_HIST_ITEMS_PER_THREAD=128,
        VEC_WIDTH=4,
        scan_fn=scan_global,
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


def onesweep_radix_sort_pairs[
    keys_dtype: DType,
    vals_dtype: DType,
    *,
    KEYS_PER_THREAD: Int,
    BINNING_TPB: Int,
](
    ctx: DeviceContext,
    mut workspace: OneSweepWorkspace[
        keys_dtype,
        vals_dtype,
        BLOCK_SIZE=BINNING_TPB,
        KEYS_PER_THREAD=KEYS_PER_THREAD,
    ],
    mut keys: DeviceBuffer[keys_dtype],
    mut values: DeviceBuffer[vals_dtype],
    size: Int,
) raises:
    validate_sort_size(size)
    require_sort_capacity(size, len(keys), "key input")
    require_sort_capacity(size, len(values), "value input")
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
        HISTOGRAM_WORDS=1,
        G_HIST_TPB=128,
        G_HIST_ITEMS_PER_THREAD=128,
        VEC_WIDTH=4,
        scan_fn=scan_global,
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


def onesweep_radix_sort_pairs[
    keys_dtype: DType, vals_dtype: DType
](
    ctx: DeviceContext,
    mut keys: DeviceBuffer[keys_dtype],
    mut values: DeviceBuffer[vals_dtype],
    size: Int,
) raises:
    validate_sort_size(size)
    require_sort_capacity(size, len(keys), "key input")
    require_sort_capacity(size, len(values), "value input")
    if size <= 1:
        return
    comptime BLOCK_SIZE = 512
    # Keep wide-key/payload staging within the same shared-memory budget.
    comptime KEYS_PER_THREAD = 15 if bit_width_of[
        keys_dtype
    ]() <= 32 and bit_width_of[vals_dtype]() <= 32 else 9
    var workspace = OneSweepWorkspace[
        keys_dtype,
        vals_dtype,
        BLOCK_SIZE=BLOCK_SIZE,
        KEYS_PER_THREAD=KEYS_PER_THREAD,
    ](ctx, size)
    onesweep_radix_sort_pairs[
        BINNING_TPB=BLOCK_SIZE, KEYS_PER_THREAD=KEYS_PER_THREAD
    ](ctx, workspace, keys, values, size)
