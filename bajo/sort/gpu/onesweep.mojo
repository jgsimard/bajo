from std.atomic import Atomic, Ordering
from std.bit import pop_count, count_trailing_zeros
from std.gpu import (
    thread_idx,
    block_idx,
    lane_id,
    grid_dim,
    WARP_SIZE,
)
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from std.gpu.primitives import warp, block
from std.gpu.sync import barrier
from std.math import ceildiv
from std.memory import stack_allocation, bitcast
from std.sys.info import bit_width_of

from .utils import circular_shift, warp_level_multi_split

comptime ordering = Ordering.RELAXED
comptime LANE_LOG = count_trailing_zeros(WARP_SIZE)  # 2^5 = 32 => 5

# Flags for Decoupled Look-back
comptime FLAG_NOT_READY = 0
comptime FLAG_REDUCTION = 1
comptime FLAG_INCLUSIVE = 2
comptime FLAG_MASK = 3


def global_histogram[
    keys_dtype: DType,
    *,
    BLOCK_SIZE: Int,
    RADIX: Int,
    VEC_WIDTH: Int,
    ITEMS_PER_THREAD: Int,
](
    sort: UnsafePointer[Scalar[keys_dtype], MutAnyOrigin],
    global_hist: UnsafePointer[UInt32, MutAnyOrigin],
    size: Int,
):
    comptime BYTES_PER_KEY = bit_width_of[keys_dtype]() / 8
    comptime PART_SIZE = BLOCK_SIZE * ITEMS_PER_THREAD
    comptime PADDED_RADIX = RADIX * 2
    comptime THREADS_PER_PARTITION = 64
    comptime NUM_PARTITIONS = BLOCK_SIZE / THREADS_PER_PARTITION

    var tid = thread_idx.x
    var bid = block_idx.x
    var gdim = grid_dim.x

    var s_hists = stack_allocation[
        PADDED_RADIX * BYTES_PER_KEY, UInt32, address_space=AddressSpace.SHARED
    ]()

    for i in range(tid, PADDED_RADIX * BYTES_PER_KEY, BLOCK_SIZE):
        s_hists[i] = 0
    barrier()

    # 64 threads : 1 histogram in shared memory
    var wave_offset = tid / THREADS_PER_PARTITION * RADIX

    @always_inline
    def _accumulate_hist[width: Int](i: Int) capturing:
        var _t = sort.load[width=width](i)
        var t = bitcast[DType.uint8, width * BYTES_PER_KEY](_t)

        comptime for v in range(width):
            comptime for b in range(BYTES_PER_KEY):
                var byte_val = Int(t[v * BYTES_PER_KEY + b])
                var s_idx = b * PADDED_RADIX + wave_offset + byte_val
                _ = Atomic.fetch_add[ordering=ordering](s_hists + s_idx, 1)

    var block_start = bid * PART_SIZE
    if bid < gdim - 1:
        for i in range(
            block_start + tid * VEC_WIDTH,
            block_start + PART_SIZE,
            BLOCK_SIZE * VEC_WIDTH,
        ):
            _accumulate_hist[VEC_WIDTH](i)
    else:
        for i in range(block_start + tid, size, BLOCK_SIZE):
            _accumulate_hist[1](i)
    barrier()

    # Reduce
    for i in range(tid, RADIX, BLOCK_SIZE):
        comptime for b in range(BYTES_PER_KEY):
            var sum: UInt32 = 0
            comptime for p in range(NUM_PARTITIONS):
                sum += s_hists[b * PADDED_RADIX + p * RADIX + i]
            var g_idx = (b * RADIX) + i
            _ = Atomic.fetch_add[ordering=ordering](global_hist + g_idx, sum)


def scan_global(
    global_hist: UnsafePointer[UInt32, MutAnyOrigin],
    pass_hist: UnsafePointer[UInt32, MutAnyOrigin],
    hist_blocks: Int,
):
    comptime RADIX = 256
    var tid = thread_idx.x
    var bid = block_idx.x

    var val = global_hist[tid + bid * RADIX]
    var out_val = block.prefix_sum[
        block_size=RADIX,
        exclusive=True,
    ](val)
    out_val = (out_val << 2) | FLAG_INCLUSIVE
    pass_hist[tid + bid * RADIX * hist_blocks] = out_val


def digit_binning[
    keys_dtype: DType,
    vals_dtype: DType,
    *,
    BITS_PER_PASS: Int,
    BLOCK_SIZE: Int,
    KEYS_PER_THREAD: Int,
    HAVE_PAYLOAD: Bool,
](
    keys_current: UnsafePointer[Scalar[keys_dtype], MutAnyOrigin],
    keys_alternate: UnsafePointer[Scalar[keys_dtype], MutAnyOrigin],
    vals_current_opt: Optional[UnsafePointer[Scalar[vals_dtype], MutAnyOrigin]],
    vals_alternate_opt: Optional[
        UnsafePointer[Scalar[vals_dtype], MutAnyOrigin]
    ],
    pass_hist: UnsafePointer[UInt32, MutAnyOrigin],
    index: UnsafePointer[UInt32, MutAnyOrigin],
    size: Int,
    radix_shift: UInt32,
):
    comptime RADIX = 2**BITS_PER_PASS
    comptime RADIX_MASK = Scalar[keys_dtype](RADIX - 1)
    comptime NUM_WARPS = BLOCK_SIZE / WARP_SIZE
    comptime WARP_PART_SIZE = WARP_SIZE * KEYS_PER_THREAD
    comptime PART_SIZE = NUM_WARPS * WARP_PART_SIZE
    comptime TOTAL_WARP_HISTS_SIZE = NUM_WARPS * RADIX
    comptime BIN_PART_SIZE = BLOCK_SIZE * KEYS_PER_THREAD

    var s_warp_histograms = stack_allocation[
        BIN_PART_SIZE, UInt32, address_space=AddressSpace.SHARED
    ]()
    var s_local_histogram = stack_allocation[
        RADIX, UInt32, address_space=AddressSpace.SHARED
    ]()

    var tid = thread_idx.x
    var lid = lane_id()
    var wid = tid / WARP_SIZE
    var gdim = grid_dim.x

    var s_warp_hist_ptr = s_warp_histograms + (wid << BITS_PER_PASS)

    for i in range(tid, TOTAL_WARP_HISTS_SIZE, BLOCK_SIZE):
        s_warp_histograms[i] = 0

    # atomically assign partition tiles
    if tid == 0:
        var idx_offset = Int(radix_shift >> 3)
        s_warp_histograms[BIN_PART_SIZE - 1] = Atomic.fetch_add[
            ordering=ordering
        ](index + idx_offset, 1)
    barrier()

    var partition_index = Int(s_warp_histograms[BIN_PART_SIZE - 1])

    # load keys
    var keys = InlineArray[Scalar[keys_dtype], KEYS_PER_THREAD](
        uninitialized=True
    )
    var BIN_SUB_PART_START = wid * WARP_PART_SIZE
    var BIN_PART_START = partition_index * PART_SIZE
    var t_base = lid + BIN_SUB_PART_START + BIN_PART_START

    var t = t_base
    comptime for i in range(KEYS_PER_THREAD):
        if partition_index < gdim - 1:
            keys[i] = keys_current[t]
        else:
            keys[i] = keys_current[t] if t < size else Scalar[keys_dtype].MAX
        t += WARP_SIZE

    # Warp-Level Multi-Split (WLMS)
    var offsets = warp_level_multi_split[
        keys_dtype, BITS_PER_PASS, KEYS_PER_THREAD
    ](keys, lid, Scalar[keys_dtype](radix_shift), s_warp_hist_ptr)
    barrier()

    # Exclusive prefix sum up the warp histograms
    if tid < RADIX:
        var reduction = s_warp_histograms[tid]
        for i in range(tid + RADIX, TOTAL_WARP_HISTS_SIZE, RADIX):
            reduction += s_warp_histograms[i]
            s_warp_histograms[i] = reduction - s_warp_histograms[i]

        _ = Atomic.fetch_add[ordering=ordering](
            pass_hist + (tid + (partition_index + 1) * RADIX),
            FLAG_REDUCTION | (reduction << 2),
        )

        var sum = warp.prefix_sum[exclusive=False](reduction)
        var shifted = circular_shift(sum)
        s_local_histogram[tid] = shifted
    barrier()

    var idx = tid << LANE_LOG
    var val: UInt32 = 0
    if tid < (RADIX >> LANE_LOG):
        val = s_local_histogram[idx]
    var exc = warp.prefix_sum[exclusive=True](val)
    if tid < (RADIX >> LANE_LOG):
        s_local_histogram[idx] = exc
    barrier()

    if tid < RADIX and lid > 0:
        s_local_histogram[tid] += s_local_histogram[tid - lid]
    barrier()

    # Update offsets
    comptime for i in range(KEYS_PER_THREAD):
        var t2 = Int((keys[i] >> Scalar[keys_dtype](radix_shift)) & RADIX_MASK)
        offsets[i] += s_local_histogram[t2]
        if wid > 0:
            offsets[i] += s_warp_histograms[wid * RADIX + t2]
    barrier()

    # Scatter keys into shared memory
    comptime for i in range(KEYS_PER_THREAD):
        s_warp_histograms[Int(offsets[i])] = UInt32(keys[i])

    # Decoupled Look-back (Spin Loop)
    if tid < RADIX:
        var reduction: UInt32 = 0
        var k = partition_index
        while True:
            var flag_payload = (pass_hist + tid + k * RADIX).load[
                volatile=True
            ]()
            var flag_mask = flag_payload & FLAG_MASK
            if flag_mask == FLAG_INCLUSIVE:
                reduction += flag_payload >> 2
                _ = Atomic.fetch_add[ordering=ordering](
                    pass_hist + tid + (partition_index + 1) * RADIX,
                    FLAG_REDUCTION | (reduction << 2),
                )
                s_local_histogram[tid] = reduction - s_local_histogram[tid]
                break
            if flag_mask == FLAG_REDUCTION:
                reduction += flag_payload >> 2
                k -= 1
    barrier()

    # Scatter to Device Memory
    var part_size = (
        BIN_PART_SIZE if partition_index < gdim - 1 else size - BIN_PART_START
    )
    comptime if HAVE_PAYLOAD:
        debug_assert["safe"](Bool(vals_current_opt))
        debug_assert["safe"](Bool(vals_alternate_opt))
        var vals_current = vals_current_opt.unsafe_value()
        var vals_alternate = vals_alternate_opt.unsafe_value()

        var vals = InlineArray[Scalar[vals_dtype], KEYS_PER_THREAD](
            uninitialized=True
        )
        var digits = InlineArray[UInt8, KEYS_PER_THREAD](uninitialized=True)

        comptime for i in range(KEYS_PER_THREAD):
            var t_idx = tid + i * BLOCK_SIZE
            if t_idx < part_size:
                var key = s_warp_histograms[t_idx]
                var d = (key >> radix_shift) & UInt32(RADIX_MASK)
                digits[i] = d.cast[DType.uint8]()
                keys_alternate[
                    s_local_histogram[Int(d)] + UInt32(t_idx)
                ] = Scalar[keys_dtype](key)
        barrier()

        var t_payload = t_base
        comptime for i in range(KEYS_PER_THREAD):
            if t_payload < size:
                vals[i] = vals_current[t_payload]
            else:
                vals[i] = Scalar[vals_dtype].MAX
            t_payload += WARP_SIZE

        comptime for i in range(KEYS_PER_THREAD):
            s_warp_histograms[Int(offsets[i])] = UInt32(vals[i])
        barrier()

        comptime for i in range(KEYS_PER_THREAD):
            var t_idx = tid + i * BLOCK_SIZE
            if t_idx < part_size:
                var d = Int(digits[i])
                vals_alternate[s_local_histogram[d] + UInt32(t_idx)] = Scalar[
                    vals_dtype
                ](s_warp_histograms[t_idx])
    else:
        for i in range(tid, part_size, BLOCK_SIZE):
            var key = s_warp_histograms[i]
            var digit = Int((key >> radix_shift) & UInt32(RADIX_MASK))
            var dst = s_local_histogram[digit] + UInt32(i)
            keys_alternate[dst] = Scalar[keys_dtype](key)


struct OneSweepWorkspace[
    keys_dtype: DType,
    vals_dtype: DType,
    *,
    BLOCK_SIZE: Int,
    KEYS_PER_THREAD: Int,
]:
    var keys_alternate: DeviceBuffer[Self.keys_dtype]
    var vals_alternate: DeviceBuffer[Self.vals_dtype]
    var global_hist: DeviceBuffer[DType.uint32]
    var pass_hist: DeviceBuffer[DType.uint32]
    var index: DeviceBuffer[DType.uint32]

    def __init__(out self, ctx: DeviceContext, size: Int) raises:
        comptime RADIX = 256
        comptime BIN_PART_SIZE = Self.BLOCK_SIZE * Self.KEYS_PER_THREAD
        comptime NUM_PASSES = bit_width_of[Self.keys_dtype]() / 8

        var binning_blocks = ceildiv(size, BIN_PART_SIZE)
        var hist_blocks = binning_blocks + 1

        self.keys_alternate = ctx.enqueue_create_buffer[Self.keys_dtype](size)
        self.vals_alternate = ctx.enqueue_create_buffer[Self.vals_dtype](size)
        self.global_hist = ctx.enqueue_create_buffer[DType.uint32](
            RADIX * NUM_PASSES
        )
        self.pass_hist = ctx.enqueue_create_buffer[DType.uint32](
            hist_blocks * RADIX * NUM_PASSES
        )
        self.index = ctx.enqueue_create_buffer[DType.uint32](NUM_PASSES)


def onesweep_radix_sort_keys[
    keys_dtype: DType, *, KEYS_PER_THREAD: Int, BINNING_TPB: Int
](
    ctx: DeviceContext,
    mut workspace: OneSweepWorkspace[
        keys_dtype,
        DType.invalid,
        BLOCK_SIZE=BINNING_TPB,
        KEYS_PER_THREAD=KEYS_PER_THREAD,
    ],
    mut keys: DeviceBuffer[keys_dtype],
    size: Int,
) raises:
    comptime BITS_PER_PASS = 8
    comptime NUM_PASSES = bit_width_of[keys_dtype]() / BITS_PER_PASS
    comptime RADIX = 2**BITS_PER_PASS
    comptime BIN_PART_SIZE = BINNING_TPB * KEYS_PER_THREAD
    comptime G_HIST_TPB = 128
    comptime G_HIST_ITEMS_PER_THREAD = 128
    comptime G_HIST_PART_SIZE = G_HIST_TPB * G_HIST_ITEMS_PER_THREAD
    comptime VEC_WIDTH = 4

    var binning_blocks = ceildiv(size, BIN_PART_SIZE)
    var hist_blocks = binning_blocks + 1
    var g_hist_blocks = ceildiv(size, G_HIST_PART_SIZE)

    workspace.index.enqueue_fill(0)
    workspace.global_hist.enqueue_fill(0)
    workspace.pass_hist.enqueue_fill(0)
    ctx.synchronize()

    keys_current = keys.unsafe_ptr()
    keys_alternate = workspace.keys_alternate.unsafe_ptr()
    dummy_v_ptr = Optional[UnsafePointer[Scalar[DType.invalid], MutAnyOrigin]]()
    global_hist = workspace.global_hist.unsafe_ptr()
    pass_hist = workspace.pass_hist.unsafe_ptr()

    # 1. Global Histogram
    comptime _ghist = global_histogram[
        keys_dtype,
        BLOCK_SIZE=G_HIST_TPB,
        RADIX=RADIX,
        VEC_WIDTH=VEC_WIDTH,
        ITEMS_PER_THREAD=G_HIST_ITEMS_PER_THREAD,
    ]
    ctx.enqueue_function[_ghist, _ghist](
        keys_current,
        global_hist,
        size,
        grid_dim=g_hist_blocks,
        block_dim=G_HIST_TPB,
    )

    # 2. Block Scan
    comptime _scan = scan_global
    ctx.enqueue_function[_scan, _scan](
        global_hist,
        pass_hist,
        hist_blocks,
        grid_dim=NUM_PASSES,
        block_dim=RADIX,
    )

    # 3. Digit Binning Passes (Decoupled Look-back)
    comptime _bin = digit_binning[
        keys_dtype,
        DType.invalid,
        BITS_PER_PASS=BITS_PER_PASS,
        BLOCK_SIZE=BINNING_TPB,
        KEYS_PER_THREAD=KEYS_PER_THREAD,
        HAVE_PAYLOAD=False,
    ]

    comptime for pass_idx in range(NUM_PASSES):
        comptime radix_shift = UInt32(pass_idx * 8)
        var pass_hist_offset = pass_idx * hist_blocks * RADIX
        var pass_hist_ptr = pass_hist + pass_hist_offset
        ctx.enqueue_function[_bin, _bin](
            keys_current,
            keys_alternate,
            dummy_v_ptr,
            dummy_v_ptr,
            pass_hist_ptr,
            workspace.index.unsafe_ptr(),
            size,
            radix_shift,
            grid_dim=binning_blocks,
            block_dim=BINNING_TPB,
        )
        swap(keys_current, keys_alternate)

    ctx.synchronize()


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
    comptime BITS_PER_PASS = 8
    comptime NUM_PASSES = bit_width_of[keys_dtype]() / BITS_PER_PASS
    comptime RADIX = 2**BITS_PER_PASS
    comptime BIN_PART_SIZE = BINNING_TPB * KEYS_PER_THREAD
    comptime G_HIST_TPB = 128
    comptime G_HIST_ITEMS_PER_THREAD = 128
    comptime G_HIST_PART_SIZE = G_HIST_TPB * G_HIST_ITEMS_PER_THREAD
    comptime VEC_WIDTH = 4

    var binning_blocks = ceildiv(size, BIN_PART_SIZE)
    var hist_blocks = binning_blocks + 1
    var g_hist_blocks = ceildiv(size, G_HIST_PART_SIZE)

    workspace.index.enqueue_fill(0)
    workspace.global_hist.enqueue_fill(0)
    workspace.pass_hist.enqueue_fill(0)
    ctx.synchronize()

    keys_current = keys.unsafe_ptr()
    keys_alternate = workspace.keys_alternate.unsafe_ptr()
    vals_current = values.unsafe_ptr()
    vals_alternate = workspace.vals_alternate.unsafe_ptr()
    global_hist = workspace.global_hist.unsafe_ptr()
    pass_hist = workspace.pass_hist.unsafe_ptr()

    # 1. Global Histogram
    comptime _ghist = global_histogram[
        keys_dtype,
        BLOCK_SIZE=G_HIST_TPB,
        RADIX=RADIX,
        VEC_WIDTH=VEC_WIDTH,
        ITEMS_PER_THREAD=G_HIST_ITEMS_PER_THREAD,
    ]
    ctx.enqueue_function[_ghist, _ghist](
        keys_current,
        global_hist,
        size,
        grid_dim=g_hist_blocks,
        block_dim=G_HIST_TPB,
    )

    # 2. Block Scan
    comptime _scan = scan_global
    ctx.enqueue_function[_scan, _scan](
        global_hist,
        pass_hist,
        hist_blocks,
        grid_dim=NUM_PASSES,
        block_dim=RADIX,
    )

    # 3. Digit Binning Passes (Decoupled Look-back)
    comptime _bin = digit_binning[
        keys_dtype,
        vals_dtype,
        BITS_PER_PASS=BITS_PER_PASS,
        BLOCK_SIZE=BINNING_TPB,
        KEYS_PER_THREAD=KEYS_PER_THREAD,
        HAVE_PAYLOAD=True,
    ]

    comptime for pass_idx in range(NUM_PASSES):
        comptime radix_shift = UInt32(pass_idx * 8)
        var pass_hist_offset = pass_idx * hist_blocks * RADIX
        ctx.enqueue_function[_bin, _bin](
            keys_current,
            keys_alternate,
            Optional(vals_current),
            Optional(vals_alternate),
            pass_hist + pass_hist_offset,
            workspace.index.unsafe_ptr(),
            size,
            radix_shift,
            grid_dim=binning_blocks,
            block_dim=BINNING_TPB,
        )
        swap(keys_current, keys_alternate)
        swap(vals_current, vals_alternate)

    ctx.synchronize()


def onesweep_radix_sort_pairs[
    keys_dtype: DType, vals_dtype: DType
](
    ctx: DeviceContext,
    mut keys: DeviceBuffer[keys_dtype],
    mut values: DeviceBuffer[vals_dtype],
    size: Int,
) raises:
    comptime BLOCK_SIZE = 512
    comptime KEYS_PER_THREAD = 15
    var workspace = OneSweepWorkspace[
        keys_dtype,
        vals_dtype,
        BLOCK_SIZE=BLOCK_SIZE,
        KEYS_PER_THREAD=KEYS_PER_THREAD,
    ](ctx, size)
    onesweep_radix_sort_pairs[
        BINNING_TPB=BLOCK_SIZE, KEYS_PER_THREAD=KEYS_PER_THREAD
    ](ctx, workspace, keys, values, size)
