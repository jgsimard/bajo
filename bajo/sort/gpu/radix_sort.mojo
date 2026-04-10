from std.bit import pop_count, count_trailing_zeros
from std.gpu import (
    thread_idx,
    block_idx,
    block_dim,
    lane_id,
    grid_dim,
    WARP_SIZE,
    warp_id,
)
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.gpu.primitives import warp, block
from std.gpu.sync import barrier
from std.math import ceildiv
from std.memory import stack_allocation
from std.os.atomic import Atomic, Consistency
from std.sys.info import bit_width_of

from .utils import DoubleBuffer, circular_shift, warp_level_multi_split

# based on DeviceRadixSort from https://github.com/b0nes164/GPUSorting


comptime LANE_LOG = count_trailing_zeros(WARP_SIZE)  # 2^5 = 32 => 5


def upsweep[
    keys_dtype: DType,
    BLOCK_SIZE: Int,
    RADIX: Int,
    VEC_WIDTH: Int,
    KEYS_PER_THREAD: Int,
](
    sort: UnsafePointer[Scalar[keys_dtype], MutAnyOrigin],
    global_hist: UnsafePointer[UInt32, MutAnyOrigin],
    pass_hist: UnsafePointer[UInt32, MutAnyOrigin],
    size: Int,
    radix_shift: Scalar[keys_dtype],
):
    comptime PART_SIZE = 512 * KEYS_PER_THREAD  # = BLOCK_SIZE * KEYS_PER_THREAD = 512 * 8
    comptime VEC_PART_SIZE = PART_SIZE / VEC_WIDTH
    comptime NUM_WARPS = BLOCK_SIZE // WARP_SIZE
    comptime PADDED_RADIX = RADIX + 1
    comptime RADIX_MASK = Scalar[keys_dtype](RADIX - 1)
    comptime ordering = Consistency.MONOTONIC

    var tid = thread_idx.x
    var bid = block_idx.x
    var gdim = grid_dim.x
    var lid = lane_id()
    var wid = warp_id()

    # Shared Memory Allocation
    var s_global_hist = stack_allocation[
        NUM_WARPS * PADDED_RADIX, UInt32, address_space=AddressSpace.SHARED
    ]()

    # Initialize shared memory using the constant BLOCK_SIZE
    for i in range(tid, NUM_WARPS * PADDED_RADIX, BLOCK_SIZE):
        s_global_hist[i] = 0
    barrier()

    # Histogram Binning
    var s_warp_hist = s_global_hist + (wid * PADDED_RADIX)

    @always_inline
    def _f[width: Int](i: Int) capturing:
        var t = sort.load[width=width](i)
        t = (t >> radix_shift) & RADIX_MASK
        comptime for j in range(width):
            _ = Atomic.fetch_add[ordering=ordering](s_warp_hist + t[j], 1)

    var block_start = bid * PART_SIZE
    if bid < gdim - 1:
        for i in range(
            block_start + (tid * VEC_WIDTH),
            block_start + PART_SIZE,
            BLOCK_SIZE * VEC_WIDTH,
        ):
            _f[VEC_WIDTH](i)
    else:
        for i in range(block_start + tid, size, BLOCK_SIZE):
            _f[1](i)

    barrier()

    # Consolidate warp histograms into the first histogram
    for i in range(tid, RADIX, BLOCK_SIZE):
        var total_val: UInt32 = 0
        comptime for w in range(NUM_WARPS):
            total_val += s_global_hist[w * PADDED_RADIX + i]

        # Store for the Scan pass
        pass_hist[i * gdim + bid] = total_val

        var scan_val = warp.prefix_sum[exclusive=False](total_val)
        var shifted = circular_shift(scan_val)
        s_global_hist[i] = shifted

    barrier()

    # Final Global Accumulation
    if tid < WARP_SIZE:
        var idx = tid << LANE_LOG
        var val: UInt32 = 0
        if tid < (RADIX >> LANE_LOG):
            val = s_global_hist[idx]
        var exclusive_val = warp.prefix_sum[exclusive=True](val)
        if tid < (RADIX >> LANE_LOG):
            s_global_hist[idx] = exclusive_val
    barrier()

    for i in range(tid, RADIX, BLOCK_SIZE):
        var prev_sum: UInt32 = 0
        if lid > 0:
            prev_sum = s_global_hist[i - lid]

        var val_to_add = s_global_hist[i] + prev_sum
        var global_idx = i + Int(radix_shift << Scalar[keys_dtype](LANE_LOG))
        _ = Atomic.fetch_add[ordering=ordering](
            global_hist + global_idx, val_to_add
        )


def scan[
    BLOCK_SIZE: Int
](pass_hist: UnsafePointer[UInt32, MutAnyOrigin], thread_blocks: Int):
    var tid = Int(thread_idx.x)
    var bid = Int(block_idx.x)

    var reduction: UInt32 = 0
    var partitions_end = (thread_blocks // BLOCK_SIZE) * BLOCK_SIZE
    var digit_offset = bid * thread_blocks

    var i = tid
    while i < partitions_end:
        var val = pass_hist[i + digit_offset]

        # This returns 0 for the first element in the block
        var exclusive = block.prefix_sum[block_size=BLOCK_SIZE, exclusive=True](
            val
        )

        # Get the total sum of all 'val' in the block to update the reduction
        var tile_total = block.sum[block_size=BLOCK_SIZE, broadcast=True](val)

        # Write back: exclusive scan result + carry-over from previous tiles
        pass_hist[i + digit_offset] = exclusive + reduction

        # Update the running total for the next tile
        reduction += tile_total
        i += BLOCK_SIZE

    # Tail Handling
    var val_tail: UInt32 = 0
    var has_data = i < thread_blocks
    if has_data:
        val_tail = pass_hist[i + digit_offset]

    # Collective call: all threads must enter
    var exclusive_tail = block.prefix_sum[
        block_size=BLOCK_SIZE, exclusive=True
    ](val_tail)

    if has_data:
        pass_hist[i + digit_offset] = exclusive_tail + reduction


def downsweep[
    keys_dtype: DType,
    vals_dtype: DType,
    BITS_PER_PASS: Int,
    BLOCK_SIZE: Int,
    KEYS_PER_THREAD: Int,
    HAVE_PAYLOAD: Bool,
](
    sort: UnsafePointer[Scalar[keys_dtype], MutAnyOrigin],
    sort_payload: UnsafePointer[Scalar[vals_dtype], MutAnyOrigin],
    alt: UnsafePointer[Scalar[keys_dtype], MutAnyOrigin],
    alt_payload: UnsafePointer[Scalar[vals_dtype], MutAnyOrigin],
    global_hist: UnsafePointer[UInt32, MutAnyOrigin],
    pass_hist: UnsafePointer[UInt32, MutAnyOrigin],
    size: Int,
    radix_shift: UInt32,
):
    comptime RADIX = 2**BITS_PER_PASS
    comptime RADIX_MASK = Scalar[keys_dtype](RADIX - 1)
    comptime NUM_WARPS = BLOCK_SIZE // WARP_SIZE
    comptime WARP_PART_SIZE = WARP_SIZE * KEYS_PER_THREAD
    comptime PART_SIZE = NUM_WARPS * WARP_PART_SIZE
    comptime TOTAL_WARP_HISTS_SIZE = NUM_WARPS * RADIX

    # Shared memory allocations
    var s_warp_histograms = stack_allocation[
        PART_SIZE, UInt32, address_space=AddressSpace.SHARED
    ]()
    var s_local_histogram = stack_allocation[
        RADIX, UInt32, address_space=AddressSpace.SHARED
    ]()

    var tid = thread_idx.x
    var bid = block_idx.x
    var gdim = grid_dim.x
    var lid = lane_id()
    var wid = warp_id()

    var s_warp_hist_ptr = s_warp_histograms + (wid << BITS_PER_PASS)

    # Clear shared memory
    for i in range(tid, TOTAL_WARP_HISTS_SIZE, BLOCK_SIZE):
        s_warp_histograms[i] = 0
    barrier()

    # Load keys
    var keys = InlineArray[Scalar[keys_dtype], KEYS_PER_THREAD](
        uninitialized=True
    )
    var BIN_SUB_PART_START = wid * WARP_PART_SIZE
    var BIN_PART_START = bid * PART_SIZE
    var t_base = lid + BIN_SUB_PART_START + BIN_PART_START

    var t = t_base
    comptime for i in range(KEYS_PER_THREAD):
        if bid < gdim - 1:
            keys[i] = sort[t]
        else:
            keys[i] = sort[t] if t < size else Scalar[keys_dtype].MAX
        t += WARP_SIZE
    barrier()

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

        var sum = warp.prefix_sum[exclusive=False](reduction)
        var shifted = circular_shift(sum)
        s_warp_histograms[tid] = shifted
    barrier()

    if tid < WARP_SIZE:
        var idx = tid << LANE_LOG
        var val: UInt32 = 0
        if tid < (RADIX >> LANE_LOG):
            val = s_warp_histograms[idx]
        val = warp.prefix_sum[exclusive=True](val)
        if tid < (RADIX >> LANE_LOG):
            s_warp_histograms[idx] = val
    barrier()

    if tid < RADIX:
        var prev_sum: UInt32 = 0
        if lid > 0:
            prev_sum = s_warp_histograms[tid - lid]
        s_warp_histograms[tid] += prev_sum
    barrier()

    # Update offsets
    comptime for i in range(KEYS_PER_THREAD):
        var t2 = Int((keys[i] >> Scalar[keys_dtype](radix_shift)) & RADIX_MASK)
        if wid > 0:
            offsets[i] += (
                s_warp_histograms[wid * RADIX + t2] + s_warp_histograms[t2]
            )
        else:
            offsets[i] += s_warp_histograms[t2]

    # Load threadblock reductions
    if tid < RADIX:
        var global_offset = global_hist[
            tid + Int(radix_shift << UInt32(LANE_LOG))
        ]
        var pass_offset = pass_hist[tid * gdim + bid]
        s_local_histogram[tid] = (
            global_offset + pass_offset - s_warp_histograms[tid]
        )
    barrier()

    # Scatter keys into shared memory then to device
    comptime for i in range(KEYS_PER_THREAD):
        s_warp_histograms[Int(offsets[i])] = UInt32(keys[i])
    barrier()

    comptime if HAVE_PAYLOAD:
        var vals = InlineArray[Scalar[vals_dtype], KEYS_PER_THREAD](
            uninitialized=True
        )
        var digits = InlineArray[UInt8, KEYS_PER_THREAD](uninitialized=True)
        if bid < gdim - 1:
            comptime for i in range(KEYS_PER_THREAD):
                var t = tid + (i * BLOCK_SIZE)
                var key = s_warp_histograms[t]
                var d = (key >> radix_shift) & UInt32(RADIX_MASK)
                digits[i] = d.cast[DType.uint8]()
                alt[s_local_histogram[Int(d)] + UInt32(t)] = Scalar[keys_dtype](
                    key
                )
            barrier()

            # Load payloads into registers
            var t_payload = t_base
            comptime for i in range(KEYS_PER_THREAD):
                vals[i] = sort_payload[t_payload]
                t_payload += WARP_SIZE

            # Scatter payloads into shared memory
            comptime for i in range(KEYS_PER_THREAD):
                s_warp_histograms[Int(offsets[i])] = UInt32(vals[i])
            barrier()

            # Scatter payloads into device memory
            comptime for i in range(KEYS_PER_THREAD):
                var t = tid + (i * BLOCK_SIZE)
                var d = Int(digits[i])
                alt_payload[s_local_histogram[d] + UInt32(t)] = Scalar[
                    vals_dtype
                ](s_warp_histograms[t])

        # tail
        else:
            var final_part_size = size - BIN_PART_START
            comptime for i in range(KEYS_PER_THREAD):
                var t = tid + (i * BLOCK_SIZE)
                if t < final_part_size:
                    var key = s_warp_histograms[t]
                    var d = (key >> radix_shift) & UInt32(RADIX_MASK)
                    digits[i] = d.cast[DType.uint8]()
                    alt[s_local_histogram[Int(d)] + UInt32(t)] = Scalar[
                        keys_dtype
                    ](key)
            barrier()

            # Load payloads into registers
            var t_payload = t_base
            comptime for i in range(KEYS_PER_THREAD):
                if t_payload < size:
                    vals[i] = sort_payload[t_payload]
                t_payload += WARP_SIZE

            # Scatter payloads into shared memory
            comptime for i in range(KEYS_PER_THREAD):
                s_warp_histograms[Int(offsets[i])] = UInt32(keys[i])
            barrier()

            # Scatter payloads into device memory
            comptime for i in range(KEYS_PER_THREAD):
                var t = tid + (i * BLOCK_SIZE)
                if t < final_part_size:
                    var d = Int(digits[i])
                    alt_payload[s_local_histogram[d] + UInt32(t)] = Scalar[
                        vals_dtype
                    ](s_warp_histograms[t])

    else:
        # Scatter runs of keys into device memory (alt buffer)
        var upper_bound = PART_SIZE if bid < gdim - 1 else size - BIN_PART_START
        for i in range(tid, upper_bound, BLOCK_SIZE):
            var key = s_warp_histograms[i]
            var digit = Int((key >> radix_shift) & UInt32(RADIX_MASK))
            var dst = s_local_histogram[digit] + UInt32(i)
            alt[dst] = Scalar[keys_dtype](key)


def device_radix_sort_keys[
    dtype: DType, BITS_PER_PASS: Int = 8, KEYS_PER_THREAD: Int = 9
](ctx: DeviceContext, mut keys: DeviceBuffer[dtype], size: Int,) raises:
    comptime NUM_PASSES = bit_width_of[dtype]() / BITS_PER_PASS
    comptime RADIX = 2**BITS_PER_PASS
    comptime RADIX_MASK = RADIX - 1
    comptime GLOBAL_HIST = NUM_PASSES * RADIX
    comptime VEC_WIDTH = 4
    comptime PART_SIZE = 512 * KEYS_PER_THREAD  # = BLOCK_SIZE * KEYS_PER_THREAD = 512 * 8

    var gdim = ceildiv(size, PART_SIZE)

    var alt_keys = ctx.enqueue_create_buffer[dtype](size)

    var global_hist = ctx.enqueue_create_buffer[DType.uint32](GLOBAL_HIST)
    var pass_hist = ctx.enqueue_create_buffer[DType.uint32](gdim * RADIX)

    # Pointer management for ping-ponging
    var db_keys = DoubleBuffer(keys.unsafe_ptr(), alt_keys.unsafe_ptr())

    comptime UPSWEEP_BLOC_SIZE = 256
    comptime SCAN_BLOCK_SIZE = 256
    comptime DOWNSWEEP_BLOCK_SIZE = 512
    comptime _upsweep = upsweep[
        dtype, UPSWEEP_BLOC_SIZE, RADIX, VEC_WIDTH, KEYS_PER_THREAD
    ]
    comptime _scan = scan[SCAN_BLOCK_SIZE]
    comptime _downsweep_keys = downsweep[
        dtype,
        dtype,
        BITS_PER_PASS,
        DOWNSWEEP_BLOCK_SIZE,
        KEYS_PER_THREAD,
        False,
    ]
    var _dummy_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin]()

    comptime for pass_idx in range(NUM_PASSES):
        var radix_shift = UInt32(pass_idx * BITS_PER_PASS)

        # Reset global histogram for this pass
        global_hist.enqueue_fill(0)

        # Upsweep (Global Histogram)
        ctx.enqueue_function[_upsweep, _upsweep](
            db_keys.current,
            global_hist.unsafe_ptr(),
            pass_hist.unsafe_ptr(),
            size,
            Scalar[dtype](radix_shift),
            grid_dim=gdim,
            block_dim=UPSWEEP_BLOC_SIZE,
        )

        # Scan (Prefix Sum of partial histograms)
        ctx.enqueue_function[_scan, _scan](
            pass_hist.unsafe_ptr(),
            gdim,
            grid_dim=RADIX,
            block_dim=SCAN_BLOCK_SIZE,
        )

        # Downsweep (Scatter keys only)
        ctx.enqueue_function[_downsweep_keys, _downsweep_keys](
            db_keys.current,
            _dummy_ptr,
            db_keys.alternate,
            _dummy_ptr,
            global_hist.unsafe_ptr(),
            pass_hist.unsafe_ptr(),
            size,
            radix_shift,
            grid_dim=gdim,
            block_dim=DOWNSWEEP_BLOCK_SIZE,
        )

        # Swap buffer pointers for the next pass
        db_keys.swap()

    comptime if NUM_PASSES % 2 != 0:
        db_keys.swap()

    ctx.synchronize()


struct RadixSortWorkspace[
    keys_dtype: DType,
    vals_dtype: DType,
    BITS_PER_PASS: Int = 8,
    KEYS_PER_THREAD: Int = 9,
]:
    var alt_keys: DeviceBuffer[Self.keys_dtype]
    var alt_vals: DeviceBuffer[Self.vals_dtype]
    var global_hist: DeviceBuffer[DType.uint32]
    var pass_hist: DeviceBuffer[DType.uint32]

    def __init__(out self, ctx: DeviceContext, size: Int) raises:
        comptime NUM_PASSES = bit_width_of[
            Self.keys_dtype
        ]() / Self.BITS_PER_PASS
        comptime RADIX = 2**Self.BITS_PER_PASS
        comptime RADIX_MASK = RADIX - 1
        comptime GLOBAL_HIST = NUM_PASSES * RADIX
        comptime PART_SIZE = 512 * Self.KEYS_PER_THREAD  # = BLOCK_SIZE * KEYS_PER_THREAD = 512 * 8

        var gdim = ceildiv(size, PART_SIZE)

        self.alt_keys = ctx.enqueue_create_buffer[Self.keys_dtype](size)
        self.alt_vals = ctx.enqueue_create_buffer[Self.vals_dtype](size)

        self.global_hist = ctx.enqueue_create_buffer[DType.uint32](GLOBAL_HIST)
        self.pass_hist = ctx.enqueue_create_buffer[DType.uint32](gdim * RADIX)


def device_radix_sort_pairs[
    keys_dtype: DType,
    vals_dtype: DType,
    BITS_PER_PASS: Int = 8,
    KEYS_PER_THREAD: Int = 9,
](
    ctx: DeviceContext,
    mut workspace: RadixSortWorkspace[
        keys_dtype, vals_dtype, BITS_PER_PASS, KEYS_PER_THREAD
    ],
    mut keys: DeviceBuffer[keys_dtype],
    mut values: DeviceBuffer[vals_dtype],
    size: Int,
) raises:
    comptime NUM_PASSES = bit_width_of[keys_dtype]() / BITS_PER_PASS
    comptime RADIX = 2**BITS_PER_PASS
    comptime RADIX_MASK = RADIX - 1
    comptime GLOBAL_HIST = NUM_PASSES * RADIX
    comptime VEC_WIDTH = 4
    comptime PART_SIZE = 512 * KEYS_PER_THREAD  # = BLOCK_SIZE * KEYS_PER_THREAD = 512 * 8

    var gdim = ceildiv(size, PART_SIZE)

    # Pointer management for ping-ponging
    var db_keys = DoubleBuffer(
        keys.unsafe_ptr(), workspace.alt_keys.unsafe_ptr()
    )
    var db_vals = DoubleBuffer(
        values.unsafe_ptr(), workspace.alt_vals.unsafe_ptr()
    )
    comptime UPSWEEP_BLOC_SIZE = 256
    comptime SCAN_BLOCK_SIZE = 256
    comptime DOWNSWEEP_BLOCK_SIZE = 512
    comptime _upsweep = upsweep[
        keys_dtype, UPSWEEP_BLOC_SIZE, RADIX, VEC_WIDTH, KEYS_PER_THREAD
    ]
    comptime _scan = scan[SCAN_BLOCK_SIZE]
    comptime _downsweep_pairs = downsweep[
        keys_dtype,
        vals_dtype,
        BITS_PER_PASS,
        DOWNSWEEP_BLOCK_SIZE,
        KEYS_PER_THREAD,
        True,
    ]

    workspace.global_hist.enqueue_fill(0)
    comptime for pass_idx in range(NUM_PASSES):
        var radix_shift = UInt32(pass_idx * BITS_PER_PASS)

        # Upsweep (Global Histogram)
        ctx.enqueue_function[_upsweep, _upsweep](
            db_keys.current,
            workspace.global_hist.unsafe_ptr(),
            workspace.pass_hist.unsafe_ptr(),
            size,
            Scalar[keys_dtype](radix_shift),
            grid_dim=gdim,
            # grid_dim= ceildiv(size, UPSWEEP_BLOC_SIZE * KEYS_PER_THREAD),
            block_dim=UPSWEEP_BLOC_SIZE,
        )

        # Scan (Prefix Sum of histograms)
        ctx.enqueue_function[_scan, _scan](
            workspace.pass_hist.unsafe_ptr(),
            gdim,
            grid_dim=RADIX,
            block_dim=SCAN_BLOCK_SIZE,
        )

        # Downsweep (Scatter)
        ctx.enqueue_function[_downsweep_pairs, _downsweep_pairs](
            db_keys.current,
            db_vals.current,
            db_keys.alternate,
            db_vals.alternate,
            workspace.global_hist.unsafe_ptr(),
            workspace.pass_hist.unsafe_ptr(),
            size,
            radix_shift,
            grid_dim=gdim,
            block_dim=DOWNSWEEP_BLOCK_SIZE,
        )

        # Swap buffers for next pass
        db_keys.swap()
        db_vals.swap()

    comptime if NUM_PASSES % 2 != 0:
        db_keys.swap()
        db_vals.swap()
    ctx.synchronize()


def device_radix_sort_pairs[
    keys_dtype: DType, vals_dtype: DType
](
    ctx: DeviceContext,
    mut keys: DeviceBuffer[keys_dtype],
    mut values: DeviceBuffer[vals_dtype],
    size: Int,
) raises:
    var workspace = RadixSortWorkspace[keys_dtype, vals_dtype](ctx, size)
    device_radix_sort_pairs(ctx, workspace, keys, values, size)
