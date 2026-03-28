from std.bit import pop_count, count_trailing_zeros
from std.gpu import (
    thread_idx_int as thread_idx,
    block_idx_int as block_idx,
    block_dim,
    lane_id_int as lane_id,
    grid_dim_int as grid_dim,
    WARP_SIZE,
    warp_id_int as warp_id,
)
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.gpu.primitives import warp, block
from std.gpu.sync import barrier
from std.math import ceildiv
from std.memory import stack_allocation
from std.os.atomic import Atomic

# based on DeviceRadixSort from https://github.com/b0nes164/GPUSorting

comptime N_BITS = 8
comptime RADIX = 2**N_BITS
comptime RADIX_MASK = RADIX - 1
comptime WARP_MASK = WARP_SIZE - 1

comptime GLOBAL_HIST = 4 * RADIX
comptime VEC_WIDTH = 4
comptime KEYS_PER_THREAD = 8
comptime PART_SIZE = 512 * KEYS_PER_THREAD  # = BLOCK_SIZE * KEYS_PER_THREAD = 512 * 8
comptime WARP_PART_SIZE = WARP_SIZE * KEYS_PER_THREAD
comptime LANE_LOG = count_trailing_zeros(WARP_SIZE)  # 2^5 = 32 => 5


@fieldwise_init
struct DoubleBuffer[dtype: DType](Copyable):
    var current: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]
    var alternate: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]

    def swap(mut self):
        swap(self.current, self.alternate)


def circular_shift(val: UInt32, lid: UInt32) -> UInt32:
    return warp.shuffle_idx(
        val, UInt32((lid + UInt32(WARP_MASK)) & UInt32(WARP_MASK))
    )


def upsweep[
    BLOCK_SIZE: Int, VEC_WIDTH: Int = 4
](
    sort: UnsafePointer[UInt32, MutAnyOrigin],
    global_hist: UnsafePointer[UInt32, MutAnyOrigin],
    pass_hist: UnsafePointer[UInt32, MutAnyOrigin],
    size: Int,
    radix_shift: UInt32,
):
    comptime VEC_PART_SIZE = PART_SIZE / VEC_WIDTH
    comptime NUM_WARPS = BLOCK_SIZE // WARP_SIZE
    comptime PADDED_RADIX = RADIX + 1

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

    if bid < gdim - 1:
        for i in range(
            tid + (bid * VEC_PART_SIZE), (bid + 1) * VEC_PART_SIZE, BLOCK_SIZE
        ):
            var t = sort.load[width=VEC_WIDTH](i * VEC_WIDTH)
            t = (t >> radix_shift) & RADIX_MASK
            comptime for j in range(VEC_WIDTH):
                _ = Atomic.fetch_add(s_warp_hist + t[j], 1)

    # tail
    else:
        for i in range(tid + (bid * PART_SIZE), size, BLOCK_SIZE):
            var t = sort[i]
            t = (t >> radix_shift) & RADIX_MASK
            _ = Atomic.fetch_add(s_warp_hist + t, 1)

    barrier()

    # Consolidate warp histograms into the first histogram
    for i in range(tid, RADIX, BLOCK_SIZE):
        var total_val: UInt32 = 0
        comptime for w in range(NUM_WARPS):
            total_val += s_global_hist[w * PADDED_RADIX + i]

        # Store for the Scan pass
        pass_hist[i * gdim + bid] = total_val

        var scan_val = warp.prefix_sum[exclusive=False](total_val)
        var shifted = circular_shift(scan_val, UInt32(lid))
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
        var global_idx = i + Int(radix_shift << UInt32(LANE_LOG))
        _ = Atomic.fetch_add(global_hist + global_idx, val_to_add)


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


@always_inline
def warp_level_multi_split(
    keys: InlineArray[UInt32, KEYS_PER_THREAD],
    lid: Int,
    radix_shift: UInt32,
    s_warp_hist_ptr: UnsafePointer[
        UInt32, MutExternalOrigin, address_space=AddressSpace.SHARED
    ],
) -> InlineArray[UInt32, KEYS_PER_THREAD]:
    comptime mask_dtype = DType.uint64 if WARP_SIZE > 32 else DType.uint32
    comptime MaskInt = SIMD[mask_dtype, 1]

    var offsets = InlineArray[UInt32, KEYS_PER_THREAD](uninitialized=True)
    var lane_mask_lt = (MaskInt(1) << MaskInt(lid)) - 1

    comptime for i in range(KEYS_PER_THREAD):
        var warp_flags: MaskInt = ~MaskInt(0)
        var key = keys[i]

        comptime for k in range(N_BITS):
            var t2 = ((key >> (radix_shift + UInt32(k))) & 1) == 1
            var ballot = warp.vote[mask_dtype](t2)
            var match_mask = ballot if t2 else ~ballot
            warp_flags &= match_mask

        var bits = UInt32(pop_count(warp_flags & lane_mask_lt))
        var pre_increment_val: UInt32 = 0

        if bits == 0:
            var digit = Int((key >> radix_shift) & RADIX_MASK)
            var count = UInt32(pop_count(warp_flags))
            pre_increment_val = Atomic.fetch_add(s_warp_hist_ptr + digit, count)

        var leader_lane = count_trailing_zeros(warp_flags)
        pre_increment_val = warp.shuffle_idx(
            pre_increment_val, UInt32(leader_lane)
        )

        offsets[i] = pre_increment_val + UInt32(bits)
    return offsets^


def downsweep[
    BLOCK_SIZE: Int, HAVE_PAYLOAD: Bool
](
    sort: UnsafePointer[UInt32, MutAnyOrigin],
    sort_payload: UnsafePointer[UInt32, MutAnyOrigin],
    alt: UnsafePointer[UInt32, MutAnyOrigin],
    alt_payload: UnsafePointer[UInt32, MutAnyOrigin],
    global_hist: UnsafePointer[UInt32, MutAnyOrigin],
    pass_hist: UnsafePointer[UInt32, MutAnyOrigin],
    size: Int,
    radix_shift: UInt32,
):
    comptime NUM_WARPS = BLOCK_SIZE // WARP_SIZE
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

    var s_warp_hist_ptr = s_warp_histograms + (wid << N_BITS)

    # Clear shared memory
    for i in range(tid, TOTAL_WARP_HISTS_SIZE, BLOCK_SIZE):
        s_warp_histograms[i] = 0
    barrier()

    # Load keys
    var keys = InlineArray[UInt32, KEYS_PER_THREAD](uninitialized=True)
    var BIN_SUB_PART_START = wid * WARP_PART_SIZE
    var BIN_PART_START = bid * PART_SIZE
    var t_base = lid + BIN_SUB_PART_START + BIN_PART_START

    var t = t_base
    comptime for i in range(KEYS_PER_THREAD):
        if bid < gdim - 1:
            keys[i] = sort[t]
        else:
            keys[i] = sort[t] if t < size else 0xFFFFFFFF
        t += WARP_SIZE
    barrier()

    # Warp-Level Multi-Split (WLMS)
    var offsets = warp_level_multi_split(
        keys, lid, radix_shift, s_warp_hist_ptr
    )
    barrier()

    # Exclusive prefix sum up the warp histograms
    if tid < RADIX:
        var reduction = s_warp_histograms[tid]
        for i in range(tid + RADIX, TOTAL_WARP_HISTS_SIZE, RADIX):
            reduction += s_warp_histograms[i]
            s_warp_histograms[i] = reduction - s_warp_histograms[i]

        var sum = warp.prefix_sum[exclusive=False](reduction)
        var shifted = circular_shift(sum, UInt32(lid))
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
        var t2 = Int((keys[i] >> radix_shift) & RADIX_MASK)
        if wid > 0:
            offsets[i] += (
                s_warp_histograms[wid * RADIX + t2] + s_warp_histograms[t2]
            )
        else:
            offsets[i] += s_warp_histograms[t2]

    # Load threadblock reductions
    if tid < RADIX:
        var global_offset = global_hist[tid + Int(radix_shift << 5)]
        var pass_offset = pass_hist[tid * gdim + bid]
        s_local_histogram[tid] = (
            global_offset + pass_offset - s_warp_histograms[tid]
        )
    barrier()

    # Scatter keys into shared memory then to device
    comptime for i in range(KEYS_PER_THREAD):
        s_warp_histograms[Int(offsets[i])] = keys[i]
    barrier()

    comptime if HAVE_PAYLOAD:
        var digits = InlineArray[UInt8, KEYS_PER_THREAD](uninitialized=True)
        if bid < gdim - 1:
            comptime for i in range(KEYS_PER_THREAD):
                var t = tid + (i * BLOCK_SIZE)
                var key = s_warp_histograms[t]
                var d = (key >> radix_shift) & RADIX_MASK
                digits[i] = d.cast[DType.uint8]()
                alt[s_local_histogram[Int(d)] + UInt32(t)] = key
            barrier()

            # Load payloads into registers
            var t_payload = t_base
            comptime for i in range(KEYS_PER_THREAD):
                keys[i] = sort_payload[t_payload]
                t_payload += WARP_SIZE

            # Scatter payloads into shared memory
            comptime for i in range(KEYS_PER_THREAD):
                s_warp_histograms[Int(offsets[i])] = keys[i]
            barrier()

            # Scatter payloads into device memory
            comptime for i in range(KEYS_PER_THREAD):
                var t = tid + (i * BLOCK_SIZE)
                var d = Int(digits[i])
                alt_payload[
                    s_local_histogram[d] + UInt32(t)
                ] = s_warp_histograms[t]

        # tail
        else:
            var final_part_size = size - BIN_PART_START
            comptime for i in range(KEYS_PER_THREAD):
                var t = tid + (i * BLOCK_SIZE)
                if t < final_part_size:
                    var key = s_warp_histograms[t]
                    var d = (key >> radix_shift) & RADIX_MASK
                    digits[i] = d.cast[DType.uint8]()
                    alt[s_local_histogram[Int(d)] + UInt32(t)] = key
            barrier()

            # Load payloads into registers
            var t_payload = t_base
            comptime for i in range(KEYS_PER_THREAD):
                if t_payload < size:
                    keys[i] = sort_payload[t_payload]
                t_payload += WARP_SIZE

            # Scatter payloads into shared memory
            comptime for i in range(KEYS_PER_THREAD):
                s_warp_histograms[Int(offsets[i])] = keys[i]
            barrier()

            # Scatter payloads into device memory
            comptime for i in range(KEYS_PER_THREAD):
                var t = tid + (i * BLOCK_SIZE)
                if t < final_part_size:
                    var d = Int(digits[i])
                    alt_payload[
                        s_local_histogram[d] + UInt32(t)
                    ] = s_warp_histograms[t]

    else:
        # Scatter runs of keys into device memory (alt buffer)
        if bid < gdim - 1:
            for i in range(tid, PART_SIZE, BLOCK_SIZE):
                var key = s_warp_histograms[i]
                var digit = Int((key >> radix_shift) & RADIX_MASK)
                var dst = s_local_histogram[digit] + UInt32(i)
                alt[dst] = key

        else:
            for i in range(tid, size - BIN_PART_START, BLOCK_SIZE):
                var key = s_warp_histograms[i]
                var digit = Int((key >> radix_shift) & RADIX_MASK)
                var dst = s_local_histogram[digit] + UInt32(i)
                alt[dst] = key


def device_radix_sort_keys(
    ctx: DeviceContext,
    mut keys: DeviceBuffer[DType.uint32],
    size: Int,
) raises:
    """Orchestrates 4 passes of the Reduce-then-Scan Radix Sort for keys only.
    """
    comptime dtype = DType.uint32
    var gdim = ceildiv(size, PART_SIZE)

    var alt_keys = ctx.enqueue_create_buffer[dtype](size)

    var global_hist = ctx.enqueue_create_buffer[dtype](GLOBAL_HIST)
    var pass_hist = ctx.enqueue_create_buffer[dtype](gdim * RADIX)

    # Pointer management for ping-ponging
    var db_keys = DoubleBuffer(keys.unsafe_ptr(), alt_keys.unsafe_ptr())

    comptime UPSWEEP_BLOC_SIZE = 256
    comptime SCAN_BLOCK_SIZE = 256
    comptime DOWNSWEEP_BLOCK_SIZE = 512
    comptime _upsweep = upsweep[UPSWEEP_BLOC_SIZE, VEC_WIDTH]
    comptime _scan = scan[SCAN_BLOCK_SIZE]
    comptime _downsweep_keys = downsweep[DOWNSWEEP_BLOCK_SIZE, False]
    var _dummy_ptr = UnsafePointer[UInt32, MutAnyOrigin]()

    comptime for pass_idx in range(4):
        var radix_shift = UInt32(pass_idx * 8)

        # Reset global histogram for this pass
        global_hist.enqueue_fill(0)

        # Upsweep (Global Histogram)
        ctx.enqueue_function[_upsweep, _upsweep](
            db_keys.current,
            global_hist.unsafe_ptr(),
            pass_hist.unsafe_ptr(),
            size,
            radix_shift,
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
            block_dim=512,
        )

        # Swap buffer pointers for the next pass
        db_keys.swap()

    # With 4 passes (even), the final sorted data is naturally back in the 'keys' buffer
    ctx.synchronize()


struct RadixSortWorkspace[dtype: DType]:
    var alt_keys: DeviceBuffer[Self.dtype]
    var alt_vals: DeviceBuffer[Self.dtype]
    var global_hist: DeviceBuffer[Self.dtype]
    var pass_hist: DeviceBuffer[Self.dtype]

    def __init__(out self, ctx: DeviceContext, size: Int) raises:
        var gdim = ceildiv(size, PART_SIZE)

        self.alt_keys = ctx.enqueue_create_buffer[Self.dtype](size)
        self.alt_vals = ctx.enqueue_create_buffer[Self.dtype](size)

        self.global_hist = ctx.enqueue_create_buffer[Self.dtype](GLOBAL_HIST)
        self.pass_hist = ctx.enqueue_create_buffer[Self.dtype](gdim * RADIX)


def device_radix_sort_pairs[
    dtype: DType
](
    ctx: DeviceContext,
    mut workspace: RadixSortWorkspace[dtype],
    mut keys: DeviceBuffer[dtype],
    mut values: DeviceBuffer[dtype],
    size: Int,
) raises:
    """Orchestrates 4 passes of the Reduce-then-Scan Radix Sort."""
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
    comptime _upsweep = upsweep[UPSWEEP_BLOC_SIZE, VEC_WIDTH]
    comptime _scan = scan[SCAN_BLOCK_SIZE]
    comptime _downsweep_pairs = downsweep[DOWNSWEEP_BLOCK_SIZE, True]

    workspace.global_hist.enqueue_fill(0)
    comptime for pass_idx in range(4):
        var radix_shift = UInt32(pass_idx * 8)

        # Upsweep (Global Histogram)
        ctx.enqueue_function[_upsweep, _upsweep](
            db_keys.current,
            workspace.global_hist.unsafe_ptr(),
            workspace.pass_hist.unsafe_ptr(),
            size,
            radix_shift,
            grid_dim=gdim,
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

    # Final result should be in the original buffers
    # (Since 4 passes is an even number, d_keys == keys.unsafe_ptr())
    ctx.synchronize()


def device_radix_sort_pairs[
    dtype: DType
](
    ctx: DeviceContext,
    mut keys: DeviceBuffer[dtype],
    mut values: DeviceBuffer[dtype],
    size: Int,
) raises:
    var workspace = RadixSortWorkspace[dtype](ctx, size)
    device_radix_sort_pairs(ctx, workspace, keys, values, size)
