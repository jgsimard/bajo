from std.bit import pop_count, count_trailing_zeros
from std.gpu import (
    thread_idx,
    block_idx,
    block_dim,
    lane_id,
    grid_dim,
    WARP_SIZE,
)
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.gpu.primitives import warp, block
from std.gpu.sync import barrier
from std.math import ceildiv
from std.memory import stack_allocation
from std.os.atomic import Atomic

comptime N_BITS = 8
comptime RADIX = 2**N_BITS
comptime RADIX_MASK = 255


comptime SEC_RADIX_START = 1 * RADIX
comptime THIRD_RADIX_START = 2 * RADIX
comptime FOURTH_RADIX_START = 3 * RADIX
comptime GLOBAL_HIST = 4 * RADIX

comptime PART_SIZE = 7680
comptime VEC_PART_SIZE = 1920

comptime BIN_PART_SIZE = 7680
comptime BIN_HISTS_SIZE = 4096
comptime BIN_SUB_PART_SIZE = 480
comptime BIN_WARPS = 16
comptime BIN_KEYS_PER_THREAD = 15
comptime LANE_LOG = 5


@fieldwise_init
struct DoubleBuffer[dtype: DType](Copyable):
    var current: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]
    var alternate: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]

    def swap(mut self):
        swap(self.current, self.alternate)


# based on DeviceRadixSort from https://github.com/b0nes164/GPUSorting


def upsweep[
    BLOCK_SIZE: Int
](
    sort: UnsafePointer[UInt32, MutAnyOrigin],
    global_hist: UnsafePointer[UInt32, MutAnyOrigin],
    pass_hist: UnsafePointer[UInt32, MutAnyOrigin],
    size: Int,
    radix_shift: UInt32,
):
    comptime NUM_WARPS = BLOCK_SIZE // WARP_SIZE
    var tid = Int(thread_idx.x)
    var bid = Int(block_idx.x)
    var gdim = Int(grid_dim.x)
    var lid = Int(lane_id())
    var warp_id = tid // WARP_SIZE

    # Shared Memory Allocation
    comptime PADDED_RADIX = RADIX + 1
    var s_global_hist = stack_allocation[
        NUM_WARPS * PADDED_RADIX, UInt32, address_space=AddressSpace.SHARED
    ]()

    # Initialize shared memory using the constant BLOCK_SIZE
    for i in range(tid, NUM_WARPS * PADDED_RADIX, BLOCK_SIZE):
        s_global_hist[i] = 0
    barrier()

    # Histogram Binning
    var s_warp_hist = s_global_hist + (warp_id * PADDED_RADIX)

    if bid < gdim - 1:
        for i in range(
            tid + (bid * VEC_PART_SIZE), (bid + 1) * VEC_PART_SIZE, BLOCK_SIZE
        ):
            var t = sort.load[width=4](i * 4)
            t = (t >> radix_shift) & RADIX_MASK
            _ = Atomic.fetch_add(s_warp_hist + t[0], 1)
            _ = Atomic.fetch_add(s_warp_hist + t[1], 1)
            _ = Atomic.fetch_add(s_warp_hist + t[2], 1)
            _ = Atomic.fetch_add(s_warp_hist + t[3], 1)

    # tail
    else:
        for j in range(tid + (bid * PART_SIZE), size, BLOCK_SIZE):
            var t = sort[j]
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
        # Circular shift: lane 0 pulls from 31, lane 1 pulls from 0, etc.
        var shifted = warp.shuffle_idx(scan_val, UInt32((lid + 31) & 31))
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
        var global_idx = i + Int(radix_shift << LANE_LOG)
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


def downsweep_keys_only(
    sort: UnsafePointer[UInt32, MutAnyOrigin],
    alt: UnsafePointer[UInt32, MutAnyOrigin],
    global_hist: UnsafePointer[UInt32, MutAnyOrigin],
    pass_hist: UnsafePointer[UInt32, MutAnyOrigin],
    size: Int,
    radix_shift: UInt32,
):
    # Shared memory allocations
    var s_warp_histograms = stack_allocation[
        BIN_PART_SIZE, UInt32, address_space=AddressSpace.SHARED
    ]()
    var s_local_histogram = stack_allocation[
        RADIX, UInt32, address_space=AddressSpace.SHARED
    ]()

    var tid = Int(thread_idx.x)
    var bdim = Int(block_dim.x)
    var bid = Int(block_idx.x)
    var gdim = Int(grid_dim.x)
    var lid = Int(lane_id())
    var warp_id = tid >> LANE_LOG

    var s_warpHist_ptr = s_warp_histograms + (warp_id << N_BITS)

    # Clear shared memory
    for i in range(tid, BIN_HISTS_SIZE, bdim):
        s_warp_histograms[i] = 0
    barrier()

    # Load keys
    var keys = InlineArray[UInt32, BIN_KEYS_PER_THREAD](uninitialized=True)
    var BIN_SUB_PART_START = warp_id * BIN_SUB_PART_SIZE
    var BIN_PART_START = bid * BIN_PART_SIZE

    if bid < gdim - 1:
        var t = Int(lane_id()) + BIN_SUB_PART_START + BIN_PART_START
        comptime for i in range(BIN_KEYS_PER_THREAD):
            keys[i] = sort[t]
            t += WARP_SIZE

    if bid == gdim - 1:
        var t = Int(lane_id()) + BIN_SUB_PART_START + BIN_PART_START
        comptime for i in range(BIN_KEYS_PER_THREAD):
            keys[i] = sort[t] if t < Int(size) else 0xFFFFFFFF
            t += WARP_SIZE
    barrier()

    # Warp-Level Multi-Split (WLMS)
    var offsets = SIMD[DType.uint32, 16](0)
    var lane_mask_lt = (UInt32(1) << UInt32(lane_id())) - 1

    comptime for i in range(BIN_KEYS_PER_THREAD):
        var warpFlags: UInt32 = 0xFFFFFFFF
        var key = keys[i]

        comptime for k in range(N_BITS):
            var t2 = ((key >> (radix_shift + UInt32(k))) & 1) == 1
            var ballot = warp.vote[DType.uint32](t2)
            var match_mask = ballot if t2 else ~ballot
            warpFlags &= match_mask

        var bits = pop_count(warpFlags & lane_mask_lt)
        var preIncrementVal: UInt32 = 0

        # Leader executes atomic increment
        if bits == 0:
            var digit = (key >> radix_shift) & RADIX_MASK
            var count = pop_count(warpFlags)
            preIncrementVal = Atomic.fetch_add(s_warpHist_ptr + digit, count)

        # Broadcast the allocated offset from the leader thread
        var leader_lane = count_trailing_zeros(warpFlags)
        preIncrementVal = warp.shuffle_idx(preIncrementVal, UInt32(leader_lane))

        offsets[i] = preIncrementVal + UInt32(bits)
    barrier()

    # Exclusive prefix sum up the warp histograms
    if tid < RADIX:
        var reduction = s_warp_histograms[tid]
        for i in range(tid + RADIX, BIN_HISTS_SIZE, RADIX):
            reduction += s_warp_histograms[i]
            s_warp_histograms[i] = reduction - s_warp_histograms[i]

        var sum = warp.prefix_sum[exclusive=False](reduction)
        var shifted = warp.shuffle_idx(sum, UInt32((lane_id() + 31) & 31))
        s_warp_histograms[tid] = shifted
    barrier()

    if tid < 32:
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
        if lane_id() > 0:
            prev_sum = s_warp_histograms[tid - Int(lane_id())]
        s_warp_histograms[tid] += prev_sum
    barrier()

    # Update offsets
    if warp_id > 0:
        comptime for i in range(BIN_KEYS_PER_THREAD):
            var t2 = Int((keys[i] >> radix_shift) & RADIX_MASK)
            offsets[i] += (
                s_warp_histograms[warp_id * RADIX + t2] + s_warp_histograms[t2]
            )
    else:
        comptime for i in range(BIN_KEYS_PER_THREAD):
            var t2 = Int((keys[i] >> radix_shift) & RADIX_MASK)
            offsets[i] += s_warp_histograms[t2]

    # Load in threadblock reductions from global/pass histograms
    if tid < RADIX:
        var global_offset = global_hist[tid + Int(radix_shift << 5)]
        var pass_offset = pass_hist[tid * gdim + bid]
        s_local_histogram[tid] = (
            global_offset + pass_offset - s_warp_histograms[tid]
        )
    barrier()

    # Scatter keys into shared memory
    comptime for i in range(BIN_KEYS_PER_THREAD):
        s_warp_histograms[Int(offsets[i])] = keys[i]
    barrier()

    # Scatter runs of keys into device memory (alt buffer)
    if bid < gdim - 1:
        for i in range(tid, BIN_PART_SIZE, bdim):
            var key = s_warp_histograms[i]
            var digit = Int((key >> radix_shift) & RADIX_MASK)
            var dst = s_local_histogram[digit] + UInt32(i)
            alt[dst] = key

    if bid == gdim - 1:
        var finalPartSize = Int(size) - BIN_PART_START
        for i in range(tid, finalPartSize, bdim):
            var key = s_warp_histograms[i]
            var digit = Int((key >> radix_shift) & RADIX_MASK)
            var dst = s_local_histogram[digit] + UInt32(i)
            alt[dst] = key


def downsweep_pairs(
    sort: UnsafePointer[UInt32, MutAnyOrigin],
    sort_payload: UnsafePointer[UInt32, MutAnyOrigin],
    alt: UnsafePointer[UInt32, MutAnyOrigin],
    alt_payload: UnsafePointer[UInt32, MutAnyOrigin],
    global_hist: UnsafePointer[UInt32, MutAnyOrigin],
    pass_hist: UnsafePointer[UInt32, MutAnyOrigin],
    size: Int,
    radix_shift: UInt32,
):
    # Shared memory allocations
    var s_warp_histograms = stack_allocation[
        BIN_PART_SIZE, UInt32, address_space=AddressSpace.SHARED
    ]()
    var s_local_histogram = stack_allocation[
        RADIX, UInt32, address_space=AddressSpace.SHARED
    ]()

    var tid = Int(thread_idx.x)
    var bdim = Int(block_dim.x)
    var bid = Int(block_idx.x)
    var gdim = Int(grid_dim.x)
    var lid = Int(lane_id())
    var warp_id = tid >> LANE_LOG

    var s_warpHist_ptr = s_warp_histograms + (warp_id << N_BITS)

    # Clear shared memory
    for i in range(tid, BIN_HISTS_SIZE, bdim):
        s_warp_histograms[i] = 0
    barrier()

    # Load keys
    var keys = InlineArray[UInt32, BIN_KEYS_PER_THREAD](uninitialized=True)
    var BIN_SUB_PART_START = warp_id * BIN_SUB_PART_SIZE
    var BIN_PART_START = bid * BIN_PART_SIZE

    if bid < gdim - 1:
        var t = Int(lane_id()) + BIN_SUB_PART_START + BIN_PART_START
        comptime for i in range(BIN_KEYS_PER_THREAD):
            keys[i] = sort[t]
            t += 32

    if bid == gdim - 1:
        var t = Int(lane_id()) + BIN_SUB_PART_START + BIN_PART_START
        comptime for i in range(BIN_KEYS_PER_THREAD):
            keys[i] = sort[t] if t < size else 0xFFFFFFFF
            t += 32
    barrier()

    # Warp-Level Multi-Split (WLMS)
    var offsets = SIMD[DType.uint32, 16](0)
    var lane_mask_lt = (UInt32(1) << UInt32(lane_id())) - 1

    comptime for i in range(BIN_KEYS_PER_THREAD):
        var warpFlags: UInt32 = 0xFFFFFFFF
        var key = keys[i]

        comptime for k in range(N_BITS):
            var t2 = ((key >> (radix_shift + UInt32(k))) & 1) == 1
            var ballot = warp.vote[DType.uint32](t2)
            var match_mask = ballot if t2 else ~ballot
            warpFlags &= match_mask

        var bits = pop_count(warpFlags & lane_mask_lt)
        var pre_increment_val: UInt32 = 0

        if bits == 0:
            var digit = Int((key >> radix_shift) & RADIX_MASK)
            var count = pop_count(warpFlags)
            pre_increment_val = Atomic.fetch_add(s_warpHist_ptr + digit, count)

        var leader_lane = count_trailing_zeros(warpFlags)
        pre_increment_val = warp.shuffle_idx(
            pre_increment_val, UInt32(leader_lane)
        )

        offsets[i] = pre_increment_val + UInt32(bits)
    barrier()

    # Exclusive prefix sum up the warp histograms
    if tid < RADIX:
        var reduction = s_warp_histograms[tid]
        for i in range(tid + RADIX, BIN_HISTS_SIZE, RADIX):
            reduction += s_warp_histograms[i]
            s_warp_histograms[i] = reduction - s_warp_histograms[i]

        var sum = warp.prefix_sum[exclusive=False](reduction)
        var shifted = warp.shuffle_idx(sum, UInt32((lane_id() + 31) & 31))
        s_warp_histograms[tid] = shifted
    barrier()

    if tid < 32:
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
        if lane_id() > 0:
            prev_sum = s_warp_histograms[tid - Int(lane_id())]
        s_warp_histograms[tid] += prev_sum
    barrier()

    # Update offsets
    if warp_id > 0:
        comptime for i in range(BIN_KEYS_PER_THREAD):
            var t2 = Int((keys[i] >> radix_shift) & RADIX_MASK)
            offsets[i] += (
                s_warp_histograms[warp_id * RADIX + t2] + s_warp_histograms[t2]
            )
    else:
        comptime for i in range(BIN_KEYS_PER_THREAD):
            var t2 = Int((keys[i] >> radix_shift) & RADIX_MASK)
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
    comptime for i in range(BIN_KEYS_PER_THREAD):
        s_warp_histograms[Int(offsets[i])] = keys[i]
    barrier()

    var digits = SIMD[DType.uint8, 16](0)
    if bid < gdim - 1:
        comptime for i in range(BIN_KEYS_PER_THREAD):
            var t = tid + (i * bdim)
            var key = s_warp_histograms[t]
            var d = (key >> radix_shift) & RADIX_MASK
            digits[i] = d.cast[DType.uint8]()
            alt[s_local_histogram[Int(d)] + UInt32(t)] = key
        barrier()

        # Load payloads into registers
        var t_payload = Int(lane_id()) + BIN_SUB_PART_START + BIN_PART_START
        comptime for i in range(BIN_KEYS_PER_THREAD):
            keys[i] = sort_payload[t_payload]
            t_payload += 32

        # Scatter payloads into shared memory
        comptime for i in range(BIN_KEYS_PER_THREAD):
            s_warp_histograms[Int(offsets[i])] = keys[i]
        barrier()

        # Scatter payloads into device memory
        comptime for i in range(BIN_KEYS_PER_THREAD):
            var t = tid + (i * bdim)
            var d = Int(digits[i])
            alt_payload[s_local_histogram[d] + UInt32(t)] = s_warp_histograms[t]

    # tail
    else:
        var final_part_size = Int(size) - BIN_PART_START
        comptime for i in range(BIN_KEYS_PER_THREAD):
            var t = tid + (i * bdim)
            if t < final_part_size:
                var key = s_warp_histograms[t]
                var d = (key >> radix_shift) & RADIX_MASK
                digits[i] = d.cast[DType.uint8]()
                alt[s_local_histogram[Int(d)] + UInt32(t)] = key
        barrier()

        # Load payloads into registers
        var t_payload = lid + BIN_SUB_PART_START + BIN_PART_START
        comptime for i in range(BIN_KEYS_PER_THREAD):
            if t_payload < Int(size):
                keys[i] = sort_payload[t_payload]
            t_payload += 32

        # Scatter payloads into shared memory
        comptime for i in range(BIN_KEYS_PER_THREAD):
            s_warp_histograms[Int(offsets[i])] = keys[i]
        barrier()

        # Scatter payloads into device memory
        comptime for i in range(BIN_KEYS_PER_THREAD):
            var t = tid + (i * bdim)
            if t < final_part_size:
                var d = Int(digits[i])
                alt_payload[
                    s_local_histogram[d] + UInt32(t)
                ] = s_warp_histograms[t]


def device_radix_sort_keys(
    ctx: DeviceContext,
    mut keys: DeviceBuffer[DType.uint32],
    size: Int,
) raises:
    """Orchestrates 4 passes of the Reduce-then-Scan Radix Sort for keys only.
    """
    comptime dtype = DType.uint32
    var gdim = ceildiv(size, BIN_PART_SIZE)

    var alt_keys = ctx.enqueue_create_buffer[dtype](size)

    var global_hist = ctx.enqueue_create_buffer[dtype](GLOBAL_HIST)
    var pass_hist = ctx.enqueue_create_buffer[dtype](gdim * RADIX)

    # Pointer management for ping-ponging
    var db_keys = DoubleBuffer(keys.unsafe_ptr(), alt_keys.unsafe_ptr())

    comptime UPSWEEP_BLOC_SIZE = 256
    comptime SCAN_BLOCK_SIZE = 256
    comptime _upsweep = upsweep[UPSWEEP_BLOC_SIZE]
    comptime _scan = scan[SCAN_BLOCK_SIZE]

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
        ctx.enqueue_function[downsweep_keys_only, downsweep_keys_only](
            db_keys.current,
            db_keys.alternate,
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


def device_radix_sort_pairs[
    dtype: DType
](
    ctx: DeviceContext,
    mut keys: DeviceBuffer[dtype],
    mut values: DeviceBuffer[dtype],
    size: Int,
) raises:
    """Orchestrates 4 passes of the Reduce-then-Scan Radix Sort."""
    var gdim = ceildiv(size, BIN_PART_SIZE)

    var alt_keys = ctx.enqueue_create_buffer[dtype](size)
    var alt_vals = ctx.enqueue_create_buffer[dtype](size)

    var global_hist = ctx.enqueue_create_buffer[dtype](GLOBAL_HIST)
    var pass_hist = ctx.enqueue_create_buffer[dtype](gdim * RADIX)

    # Pointer management for ping-ponging
    var db_keys = DoubleBuffer(keys.unsafe_ptr(), alt_keys.unsafe_ptr())
    var db_vals = DoubleBuffer(values.unsafe_ptr(), alt_vals.unsafe_ptr())

    comptime UPSWEEP_BLOC_SIZE = 256
    comptime SCAN_BLOCK_SIZE = 256
    comptime _upsweep = upsweep[UPSWEEP_BLOC_SIZE]
    comptime _scan = scan[SCAN_BLOCK_SIZE]

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

        # Scan (Prefix Sum of histograms)
        ctx.enqueue_function[_scan, _scan](
            pass_hist.unsafe_ptr(),
            gdim,
            grid_dim=RADIX,
            block_dim=SCAN_BLOCK_SIZE,
        )

        # Downsweep (Scatter)
        ctx.enqueue_function[downsweep_pairs, downsweep_pairs](
            db_keys.current,
            db_vals.current,
            db_keys.alternate,
            db_vals.alternate,
            global_hist.unsafe_ptr(),
            pass_hist.unsafe_ptr(),
            size,
            radix_shift,
            grid_dim=gdim,
            block_dim=512,
        )

        # Swap buffers for next pass
        db_keys.swap()
        db_vals.swap()

    # Final result should be in the original buffers
    # (Since 4 passes is an even number, d_keys == keys.unsafe_ptr())
    ctx.synchronize()
