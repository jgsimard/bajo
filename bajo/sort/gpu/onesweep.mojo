from std.bit import pop_count, count_trailing_zeros
from std.gpu import (
    thread_idx,
    block_idx,
    block_dim,
    lane_id,
    grid_dim,
    WARP_SIZE,
)
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from std.gpu.primitives import warp
from std.gpu.sync import barrier
from std.gpu.intrinsics import load_volatile
from std.math import ceildiv
from std.memory import stack_allocation, bitcast
from std.os.atomic import Atomic
from std.sys.info import bit_width_of

from .utils import DoubleBuffer, circular_shift, warp_level_multi_split

# Based on OneSweep from https://github.com/b0nes164/GPUSorting
# Research by Andy Adinets & Duane Merrill (Nvidia Corporation)

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
    comptime G_HIST_PART_SIZE = BLOCK_SIZE * ITEMS_PER_THREAD
    comptime G_HIST_VEC_SIZE = G_HIST_PART_SIZE / VEC_WIDTH
    comptime PADDED_RADIX = RADIX * 2

    var tid = thread_idx.x
    var bid = block_idx.x
    var gdim = grid_dim.x
    var wid = tid // 64

    comptime _stack = stack_allocation[
        PADDED_RADIX, UInt32, address_space=AddressSpace.SHARED
    ]
    var s_hist_0 = _stack()
    var s_hist_1 = _stack()
    var s_hist_2 = _stack()
    var s_hist_3 = _stack()

    for i in range(tid, PADDED_RADIX, BLOCK_SIZE):
        s_hist_0[i] = 0
        s_hist_1[i] = 0
        s_hist_2[i] = 0
        s_hist_3[i] = 0
    barrier()

    # 64 threads : 1 histogram in shared memory
    var wave_offset = wid * RADIX
    var s_w_0 = s_hist_0 + wave_offset
    var s_w_1 = s_hist_1 + wave_offset
    var s_w_2 = s_hist_2 + wave_offset
    var s_w_3 = s_hist_3 + wave_offset

    if bid < gdim - 1:
        var part_start = bid * G_HIST_VEC_SIZE
        var part_end = (bid + 1) * G_HIST_VEC_SIZE
        for i in range(tid + part_start, part_end, BLOCK_SIZE):
            # this is slower. why ?
            # var ptr_u8 = sort.bitcast[UInt8]()
            # var t = ptr_u8.load[width=16](i * 4 * 4)
            var t_u32 = sort.load[width=4](i * 4)
            var t = bitcast[DType.uint8, 16](t_u32)

            _ = Atomic.fetch_add(s_w_0 + Int(t[0]), 1)
            _ = Atomic.fetch_add(s_w_1 + Int(t[1]), 1)
            _ = Atomic.fetch_add(s_w_2 + Int(t[2]), 1)
            _ = Atomic.fetch_add(s_w_3 + Int(t[3]), 1)

            _ = Atomic.fetch_add(s_w_0 + Int(t[4]), 1)
            _ = Atomic.fetch_add(s_w_1 + Int(t[5]), 1)
            _ = Atomic.fetch_add(s_w_2 + Int(t[6]), 1)
            _ = Atomic.fetch_add(s_w_3 + Int(t[7]), 1)

            _ = Atomic.fetch_add(s_w_0 + Int(t[8]), 1)
            _ = Atomic.fetch_add(s_w_1 + Int(t[9]), 1)
            _ = Atomic.fetch_add(s_w_2 + Int(t[10]), 1)
            _ = Atomic.fetch_add(s_w_3 + Int(t[11]), 1)

            _ = Atomic.fetch_add(s_w_0 + Int(t[12]), 1)
            _ = Atomic.fetch_add(s_w_1 + Int(t[13]), 1)
            _ = Atomic.fetch_add(s_w_2 + Int(t[14]), 1)
            _ = Atomic.fetch_add(s_w_3 + Int(t[15]), 1)
    else:
        var part_start = bid * G_HIST_PART_SIZE
        for i in range(tid + part_start, size, BLOCK_SIZE):
            var t = sort[i]
            var t_bytes = bitcast[DType.uint8, 4](t)
            _ = Atomic.fetch_add(s_w_0 + Int(t_bytes[0]), 1)
            _ = Atomic.fetch_add(s_w_1 + Int(t_bytes[1]), 1)
            _ = Atomic.fetch_add(s_w_2 + Int(t_bytes[2]), 1)
            _ = Atomic.fetch_add(s_w_3 + Int(t_bytes[3]), 1)

    barrier()

    # Reduce
    for i in range(tid, RADIX, BLOCK_SIZE):
        _ = Atomic.fetch_add(global_hist + i, s_hist_0[i] + s_hist_0[i + RADIX])
        _ = Atomic.fetch_add(
            global_hist + i + RADIX, s_hist_1[i] + s_hist_1[i + RADIX]
        )
        _ = Atomic.fetch_add(
            global_hist + i + (RADIX * 2), s_hist_2[i] + s_hist_2[i + RADIX]
        )
        _ = Atomic.fetch_add(
            global_hist + i + (RADIX * 3), s_hist_3[i] + s_hist_3[i + RADIX]
        )


def scan_global(
    global_hist: UnsafePointer[UInt32, MutAnyOrigin],
    first_pass: UnsafePointer[UInt32, MutAnyOrigin],
    sec_pass: UnsafePointer[UInt32, MutAnyOrigin],
    third_pass: UnsafePointer[UInt32, MutAnyOrigin],
    fourth_pass: UnsafePointer[UInt32, MutAnyOrigin],
):
    comptime RADIX = 256
    var tid = thread_idx.x
    var bid = block_idx.x
    var lid = lane_id()

    var s_scan = stack_allocation[
        RADIX, UInt32, address_space=AddressSpace.SHARED
    ]()

    var val = global_hist[tid + (bid * RADIX)]
    var sum = warp.prefix_sum[exclusive=False](val)
    var shifted = circular_shift(sum, UInt32(lid))
    s_scan[tid] = shifted
    barrier()

    var idx = tid << LANE_LOG
    var v: UInt32 = 0
    if tid < (RADIX >> LANE_LOG):
        v = s_scan[idx]

    var exc = warp.prefix_sum[exclusive=True](v)

    if tid < (RADIX >> LANE_LOG):
        s_scan[idx] = exc
    barrier()

    var out_val = s_scan[tid]
    if lid > 0:
        out_val += s_scan[tid - lid]

    out_val = (out_val << 2) | FLAG_INCLUSIVE

    if bid == 0:
        first_pass[tid] = out_val
    elif bid == 1:
        sec_pass[tid] = out_val
    elif bid == 2:
        third_pass[tid] = out_val
    elif bid == 3:
        fourth_pass[tid] = out_val


def digit_binning[
    keys_dtype: DType,
    vals_dtype: DType,
    *,
    BITS_PER_PASS: Int,
    BLOCK_SIZE: Int,
    KEYS_PER_THREAD: Int,
    HAVE_PAYLOAD: Bool,
](
    sort: UnsafePointer[Scalar[keys_dtype], MutAnyOrigin],
    sort_payload: UnsafePointer[Scalar[vals_dtype], MutAnyOrigin],
    alt: UnsafePointer[Scalar[keys_dtype], MutAnyOrigin],
    alt_payload: UnsafePointer[Scalar[vals_dtype], MutAnyOrigin],
    pass_hist: UnsafePointer[UInt32, MutAnyOrigin],
    index: UnsafePointer[UInt32, MutAnyOrigin],
    size: Int,
    radix_shift: UInt32,
):
    comptime RADIX = 2**BITS_PER_PASS
    comptime RADIX_MASK = Scalar[keys_dtype](RADIX - 1)
    comptime NUM_WARPS = BLOCK_SIZE // WARP_SIZE
    comptime WARP_PART_SIZE = WARP_SIZE * KEYS_PER_THREAD
    comptime PART_SIZE = NUM_WARPS * WARP_PART_SIZE
    comptime TOTAL_WARP_HISTS_SIZE = NUM_WARPS * RADIX
    comptime BIN_PART_SIZE = 7680

    var s_warp_histograms = stack_allocation[
        BIN_PART_SIZE, UInt32, address_space=AddressSpace.SHARED
    ]()
    var s_local_histogram = stack_allocation[
        RADIX, UInt32, address_space=AddressSpace.SHARED
    ]()

    var tid = thread_idx.x
    var lid = lane_id()
    var wid = tid // WARP_SIZE
    var gdim = grid_dim.x

    var s_warp_hist_ptr = s_warp_histograms + (wid << BITS_PER_PASS)

    for i in range(tid, TOTAL_WARP_HISTS_SIZE, BLOCK_SIZE):
        s_warp_histograms[i] = 0

    # atomically assign partition tiles
    if tid == 0:
        var idx_offset = Int(radix_shift >> 3)
        s_warp_histograms[BIN_PART_SIZE - 1] = Atomic.fetch_add(
            index + idx_offset, 1
        )
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
            keys[i] = sort[t]
        else:
            keys[i] = sort[t] if t < size else Scalar[keys_dtype].MAX
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

        _ = Atomic.fetch_add(
            pass_hist + (tid + (partition_index + 1) * RADIX),
            FLAG_REDUCTION | (reduction << 2),
        )

        var sum = warp.prefix_sum[exclusive=False](reduction)
        var shifted = circular_shift(sum, UInt32(lid))
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
        if wid > 0:
            offsets[i] += (
                s_warp_histograms[wid * RADIX + t2] + s_local_histogram[t2]
            )
        else:
            offsets[i] += s_local_histogram[t2]
    barrier()

    # Scatter keys into shared memory
    comptime for i in range(KEYS_PER_THREAD):
        s_warp_histograms[Int(offsets[i])] = UInt32(keys[i])

    # Decoupled Look-back (Spin Loop)
    if tid < RADIX:
        var reduction: UInt32 = 0
        var k = partition_index
        while True:
            var flag_payload = load_volatile(pass_hist + (tid + k * RADIX))
            var flag_mask = flag_payload & FLAG_MASK
            if flag_mask == FLAG_INCLUSIVE:
                reduction += flag_payload >> 2
                _ = Atomic.fetch_add(
                    pass_hist + (tid + (partition_index + 1) * RADIX),
                    FLAG_REDUCTION | (reduction << 2),
                )
                s_local_histogram[tid] = reduction - s_local_histogram[tid]
                break
            if flag_mask == FLAG_REDUCTION:
                reduction += flag_payload >> 2
                k -= 1
    barrier()

    # Scatter to Device Memory
    comptime if HAVE_PAYLOAD:
        var vals = InlineArray[Scalar[vals_dtype], KEYS_PER_THREAD](
            uninitialized=True
        )
        var digits = InlineArray[UInt8, KEYS_PER_THREAD](uninitialized=True)

        if partition_index < gdim - 1:
            comptime for i in range(KEYS_PER_THREAD):
                var t_idx = tid + (i * BLOCK_SIZE)
                var key = s_warp_histograms[t_idx]
                var d = (key >> radix_shift) & UInt32(RADIX_MASK)
                digits[i] = d.cast[DType.uint8]()
                alt[s_local_histogram[Int(d)] + UInt32(t_idx)] = Scalar[
                    keys_dtype
                ](key)
            barrier()

            var t_payload = t_base
            comptime for i in range(KEYS_PER_THREAD):
                vals[i] = sort_payload[t_payload]
                t_payload += WARP_SIZE

            comptime for i in range(KEYS_PER_THREAD):
                s_warp_histograms[Int(offsets[i])] = UInt32(vals[i])
            barrier()

            comptime for i in range(KEYS_PER_THREAD):
                var t_idx = tid + (i * BLOCK_SIZE)
                var d = Int(digits[i])
                alt_payload[s_local_histogram[d] + UInt32(t_idx)] = Scalar[
                    vals_dtype
                ](s_warp_histograms[t_idx])
        else:
            var final_part_size = size - BIN_PART_START
            comptime for i in range(KEYS_PER_THREAD):
                var t_idx = tid + (i * BLOCK_SIZE)
                if t_idx < final_part_size:
                    var key = s_warp_histograms[t_idx]
                    var d = (key >> radix_shift) & UInt32(RADIX_MASK)
                    digits[i] = d.cast[DType.uint8]()
                    alt[s_local_histogram[Int(d)] + UInt32(t_idx)] = Scalar[
                        keys_dtype
                    ](key)
            barrier()

            var t_payload = t_base
            comptime for i in range(KEYS_PER_THREAD):
                if t_payload < size:
                    vals[i] = sort_payload[t_payload]
                t_payload += WARP_SIZE

            comptime for i in range(KEYS_PER_THREAD):
                s_warp_histograms[Int(offsets[i])] = UInt32(
                    vals[i]
                )  # Corrected from keys[i]
            barrier()

            comptime for i in range(KEYS_PER_THREAD):
                var t_idx = tid + (i * BLOCK_SIZE)
                if t_idx < final_part_size:
                    var d = Int(digits[i])
                    alt_payload[s_local_histogram[d] + UInt32(t_idx)] = Scalar[
                        vals_dtype
                    ](s_warp_histograms[t_idx])
    else:
        var upper_bound = (
            BIN_PART_SIZE if partition_index
            < gdim - 1 else size - BIN_PART_START
        )
        for i in range(tid, upper_bound, BLOCK_SIZE):
            var key = s_warp_histograms[i]
            var digit = Int((key >> radix_shift) & UInt32(RADIX_MASK))
            var dst = s_local_histogram[digit] + UInt32(i)
            alt[dst] = Scalar[keys_dtype](key)


struct OneSweepWorkspace[
    keys_dtype: DType,
    vals_dtype: DType,
    *,
    BLOCK_SIZE: Int,
    KEYS_PER_THREAD: Int,
]:
    var alt_keys: DeviceBuffer[Self.keys_dtype]
    var alt_vals: DeviceBuffer[Self.vals_dtype]
    var global_hist: DeviceBuffer[DType.uint32]
    var pass_hist_p1: DeviceBuffer[DType.uint32]
    var pass_hist_p2: DeviceBuffer[DType.uint32]
    var pass_hist_p3: DeviceBuffer[DType.uint32]
    var pass_hist_p4: DeviceBuffer[DType.uint32]
    var index: DeviceBuffer[DType.uint32]

    def __init__(out self, ctx: DeviceContext, size: Int) raises:
        comptime RADIX = 256
        comptime BIN_PART_SIZE = Self.BLOCK_SIZE * Self.KEYS_PER_THREAD

        var binning_blocks = ceildiv(size, BIN_PART_SIZE)

        var hist_blocks = binning_blocks + 1

        self.alt_keys = ctx.enqueue_create_buffer[Self.keys_dtype](size)
        self.alt_vals = ctx.enqueue_create_buffer[Self.vals_dtype](size)

        self.global_hist = ctx.enqueue_create_buffer[DType.uint32](RADIX * 4)
        var N = hist_blocks * RADIX
        self.pass_hist_p1 = ctx.enqueue_create_buffer[DType.uint32](N)
        self.pass_hist_p2 = ctx.enqueue_create_buffer[DType.uint32](N)
        self.pass_hist_p3 = ctx.enqueue_create_buffer[DType.uint32](N)
        self.pass_hist_p4 = ctx.enqueue_create_buffer[DType.uint32](N)

        self.index = ctx.enqueue_create_buffer[DType.uint32](4)


def device_radix_sort_onesweep_keys[
    keys_dtype: DType, KEYS_PER_THREAD: Int
](
    ctx: DeviceContext,
    mut workspace: OneSweepWorkspace[keys_dtype, keys_dtype],
    mut keys: DeviceBuffer[keys_dtype],
    size: Int,
) raises:
    comptime RADIX = 256
    comptime BIN_PART_SIZE = 7680
    comptime G_HIST_PART_SIZE = 65536
    comptime G_HIST_TPB = 128
    comptime VEC_WIDTH = 4

    var binning_blocks = ceildiv(size, BIN_PART_SIZE)
    var g_hist_blocks = ceildiv(size, G_HIST_PART_SIZE)

    workspace.index.enqueue_fill(0)
    workspace.global_hist.enqueue_fill(0)
    workspace.pass_hist_p1.enqueue_fill(0)
    workspace.pass_hist_p2.enqueue_fill(0)
    workspace.pass_hist_p3.enqueue_fill(0)
    workspace.pass_hist_p4.enqueue_fill(0)
    ctx.synchronize()

    var pass_hist: InlineArray[UnsafePointer[UInt32, MutAnyOrigin], 4] = [
        workspace.pass_hist_p1.unsafe_ptr(),
        workspace.pass_hist_p2.unsafe_ptr(),
        workspace.pass_hist_p3.unsafe_ptr(),
        workspace.pass_hist_p4.unsafe_ptr(),
    ]

    # 1. Global Histogram
    comptime _ghist = global_histogram[
        keys_dtype, BLOCK_SIZE=G_HIST_TPB, RADIX=RADIX, VEC_WIDTH=VEC_WIDTH
    ]
    ctx.enqueue_function[_ghist, _ghist](
        keys.unsafe_ptr(),
        workspace.global_hist.unsafe_ptr(),
        size,
        grid_dim=g_hist_blocks,
        block_dim=G_HIST_TPB,
    )

    # 2. Block Scan
    comptime _scan = scan_global
    ctx.enqueue_function[_scan, _scan](
        workspace.global_hist.unsafe_ptr(),
        pass_hist[0],
        pass_hist[1],
        pass_hist[2],
        pass_hist[3],
        grid_dim=4,
        block_dim=RADIX,
    )

    # 3. Digit Binning Passes (Decoupled Look-back)
    comptime _bin = digit_binning[
        keys_dtype,
        DType.invalid,
        BITS_PER_PASS=8,
        BLOCK_SIZE=512,
        KEYS_PER_THREAD=KEYS_PER_THREAD,
        HAVE_PAYLOAD=False,
    ]

    var dummy_v_ptr = UnsafePointer[Scalar[keys_dtype], MutAnyOrigin]()
    var db_keys = DoubleBuffer[keys_dtype](
        keys.unsafe_ptr(), workspace.alt_keys.unsafe_ptr()
    )

    for pass_idx in range(4):
        ctx.enqueue_function[_bin, _bin](
            db_keys.current,
            dummy_v_ptr,
            db_keys.alternate,
            dummy_v_ptr,
            pass_hist[pass_idx],
            workspace.index.unsafe_ptr(),
            size,
            UInt32(pass_idx * 8),
            grid_dim=binning_blocks,
            block_dim=512,
        )
        db_keys.swap()

    ctx.synchronize()


def device_radix_sort_onesweep_pairs[
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
    comptime G_HIST_ITEMS_PER_THREAD = 32
    comptime G_HIST_PART_SIZE = G_HIST_TPB * G_HIST_ITEMS_PER_THREAD
    comptime VEC_WIDTH = 4

    var binning_blocks = ceildiv(size, BIN_PART_SIZE)
    var g_hist_blocks = ceildiv(size, G_HIST_PART_SIZE)

    workspace.index.enqueue_fill(0)
    workspace.global_hist.enqueue_fill(0)
    workspace.pass_hist_p1.enqueue_fill(0)
    workspace.pass_hist_p2.enqueue_fill(0)
    workspace.pass_hist_p3.enqueue_fill(0)
    workspace.pass_hist_p4.enqueue_fill(0)
    ctx.synchronize()

    var pass_hist: InlineArray[UnsafePointer[UInt32, MutAnyOrigin], 4] = [
        workspace.pass_hist_p1.unsafe_ptr(),
        workspace.pass_hist_p2.unsafe_ptr(),
        workspace.pass_hist_p3.unsafe_ptr(),
        workspace.pass_hist_p4.unsafe_ptr(),
    ]

    # 1. Global Histogram
    comptime _ghist = global_histogram[
        keys_dtype,
        BLOCK_SIZE=G_HIST_TPB,
        RADIX=RADIX,
        VEC_WIDTH=VEC_WIDTH,
        ITEMS_PER_THREAD=G_HIST_ITEMS_PER_THREAD,
    ]
    ctx.enqueue_function[_ghist, _ghist](
        keys.unsafe_ptr(),
        workspace.global_hist.unsafe_ptr(),
        size,
        grid_dim=g_hist_blocks,
        block_dim=G_HIST_TPB,
    )

    # 2. Block Scan
    comptime _scan = scan_global
    ctx.enqueue_function[_scan, _scan](
        workspace.global_hist.unsafe_ptr(),
        pass_hist[0],
        pass_hist[1],
        pass_hist[2],
        pass_hist[3],
        grid_dim=4,
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

    var db_keys = DoubleBuffer[keys_dtype](
        keys.unsafe_ptr(), workspace.alt_keys.unsafe_ptr()
    )
    var db_vals = DoubleBuffer[vals_dtype](
        values.unsafe_ptr(), workspace.alt_vals.unsafe_ptr()
    )

    comptime for pass_idx in range(4):
        comptime radix_shift = UInt32(pass_idx * 8)
        ctx.enqueue_function[_bin, _bin](
            db_keys.current,
            db_vals.current,
            db_keys.alternate,
            db_vals.alternate,
            pass_hist[pass_idx],
            workspace.index.unsafe_ptr(),
            size,
            radix_shift,
            grid_dim=binning_blocks,
            block_dim=BINNING_TPB,
        )
        db_keys.swap()
        db_vals.swap()

    ctx.synchronize()
