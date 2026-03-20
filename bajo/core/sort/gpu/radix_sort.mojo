from std.bit import pop_count, count_trailing_zeros
from std.gpu import (
    global_idx,
    thread_idx,
    block_idx,
    block_dim,
    grid_dim,
    barrier,
)
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from std.gpu.primitives.block import prefix_sum as block_prefix_sum
from std.gpu.primitives.warp import prefix_sum, vote, shuffle_idx, lane_id
from std.math import ceildiv
from std.memory import UnsafePointer
from std.os.atomic import Atomic
from std.sys.info import size_of

from layout import Layout, LayoutTensor


@fieldwise_init
struct DoubleBuffer[dtype: DType](Copyable):
    var d_buffers_0: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]
    var d_buffers_1: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]
    var selector: Int

    def __init__(
        out self,
        current: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        alternate: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
    ):
        self.d_buffers_0 = current
        self.d_buffers_1 = alternate
        self.selector = 0

    def current(self) -> UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]:
        if self.selector == 0:
            return self.d_buffers_0
        return self.d_buffers_1

    def alternate(self) -> UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]:
        if self.selector == 0:
            return self.d_buffers_1
        return self.d_buffers_0

    def swap(mut self):
        self.selector ^= 1


# -------------------------------------------------------------------
# PHASE 1: Histogram
# -------------------------------------------------------------------
def radix_sort_histogram_kernel[
    dtype: DType, RADIX_BINS: Int
](
    d_keys: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    d_block_histograms: UnsafePointer[UInt32, MutAnyOrigin],
    num_items: Int,
    shift: Scalar[dtype],
):
    var tid = thread_idx.x
    var bid = block_idx.x
    var g_id = global_idx.x

    var s_hist = LayoutTensor[
        DType.uint32,
        Layout.row_major(RADIX_BINS),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    if tid < UInt(RADIX_BINS):
        s_hist[Int(tid)] = 0
    barrier()

    if g_id < UInt(num_items):
        var key = d_keys[g_id]
        var bin_idx = Int((key >> shift) & Scalar[dtype](0xFF))
        _ = Atomic.fetch_add(s_hist.ptr + (bin_idx), 1)
    barrier()

    if tid < UInt(RADIX_BINS):
        var global_offset = tid * grid_dim.x + bid
        d_block_histograms[global_offset] = s_hist[Int(tid)][0]


# -------------------------------------------------------------------
# PHASE 2 : Parallel Block Scan
# -------------------------------------------------------------------
def scan_local_kernel[
    BLOCK_SIZE: Int
](
    d_inout: UnsafePointer[UInt32, MutAnyOrigin],
    d_partials: UnsafePointer[UInt32, MutAnyOrigin],
    n: Int,
):
    var tid = thread_idx.x
    var bid = block_idx.x
    var gid = bid * UInt(BLOCK_SIZE) + tid

    var val: UInt32 = 0
    if gid < UInt(n):
        val = d_inout[gid]

    # Block-level scan
    var excl_val = block_prefix_sum[block_size=BLOCK_SIZE, exclusive=True](val)

    if gid < UInt(n):
        d_inout[gid] = excl_val

    # Save total sum (last excl + last val) to partials
    if tid == UInt(BLOCK_SIZE - 1):
        d_partials[Int(bid)] = excl_val + val


def scan_partials_kernel[
    BLOCK_SIZE: Int
](d_partials: UnsafePointer[UInt32, MutAnyOrigin], num_partials: Int):
    if thread_idx.x == 0:
        var sum: UInt32 = 0
        for i in range(num_partials):
            var val = d_partials[i]
            d_partials[i] = sum
            sum += val


def scan_add_kernel[
    BLOCK_SIZE: Int
](
    d_inout: UnsafePointer[UInt32, MutAnyOrigin],
    d_partials: UnsafePointer[UInt32, MutAnyOrigin],
    n: Int,
):
    var gid = global_idx.x
    var bid = block_idx.x
    if gid < UInt(n):
        # Add the offset belonging to this block
        d_inout[Int(gid)] += d_partials[Int(bid)]


# -------------------------------------------------------------------
# PHASE 3: Scatter
# -------------------------------------------------------------------
def radix_sort_scatter_keys_kernel[
    RADIX_BINS: Int
](
    keys_in: UnsafePointer[UInt32, MutAnyOrigin],
    keys_out: UnsafePointer[UInt32, MutAnyOrigin],
    d_block_histograms: UnsafePointer[UInt32, MutAnyOrigin],
    num_items: Int,
    shift: UInt32,
):
    var tid = thread_idx.x
    var bid = block_idx.x
    var g_id = global_idx.x

    var s_offsets = LayoutTensor[
        DType.uint32,
        Layout.row_major(RADIX_BINS),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    if tid < UInt(RADIX_BINS):
        var global_offset = tid * grid_dim.x + bid
        s_offsets[Int(tid)] = d_block_histograms[global_offset]
    barrier()

    if g_id < UInt(num_items):
        var key = keys_in[g_id]
        var bin_idx = (key >> shift) & 0xFF
        var output_pos = Atomic.fetch_add(s_offsets.ptr + Int(bin_idx), 1)
        keys_out[Int(output_pos)] = key


def radix_sort_scatter_pairs_kernel[
    dtype: DType, BLOCK_THREADS: Int, RADIX_BINS: Int
](
    keys_in: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    keys_out: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    values_in: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    values_out: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    d_block_histograms: UnsafePointer[UInt32, MutAnyOrigin],
    num_items: Int,
    shift: Scalar[dtype],
):
    var tid = thread_idx.x
    var bid = block_idx.x
    var g_id = bid * UInt(BLOCK_THREADS) + tid

    # 1. Shared Memory Allocations
    comptime LT_bin = LayoutTensor[
        DType.uint32,
        Layout.row_major(RADIX_BINS),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ]
    var s_offsets = LT_bin.stack_allocation()
    var s_local_hist = LT_bin.stack_allocation()

    comptime LT_block[T: DType] = LayoutTensor[
        T,
        Layout.row_major(BLOCK_THREADS),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ]
    var s_bins = LT_block[DType.uint32].stack_allocation()
    var s_keys_exch = LT_block[dtype].stack_allocation()
    var s_vals_exch = LT_block[dtype].stack_allocation()
    var s_pos_exch = LT_block[DType.uint32].stack_allocation()

    # 2. Init global offsets & zero local histogram
    if tid < UInt(RADIX_BINS):
        var global_offset = tid * grid_dim.x + bid
        s_offsets[Int(tid)] = d_block_histograms[global_offset]
        s_local_hist[Int(tid)] = 0
    barrier()

    # 3. Load elements, determine bins, and build a local histogram
    var key: Scalar[dtype] = 0
    var val: Scalar[dtype] = 0
    var bin_idx: UInt32 = 0
    var is_valid = g_id < UInt(num_items)

    if is_valid:
        key = keys_in[g_id]
        val = values_in[g_id]
        bin_idx = UInt32(Int((key >> shift) & Scalar[dtype](0xFF)))
        s_bins[Int(tid)] = bin_idx
        # Atomically count how many items in this block go to each bin
        _ = Atomic.fetch_add(s_local_hist.ptr + Int(bin_idx), 1)
    barrier()

    # 4. Exclusive prefix sum of the local histogram to get shared memory offsets
    if tid == 0:
        var sum: UInt32 = 0
        for i in range(RADIX_BINS):
            var count = s_local_hist[i]
            s_local_hist[i] = sum
            sum += count[0]
    barrier()

    # 5. SIMD-Optimized Local Rank & Exchange Routing
    if is_valid:
        var local_rank: UInt32 = 0
        var i: Int = 0

        # Vectorize the loop using SIMD-4 (Reduces iterations by 4x!)
        while i <= Int(tid) - 4:
            var chunk = s_bins.ptr.load[width=4](i)
            var mask = chunk.eq(bin_idx)
            local_rank += mask.cast[DType.uint32]().reduce_add()
            i += 4

        # Remainder logic
        while i < Int(tid):
            if s_bins[i] == bin_idx:
                local_rank += 1
            i += 1

        # Calculate exact destinations
        var output_pos = s_offsets[Int(bin_idx)] + local_rank
        var shared_idx = s_local_hist[Int(bin_idx)] + local_rank

        # Write to Shared Memory FIRST (This compacts the data by bin)
        s_keys_exch[Int(shared_idx)] = key
        s_vals_exch[Int(shared_idx)] = val
        s_pos_exch[Int(shared_idx)] = output_pos
    barrier()

    # 6. COALESCED Global Memory Scatter
    # Calculate exactly how many valid elements exist in this specific block
    var valid_in_block = Int(num_items) - Int(bid) * BLOCK_THREADS
    if valid_in_block > BLOCK_THREADS:
        valid_in_block = BLOCK_THREADS

    # Because elements are compacted in shared memory, contiguous threads
    # will now write to contiguous global memory addresses!
    if Int(tid) < valid_in_block:
        var out_pos = s_pos_exch[Int(tid)]
        keys_out[Int(out_pos)] = s_keys_exch[Int(tid)][0]
        values_out[Int(out_pos)] = s_vals_exch[Int(tid)][0]


# -------------------------------------------------------------------
# HOST APIS: radx_sort overloads
# -------------------------------------------------------------------


def radix_sort[
    RADIX_BITS: Int = 8
](ctx: DeviceContext, keys: DeviceBuffer[DType.uint32], size: Int) raises:
    comptime RADIX_BINS = 1 << RADIX_BITS
    comptime PASSES = size_of[DType.uint32]() * 8 / RADIX_BITS
    comptime BLOCK_THREADS = 256
    comptime SCAN_BLOCK = 512  # Number of threads for the scan kernels

    var grid_blocks = ceildiv(size, BLOCK_THREADS)
    var total_hist_bins = RADIX_BINS * grid_blocks
    var scan_grid_blocks = ceildiv(total_hist_bins, SCAN_BLOCK)

    var buf_keys_alt = ctx.enqueue_create_buffer[DType.uint32](size)
    var buf_hist = ctx.enqueue_create_buffer[DType.uint32](total_hist_bins)

    # NEW: Allocate the tiny partials array
    var buf_partials = ctx.enqueue_create_buffer[DType.uint32](scan_grid_blocks)

    var ping_pong = DoubleBuffer(keys.unsafe_ptr(), buf_keys_alt.unsafe_ptr())
    var d_hist = buf_hist.unsafe_ptr()
    var d_partials = buf_partials.unsafe_ptr()

    # kernels
    comptime histogram = radix_sort_histogram_kernel[DType.uint32, RADIX_BINS]
    comptime scan_local = scan_local_kernel[SCAN_BLOCK]
    comptime scan_partials = scan_partials_kernel[SCAN_BLOCK]
    comptime scan_add = scan_add_kernel[SCAN_BLOCK]
    comptime scatter = radix_sort_scatter_keys_kernel[RADIX_BINS]

    comptime for p in range(PASSES):
        var shift = UInt32(p) * UInt32(RADIX_BITS)

        # 1. Histogram
        ctx.enqueue_function[histogram, histogram](
            ping_pong.current(),
            d_hist,
            size,
            shift,
            grid_dim=grid_blocks,
            block_dim=BLOCK_THREADS,
        )

        # 2. Parallel Scan
        ctx.enqueue_function[scan_local, scan_local](
            d_hist,
            d_partials,
            total_hist_bins,
            grid_dim=scan_grid_blocks,
            block_dim=SCAN_BLOCK,
        )

        ctx.enqueue_function[scan_partials, scan_partials](
            d_partials,
            scan_grid_blocks,
            grid_dim=scan_grid_blocks,
            block_dim=SCAN_BLOCK,
        )

        ctx.enqueue_function[scan_add, scan_add](
            d_hist,
            d_partials,
            total_hist_bins,
            grid_dim=scan_grid_blocks,
            block_dim=SCAN_BLOCK,
        )

        # 3. Scatter
        ctx.enqueue_function[scatter, scatter](
            ping_pong.current(),
            ping_pong.alternate(),
            d_hist,
            size,
            shift,
            grid_dim=grid_blocks,
            block_dim=BLOCK_THREADS,
        )

        ping_pong.swap()


def radix_sort[
    dtype: DType, RADIX_BITS: Int = 8
](
    ctx: DeviceContext,
    keys: DeviceBuffer[dtype],
    values: DeviceBuffer[dtype],
    size: Int,
) raises:
    comptime BLOCK_THREADS = 512
    comptime SCAN_BLOCK = 512
    comptime RADIX_BINS = 2**RADIX_BITS
    comptime PASSES = size_of[dtype]() * 8 / RADIX_BITS

    var grid_blocks = ceildiv(size, BLOCK_THREADS)
    var total_hist_bins = RADIX_BINS * grid_blocks
    var scan_grid_blocks = ceildiv(total_hist_bins, SCAN_BLOCK)

    # Allocate Alternates & Histogram trackers
    var buf_keys_alt = ctx.enqueue_create_buffer[dtype](size)
    var buf_values_alt = ctx.enqueue_create_buffer[dtype](size)
    var buf_hist = ctx.enqueue_create_buffer[DType.uint32](total_hist_bins)
    var buf_partials = ctx.enqueue_create_buffer[DType.uint32](scan_grid_blocks)

    var ping_pong_keys = DoubleBuffer[dtype](
        keys.unsafe_ptr(), buf_keys_alt.unsafe_ptr()
    )
    var ping_pong_values = DoubleBuffer[dtype](
        values.unsafe_ptr(), buf_values_alt.unsafe_ptr()
    )

    var d_hist = buf_hist.unsafe_ptr()
    var d_partials = buf_partials.unsafe_ptr()

    # kernels
    comptime histogram = radix_sort_histogram_kernel[dtype, RADIX_BINS]
    comptime scan_local = scan_local_kernel[SCAN_BLOCK]
    comptime scan_partials = scan_partials_kernel[SCAN_BLOCK]
    comptime scan_add = scan_add_kernel[SCAN_BLOCK]
    comptime scatter = radix_sort_scatter_pairs_kernel[
        dtype, BLOCK_THREADS, RADIX_BINS
    ]

    comptime for p in range(PASSES):
        var shift = Scalar[dtype](p * RADIX_BITS)

        # 1. Histogram
        ctx.enqueue_function[histogram, histogram](
            ping_pong_keys.current(),
            d_hist,
            size,
            shift,
            grid_dim=grid_blocks,
            block_dim=BLOCK_THREADS,
        )

        # 2. Parallel Scan
        ctx.enqueue_function[scan_local, scan_local](
            d_hist,
            d_partials,
            total_hist_bins,
            grid_dim=scan_grid_blocks,
            block_dim=SCAN_BLOCK,
        )

        ctx.enqueue_function[scan_partials, scan_partials](
            d_partials, scan_grid_blocks, grid_dim=1, block_dim=1
        )

        ctx.enqueue_function[scan_add, scan_add](
            d_hist,
            d_partials,
            total_hist_bins,
            grid_dim=scan_grid_blocks,
            block_dim=SCAN_BLOCK,
        )

        # 3. Scatter
        ctx.enqueue_function[scatter, scatter](
            ping_pong_keys.current(),
            ping_pong_keys.alternate(),
            ping_pong_values.current(),
            ping_pong_values.alternate(),
            d_hist,
            size,
            shift,
            grid_dim=grid_blocks,
            block_dim=BLOCK_THREADS,
        )

        ping_pong_keys.swap()
        ping_pong_values.swap()

    # 4. Handle odd-byte DTypes (like DType.uint8)
    comptime if PASSES % 2 != 0:
        # If the number of passes is odd, the final sorted arrays are sitting in
        # `buf_keys_alt` and `buf_values_alt`. You must execute a device-to-device
        # copy here to put them back into the user's original `keys` and `values` buffers.
        pass

    ctx.synchronize()
