from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.sys.info import bit_width_of

from bajo.core.utils import is_power_of_2


# bitonic sort
def bitonic_sort_step[
    dtype: DType
](
    keys: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    values: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    j: Int,
    k: Int,
    size: Int,
):
    """
    Executes a single step of the Bitonic sort network.
    """
    var tid = Int(thread_idx.x + block_idx.x * block_dim.x)

    if tid < size:
        var ixj = tid ^ j

        if tid < ixj:
            var sort_dir = (tid & k) == 0
            var key_a = keys[tid]
            var key_b = keys[ixj]

            # If sort_dir is True we sort ascending, else descending
            if (sort_dir and key_a > key_b) or (not sort_dir and key_a < key_b):
                # Swap Keys
                keys[tid] = key_b
                keys[ixj] = key_a

                # Swap Values
                var val_a = values[tid]
                var val_b = values[ixj]
                values[tid] = val_b
                values[ixj] = val_a


def bitonic_sort[
    dtype: DType, //, THREADS_PER_BLOCK: Int = 256
](
    ctx: DeviceContext,
    keys: DeviceBuffer[dtype],
    values: DeviceBuffer[dtype],
    size: Int,
) raises:
    """
    Sorts keys and values in-place on the GPU using Bitonic Sort.
    """

    debug_assert["safe"](is_power_of_2(size))

    var blocks = (size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    var k = 2
    while k <= size:
        var j = k >> 1
        while j > 0:
            comptime kernel = bitonic_sort_step[dtype]
            ctx.enqueue_function[kernel, kernel](
                keys,
                values,
                j,
                k,
                size,
                grid_dim=blocks,
                block_dim=THREADS_PER_BLOCK,
            )

            ctx.synchronize()

            j >>= 1
        k <<= 1


# radix sort
fn radix_predicate[
    dtype: DType,
    //,
    NUM_BUCKETS: Int,
    MASK: Scalar[dtype],
](
    keys: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    digit_flags: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    scan_init: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    bit_shift: Int,
    size: Int,
):
    """
    Extracts the digit using the comptime-derived MASK and number of buckets.
    """
    var tid = Int(thread_idx.x + block_idx.x * block_dim.x)
    if tid < size:
        var key = keys[tid]
        var shift_val = Scalar[dtype](bit_shift)
        var digit = Int((key >> shift_val) & MASK)

        for b in range(NUM_BUCKETS):
            var is_match: Scalar[dtype] = 0
            if b == digit:
                is_match = 1

            var idx = b * size + tid
            digit_flags[idx] = is_match
            scan_init[idx] = is_match


fn scan_step[
    dtype: DType
](
    data_in: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    data_out: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    offset: Int,
    total_size: Int,
):
    """
    Global Hillis-Steele Inclusive Scan.
    """
    var tid = Int(thread_idx.x + block_idx.x * block_dim.x)
    if tid < total_size:
        if tid >= offset:
            data_out[tid] = data_in[tid] + data_in[tid - offset]
        else:
            data_out[tid] = data_in[tid]


fn radix_scatter[
    dtype: DType, //, MASK: Scalar[dtype]
](
    keys_in: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    values_in: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    keys_out: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    values_out: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    inclusive_scan: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    bit_shift: Int,
    size: Int,
):
    """
    Scatters keys and values to their exact new index using the inclusive scan.
    """
    var tid = Int(thread_idx.x + block_idx.x * block_dim.x)
    if tid < size:
        var shift_val = Scalar[dtype](bit_shift)
        var digit = Int((keys_in[tid] >> shift_val) & MASK)
        var dest_idx = inclusive_scan[digit * size + tid] - 1

        keys_out[Int(dest_idx)] = keys_in[tid]
        values_out[Int(dest_idx)] = values_in[tid]


fn copy[
    dtype: DType
](
    src_keys: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    src_values: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst_keys: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst_values: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    size: Int,
):
    """Restores data to original buffers if the algorithm takes an odd number of passes.
    """
    var tid = Int(thread_idx.x + block_idx.x * block_dim.x)
    if tid < size:
        dst_keys[tid] = src_keys[tid]
        dst_values[tid] = src_values[tid]


def radix_sort[
    dtype: DType, //, BITS_PER_PASS: Int = 4, THREADS_PER_BLOCK: Int = 256
](
    ctx: DeviceContext,
    keys: DeviceBuffer[dtype],
    values: DeviceBuffer[dtype],
    size: Int,
) raises:
    """
    GPU Radix Sort based on the number of bits requested.
    """
    comptime DTYPE_BIT_WIDTH = bit_width_of[dtype]()

    comptime assert is_power_of_2(
        BITS_PER_PASS
    ), "BITS_PER_PASS must be a power of 2"
    comptime assert (
        BITS_PER_PASS <= 16
    ), "BITS_PER_PASS > 16 will likely trigger a VRAM Out-Of-Memory crash"
    comptime assert BITS_PER_PASS <= DTYPE_BIT_WIDTH

    comptime NUM_BUCKETS = 1 << BITS_PER_PASS
    comptime MASK = Scalar[dtype](NUM_BUCKETS - 1)
    comptime NUM_PASSES = DTYPE_BIT_WIDTH // BITS_PER_PASS

    # Grid dimensions
    var blocks = (size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    var total_scan_size = size * NUM_BUCKETS
    var scan_blocks = (
        total_scan_size + THREADS_PER_BLOCK - 1
    ) // THREADS_PER_BLOCK

    # Allocations
    var keys_alt = ctx.enqueue_create_buffer[dtype](size)
    var values_alt = ctx.enqueue_create_buffer[dtype](size)

    var digit_flags = ctx.enqueue_create_buffer[dtype](total_scan_size)
    var scan_A = ctx.enqueue_create_buffer[dtype](total_scan_size)
    var scan_B = ctx.enqueue_create_buffer[dtype](total_scan_size)

    var current_is_original = True

    for pass_idx in range(NUM_PASSES):
        var bit_shift = pass_idx * BITS_PER_PASS

        var c_keys = keys if current_is_original else keys_alt
        var n_keys = keys_alt if current_is_original else keys
        var c_values = values if current_is_original else values_alt
        var n_values = values_alt if current_is_original else values

        # Step 1: Predicate (Fill Buckets)
        comptime _radix_predicate = radix_predicate[NUM_BUCKETS, MASK]
        ctx.enqueue_function[_radix_predicate, _radix_predicate](
            c_keys,
            digit_flags,
            scan_A,
            bit_shift,
            size,
            grid_dim=blocks,
            block_dim=THREADS_PER_BLOCK,
        )
        ctx.synchronize()

        # Step 2: Global Inclusive Scan
        var current_scan_is_A = True
        var offset = 1
        while offset < total_scan_size:
            comptime _scan_step = scan_step[dtype]
            if current_scan_is_A:
                ctx.enqueue_function[_scan_step, _scan_step](
                    scan_A,
                    scan_B,
                    offset,
                    total_scan_size,
                    grid_dim=scan_blocks,
                    block_dim=THREADS_PER_BLOCK,
                )
            else:
                ctx.enqueue_function[_scan_step, _scan_step](
                    scan_B,
                    scan_A,
                    offset,
                    total_scan_size,
                    grid_dim=scan_blocks,
                    block_dim=THREADS_PER_BLOCK,
                )
            ctx.synchronize()
            current_scan_is_A = not current_scan_is_A
            offset *= 2

        var final_scan = scan_B if not current_scan_is_A else scan_A

        # Step 3: Scatter
        comptime _radix_scatter = radix_scatter[MASK]
        ctx.enqueue_function[_radix_scatter, _radix_scatter](
            c_keys,
            c_values,
            n_keys,
            n_values,
            final_scan,
            bit_shift,
            size,
            grid_dim=blocks,
            block_dim=THREADS_PER_BLOCK,
        )
        ctx.synchronize()

        current_is_original = not current_is_original

    # Step 4: Cleanup odd passes
    if not current_is_original:
        comptime copy_kernel = copy[dtype]
        ctx.enqueue_function[copy_kernel, copy_kernel](
            keys_alt,
            values_alt,
            keys,
            values,
            size,
            grid_dim=blocks,
            block_dim=THREADS_PER_BLOCK,
        )
        ctx.synchronize()
