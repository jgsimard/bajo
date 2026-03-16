from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.sys.info import bit_width_of

from bajo.core.utils import is_power_of_2


fn bitonic_sort_shared[
    dtype: DType, THREADS_PER_BLOCK: Int, IS_MERGE_BLOCK: Bool
](
    keys: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    values: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    k: Int,
    size: Int,
):
    var tid = Int(thread_idx.x + block_idx.x * block_dim.x)
    var local_tid = Int(thread_idx.x)

    # put block data into shared memory
    var shared_keys = stack_allocation[
        THREADS_PER_BLOCK, Scalar[dtype], address_space=AddressSpace.SHARED
    ]()
    var shared_values = stack_allocation[
        THREADS_PER_BLOCK, Scalar[dtype], address_space=AddressSpace.SHARED
    ]()

    if tid < size:
        shared_keys[local_tid] = keys[tid]
        shared_values[local_tid] = values[tid]

    barrier()

    comptime if IS_MERGE_BLOCK:
        var j = THREADS_PER_BLOCK >> 1
        while j > 0:
            var ixj = local_tid ^ j

            if local_tid < ixj:
                var global_ixj = tid ^ j
                if global_ixj < size:
                    var sort_dir = (tid & k) == 0
                    var key_a = shared_keys[local_tid]
                    var key_b = shared_keys[ixj]

                    var swap = (key_a > key_b) if sort_dir else (key_a < key_b)
                    if swap:
                        shared_keys[local_tid] = key_b
                        shared_keys[ixj] = key_a

                        var val_a = shared_values[local_tid]
                        var val_b = shared_values[ixj]
                        shared_values[local_tid] = val_b
                        shared_values[ixj] = val_a

            barrier()
            j >>= 1
    else:
        # block bitonic sort
        var k = 2
        while k <= THREADS_PER_BLOCK:
            var j = k >> 1
            while j > 0:
                var ixj = local_tid ^ j

                # Only 1 thread processes both elements of the swap pair
                if local_tid < ixj:
                    var global_ixj = tid ^ j
                    if global_ixj < size:
                        var sort_dir = (tid & k) == 0
                        var key_a = shared_keys[local_tid]
                        var key_b = shared_keys[ixj]

                        var swap = (key_a > key_b) if sort_dir else (
                            key_a < key_b
                        )

                        if swap:
                            shared_keys[local_tid] = key_b
                            shared_keys[ixj] = key_a

                            var val_a = shared_values[local_tid]
                            var val_b = shared_values[ixj]
                            shared_values[local_tid] = val_b
                            shared_values[ixj] = val_a

                barrier()  # Sync threads before the next internal 'j' step
                j >>= 1
            k <<= 1

    # write back to global memory
    if tid < size:
        keys[tid] = shared_keys[local_tid]
        values[tid] = shared_values[local_tid]


fn bitonic_sort[
    dtype: DType, //, THREADS_PER_BLOCK: Int = 256
](
    ctx: DeviceContext,
    keys: DeviceBuffer[dtype],
    values: DeviceBuffer[dtype],
    size: Int,
) raises:
    """
    Bitonic Sort.
    """
    debug_assert["safe"](is_power_of_2(size))
    var blocks = (size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    # PHASE 1: sort independent chunks of size THREADS_PER_BLOCK in Shared Memory
    comptime shared_block_kernel = bitonic_sort_shared[
        dtype, THREADS_PER_BLOCK, False
    ]
    ctx.enqueue_function[shared_block_kernel, shared_block_kernel](
        keys, values, 0, size, grid_dim=blocks, block_dim=THREADS_PER_BLOCK
    )

    # PHASE 2: Merge across blocks.
    var k = THREADS_PER_BLOCK << 1
    while k <= size:
        var j = k >> 1
        while j > 0:
            if j >= THREADS_PER_BLOCK:
                # Swaps cross block boundaries, must use Global Memory Step
                comptime global_step_kernel = bitonic_sort_step[dtype]
                ctx.enqueue_function[global_step_kernel, global_step_kernel](
                    keys,
                    values,
                    j,
                    k,
                    size,
                    grid_dim=blocks,
                    block_dim=THREADS_PER_BLOCK,
                )
                j >>= 1
            else:
                # Remaining swaps for this 'k' fall entirely inside independent blocks.
                # Collapse all remaining j loops into a single Shared Memory kernel launch!
                comptime shared_merge_kernel = bitonic_sort_shared[
                    dtype, THREADS_PER_BLOCK, True
                ]
                ctx.enqueue_function[shared_merge_kernel, shared_merge_kernel](
                    keys,
                    values,
                    k,
                    size,
                    grid_dim=blocks,
                    block_dim=THREADS_PER_BLOCK,
                )
                # After this, the j-loop for the current k is completely finished
                break

        k <<= 1

    ctx.synchronize()


# basic version of bitonic sort
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


def bitonic_sort_basic[
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
