from std.gpu import (
    thread_idx_int as thread_idx,
    barrier,
    global_idx_int as global_idx,
)
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.gpu.primitives import warp
from std.sys.info import bit_width_of

from bajo.core.utils import is_power_of_2


@always_inline
def bitonic_sort_shared[
    dtype: DType, THREADS_PER_BLOCK: Int, IS_MERGE_BLOCK: Bool
](
    keys: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    values: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    k_merge: Int,
    size: Int,
):
    var gid = global_idx.x
    var tid = thread_idx.x

    # put block data into shared memory
    var shared_keys = stack_allocation[
        THREADS_PER_BLOCK, Scalar[dtype], address_space=AddressSpace.SHARED
    ]()
    var shared_values = stack_allocation[
        THREADS_PER_BLOCK, Scalar[dtype], address_space=AddressSpace.SHARED
    ]()

    if gid < size:
        shared_keys[tid] = keys[gid]
        shared_values[tid] = values[gid]
    else:
        shared_keys[tid] = 0xFFFFFFFF
        shared_values[tid] = 0

    barrier()

    @always_inline
    def _step(j_in: Int, k: Int) capturing:
        j = j_in
        var sort_dir = (gid & k) == 0
        while j > 0:
            if j >= 32:
                var ixj = tid ^ j

                if tid < ixj:
                    var global_ixj = gid ^ j
                    if global_ixj < size:
                        var key_a = shared_keys[tid]
                        var key_b = shared_keys[ixj]

                        swap = (key_a > key_b) if sort_dir else (key_a < key_b)
                        if swap:
                            shared_keys[tid] = key_b
                            shared_keys[ixj] = key_a

                            var val_a = shared_values[tid]
                            var val_b = shared_values[ixj]
                            shared_values[tid] = val_b
                            shared_values[ixj] = val_a

                barrier()
            else:
                # Warp Shuffle Path
                var my_key = shared_keys[tid]
                var my_val = shared_values[tid]

                var shuffle_j = j
                while shuffle_j > 0:
                    var other_key = warp.shuffle_xor(my_key, UInt32(shuffle_j))
                    var other_val = warp.shuffle_xor(my_val, UInt32(shuffle_j))

                    var is_upper = (tid & shuffle_j) == 0
                    var should_swap = (my_key > other_key) == sort_dir

                    if (is_upper and should_swap) or (
                        not is_upper and not should_swap
                    ):
                        my_key = other_key
                        my_val = other_val

                    shuffle_j >>= 1

                shared_keys[tid] = my_key
                shared_values[tid] = my_val
                barrier()
                break
            j >>= 1

    var limit = min(THREADS_PER_BLOCK, size)

    comptime if IS_MERGE_BLOCK:
        var j = limit >> 1
        _step(j, k_merge)
    else:
        # block bitonic sort
        var k = 2
        while k <= limit:
            var j = k >> 1
            _step(j, k)
            k <<= 1

    # write back to global memory
    if gid < size:
        keys[gid] = shared_keys[tid]
        values[gid] = shared_values[tid]


def bitonic_sort[
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
                break

        k <<= 1

    ctx.synchronize()


# basic version of bitonic sort
@always_inline
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
    var gid = global_idx.x

    if gid < size:
        var ixj = gid ^ j

        if gid < ixj:
            var sort_dir = (gid & k) == 0
            var key_a = keys[gid]
            var key_b = keys[ixj]

            # If sort_dir is True we sort ascending, else descending
            if (sort_dir and key_a > key_b) or (not sort_dir and key_a < key_b):
                # Swap Keys
                keys[gid] = key_b
                keys[ixj] = key_a

                # Swap Values
                swap(values[gid], values[ixj])


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
            j >>= 1
        k <<= 1
    ctx.synchronize()
