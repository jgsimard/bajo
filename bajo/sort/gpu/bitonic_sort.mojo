from std.bit import count_trailing_zeros
from std.gpu import (
    thread_idx,
    block_idx,
    barrier,
    global_idx,
    WARP_SIZE,
)
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.gpu.primitives import warp

from bajo.core.utils import is_power_of_2


@always_inline
def bitonic_sort_shared[
    keys_dtype: DType,
    vals_dtype: DType,
    *,
    THREADS_PER_BLOCK: Int,
    ITEMS_PER_THREAD: Int,
    IS_MERGE_BLOCK: Bool,
](
    keys: UnsafePointer[Scalar[keys_dtype], MutAnyOrigin],
    values: UnsafePointer[Scalar[vals_dtype], MutAnyOrigin],
    k_merge: Int,  # Target bitonic sequence length
    size: Int,
):
    """
    Sorts a tile of data of size `PART_SIZE` within a block. It has three stages: 1) SIMD, 2) Warp 3) block
    Then a frouth stage if necessary to sort between blocks
    """
    comptime PART_SIZE = THREADS_PER_BLOCK * ITEMS_PER_THREAD
    comptime MAX_WARP_K = ITEMS_PER_THREAD * WARP_SIZE
    comptime NUM_STAGES = count_trailing_zeros(PART_SIZE)

    var bid = block_idx.x
    var tid = thread_idx.x
    var block_start = bid * PART_SIZE

    var shared_keys = stack_allocation[
        PART_SIZE, Scalar[keys_dtype], address_space=AddressSpace.SHARED
    ]()
    var shared_vals = stack_allocation[
        PART_SIZE, Scalar[vals_dtype], address_space=AddressSpace.SHARED
    ]()

    var g_base = bid * PART_SIZE + (tid * ITEMS_PER_THREAD)
    var r_keys = SIMD[keys_dtype, ITEMS_PER_THREAD]()
    var r_vals = SIMD[vals_dtype, ITEMS_PER_THREAD]()

    if g_base < size:
        r_keys = keys.load[width=ITEMS_PER_THREAD](g_base)
        r_vals = values.load[width=ITEMS_PER_THREAD](g_base)
    else:
        r_keys = SIMD[keys_dtype, ITEMS_PER_THREAD](Scalar[keys_dtype].MAX)
        r_vals = SIMD[vals_dtype, ITEMS_PER_THREAD](0)

    @always_inline
    def _step(j: Int, k: Int) capturing:
        comptime for i in range(ITEMS_PER_THREAD / 2):
            var pair_id = tid + i * THREADS_PER_BLOCK

            var idx_a = ((pair_id & -j) << 1) | (pair_id & (j - 1))
            var idx_b = idx_a ^ j

            var global_idx_a = block_start + idx_a
            var sort_dir = (global_idx_a & k) == 0

            var key_a = shared_keys[idx_a]
            var key_b = shared_keys[idx_b]
            var val_a = shared_vals[idx_a]
            var val_b = shared_vals[idx_b]

            var should_swap = (key_a > key_b) if sort_dir else (key_a < key_b)

            shared_keys[idx_a] = key_b if should_swap else key_a
            shared_keys[idx_b] = key_a if should_swap else key_b
            shared_vals[idx_a] = val_b if should_swap else val_a
            shared_vals[idx_b] = val_a if should_swap else val_b

        barrier()

    comptime if not IS_MERGE_BLOCK:
        # Stage 1 & 2: SIMD -> WARP
        comptime for stage in range(1, NUM_STAGES + 1):
            comptime k = 1 << stage
            if k <= MAX_WARP_K:
                comptime for step_idx in range(stage):
                    comptime j = 1 << (stage - 1 - step_idx)
                    comptime if j < ITEMS_PER_THREAD:
                        # Stage 1: SIMD sort
                        var other_keys = SIMD[keys_dtype, ITEMS_PER_THREAD]()
                        var other_vals = SIMD[vals_dtype, ITEMS_PER_THREAD]()
                        var is_lower_vec = SIMD[DType.bool, ITEMS_PER_THREAD]()
                        var logical_i_vec = SIMD[
                            DType.uint32, ITEMS_PER_THREAD
                        ]()

                        comptime for i in range(ITEMS_PER_THREAD):
                            other_keys[i] = r_keys[i ^ j]
                            other_vals[i] = r_vals[i ^ j]
                            is_lower_vec[i] = (i & j) == 0
                            logical_i_vec[i] = UInt32(
                                tid * ITEMS_PER_THREAD + i
                            )

                        var sort_dir_vec = (logical_i_vec & UInt32(k)).eq(0)

                        var greater_mask = r_keys.gt(other_keys)
                        var less_mask = r_keys.lt(other_keys)

                        var should_swap = (sort_dir_vec & greater_mask) | (
                            ~sort_dir_vec & less_mask
                        )
                        var take_other = (is_lower_vec & should_swap) | (
                            ~is_lower_vec & ~should_swap
                        )

                        r_keys = take_other.select(other_keys, r_keys)
                        r_vals = take_other.select(other_vals, r_vals)
                    else:
                        # Stage 2: Warp sort
                        comptime thread_j = j / ITEMS_PER_THREAD
                        comptime for i in range(ITEMS_PER_THREAD):
                            var my_key = r_keys[i]
                            var my_val = r_vals[i]

                            var other_key = warp.shuffle_xor(
                                my_key, UInt32(thread_j)
                            )
                            var other_val = warp.shuffle_xor(
                                my_val, UInt32(thread_j)
                            )

                            var logical_i = tid * ITEMS_PER_THREAD + i
                            var sort_dir = ((block_start + logical_i) & k) == 0

                            var is_lower = (tid & thread_j) == 0
                            var should_swap = (
                                my_key > other_key
                            ) if sort_dir else (my_key < other_key)
                            var take_other = (is_lower and should_swap) or (
                                not is_lower and not should_swap
                            )

                            r_keys[i] = other_key if take_other else my_key
                            r_vals[i] = other_val if take_other else my_val

        # Stage 3: Block sort
        comptime if PART_SIZE > MAX_WARP_K:
            comptime for i in range(ITEMS_PER_THREAD):
                shared_keys[tid * ITEMS_PER_THREAD + i] = r_keys[i]
                shared_vals[tid * ITEMS_PER_THREAD + i] = r_vals[i]
            barrier()

            comptime start_stage = Int(count_trailing_zeros(MAX_WARP_K)) + 1
            comptime for stage in range(start_stage, NUM_STAGES + 1):
                comptime k = 1 << stage
                comptime for step_idx in range(stage):
                    comptime j = 1 << (stage - 1 - step_idx)
                    _step(j, k)

            comptime for i in range(ITEMS_PER_THREAD):
                r_keys[i] = shared_keys[tid * ITEMS_PER_THREAD + i]
                r_vals[i] = shared_vals[tid * ITEMS_PER_THREAD + i]

    else:
        # Stage 4: Cross-block merge
        comptime for i in range(ITEMS_PER_THREAD):
            shared_keys[tid * ITEMS_PER_THREAD + i] = r_keys[i]
            shared_vals[tid * ITEMS_PER_THREAD + i] = r_vals[i]
        barrier()

        var limit = min(PART_SIZE, size)
        var j = limit >> 1
        while j > 0:
            _step(j, k_merge)
            j >>= 1

        comptime for i in range(ITEMS_PER_THREAD):
            r_keys[i] = shared_keys[tid * ITEMS_PER_THREAD + i]
            r_vals[i] = shared_vals[tid * ITEMS_PER_THREAD + i]

    # write back to global memory
    if g_base < size:
        keys.store[width=ITEMS_PER_THREAD](g_base, r_keys)
        values.store[width=ITEMS_PER_THREAD](g_base, r_vals)


def bitonic_sort_pairs[
    keys_dtype: DType,
    vals_dtype: DType,
    //,
    THREADS_PER_BLOCK: Int = 256,
    ITEMS_PER_THREAD: Int = 4,
](
    ctx: DeviceContext,
    keys: DeviceBuffer[keys_dtype],
    values: DeviceBuffer[vals_dtype],
    size: Int,
) raises:
    """
    Bitonic Sort.
    """
    debug_assert["safe"](is_power_of_2(size))

    comptime PART_SIZE = THREADS_PER_BLOCK * ITEMS_PER_THREAD
    var blocks = (size + PART_SIZE - 1) / PART_SIZE

    # Phase 1: sort chunks of size PART_SIZE
    comptime shared_block_kernel = bitonic_sort_shared[
        keys_dtype,
        vals_dtype,
        THREADS_PER_BLOCK=THREADS_PER_BLOCK,
        ITEMS_PER_THREAD=ITEMS_PER_THREAD,
        IS_MERGE_BLOCK=False,
    ]
    ctx.enqueue_function[shared_block_kernel, shared_block_kernel](
        keys.unsafe_ptr(),
        values.unsafe_ptr(),
        0,
        size,
        grid_dim=blocks,
        block_dim=THREADS_PER_BLOCK,
    )

    # Phase 2: merge across blocks
    var k = PART_SIZE * 2
    while k <= size:
        var j = k / 2
        while j > 0:
            if j >= PART_SIZE:
                var total_pairs = size / 2
                var global_blocks = (
                    total_pairs + THREADS_PER_BLOCK - 1
                ) / THREADS_PER_BLOCK

                comptime global_step_kernel = bitonic_sort_step[
                    keys_dtype, vals_dtype
                ]
                ctx.enqueue_function[global_step_kernel, global_step_kernel](
                    keys.unsafe_ptr(),
                    values.unsafe_ptr(),
                    j,
                    k,
                    size,
                    grid_dim=global_blocks,
                    block_dim=THREADS_PER_BLOCK,
                )
                j /= 2
            else:
                comptime shared_merge_kernel = bitonic_sort_shared[
                    keys_dtype,
                    vals_dtype,
                    THREADS_PER_BLOCK=THREADS_PER_BLOCK,
                    ITEMS_PER_THREAD=ITEMS_PER_THREAD,
                    IS_MERGE_BLOCK=True,
                ]
                ctx.enqueue_function[shared_merge_kernel, shared_merge_kernel](
                    keys.unsafe_ptr(),
                    values.unsafe_ptr(),
                    k,
                    size,
                    grid_dim=blocks,
                    block_dim=THREADS_PER_BLOCK,
                )
                break
        k *= 2

    ctx.synchronize()


@always_inline
def bitonic_sort_step[
    keys_dtype: DType, vals_dtype: DType
](
    keys: UnsafePointer[Scalar[keys_dtype], MutAnyOrigin],
    values: UnsafePointer[Scalar[vals_dtype], MutAnyOrigin],
    j: Int,
    k: Int,
    size: Int,
):
    var pair_id = global_idx.x
    var total_pairs = size >> 1

    if pair_id >= total_pairs:
        return

    var idx_a = ((pair_id & -j) << 1) | (pair_id & (j - 1))
    var idx_b = idx_a ^ j

    var sort_dir = (idx_a & k) == 0

    var key_a = keys[idx_a]
    var key_b = keys[idx_b]
    var val_a = values[idx_a]
    var val_b = values[idx_b]

    var should_swap = (key_a > key_b) if sort_dir else (key_a < key_b)

    keys[idx_a] = key_b if should_swap else key_a
    keys[idx_b] = key_a if should_swap else key_b
    values[idx_a] = val_b if should_swap else val_a
    values[idx_b] = val_a if should_swap else val_b


def naive_bitonic_sort_pairs[
    keys_dtype: DType, vals_dtype: DType, //, THREADS_PER_BLOCK: Int = 256
](
    ctx: DeviceContext,
    keys: DeviceBuffer[keys_dtype],
    values: DeviceBuffer[vals_dtype],
    size: Int,
) raises:
    debug_assert["safe"](is_power_of_2(size))

    # 1 thread maps to 1 pair
    var total_pairs = size >> 1
    var blocks = (total_pairs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK

    var k = 2
    while k <= size:
        var j = k >> 1
        while j > 0:
            comptime kernel = bitonic_sort_step[keys_dtype, vals_dtype]
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
