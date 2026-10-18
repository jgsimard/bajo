from std.math import ceildiv
from std.memory import stack_allocation
from std.sys.info import bit_width_of
from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from max.gpu import thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, DeviceBuffer
from bajo.sort.gpu.decoupled_fallback import (
    DecoupledFallbackWorkspace,
    decoupled_fallback_radix_sort_keys,
    decoupled_fallback_radix_sort_pairs,
    global_histogram,
    scan_global_fallback,
    digit_binning,
    publish_split_histogram,
    SPLIT_READY,
    SPLIT_INCLUSIVE,
    SPLIT_STATE_MASK,
    SPLIT_VALUE_MASK,
)
from test.sort.helpers import (
    _byte_pass_keys,
    _stable_byte_order,
    _full_width_key,
)


def _check_sorted_pairs[
    keys_dtype: DType, vals_dtype: DType
](
    keys: DeviceBuffer[keys_dtype], values: DeviceBuffer[vals_dtype], size: Int
) raises:
    with keys.map_to_host() as host, values.map_to_host() as ids:
        for i in range(size):
            var original = UInt32(ids[i])
            assert_true(Int(original) < size)
            assert_equal(
                ids[i],
                Scalar[vals_dtype](
                    (UInt64(0xFEDCBA98) << 32) | UInt64(original)
                ),
            )
            assert_equal(
                host[i],
                Scalar[keys_dtype](_full_width_key(original)),
            )
            if i > 0:
                assert_true(host[i - 1] <= host[i])
                if host[i - 1] == host[i]:
                    assert_true(UInt32(ids[i - 1]) < original)


def _check_pairs[
    keys_dtype: DType,
    vals_dtype: DType,
    *,
    KEYS_PER_THREAD: Int = 9,
    G_HIST_TPB: Int = 256,
    G_HIST_ITEMS_PER_THREAD: Int = 32,
    VEC_WIDTH: SIMDLength = 4,
](size: Int) raises:
    with DeviceContext() as ctx:
        var keys = ctx.enqueue_create_buffer[keys_dtype](size)
        var values = ctx.enqueue_create_buffer[vals_dtype](size)
        var workspace = DecoupledFallbackWorkspace[
            keys_dtype,
            vals_dtype,
            BLOCK_SIZE=512,
            KEYS_PER_THREAD=KEYS_PER_THREAD,
        ](ctx, size)
        for _repeat in range(2):
            with keys.map_to_host() as host, values.map_to_host() as ids:
                for i in range(size):
                    host[i] = Scalar[keys_dtype](_full_width_key(UInt32(i)))
                    ids[i] = Scalar[vals_dtype](
                        (UInt64(0xFEDCBA98) << 32) | UInt64(i)
                    )
            decoupled_fallback_radix_sort_pairs[
                keys_dtype,
                vals_dtype,
                BINNING_TPB=512,
                KEYS_PER_THREAD=KEYS_PER_THREAD,
                G_HIST_TPB=G_HIST_TPB,
                G_HIST_ITEMS_PER_THREAD=G_HIST_ITEMS_PER_THREAD,
                VEC_WIDTH=VEC_WIDTH,
            ](ctx, workspace, keys, values, size)
            ctx.synchronize()
            _check_sorted_pairs(keys, values, size)


def test_full_width_pairs_stability_tails_and_workspace_reuse() raises:
    comptime for keys_dtype in [DType.uint32, DType.uint64]:
        comptime for vals_dtype in [DType.uint32, DType.uint64]:
            for size in [0, 1, 4608, 4609]:
                _check_pairs[keys_dtype, vals_dtype](size)


def test_keys_only_and_allocating_pairs_agree() raises:
    with DeviceContext() as ctx:
        var size = 20003
        var keys = ctx.enqueue_create_buffer[.uint64](size)
        var pairs = ctx.enqueue_create_buffer[.uint64](size)
        var ids = ctx.enqueue_create_buffer[.uint32](size)
        with keys.map_to_host() as k, pairs.map_to_host() as p, ids.map_to_host() as v:
            for i in range(size):
                k[i] = _full_width_key(UInt32(i))
                p[i] = k[i]
                v[i] = UInt32(i)
        var workspace = DecoupledFallbackWorkspace[
            .uint64, .uint64, BLOCK_SIZE=512, KEYS_PER_THREAD=9
        ](ctx, size)
        decoupled_fallback_radix_sort_keys[
            .uint64,
            BINNING_TPB=512,
            KEYS_PER_THREAD=9,
            G_HIST_TPB=64,
            G_HIST_ITEMS_PER_THREAD=64,
            VEC_WIDTH=2,
        ](ctx, workspace, keys, size)
        decoupled_fallback_radix_sort_pairs[
            BLOCK_SIZE=256,
            KEYS_PER_THREAD=7,
            G_HIST_TPB=256,
            G_HIST_ITEMS_PER_THREAD=32,
            VEC_WIDTH=8,
        ](ctx, pairs, ids, size)
        ctx.synchronize()
        with keys.map_to_host() as k, pairs.map_to_host() as p:
            for i in range(size):
                assert_equal(k[i], p[i])


def test_custom_histogram_configuration_pairs_and_workspace_reuse() raises:
    _check_pairs[
        .uint32,
        .uint64,
        G_HIST_TPB=256,
        G_HIST_ITEMS_PER_THREAD=32,
        VEC_WIDTH=8,
    ](8193)


def test_small_tiles_keep_partition_id_outside_histograms() raises:
    _check_pairs[.uint64, .uint64, KEYS_PER_THREAD=7](20003)
    _check_pairs[.uint64, .uint64, KEYS_PER_THREAD=1](2003)
    _forced_pass[True, 512, KEYS_PER_THREAD=7](20003)


def test_workspace_reuse_with_different_active_sizes() raises:
    with DeviceContext() as ctx:
        var capacity = 131073
        var keys = ctx.enqueue_create_buffer[.uint32](capacity)
        var values = ctx.enqueue_create_buffer[.uint64](capacity)
        var workspace = DecoupledFallbackWorkspace[
            .uint32, .uint64, BLOCK_SIZE=512, KEYS_PER_THREAD=9
        ](ctx, capacity)
        for size in [capacity, 17, 8193, 0, capacity]:
            with keys.map_to_host() as host, values.map_to_host() as ids:
                for i in range(size):
                    host[i] = UInt32(_full_width_key(UInt32(i)))
                    ids[i] = (UInt64(0xFEDCBA98) << 32) | UInt64(i)
            decoupled_fallback_radix_sort_pairs[
                BINNING_TPB=512, KEYS_PER_THREAD=9
            ](ctx, workspace, keys, values, size)
            ctx.synchronize()
            _check_sorted_pairs(keys, values, size)


def _forced_pass[
    pairs: Bool, block_size: Int, *, KEYS_PER_THREAD: Int = 9
](size: Int) raises:
    """Omit ALL owner reductions and inclusive prefixes, forcing helpers.

    Unlike an artificial delay, omitted publications remain missing forever
    unless the fallback reconstructs them. The final output must still match
    an independent stable counting-sort reference byte for byte.
    """
    with DeviceContext() as ctx:
        var tiles = ceildiv(size, block_size * KEYS_PER_THREAD)
        var original = _byte_pass_keys[pairs](size)
        var expected = _stable_byte_order(original)
        var keys = ctx.enqueue_create_buffer[.uint32](size)
        var output = ctx.enqueue_create_buffer[.uint32](size)
        var values = ctx.enqueue_create_buffer[.uint32](size)
        var value_output = ctx.enqueue_create_buffer[.uint32](size)
        var global_hist = ctx.enqueue_create_buffer[.uint32](4 * 256)
        var spine = ctx.enqueue_create_buffer[.uint32]((tiles + 1) * 512)
        var index = ctx.enqueue_create_buffer[.uint32](4)
        with keys.map_to_host() as k, values.map_to_host() as v:
            for i in range(size):
                k[i] = original[i]
                v[i] = UInt32(i)
        global_hist.enqueue_fill(0)
        spine.enqueue_fill(0)
        index.enqueue_fill(0)
        ctx.enqueue_function[
            global_histogram[
                .uint32,
                BLOCK_SIZE=128,
                RADIX=256,
                VEC_WIDTH=4,
                ITEMS_PER_THREAD=128,
            ]
        ](
            keys,
            global_hist,
            Int32(size),
            grid_dim=ceildiv(size, 16384),
            block_dim=128,
        )
        ctx.enqueue_function[scan_global_fallback](
            global_hist, spine, Int32(tiles + 1), grid_dim=1, block_dim=256
        )
        ctx.enqueue_function[
            digit_binning[
                .uint32,
                .uint32,
                BITS_PER_PASS=8,
                BLOCK_SIZE=block_size,
                KEYS_PER_THREAD=KEYS_PER_THREAD,
                HAVE_PAYLOAD=pairs,
                _TEST_SUPPRESS_PUBLICATION=True,
            ]
        ](
            keys,
            output,
            Optional(values.unsafe_ptr()),
            Optional(value_output.unsafe_ptr()),
            spine,
            index,
            Int32(size),
            UInt32(0),
            grid_dim=tiles,
            block_dim=block_size,
        )
        ctx.synchronize()
        with output.map_to_host() as result:
            for i in range(size):
                assert_equal(result[i], original[expected[i]])
        comptime if pairs:
            with value_output.map_to_host() as result:
                for i in range(size):
                    assert_equal(result[i], UInt32(expected[i]))
        with spine.map_to_host() as result:
            # Every nonfinal row can only have been published by a helper.
            for row in range(1, tiles):
                for word in range(512):
                    assert_equal(
                        result[row * 512 + word] & SPLIT_STATE_MASK, SPLIT_READY
                    )


def test_forced_fallback_pairs_oversubscribed_grid() raises:
    # 911 workgroups exceed the maximum resident 512-thread groups on the
    # development GPU. The same test also exercises a partial final tile.
    _forced_pass[True, 512](4194307)


def test_forced_fallback_keys_smaller_workgroups() raises:
    _forced_pass[False, 256](20003)


def _publish[
    STATE: UInt32
](counts: Pointer[UInt32, MutAnyOrigin], spine: Pointer[UInt32, MutAnyOrigin]):
    var shared = stack_allocation[
        256, UInt32, address_space=AddressSpace.SHARED
    ]()
    if thread_idx.x < 256:
        shared[unsafe_offset=thread_idx.x] = counts[unsafe_offset=thread_idx.x]
    barrier()
    publish_split_histogram[512, STATE](spine, shared, 0)


def test_split_states_full_uint32_and_late_owner_do_not_regress() raises:
    with DeviceContext() as ctx:
        var counts = ctx.enqueue_create_buffer[.uint32](256)
        var spine = ctx.enqueue_create_buffer[.uint32](512)
        spine.enqueue_fill(0)
        counts.enqueue_fill(UInt32(0x8000FFF0))
        ctx.enqueue_function[_publish[SPLIT_READY]](
            counts, spine, grid_dim=1, block_dim=512
        )
        counts.enqueue_fill(UInt32(0x80010010))
        ctx.enqueue_function[_publish[SPLIT_INCLUSIVE]](
            counts, spine, grid_dim=1, block_dim=512
        )
        counts.enqueue_fill(UInt32(0x8000FFF0))
        ctx.enqueue_function[_publish[SPLIT_READY]](
            counts, spine, grid_dim=1, block_dim=512
        )
        ctx.synchronize()
        with spine.map_to_host() as result:
            for digit in range(256):
                var low = result[2 * digit]
                var high = result[2 * digit + 1]
                assert_equal(low & SPLIT_STATE_MASK, SPLIT_INCLUSIVE)
                assert_equal(high & SPLIT_STATE_MASK, SPLIT_INCLUSIVE)
                assert_equal(
                    (low & SPLIT_VALUE_MASK)
                    | ((high & SPLIT_VALUE_MASK) << 16),
                    UInt32(0x80010010),
                )


def test_size_validation() raises:
    with DeviceContext() as ctx:
        var keys = ctx.enqueue_create_buffer[.uint32](33)
        var values = ctx.enqueue_create_buffer[.uint32](33)
        var workspace = DecoupledFallbackWorkspace[
            .uint32, .uint32, BLOCK_SIZE=512, KEYS_PER_THREAD=9
        ](ctx, 33)
        for size in [-1, 34, Int(Int32.MAX) + 1]:
            with assert_raises():
                decoupled_fallback_radix_sort_pairs[
                    .uint32, .uint32, BINNING_TPB=512, KEYS_PER_THREAD=9
                ](ctx, workspace, keys, values, size)


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
