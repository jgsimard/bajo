"""GPU integration checks: reject bad inputs before dispatch or mutation."""

from std.testing import TestSuite, assert_equal, assert_raises
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.sort.gpu import gpu_sort_pairs
from bajo.sort.gpu.bitonic_sort import (
    bitonic_sort_pairs,
    naive_bitonic_sort_pairs,
)
from bajo.sort.gpu.onesweep import (
    OneSweepWorkspace,
    onesweep_radix_sort_keys,
    onesweep_radix_sort_pairs,
)
from bajo.sort.gpu.radix_sort import (
    RadixSortWorkspace,
    device_radix_sort_keys,
    device_radix_sort_pairs,
)
from bajo.sort.gpu.decoupled_fallback import (
    DecoupledFallbackWorkspace,
    decoupled_fallback_radix_sort_keys,
    decoupled_fallback_radix_sort_pairs,
)


def _sort[
    algorithm: Int
](
    ctx: DeviceContext,
    mut keys: DeviceBuffer[.uint32],
    mut values: DeviceBuffer[.uint32],
    size: Int,
) raises:
    comptime if algorithm == 0:
        gpu_sort_pairs(ctx, keys, values, size)
    elif algorithm == 1:
        bitonic_sort_pairs(ctx, keys, values, size)
    elif algorithm == 2:
        naive_bitonic_sort_pairs(ctx, keys, values, size)
    elif algorithm == 3:
        onesweep_radix_sort_pairs(ctx, keys, values, size)
    elif algorithm == 4:
        device_radix_sort_keys(ctx, keys, size)
    elif algorithm == 5:
        device_radix_sort_pairs(ctx, keys, values, size)
    else:
        decoupled_fallback_radix_sort_pairs(ctx, keys, values, size)


def test_all_allocating_entry_points_validate_before_early_return() raises:
    with DeviceContext() as ctx:
        var keys = ctx.enqueue_create_buffer[.uint32](0)
        var values = ctx.enqueue_create_buffer[.uint32](0)
        comptime for algorithm in range(7):
            _sort[algorithm](ctx, keys, values, 0)
            for size in [-1, 1, 2, Int(Int32.MAX) + 1]:
                with assert_raises():
                    _sort[algorithm](ctx, keys, values, size)


def test_short_payload_is_rejected_before_dispatch() raises:
    with DeviceContext() as ctx:
        var keys = ctx.enqueue_create_buffer[.uint32](32)
        var values = ctx.enqueue_create_buffer[.uint32](31)
        keys.enqueue_fill(123)
        values.enqueue_fill(456)
        comptime for algorithm in [0, 1, 2, 3, 5, 6]:
            with assert_raises():
                _sort[algorithm](ctx, keys, values, 32)
        ctx.synchronize()
        with keys.map_to_host() as k, values.map_to_host() as v:
            for index in range(32):
                assert_equal(k[index], UInt32(123))
            for index in range(31):
                assert_equal(v[index], UInt32(456))


def _reject_onesweep_workspace[
    damage: Int, pairs: Bool
](
    ctx: DeviceContext,
    mut keys: DeviceBuffer[.uint32],
    mut values: DeviceBuffer[.uint32],
) raises:
    var workspace = OneSweepWorkspace[
        .uint32, .uint32, BLOCK_SIZE=512, KEYS_PER_THREAD=15
    ](ctx, 1 if damage == 0 else 32)
    comptime if damage == 1:
        workspace.vals_alternate = ctx.enqueue_create_buffer[.uint32](0)
    elif damage == 2:
        workspace.global_hist = ctx.enqueue_create_buffer[.uint32](0)
    elif damage == 3:
        workspace.pass_hist = ctx.enqueue_create_buffer[.uint32](0)
    elif damage == 4:
        workspace.index = ctx.enqueue_create_buffer[.uint32](0)

    with assert_raises():
        comptime if pairs:
            onesweep_radix_sort_pairs[BINNING_TPB=512, KEYS_PER_THREAD=15](
                ctx, workspace, keys, values, 32
            )
        else:
            onesweep_radix_sort_keys[BINNING_TPB=512, KEYS_PER_THREAD=15](
                ctx, workspace, keys, 32
            )


def _reject_radix_workspace[
    damage: Int, pairs: Bool
](
    ctx: DeviceContext,
    mut keys: DeviceBuffer[.uint32],
    mut values: DeviceBuffer[.uint32],
) raises:
    var workspace = RadixSortWorkspace[.uint32, .uint32](
        ctx, 1 if damage == 0 else 32
    )
    comptime if damage == 1:
        workspace.vals_alternate = ctx.enqueue_create_buffer[.uint32](0)
    elif damage == 2:
        workspace.global_hist = ctx.enqueue_create_buffer[.uint32](0)
    elif damage == 3:
        workspace.pass_hist = ctx.enqueue_create_buffer[.uint32](0)

    with assert_raises():
        comptime if pairs:
            device_radix_sort_pairs(ctx, workspace, keys, values, 32)
        else:
            device_radix_sort_keys(ctx, workspace, keys, 32)


def _reject_fallback_workspace[
    damage: Int, pairs: Bool
](
    ctx: DeviceContext,
    mut keys: DeviceBuffer[.uint32],
    mut values: DeviceBuffer[.uint32],
) raises:
    var workspace = DecoupledFallbackWorkspace[
        .uint32, .uint32, BLOCK_SIZE=512, KEYS_PER_THREAD=15
    ](ctx, 1 if damage == 0 else 32)
    comptime if damage == 1:
        workspace.vals_alternate = ctx.enqueue_create_buffer[.uint32](0)
    elif damage == 2:
        workspace.global_hist = ctx.enqueue_create_buffer[.uint32](0)
    elif damage == 3:
        workspace.pass_hist = ctx.enqueue_create_buffer[.uint32](0)
    elif damage == 4:
        workspace.index = ctx.enqueue_create_buffer[.uint32](0)

    with assert_raises():
        comptime if pairs:
            decoupled_fallback_radix_sort_pairs[
                BINNING_TPB=512, KEYS_PER_THREAD=15
            ](ctx, workspace, keys, values, 32)
        else:
            decoupled_fallback_radix_sort_keys[
                BINNING_TPB=512, KEYS_PER_THREAD=15
            ](ctx, workspace, keys, 32)


def test_every_retained_workspace_buffer_is_validated() raises:
    with DeviceContext() as ctx:
        var keys = ctx.enqueue_create_buffer[.uint32](32)
        var values = ctx.enqueue_create_buffer[.uint32](32)
        keys.enqueue_fill(123)
        values.enqueue_fill(456)
        comptime for pairs in [False, True]:
            comptime for damage in range(5):
                comptime if damage != 1 or pairs:
                    _reject_onesweep_workspace[damage, pairs](ctx, keys, values)
                    _reject_fallback_workspace[damage, pairs](ctx, keys, values)
                    comptime if damage != 4:
                        _reject_radix_workspace[damage, pairs](
                            ctx, keys, values
                        )
        ctx.synchronize()
        with keys.map_to_host() as k, values.map_to_host() as v:
            for index in range(32):
                assert_equal(k[index], UInt32(123))
                assert_equal(v[index], UInt32(456))


def test_general_sort_handles_non_power_of_two_prefixes() raises:
    with DeviceContext() as ctx:
        for size in [0, 1, 3, 33]:
            # Retain a sentinel immediately past the requested prefix.
            var keys = ctx.enqueue_create_buffer[.uint32](size + 1)
            var values = ctx.enqueue_create_buffer[.uint32](size + 1)
            with keys.map_to_host() as k, values.map_to_host() as v:
                for index in range(size):
                    k[index] = UInt32(size - index)
                    v[index] = UInt32(index)
                k[size] = 12345
                v[size] = 67890
            gpu_sort_pairs(ctx, keys, values, size)
            ctx.synchronize()
            with keys.map_to_host() as k, values.map_to_host() as v:
                for index in range(size):
                    assert_equal(k[index], UInt32(index + 1))
                    assert_equal(v[index], UInt32(size - index - 1))
                assert_equal(k[size], UInt32(12345))
                assert_equal(v[size], UInt32(67890))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
