from test.sort.helpers import _full_width_key

from std.testing import (
    TestSuite,
    assert_true,
    assert_equal,
)
from max.gpu.host import DeviceContext

from bajo.sort.gpu.radix_sort import (
    device_radix_sort_pairs,
    device_radix_sort_keys,
    RadixSortWorkspace,
)


def _assert_radix_keys_workspace[dtype: DType](size: Int) raises:
    with DeviceContext() as ctx:
        var keys = ctx.enqueue_create_buffer[dtype](size)
        var reference = ctx.enqueue_create_buffer[dtype](size)
        var values = ctx.enqueue_create_buffer[.uint32](size)
        with reference.map_to_host() as host, values.map_to_host() as ids:
            for i in range(size):
                host[i] = Scalar[dtype](_full_width_key(UInt32(i)))
                ids[i] = UInt32(i)
        reference.enqueue_copy_to(keys)
        device_radix_sort_pairs[dtype, .uint32](ctx, reference, values, size)
        ctx.synchronize()
        var workspace = RadixSortWorkspace[dtype, dtype](
            ctx, size, with_values=False
        )
        assert_equal(len(workspace.vals_alternate), 0)
        for repeat in range(3):
            if repeat > 0:
                with keys.map_to_host() as host:
                    for i in range(size):
                        host[i] = Scalar[dtype](_full_width_key(UInt32(i)))
            if repeat == 2:
                device_radix_sort_keys(ctx, keys, size)
            else:
                device_radix_sort_keys(ctx, workspace, keys, size)
            ctx.synchronize()
            with keys.map_to_host() as host, reference.map_to_host() as expected, values.map_to_host() as ids:
                for i in range(size):
                    assert_equal(host[i], expected[i])
                    var original = Int(ids[i])
                    assert_true(original < size)
                    assert_equal(
                        host[i],
                        Scalar[dtype](_full_width_key(UInt32(original))),
                    )
                    if i > 0:
                        assert_true(host[i - 1] <= host[i])
                        if host[i - 1] == host[i]:
                            assert_true(ids[i - 1] < ids[i])


def test_radix_keys_workspace_matches_stable_pairs() raises:
    comptime for dtype in [DType.uint32, DType.uint64]:
        for size in [1, 4608, 4609, 129 * 4608]:
            _assert_radix_keys_workspace[dtype](size)


def test_radix_empty_input_preserves_buffers() raises:
    with DeviceContext() as ctx:
        var keys = ctx.enqueue_create_buffer[.uint32](1)
        var values = ctx.enqueue_create_buffer[.uint32](1)
        keys.enqueue_fill(99)
        values.enqueue_fill(88)
        device_radix_sort_keys(ctx, keys, 0)
        device_radix_sort_pairs(ctx, keys, values, 0)
        ctx.synchronize()
        with keys.map_to_host() as host, values.map_to_host() as ids:
            assert_equal(host[0], UInt32(99))
            assert_equal(ids[0], UInt32(88))


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
