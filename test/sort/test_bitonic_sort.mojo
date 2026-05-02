from std.math import sqrt
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_true,
    assert_false,
    assert_equal,
)
from std.gpu.host import DeviceContext

from bajo.sort.gpu.bitonic_sort import (
    naive_bitonic_sort_pairs,
    bitonic_sort_pairs,
)


def test_bitonic_sort_shared() raises:
    with DeviceContext() as ctx:
        comptime dtype = DType.uint32
        comptime SIZE = 8

        keys = ctx.enqueue_create_buffer[dtype](SIZE)
        values = ctx.enqueue_create_buffer[dtype](SIZE)

        _keys: List[UInt32] = [45, 12, 89, 1, 34, 99, 2, 23]
        with keys.map_to_host() as host_keys, values.map_to_host() as host_values:
            for i in range(SIZE):
                host_keys[i] = _keys[i]
                host_values[i] = UInt32(i)

        bitonic_sort_pairs(ctx, keys, values, SIZE)
        ctx.synchronize()

        expected_keys: List[UInt32] = [1, 2, 12, 23, 34, 45, 89, 99]
        expected_vals: List[UInt32] = [3, 6, 1, 7, 4, 0, 2, 5]
        with keys.map_to_host() as host_keys, values.map_to_host() as host_values:
            for i in range(SIZE):
                assert_equal(host_keys[i], expected_keys[i])
                assert_equal(host_values[i], expected_vals[i])


def test_bitonic_sort_basic() raises:
    with DeviceContext() as ctx:
        comptime dtype = DType.uint32
        comptime SIZE = 8

        keys = ctx.enqueue_create_buffer[dtype](SIZE)
        values = ctx.enqueue_create_buffer[dtype](SIZE)

        _keys: List[UInt32] = [45, 12, 89, 1, 34, 99, 2, 23]
        with keys.map_to_host() as host_keys, values.map_to_host() as host_values:
            for i in range(SIZE):
                host_keys[i] = _keys[i]
                host_values[i] = UInt32(i)

        naive_bitonic_sort_pairs(ctx, keys, values, SIZE)

        ctx.synchronize()

        expected_keys: List[UInt32] = [1, 2, 12, 23, 34, 45, 89, 99]
        expected_vals: List[UInt32] = [3, 6, 1, 7, 4, 0, 2, 5]
        with keys.map_to_host() as host_keys, values.map_to_host() as host_values:
            for i in range(SIZE):
                assert_equal(host_keys[i], expected_keys[i])
                assert_equal(host_values[i], expected_vals[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
