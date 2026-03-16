from std.math import sqrt
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_true,
    assert_false,
    assert_equal,
)
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from bajo.core.sort.gpu import bitonic_sort, radix_sort


def test_bitonic_sort_gpu() raises:
    with DeviceContext() as ctx:
        comptime dtype = DType.uint16
        comptime SIZE = 8

        keys = ctx.enqueue_create_buffer[dtype](SIZE)
        keys.enqueue_fill(0)

        values = ctx.enqueue_create_buffer[dtype](SIZE)
        values.enqueue_fill(0)

        with keys.map_to_host() as host_keys, values.map_to_host() as host_values:
            # fmt: off
            host_keys[0] = 45; host_values[0] = 0
            host_keys[1] = 12; host_values[1] = 1
            host_keys[2] = 89; host_values[2] = 2
            host_keys[3] = 1;  host_values[3] = 3
            host_keys[4] = 34; host_values[4] = 4
            host_keys[5] = 99; host_values[5] = 5
            host_keys[6] = 2;  host_values[6] = 6
            host_keys[7] = 23; host_values[7] = 7
            # fmt: on

        bitonic_sort(ctx, keys, values, SIZE)

        ctx.synchronize()

        with keys.map_to_host() as host_keys, values.map_to_host() as host_values:
            assert_equal(host_keys[0], 1)
            assert_equal(host_keys[1], 2)
            assert_equal(host_keys[2], 12)
            assert_equal(host_keys[3], 23)
            assert_equal(host_keys[4], 34)
            assert_equal(host_keys[5], 45)
            assert_equal(host_keys[6], 89)
            assert_equal(host_keys[7], 99)

            assert_equal(host_values[0], 3)
            assert_equal(host_values[1], 6)
            assert_equal(host_values[2], 1)
            assert_equal(host_values[3], 7)
            assert_equal(host_values[4], 4)
            assert_equal(host_values[5], 0)
            assert_equal(host_values[6], 2)
            assert_equal(host_values[7], 5)


fn test_radix_sort_gpu() raises:
    with DeviceContext() as ctx:
        comptime dtype = DType.uint16
        comptime SIZE = 8
        comptime N_BITS = 8

        var keys = ctx.enqueue_create_buffer[dtype](SIZE)
        keys.enqueue_fill(0)

        var values = ctx.enqueue_create_buffer[dtype](SIZE)
        values.enqueue_fill(0)

        with keys.map_to_host() as host_keys, values.map_to_host() as host_values:
            # fmt: off
            host_keys[0] = 45; host_values[0] = 0
            host_keys[1] = 12; host_values[1] = 1
            host_keys[2] = 89; host_values[2] = 2
            host_keys[3] = 1;  host_values[3] = 3
            host_keys[4] = 34; host_values[4] = 4
            host_keys[5] = 99; host_values[5] = 5
            host_keys[6] = 2;  host_values[6] = 6
            host_keys[7] = 23; host_values[7] = 7
            # fmt: on

        radix_sort[N_BITS](ctx, keys, values, SIZE)

        ctx.synchronize()

        with keys.map_to_host() as host_keys, values.map_to_host() as host_values:
            assert_equal(host_keys[0], 1)
            assert_equal(host_keys[1], 2)
            assert_equal(host_keys[2], 12)
            assert_equal(host_keys[3], 23)
            assert_equal(host_keys[4], 34)
            assert_equal(host_keys[5], 45)
            assert_equal(host_keys[6], 89)
            assert_equal(host_keys[7], 99)

            assert_equal(host_values[0], 3)
            assert_equal(host_values[1], 6)
            assert_equal(host_values[2], 1)
            assert_equal(host_values[3], 7)
            assert_equal(host_values[4], 4)
            assert_equal(host_values[5], 0)
            assert_equal(host_values[6], 2)
            assert_equal(host_values[7], 5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
