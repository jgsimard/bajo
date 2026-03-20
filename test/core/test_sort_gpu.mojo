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

from bajo.core.sort.gpu import bitonic_sort, bitonic_sort_basic, radix_sort


def test_bitonic_sort_shared() raises:
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


def test_bitonic_sort_basic() raises:
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

        bitonic_sort_basic(ctx, keys, values, SIZE)

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


# fn test_radix_sort() raises:
# with DeviceContext() as ctx:
#     comptime dtype = DType.uint16
#     comptime SIZE = 8
#     comptime N_BITS = 8

#     var keys = ctx.enqueue_create_buffer[dtype](SIZE)
#     keys.enqueue_fill(0)

#     var values = ctx.enqueue_create_buffer[dtype](SIZE)
#     values.enqueue_fill(0)

#     with keys.map_to_host() as host_keys, values.map_to_host() as host_values:
#         # fmt: off
#         host_keys[0] = 45; host_values[0] = 0
#         host_keys[1] = 12; host_values[1] = 1
#         host_keys[2] = 89; host_values[2] = 2
#         host_keys[3] = 1;  host_values[3] = 3
#         host_keys[4] = 34; host_values[4] = 4
#         host_keys[5] = 99; host_values[5] = 5
#         host_keys[6] = 2;  host_values[6] = 6
#         host_keys[7] = 23; host_values[7] = 7
#         # fmt: on

#     radix_sort[N_BITS](ctx, keys, values, SIZE)

#     ctx.synchronize()

#     with keys.map_to_host() as host_keys, values.map_to_host() as host_values:
#         assert_equal(host_keys[0], 1)
#         assert_equal(host_keys[1], 2)
#         assert_equal(host_keys[2], 12)
#         assert_equal(host_keys[3], 23)
#         assert_equal(host_keys[4], 34)
#         assert_equal(host_keys[5], 45)
#         assert_equal(host_keys[6], 89)
#         assert_equal(host_keys[7], 99)

#         assert_equal(host_values[0], 3)
#         assert_equal(host_values[1], 6)
#         assert_equal(host_values[2], 1)
#         assert_equal(host_values[3], 7)
#         assert_equal(host_values[4], 4)
#         assert_equal(host_values[5], 0)
#         assert_equal(host_values[6], 2)
#         assert_equal(host_values[7], 5)


def test_radix_sort_basic() raises:
    """Tests basic key-only radix sort on a small, unsorted array."""
    with DeviceContext() as ctx:
        comptime dtype = DType.uint32
        comptime SIZE = 8

        keys = ctx.enqueue_create_buffer[dtype](SIZE)

        with keys.map_to_host() as host_keys:
            # fmt: off
            host_keys[0] = 45
            host_keys[1] = 12
            host_keys[2] = 89
            host_keys[3] = 1
            host_keys[4] = 34
            host_keys[5] = 99
            host_keys[6] = 2
            host_keys[7] = 23
            # fmt: on

        # Assuming the dispatcher is updated to take external buffers:
        # radix_sort(ctx, keys, SIZE)
        radix_sort(ctx, keys, SIZE)

        ctx.synchronize()

        with keys.map_to_host() as host_keys:
            assert_equal(host_keys[0], 1)
            assert_equal(host_keys[1], 2)
            assert_equal(host_keys[2], 12)
            assert_equal(host_keys[3], 23)
            assert_equal(host_keys[4], 34)
            assert_equal(host_keys[5], 45)
            assert_equal(host_keys[6], 89)
            assert_equal(host_keys[7], 99)


def test_radix_sort_key_value() raises:
    """Tests key-value radix sort to ensure payloads are swapped correctly."""
    with DeviceContext() as ctx:
        comptime dtype = DType.uint32
        comptime SIZE = 8

        keys = ctx.enqueue_create_buffer[dtype](SIZE)
        values = ctx.enqueue_create_buffer[dtype](SIZE)

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

        # Key-Value dispatcher
        radix_sort(ctx, keys, values, SIZE)

        ctx.synchronize()

        with keys.map_to_host() as host_keys, values.map_to_host() as host_values:
            # Check Keys
            assert_equal(host_keys[0], 1)
            assert_equal(host_keys[7], 99)

            # Check Values (Payloads followed the keys)
            assert_equal(host_values[0], 3)  # Key 1 was originally at index 3
            assert_equal(host_values[1], 6)  # Key 2 was originally at index 6
            assert_equal(host_values[2], 1)  # Key 12 was originally at index 1
            assert_equal(host_values[3], 7)  # Key 23 was originally at index 7
            assert_equal(host_values[4], 4)  # Key 34 was originally at index 4
            assert_equal(host_values[5], 0)  # Key 45 was originally at index 0
            assert_equal(host_values[6], 2)  # Key 89 was originally at index 2
            assert_equal(host_values[7], 5)  # Key 99 was originally at index 5


def test_radix_sort_duplicates_stability() raises:
    """
    Radix sort is a stable sort. This tests that identical keys
    retain their original relative order (checked via values).
    """
    with DeviceContext() as ctx:
        comptime dtype = DType.uint32
        comptime SIZE = 6

        keys = ctx.enqueue_create_buffer[dtype](SIZE)
        values = ctx.enqueue_create_buffer[dtype](SIZE)

        with keys.map_to_host() as host_keys, values.map_to_host() as host_values:
            # fmt: off
            host_keys[0] = 42; host_values[0] = 0
            host_keys[1] = 10; host_values[1] = 1
            host_keys[2] = 42; host_values[2] = 2 # Duplicate 1
            host_keys[3] = 42; host_values[3] = 3 # Duplicate 2
            host_keys[4] = 5;  host_values[4] = 4
            host_keys[5] = 10; host_values[5] = 5 # Duplicate 3
            # fmt: on

        radix_sort(ctx, keys, values, SIZE)
        ctx.synchronize()

        with keys.map_to_host() as host_keys, values.map_to_host() as host_values:
            # Keys check
            assert_equal(host_keys[0], 5)
            assert_equal(host_keys[1], 10)
            assert_equal(host_keys[2], 10)
            assert_equal(host_keys[3], 42)
            assert_equal(host_keys[4], 42)
            assert_equal(host_keys[5], 42)

            # Stability check: Values for the identical keys must remain in ascending order
            assert_equal(host_values[1], 1)  # First 10
            assert_equal(host_values[2], 5)  # Second 10

            assert_equal(host_values[3], 0)  # First 42
            assert_equal(host_values[4], 2)  # Second 42
            assert_equal(host_values[5], 3)  # Third 42


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
