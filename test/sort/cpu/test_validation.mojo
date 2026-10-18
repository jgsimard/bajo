"""Exercise device ABI limits without allocating huge device buffers."""

from std.testing import TestSuite, assert_equal, assert_raises

from bajo.sort.gpu.validation import (
    validate_sort_size,
    require_sort_capacity,
    validate_onesweep_size,
    validate_bitonic_size,
)


def test_sort_size_limits() raises:
    for size in [0, 1, Int(Int32.MAX)]:
        validate_sort_size(size)
    for size in [-1, Int(Int32.MAX) + 1, Int.MAX]:
        with assert_raises():
            validate_sort_size(size)


def test_buffer_prefix_capacity() raises:
    require_sort_capacity(0, 0, "empty input")
    require_sort_capacity(33, 33, "exact input")
    require_sort_capacity(33, 100, "retained scratch")
    with assert_raises():
        require_sort_capacity(1, 0, "singleton input")
    with assert_raises():
        require_sort_capacity(33, 32, "short scratch")


def test_onesweep_count_and_histogram_limits() raises:
    assert_equal(validate_onesweep_size[512 * 15, 4](0), 1024)
    assert_equal(validate_onesweep_size[512 * 15, 4](7680), 2048)
    assert_equal(validate_onesweep_size[512 * 15, 4](7681), 3072)
    # The largest representable prefix is valid with the default partition size.
    _ = validate_onesweep_size[512 * 15, 4](0x3FFFFFFF)
    with assert_raises():
        _ = validate_onesweep_size[512 * 15, 4](0x40000000)
    # Small partitions and 64-bit keys can overflow the histogram's Int32 ABI
    # before reaching the 30-bit count limit. No huge allocation is needed here.
    _ = validate_onesweep_size[256, 8](268434944)
    with assert_raises():
        _ = validate_onesweep_size[256, 8](268434945)


def test_bitonic_size_contract() raises:
    for size in [0, 1, 2, 32, 1 << 30]:
        validate_bitonic_size(size)
    for size in [-1, 3, 33, 1 << 31]:
        with assert_raises():
            validate_bitonic_size(size)


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
