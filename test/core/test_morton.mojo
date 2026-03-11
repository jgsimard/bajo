from std.testing import assert_equal, TestSuite
from std.math import clamp

from bajo.core.morton import expand_bits_2d, expand_bits_3d, morton3


def test_expand_bits_2d() raises:
    # 0b11 (3) -> 0b101 (5)
    assert_equal(expand_bits_2d(UInt32(3)), 5)

    # 0b1111 (15) -> 0b1010101 (85)
    assert_equal(expand_bits_2d(UInt32(15)), 85)


def test_expand_bits_3d() raises:
    # 0b1 (1) -> 0b1 (1)
    assert_equal(expand_bits_3d(UInt32(1)), 1)

    # 0b10 (2) -> 0b1000 (8)
    assert_equal(expand_bits_3d(UInt32(2)), 8)

    # 0b100 (4) -> 0b1000000 (64)
    assert_equal(expand_bits_3d(UInt32(4)), 64)


def test_morton3_logic() raises:
    # dim=1024 (10 bits per axis)
    # If x=1/1024, y=0, z=0 -> index should be 1 (001)
    # If x=0, y=1/1024, z=0 -> index should be 2 (010)
    # If x=0, y=0, z=1/1024 -> index should be 4 (100)

    comptime size = 1
    dim: UInt32 = 1024
    step = 1.0 / Float32(dim)

    # Origin
    assert_equal(morton3[size](0.0, 0.0, 0.0), 0)

    # X step
    assert_equal(morton3[size](step, 0.0, 0.0), 1)

    # Y step
    assert_equal(morton3[size](0.0, step, 0.0), 2)

    # Z step
    assert_equal(morton3[size](0.0, 0.0, step), 4)

    # Combined (1, 1, 1) scaling
    assert_equal(morton3[size](step, step, step), 7)  # 0b111


def test_morton3_boundaries() raises:
    # Check clamping and max value
    # Max value (1.0, 1.0, 1.0) should map to (1023, 1023, 1023)
    # Since 1023 is 10 bits of 1s, expanded it should be 30 bits of 1s
    m_max = morton3[size=1](1.0, 1.0, 1.0)
    expected: UInt32 = (1 << 30) - 1
    assert_equal(m_max, expected)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
