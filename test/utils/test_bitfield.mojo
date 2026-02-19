from testing import assert_equal, assert_true, assert_false, TestSuite
from sys import simd_width_of

from bajo.utils.bitfield import BitField


def test_type_selection():
    assert_equal(BitField[8].dtype, DType.uint8)
    assert_equal(BitField[16].dtype, DType.uint16)
    assert_equal(BitField[32].dtype, DType.uint32)
    assert_equal(BitField[64].dtype, DType.uint64)
    assert_equal(BitField[128].dtype, DType.uint64)
    assert_equal(BitField[256].dtype, DType.uint64)
    assert_equal(BitField[512].dtype, DType.uint64)


def test_8bit_functionality():
    var bf = BitField[8]()

    for i in range(8):
        assert_false(bf[i], "Bit should be initialized to False")

    # Test setting and getting
    bf[0] = True
    bf[7] = True
    assert_true(bf[0])
    assert_true(bf[7])
    assert_false(bf[3], "Isolation check failed")

    # Test clearing
    bf[7] = False
    assert_false(bf[7], "Bit should have been cleared")
    assert_true(bf[0], "Clearing bit 7 should not affect bit 0")


def test_isolation_pattern():
    var bf = BitField[32]()
    # Set every even bit
    for i in range(0, 32, 2):
        bf[i] = True

    for i in range(32):
        if i % 2 == 0:
            assert_true(bf[i], "Even bit should be True")
        else:
            assert_false(bf[i], "Odd bit should be False")


def test_multi_lane_boundaries():
    # 128 bits uses 2 lanes of uint64
    var bf = BitField[128]()

    # Test boundary of first lane
    bf[63] = True
    # Test start of second lane
    bf[64] = True

    assert_true(bf[63], "Bit 63 (end of lane 0) failed")
    assert_true(bf[64], "Bit 64 (start of lane 1) failed")

    # Flip them
    bf[63] = False
    assert_false(bf[63])
    assert_true(bf[64])


def test_512bit_stress():
    var bf = BitField[512]()
    # Set bits across various lanes
    bf[0] = True
    bf[100] = True
    bf[200] = True
    bf[300] = True
    bf[400] = True
    bf[511] = True

    assert_true(bf[0])
    assert_true(bf[100])
    assert_true(bf[200])
    assert_true(bf[300])
    assert_true(bf[400])
    assert_true(bf[511])
    assert_false(bf[510])


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
