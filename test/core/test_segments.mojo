from std.testing import TestSuite, assert_equal, assert_raises

from bajo.core import SegmentOffsets


def test_segment_offsets_from_counts() raises:
    var segments = SegmentOffsets.from_counts([3, 0, 5, 2])

    assert_equal(segments.segment_count(), 4)
    assert_equal(segments.item_count(), 10)
    assert_equal(segments.begin(0), UInt32(0))
    assert_equal(segments.end(0), UInt32(3))
    assert_equal(segments.count(1), UInt32(0))
    assert_equal(segments.begin(2), UInt32(3))
    assert_equal(segments.end(2), UInt32(8))
    assert_equal(segments.count(3), UInt32(2))


def test_segment_offsets_single_is_the_general_case() raises:
    var segments = SegmentOffsets.single(7)

    assert_equal(segments.segment_count(), 1)
    assert_equal(segments.item_count(), 7)
    assert_equal(segments.begin(0), UInt32(0))
    assert_equal(segments.end(0), UInt32(7))


def test_segment_offsets_can_describe_no_segments() raises:
    var segments = SegmentOffsets.from_counts([])

    assert_equal(segments.segment_count(), 0)
    assert_equal(segments.item_count(), 0)


def test_segment_offsets_preserve_the_uint32_limit() raises:
    var segments = SegmentOffsets.from_counts([Int(UInt32.MAX) - 1, 0, 1])
    assert_equal(segments.item_count(), Int(UInt32.MAX))
    assert_equal(segments.count(0), UInt32.MAX - 1)
    assert_equal(segments.count(1), UInt32(0))
    assert_equal(segments.count(2), UInt32(1))


def test_checked_boundary_rejects_mutated_offsets() raises:
    var segments = SegmentOffsets.from_counts([3, 2])
    segments.validate()
    segments.offsets[0] = 1
    with assert_raises():
        segments.validate()
    segments.offsets[0] = 0
    segments.offsets[1] = 6
    with assert_raises():
        segments.validate()
    segments.offsets.clear()
    with assert_raises():
        segments.validate()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
