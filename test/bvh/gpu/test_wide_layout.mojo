"""Segment compaction preserves fields across empty segments and block edges."""

from std.sys import has_accelerator
from std.testing import TestSuite, assert_equal
from max.gpu.host import DeviceContext
from bajo.core import SegmentOffsets
from bajo.bvh.gpu.utils import upload_list
from bajo.bvh.gpu.wide_layout import enqueue_compact_segmented_buffer


def _check_compaction[stride: Int]() raises:
    var capacities = SegmentOffsets.from_counts([3, 0, 5, 2, 1, 2049])
    var counts = SegmentOffsets.from_counts([1, 0, 3, 1, 0, 2047])
    var values = [Float32(i) for i in range(capacities.item_count() * stride)]
    with DeviceContext() as ctx:
        var source = upload_list(ctx, values)
        var source_offsets = upload_list(ctx, capacities.offsets)
        var target_offsets = upload_list(ctx, counts.offsets)
        var target = enqueue_compact_segmented_buffer[.float32, stride](
            ctx,
            source,
            source_offsets,
            target_offsets,
            counts.item_count(),
            counts.segment_count(),
        )
        with target.map_to_host() as result:
            assert_equal(len(result), counts.item_count() * stride)
            for segment in range(counts.segment_count()):
                for item in range(Int(counts.count(segment))):
                    for field in range(stride):
                        var target_idx = (
                            Int(counts.offsets[segment]) + item
                        ) * stride + field
                        var source_idx = (
                            Int(capacities.offsets[segment]) + item
                        ) * stride + field
                        assert_equal(result[target_idx], values[source_idx])


def test_compact_segmented_fields() raises:
    comptime for stride in [1, 7, 112, 192, 253, 640]:
        _check_compaction[stride]()


def main() raises:
    comptime if not has_accelerator():
        raise "No Accelerator found"
    TestSuite.discover_tests[__functions_in_module()]().run()
