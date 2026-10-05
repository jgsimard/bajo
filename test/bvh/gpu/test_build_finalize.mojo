"""Compact GPU offsets, validation failures, and tiled scan boundaries."""

from max.gpu.host import DeviceContext
from std.sys import has_accelerator
from std.testing import TestSuite, assert_equal, assert_raises

from bajo.bvh.gpu.build_finalize import (
    GpuCompactWideLayout,
    finalize_wide_layout_kernel,
)
from bajo.bvh.gpu.utils import _device_span, upload_list
from bajo.core import SegmentOffsets


def test_compact_offsets_across_tiles_and_empty_segments() raises:
    with DeviceContext() as ctx:
        for count in [1, 2, 31, 32, 33, 255, 256, 257, 513, 4099]:
            var nodes = List[UInt32](capacity=count)
            var leaves = List[UInt32](capacity=count)
            var node_counts = List[Int](capacity=count)
            var leaf_counts = List[Int](capacity=count)
            for index in range(count):
                var n = (index * 7) % 13
                var l = (index * 11) % 17
                nodes.append(UInt32(n))
                leaves.append(UInt32(l))
                node_counts.append(n)
                leaf_counts.append(l)
            var expected_nodes = SegmentOffsets.from_counts(node_counts^)
            var expected_leaves = SegmentOffsets.from_counts(leaf_counts^)
            var layout = GpuCompactWideLayout(
                ctx, upload_list(ctx, nodes), upload_list(ctx, leaves), count
            )
            assert_equal(layout.node_count, expected_nodes.item_count())
            assert_equal(layout.leaf_block_count, expected_leaves.item_count())
            with layout.node_segment_offsets.map_to_host() as offsets:
                for index in range(count + 1):
                    assert_equal(offsets[index], expected_nodes.offsets[index])
            with layout.leaf_block_segment_offsets.map_to_host() as offsets:
                for index in range(count + 1):
                    assert_equal(offsets[index], expected_leaves.offsets[index])


def _validate(
    nodes: List[UInt32],
    leaves: List[UInt32],
    hploc: List[UInt32],
    collapse: List[UInt32],
    encoded: List[UInt32],
    fat_leaves: Bool = True,
) raises:
    with DeviceContext() as ctx:
        var primitives = SegmentOffsets.from_counts([0, 5, 0, 3, 0])
        var capacities = SegmentOffsets.from_counts([0, 4, 0, 2, 0])
        var layout = GpuCompactWideLayout(
            ctx,
            upload_list(ctx, nodes),
            upload_list(ctx, leaves),
            5,
            primitive_offsets=Optional(upload_list(ctx, primitives.offsets)),
            node_capacity_offsets=Optional(
                upload_list(ctx, capacities.offsets)
            ),
            hploc_status=Optional(upload_list(ctx, hploc)),
            collapse_status=Optional(upload_list(ctx, collapse)),
            encoded_counts=Optional(upload_list(ctx, encoded)),
            fat_leaves=fat_leaves,
        )
        assert_equal(layout.node_count, 3)
        assert_equal(layout.leaf_block_count, 5 if fat_leaves else 8)


def test_validated_fat_and_primitive_leaf_counts() raises:
    _validate(
        [0, 2, 0, 1, 0],
        [0, 3, 0, 2, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 5, 0, 3, 0],
    )
    _validate(
        [0, 2, 0, 1, 0],
        [0, 5, 0, 3, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 5, 0, 3, 0],
        False,
    )


def test_finalization_rejects_failed_builds_and_bad_counts() raises:
    for fault in range(8):
        var nodes: List[UInt32] = [0, 2, 0, 1, 0]
        var leaves: List[UInt32] = [0, 3, 0, 2, 0]
        var hploc: List[UInt32] = [0, 0, 0, 0, 0]
        var collapse: List[UInt32] = [0, 0, 0, 0, 0]
        var encoded: List[UInt32] = [0, 5, 0, 3, 0]
        if fault == 0:
            hploc[3] = 2
        elif fault == 1:
            collapse[1] = 4
        elif fault == 2:
            nodes[0] = 1
        elif fault == 3:
            nodes[1] = 0
        elif fault == 4:
            nodes[1] = 5
        elif fault == 5:
            leaves[3] = 0
        elif fault == 6:
            leaves[3] = 4
        else:
            encoded[3] = 2
        with assert_raises():
            _validate(nodes, leaves, hploc, collapse, encoded)
    with assert_raises():
        _validate(
            [0, 2, 0, 1, 0],
            [0, 3, 0, 2, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 5, 0, 3, 0],
            False,
        )


def test_validation_reports_first_failure_across_tiles() raises:
    with DeviceContext() as ctx:
        var counts = List[UInt32](length=513, fill=0)
        var statuses = counts.copy()
        statuses[257] = 4
        statuses[512] = 5
        var input = upload_list(ctx, counts)
        var collapse = upload_list(ctx, statuses)
        var nodes = ctx.enqueue_create_buffer[.uint32](514)
        var leaves = ctx.enqueue_create_buffer[.uint32](514)
        var summary = ctx.enqueue_create_buffer[.uint32](5)
        ctx.enqueue_function[finalize_wide_layout_kernel[256]](
            _device_span[mut=False](input),
            _device_span[mut=False](input),
            _device_span[mut=False](input),
            _device_span[mut=False](input),
            _device_span[mut=False](input),
            _device_span[mut=False](collapse),
            _device_span[mut=False](input),
            _device_span[mut=True](nodes),
            _device_span[mut=True](leaves),
            _device_span[mut=True](summary),
            Int32(513),
            UInt32(4),
            grid_dim=1,
            block_dim=256,
        )
        ctx.synchronize()
        with summary.map_to_host() as result:
            assert_equal(result[2], UInt32(2))
            assert_equal(result[3], UInt32(257))
            assert_equal(result[4], UInt32(4))


def test_compact_offsets_reject_uint32_overflow() raises:
    with DeviceContext() as ctx:
        var large: List[UInt32] = [0xFFFFFFFF, 1]
        var small: List[UInt32] = [1, 1]
        with assert_raises():
            _ = GpuCompactWideLayout(
                ctx,
                upload_list(ctx, large),
                upload_list(ctx, small),
                2,
            )
        with assert_raises():
            _ = GpuCompactWideLayout(
                ctx,
                upload_list(ctx, small),
                upload_list(ctx, large),
                2,
            )


def main() raises:
    comptime if not has_accelerator():
        raise "No Accelerator found"
    TestSuite.discover_tests[__functions_in_module()]().run()
