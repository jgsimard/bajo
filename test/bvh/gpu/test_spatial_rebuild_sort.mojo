"""Parallel rebuild ordering with padded groups, ties and multiple lane waves."""

from std.testing import TestSuite, assert_equal, assert_true
from std.math import min, max
from max.gpu.host import DeviceContext
from bajo.core import AABB, Point3f32
from bajo.bvh.constants import EMPTY_LANE
from bajo.bvh.spatial import SpatialBoundsBvh
from bajo.bvh.spatial.gpu import GpuSpatialBoundsBvh
from bajo.bvh.gpu.utils import upload_list


def _check_sort[block_size: Int]() raises:
    with DeviceContext() as ctx:
        var sizes: List[Int] = [
            1,
            2,
            4,
            8,
            16,
            17,
            48,
            192,
            256,
            257,
            512,
            768,
            1024,
            1025,
            2048,
            3072,
        ]
        for count in sizes:
            var boxes = List[AABB[.WORLD]]()
            var groups = List[UInt32]()
            var items = List[Float32]()
            for i in range(count):
                # Reverse-order input, duplicate centroids and large coordinates.
                var x = Float32((count - i) % 11) * 1e35
                boxes.append(
                    AABB(Point3f32[.WORLD](x), Point3f32[.WORLD](x + 1e34))
                )
                groups.append(UInt32(i % 3))
                comptime for axis in range(3):
                    items.append(x)
                comptime for axis in range(3):
                    items.append(x + 1e34)
            var cpu = SpatialBoundsBvh(boxes, groups)
            var gpu = GpuSpatialBoundsBvh(ctx, cpu.copy())
            var input = upload_list(ctx, items)
            var bp = Int(gpu.bounds.unsafe_ptr())
            var np = Int(gpu.nodes.unsafe_ptr())
            # Change device centroids after the first sort, then replay unchanged.
            for phase in range(3):
                if phase == 1:
                    with input.map_to_host() as data:
                        for i in range(count):
                            var x = -Float32((i * 17) % 13) * 1e35
                            comptime for axis in range(3):
                                items[i * 6 + axis] = x
                                items[i * 6 + axis + 3] = x + 1e34
                                data[i * 6 + axis] = x
                                data[i * 6 + axis + 3] = x + 1e34
                gpu.enqueue_update[True, block_size](ctx, input)
                gpu.check_update(ctx)
                assert_equal(Int(gpu.bounds.unsafe_ptr()), bp)
                assert_equal(Int(gpu.nodes.unsafe_ptr()), np)
                with gpu.nodes.map_to_host() as nodes:
                    with gpu.bounds.map_to_host() as bounds:
                        var root = 0
                        var seen = List[UInt32](length=count, fill=0)
                        while root < len(nodes) // 2:
                            var end = Int(nodes[root * 2])
                            var previous = EMPTY_LANE
                            for node in range(root, end):
                                var item = nodes[node * 2 + 1]
                                if item == EMPTY_LANE:
                                    continue
                                seen[Int(item)] += 1
                                if previous != EMPTY_LANE:
                                    if root != 0:
                                        assert_equal(
                                            groups[Int(item)],
                                            groups[Int(previous)],
                                        )
                                    var a = Float64(
                                        items[Int(previous) * 6]
                                    ) + Float64(items[Int(previous) * 6 + 3])
                                    var b = Float64(
                                        items[Int(item) * 6]
                                    ) + Float64(items[Int(item) * 6 + 3])
                                    assert_true(
                                        a < b or (a == b and previous < item)
                                    )
                                previous = item
                            root = end
                        for times in seen:
                            assert_equal(times, UInt32(2))
                        # Independently reduce descendant leaf bounds at every node.
                        for node in range(len(nodes) // 2):
                            var end = Int(nodes[node * 2])
                            var lo = Point3f32[.WORLD](3e38)
                            var hi = Point3f32[.WORLD](-3e38)
                            for child in range(node, end):
                                var item = nodes[child * 2 + 1]
                                if item != EMPTY_LANE:
                                    comptime for axis in range(3):
                                        lo[axis] = min(
                                            lo[axis],
                                            items[Int(item) * 6 + axis],
                                        )
                                        hi[axis] = max(
                                            hi[axis],
                                            items[Int(item) * 6 + axis + 3],
                                        )
                            comptime for axis in range(3):
                                assert_equal(bounds[node * 6 + axis], lo[axis])
                                assert_equal(
                                    bounds[node * 6 + axis + 3], hi[axis]
                                )


def test_parallel_rebuild_padded_groups_ties_and_bounds() raises:
    _check_sort[32]()
    _check_sort[256]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
