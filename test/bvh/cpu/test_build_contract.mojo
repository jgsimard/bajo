"""Build scheduling preserves SAH ties, partition order, and H-PLOC payloads."""

from std.testing import TestSuite, assert_equal, assert_true
from bajo.core import AABB, Point3W, Vec3W
from bajo.bvh.cpu.builder import BinaryBoundsBvh, BoundsItem
from bajo.bvh.cpu.builder.types import BoundsBvhNode
from bajo.bvh.cpu.builder.sah import (
    _find_sah_split,
    _find_sah_split_parallel,
    _partition_items_by_bin_parallel,
)


def test_parallel_sah_preserves_split_ties_and_stable_partition() raises:
    var items = List[BoundsItem[.WORLD]]()
    var centroids = AABB[.WORLD].invalid()
    for i in range(37):
        var x = Float32((i * 7) % 13)
        var point = Point3W(x, x, x)
        items.append(
            BoundsItem(
                AABB[.WORLD](point, point + Vec3W(0.5, 0.5, 0.5)), UInt32(i)
            )
        )
        centroids.grow(items[i].centroid)
    var indices = [UInt32(i) for i in range(len(items))]
    var node = BoundsBvhNode[.WORLD]()
    node.set_leaf(0, UInt32(len(items)))
    var expected = _find_sah_split[.WORLD, 16](node, centroids, indices, items)
    assert_equal(expected.axis, 0)
    for workers in [1, 3, 8, 41]:
        var actual = _find_sah_split_parallel[.WORLD, 16](
            node, centroids, indices, items, workers
        )
        assert_equal(actual.axis, expected.axis)
        assert_equal(actual.bin, expected.bin)
        assert_equal(actual.cost, expected.cost)
        assert_equal(actual.pos, expected.pos)

    # Include untouched prefix/suffix entries and workers with empty chunks.
    for split_bin in [-1, 5, 16]:
        var left = List[UInt32]()
        var right = List[UInt32]()
        for i in range(2, 35):
            if items[i].centroid.x < Float32(split_bin + 1):
                left.append(UInt32(i))
            else:
                right.append(UInt32(i))
        for workers in [1, 3, 8, 41]:
            var partitioned = indices.copy()
            var result = _partition_items_by_bin_parallel[.WORLD, 16](
                partitioned, items, 2, 33, 0, split_bin, 0.0, 1.0, workers
            )
            assert_equal(result.split_idx, 2 + len(left))
            assert_equal(partitioned[0], UInt32(0))
            assert_equal(partitioned[1], UInt32(1))
            assert_equal(partitioned[35], UInt32(35))
            assert_equal(partitioned[36], UInt32(36))
            for i in range(len(left)):
                assert_equal(partitioned[2 + i], left[i])
            for i in range(len(right)):
                assert_equal(partitioned[2 + len(left) + i], right[i])


def test_parallel_hploc_microleaves_preserve_every_payload() raises:
    var items = List[BoundsItem[.WORLD]]()
    for i in range(17001):
        var point = Point3W(Float32(i % 257), Float32(i // 257), 0.0)
        items.append(
            BoundsItem(
                AABB[.WORLD](point, point + Vec3W(0.5, 0.5, 0.5)), UInt32(i)
            )
        )
    comptime for microleaf_size in [1, 4]:
        var tree = BinaryBoundsBvh[.WORLD, 16, .HPLOC, microleaf_size](
            items.copy()
        )
        var seen = List[Int](length=len(items), fill=0)
        for node in tree.nodes:
            if node.is_leaf():
                assert_true(node.item_count <= 16)
                for offset in range(Int(node.item_count)):
                    var item = Int(
                        tree.item_indices[Int(node.first_item()) + offset]
                    )
                    seen[item] += 1
        for count in seen:
            assert_equal(count, 1)


def test_parallel_sah_mixed_subtrees_preserve_every_payload() raises:
    var count = 65536
    var items = List[BoundsItem[.WORLD]](capacity=count + 128)
    for i in range(count + 128):
        var x = Float32(i % 512) * 4.0 if i < count else 1000000.0
        var y = (
            Float32(i // 512) * 4.0 if i < count else Float32(i - count) * 4.0
        )
        var point = Point3W(x, y, 0.0)
        items.append(
            BoundsItem(AABB[.WORLD](point, point + Vec3W(0.5)), UInt32(i))
        )
    var tree = BinaryBoundsBvh[.WORLD, 16, .SAH](items.copy())
    var seen = List[Int](length=len(items), fill=0)
    for node in tree.nodes:
        if node.is_leaf():
            assert_true(node.item_count <= 16)
            for offset in range(Int(node.item_count)):
                var item = Int(
                    tree.item_indices[Int(node.first_item()) + offset]
                )
                seen[item] += 1
                assert_true(items[item].bounds._min.x >= node.aabb._min.x)
                assert_true(items[item].bounds._min.y >= node.aabb._min.y)
                assert_true(items[item].bounds._max.x <= node.aabb._max.x)
                assert_true(items[item].bounds._max.y <= node.aabb._max.y)
    for occurrences in seen:
        assert_equal(occurrences, 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
