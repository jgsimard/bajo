"""Exhaustive subtree geometry and membership across construction strategies."""
from std.testing import TestSuite, assert_equal, assert_true
from std.math import min, max
from bajo.core import AABB, Point3f32
from bajo.bvh.spatial import SpatialBoundsBvh
from bajo.bvh.constants import EMPTY_LANE, f32_max


def check_forest(
    tree: SpatialBoundsBvh[.WORLD], boxes: ImmSpan[AABB[.WORLD], _]
) raises:
    var total_nodes = len(tree.nodes) // 2
    assert_equal(len(tree.bounds), total_nodes * 6)
    if len(boxes) == 0:
        assert_equal(total_nodes, 0)
        return
    var roots = List[UInt32]()
    roots.append(0)
    for root in tree.group_roots:
        roots.append(root)
    var previous_end = 0
    for scope in range(len(roots)):
        var root = Int(roots[scope])
        assert_equal(root, previous_end)
        var end = Int(tree.nodes[root * 2])
        assert_true(root < end and end <= total_nodes)
        var seen = List[Bool](length=len(boxes), fill=False)
        var leaves = 0
        for node in range(root, end):
            var escape = Int(tree.nodes[node * 2])
            assert_true(node < escape and escape <= end)
            var payload = tree.nodes[node * 2 + 1]
            if payload != EMPTY_LANE:
                assert_equal(escape, node + 1)
                assert_true(Int(payload) < len(boxes))
                assert_true(not seen[Int(payload)])
                seen[Int(payload)] = True
                leaves += 1
            else:
                var left = node + 1
                var right = Int(tree.nodes[left * 2])
                assert_true(left < right and right < escape)
                assert_equal(Int(tree.nodes[right * 2]), escape)
            # Independently scan every descendant primitive rather than
            # reproducing either builder's range scan or child-bound merge.
            comptime for axis in range(3):
                var lo = Float32(f32_max)
                var hi = -lo
                var found = False
                for descendant in range(node, escape):
                    var face = tree.nodes[descendant * 2 + 1]
                    if face != EMPTY_LANE:
                        found = True
                        lo = min(lo, boxes[Int(face)]._min[axis])
                        hi = max(hi, boxes[Int(face)]._max[axis])
                assert_true(found)
                assert_equal(tree.bounds[node * 6 + axis], lo)
                assert_equal(tree.bounds[node * 6 + axis + 3], hi)
        for face in range(len(boxes)):
            var wanted = scope == 0
            if scope > 0:
                wanted = tree.groups[face] == tree.group_ids[scope - 1]
            assert_equal(seen[face], wanted)
        assert_equal(end - root, leaves * 2 - 1)
        previous_end = end
    assert_equal(previous_end, total_nodes)


def test_construction_thresholds_uneven_groups_and_collisions() raises:
    var sizes: List[Int] = [
        0,
        1,
        2,
        3,
        257,
        4097,
        16383,
        16384,
        16385,
        40957,
        40960,
        40961,
    ]
    var labels: List[UInt32] = [19, 4, 19, 100, 7]
    for count in sizes:
        var boxes = List[AABB[.WORLD]]()
        var groups = List[UInt32]()
        for face in range(count):
            var x = Float32((face * 17) % 31) * 0.25
            var y = Float32((face * 13) % 7) * 0.125
            var z = Float32((face * 19) % 3)
            if face % 11 == 0:
                x = 0.0
                y = 0.0
                z = 0.0
            boxes.append(
                AABB(
                    Point3f32[.WORLD](x, y, z),
                    Point3f32[.WORLD](x + 0.5, y + 0.25, z + 0.125),
                )
            )
            groups.append(labels[face % len(labels)])
        var global_tree = SpatialBoundsBvh(boxes)
        check_forest(global_tree, boxes)
        var grouped_tree = SpatialBoundsBvh(boxes, groups)
        check_forest(grouped_tree, boxes)


def test_construction_zero_extents_and_overflowing_float32_centroids() raises:
    for kind in range(2):
        var boxes = List[AABB[.WORLD]]()
        for _ in range(16385):
            boxes.append(
                AABB(
                    Point3f32[.WORLD](Float32(0.0 if kind == 0 else 2.0e38)),
                    Point3f32[.WORLD](Float32(0.0 if kind == 0 else 3.0e38)),
                )
            )
        var tree = SpatialBoundsBvh(boxes)
        check_forest(tree, boxes)


def _check_parallel_group_emission_thresholds_and_refit(
    group_count: Int,
) raises:
    var boxes = List[AABB[.WORLD]]()
    var groups = List[UInt32]()
    var labels: List[UInt32] = [97, 3, 200, 17, 19, 21, 23, 25, UInt32.MAX]
    var sizes: List[Int] = [16383, 16384, 16385, 1, 2, 3, 4, 5, 6]
    for group in range(group_count):
        for item in range(sizes[group]):
            var x = Float32((item * 17) % 127) * 0.25
            var y = Float32((item * 13) % 31) * 0.125
            var z = Float32((item * 19) % 7) + Float32(group) * 0.5
            boxes.append(
                AABB(
                    Point3f32[.WORLD](x, y, z),
                    Point3f32[.WORLD](x + 0.5, y + 0.25, z + 0.125),
                )
            )
            groups.append(labels[group])
    var tree = SpatialBoundsBvh(boxes, groups)
    check_forest(tree, boxes)
    for i in range(len(boxes)):
        boxes[i] = AABB(
            Point3f32[.WORLD](
                boxes[i]._min.x + 0.125,
                boxes[i]._min.y - 0.25,
                boxes[i]._min.z + 0.5,
            ),
            Point3f32[.WORLD](
                boxes[i]._max.x + 0.125,
                boxes[i]._max.y - 0.25,
                boxes[i]._max.z + 0.5,
            ),
        )
    tree.refit(boxes)
    check_forest(tree, boxes)


def test_parallel_group_emission_thresholds_and_refit() raises:
    _check_parallel_group_emission_thresholds_and_refit(3)
    _check_parallel_group_emission_thresholds_and_refit(9)


def test_morton_packet_tails_negative_coordinates() raises:
    var counts: List[Int] = [16391, 16393]
    for count in counts:
        var boxes = List[AABB[.WORLD]]()
        var groups = List[UInt32]()
        for i in range(count):
            var x = Float32((i * 17) % 127 - 63) * 0.25
            var y = Float32((i * 13) % 31 - 15) * 0.125
            var z = Float32((i * 7) % 11 - 5) * 0.125
            boxes.append(
                AABB(
                    Point3f32[.WORLD](x, y, z),
                    Point3f32[.WORLD](x + 0.5, y + 0.25, z),
                )
            )
            groups.append(UInt32((i * 19) % 5))
        var tree = SpatialBoundsBvh(boxes)
        check_forest(tree, boxes)
        var grouped = SpatialBoundsBvh(boxes, groups)
        check_forest(grouped, boxes)


def test_many_sparse_groups_preserve_membership_and_morton_order() raises:
    for count, group_count in [
        (17, 8),
        (17, 9),
        (257, 257),
        (4097, 65),
        (16383, 65),
        (16384, 65),
        (16385, 65),
    ]:
        var boxes = List[AABB[.WORLD]](capacity=count)
        var groups = List[UInt32](capacity=count)
        var labels = List[UInt32](capacity=group_count)
        for group in range(group_count):
            # Include UInt32.MAX and non-dense, unsorted labels. Their order
            # must follow first appearance rather than a directory's iteration.
            labels.append(UInt32.MAX - UInt32(group) * UInt32(2654435761))
        for face in range(count):
            var x = Float32((face * 17) % 19)
            var y = Float32((face * 13) % 7)
            boxes.append(
                AABB(
                    Point3f32[.WORLD](x, y, 0.0),
                    Point3f32[.WORLD](x + 0.5, y + 0.25, 0.0),
                )
            )
            groups.append(labels[face % group_count])
        var tree = SpatialBoundsBvh(boxes, groups)
        assert_equal(len(tree.group_ids), group_count)
        for group in range(group_count):
            assert_equal(tree.group_ids[group], labels[group])
            assert_equal(
                tree.group_root(labels[group]), tree.group_roots[group]
            )
        check_forest(tree, boxes)
        if count >= 16384:
            # Morton collisions keep the global stable leaf order inside
            # every group. Verify this observable traversal order directly.
            var global_end = Int(tree.nodes[0])
            for group in range(group_count):
                var cursor = Int(tree.group_roots[group])
                var end = Int(tree.nodes[cursor * 2])
                for node in range(global_end):
                    var id = tree.nodes[node * 2 + 1]
                    if id == EMPTY_LANE or groups[Int(id)] != labels[group]:
                        continue
                    while (
                        cursor < end
                        and tree.nodes[cursor * 2 + 1] == EMPTY_LANE
                    ):
                        cursor += 1
                    assert_true(cursor < end)
                    assert_equal(tree.nodes[cursor * 2 + 1], id)
                    cursor += 1
                assert_equal(cursor, end)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
