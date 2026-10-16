"""Independent integer-oracle regression for finite subnormal owner bounds."""
from std.testing import TestSuite, assert_equal, assert_true
from std.memory import bitcast
from bajo.core import AABB, Point3f32
from bajo.bvh.constants import EMPTY_LANE
from bajo.bvh.spatial import SpatialBoundsBvh


@always_inline
def key(bits: UInt32) -> UInt32:
    if (bits & UInt32(0x7FFFFFFF)) == 0:
        return UInt32(0x80000000)
    return bits ^ (
        UInt32(0xFFFFFFFF) if (bits >> 31) != 0 else UInt32(0x80000000)
    )


def check_bits(
    tree: SpatialBoundsBvh[.WORLD], boxes: ImmSpan[AABB[.WORLD], _]
) raises:
    var roots = List[UInt32]()
    if len(boxes) > 0:
        roots.append(0)
    for root in tree.group_roots:
        roots.append(root)
    var previous = 0
    for scope in range(len(roots)):
        var root = Int(roots[scope])
        assert_equal(root, previous)
        var end = Int(tree.nodes[root * 2])
        var seen = List[Bool](length=len(boxes), fill=False)
        for node in range(root, end):
            var escape = Int(tree.nodes[node * 2])
            assert_true(node < escape and escape <= end)
            var id = tree.nodes[node * 2 + 1]
            if id != EMPTY_LANE:
                assert_equal(escape, node + 1)
                assert_true(Int(id) < len(boxes) and not seen[Int(id)])
                seen[Int(id)] = True
            else:
                var right = Int(tree.nodes[(node + 1) * 2])
                assert_true(node + 1 < right and right < escape)
                assert_equal(Int(tree.nodes[right * 2]), escape)
            comptime for axis in range(3):
                var lower = UInt32(0x7F7FFFFF)
                var upper = UInt32(0xFF7FFFFF)
                for descendant in range(node, escape):
                    var face = tree.nodes[descendant * 2 + 1]
                    if face != EMPTY_LANE:
                        var lo = bitcast[.uint32](boxes[Int(face)]._min[axis])
                        var hi = bitcast[.uint32](boxes[Int(face)]._max[axis])
                        if key(lo) < key(lower):
                            lower = lo
                        if key(hi) > key(upper):
                            upper = hi
                assert_equal(
                    key(bitcast[.uint32](tree.bounds[node * 6 + axis])),
                    key(lower),
                )
                assert_equal(
                    key(bitcast[.uint32](tree.bounds[node * 6 + axis + 3])),
                    key(upper),
                )
        for face in range(len(boxes)):
            var wanted = scope == 0
            if scope > 0:
                wanted = tree.groups[face] == tree.group_ids[scope - 1]
            assert_equal(seen[face], wanted)
        previous = end
    assert_equal(previous, len(tree.nodes) // 2)


def test_subnormal_build_refit_rebuild_integer_oracle() raises:
    var sizes: List[Int] = [0, 1, 3, 257, 16383, 16384, 16391, 49153]
    for count in sizes:
        var boxes = List[AABB[.WORLD]]()
        var groups = List[UInt32]()
        for i in range(count):
            var x = Float32((i * 17) % 127 - 63) * 0.25
            var y = Float32((i * 13) % 31 - 15) * 0.125
            var magnitude = UInt32(
                1 if i % 3 == 0 else 71362 if i % 3 == 1 else 0x7FFFFF
            )
            var z = bitcast[.float32](magnitude | (UInt32(i % 2) << 31))
            boxes.append(
                AABB(
                    Point3f32[.WORLD](x, y, z),
                    Point3f32[.WORLD](x + 0.5, y + 0.25, z),
                )
            )
            groups.append(UInt32(i % 3))
        var tree = SpatialBoundsBvh(boxes, groups)
        check_bits(tree, boxes)
        var bounds_ptr = Int(tree.bounds.unsafe_ptr())
        var nodes_ptr = Int(tree.nodes.unsafe_ptr())
        for i in range(count):
            var z = bitcast[.float32](UInt32(17) | (UInt32((i + 1) % 2) << 31))
            boxes[i]._min.z = z
            boxes[i]._max.z = z
        tree.refit(boxes)
        check_bits(tree, boxes)
        tree.rebuild(boxes)
        check_bits(tree, boxes)
        assert_equal(Int(tree.bounds.unsafe_ptr()), bounds_ptr)
        assert_equal(Int(tree.nodes.unsafe_ptr()), nodes_ptr)
        # A normal refit after the precise path must also update all bounds.
        for i in range(count):
            boxes[i]._min.z = Float32(-1.0)
            boxes[i]._max.z = Float32(1.0)
        tree.refit(boxes)
        check_bits(tree, boxes)


def test_inverted_subnormal_rejection_is_atomic() raises:
    var z = bitcast[.float32](UInt32(71362))
    var nz = bitcast[.float32](UInt32(0x800116C2))
    var valid: List[AABB[.WORLD]] = [
        AABB(Point3f32[.WORLD](z), Point3f32[.WORLD](z))
    ]
    var invalid: List[AABB[.WORLD]] = [
        AABB(Point3f32[.WORLD](z), Point3f32[.WORLD](nz))
    ]
    var rejected = False
    try:
        var bad = SpatialBoundsBvh(invalid)
    except:
        rejected = True
    assert_true(rejected)
    var tree = SpatialBoundsBvh(valid)
    var original_bounds = tree.bounds.copy()
    var original_nodes = tree.nodes.copy()
    for operation in range(2):
        rejected = False
        try:
            if operation == 0:
                tree.refit(invalid)
            else:
                tree.rebuild(invalid)
        except:
            rejected = True
        assert_true(rejected)
        for i in range(len(original_bounds)):
            assert_equal(
                bitcast[.uint32](tree.bounds[i]),
                bitcast[.uint32](original_bounds[i]),
            )
        for i in range(len(original_nodes)):
            assert_equal(tree.nodes[i], original_nodes[i])


def test_many_groups_precise_build_and_rebuild_preserve_roots_and_storage() raises:
    for count in [513, 16385]:
        var boxes = List[AABB[.WORLD]](capacity=count)
        var groups = List[UInt32](capacity=count)
        for i in range(count):
            var x = Float32((i * 17) % 127 - 63)
            var z = bitcast[.float32](
                UInt32(1 + i % 71) | (UInt32(i % 2) << 31)
            )
            boxes.append(
                AABB(
                    Point3f32[.WORLD](x, 0.0, z),
                    Point3f32[.WORLD](x + 0.5, 0.25, z),
                )
            )
            groups.append(UInt32.MAX - UInt32(i % 33) * UInt32(65537))
        var tree = SpatialBoundsBvh(boxes, groups)
        check_bits(tree, boxes)
        var roots = tree.group_roots.copy()
        var labels = tree.group_ids.copy()
        var bounds_ptr = Int(tree.bounds.unsafe_ptr())
        var nodes_ptr = Int(tree.nodes.unsafe_ptr())
        for i in range(count):
            boxes[i]._min.x += 0.125
            boxes[i]._max.x += 0.125
            var z = bitcast[.float32](UInt32(17) | (UInt32((i + 1) % 2) << 31))
            boxes[i]._min.z = z
            boxes[i]._max.z = z
        tree.refit(boxes)
        check_bits(tree, boxes)
        tree.rebuild(boxes)
        check_bits(tree, boxes)
        assert_equal(Int(tree.bounds.unsafe_ptr()), bounds_ptr)
        assert_equal(Int(tree.nodes.unsafe_ptr()), nodes_ptr)
        for group in range(len(labels)):
            assert_equal(tree.group_ids[group], labels[group])
            assert_equal(tree.group_roots[group], roots[group])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
