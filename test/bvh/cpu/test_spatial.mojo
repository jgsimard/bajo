from std.testing import TestSuite, assert_equal, assert_true
from bajo.bvh.spatial import SpatialBoundsBvh
from bajo.core import AABB, Point3f32, Vec3f32, Rayf32


def test_bounds_iterators_and_groups() raises:
    var boxes: List[AABB[.WORLD]] = [
        AABB(Point3f32[.WORLD](0.0), Point3f32[.WORLD](1.0)),
        AABB(Point3f32[.WORLD](2.0), Point3f32[.WORLD](3.0)),
        AABB(Point3f32[.WORLD](0.5), Point3f32[.WORLD](1.5)),
    ]
    var groups: List[UInt32] = [7, 8, 8]
    var tree = SpatialBoundsBvh(boxes, groups)
    var view = tree.view()
    var query = view.query_aabb(
        AABB(Point3f32[.WORLD](1.0), Point3f32[.WORLD](2.0))
    )
    var found = List[UInt32]()
    var index = UInt32(99)
    while query.next(index):
        found.append(index)
    sort(found)
    assert_equal(len(found), 3)
    assert_equal(found[0], UInt32(0))
    assert_equal(found[1], UInt32(1))
    assert_equal(found[2], UInt32(2))
    var previous = index
    assert_true(not query.next(index))
    assert_equal(index, previous)
    query = view.query_sphere(Point3f32[.WORLD](1.0), 0.0, tree.group_root(7))
    assert_true(query.next(index))
    assert_equal(index, UInt32(0))
    assert_true(not query.next(index))
    query = view.query_sphere(Point3f32[.WORLD](0.0), -1.0)
    assert_true(not query.next(index))
    query = view.query_aabb(boxes[0], tree.group_root(999))
    assert_true(not query.next(index))


def test_parallel_rays_cutoffs_and_capsules() raises:
    var boxes: List[AABB[.WORLD]] = [
        AABB(
            Point3f32[.WORLD](0.0, 0.0, 2.0), Point3f32[.WORLD](1.0, 1.0, 3.0)
        ),
        AABB(
            Point3f32[.WORLD](0.0, 0.0, 5.0), Point3f32[.WORLD](1.0, 1.0, 6.0)
        ),
    ]
    var tree = SpatialBoundsBvh(boxes)
    var view = tree.view()
    var query = view.query_ray(
        Rayf32[.WORLD](Point3f32[.WORLD](0.0), Vec3f32[.WORLD](0.0, 0.0, 1.0))
    )
    var index = UInt32(99)
    assert_true(query.next(index, 3.0))
    assert_equal(index, UInt32(0))
    assert_true(not query.next(index, 100.0))
    query = view.query_ray(
        Rayf32[.WORLD](
            Point3f32[.WORLD](2.0, 0.0, 0.0), Vec3f32[.WORLD](0.0, 0.0, 1.0)
        )
    )
    assert_true(not query.next(index))
    query = view.query_capsule(
        Point3f32[.WORLD](-1.0, 0.0, 0.0), Vec3f32[.WORLD](0.0, 0.0, 3.0), 1.0
    )
    assert_true(query.next(index))
    assert_equal(index, UInt32(0))
    assert_true(not query.next(index))


def test_empty_and_invalid_bounds() raises:
    var empty = SpatialBoundsBvh[.WORLD](List[AABB[.WORLD]]())
    var query = empty.view().query_sphere(Point3f32[.WORLD](0.0), 1.0)
    var index = UInt32(42)
    assert_true(not query.next(index))
    assert_equal(index, UInt32(42))
    var rejected = False
    try:
        var bad = SpatialBoundsBvh[.WORLD]([AABB[.WORLD].invalid()])
    except:
        rejected = True
    assert_true(rejected)


def test_refit_rebuild_and_failed_update_preserve_buffers() raises:
    var boxes: List[AABB[.WORLD]] = [
        AABB(Point3f32[.WORLD](0.0), Point3f32[.WORLD](1.0)),
        AABB(Point3f32[.WORLD](4.0), Point3f32[.WORLD](5.0)),
    ]
    var groups: List[UInt32] = [3, 4]
    var tree = SpatialBoundsBvh(boxes, groups)
    var bounds_address = Int(tree.bounds.unsafe_ptr())
    var nodes_address = Int(tree.nodes.unsafe_ptr())
    var original_nodes = tree.nodes.copy()
    boxes[0] = AABB(Point3f32[.WORLD](10.0), Point3f32[.WORLD](11.0))
    tree.refit(boxes)
    assert_equal(Int(tree.bounds.unsafe_ptr()), bounds_address)
    assert_equal(Int(tree.nodes.unsafe_ptr()), nodes_address)
    for i in range(len(original_nodes)):
        assert_equal(tree.nodes[i], original_nodes[i])
    var query = tree.view().query_sphere(
        Point3f32[.WORLD](10.5), 0.0, tree.group_root(3)
    )
    var index = UInt32(99)
    assert_true(query.next(index))
    assert_equal(index, UInt32(0))
    assert_true(not query.next(index))
    tree.rebuild(boxes)
    assert_equal(Int(tree.bounds.unsafe_ptr()), bounds_address)
    assert_equal(Int(tree.nodes.unsafe_ptr()), nodes_address)
    query = tree.view().query_sphere(Point3f32[.WORLD](4.5), 0.0)
    assert_true(query.next(index))
    assert_equal(index, UInt32(1))
    var saved = tree.bounds.copy()
    boxes[1] = AABB[.WORLD].invalid()
    var rejected = False
    try:
        tree.refit(boxes)
    except:
        rejected = True
    assert_true(rejected)
    for i in range(len(saved)):
        assert_equal(tree.bounds[i], saved[i])


def test_hierarchy_matches_exhaustive_overlap_sets() raises:
    var boxes = List[AABB[.WORLD]]()
    for i in range(127):
        var p = Point3f32[.WORLD](
            Float32(i % 11), Float32(i % 7), Float32(i % 5)
        )
        boxes.append(AABB(p, p + Vec3f32[.WORLD](0.25)))
    var tree = SpatialBoundsBvh(boxes)
    for x in range(13):
        var region = AABB(
            Point3f32[.WORLD](Float32(x) - 0.1, 1.0, 0.0),
            Point3f32[.WORLD](Float32(x) + 1.0, 4.0, 2.0),
        )
        var actual = List[UInt32]()
        var expected = List[UInt32]()
        var query = tree.view().query_aabb(region)
        var index = UInt32(0)
        while query.next(index):
            actual.append(index)
        for i in range(len(boxes)):
            if boxes[i].overlaps(region)[0]:
                expected.append(UInt32(i))
        sort(actual)
        assert_equal(len(actual), len(expected))
        for i in range(len(expected)):
            assert_equal(actual[i], expected[i])


def test_sphere_corner_and_zero_length_capsule_semantics() raises:
    var tree = SpatialBoundsBvh[.WORLD](
        [AABB(Point3f32[.WORLD](0.0), Point3f32[.WORLD](1.0))]
    )
    var view = tree.view()
    var index = UInt32(99)
    # Expanding the box in each axis would falsely accept this sphere corner.
    var query = view.query_sphere(Point3f32[.WORLD](2.0, 2.0, 0.5), 1.0)
    assert_true(not query.next(index))
    query = view.query_sphere(Point3f32[.WORLD](2.0, 1.0, 0.5), 1.0)
    assert_true(query.next(index))
    assert_equal(index, UInt32(0))
    # Capsule queries deliberately return conservative broad-phase candidates.
    query = view.query_capsule(
        Point3f32[.WORLD](2.0, 2.0, 0.5), Vec3f32[.WORLD](0.0), 1.0
    )
    assert_true(query.next(index))
    query = view.query_ray(
        Rayf32[.WORLD](Point3f32[.WORLD](0.5), Vec3f32[.WORLD](0.0))
    )
    assert_true(query.next(index))
    query = view.query_ray(
        Rayf32[.WORLD](Point3f32[.WORLD](2.0), Vec3f32[.WORLD](0.0))
    )
    assert_true(not query.next(index))


def test_group_mismatch_and_update_count_fail_atomically() raises:
    var boxes: List[AABB[.WORLD]] = [
        AABB(Point3f32[.WORLD](0.0), Point3f32[.WORLD](1.0))
    ]
    var rejected = False
    try:
        var groups: List[UInt32] = [1, 2]
        var bad = SpatialBoundsBvh(boxes, groups)
    except:
        rejected = True
    assert_true(rejected)
    var tree = SpatialBoundsBvh(boxes)
    rejected = False
    try:
        tree.rebuild(List[AABB[.WORLD]]())
    except:
        rejected = True
    assert_true(rejected)
    var query = tree.view().query_sphere(Point3f32[.WORLD](0.0), 0.0)
    var index = UInt32(99)
    assert_true(query.next(index))
    assert_equal(index, UInt32(0))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
