from std.testing import TestSuite, assert_equal
from bajo.core import AABB, Point3f32, Vec3f32, Rayf32
from bajo.bvh.spatial import (
    SpatialBoundsBvh,
    BoundsQueryView,
    BoundsQuery,
)
from test.bvh.spatial_fixtures import tetrahedron


def test_serial_batch_capacity_partial_exhaustion_and_zero_capacity() raises:
    var boxes = List[AABB[.WORLD]]()
    var groups = List[UInt32]()
    for i in range(11):
        boxes.append(
            AABB(
                Point3f32[.WORLD](Float32(i) * 0.1),
                Point3f32[.WORLD](Float32(i) * 0.1 + 0.1),
            )
        )
        groups.append(UInt32(i % 2))
    var tree = SpatialBoundsBvh(boxes, groups)
    var batch = tree.view().query_aabb(
        AABB(Point3f32[.WORLD](0.0), Point3f32[.WORLD](2.0))
    )
    var zero = List[UInt32]()
    assert_equal(batch.next_batch(zero), 0)
    var output = List[UInt32](length=4, fill=99)
    var found = List[UInt32]()
    assert_equal(batch.next_batch(output), 4)
    for item in output:
        found.append(item)
    assert_equal(batch.next_batch(output), 4)
    for item in output:
        found.append(item)
    output[3] = 99
    assert_equal(batch.next_batch(output), 3)
    assert_equal(output[3], UInt32(99))
    for i in range(3):
        found.append(output[i])
    sort(found)
    for i in range(11):
        assert_equal(found[i], UInt32(i))
    var previous = output.copy()
    assert_equal(batch.next_batch(output), 0)
    for i in range(4):
        assert_equal(output[i], previous[i])
    batch = tree.view().query_sphere(
        Point3f32[.WORLD](0.5), 1.0, tree.group_root(1)
    )
    var count = 0
    while True:
        var filled = batch.next_batch(output)
        if filled == 0:
            break
        for i in range(filled):
            assert_equal(output[i] % 2, UInt32(1))
        count += filled
    assert_equal(count, 5)


def test_serial_batch_ray_cutoff_and_empty_tree() raises:
    var boxes: List[AABB[.WORLD]] = [
        AABB(
            Point3f32[.WORLD](0.0, 0.0, 1.0), Point3f32[.WORLD](1.0, 1.0, 2.0)
        ),
        AABB(
            Point3f32[.WORLD](0.0, 0.0, 3.0), Point3f32[.WORLD](1.0, 1.0, 4.0)
        ),
    ]
    var tree = SpatialBoundsBvh(boxes)
    var batch = tree.view().query_ray(
        Rayf32[.WORLD](
            Point3f32[.WORLD](0.5, 0.5, 0.0), Vec3f32[.WORLD](0.0, 0.0, 1.0)
        )
    )
    var output = List[UInt32](length=4, fill=99)
    assert_equal(batch.next_batch(output, 2.0), 1)
    assert_equal(output[0], UInt32(0))
    assert_equal(batch.next_batch(output, 10.0), 0)
    var empty = SpatialBoundsBvh[.WORLD](List[AABB[.WORLD]]())
    var empty_batch = empty.view().query_sphere(Point3f32[.WORLD](0.0), 1.0)
    assert_equal(empty_batch.next_batch(output), 0)


def test_mesh_aabb_batch_keeps_original_face_ids_and_groups() raises:
    var mesh = tetrahedron()
    var box = AABB(Point3f32[.WORLD](0.0), Point3f32[.WORLD](1.0))
    var roots: List[UInt32] = [0, mesh.group_root(7), mesh.group_root(999)]
    for r in range(3):
        var batch = mesh.view().query_aabb(box, roots[r])
        var output = List[UInt32](length=3, fill=99)
        var counts = List[Int](length=4, fill=0)
        while True:
            var count = batch.next_batch(output)
            if count == 0:
                break
            for i in range(count):
                counts[Int(output[i])] += 1
            for i in range(count, 3):
                assert_equal(output[i], UInt32(99))
            for i in range(3):
                output[i] = 99
        for face in range(4):
            assert_equal(
                counts[face], 1 if r == 0 or (r == 1 and face < 2) else 0
            )


def make_query[
    origin: ImmOrigin
](
    view: BoundsQueryView[origin, .WORLD], kind: Int, root: UInt32
) -> BoundsQuery[origin, .WORLD]:
    var p = Point3f32[.WORLD](0.23, 0.17, -0.25)
    var ray = Rayf32[.WORLD](
        p, Vec3f32[.WORLD](0.0, 0.0, 1.0), t_min=-1.0, t_max=4.0
    )
    var query = view.query_aabb(
        AABB(p - Vec3f32[.WORLD](0.4), p + Vec3f32[.WORLD](0.4)), root
    )
    if kind == 1:
        query = view.query_ray(ray, root)
    elif kind == 2:
        query = view.query_sphere(p, 0.4, root)
    elif kind == 3:
        query = view.query_capsule(p, ray.d, 0.2, root)
    return query^


def check_packet_batches[
    width: SIMDLength
](tree: SpatialBoundsBvh[.WORLD]) raises:
    var capacities: List[Int] = [0, 1, 3, 8, 17, 64]
    var roots: List[UInt32] = [
        0,
        tree.group_root(0),
        tree.group_root(1),
        tree.group_root(2),
        tree.group_root(99),
    ]
    var view = tree.view()
    for kind in range(4):
        for root in roots:
            for capacity in capacities:
                var scalar = make_query(view, kind, root)
                var batch = scalar.copy()
                var expected = List[Int](length=len(tree.item_bounds), fill=0)
                var actual = expected.copy()
                var item = UInt32(0)
                while scalar.next(item, 1.5):
                    expected[Int(item)] += 1
                var output = List[UInt32](length=capacity, fill=99)
                if capacity == 0:
                    var cursor = batch.cursor
                    assert_equal(batch.next_batch[width](output, 0.1), 0)
                    assert_equal(batch.cursor, cursor)
                    output.append(UInt32(99))
                while True:
                    var count = batch.next_batch[width](output, 1.5)
                    for i in range(count):
                        actual[Int(output[i])] += 1
                    for i in range(count, len(output)):
                        assert_equal(output[i], UInt32(99))
                    for i in range(len(output)):
                        output[i] = 99
                    if count == 0:
                        break
                for i in range(len(actual)):
                    assert_equal(actual[i], expected[i])


def test_packet_batches_match_scalar_for_widths_groups_tails_and_updates() raises:
    var boxes = List[AABB[.WORLD]]()
    var groups = List[UInt32]()
    for i in range(37):
        var p = Point3f32[.WORLD](
            Float32(i % 5) * 0.11,
            Float32((i * 7) % 5) * 0.07,
            Float32(i % 11) * 0.09 - 0.4,
        )
        boxes.append(AABB(p, p + Vec3f32[.WORLD](0.08)))
        groups.append(UInt32(i % 3))
    var tree = SpatialBoundsBvh(boxes, groups)
    for phase in range(3):
        check_packet_batches[1](tree)
        check_packet_batches[2](tree)
        check_packet_batches[4](tree)
        check_packet_batches[8](tree)
        check_packet_batches[16](tree)
        for i in range(len(boxes)):
            boxes[i] = AABB(
                boxes[i]._min + Vec3f32[.WORLD](0.03),
                boxes[i]._max + Vec3f32[.WORLD](0.03),
            )
        if phase == 0:
            tree.refit(boxes)
        elif phase == 1:
            tree.rebuild(boxes)


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
