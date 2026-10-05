from std.testing import TestSuite, assert_true, assert_equal
from bajo.core import AABB, Point3f32, Vec3f32, Rayf32
from bajo.bvh.spatial import SpatialBoundsBvh, BoundsQueryBatch
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
    var batch = BoundsQueryBatch(
        tree.view().query_aabb(
            AABB(Point3f32[.WORLD](0.0), Point3f32[.WORLD](2.0))
        )
    )
    var zero = List[UInt32]()
    assert_equal(batch.next(zero), 0)
    var output = List[UInt32](length=4, fill=99)
    var found = List[UInt32]()
    assert_equal(batch.next(output), 4)
    for item in output:
        found.append(item)
    assert_equal(batch.next(output), 4)
    for item in output:
        found.append(item)
    output[3] = 99
    assert_equal(batch.next(output), 3)
    assert_equal(output[3], UInt32(99))
    for i in range(3):
        found.append(output[i])
    sort(found)
    for i in range(11):
        assert_equal(found[i], UInt32(i))
    var previous = output.copy()
    assert_equal(batch.next(output), 0)
    for i in range(4):
        assert_equal(output[i], previous[i])
    batch = BoundsQueryBatch(
        tree.view().query_sphere(
            Point3f32[.WORLD](0.5), 1.0, tree.group_root(1)
        )
    )
    var count = 0
    while True:
        var filled = batch.next(output)
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
    var batch = BoundsQueryBatch(
        tree.view().query_ray(
            Rayf32[.WORLD](
                Point3f32[.WORLD](0.5, 0.5, 0.0), Vec3f32[.WORLD](0.0, 0.0, 1.0)
            )
        )
    )
    var output = List[UInt32](length=4, fill=99)
    assert_equal(batch.next(output, 2.0), 1)
    assert_equal(output[0], UInt32(0))
    assert_equal(batch.next(output, 10.0), 0)
    var empty = SpatialBoundsBvh[.WORLD](List[AABB[.WORLD]]())
    var empty_batch = BoundsQueryBatch(
        empty.view().query_sphere(Point3f32[.WORLD](0.0), 1.0)
    )
    assert_equal(empty_batch.next(output), 0)


def test_mesh_aabb_batch_keeps_original_face_ids_and_groups() raises:
    var mesh = tetrahedron()
    var box = AABB(Point3f32[.WORLD](0.0), Point3f32[.WORLD](1.0))
    var roots: List[UInt32] = [0, mesh.group_root(7), mesh.group_root(999)]
    for r in range(3):
        var batch = BoundsQueryBatch(mesh.view().query_aabb(box, roots[r]))
        var output = List[UInt32](length=3, fill=99)
        var counts = List[Int](length=4, fill=0)
        while True:
            var count = batch.next(output)
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
