from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_almost_equal,
)
from std.math import sqrt, atan2, abs
from std.memory import bitcast
from bajo.bvh.spatial.mesh import SpatialMesh
from bajo.bvh.constants import EMPTY_LANE
from bajo.bvh.spatial.geometry import angle_atan2
from bajo.core import AABB, Point3f32, Vec3f32, Rayf32
from test.bvh.spatial_fixtures import tetrahedron


def test_projection_regions_barycentrics_and_spheres() raises:
    var points: List[Point3f32[.WORLD]] = [
        Point3f32[.WORLD](0.0),
        Point3f32[.WORLD](1.0, 0.0, 0.0),
        Point3f32[.WORLD](0.0, 1.0, 0.0),
    ]
    var indices: List[UInt32] = [0, 1, 2]
    var mesh = SpatialMesh(points, indices)
    var probes: List[Point3f32[.WORLD]] = [
        Point3f32[.WORLD](0.2, 0.3, 2.0),
        Point3f32[.WORLD](-1.0, -1.0, 0.0),
        Point3f32[.WORLD](2.0, -1.0, 0.0),
        Point3f32[.WORLD](-1.0, 2.0, 0.0),
        Point3f32[.WORLD](0.5, -1.0, 0.0),
        Point3f32[.WORLD](-1.0, 0.5, 0.0),
        Point3f32[.WORLD](1.0, 1.0, 0.0),
    ]
    var expected: List[Point3f32[.WORLD]] = [
        Point3f32[.WORLD](0.2, 0.3, 0.0),
        points[0],
        points[1],
        points[2],
        Point3f32[.WORLD](0.5, 0.0, 0.0),
        Point3f32[.WORLD](0.0, 0.5, 0.0),
        Point3f32[.WORLD](0.5, 0.5, 0.0),
    ]
    var distances: List[Float64] = [
        2.0,
        sqrt(Float64(2.0)),
        sqrt(Float64(2.0)),
        sqrt(Float64(2.0)),
        1.0,
        1.0,
        sqrt(Float64(0.5)),
    ]
    var view = mesh.view()
    for i in range(len(probes)):
        var result = view.query_point_no_sign(probes[i])
        assert_true(result.result)
        assert_equal(result.face, UInt32(0))
        assert_almost_equal(result.distance, distances[i])
        var reconstructed = view.eval_position(result.face, result.u, result.v)
        for axis in range(3):
            assert_almost_equal(result.point[axis], expected[i][axis])
            assert_almost_equal(reconstructed[axis], expected[i][axis])
    assert_true(not view.query_point_no_sign(probes[0], 1.99).result)
    assert_true(view.query_point_no_sign(probes[0], 2.0).result)
    var face = UInt32(99)
    var sphere = view.query_sphere(Point3f32[.WORLD](1.0, 1.0, 0.0), 0.5)
    assert_true(
        not sphere.next(face)
    )  # Inside triangle AABB, outside triangle.
    assert_equal(face, UInt32(99))
    sphere = view.query_sphere(probes[0], 2.0)
    assert_true(sphere.next(face))
    assert_true(not sphere.next(face))
    var furthest = view.query_furthest_point_no_sign(Point3f32[.WORLD](0.0))
    assert_true(furthest.result)
    assert_equal(furthest.face, UInt32(0))
    assert_almost_equal(furthest.distance, Float64(1.0))
    assert_true(
        not view.query_furthest_point_no_sign(
            Point3f32[.WORLD](0.0), 1.01
        ).result
    )


def test_ray_intervals_counts_and_shared_edges() raises:
    var mesh = tetrahedron()
    var view = mesh.view()
    var ray = Rayf32[.WORLD](
        Point3f32[.WORLD](0.2, 0.2, 2.0), Vec3f32[.WORLD](0.0, 0.0, -2.0)
    )
    var hit = view.query_ray(ray)
    assert_true(hit.result and hit.front_face)
    assert_equal(hit.face, UInt32(3))
    assert_almost_equal(hit.t, Float64(0.7), atol=1e-7)
    assert_equal(view.query_ray_count_intersections(ray), UInt32(2))
    assert_true(view.query_ray_anyhit(ray))
    var interval = Rayf32[.WORLD](
        Point3f32[.WORLD](0.2, 0.2, 1.0),
        Vec3f32[.WORLD](0.0, 0.0, -1.0),
        t_min=1.0,
        t_max=1.0,
    )
    hit = view.query_ray(interval)
    assert_true(hit.result)
    assert_equal(hit.face, UInt32(0))
    assert_equal(hit.t, Float64(1.0))
    var vertex_ray = Rayf32[.WORLD](
        Point3f32[.WORLD](0.0, 0.0, -1.0), Vec3f32[.WORLD](0.0, 0.0, 1.0)
    )
    assert_equal(view.query_ray_count_intersections(vertex_ray), UInt32(2))
    assert_equal(view.query_ray(vertex_ray).face, UInt32(0))
    assert_equal(
        view.query_ray_count_intersections(ray, mesh.group_root(7)), UInt32(1)
    )
    assert_true(not view.query_ray(ray, mesh.group_root(999)).result)


def test_closed_tetrahedron_signs_against_analytic_grid() raises:
    var mesh = tetrahedron()
    var view = mesh.view()
    for x in range(-2, 8):
        for y in range(-2, 8):
            for z in range(-2, 8):
                var p = Point3f32[.WORLD](
                    Float32(x) * 0.13 + 0.017,
                    Float32(y) * 0.13 + 0.031,
                    Float32(z) * 0.13 + 0.043,
                )
                var inside = (
                    p.x > 0.0
                    and p.y > 0.0
                    and p.z > 0.0
                    and p.x + p.y + p.z < 1.0
                )
                var expected = Float32(-1.0 if inside else 1.0)
                assert_equal(view.query_point_sign_normal(p).sign, expected)
                assert_equal(view.query_point_sign_parity(p).sign, expected)
                assert_equal(
                    view.query_point_sign_winding_number(p).sign, expected
                )
                assert_almost_equal(
                    abs(view.winding_number(p)),
                    Float64(1.0 if inside else 0.0),
                    atol=1e-7,
                )
    var on_face = Point3f32[.WORLD](0.25, 0.25, 0.0)
    assert_equal(view.query_point(on_face).sign, Float32(0.0))
    assert_equal(view.query_point_sign_parity(on_face).sign, Float32(0.0))
    assert_equal(
        view.query_point_sign_winding_number(on_face).sign, Float32(0.0)
    )


def test_degenerate_faces_and_extreme_scales() raises:
    var points: List[Point3f32[.WORLD]] = [
        Point3f32[.WORLD](0.0),
        Point3f32[.WORLD](1.0, 0.0, 0.0),
        Point3f32[.WORLD](2.0, 0.0, 0.0),
    ]
    var indices: List[UInt32] = [0, 1, 2, 0, 0, 0]
    var mesh = SpatialMesh(points, indices)
    var view = mesh.view()
    var hit = view.query_point_no_sign(Point3f32[.WORLD](1.5, 1.0, 0.0))
    assert_true(hit.result)
    assert_almost_equal(hit.point.x, Float32(1.5))
    assert_equal(hit.distance, Float64(1.0))
    assert_true(
        not view.query_ray_anyhit(
            Rayf32[.WORLD](
                Point3f32[.WORLD](0.0, 0.0, 1.0),
                Vec3f32[.WORLD](0.0, 0.0, -1.0),
            )
        )
    )
    assert_equal(view.eval_normal(0).z, Float32(0.0))
    for scale in List[Float32]([Float32(1e-20), Float32(1e20)]):
        var scaled: List[Point3f32[.WORLD]] = [
            Point3f32[.WORLD](0.0),
            Point3f32[.WORLD](scale, 0.0, 0.0),
            Point3f32[.WORLD](0.0, scale, 0.0),
        ]
        var triangle: List[UInt32] = [0, 1, 2]
        var extreme = SpatialMesh(scaled, triangle)
        var p = Point3f32[.WORLD](scale * 0.25, scale * 0.25, scale)
        var projection = extreme.view().query_point_no_sign(p)
        assert_true(projection.result)
        assert_almost_equal(
            projection.distance / Float64(scale), Float64(1.0), atol=1e-6
        )
        var ray = Rayf32[.WORLD](p, Vec3f32[.WORLD](0.0, 0.0, -scale))
        assert_almost_equal(
            extreme.view().query_ray(ray).t, Float64(1.0), atol=1e-6
        )


def test_mesh_updates_velocities_and_failed_update_atomicity() raises:
    var mesh = tetrahedron()
    var pp = Int(mesh.points.unsafe_ptr())
    var np = Int(mesh.bvh.nodes.unsafe_ptr())
    var bp = Int(mesh.bvh.bounds.unsafe_ptr())
    var vp = Int(mesh.velocities.unsafe_ptr())
    var velocities: List[Vec3f32[.WORLD]] = [
        Vec3f32[.WORLD](1.0),
        Vec3f32[.WORLD](2.0),
        Vec3f32[.WORLD](3.0),
        Vec3f32[.WORLD](4.0),
    ]
    mesh.set_velocities(velocities)
    assert_almost_equal(
        mesh.view().eval_velocity(0, 0.5, 0.25).x, Float32(1.75)
    )
    var points: List[Point3f32[.WORLD]] = [
        Point3f32[.WORLD](2.0),
        Point3f32[.WORLD](3.0, 2.0, 2.0),
        Point3f32[.WORLD](2.0, 3.0, 2.0),
        Point3f32[.WORLD](2.0, 2.0, 3.0),
    ]
    for phase in range(2):
        if phase == 0:
            mesh.refit(points)
        else:
            mesh.rebuild(points)
        assert_equal(Int(mesh.points.unsafe_ptr()), pp)
        assert_equal(Int(mesh.bvh.nodes.unsafe_ptr()), np)
        assert_equal(Int(mesh.bvh.bounds.unsafe_ptr()), bp)
        assert_equal(Int(mesh.velocities.unsafe_ptr()), vp)
        assert_equal(
            mesh.view().query_point(Point3f32[.WORLD](2.1)).sign, Float32(-1.0)
        )
        assert_equal(
            mesh.view()
            .query_point_no_sign(
                Point3f32[.WORLD](2.1), root=mesh.group_root(7)
            )
            .face,
            UInt32(0),
        )
    var failed = False
    points[0] = Point3f32[.WORLD](bitcast[.float32](UInt32(0x7FC00000)))
    try:
        mesh.refit(points)
    except:
        failed = True
    assert_true(failed)
    assert_equal(mesh.points[0], Float32(2.0))
    assert_equal(
        mesh.view().query_point(Point3f32[.WORLD](2.1)).sign, Float32(-1.0)
    )
    mesh.set_velocities(List[Vec3f32[.WORLD]]())
    assert_equal(mesh.view().eval_velocity(0, 0.5, 0.25).x, Float32(0.0))


def test_empty_invalid_mesh_inputs_and_queries() raises:
    var empty = SpatialMesh[.WORLD](List[Point3f32[.WORLD]](), List[UInt32]())
    var p = Point3f32[.WORLD](0.0)
    var view = empty.view()
    assert_true(not view.query_point(p).result)
    assert_equal(
        view.query_ray_count_intersections(
            Rayf32[.WORLD](p, Vec3f32[.WORLD](1.0))
        ),
        UInt32(0),
    )
    var mesh = tetrahedron()
    var nan = bitcast[.float32](UInt32(0x7FC00000))
    assert_true(
        not mesh.view().query_point_no_sign(Point3f32[.WORLD](nan)).result
    )
    assert_true(not mesh.view().query_point_no_sign(p, -1.0).result)
    assert_true(not mesh.view().query_point_sign_normal(p, epsilon=-1.0).result)
    assert_true(
        not mesh.view().query_point_sign_winding_number(p, threshold=nan).result
    )
    assert_true(
        not mesh.view()
        .query_ray(Rayf32[.WORLD](p, Vec3f32[.WORLD](nan)))
        .result
    )
    var rejected = False
    try:
        var bad = SpatialMesh[.WORLD](
            List[Point3f32[.WORLD]](),
            List[UInt32]([UInt32(0), UInt32(1), UInt32(2)]),
        )
    except:
        rejected = True
    assert_true(rejected)
    rejected = False
    try:
        var bad_indices: List[UInt32] = [0]
        var bad = SpatialMesh[.WORLD](List[Point3f32[.WORLD]]([p]), bad_indices)
    except:
        rejected = True
    assert_true(rejected)


def test_portable_angle_against_libm() raises:
    for i in range(-1000, 1001):
        var y = Float64(i) / 137.0
        for x in List[Float64](
            [
                Float64(-1000.0),
                Float64(-1.0),
                Float64(-0.001),
                Float64(0.0),
                Float64(0.001),
                Float64(1.0),
                Float64(1000.0),
            ]
        ):
            assert_almost_equal(angle_atan2(y, x), atan2(y, x), atol=2e-15)
    assert_equal(angle_atan2(Float64(0.0), Float64(0.0)), Float64(0.0))


def test_parity_retries_boundary_hits_on_closed_octahedron() raises:
    # The first three parity directions meet four-face vertices. Counting each
    # face would report even parity from this interior point.
    var directions: List[Point3f32[.WORLD]] = []
    for sample in range(3):
        directions.append(
            Point3f32[.WORLD](
                1.0,
                0.173 + Float32((sample * 37 + 11) % 101) * 0.013,
                0.219 + Float32((sample * sample * 19 + 23) % 103) * 0.017,
            )
        )
    var points = directions.copy()
    for p in directions:
        points.append(Point3f32[.WORLD](-p.x, -p.y, -p.z))
    var indices: List[UInt32] = [
        0,
        1,
        2,
        0,
        2,
        4,
        0,
        4,
        5,
        0,
        5,
        1,
        3,
        2,
        1,
        3,
        4,
        2,
        3,
        5,
        4,
        3,
        1,
        5,
    ]
    var mesh = SpatialMesh(points, indices)
    assert_equal(
        mesh.view().query_point_sign_parity(Point3f32[.WORLD](0.0)).sign,
        Float32(-1.0),
    )


def test_skinny_triangle_projection_avoids_gram_cancellation() raises:
    var points: List[Point3f32[.WORLD]] = [
        Point3f32[.WORLD](0.0),
        Point3f32[.WORLD](1.0, 0.0, 0.0),
        Point3f32[.WORLD](1.0, 1e-20, 0.0),
    ]
    var indices: List[UInt32] = [0, 1, 2]
    var mesh = SpatialMesh(points, indices)
    var p = Point3f32[.WORLD](0.5, 2.5e-21, 0.0)
    var hit = mesh.view().query_point_no_sign(p, 0.0)
    assert_true(hit.result)
    assert_equal(hit.distance, Float64(0.0))
    assert_almost_equal(hit.u, Float32(0.5))
    assert_almost_equal(hit.v, Float32(0.25), atol=1e-6)


def test_mesh_bounds_groups_and_reversed_orientation() raises:
    var mesh = tetrahedron()
    var query = mesh.view().query_aabb(
        AABB(Point3f32[.WORLD](0.0), Point3f32[.WORLD](1.0)), mesh.group_root(8)
    )
    var found = List[UInt32]()
    var face = UInt32(99)
    while query.next(face):
        found.append(face)
    sort(found)
    assert_equal(len(found), 2)
    assert_equal(found[0], UInt32(2))
    assert_equal(found[1], UInt32(3))
    assert_true(not query.next(face))
    var points: List[Point3f32[.WORLD]] = []
    for vertex in range(4):
        points.append(mesh.view().vertex(UInt32(vertex)))
    var reversed = mesh.indices.copy()
    for index in range(0, len(reversed), 3):
        var old = reversed[index + 1]
        reversed[index + 1] = reversed[index + 2]
        reversed[index + 2] = old
    var inward = SpatialMesh(points, reversed)
    var inside = Point3f32[.WORLD](0.1)
    assert_equal(
        inward.view().query_point_sign_normal(inside).sign, Float32(1.0)
    )
    assert_equal(
        inward.view().query_point_sign_parity(inside).sign, Float32(-1.0)
    )
    assert_equal(
        inward.view().query_point_sign_winding_number(inside).sign,
        Float32(-1.0),
    )
    assert_almost_equal(
        mesh.view().query_point_no_sign(inside).distance,
        Float64(Float32(0.1)),
        atol=1e-8,
    )


def test_normal_sign_preserves_barycentrics_and_velocity() raises:
    var mesh = tetrahedron()
    var velocities: List[Vec3f32[.WORLD]] = [
        Vec3f32[.WORLD](1.0, -1.0, 0.5),
        Vec3f32[.WORLD](3.0, -1.0, 0.5),
        Vec3f32[.WORLD](1.0, 2.0, 0.5),
        Vec3f32[.WORLD](1.0, -1.0, 4.5),
    ]
    mesh.set_velocities(velocities)
    var view = mesh.view()
    for lane in range(128):
        var p = Point3f32[.WORLD](
            Float32(lane % 7) * 0.19 - 0.1,
            Float32(lane % 5) * 0.23 + 0.017,
            Float32(lane % 3) * 0.37 + 0.043,
        )
        var unsigned = view.query_point_no_sign(p)
        var signed = view.query_point_sign_normal(p)
        assert_equal(signed.face, unsigned.face)
        assert_equal(signed.u, unsigned.u)
        assert_equal(signed.v, unsigned.v)
        var position = view.eval_position(signed.face, signed.u, signed.v)
        var velocity = view.eval_velocity(signed.face, signed.u, signed.v)
        assert_almost_equal(position.x, signed.point.x, atol=1e-6)
        assert_almost_equal(position.y, signed.point.y, atol=1e-6)
        assert_almost_equal(position.z, signed.point.z, atol=1e-6)
        assert_almost_equal(velocity.x, signed.point.x * 2.0 + 1.0, atol=1e-6)
        assert_almost_equal(velocity.y, signed.point.y * 3.0 - 1.0, atol=1e-6)
        assert_almost_equal(velocity.z, signed.point.z * 4.0 + 0.5, atol=1e-6)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
