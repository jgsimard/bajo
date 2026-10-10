"""Distance-pruning and SIMD width checks against exhaustive triangle scans."""
from std.testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_true,
)
from std.math import sqrt, atan2
from bajo.core import Point3f32, AABB, Rayf32, Vec3f32, Vec3
from bajo.bvh.spatial import SpatialMesh, SpatialBoundsBvh
from bajo.bvh.cpu.builder import BoundsItem
from bajo.core.intersect import (
    project_triangle,
    angle_atan2,
    intersect_triangle,
)
from bajo.bvh.constants import EMPTY_LANE
from bajo.bvh.spatial.query import (
    ray_overlaps_bounds,
    ray_overlaps_bounds_cached,
)


def check_closest[
    width: SIMDLength
](
    mesh: SpatialMesh[.WORLD],
    point: Point3f32[.WORLD],
    cutoff: Float32,
    group: Int,
) raises:
    var best2 = Float64(cutoff) * Float64(cutoff)
    var expected = EMPTY_LANE
    var eu = Float32(0.0)
    var ev = Float32(0.0)
    var ep = Point3f32[.WORLD](0.0)
    var view = mesh.view()
    for face in range(len(mesh.indices) // 3):
        if group >= 0 and mesh.bvh.groups[face] != UInt32(group):
            continue
        var projected = project_triangle(
            point,
            view.face_vertex(UInt32(face), 0),
            view.face_vertex(UInt32(face), 1),
            view.face_vertex(UInt32(face), 2),
        )
        if projected.distance2 <= best2 and (
            expected == EMPTY_LANE
            or projected.distance2 < best2
            or UInt32(face) < expected
        ):
            expected = UInt32(face)
            best2 = projected.distance2
            eu = Float32(projected.u)
            ev = Float32(projected.v)
            ep = Point3f32[.WORLD](
                Float32(projected.point.x),
                Float32(projected.point.y),
                Float32(projected.point.z),
            )
    var root = UInt32(0) if group < 0 else mesh.group_root(UInt32(group))
    var actual = view.query_point_no_sign[width](point, cutoff, root)
    assert_equal(actual.result, expected != EMPTY_LANE)
    assert_equal(actual.face, expected)
    if actual.result:
        assert_equal(actual.u, eu)
        assert_equal(actual.v, ev)
        assert_almost_equal(actual.distance, sqrt(best2), atol=1e-12)
        assert_equal(actual.point.x, ep.x)
        assert_equal(actual.point.y, ep.y)
        assert_equal(actual.point.z, ep.z)


def check_furthest[
    width: SIMDLength
](
    mesh: SpatialMesh[.WORLD],
    point: Point3f32[.WORLD],
    cutoff: Float32,
    group: Int,
) raises:
    var best2 = Float64(cutoff) * Float64(cutoff)
    var expected = EMPTY_LANE
    var eu = Float32(0.0)
    var ev = Float32(0.0)
    var ep = Point3f32[.WORLD](0.0)
    var view = mesh.view()
    for face in range(len(mesh.indices) // 3):
        if group >= 0 and mesh.bvh.groups[face] != UInt32(group):
            continue
        for corner in range(3):
            var vertex = view.face_vertex(UInt32(face), corner)
            var dx = Float64(vertex.x) - Float64(point.x)
            var dy = Float64(vertex.y) - Float64(point.y)
            var dz = Float64(vertex.z) - Float64(point.z)
            var distance2 = dx * dx + dy * dy + dz * dz
            if distance2 >= best2 and (
                expected == EMPTY_LANE
                or distance2 > best2
                or UInt32(face) < expected
            ):
                expected = UInt32(face)
                best2 = distance2
                eu = Float32(1.0 if corner == 0 else 0.0)
                ev = Float32(1.0 if corner == 1 else 0.0)
                ep = vertex
    var root = UInt32(0) if group < 0 else mesh.group_root(UInt32(group))
    var actual = view.query_furthest_point_no_sign[width](point, cutoff, root)
    assert_equal(actual.result, expected != EMPTY_LANE)
    assert_equal(actual.face, expected)
    if actual.result:
        assert_equal(actual.u, eu)
        assert_equal(actual.v, ev)
        assert_almost_equal(actual.distance, sqrt(best2), atol=1e-12)
        assert_equal(actual.point.x, ep.x)
        assert_equal(actual.point.y, ep.y)
        assert_equal(actual.point.z, ep.z)


def check_fixture[width: SIMDLength](mesh: SpatialMesh[.WORLD]) raises:
    var cutoffs: List[Float32] = [0.0, 0.1, 0.5, 4.0]
    for q in range(128):
        var p = Point3f32[.WORLD](
            Float32(q % 11) * 0.193 - 0.3,
            Float32((q * 17) % 13) * 0.157 - 0.2,
            Float32((q * 23) % 7) * 0.211 - 0.1,
        )
        for cutoff in cutoffs:
            for group in range(-1, 4):
                check_closest[width](mesh, p, cutoff, group)
                check_furthest[width](mesh, p, cutoff, group)
    for vertex in range(len(mesh.points) // 3):
        check_closest[width](mesh, mesh.view().vertex(UInt32(vertex)), 0.0, -1)


def test_distance_pruning_widths_groups_updates_and_degenerate_ties() raises:
    var points = List[Point3f32[.WORLD]]()
    for i in range(48):
        points.append(
            Point3f32[.WORLD](
                Float32(i % 7) * 0.31,
                Float32((i * 13) % 11) * 0.19,
                Float32((i * 17) % 9) * 0.23,
            )
        )
    var indices = List[UInt32]()
    var groups = List[UInt32]()
    for i in range(96):
        indices.extend(
            [
                UInt32(i % 48),
                UInt32((i * 7 + 3) % 48),
                UInt32((i * 17 + 5) % 48),
            ]
        )
        groups.append(UInt32(i % 3))
    # Duplicate original geometry and a point/segment face exercise ties.
    indices.extend(
        [
            UInt32(0),
            UInt32(0),
            UInt32(0),
            UInt32(1),
            UInt32(1),
            UInt32(2),
            UInt32(0),
            UInt32(3),
            UInt32(5),
        ]
    )
    groups.extend([UInt32(0), UInt32(1), UInt32(2)])
    var mesh = SpatialMesh(points, indices, groups)
    for phase in range(3):
        if phase > 0:
            for i in range(len(points)):
                points[i].x += 0.113
                points[i].y -= 0.071
            if phase == 1:
                mesh.refit(points)
            else:
                mesh.rebuild(points)
        check_fixture[1](mesh)
        check_fixture[2](mesh)
        check_fixture[4](mesh)
        check_fixture[8](mesh)


def check_winding_cache[width: SIMDLength]() raises:
    var points = List[Point3f32[.WORLD]]()
    var indices = List[UInt32]()
    var groups = List[UInt32]()
    var order: List[UInt32] = [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]
    for tet in range(3):
        var x = Float32(tet) * 3.0
        points.extend(
            [
                Point3f32[.WORLD](x, 0.0, 0.0),
                Point3f32[.WORLD](x + 1.0, 0.0, 0.0),
                Point3f32[.WORLD](x, 1.0, 0.0),
                Point3f32[.WORLD](x, 0.0, 1.0),
            ]
        )
        for index in order:
            indices.append(UInt32(tet * 4) + index)
        for _ in range(4):
            groups.append(UInt32(tet % 2))
    indices.extend([UInt32(0), UInt32(0), UInt32(0)])
    groups.append(0)
    var mesh = SpatialMesh[.WORLD, width](points, indices, groups)
    var cache_address = Int(mesh.packed_triangles.unsafe_ptr())
    for phase in range(3):
        if phase > 0:
            for vertex in range(len(points)):
                points[vertex].x += 0.13
            if phase == 1:
                mesh.refit(points)
            else:
                mesh.rebuild(points)
        assert_equal(Int(mesh.packed_triangles.unsafe_ptr()), cache_address)
        var roots: List[UInt32] = [
            0,
            mesh.group_root(0),
            mesh.group_root(1),
            mesh.group_root(999),
        ]
        for q in range(64):
            var p = Point3f32[.WORLD](
                Float32(q % 7) * 0.17 - 0.05 + Float32(q % 3) * 3.0,
                Float32(q % 5) * 0.23 + 0.01,
                Float32(q % 3) * 0.31 + 0.02,
            )
            for root in roots:
                var view = mesh.view()
                assert_almost_equal(
                    view.winding_number(p, root),
                    view._winding_number[1](p, root),
                    atol=1e-12,
                    rtol=1e-12,
                )
    var previous = mesh.packed_triangles.copy()
    var invalid = points.copy()
    invalid.append(Point3f32[.WORLD](0.0))
    var failed = False
    try:
        mesh.refit(invalid)
    except:
        failed = True
    assert_true(failed)
    for i in range(len(previous)):
        assert_equal(mesh.packed_triangles[i], previous[i])


def check_vector_angles[width: SIMDLength]() raises:
    for q in range(1000):
        var x = SIMD[.float64, width](0.0)
        var y = SIMD[.float64, width](0.0)
        comptime for lane in range(width):
            x[lane] = Float64((q * 17 + Int(lane) * 11) % 127 - 63)
            y[lane] = Float64((q * 23 + Int(lane) * 7) % 131 - 65)
        var actual = angle_atan2(y, x)
        comptime for lane in range(width):
            assert_almost_equal(
                actual[lane], atan2(y[lane], x[lane]), atol=1e-15, rtol=1e-15
            )


def test_winding_cache_widths_tails_groups_updates_and_angles() raises:
    check_winding_cache[1]()
    check_winding_cache[2]()
    check_winding_cache[4]()
    check_winding_cache[8]()
    check_winding_cache[16]()
    check_vector_angles[1]()
    check_vector_angles[2]()
    check_vector_angles[4]()
    check_vector_angles[8]()
    check_vector_angles[16]()


def check_bounds_refit[width: SIMDLength]() raises:
    for count in range(0, 20):
        var boxes = List[AABB[.WORLD]]()
        var groups = List[UInt32]()
        for i in range(count):
            var p = Point3f32[.WORLD](
                Float32(i) * 0.21, Float32(i % 3) * 0.13, Float32(i % 5) * 0.11
            )
            boxes.append(
                AABB(p, Point3f32[.WORLD](p.x + 0.1, p.y + 0.2, p.z + 0.3))
            )
            groups.append(UInt32(i % 3))
        var tree = SpatialBoundsBvh(boxes, groups)
        for phase in range(2):
            for i in range(count):
                var x = Float32(count - i) * 0.17 + Float32(phase) * 0.21
                boxes[i] = AABB(
                    Point3f32[.WORLD](x, -0.13, Float32(i % 5) * 0.17),
                    Point3f32[.WORLD](
                        x + 0.19, 0.33, Float32(i % 5) * 0.17 + 0.21
                    ),
                )
            tree.refit[width](boxes)
            for q in range(20):
                var box = AABB(
                    Point3f32[.WORLD](Float32(q) * 0.19, -0.2, 0.2),
                    Point3f32[.WORLD](Float32(q) * 0.19 + 0.31, 0.2, 0.5),
                )
                for group in range(-1, 4):
                    var root = UInt32(0) if group < 0 else tree.group_root(
                        UInt32(group)
                    )
                    var query = tree.view().query_aabb(box, root)
                    var seen = List[Int](length=count, fill=0)
                    var item = UInt32(0)
                    while query.next(item):
                        seen[Int(item)] += 1
                    for i in range(count):
                        var expected = boxes[i].overlaps(box)[0] and (
                            group < 0 or groups[i] == UInt32(group)
                        )
                        assert_equal(seen[i], 1 if expected else 0)


def test_bounds_refit_simd_widths_empty_tails_groups_and_deformation() raises:
    check_bounds_refit[1]()
    check_bounds_refit[2]()
    check_bounds_refit[4]()
    check_bounds_refit[8]()
    check_bounds_refit[16]()


def test_cached_ray_slabs_match_division_at_boundaries_and_scales() raises:
    var scales: List[Float32] = [1e-30, 1e-10, 1.0, 1e10, 1e30]
    var cutoffs: List[Float32] = [
        -1.0,
        0.0,
        0.99999994,
        1.0,
        1.00000012,
        2.0,
        10.0,
    ]
    for scale in scales:
        var bounds = AABB(Point3f32[.WORLD](-scale), Point3f32[.WORLD](scale))
        for q in range(256):
            var origin = Point3f32[.WORLD](
                Float32(q % 7 - 3) * scale,
                Float32((q * 11) % 7 - 3) * scale,
                Float32((q * 13) % 7 - 3) * scale,
            )
            var direction = Vec3f32[.WORLD](
                Float32(q % 5 - 2) * scale,
                Float32((q * 3) % 5 - 2) * scale,
                Float32((q * 7) % 5 - 2) * scale,
            )
            var inverse = Vec3[.float64, .WORLD](0.0)
            comptime for axis in range(3):
                var d = Float64(direction[axis])
                inverse.__setitem__[axis](1.0 / d if d != 0.0 else 0.0)
            for cutoff in cutoffs:
                var ray = Rayf32[.WORLD](origin, direction, t_max=10.0)
                for expanded in range(2):
                    var radius = scale * Float32(expanded) * 0.25
                    assert_equal(
                        ray_overlaps_bounds_cached(
                            ray, bounds, cutoff, radius, inverse
                        ),
                        ray_overlaps_bounds(ray, bounds, cutoff, radius),
                    )


def test_closest_ray_tightening_preserves_ties_and_negative_intervals() raises:
    var points = List[Point3f32[.WORLD]]()
    var indices = List[UInt32]()
    var groups = List[UInt32]()
    for face in range(64):
        var z = Float32(face % 16 - 8) * 0.17
        var x = Float32((face * 7) % 5 - 2) * 0.13
        var base = UInt32(len(points))
        points.extend(
            [
                Point3f32[.WORLD](x - 1.0, -1.0, z),
                Point3f32[.WORLD](x + 1.0, -1.0, z),
                Point3f32[.WORLD](x, 1.0, z),
            ]
        )
        indices.extend([base, base + 1, base + 2])
        groups.append(UInt32(face % 3))
    var mesh = SpatialMesh(points, indices, groups=groups)
    var view = mesh.view()
    for q in range(128):
        var ray = Rayf32[.WORLD](
            Point3f32[.WORLD](
                Float32(q % 7 - 3) * 0.11,
                Float32((q * 17) % 7 - 3) * 0.09,
                Float32(q % 3) * 0.17,
            ),
            Vec3f32[.WORLD](0.0, 0.0, Float32(1.0 if q % 2 == 0 else -1.0)),
            t_min=-2.0,
            t_max=2.0,
        )
        for group in range(-1, 4):
            var expected = EMPTY_LANE
            var best = Float64(ray.t_max)
            for face in range(64):
                if group >= 0 and groups[face] != UInt32(group):
                    continue
                var hit = intersect_triangle(
                    ray,
                    view.face_vertex(UInt32(face), 0),
                    view.face_vertex(UInt32(face), 1),
                    view.face_vertex(UInt32(face), 2),
                )
                if hit.result and (
                    hit.t < best or (hit.t == best and UInt32(face) < expected)
                ):
                    best = hit.t
                    expected = UInt32(face)
            var root = UInt32(0) if group < 0 else mesh.group_root(
                UInt32(group)
            )
            comptime for exponent in range(5):
                comptime width = 1 << exponent
                var actual = view.query_ray[width](ray, root)
                assert_equal(actual.result, expected != EMPTY_LANE)
                assert_equal(actual.face, expected)
                if actual.result:
                    assert_equal(actual.t, best)
                    var hit = intersect_triangle(
                        ray,
                        view.face_vertex(expected, 0),
                        view.face_vertex(expected, 1),
                        view.face_vertex(expected, 2),
                    )
                    assert_equal(actual.u, Float32(hit.u))
                    assert_equal(actual.v, Float32(hit.v))
                    var normal = view.eval_normal(expected)
                    assert_equal(actual.normal.x, normal.x)
                    assert_equal(actual.normal.y, normal.y)
                    assert_equal(actual.normal.z, normal.z)


def check_ordered_ray(
    mesh: SpatialMesh[.WORLD], ray: Rayf32[.WORLD], group: Int
) raises:
    var view = mesh.view()
    var expected = EMPTY_LANE
    var expected_hits = UInt32(0)
    var best = Float64(ray.t_max)
    for face in range(len(mesh.indices) // 3):
        if group >= 0 and mesh.bvh.groups[face] != UInt32(group):
            continue
        var hit = intersect_triangle(
            ray,
            view.face_vertex(UInt32(face), 0),
            view.face_vertex(UInt32(face), 1),
            view.face_vertex(UInt32(face), 2),
        )
        if hit.result:
            expected_hits += 1
        if hit.result and (
            hit.t < best or (hit.t == best and UInt32(face) < expected)
        ):
            best = hit.t
            expected = UInt32(face)
    var root = UInt32(0) if group < 0 else mesh.group_root(UInt32(group))
    comptime for exponent in range(5):
        var actual = view.query_ray[1 << exponent](ray, root)
        assert_equal(
            view.query_ray_count_intersections[1 << exponent](ray, root),
            expected_hits,
        )
        assert_equal(
            view.query_ray_anyhit[1 << exponent](ray, root), expected_hits != 0
        )
        assert_equal(actual.result, expected != EMPTY_LANE)
        assert_equal(actual.face, expected)
        if actual.result:
            assert_equal(actual.t, best)


def test_ordered_ray_scales_oblique_parallel_and_updates() raises:
    var scales: List[Float32] = [1e-30, 1e-10, 1.0, 1e10, 1e30]
    for scale in scales:
        var points = List[Point3f32[.WORLD]]()
        var indices = List[UInt32]()
        var groups = List[UInt32]()
        for face in range(32):
            var x = Float32(face % 4 - 2) * scale
            var z = Float32(face // 4 - 4) * scale
            var base = UInt32(len(points))
            points.extend(
                [
                    Point3f32[.WORLD](x - scale, -scale, z),
                    Point3f32[.WORLD](x + scale, -scale, z + scale * 0.13),
                    Point3f32[.WORLD](x, scale, z - scale * 0.07),
                ]
            )
            indices.extend([base, base + 1, base + 2])
            groups.append(UInt32(face % 3))
        var mesh = SpatialMesh(points, indices, groups=groups)
        for phase in range(3):
            if phase != 0:
                for vertex in range(len(points)):
                    points[vertex] = Point3f32[.WORLD](
                        points[vertex].x
                        + scale * Float32(vertex % 3 - 1) * 0.23,
                        points[vertex].y - scale * 0.17,
                        points[vertex].z + scale * 0.11,
                    )
                if phase == 1:
                    mesh.refit(points)
                else:
                    mesh.rebuild(points)
            for q in range(128):
                var ray = Rayf32[.WORLD](
                    Point3f32[.WORLD](
                        scale * Float32(q % 7 - 3) * 0.31,
                        scale * Float32((q * 17) % 7 - 3) * 0.27,
                        scale * Float32(q % 3 - 1),
                    ),
                    Vec3f32[.WORLD](
                        scale * Float32(q % 3 - 1) * 0.13,
                        scale * Float32(q % 5 - 2) * 0.19,
                        scale * Float32(1.0 if q % 2 == 0 else -1.0),
                    ),
                    t_min=-2.0,
                    t_max=2.0,
                )
                for group in range(-1, 4):
                    check_ordered_ray(mesh, ray, group)


def check_build_reduction[width: SIMDLength]() raises:
    var items = List[BoundsItem[.WORLD]]()
    var order = List[UInt32]()
    for i in range(39):
        var lo = Point3f32[.WORLD](
            -Float32(i % 5), -Float32(i % 7), -Float32(i % 11)
        )
        var hi = Point3f32[.WORLD](
            Float32(i % 13), Float32(i % 17), Float32(i % 19)
        )
        items.append(BoundsItem(AABB(lo, hi), UInt32(i)))
        order.append(UInt32(38 - i))
    var boxes = List[AABB[.WORLD]]()
    var tree = SpatialBoundsBvh(boxes)
    for first in range(20):
        for count in range(1, 20):
            var expected = AABB[.WORLD].invalid()
            for i in range(first, first + count):
                expected = AABB[.WORLD].merge(
                    expected, items[Int(order[i])].bounds
                )
            var actual = tree._range_bounds[width](items, order, first, count)
            comptime for axis in range(3):
                assert_equal(actual._min[axis], expected._min[axis])
                assert_equal(actual._max[axis], expected._max[axis])


def test_direct_builder_bounds_reduction_widths_and_subsets() raises:
    check_build_reduction[1]()
    check_build_reduction[2]()
    check_build_reduction[4]()
    check_build_reduction[8]()
    check_build_reduction[16]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
