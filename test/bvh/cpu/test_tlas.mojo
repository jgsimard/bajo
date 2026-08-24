from std.testing import TestSuite, assert_true, assert_almost_equal

from bajo.bvh.constants import PrimitiveKind
from bajo.bvh.types import Instance, Sphere, Hit
from bajo.bvh.cpu.blas_set import (
    build_cpu_sphere_blas_set,
    build_cpu_triangle_blas_set,
)
from bajo.bvh.cpu.tlas import CpuTlas
from bajo.core import (
    AABB,
    Affine3f32,
    Vec3f32,
    Point3f32,
    Vec3W,
    Point3W,
    Rayf32,
)


def test_instance_derives_inverse_from_transform() raises:
    var transform = Affine3f32[.LOCAL, .WORLD].from_scale(
        Vec3f32[.LOCAL](2.0, 4.0, 5.0)
    )
    var bounds = AABB[.LOCAL](Point3f32[.LOCAL](-1.0), Point3f32[.LOCAL](1.0))
    var instance = Instance(transform, UInt32(0), bounds, .SPHERE)
    var local_point = Point3f32[.LOCAL](1.5, -2.0, 3.0)
    var round_trip = instance.inv_transform.point(
        instance.transform.point(local_point)
    )
    assert_almost_equal(round_trip.x, local_point.x)
    assert_almost_equal(round_trip.y, local_point.y)
    assert_almost_equal(round_trip.z, local_point.z)


def _triangle_vertices() -> List[Point3f32[.LOCAL]]:
    return [
        Point3f32[.LOCAL](-1.0, -1.0, 2.0),
        Point3f32[.LOCAL](1.0, -1.0, 2.0),
        Point3f32[.LOCAL](0.0, 1.0, 2.0),
    ]


def _triangle_bounds(
    vertices: List[Point3f32[.LOCAL]],
) -> AABB[.LOCAL]:
    return AABB(vertices[0], vertices[1], vertices[2])


def _sphere_values() -> List[Sphere[.LOCAL]]:
    return [Sphere(Point3f32[.LOCAL](0.0, 0.0, 2.0), 1.0)]


def _instance(
    kind: PrimitiveKind,
    blas_idx: UInt32,
    bounds: AABB[.LOCAL],
    tx: Float32,
    ty: Float32 = 0.0,
    tz: Float32 = 0.0,
) -> Instance:
    return Instance(
        Affine3f32[.LOCAL, .WORLD].from_translation(
            Vec3f32[.WORLD](tx, ty, tz)
        ),
        blas_idx,
        bounds,
        kind,
    )


def _ray(x: Float32 = 0.0) -> Rayf32[.WORLD]:
    return Rayf32[.WORLD](Point3W(x, 0.0, 0.0), Vec3W(0.0, 0.0, 1.0))


def _assert_hit(hit: Hit, inst: UInt32, t: Float32) raises:
    assert_true(hit.inst == inst)
    assert_true(hit.prim == 0)
    assert_almost_equal(hit.t, t)


def test_tlas_triangle_single_instance_cases() raises:
    var vertices = _triangle_vertices()
    var bounds = _triangle_bounds(vertices)
    var blases = build_cpu_triangle_blas_set[4]([vertices^])
    var identity = CpuTlas[4]([_instance(.TRIANGLE, 0, bounds, 0.0)])
    var hit = identity.trace_blases[4, 4, .CLOSEST_HIT](_ray(), blases)
    _assert_hit(hit, 0, 2.0)
    assert_almost_equal(hit.normal.z, 1.0)

    var translated = CpuTlas[4]([_instance(.TRIANGLE, 0, bounds, 5.0)])
    _assert_hit(
        translated.trace_blases[4, 4, .CLOSEST_HIT](_ray(5.0), blases),
        0,
        2.0,
    )
    assert_true(
        not translated.trace_blases[4, 4, .CLOSEST_HIT](_ray(), blases).is_hit()
    )


def test_tlas_triangle_two_instance_cases() raises:
    var first = _triangle_vertices()
    var bounds = _triangle_bounds(first)
    var blases = build_cpu_triangle_blas_set[4]([first^, _triangle_vertices()])
    var near_far = CpuTlas[4](
        [
            _instance(.TRIANGLE, 0, bounds, 0.0),
            _instance(.TRIANGLE, 1, bounds, 0.0, tz=6.0),
        ]
    )
    _assert_hit(
        near_far.trace_blases[4, 4, .CLOSEST_HIT](_ray(), blases),
        0,
        2.0,
    )
    var left_right = CpuTlas[4](
        [
            _instance(.TRIANGLE, 0, bounds, -5.0),
            _instance(.TRIANGLE, 1, bounds, 5.0),
        ]
    )
    _assert_hit(
        left_right.trace_blases[4, 4, .CLOSEST_HIT](_ray(5.0), blases),
        1,
        2.0,
    )


def _test_tlas_triangle_leaf_width[leaf_width: SIMDLength]() raises:
    var vertices = _triangle_vertices()
    var bounds = _triangle_bounds(vertices)
    var blases = build_cpu_triangle_blas_set[4]([vertices^])
    var tlas = CpuTlas[4, leaf_width](
        [
            _instance(.TRIANGLE, 0, bounds, -5.0),
            _instance(.TRIANGLE, 0, bounds, 5.0),
            _instance(.TRIANGLE, 0, bounds, 10.0),
        ]
    )
    _assert_hit(
        tlas.trace_blases[4, 4, .CLOSEST_HIT](_ray(5.0), blases),
        1,
        2.0,
    )
    assert_true(
        tlas.trace_blases[4, 4, .ANY_HIT](_ray(5.0), blases).is_occluded()
    )


def test_tlas_triangle_decoupled_leaf_widths() raises:
    _test_tlas_triangle_leaf_width[1]()
    _test_tlas_triangle_leaf_width[2]()


def test_tlas_leaf_instance_bounds_filter_missed_blas() raises:
    var vertices = _triangle_vertices()
    var bounds = _triangle_bounds(vertices)
    var blases = build_cpu_triangle_blas_set[4]([vertices^])
    var tlas = CpuTlas[4, 4](
        [
            _instance(.TRIANGLE, 99, bounds, -5.0),
            _instance(.TRIANGLE, 0, bounds, 5.0),
        ]
    )
    _assert_hit(
        tlas.trace_blases[4, 4, .CLOSEST_HIT](_ray(5.0), blases),
        1,
        2.0,
    )


def test_tlas_triangle_shadow_cases() raises:
    var vertices = _triangle_vertices()
    var bounds = _triangle_bounds(vertices)
    var blases = build_cpu_triangle_blas_set[4]([vertices^])
    var tlas = CpuTlas[4]([_instance(.TRIANGLE, 0, bounds, 5.0)])
    assert_true(
        tlas.trace_blases[4, 4, .ANY_HIT](_ray(5.0), blases).is_occluded()
    )
    assert_true(
        not tlas.trace_blases[4, 4, .ANY_HIT](_ray(), blases).is_occluded()
    )


def test_tlas_sphere_single_instance_cases() raises:
    var spheres = _sphere_values()
    var bounds = spheres[0].bounds()
    var blases = build_cpu_sphere_blas_set[4]([spheres^])
    var identity = CpuTlas[4]([_instance(.SPHERE, 0, bounds, 0.0)])
    var hit = identity.trace_blases[4, 4, .CLOSEST_HIT](_ray(), blases)
    _assert_hit(hit, 0, 1.0)
    assert_almost_equal(hit.normal.z, -1.0)

    var translated = CpuTlas[4]([_instance(.SPHERE, 0, bounds, 5.0)])
    _assert_hit(
        translated.trace_blases[4, 4, .CLOSEST_HIT](_ray(5.0), blases),
        0,
        1.0,
    )
    assert_true(
        not translated.trace_blases[4, 4, .CLOSEST_HIT](_ray(), blases).is_hit()
    )


def test_tlas_sphere_two_instance_cases() raises:
    var first = _sphere_values()
    var bounds = first[0].bounds()
    var blases = build_cpu_sphere_blas_set[4]([first^, _sphere_values()])
    var near_far = CpuTlas[4](
        [
            _instance(.SPHERE, 0, bounds, 0.0),
            _instance(.SPHERE, 1, bounds, 0.0, tz=6.0),
        ]
    )
    _assert_hit(
        near_far.trace_blases[4, 4, .CLOSEST_HIT](_ray(), blases),
        0,
        1.0,
    )
    var left_right = CpuTlas[4](
        [
            _instance(.SPHERE, 0, bounds, -5.0),
            _instance(.SPHERE, 1, bounds, 5.0),
        ]
    )
    _assert_hit(
        left_right.trace_blases[4, 4, .CLOSEST_HIT](_ray(5.0), blases),
        1,
        1.0,
    )


def test_tlas_sphere_nonuniform_scale_normal() raises:
    var spheres = _sphere_values()
    var bounds = spheres[0].bounds()
    var blases = build_cpu_sphere_blas_set[4]([spheres^])
    var transform = Affine3f32[.LOCAL, .WORLD].from_scale(
        Vec3f32[.LOCAL](2.0, 1.0, 1.0)
    )
    var tlas = CpuTlas[4]([Instance(transform, UInt32(0), bounds, .SPHERE)])
    var hit = tlas.trace_blases[4, 4, .CLOSEST_HIT](_ray(1.0), blases)
    _assert_hit(hit, 0, 1.1339746)
    assert_almost_equal(hit.normal.x, 0.2773501, atol=1.0e-5)
    assert_almost_equal(hit.normal.z, -0.9607689, atol=1.0e-5)


def test_tlas_sphere_shadow_cases() raises:
    var spheres = _sphere_values()
    var bounds = spheres[0].bounds()
    var blases = build_cpu_sphere_blas_set[4]([spheres^])
    var tlas = CpuTlas[4]([_instance(.SPHERE, 0, bounds, 5.0)])
    assert_true(
        tlas.trace_blases[4, 4, .ANY_HIT](_ray(5.0), blases).is_occluded()
    )
    assert_true(
        not tlas.trace_blases[4, 4, .ANY_HIT](_ray(), blases).is_occluded()
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
