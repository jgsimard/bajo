from std.testing import TestSuite, assert_true, assert_almost_equal

from bajo.bvh.constants import TRACE, Primitive
from bajo.bvh.types import Ray, Instance, Sphere, Hit
from bajo.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.bvh.cpu.sphere_bvh import SphereBvh
from bajo.bvh.cpu.tlas import Tlas
from bajo.core import Affine3f32, Vec3f32, Point3f32


def _make_one_local_triangle_z2() -> List[Point3f32]:
    return [
        Point3f32(-1.0, -1.0, 2.0),
        Point3f32(1.0, -1.0, 2.0),
        Point3f32(0.0, 1.0, 2.0),
    ]


def _make_one_local_sphere_z2() -> List[Sphere]:
    return [Sphere(Point3f32(0.0, 0.0, 2.0), 1.0)]


def _triangle_instance[
    width: SIMDSize
](
    blas_idx: UInt32,
    tx: Float32,
    ty: Float32,
    tz: Float32,
    blas: TriangleBvh[width],
) -> Instance:
    t = Vec3f32(tx, ty, tz)
    return Instance(
        Affine3f32.from_translation(t),
        Affine3f32.from_translation(-t),
        blas_idx,
        blas.bounds(),
        Primitive.TRIANGLE,
    )


def _sphere_instance[
    width: SIMDSize
](
    blas_idx: UInt32,
    tx: Float32,
    ty: Float32,
    tz: Float32,
    blas: SphereBvh[width],
) -> Instance:
    t = Vec3f32(tx, ty, tz)
    return Instance(
        Affine3f32.from_translation(t),
        Affine3f32.from_translation(-t),
        blas_idx,
        blas.bounds(),
        Primitive.SPHERE,
    )


def _assert_hit(
    hit: Hit,
    inst: UInt32,
    prim: UInt32,
    t: Float32,
) raises:
    assert_true(hit.inst == inst)
    assert_true(hit.prim == prim)
    assert_almost_equal(hit.t, t)


# -----------------------------------------------------------------------------
# Triangle TLAS
# -----------------------------------------------------------------------------
def test_tlas_triangle_single_instance_cases() raises:
    var verts = _make_one_local_triangle_z2()

    var blas = TriangleBvh[4](verts^)

    var blases = [blas.copy()]

    # Identity hit.
    var identity_instances = [_triangle_instance[4](0, 0.0, 0.0, 0.0, blas)]
    var identity_tlas = Tlas[4](identity_instances)

    var identity_ray = Ray(Point3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    _assert_hit(
        identity_tlas.trace[TriangleBvh[4], TRACE.CLOSEST_HIT](
            identity_ray,
            blases.unsafe_ptr(),
        ),
        0,
        0,
        2.0,
    )

    # Translated hit.
    var translated_instances = [_triangle_instance[4](0, 5.0, 0.0, 0.0, blas)]
    var translated_tlas = Tlas[4](translated_instances)

    var translated_hit_ray = Ray(
        Point3f32(5.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0)
    )
    _assert_hit(
        translated_tlas.trace[TriangleBvh[4], TRACE.CLOSEST_HIT](
            translated_hit_ray,
            blases.unsafe_ptr(),
        ),
        0,
        0,
        2.0,
    )

    # Translated miss.
    var translated_miss_ray = Ray(
        Point3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0)
    )
    var hit = translated_tlas.trace[TriangleBvh[4], TRACE.CLOSEST_HIT](
        translated_miss_ray,
        blases.unsafe_ptr(),
    )
    assert_true(not hit.is_hit())


def test_tlas_triangle_two_instance_cases() raises:
    var verts = _make_one_local_triangle_z2()

    var first_blas = TriangleBvh[4](verts.copy())
    var second_blas = TriangleBvh[4](verts^)

    var blases = [first_blas.copy(), second_blas.copy()]

    # Near/far along z: nearest should win.
    var near_far_instances = [
        _triangle_instance[4](0, 0.0, 0.0, 0.0, first_blas),
        _triangle_instance[4](1, 0.0, 0.0, 6.0, second_blas),
    ]

    var near_far_tlas = Tlas[4](near_far_instances)

    var center_ray = Ray(
        Point3f32(0.0, 0.0, 0.0),
        Vec3f32(0.0, 0.0, 1.0),
    )
    _assert_hit(
        near_far_tlas.trace[TriangleBvh[4], TRACE.CLOSEST_HIT](
            center_ray,
            blases.unsafe_ptr(),
        ),
        0,
        0,
        2.0,
    )
    # Left/right along x: ray targets second instance.
    var left_right_instances = [
        _triangle_instance[4](0, -5.0, 0.0, 0.0, first_blas),
        _triangle_instance[4](1, 5.0, 0.0, 0.0, second_blas),
    ]
    var left_right_tlas = Tlas[4](left_right_instances)

    var right_ray = Ray(Point3f32(5.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    _assert_hit(
        left_right_tlas.trace[TriangleBvh[4], TRACE.CLOSEST_HIT](
            right_ray,
            blases.unsafe_ptr(),
        ),
        1,
        0,
        2.0,
    )


def test_tlas_triangle_shadow_cases() raises:
    var verts = _make_one_local_triangle_z2()

    var blas = TriangleBvh[4](verts^)

    var blases = [blas.copy()]

    var instances = [_triangle_instance[4](0, 5.0, 0.0, 0.0, blas)]

    var tlas = Tlas[4](instances)

    var ray_hit = Ray(Point3f32(5.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    assert_true(
        tlas.trace[TriangleBvh[4], TRACE.ANY_HIT](
            ray_hit,
            blases.unsafe_ptr(),
        ).is_occluded()
    )

    var ray_miss = Ray(
        Point3f32(0.0, 0.0, 0.0),
        Vec3f32(0.0, 0.0, 1.0),
    )
    assert_true(
        not tlas.trace[TriangleBvh[4], TRACE.ANY_HIT](
            ray_miss,
            blases.unsafe_ptr(),
        ).is_occluded()
    )


# -----------------------------------------------------------------------------
# Sphere TLAS
# -----------------------------------------------------------------------------
def test_tlas_sphere_single_instance_cases() raises:
    var spheres = _make_one_local_sphere_z2()

    var blas = SphereBvh[4](spheres^)

    var blases = [blas.copy()]

    # Identity hit.
    var identity_instances = [_sphere_instance[4](0, 0.0, 0.0, 0.0, blas)]

    var identity_tlas = Tlas[4](identity_instances)

    var identity_ray = Ray(Point3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    _assert_hit(
        identity_tlas.trace[SphereBvh[4], TRACE.CLOSEST_HIT](
            identity_ray,
            blases.unsafe_ptr(),
        ),
        0,
        0,
        1.0,
    )

    # Translated hit.
    var translated_instances = [_sphere_instance[4](0, 5.0, 0.0, 0.0, blas)]
    var translated_tlas = Tlas[4](translated_instances)

    var translated_hit_ray = Ray(
        Point3f32(5.0, 0.0, 0.0),
        Vec3f32(0.0, 0.0, 1.0),
    )
    _assert_hit(
        translated_tlas.trace[SphereBvh[4], TRACE.CLOSEST_HIT](
            translated_hit_ray,
            blases.unsafe_ptr(),
        ),
        0,
        0,
        1.0,
    )

    # Translated miss.
    var translated_miss_ray = Ray(
        Point3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0)
    )

    var hit = translated_tlas.trace[SphereBvh[4], TRACE.CLOSEST_HIT](
        translated_miss_ray,
        blases.unsafe_ptr(),
    )
    assert_true(not hit.is_hit())


def test_tlas_sphere_two_instance_cases() raises:
    var spheres = _make_one_local_sphere_z2()

    var first_blas = SphereBvh[4](spheres.copy())
    var second_blas = SphereBvh[4](spheres^)

    var blases = [first_blas.copy(), second_blas.copy()]

    # Near/far along z: nearest should win.
    var near_far_instances = [
        _sphere_instance[4](0, 0.0, 0.0, 0.0, first_blas),
        _sphere_instance[4](1, 0.0, 0.0, 6.0, second_blas),
    ]
    var near_far_tlas = Tlas[4](near_far_instances)

    var center_ray = Ray(Point3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    _assert_hit(
        near_far_tlas.trace[SphereBvh[4], TRACE.CLOSEST_HIT](
            center_ray,
            blases.unsafe_ptr(),
        ),
        0,
        0,
        1.0,
    )

    # Left/right along x: ray targets second instance.
    var left_right_instances = [
        _sphere_instance[4](0, -5.0, 0.0, 0.0, first_blas),
        _sphere_instance[4](1, 5.0, 0.0, 0.0, second_blas),
    ]
    var left_right_tlas = Tlas[4](left_right_instances)

    var right_ray = Ray(Point3f32(5.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    _assert_hit(
        left_right_tlas.trace[SphereBvh[4], TRACE.CLOSEST_HIT](
            right_ray,
            blases.unsafe_ptr(),
        ),
        1,
        0,
        1.0,
    )


def test_tlas_sphere_shadow_cases() raises:
    var spheres = _make_one_local_sphere_z2()

    var blas = SphereBvh[4](spheres^)

    var blases = [blas.copy()]

    var instances = [_sphere_instance[4](0, 5.0, 0.0, 0.0, blas)]

    var tlas = Tlas[4](instances)

    var ray_hit = Ray(Point3f32(5.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    assert_true(
        tlas.trace[SphereBvh[4], TRACE.ANY_HIT](
            ray_hit, blases.unsafe_ptr()
        ).is_occluded()
    )

    var ray_miss = Ray(Point3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    assert_true(
        not tlas.trace[SphereBvh[4], TRACE.ANY_HIT](
            ray_miss,
            blases.unsafe_ptr(),
        ).is_occluded()
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
