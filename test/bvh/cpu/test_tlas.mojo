from std.testing import TestSuite, assert_true, assert_almost_equal

from bajo.bvh.constants import TRACE, Primitive
from bajo.bvh.types import Instance, Sphere, Hit
from bajo.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.bvh.cpu.sphere_bvh import SphereBvh
from bajo.bvh.cpu.tlas import Tlas
from bajo.core import (
    AABB,
    Affine3f32,
    Vec3f32,
    Point3f32,
    Frame,
    Vec3W,
    Point3W,
    Rayf32,
)


def test_instance_derives_inverse_from_transform() raises:
    var transform = Affine3f32[Frame.LOCAL, Frame.WORLD].from_scale(
        Vec3f32[Frame.LOCAL](2.0, 4.0, 5.0)
    )
    var bounds = AABB[Frame.LOCAL](
        Point3f32[Frame.LOCAL](-1.0),
        Point3f32[Frame.LOCAL](1.0),
    )
    var instance = Instance(transform, UInt32(0), bounds, Primitive.SPHERE)
    var local_point = Point3f32[Frame.LOCAL](1.5, -2.0, 3.0)
    var world_point = instance.transform.point(local_point)
    var round_trip = instance.inv_transform.point(world_point)

    assert_almost_equal(round_trip.x, local_point.x)
    assert_almost_equal(round_trip.y, local_point.y)
    assert_almost_equal(round_trip.z, local_point.z)


def _make_one_local_triangle_z2[frame: Frame]() -> List[Point3f32[frame]]:
    return [
        Point3f32[frame](-1.0, -1.0, 2.0),
        Point3f32[frame](1.0, -1.0, 2.0),
        Point3f32[frame](0.0, 1.0, 2.0),
    ]


def _make_one_local_sphere_z2[frame: Frame]() -> List[Sphere[frame]]:
    return [Sphere(Point3f32[frame](0.0, 0.0, 2.0), 1.0)]


def _triangle_instance[
    width: SIMDSize
](
    blas_idx: UInt32,
    tx: Float32,
    ty: Float32,
    tz: Float32,
    blas: TriangleBvh[Frame.LOCAL, width],
) -> Instance:
    var t_world = Vec3f32[Frame.WORLD](tx, ty, tz)
    return Instance(
        Affine3f32[Frame.LOCAL, Frame.WORLD].from_translation(t_world),
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
    blas: SphereBvh[Frame.LOCAL, width],
) -> Instance:
    var t_world = Vec3f32[Frame.WORLD](tx, ty, tz)
    return Instance(
        Affine3f32[Frame.LOCAL, Frame.WORLD].from_translation(t_world),
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
    var verts = _make_one_local_triangle_z2[Frame.LOCAL]()

    var blas = TriangleBvh[Frame.LOCAL, 4](verts^)

    var blases = [blas.copy()]

    # Identity hit.
    var identity_instances = [_triangle_instance[4](0, 0.0, 0.0, 0.0, blas)]
    var identity_tlas = Tlas[4](identity_instances)

    var identity_ray = Rayf32[Frame.WORLD](
        Point3W(0.0, 0.0, 0.0), Vec3W(0.0, 0.0, 1.0)
    )
    var identity_hit = identity_tlas.trace[
        TriangleBvh[Frame.LOCAL, 4], TRACE.CLOSEST_HIT
    ](
        identity_ray,
        blases.unsafe_ptr(),
    )
    _assert_hit(identity_hit, 0, 0, 2.0)
    assert_almost_equal(identity_hit.normal.x, 0.0)
    assert_almost_equal(identity_hit.normal.y, 0.0)
    assert_almost_equal(identity_hit.normal.z, 1.0)

    # Translated hit.
    var translated_instances = [_triangle_instance[4](0, 5.0, 0.0, 0.0, blas)]
    var translated_tlas = Tlas[4](translated_instances)

    var translated_hit_ray = Rayf32[Frame.WORLD](
        Point3W(5.0, 0.0, 0.0), Vec3W(0.0, 0.0, 1.0)
    )
    _assert_hit(
        translated_tlas.trace[TriangleBvh[Frame.LOCAL, 4], TRACE.CLOSEST_HIT](
            translated_hit_ray,
            blases.unsafe_ptr(),
        ),
        0,
        0,
        2.0,
    )

    # Translated miss.
    var translated_miss_ray = Rayf32[Frame.WORLD](
        Point3W(0.0, 0.0, 0.0), Vec3W(0.0, 0.0, 1.0)
    )
    var hit = translated_tlas.trace[
        TriangleBvh[Frame.LOCAL, 4], TRACE.CLOSEST_HIT
    ](
        translated_miss_ray,
        blases.unsafe_ptr(),
    )
    assert_true(not hit.is_hit())


def test_tlas_triangle_two_instance_cases() raises:
    var verts = _make_one_local_triangle_z2[Frame.LOCAL]()

    var first_blas = TriangleBvh[Frame.LOCAL, 4](verts.copy())
    var second_blas = TriangleBvh[Frame.LOCAL, 4](verts^)

    var blases = [first_blas.copy(), second_blas.copy()]

    # Near/far along z: nearest should win.
    var near_far_instances = [
        _triangle_instance[4](0, 0.0, 0.0, 0.0, first_blas),
        _triangle_instance[4](1, 0.0, 0.0, 6.0, second_blas),
    ]

    var near_far_tlas = Tlas[4](near_far_instances)

    var center_ray = Rayf32[Frame.WORLD](
        Point3W(0.0, 0.0, 0.0),
        Vec3W(0.0, 0.0, 1.0),
    )
    _assert_hit(
        near_far_tlas.trace[TriangleBvh[Frame.LOCAL, 4], TRACE.CLOSEST_HIT](
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

    var right_ray = Rayf32[Frame.WORLD](
        Point3W(5.0, 0.0, 0.0), Vec3W(0.0, 0.0, 1.0)
    )
    _assert_hit(
        left_right_tlas.trace[TriangleBvh[Frame.LOCAL, 4], TRACE.CLOSEST_HIT](
            right_ray,
            blases.unsafe_ptr(),
        ),
        1,
        0,
        2.0,
    )


def test_tlas_triangle_shadow_cases() raises:
    var verts = _make_one_local_triangle_z2[Frame.LOCAL]()

    var blas = TriangleBvh[Frame.LOCAL, 4](verts^)

    var blases = [blas.copy()]

    var instances = [_triangle_instance[4](0, 5.0, 0.0, 0.0, blas)]

    var tlas = Tlas[4](instances)

    var ray_hit = Rayf32[Frame.WORLD](
        Point3W(5.0, 0.0, 0.0), Vec3W(0.0, 0.0, 1.0)
    )
    assert_true(
        tlas.trace[TriangleBvh[Frame.LOCAL, 4], TRACE.ANY_HIT](
            ray_hit,
            blases.unsafe_ptr(),
        ).is_occluded()
    )

    var ray_miss = Rayf32[Frame.WORLD](
        Point3W(0.0, 0.0, 0.0),
        Vec3W(0.0, 0.0, 1.0),
    )
    assert_true(
        not tlas.trace[TriangleBvh[Frame.LOCAL, 4], TRACE.ANY_HIT](
            ray_miss,
            blases.unsafe_ptr(),
        ).is_occluded()
    )


# -----------------------------------------------------------------------------
# Sphere TLAS
# -----------------------------------------------------------------------------
def test_tlas_sphere_single_instance_cases() raises:
    var spheres = _make_one_local_sphere_z2[Frame.LOCAL]()

    var blas = SphereBvh[Frame.LOCAL, 4](spheres^)

    var blases = [blas.copy()]

    # Identity hit.
    var identity_instances = [_sphere_instance[4](0, 0.0, 0.0, 0.0, blas)]

    var identity_tlas = Tlas[4](identity_instances)

    var identity_ray = Rayf32[Frame.WORLD](
        Point3W(0.0, 0.0, 0.0), Vec3W(0.0, 0.0, 1.0)
    )
    var identity_hit = identity_tlas.trace[
        SphereBvh[Frame.LOCAL, 4], TRACE.CLOSEST_HIT
    ](
        identity_ray,
        blases.unsafe_ptr(),
    )
    _assert_hit(identity_hit, 0, 0, 1.0)
    assert_almost_equal(identity_hit.normal.x, 0.0)
    assert_almost_equal(identity_hit.normal.y, 0.0)
    assert_almost_equal(identity_hit.normal.z, -1.0)

    # Translated hit.
    var translated_instances = [_sphere_instance[4](0, 5.0, 0.0, 0.0, blas)]
    var translated_tlas = Tlas[4](translated_instances)

    var translated_hit_ray = Rayf32[Frame.WORLD](
        Point3W(5.0, 0.0, 0.0),
        Vec3W(0.0, 0.0, 1.0),
    )
    _assert_hit(
        translated_tlas.trace[SphereBvh[Frame.LOCAL, 4], TRACE.CLOSEST_HIT](
            translated_hit_ray,
            blases.unsafe_ptr(),
        ),
        0,
        0,
        1.0,
    )

    # Translated miss.
    var translated_miss_ray = Rayf32[Frame.WORLD](
        Point3W(0.0, 0.0, 0.0), Vec3W(0.0, 0.0, 1.0)
    )

    var hit = translated_tlas.trace[
        SphereBvh[Frame.LOCAL, 4], TRACE.CLOSEST_HIT
    ](
        translated_miss_ray,
        blases.unsafe_ptr(),
    )
    assert_true(not hit.is_hit())


def test_tlas_sphere_two_instance_cases() raises:
    var spheres = _make_one_local_sphere_z2[Frame.LOCAL]()

    var first_blas = SphereBvh[Frame.LOCAL, 4](spheres.copy())
    var second_blas = SphereBvh[Frame.LOCAL, 4](spheres^)

    var blases = [first_blas.copy(), second_blas.copy()]

    # Near/far along z: nearest should win.
    var near_far_instances = [
        _sphere_instance[4](0, 0.0, 0.0, 0.0, first_blas),
        _sphere_instance[4](1, 0.0, 0.0, 6.0, second_blas),
    ]
    var near_far_tlas = Tlas[4](near_far_instances)

    var center_ray = Rayf32[Frame.WORLD](
        Point3W(0.0, 0.0, 0.0), Vec3W(0.0, 0.0, 1.0)
    )
    _assert_hit(
        near_far_tlas.trace[SphereBvh[Frame.LOCAL, 4], TRACE.CLOSEST_HIT](
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

    var right_ray = Rayf32[Frame.WORLD](
        Point3W(5.0, 0.0, 0.0), Vec3W(0.0, 0.0, 1.0)
    )
    _assert_hit(
        left_right_tlas.trace[SphereBvh[Frame.LOCAL, 4], TRACE.CLOSEST_HIT](
            right_ray,
            blases.unsafe_ptr(),
        ),
        1,
        0,
        1.0,
    )


def test_tlas_sphere_nonuniform_scale_normal() raises:
    var spheres = _make_one_local_sphere_z2[Frame.LOCAL]()
    var blas = SphereBvh[Frame.LOCAL, 4](spheres^)
    var blases = [blas.copy()]

    var transform = Affine3f32[Frame.LOCAL, Frame.WORLD].from_scale(
        Vec3f32[Frame.LOCAL](2.0, 1.0, 1.0)
    )
    var instances = [
        Instance(
            transform,
            UInt32(0),
            blas.bounds(),
            Primitive.SPHERE,
        )
    ]
    var tlas = Tlas[4](instances)
    var ray = Rayf32[Frame.WORLD](Point3W(1.0, 0.0, 0.0), Vec3W(0.0, 0.0, 1.0))
    var hit = tlas.trace[SphereBvh[Frame.LOCAL, 4], TRACE.CLOSEST_HIT](
        ray, blases.unsafe_ptr()
    )

    _assert_hit(hit, 0, 0, 1.1339746)
    assert_almost_equal(hit.normal.x, 0.2773501, atol=1.0e-5)
    assert_almost_equal(hit.normal.y, 0.0, atol=1.0e-5)
    assert_almost_equal(hit.normal.z, -0.9607689, atol=1.0e-5)


def test_tlas_sphere_shadow_cases() raises:
    var spheres = _make_one_local_sphere_z2[Frame.LOCAL]()

    var blas = SphereBvh[Frame.LOCAL, 4](spheres^)

    var blases = [blas.copy()]

    var instances = [_sphere_instance[4](0, 5.0, 0.0, 0.0, blas)]

    var tlas = Tlas[4](instances)

    var ray_hit = Rayf32[Frame.WORLD](
        Point3W(5.0, 0.0, 0.0), Vec3W(0.0, 0.0, 1.0)
    )
    assert_true(
        tlas.trace[SphereBvh[Frame.LOCAL, 4], TRACE.ANY_HIT](
            ray_hit, blases.unsafe_ptr()
        ).is_occluded()
    )

    var ray_miss = Rayf32[Frame.WORLD](
        Point3W(0.0, 0.0, 0.0), Vec3W(0.0, 0.0, 1.0)
    )
    assert_true(
        not tlas.trace[SphereBvh[Frame.LOCAL, 4], TRACE.ANY_HIT](
            ray_miss,
            blases.unsafe_ptr(),
        ).is_occluded()
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
