from std.testing import TestSuite, assert_true, assert_almost_equal

from bajo.bvh.constants import TRACE, Primitive
from bajo.bvh.types import Ray, Instance, Sphere
from bajo.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.bvh.cpu.sphere_bvh import SphereBvh
from bajo.bvh.cpu.tlas import Tlas
from bajo.core.transform import Affine3f32
from bajo.core.vec import Vec3f32


def _make_one_local_triangle_z2() -> List[Vec3f32]:
    var verts = List[Vec3f32](capacity=3)

    verts.append(Vec3f32(-1.0, -1.0, 2.0))
    verts.append(Vec3f32(1.0, -1.0, 2.0))
    verts.append(Vec3f32(0.0, 1.0, 2.0))

    return verts^


def _make_one_local_sphere_z2() -> List[Sphere]:
    return [Sphere(Vec3f32(0.0, 0.0, 2.0), 1.0)]


def _triangle_instance[
    width: Int
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
    width: Int
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
        Primitive.TRIANGLE,
    )


# -----------------------------------------------------------------------------
# Triangle TLAS
# -----------------------------------------------------------------------------
def test_tlas_triangle_identity_instance_hit() raises:
    var verts = _make_one_local_triangle_z2()

    var blas = TriangleBvh[4](
        verts.unsafe_ptr(),
        UInt32(len(verts) / 3),
    )

    var blases = List[TriangleBvh[4]](capacity=1)
    blases.append(blas.copy())

    var instances = List[Instance](capacity=1)
    instances.append(_triangle_instance[4](0, 0.0, 0.0, 0.0, blas))

    var tlas = Tlas[4](instances)

    var ray = Ray(Vec3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    var hit = tlas.trace[TriangleBvh[4], TRACE.CLOSEST_HIT](
        ray, blases.unsafe_ptr()
    )

    assert_true(hit.inst == 0)
    assert_true(hit.prim == 0)
    assert_almost_equal(hit.t, 2.0)


def test_tlas_translated_triangle_instance_hit() raises:
    var verts = _make_one_local_triangle_z2()

    var blas = TriangleBvh[4](
        verts.unsafe_ptr(),
        UInt32(len(verts) / 3),
    )

    var blases = List[TriangleBvh[4]](capacity=1)
    blases.append(blas.copy())

    var instances = List[Instance](capacity=1)
    instances.append(_triangle_instance[4](0, 5.0, 0.0, 0.0, blas))

    var tlas = Tlas[4](instances)

    # World ray at x = 5 becomes local x = 0 after inverse transform.
    var ray = Ray(Vec3f32(5.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    var hit = tlas.trace[TriangleBvh[4], TRACE.CLOSEST_HIT](
        ray, blases.unsafe_ptr()
    )

    assert_true(hit.inst == 0)
    assert_true(hit.prim == 0)
    assert_almost_equal(hit.t, 2.0)


def test_tlas_translated_triangle_instance_miss() raises:
    var verts = _make_one_local_triangle_z2()

    var blas = TriangleBvh[4](
        verts.unsafe_ptr(),
        UInt32(len(verts) / 3),
    )

    var blases = List[TriangleBvh[4]](capacity=1)
    blases.append(blas.copy())

    var instances = List[Instance](capacity=1)
    instances.append(_triangle_instance[4](0, 5.0, 0.0, 0.0, blas))

    var tlas = Tlas[4](instances)

    var ray = Ray(Vec3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    var hit = tlas.trace[TriangleBvh[4], TRACE.CLOSEST_HIT](
        ray, blases.unsafe_ptr()
    )

    assert_true(not hit.is_hit())


def test_tlas_translated_triangle_two_instances_nearest_wins() raises:
    var verts = _make_one_local_triangle_z2()

    var near_blas = TriangleBvh[4](
        verts.unsafe_ptr(),
        UInt32(len(verts) / 3),
    )
    var far_blas = TriangleBvh[4](
        verts.unsafe_ptr(),
        UInt32(len(verts) / 3),
    )

    var blases = List[TriangleBvh[4]](capacity=2)
    blases.append(near_blas.copy())
    blases.append(far_blas.copy())

    var instances = List[Instance](capacity=2)
    instances.append(_triangle_instance[4](0, 0.0, 0.0, 0.0, near_blas))
    instances.append(_triangle_instance[4](1, 0.0, 0.0, 6.0, far_blas))

    var tlas = Tlas[4](instances)

    var ray = Ray(Vec3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    var hit = tlas.trace[TriangleBvh[4], TRACE.CLOSEST_HIT](
        ray, blases.unsafe_ptr()
    )

    assert_true(hit.inst == 0)
    assert_true(hit.prim == 0)
    assert_almost_equal(hit.t, 2.0)


def test_tlas_triangle_two_instances_far_wins_when_ray_targets_far() raises:
    var verts = _make_one_local_triangle_z2()

    var left_blas = TriangleBvh[4](
        verts.unsafe_ptr(),
        UInt32(len(verts) / 3),
    )
    var right_blas = TriangleBvh[4](
        verts.unsafe_ptr(),
        UInt32(len(verts) / 3),
    )

    var blases = List[TriangleBvh[4]](capacity=2)
    blases.append(left_blas.copy())
    blases.append(right_blas.copy())

    var instances = List[Instance](capacity=2)
    instances.append(_triangle_instance[4](0, -5.0, 0.0, 0.0, left_blas))
    instances.append(_triangle_instance[4](1, 5.0, 0.0, 0.0, right_blas))

    var tlas = Tlas[4](instances)

    var ray = Ray(Vec3f32(5.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    var hit = tlas.trace[TriangleBvh[4], TRACE.CLOSEST_HIT](
        ray, blases.unsafe_ptr()
    )

    assert_true(hit.inst == 1)
    assert_true(hit.prim == 0)
    assert_almost_equal(hit.t, 2.0)


def test_tlas_translated_triangle_shadow_hit_and_miss() raises:
    var verts = _make_one_local_triangle_z2()

    var blas = TriangleBvh[4](
        verts.unsafe_ptr(),
        UInt32(len(verts) / 3),
    )

    var blases = List[TriangleBvh[4]](capacity=1)
    blases.append(blas.copy())

    var instances = List[Instance](capacity=1)
    instances.append(_triangle_instance[4](0, 5.0, 0.0, 0.0, blas))

    var tlas = Tlas[4](instances)

    var ray_hit = Ray(Vec3f32(5.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    assert_true(
        tlas.trace[TriangleBvh[4], TRACE.ANY_HIT](
            ray_hit, blases.unsafe_ptr()
        ).is_occluded()
    )

    var ray_miss = Ray(Vec3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    assert_true(
        not tlas.trace[TriangleBvh[4], TRACE.ANY_HIT](
            ray_miss, blases.unsafe_ptr()
        ).is_occluded()
    )


# -----------------------------------------------------------------------------
# Sphere TLAS
# -----------------------------------------------------------------------------
def test_tlas_sphere_identity_instance_hit() raises:
    var spheres = _make_one_local_sphere_z2()

    var blas = SphereBvh[4](
        spheres.unsafe_ptr(),
        UInt32(len(spheres)),
    )

    var blases = List[SphereBvh[4]](capacity=1)
    blases.append(blas.copy())

    var instances = List[Instance](capacity=1)
    instances.append(_sphere_instance[4](0, 0.0, 0.0, 0.0, blas))

    var tlas = Tlas[4](instances)

    var ray = Ray(Vec3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    var hit = tlas.trace[SphereBvh[4], TRACE.CLOSEST_HIT](
        ray, blases.unsafe_ptr()
    )

    assert_true(hit.inst == 0)
    assert_true(hit.prim == 0)
    assert_almost_equal(hit.t, 1.0)


def test_tlas_translated_sphere_instance_hit() raises:
    var spheres = _make_one_local_sphere_z2()

    var blas = SphereBvh[4](
        spheres.unsafe_ptr(),
        UInt32(len(spheres)),
    )

    var blases = List[SphereBvh[4]](capacity=1)
    blases.append(blas.copy())

    var instances = List[Instance](capacity=1)
    instances.append(_sphere_instance[4](0, 5.0, 0.0, 0.0, blas))

    var tlas = Tlas[4](instances)

    var ray = Ray(Vec3f32(5.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    var hit = tlas.trace[SphereBvh[4], TRACE.CLOSEST_HIT](
        ray, blases.unsafe_ptr()
    )

    assert_true(hit.inst == 0)
    assert_true(hit.prim == 0)
    assert_almost_equal(hit.t, 1.0)


def test_tlas_translated_sphere_instance_miss() raises:
    var spheres = _make_one_local_sphere_z2()

    var blas = SphereBvh[4](
        spheres.unsafe_ptr(),
        UInt32(len(spheres)),
    )

    var blases = List[SphereBvh[4]](capacity=1)
    blases.append(blas.copy())

    var instances = List[Instance](capacity=1)
    instances.append(_sphere_instance[4](0, 5.0, 0.0, 0.0, blas))

    var tlas = Tlas[4](instances)

    var ray = Ray(Vec3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    var hit = tlas.trace[SphereBvh[4], TRACE.CLOSEST_HIT](
        ray, blases.unsafe_ptr()
    )

    assert_true(not hit.is_hit())


def test_tlas_translated_sphere_two_instances_nearest_wins() raises:
    var spheres = _make_one_local_sphere_z2()

    var near_blas = SphereBvh[4](
        spheres.unsafe_ptr(),
        UInt32(len(spheres)),
    )
    var far_blas = SphereBvh[4](
        spheres.unsafe_ptr(),
        UInt32(len(spheres)),
    )

    var blases = List[SphereBvh[4]](capacity=2)
    blases.append(near_blas.copy())
    blases.append(far_blas.copy())

    var instances = List[Instance](capacity=2)
    instances.append(_sphere_instance[4](0, 0.0, 0.0, 0.0, near_blas))
    instances.append(_sphere_instance[4](1, 0.0, 0.0, 6.0, far_blas))

    var tlas = Tlas[4](instances)

    var ray = Ray(Vec3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    var hit = tlas.trace[SphereBvh[4], TRACE.CLOSEST_HIT](
        ray, blases.unsafe_ptr()
    )

    assert_true(hit.inst == 0)
    assert_true(hit.prim == 0)
    assert_almost_equal(hit.t, 1.0)


def test_cpu_tlas_sphere_two_instances_far_wins_when_ray_targets_far() raises:
    var spheres = _make_one_local_sphere_z2()

    var left_blas = SphereBvh[4](
        spheres.unsafe_ptr(),
        UInt32(len(spheres)),
    )
    var right_blas = SphereBvh[4](
        spheres.unsafe_ptr(),
        UInt32(len(spheres)),
    )

    var blases = List[SphereBvh[4]](capacity=2)
    blases.append(left_blas.copy())
    blases.append(right_blas.copy())

    var instances = List[Instance](capacity=2)
    instances.append(_sphere_instance[4](0, -5.0, 0.0, 0.0, left_blas))
    instances.append(_sphere_instance[4](1, 5.0, 0.0, 0.0, right_blas))

    var tlas = Tlas[4](instances)

    var ray = Ray(Vec3f32(5.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    var hit = tlas.trace[SphereBvh[4], TRACE.CLOSEST_HIT](
        ray, blases.unsafe_ptr()
    )

    assert_true(hit.inst == 1)
    assert_true(hit.prim == 0)
    assert_almost_equal(hit.t, 1.0)


def test_tlas_translated_sphere_shadow_hit_and_miss() raises:
    var spheres = _make_one_local_sphere_z2()

    var blas = SphereBvh[4](
        spheres.unsafe_ptr(),
        UInt32(len(spheres)),
    )

    var blases = List[SphereBvh[4]](capacity=1)
    blases.append(blas.copy())

    var instances = List[Instance](capacity=1)
    instances.append(_sphere_instance[4](0, 5.0, 0.0, 0.0, blas))

    var tlas = Tlas[4](instances)

    var ray_hit = Ray(Vec3f32(5.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    assert_true(
        tlas.trace[SphereBvh[4], TRACE.ANY_HIT](
            ray_hit, blases.unsafe_ptr()
        ).is_occluded()
    )

    var ray_miss = Ray(Vec3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
    assert_true(
        not tlas.trace[SphereBvh[4], TRACE.ANY_HIT](
            ray_miss, blases.unsafe_ptr()
        ).is_occluded()
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
