from std.testing import TestSuite, assert_almost_equal, assert_true
from std.memory.alloc import alloc, dealloc, Layout

from bajo.core import (
    Affine3,
    Affine3f32,
    Vec3,
    Vec3f32,
    Vec3W,
    Point3,
    Point3f32,
    Point3W,
    Normal3f32,
    Ray,
    assert_vec_equal,
    normalize,
    Quat,
    Frame,
    GeoKind,
)
from bajo.core.utils import degrees_to_radians


def test_identity() raises:
    var p_w = Point3f32[.WORLD](1, 2, 3)
    var v_w = Vec3f32[.WORLD](4, 5, 6)
    var p_c = Point3f32[.CAMERA](1, 2, 3)
    var v_c = Vec3f32[.CAMERA](4, 5, 6)
    var m = Affine3f32[.WORLD, .CAMERA].identity()

    assert_vec_equal(m.point(p_w), p_c)
    assert_vec_equal(m.vector(v_w), v_c)
    assert_vec_equal(m.translation(), Vec3f32[.CAMERA](0, 0, 0))


def test_translation() raises:
    var p = Point3W(1, 2, 3)
    var v = Vec3W(4, 5, 6)
    var t = Vec3W(10, -2, 3.5)
    var tc = Vec3f32[.CAMERA](10, -2, 3.5)
    var m = Affine3f32[.WORLD, .CAMERA].from_translation(t)

    assert_vec_equal(m.point(p), Point3f32[.CAMERA](11, 0, 6.5))
    assert_vec_equal(m.vector(v), Vec3f32[.CAMERA](4, 5, 6))
    assert_vec_equal(m.translation(), tc)


def test_scale() raises:
    var p = Point3W(1, 2, 3)
    var v = Vec3W(4, 5, 6)
    var s = Vec3W(2, 3, 4)
    var m = Affine3f32[.WORLD, .CAMERA].from_scale(s)

    assert_vec_equal(m.point(p), Point3f32[.CAMERA](2, 6, 12))
    assert_vec_equal(m.vector(v), Vec3f32[.CAMERA](8, 15, 24))
    assert_vec_equal(m.translation(), Vec3f32[.CAMERA](0, 0, 0))


def test_rotation_scale_from_quat() raises:
    var axis = Vec3W(1, 0, 0)
    var angle = degrees_to_radians(Float32(30))
    var q = Quat.from_axis_angle(axis, angle)
    var s = Vec3W(2, 3, 4)
    var v = Vec3W(1, 2, 3)
    var p = Point3W(1, 2, 3)
    var m = Affine3f32[.WORLD, .CAMERA].from_rotation_scale(q, s)

    var expected = q.rotate(
        Vec3W(v.x * s.x, v.y * s.y, v.z * s.z)
    ).unsafe_convert[new_frame=.CAMERA]()

    assert_vec_equal(m.vector(v), expected)
    assert_vec_equal(
        m.point(p), expected.unsafe_convert[new_kind=GeoKind.POINT]()
    )


def test_rotation_scale_translation_from_quat() raises:
    var axis = Vec3W(0, 1, 0)
    var angle = degrees_to_radians(Float32(45))
    var q = Quat.from_axis_angle(axis, angle)

    var s = Vec3W(2, 3, 4)
    var t = Vec3W(10, 20, 30)
    var v = Vec3W(1, 2, 3)
    var p = Point3W(1, 2, 3)

    var m = Affine3f32[.WORLD, .CAMERA].from_rotation_scale_translation(q, s, t)

    var expected_v = q.rotate(Vec3W(p.x * s.x, p.y * s.y, p.z * s.z))
    var expected_p = (expected_v + t).unsafe_convert[
        new_frame=.CAMERA, new_kind=GeoKind.POINT
    ]()

    assert_vec_equal(
        m.vector(v), expected_v.unsafe_convert[new_frame=.CAMERA]()
    )
    assert_vec_equal(m.point(p), expected_p)
    assert_vec_equal(m.translation(), t.unsafe_convert[new_frame=.CAMERA]())


def test_width4_translation_and_scale() raises:
    comptime T = DType.float32
    comptime W = 4
    comptime From = Frame.WORLD
    comptime To = Frame.CAMERA

    var p = Point3[T, From, W](1.0, 2.0, 3.0)
    var v = Vec3[T, From, W](1.0, 2.0, 3.0)
    var t = Vec3[T, From, W](10.0, 20.0, 30.0)
    var s = Vec3[T, From, W](2.0, 3.0, 4.0)

    var mt = Affine3[T, From, To, W].from_translation(t)
    var ms = Affine3[T, From, To, W].from_scale(s)

    assert_vec_equal(
        mt.point(p),
        Point3[T, To, W](11.0, 22.0, 33.0),
    )

    assert_vec_equal(mt.vector(v), Vec3[T, To, W](1.0, 2.0, 3.0))

    assert_vec_equal(
        ms.point(p),
        Point3[T, To, W](2.0, 6.0, 12.0),
    )

    assert_vec_equal(
        ms.vector(v),
        Vec3[T, To, W](2.0, 6.0, 12.0),
    )


def test_normal_uses_inverse_transpose_for_nonuniform_scale() raises:
    var transform = Affine3f32[.WORLD, .CAMERA].from_scale(Vec3W(2.0, 3.0, 4.0))
    var inverse = Affine3f32[.CAMERA, .WORLD].from_scale(
        Vec3f32[.CAMERA](0.5, 1.0 / 3.0, 0.25)
    )
    var normal = Normal3f32[.WORLD](1.0, 1.0, 0.0)
    var expected = normalize(
        Vec3f32[.CAMERA](0.5, 1.0 / 3.0, 0.0)
    ).unsafe_convert[new_kind=GeoKind.NORMAL]()

    assert_vec_equal(transform.normal(normal, inverse), expected)


def test_load_store() raises:
    var data = List[Float32](length=12, fill=0.0)
    # fmt: off
    var m = Affine3f32[.WORLD, .CAMERA](
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
    )
    # fmt: on

    var v = Vec3W(2, 3, 4)
    var p = Point3W(2, 3, 4)

    m.store(data, 0)
    var loaded = Affine3f32[.WORLD, .CAMERA].load(data, 0)

    assert_vec_equal(loaded.point(p), m.point(p))
    assert_vec_equal(loaded.vector(v), m.vector(v))

    assert_almost_equal(loaded.m00[0], 1.0)
    assert_almost_equal(loaded.m01[0], 2.0)
    assert_almost_equal(loaded.m02[0], 3.0)
    assert_almost_equal(loaded.tx[0], 4.0)

    assert_almost_equal(loaded.m10[0], 5.0)
    assert_almost_equal(loaded.m11[0], 6.0)
    assert_almost_equal(loaded.m12[0], 7.0)
    assert_almost_equal(loaded.ty[0], 8.0)

    assert_almost_equal(loaded.m20[0], 9.0)
    assert_almost_equal(loaded.m21[0], 10.0)
    assert_almost_equal(loaded.m22[0], 11.0)
    assert_almost_equal(loaded.tz[0], 12.0)


def test_load_transform_helpers() raises:
    # fmt: off
    var arr = [
        Float32(1), 2, 3, 4, 
        5, 6, 7, 8, 
        9, 10, 11, 12
    ]
    # fmt: on
    var p = Point3W(2, 3, 4)

    var loaded = Affine3f32[.WORLD, .CAMERA].load(arr, 0)

    # p = M * p_in + t
    assert_vec_equal(
        loaded.point(p),
        Point3f32[.CAMERA](
            1 * 2 + 2 * 3 + 3 * 4 + 4,
            5 * 2 + 6 * 3 + 7 * 4 + 8,
            9 * 2 + 10 * 3 + 11 * 4 + 12,
        ),
    )

    # v = M * v_in
    var v = Vec3W(2, 3, 4)
    assert_vec_equal(
        loaded.vector(v),
        Vec3f32[.CAMERA](
            1 * 2 + 2 * 3 + 3 * 4,
            5 * 2 + 6 * 3 + 7 * 4,
            9 * 2 + 10 * 3 + 11 * 4,
        ),
    )


def test_inverse_translation_scale() raises:
    var s = Vec3W(2, 4, 5)
    var t = Vec3W(10, 20, 30)
    var p = Point3W(1, 2, 3)

    var m = Affine3f32[.WORLD, .CAMERA].from_rotation_scale_translation(
        Quat[.WORLD].identity(),
        s,
        t,
    )

    var res = m.inverse()

    assert_true(res.mask[0])

    var p2 = m.point(p)
    assert_vec_equal(res.inv.point(p2), p)


def test_inverse_rotation_scale_translation() raises:
    var axis = Vec3W(0, 1, 0)
    var angle = degrees_to_radians(Float32(45))
    var q = Quat.from_axis_angle(axis, angle)

    var s = Vec3W(2, 3, 4)
    var t = Vec3W(10, 20, 30)
    var p = Point3W(1, 2, 3)

    var m = Affine3f32[.WORLD, .CAMERA].from_rotation_scale_translation(q, s, t)
    var res = m.inverse()

    assert_true(res.mask[0])

    var p2 = m.point(p)
    assert_vec_equal(res.inv.point(p2), p, atol=1e-4)


def test_inverse_singular_scale() raises:
    var m = Affine3f32[.WORLD, .CAMERA].from_scale(Vec3W(1, 0, 1))
    var res = m.inverse()
    assert_true(not res.mask[0])


def test_inverse_identity() raises:
    var m = Affine3f32[.WORLD, .CAMERA].identity()
    var res = m.inverse()

    assert_true(res.mask[0])

    var p = Point3f32[.CAMERA](1, 2, 3)
    var v = Vec3f32[.CAMERA](4, 5, 6)

    assert_vec_equal(res.inv.point(p), Point3f32[.WORLD](1, 2, 3))
    assert_vec_equal(res.inv.vector(v), Vec3f32[.WORLD](4, 5, 6))


def test_ray_transform() raises:
    var ray = Ray[.float64, .WORLD](
        Point3[DType.float64, .WORLD](1, 2, 3),
        Vec3[.float64, .WORLD](4, 5, 6),
        0.25,
        10,
    )
    # fmt: off
    var transform = Affine3[DType.float64, .WORLD, .CAMERA](
        2, 0, 0, 10,
        0, 3, 0, -2,
        0, 0, 4, 3,
    )
    # fmt: on
    var transformed = transform.ray(ray, 7)

    assert_vec_equal(transformed.o, Point3[DType.float64, .CAMERA](12, 4, 15))
    assert_vec_equal(transformed.d, Vec3[.float64, .CAMERA](8, 15, 24))
    assert_almost_equal(transformed.t_min, 0.25)
    assert_almost_equal(transformed.t_max, 7)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
