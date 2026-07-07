from std.testing import TestSuite, assert_almost_equal, assert_true

from bajo.core import (
    Affine3,
    Affine3f32,
    Vec3,
    Vec3f32,
    Point3,
    Point3f32,
    assert_vec_equal,
    Quat,
    Frame,
)
from bajo.core.utils import degrees_to_radians


def test_identity() raises:
    p_w = Point3f32[Frame.WORLD](1, 2, 3)
    v_w = Vec3f32[Frame.WORLD](4, 5, 6)
    p_c = Point3f32[Frame.CAMERA](1, 2, 3)
    v_c = Vec3f32[Frame.CAMERA](4, 5, 6)
    m = Affine3f32[Frame.WORLD, Frame.CAMERA].identity()

    assert_vec_equal(m.point(p_w), p_c)
    assert_vec_equal(m.vector(v_w), v_c)
    assert_vec_equal(m.translation(), Vec3f32[Frame.CAMERA](0, 0, 0))


def test_translation() raises:
    p = Point3f32(1, 2, 3)
    v = Vec3f32(4, 5, 6)
    t = Vec3f32(10, -2, 3.5)

    m = Affine3f32.from_translation(t)

    assert_vec_equal(m.point(p), Point3f32(11, 0, 6.5))
    assert_vec_equal(m.vector(v), v)
    assert_vec_equal(m.translation(), t)


def test_scale() raises:
    p = Point3f32(1, 2, 3)
    v = Vec3f32(4, 5, 6)
    s = Vec3f32(2, 3, 4)

    m = Affine3f32.from_scale(s)

    assert_vec_equal(m.point(p), Point3f32(2, 6, 12))
    assert_vec_equal(m.vector(v), Vec3f32(8, 15, 24))
    assert_vec_equal(m.translation(), Vec3f32(0, 0, 0))


def test_rotation_scale_from_quat() raises:
    axis = Vec3f32(1, 0, 0)
    angle = degrees_to_radians(Float32(30))
    q = Quat.from_axis_angle(axis, angle)
    s = Vec3f32(2, 3, 4)
    v = Vec3f32(1, 2, 3)
    p = Point3f32(1, 2, 3)
    m = Affine3f32.from_rotation_scale(q, s)

    expected = q.rotate(Vec3f32(v.x * s.x, v.y * s.y, v.z * s.z))

    assert_vec_equal(m.vector(v), expected)
    assert_vec_equal(m.point(p), expected.to_point())


def test_rotation_scale_translation_from_quat() raises:
    axis = Vec3f32(0, 1, 0)
    angle = degrees_to_radians(Float32(45))
    q = Quat.from_axis_angle(axis, angle)

    s = Vec3f32(2, 3, 4)
    t = Vec3f32(10, 20, 30)
    v = Vec3f32(1, 2, 3)
    p = Point3f32(1, 2, 3)

    m = Affine3f32.from_rotation_scale_translation(q, s, t)

    expected_v = q.rotate(Vec3f32(p.x * s.x, p.y * s.y, p.z * s.z))
    expected_p = expected_v + t

    assert_vec_equal(m.vector(v), expected_v)
    assert_vec_equal(m.point(p), expected_p.to_point())
    assert_vec_equal(m.translation(), t)


def test_width4_translation_and_scale() raises:
    comptime T = DType.float32
    comptime W = 4

    p = Point3[T, W](1.0, 2.0, 3.0)
    v = Vec3[T, W](1.0, 2.0, 3.0)
    t = Vec3[T, W](10.0, 20.0, 30.0)
    s = Vec3[T, W](2.0, 3.0, 4.0)

    mt = Affine3[T, W].from_translation(t)
    ms = Affine3[T, W].from_scale(s)

    assert_vec_equal(
        mt.point(p),
        Point3[T, W](11.0, 22.0, 33.0),
    )

    assert_vec_equal(mt.vector(v), v)

    assert_vec_equal(
        ms.point(p),
        Point3[T, W](2.0, 6.0, 12.0),
    )

    assert_vec_equal(
        ms.vector(v),
        Vec3[T, W](2.0, 6.0, 12.0),
    )


def test_load_store() raises:
    var ptr = alloc[Float32](12)
    # fmt: off
    m = Affine3f32(
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
    )
    # fmt: on

    v = Vec3f32(2, 3, 4)
    p = Point3f32(2, 3, 4)

    m.store(ptr, 0)
    loaded = Affine3f32.load(ptr, 0)

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

    ptr.free()


def test_load_transform_helpers() raises:
    # fmt: off
    var arr = [
        Float32(1), 2, 3, 4, 
        5, 6, 7, 8, 
        9, 10, 11, 12
    ]
    # fmt: on
    p = Point3f32(2, 3, 4)

    loaded = Affine3f32.load(arr.unsafe_ptr(), 0)

    # p = M * p_in + t
    assert_vec_equal(
        loaded.point(p),
        Point3f32(
            1 * 2 + 2 * 3 + 3 * 4 + 4,
            5 * 2 + 6 * 3 + 7 * 4 + 8,
            9 * 2 + 10 * 3 + 11 * 4 + 12,
        ),
    )

    # v = M * v_in
    v = Vec3f32(2, 3, 4)
    assert_vec_equal(
        loaded.vector(v),
        Vec3f32(
            1 * 2 + 2 * 3 + 3 * 4,
            5 * 2 + 6 * 3 + 7 * 4,
            9 * 2 + 10 * 3 + 11 * 4,
        ),
    )


def test_inverse_translation_scale() raises:
    s = Vec3f32(2, 4, 5)
    t = Vec3f32(10, 20, 30)
    p = Point3f32(1, 2, 3)

    m = Affine3f32.from_rotation_scale_translation(
        Quat.identity(),
        s,
        t,
    )

    res = m.inverse()

    assert_true(res.mask[0])

    p2 = m.point(p)
    assert_vec_equal(res.inv.point(p2), p)


def test_inverse_rotation_scale_translation() raises:
    axis = Vec3f32(0, 1, 0)
    angle = degrees_to_radians(Float32(45))
    q = Quat.from_axis_angle(axis, angle)

    s = Vec3f32(2, 3, 4)
    t = Vec3f32(10, 20, 30)
    p = Point3f32(1, 2, 3)

    m = Affine3f32.from_rotation_scale_translation(q, s, t)
    res = m.inverse()

    assert_true(res.mask[0])

    p2 = m.point(p)
    assert_vec_equal(res.inv.point(p2), p, atol=1e-4)


def test_inverse_singular_scale() raises:
    m = Affine3f32.from_scale(Vec3f32(1, 0, 1))
    res = m.inverse()
    assert_true(not res.mask[0])


def test_inverse_identity() raises:
    m = Affine3f32.identity()
    res = m.inverse()

    assert_true(res.mask[0])

    p = Point3f32(1, 2, 3)
    v = Vec3f32(4, 5, 6)

    assert_vec_equal(res.inv.point(p), p)
    assert_vec_equal(res.inv.vector(v), v)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
