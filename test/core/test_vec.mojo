from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_true,
    assert_false,
)

from bajo.core.vec import (
    Vec2f32,
    Vec3f32,
    dot,
    cross,
    length,
    normalize,
    assert_vec_equal,
)


def test_length_normalize() raises:
    v = Vec3f32(3, 0, 0)
    assert_almost_equal(length(v), 3.0)

    n = normalize(v)
    assert_vec_equal(n, Vec3f32(1, 0, 0))


def test_dot() raises:
    a = Vec3f32(1, 2, 3)
    b = Vec3f32(4, 5, 6)
    assert_almost_equal(dot(a, b), 32.0)


def test_vec3_add_cross() raises:
    v1 = Vec3f32(1, 2, 3)
    v2 = Vec3f32(4, 5, 6)

    assert_vec_equal(v1 + v2, Vec3f32(5, 7, 9))

    # 1,2,3 x 4,5,6 = -3, 6, -3
    assert_vec_equal(cross(v1, v2), Vec3f32(-3, 6, -3))


def test_near_zero() raises:
    assert_true(Vec3f32(1e-9).is_near_zero())
    assert_true(Vec3f32(0).is_near_zero())

    assert_false(Vec3f32(0.1).is_near_zero())
    assert_false(Vec3f32(1e-9, 1e-9, 0.1).is_near_zero())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
