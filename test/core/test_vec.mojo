from testing import (
    TestSuite,
    assert_almost_equal,
    assert_true,
    assert_false,
)

from bajo.core.vec import (
    Vec2f32,
    Vec3f32,
    Vec4f32,
    dot,
    cross,
    length,
    normalize,
    lerp,
    assert_vec_equal,
)


fn test_length_normalize() raises:
    v = Vec3f32(3, 0, 0)
    assert_almost_equal(length(v), 3.0)

    n = normalize(v)
    assert_vec_equal(n, Vec3f32(1, 0, 0))


fn test_dot() raises:
    a = Vec3f32(1, 2, 3)
    b = Vec3f32(4, 5, 6)
    assert_almost_equal(dot(a, b), 32.0)


fn test_lerp() raises:
    mid = lerp(Vec3f32(1, 2, 3), Vec3f32(11, 12, 13), 0.5)
    assert_vec_equal(mid, Vec3f32(6, 7, 8))


fn test_vec3_add_cross() raises:
    v1 = Vec3f32(1, 2, 3)
    v2 = Vec3f32(4, 5, 6)

    assert_vec_equal(v1 + v2, Vec3f32(5, 7, 9))

    # 1,2,3 x 4,5,6 = -3, 6, -3
    assert_vec_equal(cross(v1, v2), Vec3f32(-3, 6, -3))


fn test_cross_2d() raises:
    a = Vec2f32(1, 0)
    b = Vec2f32(0, 1)
    # The cross product of unit X and unit Y in 2D is 1.0
    assert_almost_equal(cross(a, b), 1.0)


fn test_swizzles() raises:
    v = Vec4f32(1, 2, 3, 4)
    assert_vec_equal(v.xy(), Vec2f32(1, 2))
    assert_vec_equal(v.yz(), Vec2f32(2, 3))
    assert_vec_equal(v.xyz(), Vec3f32(1, 2, 3))

    # Test indexing
    v[0] = 10
    assert_almost_equal(v.x(), 10.0)


fn test_near_zero() raises:
    v_tiny = Vec3f32(1e-9)
    v_zero = Vec3f32(0)
    assert_true(v_tiny.near_zero())
    assert_true(v_zero.near_zero())

    v_large = Vec3f32(0.1)
    assert_false(v_large.near_zero())


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
