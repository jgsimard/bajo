from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_true,
    assert_false,
)

from bajo.core import (
    Vec3W,
    dot,
    cross,
    length,
    normalize,
    assert_vec_equal,
)


def test_length_normalize() raises:
    v = Vec3W(3, 0, 0)
    assert_almost_equal(length(v), 3.0)

    n = normalize(v)
    assert_vec_equal(n, Vec3W(1, 0, 0))


def test_dot() raises:
    a = Vec3W(1, 2, 3)
    b = Vec3W(4, 5, 6)
    assert_almost_equal(dot(a, b), 32.0)


def test_vec3_add_cross() raises:
    v1 = Vec3W(1, 2, 3)
    v2 = Vec3W(4, 5, 6)

    assert_vec_equal(v1 + v2, Vec3W(5, 7, 9))

    # 1,2,3 x 4,5,6 = -3, 6, -3
    assert_vec_equal(cross(v1, v2), Vec3W(-3, 6, -3))


def test_near_zero() raises:
    assert_true(Vec3W(1e-9).is_near_zero())
    assert_true(Vec3W(0).is_near_zero())

    assert_false(Vec3W(0.1).is_near_zero())
    assert_false(Vec3W(1e-9, 1e-9, 0.1).is_near_zero())


def test_safe_inv_zero_and_nonzero_components() raises:
    inv = Vec3W(2.0, 0.0, -4.0).safe_inv()

    assert_almost_equal(inv.x, 0.5)
    assert_almost_equal(inv.y, 0.0)
    assert_almost_equal(inv.z, -0.25)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
