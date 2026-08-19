from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_true,
    assert_false,
)

from bajo.core import (
    Frame,
    Vec3,
    Vec3W,
    dot,
    cross,
    length,
    normalize,
    assert_vec_equal,
)


def test_select_uses_mask_per_simd_lane() raises:
    var if_true = Vec3[DType.float32, Frame.WORLD, 4](
        SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0),
        SIMD[DType.float32, 4](5.0, 6.0, 7.0, 8.0),
        SIMD[DType.float32, 4](9.0, 10.0, 11.0, 12.0),
    )
    var if_false = Vec3[DType.float32, Frame.WORLD, 4](-1.0)
    var selected = Vec3.select(
        SIMD[DType.bool, 4](True, False, True, False), if_true, if_false
    )

    assert_almost_equal(selected.x[0], 1.0)
    assert_almost_equal(selected.x[1], -1.0)
    assert_almost_equal(selected.y[2], 7.0)
    assert_almost_equal(selected.z[3], -1.0)


def test_length_normalize() raises:
    var v = Vec3W(3, 0, 0)
    assert_almost_equal(length(v), 3.0)

    var n = normalize(v)
    assert_vec_equal(n, Vec3W(1, 0, 0))


def test_dot() raises:
    var a = Vec3W(1, 2, 3)
    var b = Vec3W(4, 5, 6)
    assert_almost_equal(dot(a, b), 32.0)


def test_vec3_add_cross() raises:
    var v1 = Vec3W(1, 2, 3)
    var v2 = Vec3W(4, 5, 6)

    assert_vec_equal(v1 + v2, Vec3W(5, 7, 9))

    # 1,2,3 x 4,5,6 = -3, 6, -3
    assert_vec_equal(cross(v1, v2), Vec3W(-3, 6, -3))


def test_near_zero() raises:
    assert_true(Vec3W(1e-9).is_near_zero())
    assert_true(Vec3W(0).is_near_zero())

    assert_false(Vec3W(0.1).is_near_zero())
    assert_false(Vec3W(1e-9, 1e-9, 0.1).is_near_zero())


def test_safe_inv_zero_and_nonzero_components() raises:
    var inv = Vec3W(2.0, 0.0, -4.0).safe_inv()

    assert_almost_equal(inv.x, 0.5)
    assert_almost_equal(inv.y, 0.0)
    assert_almost_equal(inv.z, -0.25)


def test_load_store_span_with_nonzero_base() raises:
    var data = List[Float32](length=8, fill=-1.0)
    var value = Vec3W(2.0, 3.0, 4.0)

    value.store(data, 3)
    var loaded = Vec3W.load(data, 3)

    assert_vec_equal(loaded, value)
    assert_almost_equal(data[2], -1.0)
    assert_almost_equal(data[6], -1.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
