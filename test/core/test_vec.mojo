from test.helpers import assert_vec_equal

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_true,
    assert_false,
)

from bajo.core import (
    Vec3,
    Vec3W,
    Point3f32,
    dot,
    cross,
    length,
    normalize,
)


def test_select_uses_mask_per_simd_lane() raises:
    var if_true = Vec3[.float32, .WORLD, 4](
        SIMD[.float32, 4](1.0, 2.0, 3.0, 4.0),
        SIMD[.float32, 4](5.0, 6.0, 7.0, 8.0),
        SIMD[.float32, 4](9.0, 10.0, 11.0, 12.0),
    )
    var if_false = Vec3[.float32, .WORLD, 4](-1.0)
    var selected = Vec3.select(
        SIMD[.bool, 4](True, False, True, False), if_true, if_false
    )

    assert_almost_equal(selected.x[0], 1.0)
    assert_almost_equal(selected.x[1], -1.0)
    assert_almost_equal(selected.y[2], 7.0)
    assert_almost_equal(selected.z[3], -1.0)


def test_component_abs_and_clamp_preserve_packet_lanes() raises:
    var value = Vec3[.float32, .WORLD, 4](
        SIMD[.float32, 4](-2.0, -0.5, 0.5, 3.0),
        SIMD[.float32, 4](4.0, -3.0, 2.0, -1.0),
        SIMD[.float32, 4](-0.25, 0.25, -5.0, 5.0),
    )
    assert_vec_equal(
        value.abs(),
        Vec3[.float32, .WORLD, 4](
            SIMD[.float32, 4](2.0, 0.5, 0.5, 3.0),
            SIMD[.float32, 4](4.0, 3.0, 2.0, 1.0),
            SIMD[.float32, 4](0.25, 0.25, 5.0, 5.0),
        ),
    )
    var lower = SIMD[.float32, 4](-1.0, 0.0, 0.0, 1.0)
    var upper = SIMD[.float32, 4](1.0, 1.0, 2.0, 2.0)
    assert_vec_equal(
        value.clamp(lower, upper),
        Vec3[.float32, .WORLD, 4](
            SIMD[.float32, 4](-1.0, 0.0, 0.5, 2.0),
            SIMD[.float32, 4](1.0, 0.0, 2.0, 1.0),
            SIMD[.float32, 4](-0.25, 0.25, 0.0, 2.0),
        ),
    )


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
    var pointer_loaded = Vec3W(data.unsafe_ptr().unsafe_offset(3))
    var point = Point3f32[.LOCAL](data.unsafe_ptr().unsafe_offset(3))

    assert_vec_equal(loaded, value)
    assert_vec_equal(pointer_loaded, value)
    assert_almost_equal(point.x, 2.0)
    assert_almost_equal(point.y, 3.0)
    assert_almost_equal(point.z, 4.0)
    assert_almost_equal(data[2], -1.0)
    assert_almost_equal(data[6], -1.0)


def test_compiletime_indexed_assignment_and_setter_compatibility() raises:
    var value = Vec3[.float32, .WORLD, 4](0.0)
    comptime for axis in range(3):
        value[axis] = SIMD[.float32, 4](Float32(axis + 1))
    value[1] += SIMD[.float32, 4](2.0)
    assert_vec_equal(value, Vec3[.float32, .WORLD, 4](1.0, 4.0, 3.0))
    value.__setitem__[0](SIMD[.float32, 4](5.0))
    value.set_axis[2](SIMD[.float32, 4](6.0))
    assert_vec_equal(value, Vec3[.float32, .WORLD, 4](5.0, 4.0, 6.0))
    var axis = 1
    value[axis] = SIMD[.float32, 4](7.0)
    assert_vec_equal(value, Vec3[.float32, .WORLD, 4](5.0, 7.0, 6.0))


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
