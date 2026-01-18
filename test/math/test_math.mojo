from testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_true,
    assert_false,
)
from bajo.bmath import (
    Quat,
    Vector,
    Vec3f,
    Vec2f,
    Diag3f,
    Mat3f,
    Mat3x4f,
    AABB,
    degrees_to_radians,
    cross,
    dot,
    length,
    normalize,
    lerp,
    solve_quadratic,
    QuadraticSolutions,
)


# Helpers
fn assert_quat_equal(a: Quat, b: Quat) raises:
    assert_almost_equal(a.data, b.data, atol=1e-6)


fn assert_vec_equal[
    s: Int
](a: Vector[DType.float32, s], b: Vector[DType.float32, s]) raises:
    assert_almost_equal(a.data, b.data, atol=1e-5)


# Tests
fn test_solve_quadratic() raises:
    # x^2 - 5x + 6 = 0 -> roots 2, 3
    var a = SIMD[DType.float32, 2](1.0, 1.0)
    var b = SIMD[DType.float32, 2](
        -5.0, 0.0
    )  # Lane 0: has roots, Lane 1: no roots
    var c = SIMD[DType.float32, 2](6.0, 1.0)

    var res = solve_quadratic(a, b, c)
    var t0 = res.roots_0
    var t1 = res.roots_1
    var mask = res.mask

    # Lane 0 should be solved
    assert_true(mask[0])
    var root_a = t0[0]
    var root_b = t1[0]
    assert_almost_equal(min(root_a, root_b), 2.0)
    assert_almost_equal(max(root_a, root_b), 3.0)

    # Lane 1 (x^2 + 1 = 0) should not be solved
    assert_false(mask[1])


fn test_vector_math() raises:
    var v = Vec3f(3, 0, 0)
    assert_almost_equal(length(v), 3.0)

    var n = normalize(v)
    assert_vec_equal(n, Vec3f(1, 0, 0))

    var a = Vec3f(1, 2, 3)
    var b = Vec3f(4, 5, 6)
    assert_almost_equal(dot(a, b), 32.0)

    # Test Lerp
    var mid = lerp(Vec3f(0, 0, 0), Vec3f(10, 10, 10), 0.5)
    assert_vec_equal(mid, Vec3f(5, 5, 5))


fn test_matrix_basics() raises:
    # Transpose check
    var m = Mat3f(Vec3f(1, 2, 3), Vec3f(4, 5, 6), Vec3f(7, 8, 9))
    var mt = m.transpose()
    assert_almost_equal(mt[0].y(), 4.0)
    assert_almost_equal(mt[1].x(), 2.0)

    # Determinant of Identity
    var identity = Mat3f(Vec3f(1, 0, 0), Vec3f(0, 1, 0), Vec3f(0, 0, 1))
    assert_almost_equal(identity.determinant(), 1.0)


fn test_quaternion_rotation() raises:
    # Rotate (1, 0, 0) 90 degrees around Y axis -> should be (0, 0, -1)
    var angle = degrees_to_radians(Scalar[DType.float32](90))
    var q = Quat.angle_axis(angle, Vec3f(0, 1, 0))
    var v = Vec3f(1, 0, 0)
    var rotated = q.rotate_vec(v)
    assert_vec_equal(rotated, Vec3f(0, 0, -1))


fn test_trs_composition_decomposition() raises:
    var t = Vec3f(10, 20, 30)
    var r = Quat.angle_axis(
        degrees_to_radians(Scalar[DType.float32](45)), Vec3f(0, 0, 1)
    )
    var s = Diag3f(2, 2, 2)

    # Create Mat3x4 from TRS
    var mat = Mat3x4f.from_trs(t, r, s)

    # Transform a point
    var p = Vec3f(1, 0, 0)
    var p_tx = mat.txfm_point(p)
    # Expected: Scaled(2,0,0) -> Rotated 45deg on Z(1.414, 1.414, 0) -> Translated(11.414, 21.414, 30)
    assert_almost_equal(p_tx.x(), 11.41421, atol=1e-4)
    assert_almost_equal(p_tx.y(), 21.41421, atol=1e-4)

    # Decompose back
    var decomposed = mat.decompose()
    var dec_t = decomposed[0]
    var dec_s = decomposed[2]

    assert_vec_equal(dec_t, t)
    assert_almost_equal(dec_s.d0, 2.0)


fn test_quaternions() raises:
    var q1 = Quat.angle_axis(0, Vec3f(0, 1, 0))
    assert_quat_equal(q1, Quat(1, 0, 0, 0))

    var q2 = Quat.angle_axis(
        degrees_to_radians(Scalar[DType.float32](45)), Vec3f(0, 1, 0)
    )
    assert_quat_equal(q2, Quat(0.9238795, 0, 0.3826834, 0))

    var q3 = Quat.angle_axis(
        degrees_to_radians(Scalar[DType.float32](45)), Vec3f(1, 0, 0)
    )
    assert_quat_equal(q3, Quat(0.9238795, 0.3826834, 0, 0))

    var m1 = q2 * q3
    assert_quat_equal(m1, Quat(0.853553, 0.353553, 0.353553, -0.146447))


fn test_vec3_operations() raises:
    var v1 = Vec3f(1, 2, 3)
    var v2 = Vec3f(4, 5, 6)

    assert_vec_equal(v1 + v2, Vec3f(5, 7, 9))

    # 1,2,3 x 4,5,6 = -3, 6, -3
    assert_vec_equal(cross(v1, v2), Vec3f(-3, 6, -3))


fn test_aabb_logic() raises:
    var box = AABB(Vec3f(0), Vec3f(2, 2, 2))

    assert_true(box.contains(Vec3f(1)))
    assert_false(box.contains(Vec3f(3, 1, 1)))

    var box_overlap = AABB(Vec3f(1), Vec3f(3))
    var box_far = AABB(Vec3f(4), Vec3f(5))
    assert_true(box.overlaps(box_overlap))
    assert_false(box.overlaps(box_far))

    # Ray Intersect
    var ray_o = Vec3f(-1, 1, 1)
    var ray_d = Vec3f(1, 0, 0)
    # Diag3.inv handles the 1/d calculation used in ray_intersects
    var inv_d = Diag3f.from_vec(ray_d).inv()

    assert_true(box.ray_intersects(ray_o, inv_d, 0.0, 10.0))


fn test_aabb_apply_trs_rotated() raises:
    var box = AABB(Vec3f(-1), Vec3f(1))

    # Rotate 45 degrees around Z
    var angle = degrees_to_radians(Scalar[DType.float32](45))
    var r = Quat.angle_axis(angle, Vec3f(0, 0, 1))
    var t = Vec3f(0)
    var s = Diag3f(1, 1, 1)

    var new_box = box.apply_trs(t, r, s)

    # After 45 deg rotation, the corners move to ±sqrt(2)
    # The new AABB should expand to fit the diamond shape
    var expected_limit = Float32(1.41421356)
    assert_almost_equal(new_box.pMax.x(), expected_limit, atol=1e-5)
    assert_almost_equal(new_box.pMax.y(), expected_limit, atol=1e-5)
    # Z should remain 1.0
    assert_almost_equal(new_box.pMax.z(), 1.0, atol=1e-5)

    fn test_quat_multiplication() raises:
        # Rotate 90 X then 90 Y
        var qx = Quat.angle_axis(
            degrees_to_radians(Scalar[DType.float32](90)), Vec3f(1, 0, 0)
        )
        var qy = Quat.angle_axis(
            degrees_to_radians(Scalar[DType.float32](90)), Vec3f(0, 1, 0)
        )

        var q_combined = qy * qx  # Note: Order matters (Local vs World)

        # Rotate a vector (0, 0, 1)
        # 1. Rotate 90 around X -> (0, -1, 0)
        # 2. Rotate 90 around Y -> (0, -1, 0) ... Y doesn't affect it
        var v = Vec3f(0, 0, 1)
        var result = q_combined.rotate_vec(v)
        assert_vec_equal(result, Vec3f(1, 0, 0))


fn test_swizzles() raises:
    var v = Vector[DType.float32, 4](1, 2, 3, 4)
    assert_vec_equal(v.xy(), Vec2f(1, 2))
    assert_vec_equal(v.yz(), Vec2f(2, 3))
    assert_vec_equal(v.xyz(), Vec3f(1, 2, 3))

    # Test indexing
    v[0] = 10
    assert_almost_equal(v.x(), 10.0)


fn test_matrix_composition() raises:
    var parent_mat = Mat3x4f.from_trs(
        Vec3f(10, 0, 0), Quat.identity(), Diag3f(1, 1, 1)
    )
    var child_mat = Mat3x4f.from_trs(
        Vec3f(5, 0, 0), Quat.identity(), Diag3f(1, 1, 1)
    )

    var combined = parent_mat.compose(child_mat)

    var p = Vec3f(0, 0, 0)
    var result = combined.txfm_point(p)

    # Should be at 15 (10 + 5)
    assert_almost_equal(result.x(), 15.0)


fn test_matrix_roundtrip() raises:
    var r = Quat.angle_axis(
        degrees_to_radians(Scalar[DType.float32](30)), Vec3f(1, 0, 0)
    )
    var m = Mat3f.from_quat(r)
    # Rotating a vector by the matrix should match rotating by the quaternion
    var v = Vec3f(0, 1, 0)
    assert_vec_equal(m * v, r.rotate_vec(v))


fn test_vector_near_zero() raises:
    var v_tiny = Vec3f(1e-9)
    var v_zero = Vec3f(0)
    assert_true(v_tiny.near_zero())
    assert_true(v_zero.near_zero())

    var v_large = Vec3f(0.1)
    assert_false(v_large.near_zero())


fn test_cross_product_2d() raises:
    var a = Vec2f(1, 0)
    var b = Vec2f(0, 1)
    # The cross product of unit X and unit Y in 2D is 1.0
    assert_almost_equal(cross(a, b), 1.0)


fn test_aabb_merge() raises:
    var a = AABB(Vec3f(0), Vec3f(1))
    var b = AABB(Vec3f(-1), Vec3f(0.5))
    var merged = AABB.merge(a, b)
    assert_vec_equal(merged.pMin, Vec3f(-1))
    assert_vec_equal(merged.pMax, Vec3f(1))


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
