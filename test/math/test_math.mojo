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
    # Lane 0: x^2 - 5x + 6 = 0 -> roots 2, 3
    # Lane 1: x^2 + 0x + 1 = 0 -> no real roots
    comptime T = SIMD[DType.float32, 2]
    a = T(1.0, 1.0)
    b = T(-5.0, 0.0)
    c = T(6.0, 1.0)

    res = solve_quadratic(a, b, c)

    # Lane 0
    assert_true(res.mask[0])
    assert_almost_equal(res.roots_0[0], 2.0)
    assert_almost_equal(res.roots_1[0], 3.0)

    # Lane 1
    assert_false(res.mask[1])


fn test_vector_math() raises:
    v = Vec3f(3, 0, 0)
    assert_almost_equal(length(v), 3.0)

    n = normalize(v)
    assert_vec_equal(n, Vec3f(1, 0, 0))

    a = Vec3f(1, 2, 3)
    b = Vec3f(4, 5, 6)
    assert_almost_equal(dot(a, b), 32.0)

    # Test Lerp
    mid = lerp(Vec3f(1, 2, 3), Vec3f(11, 12, 13), 0.5)
    assert_vec_equal(mid, Vec3f(6, 7, 8))


fn test_matrix_basics() raises:
    # Transpose check
    m = Mat3f(Vec3f(1, 2, 3), Vec3f(4, 5, 6), Vec3f(7, 8, 9))
    mt = m.transpose()
    assert_almost_equal(mt[0].y(), 4.0)
    assert_almost_equal(mt[1].x(), 2.0)

    # Determinant of Identity
    identity = Mat3f(Vec3f(1, 0, 0), Vec3f(0, 1, 0), Vec3f(0, 0, 1))
    assert_almost_equal(identity.determinant(), 1.0)


fn test_quaternion_rotation() raises:
    # Rotate (1, 0, 0) 90 degrees around Y axis -> should be (0, 0, -1)
    angle = degrees_to_radians(Float32(90))
    q = Quat.angle_axis(angle, Vec3f(0, 1, 0))
    v = Vec3f(1, 0, 0)
    rotated = q.rotate_vec(v)
    assert_vec_equal(rotated, Vec3f(0, 0, -1))


fn test_trs_composition_decomposition() raises:
    t = Vec3f(10, 20, 30)

    angle = degrees_to_radians(Float32(45))
    r = Quat.angle_axis(angle, Vec3f(0, 0, 1))
    s = Diag3f(2, 2, 2)

    # Create Mat3x4 from TRS
    mat = Mat3x4f.from_trs(t, r, s)

    # Transform a point
    p = Vec3f(1, 0, 0)
    p_tx = mat.txfm_point(p)
    # Expected: Scaled(2,0,0) -> Rotated 45deg on Z(1.414, 1.414, 0) -> Translated(11.414, 21.414, 30)
    assert_almost_equal(p_tx.x(), 11.41421, atol=1e-4)
    assert_almost_equal(p_tx.y(), 21.41421, atol=1e-4)

    # Decompose back
    decomposed = mat.decompose()
    dec_t = decomposed[0]
    dec_s = decomposed[2]

    assert_vec_equal(dec_t, t)
    assert_almost_equal(dec_s.d0, 2.0)


fn test_quaternions() raises:
    q1 = Quat.angle_axis(0, Vec3f(0, 1, 0))
    assert_quat_equal(q1, Quat(1, 0, 0, 0))

    angle = degrees_to_radians(Float32(45))
    q2 = Quat.angle_axis(angle, Vec3f(0, 1, 0))
    assert_quat_equal(q2, Quat(0.9238795, 0, 0.3826834, 0))

    q3 = Quat.angle_axis(angle, Vec3f(1, 0, 0))
    assert_quat_equal(q3, Quat(0.9238795, 0.3826834, 0, 0))

    m1 = q2 * q3
    assert_quat_equal(m1, Quat(0.853553, 0.353553, 0.353553, -0.146447))


fn test_vec3_operations() raises:
    v1 = Vec3f(1, 2, 3)
    v2 = Vec3f(4, 5, 6)

    assert_vec_equal(v1 + v2, Vec3f(5, 7, 9))

    # 1,2,3 x 4,5,6 = -3, 6, -3
    assert_vec_equal(cross(v1, v2), Vec3f(-3, 6, -3))


fn test_aabb_logic() raises:
    box = AABB(Vec3f(0), Vec3f(2, 2, 2))

    assert_true(box.contains(Vec3f(1)))
    assert_false(box.contains(Vec3f(3, 1, 1)))

    box_overlap = AABB(Vec3f(1), Vec3f(3))
    box_far = AABB(Vec3f(4), Vec3f(5))
    assert_true(box.overlaps(box_overlap))
    assert_false(box.overlaps(box_far))

    # Ray Intersect
    ray_o = Vec3f(-1, 1, 1)
    ray_d = Vec3f(1, 0, 0)
    # Diag3.inv handles the 1/d calculation used in ray_intersects
    inv_d = Diag3f.from_vec(ray_d).inv()

    assert_true(box.ray_intersects(ray_o, inv_d, 0.0, 10.0))


fn test_aabb_apply_trs_rotated() raises:
    box = AABB(Vec3f(-1), Vec3f(1))

    # Rotate 45 degrees around Z
    angle = degrees_to_radians(Float32(45))
    r = Quat.angle_axis(angle, Vec3f(0, 0, 1))
    t = Vec3f(0)
    s = Diag3f(1, 1, 1)

    new_box = box.apply_trs(t, r, s)

    # After 45 deg rotation, the corners move to ±sqrt(2)
    # The new AABB should expand to fit the diamond shape
    expected_limit = Float32(1.41421356)
    assert_almost_equal(new_box.max.x(), expected_limit, atol=1e-5)
    assert_almost_equal(new_box.max.y(), expected_limit, atol=1e-5)
    # Z should remain 1.0
    assert_almost_equal(new_box.max.z(), 1.0, atol=1e-5)

fn test_quat_multiplication() raises:
    # Rotate 90 X then 90 Y
    angle = degrees_to_radians(Float32(90))
    qx = Quat.angle_axis(angle, Vec3f(1, 0, 0))
    qy = Quat.angle_axis(angle, Vec3f(0, 1, 0))

    q_combined = qy * qx  # Note: Order matters (Local vs World)

    # Rotate a vector (0, 0, 1)
    # 1. Rotate 90 around X -> (0, -1, 0)
    # 2. Rotate 90 around Y -> (0, -1, 0) ... Y doesn't affect it
    v = Vec3f(0, 0, 1)
    result = q_combined.rotate_vec(v)
    assert_vec_equal(result, Vec3f(1, 0, 0))


fn test_swizzles() raises:
    v = Vector[DType.float32, 4](1, 2, 3, 4)
    assert_vec_equal(v.xy(), Vec2f(1, 2))
    assert_vec_equal(v.yz(), Vec2f(2, 3))
    assert_vec_equal(v.xyz(), Vec3f(1, 2, 3))

    # Test indexing
    v[0] = 10
    assert_almost_equal(v.x(), 10.0)


fn test_matrix_composition() raises:
    parent_mat = Mat3x4f.from_trs(
        Vec3f(10, 0, 0), Quat.identity(), Diag3f(1, 1, 1)
    )
    child_mat = Mat3x4f.from_trs(
        Vec3f(5, 0, 0), Quat.identity(), Diag3f(1, 1, 1)
    )

    combined = parent_mat.compose(child_mat)

    p = Vec3f(0, 0, 0)
    result = combined.txfm_point(p)

    # Should be at 15 (10 + 5)
    assert_almost_equal(result.x(), 15.0)


fn test_matrix_roundtrip() raises:
    angle = degrees_to_radians(Float32(30))
    r = Quat.angle_axis(angle, Vec3f(1, 0, 0))
    m = Mat3f.from_quat(r)
    # Rotating a vector by the matrix should match rotating by the quaternion
    v = Vec3f(0, 1, 0)
    assert_vec_equal(m * v, r.rotate_vec(v))


fn test_vector_near_zero() raises:
    v_tiny = Vec3f(1e-9)
    v_zero = Vec3f(0)
    assert_true(v_tiny.near_zero())
    assert_true(v_zero.near_zero())

    v_large = Vec3f(0.1)
    assert_false(v_large.near_zero())


fn test_cross_product_2d() raises:
    a = Vec2f(1, 0)
    b = Vec2f(0, 1)
    # The cross product of unit X and unit Y in 2D is 1.0
    assert_almost_equal(cross(a, b), 1.0)


fn test_aabb_merge() raises:
    a = AABB(Vec3f(0), Vec3f(1))
    b = AABB(Vec3f(-1), Vec3f(0.5))
    merged = AABB.merge(a, b)
    assert_vec_equal(merged.min, Vec3f(-1))
    assert_vec_equal(merged.max, Vec3f(1))


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
