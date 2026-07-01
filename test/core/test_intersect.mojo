from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
    assert_false,
)

from bajo.core.intersect import (
    closest_point_to_aabb,
    closest_point_to_triangle,
    furthest_point_to_triangle,
    intersect_ray_aabb,
    intersect_ray_aabb_rcp,
    intersect_aabb_aabb,
    no_div_tri_tri_isect,
    intersect_tri_tri,
)

from bajo.core.vec import Vec3, assert_vec_equal


# AABB
def test_closest_point_to_aabb() raises:
    comptime dtype = DType.float64
    lower = Vec3(0.0)
    upper = Vec3(10.0)

    # Point inside (should return itself)
    p_in = Vec3(5.0)
    c_in = closest_point_to_aabb(p_in, lower, upper)
    expected_c_in = Vec3(5.0)
    assert_vec_equal(c_in, expected_c_in)

    # Point outside (should clamp to surface/corner)
    p_out = Vec3(-5.0, 15.0, 5.0)
    c_out = closest_point_to_aabb(p_out, lower, upper)
    expected_c_out = Vec3(0.0, 10.0, 5.0)
    assert_vec_equal(c_out, expected_c_out)


def test_intersect_aabb_aabb() raises:
    a_lower = Vec3(0.0)
    a_upper = Vec3(10.0)

    # Overlapping AABB
    b_lower = Vec3(5.0)
    b_upper = Vec3(15.0)
    assert_true(intersect_aabb_aabb(a_lower, a_upper, b_lower, b_upper))

    # Completely Separated AABB
    c_lower = Vec3(20.0)
    c_upper = Vec3(30.0)
    assert_false(intersect_aabb_aabb(a_lower, a_upper, c_lower, c_upper))


def test_intersect_ray_aabb() raises:
    lower = Vec3(-1.0)
    upper = Vec3(1.0)

    # Ray hitting the AABB head on
    pos_hit = Vec3(0.01, 0.01, 5.0)
    dir_hit = Vec3(0.0, 0.0, -1.0)
    # Prevent Inf*0 NaNs for exact 0
    rcp_dir_hit = Vec3(1e9, 1e9, 1.0 / dir_hit.z)

    res = intersect_ray_aabb(pos_hit, rcp_dir_hit, lower, upper, 1e30)
    assert_true(res.mask)
    # Starts at z=5, hits z=1, distance is 4
    assert_almost_equal(res.t, 4.0, atol=1e-4)

    # Ray missing the AABB
    pos_miss = Vec3([5.0, 5.0, 5.0])
    dir_miss = Vec3([0.0, 1.0, 0.0])
    rcp_dir_miss = Vec3(1e9, 1.0 / dir_miss.y, 1e9)
    res = intersect_ray_aabb(pos_miss, rcp_dir_miss, lower, upper, 1e30)
    assert_false(res.mask)


def test_intersect_ray_aabb_rcp_name_and_inside_hit() raises:
    lower = Vec3(-1.0)
    upper = Vec3(1.0)

    pos_inside = Vec3(0.0, 0.0, 0.0)
    rcp_dir = Vec3(1.0, 1e9, 1e9)

    res = intersect_ray_aabb_rcp(pos_inside, rcp_dir, lower, upper, 1e30)
    assert_true(res.mask)
    assert_almost_equal(res.t, 0.0, atol=1e-4)


def test_intersect_ray_aabb_parallel_outside_miss() raises:
    lower = Vec3(-1.0)
    upper = Vec3(1.0)

    pos = Vec3(2.0, 0.0, 0.0)
    rcp_dir = Vec3(1e9, 1.0, 1e9)

    res = intersect_ray_aabb_rcp(pos, rcp_dir, lower, upper, 1e30)
    assert_false(res.mask)


# Point / Triangle Distance Tests
def test_closest_point_to_triangle() raises:
    a = Vec3(0.0, 0.0, 0.0)
    b = Vec3(10.0, 0.0, 0.0)
    c = Vec3(0.0, 10.0, 0.0)

    # 1. Point directly above the face
    p_face = Vec3(2.0, 2.0, 5.0)
    uv_face = closest_point_to_triangle(a, b, c, p_face)
    # Reconstruct point: P = u*A + v*B + (1-u-v)*C
    # Should project to (2.0, 2.0, 0.0). Since A is origin, u is weight of A.
    # Function returns Vec2(u, v) where weight(A)=u, weight(B)=v, weight(C)=1-u-v.
    # So 0.6*0 + 0.2*10 + 0.2*0 = 2.0 (X). -> v = 0.2
    # 0.6*0 + 0.2*0 + 0.2*10 = 2.0 (Y). -> 1-u-v = 0.2 -> u = 0.6
    assert_almost_equal(uv_face.x, 0.6, atol=1e-4)
    assert_almost_equal(uv_face.y, 0.2, atol=1e-4)

    # 2. Point far past Vertex B
    p_vert = Vec3(15.0, -2.0, 0.0)
    uv_vert = closest_point_to_triangle(a, b, c, p_vert)
    # Expected to clamp to Vertex B. (u=0, v=1)
    assert_almost_equal(uv_vert.x, 0.0, atol=1e-4)
    assert_almost_equal(uv_vert.y, 1.0, atol=1e-4)


def test_furthest_point_to_triangle() raises:
    a = Vec3(0.0, 0.0, 0.0)
    b = Vec3(10.0, 0.0, 0.0)
    c = Vec3(0.0, 10.0, 0.0)

    # Point far in -X direction. Vertex B (+10 X) should be the furthest.
    p_far = Vec3(-100.0, 0.0, 0.0)
    uv_far = furthest_point_to_triangle(a, b, c, p_far)

    # Vertex B is returned as (0, 1)
    assert_equal(uv_far.x, 0.0)
    assert_equal(uv_far.y, 1.0)


# Ray / Triangle Intersection Tests


# Triangle / Triangle Intersection Tests
def _test_no_div_tri_tri_isect[T: DType]() raises:
    # Base triangle on XY plane
    t1_0 = Vec3[T](0.0, 0.0, 0.0)
    t1_1 = Vec3[T](2.0, 0.0, 0.0)
    t1_2 = Vec3[T](0.0, 2.0, 0.0)

    # 1. Intersecting Triangle (pierces through the middle)
    t2_0 = Vec3[T](0.5, 0.5, -1.0)
    t2_1 = Vec3[T](0.5, 0.5, 1.0)
    t2_2 = Vec3[T](2.0, 2.0, 0.0)
    assert_true(no_div_tri_tri_isect(t1_0, t1_1, t1_2, t2_0, t2_1, t2_2))

    # 2. Separated Triangle (High above on Z axis)
    t3_0 = Vec3[T](0.0, 0.0, 5.0)
    t3_1 = Vec3[T](2.0, 0.0, 5.0)
    t3_2 = Vec3[T](0.0, 2.0, 5.0)
    assert_false(no_div_tri_tri_isect(t1_0, t1_1, t1_2, t3_0, t3_1, t3_2))

    # 3. Coplanar Intersecting (On same plane, overlapping)
    t4_0 = Vec3[T](1.0, 1.0, 0.0)
    t4_1 = Vec3[T](3.0, 1.0, 0.0)
    t4_2 = Vec3[T](1.0, 3.0, 0.0)
    assert_true(no_div_tri_tri_isect(t1_0, t1_1, t1_2, t4_0, t4_1, t4_2))


def test_no_div_tri_tri_isect() raises:
    _test_no_div_tri_tri_isect[DType.float16]()
    _test_no_div_tri_tri_isect[DType.float32]()
    _test_no_div_tri_tri_isect[DType.float64]()


def _test_intersect_tri_tri[T: DType]() raises:
    """From warp/warp/tests/test_intersect.py."""

    v0 = Vec3[T](0.0, 0.0, 0.0)
    v1 = Vec3[T](1.0, 0.0, 0.0)
    v2 = Vec3[T](0.0, 0.0, 1.0)
    u0 = Vec3[T](0.5, -0.5, 0.0)
    u1 = Vec3[T](0.5, -0.5, 1.0)
    u2 = Vec3[T](0.5, 0.5, 0.0)

    assert_true(intersect_tri_tri(v0, v1, v2, u0, u1, u2))

    v0 = Vec3[T](0.0, 0.0, 0.0)
    v1 = Vec3[T](1.0, 0.0, 0.0)
    v2 = Vec3[T](0.0, 0.0, 1.0)
    u0 = Vec3[T](-0.5, -0.5, 0.0)
    u1 = Vec3[T](-0.5, -0.5, 1.0)
    u2 = Vec3[T](-0.5, 0.5, 0.0)

    assert_false(intersect_tri_tri(v0, v1, v2, u0, u1, u2))


def test_intersect_tri_tri() raises:
    """From warp/warp/tests/test_intersect.py."""

    _test_intersect_tri_tri[DType.float16]()
    _test_intersect_tri_tri[DType.float32]()
    _test_intersect_tri_tri[DType.float64]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
