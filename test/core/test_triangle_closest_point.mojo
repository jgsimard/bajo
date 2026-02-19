from testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_true,
    assert_false,
)

from math import abs

from bajo.core.vec import Vec3, Vec3f32, min as vmin, max as vmax, dot, length

from bajo.core.math import PhiloxRNG


@always_inline
fn triangle_closest_point_barycentric[
    dtype: DType
](a: Vec3[dtype], b: Vec3[dtype], c: Vec3[dtype], p: Vec3[dtype]) -> Vec3[
    dtype
]:
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = dot(ab, ap)
    d2 = dot(ac, ap)

    if d1 <= 0.0 and d2 <= 0.0:
        return Vec3[dtype](1.0, 0.0, 0.0)

    bp = p - b
    d3 = dot(ab, bp)
    d4 = dot(ac, bp)

    if d3 >= 0.0 and d4 <= d3:
        return Vec3[dtype](0.0, 1.0, 0.0)

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return Vec3[dtype](1.0 - v, v, 0.0)

    cp = p - c
    d5 = dot(ab, cp)
    d6 = dot(ac, cp)

    if d6 >= 0.0 and d5 <= d6:
        return Vec3[dtype](0.0, 0.0, 1.0)

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return Vec3[dtype](1.0 - w, 0.0, w)

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return Vec3[dtype](0.0, 1.0 - w, w)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom

    return Vec3[dtype](1.0 - v - w, v, w)


@always_inline
fn check_edge_feasible_region[
    dtype: DType
](
    p: Vec3[dtype],
    a: Vec3[dtype],
    b: Vec3[dtype],
    c: Vec3[dtype],
    eps: Scalar[dtype],
) -> Bool:
    ap = p - a
    bp = p - b
    ab = b - a

    if dot(ap, ab) < -eps:
        return False
    if dot(bp, ab) > eps:
        return False

    ab_sqr_norm = dot(ab, ab)
    if ab_sqr_norm < eps:
        return False

    t = dot(ab, c - a) / ab_sqr_norm
    perpendicular_foot = a + ab * t

    if dot(c - perpendicular_foot, p - perpendicular_foot) > eps:
        return False

    return True


@always_inline
fn check_vertex_feasible_region[
    dtype: DType
](
    p: Vec3[dtype],
    a: Vec3[dtype],
    b: Vec3[dtype],
    c: Vec3[dtype],
    eps: Scalar[dtype],
) -> Bool:
    ap = p - a
    ba = a - b
    ca = a - c

    if dot(ap, ba) < -eps:
        return False
    # Note: The Warp test had a potential typo (wp.dot(p, ca)),
    # fixed here to match the logic of vertex region checking (ap dot ca)
    if dot(ap, ca) < -eps:
        return False

    return True


fn test_triangle_closest_point_f32() raises:
    _test_triangle_closest_point[DType.float32, 1e-5]()


fn test_triangle_closest_point_f16() raises:
    _test_triangle_closest_point[DType.float16, 1e-3]()


fn _test_triangle_closest_point[dtype: DType, eps: Scalar[dtype]]() raises:
    rng = PhiloxRNG(seed=123, id=321)
    # Setup triangle
    a = Vec3[dtype](1.0, 0.0, 0.0)
    b = Vec3[dtype](0.0, 0.0, 0.0)
    c = Vec3[dtype](0.0, 1.0, 0.0)

    tri = InlineArray[Vec3[dtype], 3](uninitialized=True)
    tri[0] = a.copy()
    tri[1] = b.copy()
    tri[2] = c.copy()

    for i in range(100):
        # Generate random point on sphere r=2
        l = Scalar[dtype](0.0)
        p = Vec3[dtype](0, 0, 0)

        while l < eps:
            r1 = Scalar[dtype](rng.next_f32())
            r2 = Scalar[dtype](rng.next_f32())
            r3 = Scalar[dtype](rng.next_f32())
            p = Vec3[dtype](r1, r2, r3)
            l = length(p)

        p = p * (2.0 / l)

        bary = triangle_closest_point_barycentric(a, b, c, p)

        for dim in range(3):
            v1_idx = (dim + 1) % 3
            v2_idx = (dim + 2) % 3

            v1 = tri[v1_idx].copy()
            v2 = tri[v2_idx].copy()
            v3 = tri[dim].copy()

            # Case: Closest point is on an edge
            if bary[dim] == 0.0 and bary[v1_idx] != 0.0 and bary[v2_idx] != 0.0:
                if not check_edge_feasible_region(p, v1, v2, v3, eps):
                    raise "Failed edge feasible region at iteration {}".format(
                        i
                    )

                # Check perpendicularity
                closest_p = a * bary[0] + b * bary[1] + c * bary[2]
                e = v1 - v2
                err = dot(e, closest_p - p)
                if abs(err) > eps:
                    raise "Failed perpendicularity at iteration".format(i)

            # Case: Closest point is a vertex
            if bary[v1_idx] == 0.0 and bary[v2_idx] == 0.0:
                if not check_vertex_feasible_region(p, v3, v1, v2, eps):
                    raise "Failed vertex feasible region at iteration".format(i)

            # Case: Closest point is inside the face
            if bary[dim] != 0.0 and bary[v1_idx] != 0.0 and bary[v2_idx] != 0.0:
                closest_p = a * bary[0] + b * bary[1] + c * bary[2]
                e1 = v1 - v2
                e2 = v1 - v3
                if (
                    abs(dot(e1, closest_p - p)) > eps
                    or abs(dot(e2, closest_p - p)) > eps
                ):
                    raise "Failed face perpendicularity at iteration".format(i)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
