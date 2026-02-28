from math import fma, abs, max, clamp

from bajo.core.vec import (
    Vec2,
    Vec3,
    Vec3f32,
    min as vmin,
    max as vmax,
    dot,
    cross,
    length,
)


fn closest_point_to_aabb[
    dtype: DType
](p: Vec3[dtype], lower: Vec3[dtype], upper: Vec3[dtype]) -> Vec3[dtype]:
    c_array = InlineArray[Scalar[dtype], 3](uninitialized=True)
    v: Scalar[dtype]

    # X component
    v = p[0]
    if v < lower[0]:
        v = lower[0]
    if v > upper[0]:
        v = upper[0]
    c_array[0] = v

    # Y component
    v = p[1]
    if v < lower[1]:
        v = lower[1]
    if v > upper[1]:
        v = upper[1]
    c_array[1] = v

    # Z component
    v = p[2]
    if v < lower[2]:
        v = lower[2]
    if v > upper[2]:
        v = upper[2]
    c_array[2] = v

    return Vec3[dtype](c_array^)


fn closest_point_to_triangle[
    dtype: DType
](a: Vec3[dtype], b: Vec3[dtype], c: Vec3[dtype], p: Vec3[dtype]) -> Vec2[
    dtype
]:
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = dot(ab, ap)
    d2 = dot(ac, ap)

    if d1 <= 0 and d2 <= 0:
        # Vertex A: v=0, w=0, u=1
        return Vec2[dtype](1, 0)

    bp = p - b
    d3 = dot(ab, bp)
    d4 = dot(ac, bp)
    if d3 >= 0 and d4 <= d3:
        # Vertex B: v=1, w=0, u=0
        return Vec2[dtype](0, 1)

    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        # Edge AB
        v = d1 / (d1 - d3)
        u = 1 - v
        return Vec2[dtype](u, v)

    cp = p - c
    d5 = dot(ab, cp)
    d6 = dot(ac, cp)
    if d6 >= 0 and d5 <= d6:
        # Vertex C: v=0, w=1, u=0
        return Vec2[dtype](0, 0)

    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        # Edge AC
        w = d2 / (d2 - d6)
        u = 1 - w
        return Vec2[dtype](u, 0)

    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        # Edge BC
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        v = 1 - w
        return Vec2[dtype](0, v)

    # Inside Face
    denom = 1 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    u = 1 - v - w
    return Vec2[dtype](u, v)


fn furthest_point_to_triangle[
    dtype: DType
](a: Vec3[dtype], b: Vec3[dtype], c: Vec3[dtype], p: Vec3[dtype]) -> Vec2[
    dtype
]:
    pa = p - a
    pb = p - b
    pc = p - c

    dist_a = dot(pa, pa)
    dist_b = dot(pb, pb)
    dist_c = dot(pc, pc)

    # a is furthest
    if dist_a > dist_b and dist_a > dist_c:
        return Vec2[dtype](1.0, 0.0)

    # b is furthest
    if dist_b > dist_c:
        return Vec2[dtype](0.0, 1.0)

    #  c is furthest
    return Vec2[dtype](0.0, 0.0)


# TODO: use optinal ?
fn intersect_ray_aabb[
    dtype: DType
](
    pos: Vec3[dtype],
    rcp_dir: Vec3[dtype],
    lower: Vec3[dtype],
    upper: Vec3[dtype],
    mut t: Scalar[dtype],
) -> Bool:
    # X axis
    l1 = (lower[0] - pos[0]) * rcp_dir[0]
    l2 = (upper[0] - pos[0]) * rcp_dir[0]
    lmin = min(l1, l2)
    lmax = max(l1, l2)

    # Y axis
    l1 = (lower[1] - pos[1]) * rcp_dir[1]
    l2 = (upper[1] - pos[1]) * rcp_dir[1]
    lmin = max(min(l1, l2), lmin)
    lmax = min(max(l1, l2), lmax)

    # Z axis
    l1 = (lower[2] - pos[2]) * rcp_dir[2]
    l2 = (upper[2] - pos[2]) * rcp_dir[2]
    lmin = max(min(l1, l2), lmin)
    lmax = min(max(l1, l2), lmax)

    hit = (lmax >= 0.0) and (lmax >= lmin)
    if hit:
        t = lmin

    return hit


fn intersect_aabb_aabb[
    dtype: DType
](
    a_lower: Vec3[dtype],
    a_upper: Vec3[dtype],
    b_lower: Vec3[dtype],
    b_upper: Vec3[dtype],
) -> Bool:
    if (
        a_lower[0] > b_upper[0]
        or a_lower[1] > b_upper[1]
        or a_lower[2] > b_upper[2]
        or a_upper[0] < b_lower[0]
        or a_upper[1] < b_lower[1]
        or a_upper[2] < b_lower[2]
    ):
        return False
    else:
        return True


# Moller and Trumbore's method
@always_inline
fn intersect_ray_tri_moller[
    dtype: DType
](
    p: Vec3[dtype],
    dir: Vec3[dtype],
    a: Vec3[dtype],
    b: Vec3[dtype],
    c: Vec3[dtype],
    mut t: Scalar[dtype],
    mut u: Scalar[dtype],
    mut v: Scalar[dtype],
    mut w: Scalar[dtype],
    mut sign: Scalar[dtype],
    normal: UnsafePointer[Vec3[dtype], MutAnyOrigin],
) -> Bool:
    comptime assert dtype in [DType.float32, DType.float64]
    comptime EPSILON = Scalar[dtype](1e-8 if dtype == DType.float32 else 1e-16)

    ab = b - a
    ac = c - a
    n = cross(ab, ac)

    d = dot(-dir, n)

    if abs(d) < EPSILON:
        return False

    ood = 1.0 / d  # Infinity arithmetic handles d=0
    ap = p - a

    t = dot(ap, n) * ood
    if t < 0.0:
        return False

    e = cross(-dir, ap)
    v = dot(ac, e) * ood
    if v < 0.0 or v > 1.0:
        return False

    w = -dot(ab, e) * ood
    if w < 0.0 or (v + w) > 1.0:
        return False

    u = 1.0 - v - w

    if normal:
        normal[] = n.copy()

    sign = d

    return True


@always_inline
fn intersect_ray_tri_rtcd[
    dtype: DType
](
    p: Vec3[dtype],
    dir: Vec3[dtype],
    a: Vec3[dtype],
    b: Vec3[dtype],
    c: Vec3[dtype],
    mut t: Scalar[dtype],
    mut u: Scalar[dtype],
    mut v: Scalar[dtype],
    mut w: Scalar[dtype],
    mut sign: Scalar[dtype],
    normal: UnsafePointer[Vec3[dtype], MutAnyOrigin],
) -> Bool:
    comptime assert dtype in [DType.float32, DType.float64]
    comptime EPSILON = Scalar[dtype](1e-8 if dtype == DType.float32 else 1e-16)

    ab = b - a
    ac = c - a

    # calculate normal
    n = cross(ab, ac)

    # need to solve a system of three equations to give t, u, v
    d = dot(-dir, n)

    # if dir is parallel to triangle plane or points away from triangle
    if d <= 0.0:
        return False

    ap = p - a
    t = dot(ap, n)

    # ignores tris behind
    if t < 0.0:
        return False

    # compute barycentric coordinates
    e = cross(-dir, ap)
    v = dot(ac, e)
    if v < 0.0 or v > d:
        return False

    w = -dot(ab, e)
    if w < 0.0 or (v + w) > d:
        return False

    ood = 1.0 / d
    t *= ood
    v *= ood
    w *= ood
    u = 1.0 - v - w

    # optionally write out normal
    if normal:
        normal[] = n.copy()

    # Note: sign was in the signature but not assigned in your C++ snippet.
    # If you want to match the Moller version, you'd add: sign = d

    return True


@always_inline
fn diff_product[
    dtype: DType
](
    a: Scalar[dtype], b: Scalar[dtype], c: Scalar[dtype], d: Scalar[dtype]
) -> Scalar[dtype]:
    """
    Computes the difference of products a*b - c*d using
    FMA instructions for improved numerical precision.
    """
    cd = c * d
    diff = fma(a, b, -cd)
    error = fma(-c, d, cd)

    return diff + error


@always_inline
fn max_dim[dtype: DType](v: Vec3[dtype]) -> Int:
    x = abs(v[0])
    y = abs(v[1])
    z = abs(v[2])
    if x > y and x > z:
        return 0
    if y > z:
        return 1
    return 2


@always_inline
fn intersect_ray_tri_woop[
    dtype: DType
](
    p: Vec3[dtype],
    dir: Vec3[dtype],
    a: Vec3[dtype],
    b: Vec3[dtype],
    c: Vec3[dtype],
    mut t: Scalar[dtype],
    mut u: Scalar[dtype],
    mut v: Scalar[dtype],
    mut sign: Scalar[dtype],
    normal: UnsafePointer[Vec3[dtype], MutAnyOrigin],
) -> Bool:
    """
    Woop intersection (water-tight ray-triangle)
    http://jcgt.org/published/0002/01/05/.
    """
    # precompute dimensions
    kz = max_dim(dir)
    kx = kz + 1
    if kx == 3:
        kx = 0
    ky = kx + 1
    if ky == 3:
        ky = 0

    if dir[kz] < 0.0:
        tmp = kx
        kx = ky
        ky = tmp

    Sx = dir[kx] / dir[kz]
    Sy = dir[ky] / dir[kz]
    Sz = 1.0 / dir[kz]

    # transform vertices to ray space
    A = a - p
    B = b - p
    C = c - p

    Ax = A[kx] - Sx * A[kz]
    Ay = A[ky] - Sy * A[kz]
    Bx = B[kx] - Sx * B[kz]
    By = B[ky] - Sy * B[kz]
    Cx = C[kx] - Sx * C[kz]
    Cy = C[ky] - Sy * C[kz]

    # barycentric coordinates
    U = diff_product(Cx, By, Cy, Bx)
    V = diff_product(Ax, Cy, Ay, Cx)
    W = diff_product(Bx, Ay, By, Ax)

    # robustness fallback using Float64
    @parameter
    if dtype != DType.float64:
        if U == 0.0 or V == 0.0 or W == 0.0:
            comptime f64 = Float64
            U = Scalar[dtype](diff_product(f64(Cx), f64(By), f64(Cy), f64(Bx)))
            V = Scalar[dtype](diff_product(f64(Ax), f64(Cy), f64(Ay), f64(Cx)))
            W = Scalar[dtype](diff_product(f64(Bx), f64(Ay), f64(By), f64(Ax)))

    # edge tests
    if (U < 0.0 or V < 0.0 or W < 0.0) and (U > 0.0 or V > 0.0 or W > 0.0):
        return False

    det = U + V + W
    if det == 0.0:
        return False

    Az = Sz * A[kz]
    Bz = Sz * B[kz]
    Cz = Sz * C[kz]
    T = U * Az + V * Bz + W * Cz

    # Sign check: replacing xorf(T, det_sign) < 0.0 with (T * det < 0.0)
    # This is numerically equivalent for checking if T and det have different signs
    if T * det < 0.0:
        return False

    rcpDet = 1.0 / det
    u = U * rcpDet
    v = V * rcpDet
    t = T * rcpDet
    sign = det

    if normal:
        ab = b - a
        ac = c - a
        normal[] = cross(ab, ac)

    return True


@always_inline
fn edge_edge_test[
    dtype: DType
](
    v0: Vec3[dtype],
    u0: Vec3[dtype],
    u1: Vec3[dtype],
    i0: Int,
    i1: Int,
    Ax: Scalar[dtype],
    Ay: Scalar[dtype],
) -> Bool:
    """
    This edge to edge test is based on Franlin Antonio's gem:
    "Faster Line Segment Intersection", in Graphics Gems III,
    pp. 199-202.
    """
    Bx = u0[i0] - u1[i0]
    By = u0[i1] - u1[i1]
    Cx = v0[i0] - u0[i0]
    Cy = v0[i1] - u0[i1]
    f = diff_product(Ay, Bx, Ax, By)
    d = diff_product(By, Cx, Bx, Cy)

    if (f > 0.0 and d >= 0.0 and d <= f) or (f < 0.0 and d <= 0.0 and d >= f):
        e = Ax * Cy - Ay * Cx
        if f > 0.0:
            if e >= 0.0 and e <= f:
                return True
        else:
            if e <= 0.0 and e >= f:
                return True
    return False


@always_inline
fn edge_against_tri_edges[
    dtype: DType
](
    v0: Vec3[dtype],
    v1: Vec3[dtype],
    u0: Vec3[dtype],
    u1: Vec3[dtype],
    u2: Vec3[dtype],
    i0: Int,
    i1: Int,
) -> Bool:
    Ax = v1[i0] - v0[i0]
    Ay = v1[i1] - v0[i1]
    if edge_edge_test(v0, u0, u1, i0, i1, Ax, Ay):
        return True
    if edge_edge_test(v0, u1, u2, i0, i1, Ax, Ay):
        return True
    if edge_edge_test(v0, u2, u0, i0, i1, Ax, Ay):
        return True
    return False


@always_inline
fn point_in_tri[
    dtype: DType
](
    v0: Vec3[dtype],
    u0: Vec3[dtype],
    u1: Vec3[dtype],
    u2: Vec3[dtype],
    i0: Int,
    i1: Int,
) -> Bool:
    fn check(p1: Vec3[dtype], p2: Vec3[dtype]) unified {read} -> Scalar[dtype]:
        a = p2[i1] - p1[i1]
        b = -(p2[i0] - p1[i0])
        c = -a * p1[i0] - b * p1[i1]
        return a * v0[i0] + b * v0[i1] + c

    d0 = check(u0, u1)
    d1 = check(u1, u2)
    d2 = check(u2, u0)

    if (d0 * d1 > 0.0) and (d0 * d2 > 0.0):
        return True
    return False


@always_inline
fn coplanar_tri_tri[
    dtype: DType
](
    n: Vec3[dtype],
    v0: Vec3[dtype],
    v1: Vec3[dtype],
    v2: Vec3[dtype],
    u0: Vec3[dtype],
    u1: Vec3[dtype],
    u2: Vec3[dtype],
) -> Bool:
    a_abs = Vec3[dtype](abs(n[0]), abs(n[1]), abs(n[2]))
    i0: Int
    i1: Int

    if a_abs[0] > a_abs[1]:
        if a_abs[0] > a_abs[2]:
            i0 = 1
            i1 = 2
        else:
            i0 = 0
            i1 = 1
    else:
        if a_abs[2] > a_abs[1]:
            i0 = 0
            i1 = 1
        else:
            i0 = 0
            i1 = 2

    if edge_against_tri_edges(v0, v1, u0, u1, u2, i0, i1):
        return True
    if edge_against_tri_edges(v1, v2, u0, u1, u2, i0, i1):
        return True
    if edge_against_tri_edges(v2, v0, u0, u1, u2, i0, i1):
        return True

    if point_in_tri(v0, u0, u1, u2, i0, i1):
        return True
    if point_in_tri(u0, v0, v1, v2, i0, i1):
        return True

    return False


@fieldwise_init
struct Intervals[dtype: DType](Copyable):
    var a: Scalar[Self.dtype]
    var b: Scalar[Self.dtype]
    var c: Scalar[Self.dtype]
    var x0: Scalar[Self.dtype]
    var x1: Scalar[Self.dtype]
    var is_coplanar: Bool


@always_inline
fn get_intervals[
    dtype: DType
](
    vv0: Scalar[dtype],
    vv1: Scalar[dtype],
    vv2: Scalar[dtype],
    d0: Scalar[dtype],
    d1: Scalar[dtype],
    d2: Scalar[dtype],
    d0d1: Scalar[dtype],
    d0d2: Scalar[dtype],
) -> Intervals[dtype]:
    """Returns (a, b, c, x0, x1, is_coplanar)."""
    if d0d1 > 0.0:
        return Intervals(
            vv2,
            (vv0 - vv2) * d2,
            (vv1 - vv2) * d2,
            d2 - d0,
            d2 - d1,
            False,
        )
    elif d0d2 > 0.0:
        return Intervals(
            vv1,
            (vv0 - vv1) * d1,
            (vv2 - vv1) * d1,
            d1 - d0,
            d1 - d2,
            False,
        )
    elif (d1 * d2 > 0.0) or (d0 != 0.0):
        return Intervals(
            vv0,
            (vv1 - vv0) * d0,
            (vv2 - vv0) * d0,
            d0 - d1,
            d0 - d2,
            False,
        )
    elif d1 != 0.0:
        return Intervals(
            vv1,
            (vv0 - vv1) * d1,
            (vv2 - vv1) * d1,
            d1 - d0,
            d1 - d2,
            False,
        )
    elif d2 != 0.0:
        return Intervals(
            vv2,
            (vv0 - vv2) * d2,
            (vv1 - vv2) * d2,
            d2 - d0,
            d2 - d1,
            False,
        )
    return Intervals(Scalar[dtype](0), 0, 0, 0, 0, True)


comptime intersect_tri_tri = no_div_tri_tri_isect


@always_inline
fn no_div_tri_tri_isect[
    dtype: DType
](
    v0: Vec3[dtype],
    v1: Vec3[dtype],
    v2: Vec3[dtype],
    u0: Vec3[dtype],
    u1: Vec3[dtype],
    u2: Vec3[dtype],
) -> Bool:
    comptime EPSILON = 0.000001

    # plane Equation for Tri 1
    e1 = v1 - v0
    e2 = v2 - v0
    n1 = cross(e1, e2)
    d1_plane = -dot(n1, v0)

    # signed distances from Tri 2 to Plane 1
    du0 = dot(n1, u0) + d1_plane
    du1 = dot(n1, u1) + d1_plane
    du2 = dot(n1, u2) + d1_plane

    if abs(du0) < EPSILON:
        du0 = 0.0
    if abs(du1) < EPSILON:
        du1 = 0.0
    if abs(du2) < EPSILON:
        du2 = 0.0

    if (du0 * du1 > 0.0) and (du0 * du2 > 0.0):
        return False

    # plane Equation for Tri 2
    n2 = cross(u1 - u0, u2 - u0)
    d2_plane = -dot(n2, u0)

    # signed distances from Tri 1 to Plane 2
    dv0 = dot(n2, v0) + d2_plane
    dv1 = dot(n2, v1) + d2_plane
    dv2 = dot(n2, v2) + d2_plane

    if abs(dv0) < EPSILON:
        dv0 = 0.0
    if abs(dv1) < EPSILON:
        dv1 = 0.0
    if abs(dv2) < EPSILON:
        dv2 = 0.0

    if (dv0 * dv1 > 0.0) and (dv0 * dv2 > 0.0):
        return False

    # projection onto intersection line
    d_dir = cross(n1, n2)
    index = max_dim(d_dir)  # Using the helper we wrote earlier

    # compute intervals
    res1 = get_intervals(
        v0[index], v1[index], v2[index], dv0, dv1, dv2, dv0 * dv1, dv0 * dv2
    )
    if res1.is_coplanar:
        return coplanar_tri_tri(n1, v0, v1, v2, u0, u1, u2)

    res2 = get_intervals(
        u0[index], u1[index], u2[index], du0, du1, du2, du0 * du1, du0 * du2
    )
    if res2.is_coplanar:
        return coplanar_tri_tri(n1, v0, v1, v2, u0, u1, u2)

    # overlap check
    xx = res1.x0 * res1.x1
    yy = res2.x0 * res2.x1
    xxyy = xx * yy

    s1_0 = res1.a * xxyy + res1.b * res1.x1 * yy
    s1_1 = res1.a * xxyy + res1.c * res1.x0 * yy
    s2_0 = res2.a * xxyy + res2.b * xx * res2.x1
    s2_1 = res2.a * xxyy + res2.c * xx * res2.x0

    @always_inline
    fn _sort[dtype: DType](mut a: Scalar[dtype], mut b: Scalar[dtype]):
        if a > b:
            swap(a, b)

    _sort(s1_0, s1_1)
    _sort(s2_0, s2_1)

    return not (s1_1 < s2_0 or s2_1 < s1_0)


fn closest_point_edge_edge[
    dtype: DType
](
    p1: Vec3[dtype],
    q1: Vec3[dtype],
    p2: Vec3[dtype],
    q2: Vec3[dtype],
    epsilon: Scalar[dtype],
) -> Vec3[dtype]:
    """Find the closest points between two edges.

    Args:
        p1: First point of first edge.
        q1: Second point of first edge.
        p2: First point of second edge.
        q2: Second point of second edge.
        epsilon: Zero tolerance for determining if points in an edge are degenerate.

    Returns:
        Barycentric weights to the points on each edge, as well as the closest distance between the edges.
        vec3 output containing (s,t,d), where ``s`` in [0,1] is the barycentric weight for the first edge, ``t`` is the barycentric weight for the second edge, and ``d`` is the distance between the two edges at these two closest points.
    """
    # direction vectors of each segment/edge
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2

    a = dot(d1, d1)  # squared length of segment s1, always nonnegative
    e = dot(d2, d2)  # squared length of segment s2, always nonnegative
    f = dot(d2, r)

    s = Scalar[dtype](0.0)
    t = Scalar[dtype](0.0)
    dist = length(p2 - p1)

    # Check if either or both segments degenerate into points
    if a <= epsilon and e <= epsilon:
        # both segments degenerate into points
        return Vec3[dtype](s, t, dist)

    if a <= epsilon:
        s = Scalar[dtype](0.0)
        t = Scalar[dtype](f / e)  # s = 0 => t = (b*s + f) / e = f / e
    else:
        c = dot(d1, r)
        if e <= epsilon:
            # second segment generates into a point
            s = clamp(-c / a, 0.0, 1.0)  # t = 0 => s = (b*t-c)/a = -c/a
            t = Scalar[dtype](0.0)
        else:
            # The general nondegenerate case starts here
            b = dot(d1, d2)
            denom = a * e - b * b  # always nonnegative

            # if segments not parallel, compute closest point on L1 to L2 and
            # clamp to segment S1. Else pick arbitrary s (here 0)
            if denom != 0.0:
                s = clamp((b * f - c * e) / denom, 0.0, 1.0)
            else:
                s = 0.0

            # compute point on L2 closest to S1(s) using
            # t = dot((p1+d2*s) - p2,d2)/dot(d2,d2) = (b*s+f)/e
            t = (b * s + f) / e

            # if t in [0,1] done. Else clamp t, recompute s for the new value
            # of t using s = dot((p2+d2*t-p1,d1)/dot(d1,d1) = (t*b - c)/a
            # and clamp s to [0,1]
            if t < 0.0:
                t = 0.0
                s = clamp(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = clamp((b - c) / a, 0.0, 1.0)

    c1 = p1 + (q1 - p1) * s
    c2 = p2 + (q2 - p2) * t
    dist = length(c2 - c1)
    return Vec3[dtype](s, t, dist)


fn main():
    print("hello warp.intersect")
    a = Vec3f32(1.0, 2.0, 3.0)
    b = Vec3f32(4.0, 5.0, 6.0)
    c = Vec3f32(7.0, 8.0, 8.0)
    p = Vec3f32(9.0, 1.0, 2.0)
    lower = Vec3f32(3.0, 4.0, 5.0)
    upper = Vec3f32(6.0, 7.0, 8.0)
    b_lower = Vec3f32(13.0, 14.0, 5.0)
    b_upper = Vec3f32(16.0, 17.0, 8.0)
    os = Vec3f32(9.0, 7.0, 6.0)
    rcp_dir = Vec3f32(5.0, 4.0, 3.0)
    t = Float32(1.0)

    print(closest_point_to_aabb(p, lower, upper))
    print(closest_point_to_triangle(a, b, c, p))
    print(furthest_point_to_triangle(a, b, c, p))
    print(intersect_ray_aabb(os, rcp_dir, lower, upper, t))
    print(intersect_aabb_aabb(lower, upper, b_lower, b_upper))
