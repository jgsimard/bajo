from std.math import fma, abs, max, clamp

from bajo.core.vec import (
    Vec2,
    Vec3,
    dot,
    vmin,
    vmax,
    vclamp,
    cross,
    length,
)


# Result structs
@fieldwise_init
struct RayTriHit[dtype: DType, width: Int](TrivialRegisterPassable, Writable):
    var mask: SIMD[DType.bool, Self.width]
    var t: SIMD[Self.dtype, Self.width]
    var u: SIMD[Self.dtype, Self.width]
    var v: SIMD[Self.dtype, Self.width]


@fieldwise_init
struct RayAabbHit[dtype: DType, width: Int](TrivialRegisterPassable, Writable):
    """RayAabbHit.

    - mask: SIMD[DType.bool, Self.width]
    - tmin: SIMD[Self.dtype, Self.width]
    """

    var mask: SIMD[DType.bool, Self.width]
    var tmin: SIMD[Self.dtype, Self.width]
    # var tmax: SIMD[Self.dtype, Self.width]


@always_inline
def diff_product[
    dtype: DType, width: Int
](
    a: SIMD[dtype, width],
    b: SIMD[dtype, width],
    c: SIMD[dtype, width],
    d: SIMD[dtype, width],
) -> SIMD[dtype, width]:
    """
    Computes the difference of products a*b - c*d using
    FMA instructions for improved numerical precision.
    """
    cd = c * d
    diff = fma(a, b, -cd)
    error = fma(-c, d, cd)
    return diff + error


# intersection functions
@always_inline
def closest_point_to_aabb[
    dtype: DType, width: Int
](
    p: Vec3[dtype, width],
    lower: Vec3[dtype, width],
    upper: Vec3[dtype, width],
) -> Vec3[dtype, width]:
    return vclamp(p, lower, upper)


def closest_point_to_triangle[
    dtype: DType, width: Int
](
    a: Vec3[dtype, width],
    b: Vec3[dtype, width],
    c: Vec3[dtype, width],
    p: Vec3[dtype, width],
) -> Vec2[dtype, width]:
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = dot(ab, ap)
    d2 = dot(ac, ap)

    if d1 <= 0 and d2 <= 0:
        # Vertex A: v=0, w=0, u=1
        return Vec2[dtype, width](1, 0)

    bp = p - b
    d3 = dot(ab, bp)
    d4 = dot(ac, bp)
    if d3 >= 0 and d4 <= d3:
        # Vertex B: v=1, w=0, u=0
        return Vec2[dtype, width](0, 1)

    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        # Edge AB
        v = d1 / (d1 - d3)
        u = 1 - v
        return Vec2[dtype, width](u, v)

    cp = p - c
    d5 = dot(ab, cp)
    d6 = dot(ac, cp)
    if d6 >= 0 and d5 <= d6:
        # Vertex C: v=0, w=1, u=0
        return Vec2[dtype, width](0, 0)

    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        # Edge AC
        w = d2 / (d2 - d6)
        u = 1 - w
        return Vec2[dtype, width](u, 0)

    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        # Edge BC
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        v = 1 - w
        return Vec2[dtype, width](0, v)

    # Inside Face
    denom = 1 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    u = 1 - v - w
    return Vec2[dtype, width](u, v)


def furthest_point_to_triangle[
    dtype: DType, width: Int
](
    a: Vec3[dtype, width],
    b: Vec3[dtype, width],
    c: Vec3[dtype, width],
    p: Vec3[dtype, width],
) -> Vec2[dtype, width]:
    pa = p - a
    pb = p - b
    pc = p - c

    dist_a = dot(pa, pa)
    dist_b = dot(pb, pb)
    dist_c = dot(pc, pc)

    # a is furthest
    if dist_a > dist_b and dist_a > dist_c:
        return Vec2[dtype, width](1, 0)

    # b is furthest
    if dist_b > dist_c:
        return Vec2[dtype, width](0, 1)

    #  c is furthest
    return Vec2[dtype, width](0, 0)


@always_inline
def _axis_t_near[
    dtype: DType, width: Int
](
    o: SIMD[dtype, width],
    rd: SIMD[dtype, width],
    mn: SIMD[dtype, width],
    mx: SIMD[dtype, width],
) -> SIMD[dtype, width]:
    var t0 = (mn - o) * rd
    var t1 = (mx - o) * rd
    return min(t0, t1)


@always_inline
def _axis_t_far[
    dtype: DType, width: Int
](
    o: SIMD[dtype, width],
    rd: SIMD[dtype, width],
    mn: SIMD[dtype, width],
    mx: SIMD[dtype, width],
) -> SIMD[dtype, width]:
    var t0 = (mn - o) * rd
    var t1 = (mx - o) * rd
    return max(t0, t1)


@always_inline
def intersect_ray_aabb[
    dtype: DType, width: Int
](
    o: Vec3[dtype, width],
    rd: Vec3[dtype, width],
    bmin: Vec3[dtype, width],
    bmax: Vec3[dtype, width],
    t_max: SIMD[dtype, width],
) -> RayAabbHit[dtype, width]:
    comptime assert dtype in [DType.float32, DType.float64]

    var tx1 = _axis_t_near(o.x, rd.x, bmin.x, bmax.x)
    var tx2 = _axis_t_far(o.x, rd.x, bmin.x, bmax.x)
    var ty1 = _axis_t_near(o.y, rd.y, bmin.y, bmax.y)
    var ty2 = _axis_t_far(o.y, rd.y, bmin.y, bmax.y)
    var tz1 = _axis_t_near(o.z, rd.z, bmin.z, bmax.z)
    var tz2 = _axis_t_far(o.z, rd.z, bmin.z, bmax.z)

    var tmin = max(max(tx1, ty1), max(tz1, 0.0))
    var tmax = min(min(tx2, ty2), min(tz2, t_max))

    var mask = tmin.le(tmax)

    return RayAabbHit(mask, tmin)


@always_inline
def intersect_aabb_aabb[
    dtype: DType, width: Int
](
    a_lower: Vec3[dtype, width],
    a_upper: Vec3[dtype, width],
    b_lower: Vec3[dtype, width],
    b_upper: Vec3[dtype, width],
) -> SIMD[DType.bool, width]:
    return (
        a_lower.x.le(b_upper.x)
        & a_lower.y.le(b_upper.y)
        & a_lower.z.le(b_upper.z)
        & a_upper.x.ge(b_lower.x)
        & a_upper.y.ge(b_lower.y)
        & a_upper.z.ge(b_lower.z)
    )


@always_inline
def intersect_ray_tri[
    dtype: DType, width: Int
](
    o: Vec3[dtype, width],
    d: Vec3[dtype, width],
    v0: Vec3[dtype, width],
    v1: Vec3[dtype, width],
    v2: Vec3[dtype, width],
    t_max: SIMD[dtype, width],
    t_min: SIMD[dtype, width] = SIMD[dtype, width](1.0e-4),
) -> RayTriHit[dtype, width]:
    """Moller and Trumbore's method."""
    comptime assert dtype in [DType.float32, DType.float64]
    comptime EPSILON = Scalar[dtype](1e-8 if dtype == DType.float32 else 1e-16)
    comptime BVH_INF = SIMD[dtype, width](3.4028234663852886e38)

    var e1 = v1 - v0
    var e2 = v2 - v0

    var p = cross(d, e2)
    var det = dot(e1, p)

    var det_ok = det.gt(EPSILON) | det.lt(-EPSILON)
    var inv_det = Scalar[dtype](1.0) / det

    var tv = o - v0
    var u = dot(tv, p) * inv_det

    var q = cross(tv, e1)
    var v = dot(d, q) * inv_det
    var t = dot(e2, q) * inv_det

    var mask = (
        det_ok
        & u.ge(0.0)
        & u.le(1.0)
        & v.ge(0.0)
        & (u + v).le(1.0)
        & t.gt(t_min)
        & t.lt(t_max)
    )

    return RayTriHit[dtype, width](
        mask,
        mask.select(t, BVH_INF),
        u,
        v,
    )


@always_inline
def intersect_ray_tri[
    dtype: DType
](
    vertices: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    prim_idx: UInt32,
    o: Vec3[dtype],
    d: Vec3[dtype],
    t_max: Scalar[dtype],
) -> RayTriHit[dtype, 1]:
    var base = Int(prim_idx) * 9
    var v0x = vertices[base + 0]
    var v0y = vertices[base + 1]
    var v0z = vertices[base + 2]
    var v1x = vertices[base + 3]
    var v1y = vertices[base + 4]
    var v1z = vertices[base + 5]
    var v2x = vertices[base + 6]
    var v2y = vertices[base + 7]
    var v2z = vertices[base + 8]

    return intersect_ray_tri(
        o,
        d,
        Vec3[dtype](v0x, v0y, v0z),
        Vec3[dtype](v1x, v1y, v1z),
        Vec3[dtype](v2x, v2y, v2z),
        t_max,
    )


@always_inline
def intersect_ray_tri_rtcd[
    dtype: DType, width: Int
](
    p: Vec3[dtype, width],
    dir: Vec3[dtype, width],
    a: Vec3[dtype, width],
    b: Vec3[dtype, width],
    c: Vec3[dtype, width],
    mut t: SIMD[dtype, width],
    mut u: SIMD[dtype, width],
    mut v: SIMD[dtype, width],
    mut w: SIMD[dtype, width],
    mut sign: SIMD[dtype, width],
    normal: Optional[UnsafePointer[Vec3[dtype, width], MutAnyOrigin]],
) -> Bool:
    comptime assert width == 1
    comptime assert dtype in [DType.float32, DType.float64]

    var ab = b - a
    var ac = c - a

    # Calculate normal.
    var n = cross(ab, ac)

    # Need to solve a system of three equations to give t, u, v.
    var d = dot(-dir, n)

    # If dir is parallel to triangle plane or points away from triangle.
    if d <= 0.0:
        return False

    var ap = p - a
    t = dot(ap, n)

    # Ignore tris behind the ray.
    if t < 0.0:
        return False

    # Compute barycentric coordinates.
    var e = cross(-dir, ap)

    v = dot(ac, e)
    if v < 0.0 or v > d:
        return False

    w = -dot(ab, e)
    if w < 0.0 or (v + w) > d:
        return False

    var ood = 1.0 / d
    t *= ood
    v *= ood
    w *= ood
    u = 1.0 - v - w

    if normal:
        normal.unsafe_value()[] = n

    sign = d
    return True


@always_inline
def max_dim[dtype: DType, width: Int](v: Vec3[dtype, width]) -> Int:
    comptime assert width == 1

    var x = abs(v.x)
    var y = abs(v.y)
    var z = abs(v.z)

    if x > y and x > z:
        return 0
    if y > z:
        return 1
    return 2


@always_inline
def _v3_get[
    dtype: DType, width: Int
](v: Vec3[dtype, width], i: Int) -> SIMD[dtype, width]:
    if i == 0:
        return v.x
    if i == 1:
        return v.y
    return v.z


@always_inline
def intersect_ray_tri_woop[
    dtype: DType, width: Int
](
    p: Vec3[dtype, width],
    dir: Vec3[dtype, width],
    a: Vec3[dtype, width],
    b: Vec3[dtype, width],
    c: Vec3[dtype, width],
    mut t: SIMD[dtype, width],
    mut u: SIMD[dtype, width],
    mut v: SIMD[dtype, width],
    mut sign: SIMD[dtype, width],
    normal: Optional[UnsafePointer[Vec3[dtype, width], MutAnyOrigin]],
) -> Bool:
    """
    Woop intersection, watertight ray-triangle.
    """
    comptime assert width == 1
    comptime assert dtype in [DType.float32, DType.float64]

    # Precompute dimensions.
    var kz = max_dim(dir)
    var kx = kz + 1
    if kx == 3:
        kx = 0

    var ky = kx + 1
    if ky == 3:
        ky = 0

    if _v3_get(dir, kz) < 0.0:
        var tmp = kx
        kx = ky
        ky = tmp

    var dir_kz = _v3_get(dir, kz)
    var Sx = _v3_get(dir, kx) / dir_kz
    var Sy = _v3_get(dir, ky) / dir_kz
    var Sz = 1.0 / dir_kz

    # Transform vertices to ray space.
    var A = a - p
    var B = b - p
    var C = c - p

    var Akz = _v3_get(A, kz)
    var Bkz = _v3_get(B, kz)
    var Ckz = _v3_get(C, kz)

    var Ax = _v3_get(A, kx) - Sx * Akz
    var Ay = _v3_get(A, ky) - Sy * Akz
    var Bx = _v3_get(B, kx) - Sx * Bkz
    var By = _v3_get(B, ky) - Sy * Bkz
    var Cx = _v3_get(C, kx) - Sx * Ckz
    var Cy = _v3_get(C, ky) - Sy * Ckz

    # Barycentric coordinates.
    var U = diff_product(Cx, By, Cy, Bx)
    var V = diff_product(Ax, Cy, Ay, Cx)
    var W = diff_product(Bx, Ay, By, Ax)

    # # Robustness fallback using Float64.
    # comptime if dtype != DType.float64:
    #     if U == 0.0 or V == 0.0 or W == 0.0:
    #         comptime f64 = SIMD[DType.float64, width]
    #         U = diff_product(f64(Cx), f64(By), f64(Cy), f64(Bx))
    #         V = diff_product(f64(Ax), f64(Cy), f64(Ay), f64(Cx))
    #         W = diff_product(f64(Bx), f64(Ay), f64(By), f64(Ax))

    # Edge tests.
    if (U < 0.0 or V < 0.0 or W < 0.0) and (U > 0.0 or V > 0.0 or W > 0.0):
        return False

    var det = U + V + W
    if det == 0.0:
        return False

    var Az = Sz * Akz
    var Bz = Sz * Bkz
    var Cz = Sz * Ckz
    var T = U * Az + V * Bz + W * Cz

    # Sign check.
    if T * det < 0.0:
        return False

    var rcpDet = 1.0 / det
    u = U * rcpDet
    v = V * rcpDet
    t = T * rcpDet
    sign = det

    if normal:
        normal.unsafe_value()[] = cross(b - a, c - a)

    return True


@always_inline
def edge_edge_test[
    dtype: DType, width: Int
](
    v0: Vec3[dtype, width],
    u0: Vec3[dtype, width],
    u1: Vec3[dtype, width],
    i0: Int,
    i1: Int,
    Ax: SIMD[dtype, width],
    Ay: SIMD[dtype, width],
) -> Bool:
    comptime assert width == 1

    var Bx = _v3_get(u0, i0) - _v3_get(u1, i0)
    var By = _v3_get(u0, i1) - _v3_get(u1, i1)
    var Cx = _v3_get(v0, i0) - _v3_get(u0, i0)
    var Cy = _v3_get(v0, i1) - _v3_get(u0, i1)

    var f = diff_product(Ay, Bx, Ax, By)
    var d = diff_product(By, Cx, Bx, Cy)

    if (f > 0.0 and d >= 0.0 and d <= f) or (f < 0.0 and d <= 0.0 and d >= f):
        var e = diff_product(Ax, Cy, Ay, Cx)

        if f > 0.0:
            if e >= 0.0 and e <= f:
                return True
        else:
            if e <= 0.0 and e >= f:
                return True

    return False


@always_inline
def edge_against_tri_edges[
    dtype: DType, width: Int
](
    v0: Vec3[dtype, width],
    v1: Vec3[dtype, width],
    u0: Vec3[dtype, width],
    u1: Vec3[dtype, width],
    u2: Vec3[dtype, width],
    i0: Int,
    i1: Int,
) -> Bool:
    comptime assert width == 1

    var Ax = _v3_get(v1, i0) - _v3_get(v0, i0)
    var Ay = _v3_get(v1, i1) - _v3_get(v0, i1)

    if edge_edge_test(v0, u0, u1, i0, i1, Ax, Ay):
        return True
    if edge_edge_test(v0, u1, u2, i0, i1, Ax, Ay):
        return True
    if edge_edge_test(v0, u2, u0, i0, i1, Ax, Ay):
        return True

    return False


@always_inline
def _point_in_tri_check[
    dtype: DType, width: Int
](
    v0: Vec3[dtype, width],
    p1: Vec3[dtype, width],
    p2: Vec3[dtype, width],
    i0: Int,
    i1: Int,
) -> SIMD[dtype, width]:
    var a = _v3_get(p2, i1) - _v3_get(p1, i1)
    var b = -(_v3_get(p2, i0) - _v3_get(p1, i0))
    var c = -a * _v3_get(p1, i0) - b * _v3_get(p1, i1)
    return a * _v3_get(v0, i0) + b * _v3_get(v0, i1) + c


@always_inline
def point_in_tri[
    dtype: DType, width: Int
](
    v0: Vec3[dtype, width],
    u0: Vec3[dtype, width],
    u1: Vec3[dtype, width],
    u2: Vec3[dtype, width],
    i0: Int,
    i1: Int,
) -> Bool:
    comptime assert width == 1

    var d0 = _point_in_tri_check(v0, u0, u1, i0, i1)
    var d1 = _point_in_tri_check(v0, u1, u2, i0, i1)
    var d2 = _point_in_tri_check(v0, u2, u0, i0, i1)

    return (d0 * d1 > 0.0) and (d0 * d2 > 0.0)


@always_inline
def coplanar_tri_tri[
    dtype: DType, width: Int
](
    n: Vec3[dtype, width],
    v0: Vec3[dtype, width],
    v1: Vec3[dtype, width],
    v2: Vec3[dtype, width],
    u0: Vec3[dtype, width],
    u1: Vec3[dtype, width],
    u2: Vec3[dtype, width],
) -> Bool:
    comptime assert width == 1

    var ax = abs(n.x)
    var ay = abs(n.y)
    var az = abs(n.z)

    var i0: Int
    var i1: Int

    if ax > ay:
        if ax > az:
            i0 = 1
            i1 = 2
        else:
            i0 = 0
            i1 = 1
    else:
        if az > ay:
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
struct Intervals[dtype: DType, width: Int](Movable):
    var a: SIMD[Self.dtype, Self.width]
    var b: SIMD[Self.dtype, Self.width]
    var c: SIMD[Self.dtype, Self.width]
    var x0: SIMD[Self.dtype, Self.width]
    var x1: SIMD[Self.dtype, Self.width]
    var is_coplanar: Bool


@always_inline
def get_intervals[
    dtype: DType, width: Int
](
    vv0: SIMD[dtype, width],
    vv1: SIMD[dtype, width],
    vv2: SIMD[dtype, width],
    d0: SIMD[dtype, width],
    d1: SIMD[dtype, width],
    d2: SIMD[dtype, width],
    d0d1: SIMD[dtype, width],
    d0d2: SIMD[dtype, width],
) -> Intervals[dtype, width]:
    comptime assert width == 1

    if d0d1 > 0.0:
        return Intervals(
            vv2,
            (vv0 - vv2) * d2,
            (vv1 - vv2) * d2,
            d2 - d0,
            d2 - d1,
            False,
        )

    if d0d2 > 0.0:
        return Intervals(
            vv1,
            (vv0 - vv1) * d1,
            (vv2 - vv1) * d1,
            d1 - d0,
            d1 - d2,
            False,
        )

    if (d1 * d2 > 0.0) or (d0 != 0.0):
        return Intervals(
            vv0,
            (vv1 - vv0) * d0,
            (vv2 - vv0) * d0,
            d0 - d1,
            d0 - d2,
            False,
        )

    if d1 != 0.0:
        return Intervals(
            vv1,
            (vv0 - vv1) * d1,
            (vv2 - vv1) * d1,
            d1 - d0,
            d1 - d2,
            False,
        )

    if d2 != 0.0:
        return Intervals(
            vv2,
            (vv0 - vv2) * d2,
            (vv1 - vv2) * d2,
            d2 - d0,
            d2 - d1,
            False,
        )

    return Intervals(
        SIMD[dtype, width](0.0),
        SIMD[dtype, width](0.0),
        SIMD[dtype, width](0.0),
        SIMD[dtype, width](0.0),
        SIMD[dtype, width](0.0),
        True,
    )


comptime intersect_tri_tri = no_div_tri_tri_isect


@always_inline
def no_div_tri_tri_isect[
    dtype: DType, width: Int
](
    v0: Vec3[dtype, width],
    v1: Vec3[dtype, width],
    v2: Vec3[dtype, width],
    u0: Vec3[dtype, width],
    u1: Vec3[dtype, width],
    u2: Vec3[dtype, width],
) -> Bool:
    comptime assert width == 1
    comptime EPSILON = 0.000001

    # Plane equation for tri 1.
    var e1 = v1 - v0
    var e2 = v2 - v0
    var n1 = cross(e1, e2)
    var d1_plane = -dot(n1, v0)

    # Signed distances from tri 2 to plane 1.
    var du0 = dot(n1, u0) + d1_plane
    var du1 = dot(n1, u1) + d1_plane
    var du2 = dot(n1, u2) + d1_plane

    if abs(du0) < EPSILON:
        du0 = 0.0
    if abs(du1) < EPSILON:
        du1 = 0.0
    if abs(du2) < EPSILON:
        du2 = 0.0

    if (du0 * du1 > 0.0) and (du0 * du2 > 0.0):
        return False

    # Plane equation for tri 2.
    var n2 = cross(u1 - u0, u2 - u0)
    var d2_plane = -dot(n2, u0)

    # Signed distances from tri 1 to plane 2.
    var dv0 = dot(n2, v0) + d2_plane
    var dv1 = dot(n2, v1) + d2_plane
    var dv2 = dot(n2, v2) + d2_plane

    if abs(dv0) < EPSILON:
        dv0 = 0.0
    if abs(dv1) < EPSILON:
        dv1 = 0.0
    if abs(dv2) < EPSILON:
        dv2 = 0.0

    if (dv0 * dv1 > 0.0) and (dv0 * dv2 > 0.0):
        return False

    # Projection onto intersection line.
    var d_dir = cross(n1, n2)
    var index = max_dim(d_dir)

    var res1 = get_intervals(
        _v3_get(v0, index),
        _v3_get(v1, index),
        _v3_get(v2, index),
        dv0,
        dv1,
        dv2,
        dv0 * dv1,
        dv0 * dv2,
    )

    if res1.is_coplanar:
        return coplanar_tri_tri(n1, v0, v1, v2, u0, u1, u2)

    var res2 = get_intervals(
        _v3_get(u0, index),
        _v3_get(u1, index),
        _v3_get(u2, index),
        du0,
        du1,
        du2,
        du0 * du1,
        du0 * du2,
    )

    if res2.is_coplanar:
        return coplanar_tri_tri(n1, v0, v1, v2, u0, u1, u2)

    # Overlap check.
    var xx = res1.x0 * res1.x1
    var yy = res2.x0 * res2.x1
    var xxyy = xx * yy

    var s1_0 = res1.a * xxyy + res1.b * res1.x1 * yy
    var s1_1 = res1.a * xxyy + res1.c * res1.x0 * yy
    var s2_0 = res2.a * xxyy + res2.b * xx * res2.x1
    var s2_1 = res2.a * xxyy + res2.c * xx * res2.x0

    if s1_0 > s1_1:
        var tmp = s1_0
        s1_0 = s1_1
        s1_1 = tmp

    if s2_0 > s2_1:
        var tmp = s2_0
        s2_0 = s2_1
        s2_1 = tmp

    return not (s1_1 < s2_0 or s2_1 < s1_0)


def closest_point_edge_edge[
    dtype: DType, width: Int
](
    p1: Vec3[dtype, width],
    q1: Vec3[dtype, width],
    p2: Vec3[dtype, width],
    q2: Vec3[dtype, width],
    epsilon: SIMD[dtype, width],
) -> Vec3[dtype, width]:
    """Return (s, t, distance) for the closest points between two edges."""
    comptime assert width == 1

    var d1 = q1 - p1
    var d2 = q2 - p2
    var r = p1 - p2

    var a = dot(d1, d1)
    var e = dot(d2, d2)
    var f = dot(d2, r)

    var s = SIMD[dtype, width](0.0)
    var t = SIMD[dtype, width](0.0)
    var dist = length(p2 - p1)

    # Both segments degenerate into points.
    if a <= epsilon and e <= epsilon:
        return Vec3[dtype, width](s, t, dist)

    if a <= epsilon:
        s = 0.0
        t = f / e
    else:
        var c = dot(d1, r)

        if e <= epsilon:
            t = 0.0
            s = clamp(-c / a, 0.0, 1.0)
        else:
            var b = dot(d1, d2)
            var denom = a * e - b * b

            if denom != 0.0:
                s = clamp((b * f - c * e) / denom, 0.0, 1.0)
            else:
                s = 0.0

            t = (b * s + f) / e

            if t < 0.0:
                t = 0.0
                s = clamp(-c / a, 0.0, 1.0)

            if t > 1.0:
                t = 1.0
                s = clamp((b - c) / a, 0.0, 1.0)

    var c1 = p1 + (q1 - p1) * s
    var c2 = p2 + (q2 - p2) * t
    dist = length(c2 - c1)

    return Vec3[dtype, width](s, t, dist)
