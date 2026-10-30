from std.math import fma, abs, clamp, sqrt, min, max, pi
from std.sys.info import is_gpu
from std.utils.numerics import max_finite, isfinite
from std.memory import bitcast

from bajo.core.vec import (
    Vec2,
    Vec3,
    Point3,
    dot,
    vmin,
    vmax,
    vclamp,
    cross,
    length,
    longest_axis,
)
from bajo.core.aabb import AxisAlignedBoundingBox
from bajo.core.utils import fmin, fmax
from bajo.core.frame import Frame
from bajo.core.ray import Ray


# Result structs
@fieldwise_init
struct RayTriHit[dtype: DType, width: SIMDLength](
    TrivialRegisterPassable, Writable
):
    var mask: SIMD[.bool, Self.width]
    var t: SIMD[Self.dtype, Self.width]
    var u: SIMD[Self.dtype, Self.width]
    var v: SIMD[Self.dtype, Self.width]


@fieldwise_init
struct RayTriScaledHit[dtype: DType, width: SIMDLength](
    TrivialRegisterPassable
):
    """Division-free Moller-Trumbore candidate values.

    All numerators have been oriented to a positive determinant, so valid
    lanes can be tested against abs_det without calculating a reciprocal.
    """

    var mask: SIMD[.bool, Self.width]
    var abs_det: SIMD[Self.dtype, Self.width]
    var t_scaled: SIMD[Self.dtype, Self.width]
    var u_scaled: SIMD[Self.dtype, Self.width]
    var v_scaled: SIMD[Self.dtype, Self.width]


@fieldwise_init
struct RayDistanceHit[dtype: DType, width: SIMDLength](TrivialRegisterPassable):
    """RayDistanceHit.

    - mask: SIMD[.bool, Self.width]
    - t   : SIMD[Self.dtype, Self.width]
    """

    var mask: SIMD[.bool, Self.width]
    var t: SIMD[Self.dtype, Self.width]


def diff_product[
    dtype: DType, width: SIMDLength
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
    var cd = c * d
    var diff = fma(a, b, -cd)
    var error = fma(-c, d, cd)
    return diff + error


# intersection functions
def closest_point_to_aabb[
    dtype: DType, frame: Frame, width: SIMDLength
](
    p: Vec3[dtype, frame, width],
    lower: Vec3[dtype, frame, width],
    upper: Vec3[dtype, frame, width],
) -> Vec3[dtype, frame, width]:
    return vclamp(p, lower, upper)


def closest_point_to_triangle[
    dtype: DType, frame: Frame, width: SIMDLength
](
    a: Vec3[dtype, frame, width],
    b: Vec3[dtype, frame, width],
    c: Vec3[dtype, frame, width],
    p: Vec3[dtype, frame, width],
) -> Vec2[dtype, frame, width]:
    comptime assert width == 1, "current limitation :("
    var ab = b - a
    var ac = c - a
    var ap = p - a

    var d1 = dot(ab, ap)
    var d2 = dot(ac, ap)

    if d1 <= 0 and d2 <= 0:
        # Vertex A: v=0, w=0, u=1
        return Vec2[dtype, frame, width](1, 0)

    var bp = p - b
    var d3 = dot(ab, bp)
    var d4 = dot(ac, bp)
    if d3 >= 0 and d4 <= d3:
        # Vertex B: v=1, w=0, u=0
        return Vec2[dtype, frame, width](0, 1)

    var vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        # Edge AB
        var v = d1 / (d1 - d3)
        var u = 1 - v
        return Vec2[dtype, frame, width](u, v)

    var cp = p - c
    var d5 = dot(ab, cp)
    var d6 = dot(ac, cp)
    if d6 >= 0 and d5 <= d6:
        # Vertex C: v=0, w=1, u=0
        return Vec2[dtype, frame, width](0, 0)

    var vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        # Edge AC
        var w = d2 / (d2 - d6)
        var u = 1 - w
        return Vec2[dtype, frame, width](u, 0)

    var va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        # Edge BC
        var w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        var v = 1 - w
        return Vec2[dtype, frame, width](0, v)

    # Inside Face
    var denom = 1 / (va + vb + vc)
    var v = vb * denom
    var w = vc * denom
    var u = 1 - v - w
    return Vec2[dtype, frame, width](u, v)


def furthest_point_to_triangle[
    dtype: DType, frame: Frame, width: SIMDLength
](
    a: Vec3[dtype, frame, width],
    b: Vec3[dtype, frame, width],
    c: Vec3[dtype, frame, width],
    p: Vec3[dtype, frame, width],
) -> Vec2[dtype, frame, width]:
    comptime assert width == 1, "current limitation :("

    var pa = p - a
    var pb = p - b
    var pc = p - c

    var dist_a = dot(pa, pa)
    var dist_b = dot(pb, pb)
    var dist_c = dot(pc, pc)

    # a is furthest
    if dist_a > dist_b and dist_a > dist_c:
        return Vec2[dtype, frame, width](1, 0)

    # b is furthest
    if dist_b > dist_c:
        return Vec2[dtype, frame, width](0, 1)

    #  c is furthest
    return Vec2[dtype, frame, width](0, 0)


def intersect_ray_aabb_rcp[
    dtype: DType, frame: Frame, width: SIMDLength
](
    o: Point3[dtype, frame, width],
    rcp_d: Vec3[dtype, frame, width],
    aabb: AxisAlignedBoundingBox[dtype, frame, width],
    t_max: SIMD[dtype, width],
) -> RayDistanceHit[dtype, width]:
    return intersect_ray_aabb_rcp(o, rcp_d, aabb._min, aabb._max, t_max)


def intersect_ray_aabb_rcp[
    dtype: DType, frame: Frame, width: SIMDLength
](
    o: Point3[dtype, frame, width],
    rcp_d: Vec3[dtype, frame, width],
    bmin: Point3[dtype, frame, width],
    bmax: Point3[dtype, frame, width],
    t_max: SIMD[dtype, width],
) -> RayDistanceHit[dtype, width]:
    """Intersect a ray with an AABB using precomputed reciprocal direction."""
    comptime assert dtype.is_floating_point()

    var t0 = (bmin - o) * rcp_d
    var t1 = (bmax - o) * rcp_d
    var t_near = vmin(t0, t1)
    var t_far = vmax(t0, t1)

    var tmin = fmax(fmax(t_near.x, t_near.y), fmax(t_near.z, 0.0))
    var tmax = fmin(fmin(t_far.x, t_far.y), fmin(t_far.z, t_max))

    var mask = tmin.le(tmax)

    return RayDistanceHit(mask, tmin)


@always_inline
def intersect_ray_aabb_octant_fma[
    dtype: DType,
    frame: Frame,
    width: SIMDLength,
    positive_x: Bool,
    positive_y: Bool,
    positive_z: Bool,
](
    origin_rcp_d: Vec3[dtype, frame, width],
    rcp_d: Vec3[dtype, frame, width],
    aabb: AxisAlignedBoundingBox[dtype, frame, width],
    t_max: SIMD[dtype, width],
) -> RayDistanceHit[dtype, width]:
    """Intersect using ray-octant-selected bounds and fused multiply-subtract.
    """
    comptime assert dtype in [DType.float32, DType.float64]

    var tx_near: SIMD[dtype, width]
    var tx_far: SIMD[dtype, width]
    comptime if positive_x:
        tx_near = fma(aabb._min.x, rcp_d.x, -origin_rcp_d.x)
        tx_far = fma(aabb._max.x, rcp_d.x, -origin_rcp_d.x)
    else:
        tx_near = fma(aabb._max.x, rcp_d.x, -origin_rcp_d.x)
        tx_far = fma(aabb._min.x, rcp_d.x, -origin_rcp_d.x)

    var ty_near: SIMD[dtype, width]
    var ty_far: SIMD[dtype, width]
    comptime if positive_y:
        ty_near = fma(aabb._min.y, rcp_d.y, -origin_rcp_d.y)
        ty_far = fma(aabb._max.y, rcp_d.y, -origin_rcp_d.y)
    else:
        ty_near = fma(aabb._max.y, rcp_d.y, -origin_rcp_d.y)
        ty_far = fma(aabb._min.y, rcp_d.y, -origin_rcp_d.y)

    var tz_near: SIMD[dtype, width]
    var tz_far: SIMD[dtype, width]
    comptime if positive_z:
        tz_near = fma(aabb._min.z, rcp_d.z, -origin_rcp_d.z)
        tz_far = fma(aabb._max.z, rcp_d.z, -origin_rcp_d.z)
    else:
        tz_near = fma(aabb._max.z, rcp_d.z, -origin_rcp_d.z)
        tz_far = fma(aabb._min.z, rcp_d.z, -origin_rcp_d.z)

    var tmin = fmax(fmax(tx_near, ty_near), fmax(tz_near, 0.0))
    var tmax = fmin(fmin(tx_far, ty_far), fmin(tz_far, t_max))
    return RayDistanceHit(tmin.le(tmax), tmin)


def intersect_aabb_aabb[
    dtype: DType, frame: Frame, width: SIMDLength
](
    a_lower: Vec3[dtype, frame, width],
    a_upper: Vec3[dtype, frame, width],
    b_lower: Vec3[dtype, frame, width],
    b_upper: Vec3[dtype, frame, width],
) -> SIMD[.bool, width]:
    return (
        a_lower.x.le(b_upper.x)
        & a_lower.y.le(b_upper.y)
        & a_lower.z.le(b_upper.z)
        & a_upper.x.ge(b_lower.x)
        & a_upper.y.ge(b_lower.y)
        & a_upper.z.ge(b_lower.z)
    )


def intersect_ray_sphere[
    dtype: DType,
    frame: Frame,
    width: SIMDLength,
](
    o: Point3[dtype, frame, width],
    d: Vec3[dtype, frame, width],
    center: Point3[dtype, frame, width],
    radius: SIMD[dtype, width],
    t_max: SIMD[dtype, width],
    t_min: SIMD[dtype, width] = SIMD[dtype, width](1.0e-4),
) -> RayDistanceHit[dtype, width]:
    comptime assert dtype.is_floating_point()

    var oc = o - center

    var a = dot(d, d)
    var half_b = dot(oc, d)
    var c = dot(oc, oc) - radius * radius

    var det = half_b * half_b - a * c
    var det_ok = det.ge(0.0)

    var sqrt_det = sqrt(max(det, 0.0))
    var inv_a = Scalar[dtype](1.0) / a

    var t0 = (-half_b - sqrt_det) * inv_a
    var t1 = (-half_b + sqrt_det) * inv_a

    var near_mask = det_ok & t0.gt(t_min) & t0.lt(t_max)

    var far_mask = det_ok & (~near_mask) & t1.gt(t_min) & t1.lt(t_max)

    var mask = near_mask | far_mask

    var t = near_mask.select(
        t0,
        far_mask.select(t1, max_finite[dtype]()),
    )

    return RayDistanceHit(mask, t)


def intersect_ray_sphere_coefficients[
    dtype: DType,
    frame: Frame,
    width: SIMDLength,
](
    o: Point3[dtype, frame, width],
    d: Vec3[dtype, frame, width],
    center: Point3[dtype, frame, width],
    radius: SIMD[dtype, width],
    a: SIMD[dtype, width],
    inv_a: SIMD[dtype, width],
    t_max: SIMD[dtype, width],
    t_min: SIMD[dtype, width] = SIMD[dtype, width](1.0e-4),
) -> RayDistanceHit[dtype, width]:
    comptime assert dtype.is_floating_point()

    var oc = o - center

    var half_b = dot(oc, d)
    var c = dot(oc, oc) - radius * radius

    var det = half_b * half_b - a * c
    var det_ok = det.ge(0.0)

    var sqrt_det = sqrt(max(det, 0.0))
    var t0 = (-half_b - sqrt_det) * inv_a
    var t1 = (-half_b + sqrt_det) * inv_a

    var near_mask = det_ok & t0.gt(t_min) & t0.lt(t_max)
    var far_mask = det_ok & (~near_mask) & t1.gt(t_min) & t1.lt(t_max)
    var mask = near_mask | far_mask

    var t = near_mask.select(
        t0,
        far_mask.select(t1, max_finite[dtype]()),
    )

    return RayDistanceHit(mask, t)


def intersect_ray_tri[
    dtype: DType, frame: Frame, width: SIMDLength
](
    o: Point3[dtype, frame, width],
    d: Vec3[dtype, frame, width],
    v0: Point3[dtype, frame, width],
    v1: Point3[dtype, frame, width],
    v2: Point3[dtype, frame, width],
    t_max: SIMD[dtype, width],
    t_min: SIMD[dtype, width] = SIMD[dtype, width](1.0e-4),
) -> RayTriHit[dtype, width]:
    """Moller and Trumbore's method."""
    return intersect_ray_tri_edges(
        o,
        d,
        v0,
        v1 - v0,
        v2 - v0,
        t_max,
        t_min,
    )


def intersect_ray_tri_edges[
    dtype: DType, frame: Frame, width: SIMDLength, robust: Bool = False
](
    o: Point3[dtype, frame, width],
    d: Vec3[dtype, frame, width],
    v0: Point3[dtype, frame, width],
    e1: Vec3[dtype, frame, width],
    e2: Vec3[dtype, frame, width],
    t_max: SIMD[dtype, width],
    t_min: SIMD[dtype, width] = SIMD[dtype, width](1.0e-4),
) -> RayTriHit[dtype, width]:
    """Moller-Trumbore intersection with precomputed triangle edges.

    Robust mode uses exact-zero determinants and inclusive intervals. Its
    Float64 scalar specialization backs spatial queries without an absolute
    scale cutoff. The default retains the rendering predicate's policy.
    """
    comptime assert dtype.is_floating_point()
    comptime EPSILON = Scalar[dtype](1e-8 if dtype == .float32 else 1e-16)
    comptime INF = max_finite[dtype]()

    var p = cross(d, e2)
    var det = dot(e1, p)

    var det_ok = det.ne(0.0) if robust else det.gt(EPSILON) | det.lt(-EPSILON)
    comptime if robust and width == 1:
        if not det_ok[0]:
            return RayTriHit(det_ok, SIMD[dtype, width](INF), 0.0, 0.0)
    var inv_det = 1.0 / det

    var tv = o - v0
    var u = dot(tv, p) / det if robust else dot(tv, p) * inv_det
    comptime if robust and width == 1 and not is_gpu():
        if u[0] < 0.0 or u[0] > 1.0:
            return RayTriHit(
                SIMD[.bool, width](fill=False), SIMD[dtype, width](INF), u, 0.0
            )

    var q = cross(tv, e1)
    var v = dot(d, q) / det if robust else dot(d, q) * inv_det
    comptime if robust and width == 1 and not is_gpu():
        if v[0] < 0.0 or u[0] + v[0] > 1.0:
            return RayTriHit(
                SIMD[.bool, width](fill=False), SIMD[dtype, width](INF), u, v
            )
    var t = dot(e2, q) / det if robust else dot(e2, q) * inv_det

    var mask = det_ok & u.ge(0.0) & u.le(1.0) & v.ge(0.0) & (u + v).le(1.0)
    comptime if robust:
        mask &= t.ge(t_min) & t.le(t_max)
    else:
        mask &= t.gt(t_min) & t.lt(t_max)

    return RayTriHit(
        mask,
        mask.select(t, INF),
        u,
        v,
    )


def intersect_ray_tri_edges_scaled[
    dtype: DType, frame: Frame, width: SIMDLength
](
    o: Point3[dtype, frame, width],
    d: Vec3[dtype, frame, width],
    v0: Point3[dtype, frame, width],
    e1: Vec3[dtype, frame, width],
    e2: Vec3[dtype, frame, width],
    t_max: SIMD[dtype, width],
    t_min: SIMD[dtype, width] = SIMD[dtype, width](1.0e-4),
) -> RayTriScaledHit[dtype, width]:
    """Return determinant-scaled Moller-Trumbore candidates."""
    comptime assert dtype.is_floating_point()
    comptime EPSILON = Scalar[dtype](1e-8 if dtype == .float32 else 1e-16)

    var p = cross(d, e2)
    var det = dot(e1, p)
    var positive_det = det.ge(0.0)
    var abs_det = positive_det.select(det, -det)

    var tv = o - v0
    var u_num = dot(tv, p)
    var u_scaled = positive_det.select(u_num, -u_num)

    var q = cross(tv, e1)
    var v_num = dot(d, q)
    var t_num = dot(e2, q)
    var v_scaled = positive_det.select(v_num, -v_num)
    var t_scaled = positive_det.select(t_num, -t_num)

    var mask = (
        abs_det.gt(EPSILON)
        & u_scaled.ge(0.0)
        & u_scaled.le(abs_det)
        & v_scaled.ge(0.0)
        & (u_scaled + v_scaled).le(abs_det)
        & t_scaled.gt(t_min * abs_det)
        & t_scaled.lt(t_max * abs_det)
    )

    return RayTriScaledHit(
        mask,
        abs_det,
        t_scaled,
        u_scaled,
        v_scaled,
    )


def intersect_ray_tri[
    dtype: DType, frame: Frame
](
    vertices: ImmSpan[Scalar[dtype], _],
    prim_idx: UInt32,
    o: Point3[dtype, frame],
    d: Vec3[dtype, frame],
    t_max: Scalar[dtype],
) -> RayTriHit[dtype, 1]:
    var base = Int(prim_idx) * 9
    debug_assert["safe", _use_compiler_assume=True](
        base >= 0 and base <= len(vertices) - 9,
        "triangle load is outside the input span",
    )
    var v0 = Point3[frame=frame](vertices.unsafe_ptr().unsafe_offset(base))
    var v1 = Point3[frame=frame](vertices.unsafe_ptr().unsafe_offset(base + 3))
    var v2 = Point3[frame=frame](vertices.unsafe_ptr().unsafe_offset(base + 6))

    return intersect_ray_tri(
        o,
        d,
        v0,
        v1,
        v2,
        t_max,
    )


def edge_edge_test[
    dtype: DType, frame: Frame, width: SIMDLength
](
    v0: Vec3[dtype, frame, width],
    u0: Vec3[dtype, frame, width],
    u1: Vec3[dtype, frame, width],
    i0: Int,
    i1: Int,
    Ax: SIMD[dtype, width],
    Ay: SIMD[dtype, width],
) -> Bool:
    comptime assert width == 1

    var Bx = u0[i0] - u1[i0]
    var By = u0[i1] - u1[i1]
    var Cx = v0[i0] - u0[i0]
    var Cy = v0[i1] - u0[i1]

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


def edge_against_tri_edges[
    dtype: DType, frame: Frame, width: SIMDLength
](
    v0: Vec3[dtype, frame, width],
    v1: Vec3[dtype, frame, width],
    u0: Vec3[dtype, frame, width],
    u1: Vec3[dtype, frame, width],
    u2: Vec3[dtype, frame, width],
    i0: Int,
    i1: Int,
) -> Bool:
    comptime assert width == 1

    var Ax = v1[i0] - v0[i0]
    var Ay = v1[i1] - v0[i1]

    if edge_edge_test(v0, u0, u1, i0, i1, Ax, Ay):
        return True
    if edge_edge_test(v0, u1, u2, i0, i1, Ax, Ay):
        return True
    if edge_edge_test(v0, u2, u0, i0, i1, Ax, Ay):
        return True

    return False


def _point_in_tri_check[
    dtype: DType, frame: Frame, width: SIMDLength
](
    v0: Vec3[dtype, frame, width],
    p1: Vec3[dtype, frame, width],
    p2: Vec3[dtype, frame, width],
    i0: Int,
    i1: Int,
) -> SIMD[dtype, width]:
    var a = p2[i1] - p1[i1]
    var b = -(p2[i0] - p1[i0])
    var c = -a * p1[i0] - b * p1[i1]
    return a * v0[i0] + b * v0[i1] + c


def point_in_tri[
    dtype: DType, frame: Frame, width: SIMDLength
](
    v0: Vec3[dtype, frame, width],
    u0: Vec3[dtype, frame, width],
    u1: Vec3[dtype, frame, width],
    u2: Vec3[dtype, frame, width],
    i0: Int,
    i1: Int,
) -> Bool:
    comptime assert width == 1

    var d0 = _point_in_tri_check(v0, u0, u1, i0, i1)
    var d1 = _point_in_tri_check(v0, u1, u2, i0, i1)
    var d2 = _point_in_tri_check(v0, u2, u0, i0, i1)

    return (d0 * d1 > 0.0) and (d0 * d2 > 0.0)


def coplanar_tri_tri[
    dtype: DType, frame: Frame, width: SIMDLength
](
    n: Vec3[dtype, frame, width],
    v0: Vec3[dtype, frame, width],
    v1: Vec3[dtype, frame, width],
    v2: Vec3[dtype, frame, width],
    u0: Vec3[dtype, frame, width],
    u1: Vec3[dtype, frame, width],
    u2: Vec3[dtype, frame, width],
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
struct Intervals[dtype: DType, width: SIMDLength]:
    var a: SIMD[Self.dtype, Self.width]
    var b: SIMD[Self.dtype, Self.width]
    var c: SIMD[Self.dtype, Self.width]
    var x0: SIMD[Self.dtype, Self.width]
    var x1: SIMD[Self.dtype, Self.width]
    var is_coplanar: Bool


def get_intervals[
    dtype: DType, width: SIMDLength
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


def no_div_tri_tri_isect[
    dtype: DType, frame: Frame, width: SIMDLength
](
    v0: Vec3[dtype, frame, width],
    v1: Vec3[dtype, frame, width],
    v2: Vec3[dtype, frame, width],
    u0: Vec3[dtype, frame, width],
    u1: Vec3[dtype, frame, width],
    u2: Vec3[dtype, frame, width],
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
    var index = longest_axis[absolute=True](d_dir)

    var res1 = get_intervals(
        v0[index],
        v1[index],
        v2[index],
        dv0,
        dv1,
        dv2,
        dv0 * dv1,
        dv0 * dv2,
    )

    if res1.is_coplanar:
        return coplanar_tri_tri(n1, v0, v1, v2, u0, u1, u2)

    var res2 = get_intervals(
        u0[index],
        u1[index],
        u2[index],
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
        swap(s1_0, s1_1)

    if s2_0 > s2_1:
        swap(s2_0, s2_1)

    return not (s1_1 < s2_0 or s2_1 < s1_0)


def closest_point_edge_edge[
    dtype: DType, frame: Frame, width: SIMDLength
](
    p1: Vec3[dtype, frame, width],
    q1: Vec3[dtype, frame, width],
    p2: Vec3[dtype, frame, width],
    q2: Vec3[dtype, frame, width],
    epsilon: SIMD[dtype, width],
) -> Vec3[dtype, frame, width]:
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
        return Vec3[dtype, frame, width](s, t, dist)

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

    return Vec3[dtype, frame, width](s, t, dist)


# Robust Float32 geometry uses Float64 intermediates, inclusive ray intervals
# and exact-zero degeneracy tests. The rendering predicates above retain their
# dtype-specific determinant tolerance and open ray intervals.


def vector64[
    frame: Frame, width: SIMDLength
](p: Point3[.float32, frame, width]) -> Vec3[.float64, frame, width]:
    """Convert scalar or packet point coordinates to Float64 vectors."""
    return Vec3[.float64, frame, width](
        p.x.cast[.float64](), p.y.cast[.float64](), p.z.cast[.float64]()
    )


@fieldwise_init
struct TriangleProjection[frame: Frame](TrivialRegisterPassable):
    var point: Vec3[.float64, Self.frame]
    var u: Float64
    var v: Float64
    var distance2: Float64


def project_triangle[
    frame: Frame
](
    point: Point3[.float32, frame],
    a: Point3[.float32, frame],
    b: Point3[.float32, frame],
    c: Point3[.float32, frame],
) -> TriangleProjection[frame]:
    var p = vector64(point)
    var av = vector64(a)
    var bv = vector64(b)
    var cv = vector64(c)
    var ab = bv - av
    var ac = cv - av
    var normal = cross(ab, ac)
    var area2 = dot(normal, normal)
    if area2 > 0.0:
        var q = p - normal * (dot(p - av, normal) / area2)
        var weight_b = dot(cross(q - av, ac), normal) / area2
        var weight_c = dot(cross(ab, q - av), normal) / area2
        var weight_a = 1.0 - weight_b - weight_c
        if weight_a >= 0.0 and weight_b >= 0.0 and weight_c >= 0.0:
            var delta = q - p
            return TriangleProjection(q, weight_a, weight_b, dot(delta, delta))
    var best = TriangleProjection(
        av, Float64(1.0), Float64(0.0), dot(av - p, av - p)
    )
    comptime for edge in range(3):
        var start = av
        var end = bv
        comptime match edge:
            case 1:
                start = bv
                end = cv
            case 2:
                start = cv
                end = av
            case _:
                pass
        var direction = end - start
        var length2 = dot(direction, direction)
        var t = Float64(0.0)
        if length2 > 0.0:
            t = (dot(p - start, direction) / length2).clamp(0.0, 1.0)
        var q = start + direction * t
        var distance2 = dot(q - p, q - p)
        if distance2 < best.distance2:
            var u = Float64(1.0) - t
            var v = t
            comptime match edge:
                case 1:
                    u = 0.0
                    v = 1.0 - t
                case 2:
                    u = t
                    v = 0.0
                case _:
                    pass
            best = TriangleProjection(q, u, v, distance2)
    return best


@fieldwise_init
struct TriangleRayHit(TrivialRegisterPassable):
    var result: Bool
    var t: Float64
    var u: Float64
    var v: Float64


def intersect_triangle[
    frame: Frame
](
    ray: Ray[.float32, frame],
    a: Point3[.float32, frame],
    b: Point3[.float32, frame],
    c: Point3[.float32, frame],
) -> TriangleRayHit:
    var av = vector64(a)
    var hit = intersect_ray_tri_edges[.float64, frame, 1, True](
        Point3[.float64, frame](
            Float64(ray.o.x), Float64(ray.o.y), Float64(ray.o.z)
        ),
        Vec3[.float64, frame](
            Float64(ray.d.x), Float64(ray.d.y), Float64(ray.d.z)
        ),
        Point3[.float64, frame](av.x, av.y, av.z),
        vector64(b) - av,
        vector64(c) - av,
        Float64(ray.t_max),
        Float64(ray.t_min),
    )
    return TriangleRayHit(
        hit.mask[0], hit.t[0], 1.0 - hit.u[0] - hit.v[0], hit.u[0]
    )


def triangle_normal[
    frame: Frame
](
    a: Point3[.float32, frame],
    b: Point3[.float32, frame],
    c: Point3[.float32, frame],
) -> Vec3[.float64, frame]:
    var normal = cross(vector64(b) - vector64(a), vector64(c) - vector64(a))
    var length2 = dot(normal, normal)
    if length2 > 0.0:
        return normal / sqrt(length2)
    return Vec3[.float64, frame](0.0)


@always_inline
def angle_atan2[
    width: SIMDLength = 1
](y: SIMD[.float64, width], x: SIMD[.float64, width]) -> SIMD[.float64, width]:
    """Portable scalar/SIMD double atan2 without a host libm dependency.

    Reduce to |r| <= tan(pi/16), then evaluate 11 alternating-series terms.
    The truncation error is below 2e-17; width 1 is the scalar specialization.
    """
    comptime pi_2 = pi / 2
    comptime pi_4 = pi / 4
    comptime pi_8 = pi / 8
    # Half-angle identities avoid tan's host libm call at compile time.
    comptime tan_pi_8 = 1 / (sqrt(Float64(2)) + 1)
    comptime tan_pi_16 = tan_pi_8 / (sqrt(1 + tan_pi_8 * tan_pi_8) + 1)
    comptime tan_3pi_16 = (1 - tan_pi_16) / (1 + tan_pi_16)

    var ax = abs(x)
    var ay = abs(y)
    var swapped = ay.gt(ax)
    var maximum = max(ax, ay)
    var ratio = min(ax, ay) / maximum.eq(0.0).select(1.0, maximum)
    var upper = ratio.gt(tan_3pi_16)
    var middle = ratio.gt(tan_pi_16)
    var pivot = upper.select(1.0, middle.select(tan_pi_8, 0.0))
    var offset = upper.select(pi_4, middle.select(pi_8, 0.0))
    ratio = (ratio - pivot) / (1.0 + ratio * pivot)
    var ratio2 = ratio * ratio
    var polynomial = SIMD[.float64, width](1.0 / 21.0)
    comptime for term in range(9, -1, -1):
        polynomial = fma(
            polynomial,
            ratio2,
            SIMD[.float64, width](
                (1.0 if term % 2 == 0 else -1.0) / Float64(2 * Int(term) + 1)
            ),
        )
    var angle = fma(ratio, polynomial, offset)
    angle = swapped.select(pi_2 - angle, angle)
    angle = x.lt(0.0).select(pi - angle, angle)
    return y.lt(0.0).select(-angle, angle)


@always_inline
def solid_angle[
    frame: Frame, width: SIMDLength = 1
](
    point: Point3[.float32, frame],
    a: Vec3[.float64, frame, width],
    b: Vec3[.float64, frame, width],
    c: Vec3[.float64, frame, width],
) -> SIMD[.float64, width]:
    """Signed triangle solid angle, specialized for scalar or SIMD vertices."""
    var p = Vec3[.float64, frame, width](
        SIMD[.float64, width](Float64(point.x)),
        SIMD[.float64, width](Float64(point.y)),
        SIMD[.float64, width](Float64(point.z)),
    )
    var av = a - p
    var bv = b - p
    var cv = c - p
    var numerator = dot(av, cross(bv, cv))
    var la = length(av)
    var lb = length(bv)
    var lc = length(cv)
    var denominator = (
        la * lb * lc + dot(av, bv) * lc + dot(bv, cv) * la + dot(cv, av) * lb
    )
    return numerator.eq(0.0).select(
        0.0, 2.0 * angle_atan2(numerator, denominator)
    )


@always_inline
def ray_triangle_filter[
    frame: Frame
](
    ray: Ray[.float32, frame],
    a: Point3[.float32, frame],
    b: Point3[.float32, frame],
    c: Point3[.float32, frame],
) -> Int:
    """Certified miss/hit, or -1 when the Float64 predicate is required."""
    return ray_triangle_interval(ray, a, b, c)[0]


@always_inline
def ray_triangle_interval[
    frame: Frame
](
    ray: Ray[.float32, frame],
    a: Point3[.float32, frame],
    b: Point3[.float32, frame],
    c: Point3[.float32, frame],
) -> Tuple[Int, Float32, Float32]:
    """Return (status, distance, error); a certified hit has status 1.

    The distance interval encloses the Float64 hit. Status 0 means a certified
    miss and -1 requires Float64; their distance/error fields are unused.
    Certification excludes determinant, barycentric and interval boundaries.

    Absolute forward-error margins use the magnitudes of all triple products,
    so cancellation does not make the filter overconfident. Subnormal and
    overflowing arithmetic stays on the robust double-precision path.
    """
    var e1 = b - a
    var e2 = c - a
    var tv = ray.o - a
    var m1 = e1.abs().max_component()
    var m2 = e2.abs().max_component()
    var mt = tv.abs().max_component()
    var md = ray.d.abs().max_component()
    var scale = max(max(m1, m2), max(mt, md))
    if not isfinite(scale) or scale > 1e10 or scale < 1e-10:
        return (-1, 0.0, 0.0)
    var de = m1 * m2 * md * 0.000008
    var ue = mt * m2 * md * 0.000008
    var ve = mt * m1 * md * 0.000008
    var te = mt * m1 * m2 * 0.000008
    if de < 1e-30 or ue < 1e-30 or ve < 1e-30 or te < 1e-30:
        return (-1, 0.0, 0.0)
    var p = cross(ray.d, e2)
    var det = dot(e1, p)
    var ad = abs(det)
    if ad <= de:
        return (-1, 0.0, 0.0)
    var orientation = Float32(1.0 if det > 0.0 else -1.0)
    var u = dot(tv, p) * orientation
    var q = cross(tv, e1)
    var v = dot(ray.d, q) * orientation
    var w = ad - u - v
    var we = de + ue + ve
    if u < -ue or v < -ve or w < -we:
        return (0, 0.0, 0.0)
    var ts = dot(e2, q) * orientation
    var t = ts / ad
    var error = (te + abs(t) * de) / (ad - de) + abs(t) * 0.000002
    if not isfinite(error) or error < 1e-30:
        return (-1, 0.0, 0.0)
    if t < ray.t_min - error or t > ray.t_max + error:
        return (0, 0.0, 0.0)
    if (
        u > ue
        and v > ve
        and w > we
        and t > ray.t_min + error
        and t < ray.t_max - error
    ):
        return (1, t, error)
    return (-1, 0.0, 0.0)


@always_inline
def project_triangle_filtered[
    frame: Frame
](
    point: Point3[.float32, frame],
    a: Point3[.float32, frame],
    b: Point3[.float32, frame],
    c: Point3[.float32, frame],
) -> Tuple[TriangleProjection[frame], Float64]:
    var hit, error, _ = project_triangle_filtered_with_exactness(point, a, b, c)
    return (hit, error)


@always_inline
def project_triangle_filtered_with_exactness[
    frame: Frame
](
    point: Point3[.float32, frame],
    a: Point3[.float32, frame],
    b: Point3[.float32, frame],
    c: Point3[.float32, frame],
) -> Tuple[TriangleProjection[frame], Float64, Bool]:
    """Filtered region projection for well-conditioned faces.

    Ill-conditioned geometry retains the double cross-product predicate.
    Close comparisons and provisional winners are refined with that predicate too.
    The third result is true only when project_triangle computed every field;
    certified vertex results still need its canonical endpoint/barycentric form.
    """
    var ab32 = b - a
    var ac32 = c - a
    var ap32 = point - a
    var bp32 = point - b
    var cp32 = point - c
    var edge_scale = Float32(0.0)
    var point_scale = Float32(0.0)
    comptime for axis in range(3):
        edge_scale = max(edge_scale, max(abs(ab32[axis]), abs(ac32[axis])))
        point_scale = max(
            point_scale,
            max(abs(ap32[axis]), max(abs(bp32[axis]), abs(cp32[axis]))),
        )
    var margin = edge_scale * point_scale * 0.000008
    if isfinite(margin) and margin > 1e-30:
        var d1 = dot(ab32, ap32)
        var d2 = dot(ac32, ap32)
        var d3 = dot(ab32, bp32)
        var d4 = dot(ac32, bp32)
        var d5 = dot(ab32, cp32)
        var d6 = dot(ac32, cp32)
        var corner = -1
        if d1 < -margin and d2 < -margin:
            corner = 0
        elif d3 > margin and d3 - d4 > margin * 2.0:
            corner = 1
        elif d6 > margin and d6 - d5 > margin * 2.0:
            corner = 2
        if corner >= 0:
            var vertex = a if corner == 0 else b if corner == 1 else c
            var q = vector64(vertex)
            var delta = q - vector64(point)
            return (
                TriangleProjection(
                    q,
                    Float64(1.0 if corner == 0 else 0.0),
                    Float64(1.0 if corner == 1 else 0.0),
                    dot(delta, delta),
                ),
                0.0,
                False,
            )
    var n32 = cross(ab32, ac32)
    var edge2 = max(dot(ab32, ab32), dot(ac32, ac32))
    var area2 = dot(n32, n32)
    var magnitude = max(edge_scale, point_scale)
    comptime for axis in range(3):
        magnitude = max(
            magnitude,
            max(
                max(abs(a[axis]), abs(b[axis])),
                max(abs(c[axis]), abs(point[axis])),
            ),
        )
    if (
        not isfinite(area2)
        or magnitude > 1e8
        or magnitude < 1e-8
        or area2 <= edge2 * edge2 * 0.0625
    ):
        return (project_triangle(point, a, b, c), 0.0, True)
    var av = Vec3[.float32, frame](a.x, a.y, a.z)
    var bv = Vec3[.float32, frame](b.x, b.y, b.z)
    var cv = Vec3[.float32, frame](c.x, c.y, c.z)
    var pv = Vec3[.float32, frame](point.x, point.y, point.z)
    var uv = closest_point_to_triangle(av, bv, cv, pv)
    var q = av + ab32 * uv.y + ac32 * (1.0 - uv.x - uv.y)
    var delta = q - pv
    var distance2 = dot(delta, delta)
    var position_error = magnitude * 0.00001
    var error = (
        2.0 * sqrt(distance2) + position_error
    ) * position_error + distance2 * 0.000002
    if not isfinite(distance2) or not isfinite(error):
        return (project_triangle(point, a, b, c), 0.0, True)
    return (
        TriangleProjection(
            Vec3[.float64, frame](Float64(q.x), Float64(q.y), Float64(q.z)),
            Float64(uv.x),
            Float64(uv.y),
            Float64(distance2),
        ),
        Float64(error),
        False,
    )


@always_inline
def triangle_normal_f32[
    frame: Frame
](
    a: Point3[.float32, frame],
    b: Point3[.float32, frame],
    c: Point3[.float32, frame],
) -> Vec3[.float32, frame]:
    """Robust normal with exact Float32 shortcuts for structured directions.

    Float32 input coordinates bound the nonzero Float64 cross product away
    from overflow/underflow. Axis directions round to +/-1. Equal-magnitude nonzero
    components round to +/-1/sqrt(count), far from a Float32 rounding boundary.
    Other normals retain the original Float64 normalization.
    """
    var n = cross(vector64(b) - vector64(a), vector64(c) - vector64(a))
    var nonzero = Int(n.x != 0.0) + Int(n.y != 0.0) + Int(n.z != 0.0)
    var magnitude = max(abs(n.x), max(abs(n.y), abs(n.z)))
    if (
        isfinite(magnitude)
        and magnitude > 0.0
        and (
            (n.x == 0.0 or abs(n.x) == magnitude)
            and (n.y == 0.0 or abs(n.y) == magnitude)
            and (n.z == 0.0 or abs(n.z) == magnitude)
        )
    ):
        comptime unit2 = Float32(1.0 / sqrt(Float64(2.0)))
        comptime unit3 = Float32(1.0 / sqrt(Float64(3.0)))
        var unit = (
            Float32(1.0) if nonzero == 1 else unit2 if nonzero == 2 else unit3
        )
        var result = Vec3[.float32, frame](0.0)
        comptime for axis in range(3):
            result[axis] = (
                Float32(n[axis]) if n[axis]
                == 0.0 else -unit if n[axis]
                < 0.0 else unit
            )
        return result
    var length2 = dot(n, n)
    if length2 > 0.0:
        n = n / sqrt(length2)
        return Vec3[.float32, frame](Float32(n.x), Float32(n.y), Float32(n.z))
    return Vec3[.float32, frame](0.0)


@always_inline
def normal_sign_filter[
    frame: Frame
](
    point: Point3[.float32, frame],
    closest: Point3[.float32, frame],
    a: Point3[.float32, frame],
    b: Point3[.float32, frame],
    c: Point3[.float32, frame],
) -> Int:
    """Return +/-1 only for a certified original Float64 normal-dot sign.

    Return zero for uncertain, nonfinite or out-of-range inputs. The caller
    preserves the original normalized Float64 calculation for those cases.
    """
    var largest = UInt32(0)
    comptime for axis in range(3):
        var values = (point[axis], closest[axis], a[axis], b[axis], c[axis])
        comptime for i in range(5):
            var value = values[i]
            largest = max(largest, bitcast[.uint32](value) & UInt32(0x7FFFFFFF))
    if largest > UInt32(0x501502F9):
        return 0
    var e1 = b - a
    var e2 = c - a
    var delta = point - closest
    var total = Float32(0)
    var permanent = Float32(0)
    comptime for axis in range(3):
        comptime j = (axis + 1) % 3
        comptime k = (axis + 2) % 3
        var normal = e1[j] * e2[k] - e1[k] * e2[j]
        total += normal * delta[axis]
        var upper = abs(e1[j] * e2[k]) + abs(e1[k] * e2[j])
        permanent += upper * abs(delta[axis])
    # P sums absolute cross-product terms times the absolute displacement.
    # 2^-12 * P is deliberately much larger than gamma_64 for Float32,
    # covering edge/delta subtraction, cross/dot rounding, and the much
    # smaller Float64 normalization/dot error. FMA contraction is included.
    # The input limit M <= 1e10 prevents overflow. Even a loose 256 *
    # min_normal_f32 * M^2 allowance for underflow/FTZ is below 1e-14.
    # A positive normal length preserves sign; degeneracy stays uncertain.
    var error = permanent * Float32(0.000244140625) + Float32(1e-14)
    if total > error:
        return 1
    if total < -error:
        return -1
    return 0


@always_inline
def ray_overlaps_bounds[
    frame: Frame
](
    ray: Ray[.float32, frame],
    bounds: AxisAlignedBoundingBox[.float32, frame],
    cutoff: Float32,
    radius: Float32 = 0.0,
) -> Bool:
    """Conservative expanded-box slab test; handles parallel and zero rays."""
    var lower = Float64(ray.t_min)
    var upper = Float64(min(ray.t_max, cutoff))
    if lower > upper:
        return False
    comptime for axis in range(3):
        var origin = Float64(ray.o[axis])
        var direction = Float64(ray.d[axis])
        var lo = Float64(bounds._min[axis]) - Float64(radius)
        var hi = Float64(bounds._max[axis]) + Float64(radius)
        if direction == 0.0:
            if origin < lo or origin > hi:
                return False
        else:
            var a = (lo - origin) / direction
            var b = (hi - origin) / direction
            lower = max(lower, min(a, b))
            upper = min(upper, max(a, b))
            if lower > upper:
                return False
    return True


@always_inline
def _gpu_ray_bounds_interval[
    frame: Frame
](
    ray: Ray[.float32, frame],
    bounds: AxisAlignedBoundingBox[.float32, frame],
    cutoff: Float32,
    radius: Float32,
    inverse: Vec3[.float64, frame],
) -> Tuple[Int, Float32]:
    """Return 0/1 for separated intervals, -1 when Float64 is needed.

    Bound both interval endpoints by the largest per-axis error. Strict
    comparisons use twice that error; uncertain intervals retain Float64.
    The generous margin
    covers Float32 subtraction, reciprocal conversion and multiplication;
    overflow, underflow and touching intervals use the original exact test.
    """
    var lower = ray.t_min
    var upper = min(ray.t_max, cutoff)
    var interval_error = Float32(0.0)
    if lower > upper:
        return (0, 0.0)
    comptime for axis in range(3):
        var origin = ray.o[axis]
        var lo = bounds._min[axis] - radius
        var hi = bounds._max[axis] + radius
        var scale = max(abs(bounds._min[axis]), abs(bounds._max[axis]))
        var error = (scale + abs(radius) + abs(origin)) * 0.000002
        if not isfinite(error) or error < 1e-30:
            return (-1, 0.0)
        if ray.d[axis] == 0.0:
            if origin < lo - error or origin > hi + error:
                return (0, 0.0)
            if origin <= lo + error or origin >= hi - error:
                return (-1, 0.0)
        else:
            var direction = abs(ray.d[axis])
            if direction < 1e-30 or direction > 1e30:
                return (-1, 0.0)
            var inv = Float32(inverse[axis])
            var a = (lo - origin) * inv
            var b = (hi - origin) * inv
            error = error * abs(inv) + max(abs(a), abs(b)) * 0.000002
            if not isfinite(error) or error < 1e-30:
                return (-1, 0.0)
            interval_error = max(interval_error, error)
            lower = max(lower, min(a, b))
            upper = min(upper, max(a, b))
            if lower > upper + interval_error * 2.0:
                return (0, 0.0)
    if lower < upper - interval_error * 2.0:
        return (1, lower - interval_error)
    return (-1, 0.0)


@always_inline
def _ray_overlaps_bounds_cached_impl[
    frame: Frame, width: SIMDLength = 1
](
    ray: Ray[.float32, frame],
    bounds: AxisAlignedBoundingBox[.float32, frame, width],
    cutoff: Float32,
    radius: Float32,
    inverse: Vec3[.float64, frame],
    active: SIMD[.bool, width] = SIMD[.bool, width](fill=True),
) -> SIMD[.bool, width]:
    var lower = SIMD[.float64, width](Float64(ray.t_min))
    var upper = SIMD[.float64, width](Float64(min(ray.t_max, cutoff)))
    var valid = active & lower.le(upper)
    comptime if width == 1:
        if not valid[0]:
            return valid
    comptime for axis in range(3):
        var origin = SIMD[.float64, width](Float64(ray.o[axis]))
        var lo = bounds._min[axis].cast[.float64]() - Float64(radius)
        var hi = bounds._max[axis].cast[.float64]() + Float64(radius)
        if ray.d[axis] == 0.0:
            valid &= origin.ge(lo) & origin.le(hi)
            comptime if width == 1:
                if not valid[0]:
                    return valid
        else:
            var a = (lo - origin) * inverse[axis]
            var b = (hi - origin) * inverse[axis]
            lower = max(lower, min(a, b))
            upper = min(upper, max(a, b))
            comptime if width == 1:
                if lower[0] > upper[0]:
                    if (
                        lower[0] - upper[0]
                        <= max(abs(lower[0]), abs(upper[0])) * 1e-14
                    ):
                        var box = AxisAlignedBoundingBox[.float32, frame](
                            Point3[.float32, frame](
                                bounds._min.x[0],
                                bounds._min.y[0],
                                bounds._min.z[0],
                            ),
                            Point3[.float32, frame](
                                bounds._max.x[0],
                                bounds._max.y[0],
                                bounds._max.z[0],
                            ),
                        )
                        return SIMD[.bool, width](
                            fill=ray_overlaps_bounds(ray, box, cutoff, radius)
                        )
                    return SIMD[.bool, width](fill=False)
    var result = valid & lower.le(upper)
    var boundary = abs(upper - lower).le(max(abs(lower), abs(upper)) * 1e-14)
    comptime for lane in range(width):
        if active[lane] and boundary[lane]:
            var box = AxisAlignedBoundingBox[.float32, frame](
                Point3[.float32, frame](
                    bounds._min.x[lane],
                    bounds._min.y[lane],
                    bounds._min.z[lane],
                ),
                Point3[.float32, frame](
                    bounds._max.x[lane],
                    bounds._max.y[lane],
                    bounds._max.z[lane],
                ),
            )
            result[lane] = ray_overlaps_bounds(ray, box, cutoff, radius)
    return result


@no_inline
def _ray_overlaps_bounds_cached_exact[
    frame: Frame, width: SIMDLength = 1
](
    ray: Ray[.float32, frame],
    bounds: AxisAlignedBoundingBox[.float32, frame, width],
    cutoff: Float32,
    radius: Float32,
    inverse: Vec3[.float64, frame],
    active: SIMD[.bool, width] = SIMD[.bool, width](fill=True),
) -> SIMD[.bool, width]:
    # Keep uncertain GPU slabs in a separate function; the common Float32
    # traversal path need not inline the full Float64 boundary calculation.
    return _ray_overlaps_bounds_cached_impl(
        ray, bounds, cutoff, radius, inverse, active
    )


@always_inline
def ray_overlaps_bounds_cached[
    frame: Frame, width: SIMDLength = 1
](
    ray: Ray[.float32, frame],
    bounds: AxisAlignedBoundingBox[.float32, frame, width],
    cutoff: Float32,
    radius: Float32,
    inverse: Vec3[.float64, frame],
    active: SIMD[.bool, width] = SIMD[.bool, width](fill=True),
) -> SIMD[.bool, width]:
    """Cached scalar/SIMD slabs with division fallback near interval boundaries.
    """
    comptime if is_gpu() and width == 1:
        if not active[0]:
            return active
        var box = AxisAlignedBoundingBox[.float32, frame](
            Point3[.float32, frame](
                bounds._min.x[0], bounds._min.y[0], bounds._min.z[0]
            ),
            Point3[.float32, frame](
                bounds._max.x[0], bounds._max.y[0], bounds._max.z[0]
            ),
        )
        var filtered = _gpu_ray_bounds_interval(
            ray, box, cutoff, radius, inverse
        )[0]
        if filtered >= 0:
            return SIMD[.bool, width](fill=filtered == 1)
        return _ray_overlaps_bounds_cached_exact(
            ray, bounds, cutoff, radius, inverse, active
        )
    return _ray_overlaps_bounds_cached_impl(
        ray, bounds, cutoff, radius, inverse, active
    )


@no_inline
def _sphere_overlaps_exact[
    frame: Frame, width: SIMDLength
](
    center: Point3[.float32, frame],
    radius: Float32,
    bounds: AxisAlignedBoundingBox[.float32, frame, width],
) -> SIMD[.bool, width]:
    # Preserve touching and extreme-scale results without inlining this
    # calculation into every node visited by the GPU Float32 filter.
    return bounds.distance2(center).le(Float64(radius) * Float64(radius))


@always_inline
def sphere_overlaps_bounds[
    frame: Frame, width: SIMDLength = 1
](
    center: Point3[.float32, frame],
    radius: Float32,
    bounds: AxisAlignedBoundingBox[.float32, frame, width],
) -> SIMD[.bool, width]:
    comptime if is_gpu() and width == 1:
        var distance2 = Float32(0.0)
        comptime for axis in range(3):
            var delta = max(
                max(
                    bounds._min[axis][0] - center[axis],
                    center[axis] - bounds._max[axis][0],
                ),
                Float32(0.0),
            )
            distance2 += delta * delta
        var radius2 = radius * radius
        var scale = max(distance2, radius2)
        # Normal finite arithmetic away from tangency is unambiguous. Retain
        # Float64 for tiny/overflowing distances and touching spheres.
        if isfinite(scale) and scale > 1e-30:
            if abs(distance2 - radius2) > scale * 0.000002:
                return SIMD[.bool, width](fill=distance2 < radius2)
        return _sphere_overlaps_exact(center, radius, bounds)
    return bounds.distance2(center).le(Float64(radius) * Float64(radius))
