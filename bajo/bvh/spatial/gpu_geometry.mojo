"""Filtered single-precision predicates; uncertain geometry uses Float64."""

from std.math import abs, max, sqrt
from std.memory import bitcast
from std.utils.numerics import isfinite
from bajo.core import Frame, Rayf32, Point3f32, Vec3f32, Vec3, cross, dot

from bajo.core.intersect import (
    TriangleProjection,
    project_triangle,
    closest_point_to_triangle,
    vector64,
)


@always_inline
def ray_triangle_filter[
    frame: Frame
](
    ray: Rayf32[frame],
    a: Point3f32[frame],
    b: Point3f32[frame],
    c: Point3f32[frame],
) -> Int:
    """Certified miss/hit, or -1 when the Float64 predicate is required."""
    return ray_triangle_interval(ray, a, b, c)[0]


@always_inline
def ray_triangle_interval[
    frame: Frame
](
    ray: Rayf32[frame],
    a: Point3f32[frame],
    b: Point3f32[frame],
    c: Point3f32[frame],
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
    var m1 = max(max(abs(e1.x), abs(e1.y)), abs(e1.z))
    var m2 = max(max(abs(e2.x), abs(e2.y)), abs(e2.z))
    var mt = max(max(abs(tv.x), abs(tv.y)), abs(tv.z))
    var md = max(max(abs(ray.d.x), abs(ray.d.y)), abs(ray.d.z))
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
    point: Point3f32[frame],
    a: Point3f32[frame],
    b: Point3f32[frame],
    c: Point3f32[frame],
) -> Tuple[TriangleProjection[frame], Float64]:
    var hit, error, _ = project_triangle_filtered_with_exactness(point, a, b, c)
    return (hit, error)


@always_inline
def project_triangle_filtered_with_exactness[
    frame: Frame
](
    point: Point3f32[frame],
    a: Point3f32[frame],
    b: Point3f32[frame],
    c: Point3f32[frame],
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
    var av = Vec3f32[frame](a.x, a.y, a.z)
    var bv = Vec3f32[frame](b.x, b.y, b.z)
    var cv = Vec3f32[frame](c.x, c.y, c.z)
    var pv = Vec3f32[frame](point.x, point.y, point.z)
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
](a: Point3f32[frame], b: Point3f32[frame], c: Point3f32[frame]) -> Vec3f32[
    frame
]:
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
        var result = Vec3f32[frame](0.0)
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
        return Vec3f32[frame](Float32(n.x), Float32(n.y), Float32(n.z))
    return Vec3f32[frame](0.0)


@always_inline
def normal_sign_filter[
    frame: Frame
](
    point: Point3f32[frame],
    closest: Point3f32[frame],
    a: Point3f32[frame],
    b: Point3f32[frame],
    c: Point3f32[frame],
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
