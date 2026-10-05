"""Scalar geometry predicates shared by spatial CPU and GPU mesh queries.

Double intermediates avoid Float32 overflow and handle small triangles without
an absolute determinant epsilon. Degenerate faces remain valid point/segment
geometry for distance queries, and never contribute ray hits or solid angle.
"""

from std.math import abs, max, min, sqrt
from bajo.core import Frame, Point3f32, Rayf32, Vec3, cross, dot


@no_inline
def angle_atan2(y: Float64, x: Float64) -> Float64:
    """Portable double atan2 for finite geometry intermediates.

    Mojo's double std.math.atan2 emits a host libm call in GPU kernels. Reduce
    the ratio to |r| <= tan(pi/8), then use 20 terms of the alternating series
    (truncation below 2e-17). This shared implementation needs no device libm.
    """
    var ax = abs(x)
    var ay = abs(y)
    if ax == 0.0 and ay == 0.0:
        return 0.0
    var swapped = ay > ax
    var ratio = ax / ay if swapped else ay / ax
    var offset = Float64(0.0)
    if ratio > 0.4142135623730950488:
        ratio = (ratio - 1.0) / (ratio + 1.0)
        offset = 0.7853981633974483096
    var ratio2 = ratio * ratio
    var power = ratio
    var angle = Float64(0.0)
    for term in range(20):
        angle += power / Float64(2 * term + 1)
        power *= -ratio2
    angle += offset
    if swapped:
        angle = 1.5707963267948966192 - angle
    if x < 0.0:
        angle = 3.1415926535897932385 - angle
    return -angle if y < 0.0 else angle


def vector64[frame: Frame](p: Point3f32[frame]) -> Vec3[.float64, frame]:
    return Vec3[.float64, frame](Float64(p.x), Float64(p.y), Float64(p.z))


@fieldwise_init
struct TriangleProjection[frame: Frame](TrivialRegisterPassable):
    var point: Vec3[.float64, Self.frame]
    var u: Float64
    var v: Float64
    var distance2: Float64


def project_triangle[
    frame: Frame
](
    point: Point3f32[frame],
    a: Point3f32[frame],
    b: Point3f32[frame],
    c: Point3f32[frame],
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
    for edge in range(3):
        var start = av
        var end = bv
        if edge == 1:
            start = bv
            end = cv
        elif edge == 2:
            start = cv
            end = av
        var direction = end - start
        var length2 = dot(direction, direction)
        var t = Float64(0.0)
        if length2 > 0.0:
            t = max(0.0, min(1.0, dot(p - start, direction) / length2))
        var q = start + direction * t
        var distance2 = dot(q - p, q - p)
        if distance2 < best.distance2:
            var u = Float64(1.0) - t
            var v = t
            if edge == 1:
                u = 0.0
                v = 1.0 - t
            elif edge == 2:
                u = t
                v = 0.0
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
    ray: Rayf32[frame],
    a: Point3f32[frame],
    b: Point3f32[frame],
    c: Point3f32[frame],
) -> TriangleRayHit:
    var av = vector64(a)
    var e1 = vector64(b) - av
    var e2 = vector64(c) - av
    var direction = Vec3[.float64, frame](
        Float64(ray.d.x), Float64(ray.d.y), Float64(ray.d.z)
    )
    var p = cross(direction, e2)
    var determinant = dot(e1, p)
    if determinant == 0.0:
        return TriangleRayHit(False, 0.0, 0.0, 0.0)
    var offset = vector64(ray.o) - av
    var weight_b = dot(offset, p) / determinant
    var q = cross(offset, e1)
    var weight_c = dot(direction, q) / determinant
    var t = dot(e2, q) / determinant
    return TriangleRayHit(
        weight_b >= 0.0
        and weight_c >= 0.0
        and weight_b + weight_c <= 1.0
        and t >= Float64(ray.t_min)
        and t <= Float64(ray.t_max),
        t,
        1.0 - weight_b - weight_c,
        weight_b,
    )


def triangle_normal[
    frame: Frame
](
    a: Point3f32[frame],
    b: Point3f32[frame],
    c: Point3f32[frame],
) -> Vec3[
    .float64, frame
]:
    var normal = cross(vector64(b) - vector64(a), vector64(c) - vector64(a))
    var length2 = dot(normal, normal)
    if length2 > 0.0:
        return normal / sqrt(length2)
    return Vec3[.float64, frame](0.0)


def solid_angle[
    frame: Frame
](
    point: Point3f32[frame],
    a: Point3f32[frame],
    b: Point3f32[frame],
    c: Point3f32[frame],
) -> Float64:
    var av = vector64(a) - vector64(point)
    var bv = vector64(b) - vector64(point)
    var cv = vector64(c) - vector64(point)
    var numerator = dot(av, cross(bv, cv))
    if numerator == 0.0:
        return 0.0
    var la = sqrt(dot(av, av))
    var lb = sqrt(dot(bv, bv))
    var lc = sqrt(dot(cv, cv))
    var denominator = (
        la * lb * lc + dot(av, bv) * lc + dot(bv, cv) * la + dot(cv, av) * lb
    )
    return 2.0 * angle_atan2(numerator, denominator)
