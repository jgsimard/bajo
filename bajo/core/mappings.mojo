"""Equal-area mappings between a unit square and a unit sphere."""

from std.math import abs, cos, pi, sin, sqrt
from .frame import Frame
from .vec import Vec3, normalize


@fieldwise_init
struct SquareUV[length: SIMDLength = 1](
    Copyable, TrivialRegisterPassable, Writable
):
    var u: SIMD[.float32, Self.length]
    var v: SIMD[.float32, Self.length]


@always_inline
def equal_area_square_to_sphere[
    frame: Frame, length: SIMDLength
](u01: SIMD[.float32, length], v01: SIMD[.float32, length]) -> Vec3[
    .float32, frame, length
]:
    """PBRT v4's equal-area unit-square to unit-sphere mapping."""
    var u = 2.0 * u01 - 1.0
    var v = 2.0 * v01 - 1.0
    var up = abs(u)
    var vp = abs(v)
    var signed_distance = 1.0 - (up + vp)
    var r = 1.0 - abs(signed_distance)
    var nonzero_r = r.ne(0.0)
    var phi = (
        nonzero_r.select((vp - up) / nonzero_r.select(r, 1.0), 0.0) + 1.0
    ) * (pi / 4.0)
    var z = 1.0 - r * r
    z = signed_distance.lt(0.0).select(-z, z)
    var cos_phi = cos(phi)
    var sin_phi = sin(phi)
    cos_phi = u.lt(0.0).select(-cos_phi, cos_phi)
    sin_phi = v.lt(0.0).select(-sin_phi, sin_phi)
    var radial = r * sqrt(max(2.0 - r * r, 0.0))
    return Vec3[.float32, frame, length](cos_phi * radial, sin_phi * radial, z)


@always_inline
def equal_area_sphere_to_square[
    frame: Frame, length: SIMDLength
](direction: Vec3[.float32, frame, length]) -> SquareUV[length]:
    """PBRT v4's equal-area unit-sphere to unit-square mapping."""
    var d = normalize(direction)
    var x = abs(d.x)
    var y = abs(d.y)
    var z = abs(d.z)
    var r = sqrt(max(Float32(1.0) - z, Float32(0.0)))
    var a = max(x, y)
    var nonzero_a = a.ne(0.0)
    var b = min(x, y) / nonzero_a.select(a, Float32(1.0))

    # PBRT's 6th-degree minimax approximation of atan(b) * 2 / pi.
    var phi = SIMD[.float32, length](
        Float32(-0.251390972343483509333252996350e-1)
    )
    phi = Float32(0.419038818029165735901852432784e-1) + b * phi
    phi = Float32(0.881770664775316294736387951347e-1) + b * phi
    phi = Float32(-0.247333733281268944196501420480) + b * phi
    phi = Float32(0.61572017898280213493197203466e-2) + b * phi
    phi = Float32(0.636226545274016134946890922156) + b * phi
    phi = Float32(0.406758566246788489601959989e-5) + b * phi
    phi = x.lt(y).select(1.0 - phi, phi)

    var v = phi * r
    var u = r - v
    var original_u = u
    var south = d.z.lt(0.0)
    u = south.select(1.0 - v, u)
    v = south.select(1.0 - original_u, v)
    u = d.x.lt(0.0).select(-u, u)
    v = d.y.lt(0.0).select(-v, v)
    return SquareUV[length](0.5 * (u + 1.0), 0.5 * (v + 1.0))
