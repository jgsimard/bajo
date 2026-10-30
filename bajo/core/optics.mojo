from std.math import abs, exp, fma, sqrt
from std.utils.numerics import max_finite

from .frame import Frame
from .vec import Vec3, dot, length2


def reflect[
    dtype: DType, frame: Frame, length: SIMDLength
](v: Vec3[dtype, frame, length], n: Vec3[dtype, frame, length]) -> Vec3[
    dtype, frame, length
]:
    return v - 2.0 * dot(v, n) * n


def refract[
    dtype: DType, frame: Frame, length: SIMDLength
](
    uv: Vec3[dtype, frame, length],
    n: Vec3[dtype, frame, length],
    etai_over_etat: SIMD[dtype, length],
) -> Vec3[dtype, frame, length]:
    var cos_theta = min(dot(-uv, n), 1.0)
    var r_out_perp = etai_over_etat * (uv + cos_theta * n)
    var r_out_parallel = -sqrt(abs(1.0 - length2(r_out_perp))) * n
    return r_out_perp + r_out_parallel


def fresnel_schlick[
    dtype: DType, length: SIMDLength
](cosine: SIMD[dtype, length], ref_idx: SIMD[dtype, length]) -> SIMD[
    dtype, length
]:
    """Schlick approximation for dielectric Fresnel reflectance."""
    var root = (1.0 - ref_idx) / (1.0 + ref_idx)
    var r2 = root * root
    var x = 1.0 - cosine
    var x2 = x * x
    var x5 = x2 * x2 * x
    return fma(1.0 - r2, x5, r2)


@always_inline
def fresnel_dielectric[
    length: SIMDLength = 1
](cosine_in: SIMD[.float32, length], eta: SIMD[.float32, length]) -> SIMD[
    .float32, length
]:
    var cosine = cosine_in.clamp(-1.0, 1.0)
    var relative_eta = cosine.lt(0.0).select(1.0 / eta, eta)
    cosine = abs(cosine)
    var sin2_t = max(SIMD[.float32, length](0.0), 1.0 - cosine * cosine) / (
        relative_eta * relative_eta
    )
    if sin2_t.ge(1.0).reduce_and():
        return SIMD[.float32, length](1.0)
    var cos_t = sqrt(max(SIMD[.float32, length](0.0), 1.0 - sin2_t))
    var parallel = (relative_eta * cosine - cos_t) / (
        relative_eta * cosine + cos_t
    )
    var perpendicular = (cosine - relative_eta * cos_t) / (
        cosine + relative_eta * cos_t
    )
    return sin2_t.ge(1.0).select(
        SIMD[.float32, length](1.0),
        0.5 * (parallel * parallel + perpendicular * perpendicular),
    )


@always_inline
def fresnel_conductor[
    length: SIMDLength = 1
](
    cosine: SIMD[.float32, length],
    eta: Vec3[.float32, .WORLD, length],
    k: Vec3[.float32, .WORLD, length],
) -> Vec3[.float32, .WORLD, length]:
    var c = abs(cosine).clamp(0.0, 1.0)
    var c2 = c * c
    var s2 = 1.0 - c2
    var result = Vec3[.float32, .WORLD, length](0.0)
    comptime for channel in range(3):
        var e2 = eta[channel] * eta[channel]
        var k2 = k[channel] * k[channel]
        var t = e2 - k2 - s2
        var magnitude = sqrt(t * t + 4.0 * e2 * k2)
        var real = sqrt(max(SIMD[.float32, length](0.0), (magnitude + t) * 0.5))
        var rs = (magnitude + c2 - 2.0 * c * real) / max(
            SIMD[.float32, length](1.0e-20), magnitude + c2 + 2.0 * c * real
        )
        var rp = (
            rs
            * (c2 * magnitude + s2 * s2 - 2.0 * c * real * s2)
            / max(
                SIMD[.float32, length](1.0e-20),
                c2 * magnitude + s2 * s2 + 2.0 * c * real * s2,
            )
        )
        result[channel] = ((rs + rp) * 0.5).clamp(0.0, 1.0)
    return result


@always_inline
def medium_transmittance[
    length: SIMDLength = 1
](
    sigma_a: Vec3[.float32, .WORLD, length],
    distance: SIMD[.float32, length],
) -> Vec3[.float32, .WORLD, length]:
    # Mask the distance before multiplication: zero absorption is transparent
    # even for infinite segments, without forming 0 * infinity.
    var result = Vec3[.float32, .WORLD, length](0.0)
    comptime for axis in range(3):
        var depth = sigma_a[axis] * sigma_a[axis].ne(0.0).select(distance, 0.0)
        # SIMD exp may clamp its argument; preserve exact zero for an
        # infinitely absorbing segment (including overflowed optical depth).
        result[axis] = depth.gt(max_finite[.float32]()).select(
            Float32(0.0), exp(-depth)
        )
    return result
