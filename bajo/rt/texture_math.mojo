"""Material interpretation of sampled colors and displacement textures."""

from bajo.core.color import luminance

from std.math import sqrt
from bajo.core import Vec3, cross, dot, length2, normalize, orthonormal_basis


@always_inline
def mapped_normal[
    length: SIMDLength = 1
](
    normal: Vec3[.float32, .WORLD, length],
    dpdu: Vec3[.float32, .WORLD, length],
    color: Vec3[.float32, .WORLD, length],
) -> Vec3[.float32, .WORLD, length]:
    """Transform a linear tangent-space normal using the UV tangent frame."""
    var projected = dpdu - normal * dot(normal, dpdu)
    var fallback, _ = orthonormal_basis(normal)
    var tangent = Vec3.select(
        length2(projected).gt(1.0e-20), projected, fallback
    )
    tangent = normalize(tangent)
    var bitangent = cross(normal, tangent)
    var local = color * 2.0 - Vec3[.float32, .WORLD, length](1.0)
    var result = tangent * local.x + bitangent * local.y + normal * local.z
    result = Vec3.select(length2(result).gt(1.0e-20), result, normal)
    result = normalize(result)
    return Vec3.select(dot(result, normal).lt(0.0), -result, result)


@always_inline
def albedo_from_texture[
    length: SIMDLength = 1
](
    albedo: Vec3[.float32, .WORLD, length],
    color: Vec3[.float32, .WORLD, length],
    textured: SIMD[.bool, length] = SIMD[.bool, length](fill=True),
) -> Vec3[.float32, .WORLD, length]:
    var clamped = (albedo * color).clamp(0.0, 1.0)
    return Vec3.select(textured, clamped, albedo)


@always_inline
def roughness_from_texture[
    length: SIMDLength = 1
](
    color: Vec3[.float32, .WORLD, length],
    scale: SIMD[.float32, length],
    remap: SIMD[.bool, length],
) -> SIMD[.float32, length]:
    var roughness = (scale * luminance(color)).clamp(0.0, 1.0)
    if remap.reduce_or():
        roughness = remap.select(sqrt(roughness), roughness)
    return roughness


@always_inline
def bump_slope[
    length: SIMDLength = 1
](
    before: Vec3[.float32, .WORLD, length],
    after: Vec3[.float32, .WORLD, length],
    texels: SIMD[.float32, length],
    uv_scale: SIMD[.float32, length],
    scale: SIMD[.float32, length],
) -> SIMD[.float32, length]:
    return (luminance(after) - luminance(before)) * texels * uv_scale * scale
