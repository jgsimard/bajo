"""Texture addressing and bump-normal math shared by host and device accessors."""

from bajo.core.color import luminance

from std.math import floor, sqrt
from bajo.core import Vec3, cross, dot, length2, normalize, orthonormal_basis
from bajo.core.tensor_layout import (
    RecordTile,
    gather_record_vec3,
    gather_record_field,
)


@always_inline
def image_pixel_index[
    length: SIMDLength = 1
](
    width: Int,
    height: Int,
    u: SIMD[.float32, length],
    v: SIMD[.float32, length],
) -> SIMD[.int64, length]:
    """Repeat UVs and map their lower-left origin to top-to-bottom RGB rows."""
    var wrapped_u = u - floor(u)
    var wrapped_v = v - floor(v)
    var x = min(
        (wrapped_u * Float32(width)).cast[.int64](),
        SIMD[.int64, length](width - 1),
    )
    var y = min(
        ((Float32(1.0) - wrapped_v) * Float32(height)).cast[.int64](),
        SIMD[.int64, length](height - 1),
    )
    return y * SIMD[.int64, length](width) + x


@always_inline
def sample_image_texture[
    length: SIMDLength,
](
    pixels: RecordTile[.float32, 3, _],
    width: Int,
    height: Int,
    u: SIMD[.float32, length],
    v: SIMD[.float32, length],
) -> Vec3[.float32, .WORLD, length]:
    return gather_record_vec3(pixels, image_pixel_index(width, height, u, v))


@always_inline
def sample_image_texture_bilinear[
    length: SIMDLength,
](
    pixels: RecordTile[.float32, 3, _],
    width: Int,
    height: Int,
    u: SIMD[.float32, length],
    v: SIMD[.float32, length],
) -> Vec3[.float32, .WORLD, length]:
    var x = (u - floor(u)) * Float32(width) - 0.5
    var y = (1.0 - (v - floor(v))) * Float32(height) - 0.5
    var fx = floor(x)
    var fy = floor(y)
    var dx = x - fx
    var dy = y - fy
    var x0 = fx.cast[.int64]()
    var y0 = fy.cast[.int64]()
    var w = SIMD[.int64, length](width)
    var h = SIMD[.int64, length](height)
    var x1 = ((x0 + 1) % w + w) % w
    var y1 = ((y0 + 1) % h + h) % h
    x0 = (x0 % w + w) % w
    y0 = (y0 % h + h) % h
    var top = (
        gather_record_vec3(pixels, y0 * w + x0) * (1.0 - dx)
        + gather_record_vec3(pixels, y0 * w + x1) * dx
    )
    var bottom = (
        gather_record_vec3(pixels, y1 * w + x0) * (1.0 - dx)
        + gather_record_vec3(pixels, y1 * w + x1) * dx
    )
    return top * (1.0 - dy) + bottom * dy


@always_inline
def sample_scalar_texture_bilinear[
    length: SIMDLength,
](
    pixels: RecordTile[.float32, 1, _],
    width: Int,
    height: Int,
    u: SIMD[.float32, length],
    v: SIMD[.float32, length],
) -> SIMD[.float32, length]:
    var x = (u - floor(u)) * Float32(width) - 0.5
    var y = (1.0 - (v - floor(v))) * Float32(height) - 0.5
    var fx = floor(x)
    var fy = floor(y)
    var dx = x - fx
    var dy = y - fy
    var x0 = fx.cast[.int64]()
    var y0 = fy.cast[.int64]()
    var w = SIMD[.int64, length](width)
    var h = SIMD[.int64, length](height)
    var x1 = ((x0 + 1) % w + w) % w
    var y1 = ((y0 + 1) % h + h) % h
    x0 = (x0 % w + w) % w
    y0 = (y0 % h + h) % h
    var top = (
        gather_record_field[0](pixels, y0 * w + x0) * (1.0 - dx)
        + gather_record_field[0](pixels, y0 * w + x1) * dx
    )
    var bottom = (
        gather_record_field[0](pixels, y1 * w + x0) * (1.0 - dx)
        + gather_record_field[0](pixels, y1 * w + x1) * dx
    )
    return top * (1.0 - dy) + bottom * dy


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
def load_environment_texel[
    length: SIMDLength,
](
    pixels: RecordTile[.float32, 3, _],
    index: SIMD[.int64, length],
) -> Vec3[
    .float32, .WORLD, length
]:
    var value = gather_record_vec3(pixels, index)
    return Vec3[.float32, .WORLD, length](
        max(value.x, 0.0), max(value.y, 0.0), max(value.z, 0.0)
    )


@always_inline
def sample_environment_texture[
    length: SIMDLength,
](
    pixels: RecordTile[.float32, 3, _],
    width: Int,
    height: Int,
    u: SIMD[.float32, length],
    v: SIMD[.float32, length],
) -> Vec3[.float32, .WORLD, length]:
    """Clamp equal-area UVs to the environment's top-to-bottom RGB rows."""
    var x = (
        (u * Float32(width))
        .cast[.int64]()
        .clamp(0, SIMD[.int64, length](width - 1))
    )
    var y = (
        (v * Float32(height))
        .cast[.int64]()
        .clamp(0, SIMD[.int64, length](height - 1))
    )
    return load_environment_texel(pixels, y * SIMD[.int64, length](width) + x)


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


@always_inline
def bump_normal[
    length: SIMDLength = 1
](
    normal: Vec3[.float32, .WORLD, length],
    slope_u: SIMD[.float32, length],
    slope_v: SIMD[.float32, length],
) -> Vec3[.float32, .WORLD, length]:
    var tangent, bitangent = orthonormal_basis(normal)
    return normalize(normal - tangent * slope_u - bitangent * slope_v)
