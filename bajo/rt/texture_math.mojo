"""Texture addressing and bump-normal math shared by host and device accessors."""

from bajo.core.color import luminance

from std.math import floor, sqrt
from bajo.core import Vec3, normalize, orthonormal_basis
from bajo.core.tensor_layout import RecordTile, gather_record_vec3


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
    mut: Bool,
    origin: Origin[mut=mut],
    //,
    length: SIMDLength,
](
    pixels: RecordTile[.float32, 3, origin],
    width: Int,
    height: Int,
    u: SIMD[.float32, length],
    v: SIMD[.float32, length],
) -> Vec3[.float32, .WORLD, length]:
    return gather_record_vec3(pixels, image_pixel_index(width, height, u, v))


@always_inline
def load_environment_texel[
    mut: Bool,
    origin: Origin[mut=mut],
    //,
    length: SIMDLength,
](
    pixels: RecordTile[.float32, 3, origin],
    index: SIMD[.int64, length],
) -> Vec3[.float32, .WORLD, length]:
    var value = gather_record_vec3(pixels, index)
    return Vec3[.float32, .WORLD, length](
        max(value.x, 0.0), max(value.y, 0.0), max(value.z, 0.0)
    )


@always_inline
def sample_environment_texture[
    mut: Bool,
    origin: Origin[mut=mut],
    //,
    length: SIMDLength,
](
    pixels: RecordTile[.float32, 3, origin],
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
