"""Material interpretation of sampled colors and displacement textures."""

from bajo.core.color import luminance

from std.math import sqrt
from bajo.core import Vec3
from bajo.core.image import (
    image_pixel_index,
    sample_image_texture,
    load_environment_texel,
    sample_environment_texture,
)
from bajo.core.geometry import normal_from_slopes as bump_normal
from bajo.core.tensor_layout import RecordTile, gather_record_vec3


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
