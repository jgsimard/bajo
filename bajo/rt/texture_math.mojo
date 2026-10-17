"""Texture addressing and bump-normal math shared by host and device accessors."""

from std.math import abs, floor
from bajo.core import Vec3, cross, normalize


@always_inline
def image_pixel_offset[
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
    return 3 * (y * SIMD[.int64, length](width) + x)


@always_inline
def texture_luminance[
    length: SIMDLength = 1
](color: Vec3[.float32, .WORLD, length]) -> SIMD[.float32, length]:
    return (
        Float32(0.2126) * color.x
        + Float32(0.7152) * color.y
        + Float32(0.0722) * color.z
    )


@always_inline
def bump_normal[
    length: SIMDLength = 1
](
    normal: Vec3[.float32, .WORLD, length],
    slope_u: SIMD[.float32, length],
    slope_v: SIMD[.float32, length],
) -> Vec3[.float32, .WORLD, length]:
    var near_y = abs(normal.y).gt(0.99)
    var helper = Vec3[.float32, .WORLD, length](
        near_y.select(Float32(1.0), 0.0), near_y.select(Float32(0.0), 1.0), 0.0
    )
    var tangent = normalize(cross(helper, normal))
    var bitangent = cross(normal, tangent)
    return normalize(normal - tangent * slope_u - bitangent * slope_v)
