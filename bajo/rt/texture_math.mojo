"""Texture addressing and bump-normal math shared by host and device accessors."""

from std.math import abs, floor
from bajo.core import Vec3f32, cross, normalize


@always_inline
def image_pixel_offset(width: Int, height: Int, u: Float32, v: Float32) -> Int:
    """Repeat UVs and map their lower-left origin to top-to-bottom RGB rows."""
    var wrapped_u = u - floor(u)
    var wrapped_v = v - floor(v)
    var x = min(Int(wrapped_u * Float32(width)), width - 1)
    var y = min(Int((Float32(1.0) - wrapped_v) * Float32(height)), height - 1)
    return 3 * (y * width + x)


@always_inline
def texture_luminance(color: Vec3f32[.WORLD]) -> Float32:
    return (
        Float32(0.2126) * color.x
        + Float32(0.7152) * color.y
        + Float32(0.0722) * color.z
    )


@always_inline
def bump_normal(
    normal: Vec3f32[.WORLD], slope_u: Float32, slope_v: Float32
) -> Vec3f32[.WORLD]:
    var helper = Vec3f32[.WORLD](0.0, 1.0, 0.0)
    if abs(normal.y) > 0.99:
        helper = Vec3f32[.WORLD](1.0, 0.0, 0.0)
    var tangent = normalize(cross(helper, normal))
    var bitangent = cross(normal, tangent)
    return normalize(normal - tangent * slope_u - bitangent * slope_v)
