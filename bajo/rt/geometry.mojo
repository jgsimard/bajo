"""Host/device-safe geometry policies shared by CPU and GPU RT."""

from std.math import sqrt
from std.utils.numerics import isfinite

from bajo.core import Frame, Point3f32, Vec3, cross, dot, length2


@fieldwise_init
struct _OrientedSurfaceNormal[frame: Frame, length: SIMDLength = 1](
    TrivialRegisterPassable
):
    """An outward normal oriented against the incident ray direction."""

    var normal: Vec3[.float32, Self.frame, Self.length]
    var front_face: SIMD[.bool, Self.length]


@always_inline
def orient_surface_normal[
    frame: Frame, length: SIMDLength
](
    ray_direction: Vec3[.float32, frame, length],
    outward_normal: Vec3[.float32, frame, length],
) -> _OrientedSurfaceNormal[frame, length]:
    """Return the consistently oriented normal and front-face classification."""
    var front_face = dot(ray_direction, outward_normal).lt(0.0)
    return _OrientedSurfaceNormal[frame, length](
        Vec3.select(front_face, outward_normal, -outward_normal), front_face
    )


@always_inline
def triangle_is_valid[
    frame: Frame
](v0: Point3f32[frame], v1: Point3f32[frame], v2: Point3f32[frame]) -> Bool:
    if not (v0.is_finite()[0] and v1.is_finite()[0] and v2.is_finite()[0]):
        return False
    var twice_area_squared = length2(cross(v1 - v0, v2 - v0))[0]
    return isfinite(twice_area_squared) and twice_area_squared > 0.0


@always_inline
def triangle_area[
    frame: Frame
](v0: Point3f32[frame], v1: Point3f32[frame], v2: Point3f32[frame]) -> Float32:
    return 0.5 * sqrt(length2(cross(v1 - v0, v2 - v0)))
