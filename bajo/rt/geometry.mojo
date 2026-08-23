"""Host/device-safe geometry policies shared by CPU and GPU RT."""

from std.math import abs

from bajo.bvh import Sphere
from bajo.core import Frame, Vec3, dot


@fieldwise_init
struct _OrientedSurfaceNormal[frame: Frame, length: SIMDLength = 1](
    TrivialRegisterPassable
):
    """An outward normal oriented against the incident ray direction."""

    var normal: Vec3[.float32, Self.frame, Self.length]
    var front_face: SIMD[DType.bool, Self.length]


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
def sphere_unsigned_radius[frame: Frame](sphere: Sphere[frame]) -> Float32:
    """Return the physical radius while preserving the stored orientation sign.
    """
    return abs(sphere.radius)


@always_inline
def sphere_for_acceleration[
    frame: Frame
](sphere: Sphere[frame]) -> Sphere[frame]:
    """Preserve signed-radius shading semantics outside the acceleration data.
    """
    return Sphere[frame](sphere.center, sphere_unsigned_radius(sphere))
