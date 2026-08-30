"""Host/device-safe camera and secondary-ray construction policies."""

from bajo.bvh import Camera
from bajo.bvh.constants import f32_max
from bajo.core import Point3, Ray, Vec3, normalize
from bajo.core.random import Rng, random_on_hemisphere


comptime RT_RAY_T_MIN = Float32(0.001)
comptime RT_SHADOW_END_OFFSET = Float32(0.002)
comptime RT_AO_DISTANCE = Float32(4.0)


@always_inline
def make_camera_ray_from_samples[
    length: SIMDLength
](
    camera: Camera,
    px: SIMD[.float32, length],
    py: SIMD[.float32, length],
    width: Float32,
    height: Float32,
    pixel_u: SIMD[.float32, length],
    pixel_v: SIMD[.float32, length],
    lens_u: SIMD[.float32, length],
    lens_v: SIMD[.float32, length],
) -> Ray[.float32, .WORLD, length]:
    """Construct a primary ray from backend-provided camera samples."""
    return camera.make_ray_sampled[length](
        px,
        py,
        width,
        height,
        pixel_u,
        pixel_v,
        lens_u,
        lens_v,
        RT_RAY_T_MIN,
    )


@always_inline
def spawn_surface_ray[
    length: SIMDLength
](
    point: Point3[.float32, .WORLD, length],
    direction: Vec3[.float32, .WORLD, length],
    t_max: SIMD[.float32, length] = f32_max,
) -> Ray[.float32, .WORLD, length]:
    """Spawn a secondary ray using the renderer's shared lower bound."""
    return Ray[.float32, .WORLD, length](point, direction, RT_RAY_T_MIN, t_max)


@always_inline
def make_ao_ray(
    point: Point3[.float32, .WORLD, 1],
    normal: Vec3[.float32, .WORLD, 1],
    mut rng: Rng,
) -> Ray[.float32, .WORLD, 1]:
    """Sample the shared finite-distance ambient-occlusion query."""
    var direction = normalize(random_on_hemisphere[.WORLD](rng, normal))
    return spawn_surface_ray(point, direction, RT_AO_DISTANCE)
