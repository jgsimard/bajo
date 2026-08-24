"""Shared camera-ray launch ABI for GPU BVH front ends."""

from std.math import ceildiv
from max.gpu.host import DeviceBuffer

from bajo.bvh.camera import Camera
from bajo.bvh.constants import f32_max
from bajo.bvh.types import Hit
from bajo.core import Point3f32, Rayf32, Vec3f32, normalize


def validate_camera_launch(
    d_camera_params: DeviceBuffer[.float32],
    d_hits: DeviceBuffer[.float32],
    ray_count: Int,
    width: Int,
    height: Int,
) raises:
    debug_assert["safe", _use_compiler_assume=True](
        ray_count > 0 and width > 0 and height > 0,
        "camera launch dimensions must be positive",
    )
    debug_assert["safe", _use_compiler_assume=True](
        len(d_camera_params)
        >= ceildiv(ray_count, width * height) * Camera.STRIDE,
        "camera parameter buffer is too short",
    )
    debug_assert["safe", _use_compiler_assume=True](
        len(d_hits) >= ray_count * Hit.STRIDE,
        "hit output buffer is too short",
    )


@always_inline
def _camera_ray(
    camera_params: ImmPointer[Float32, _],
    ray_count: Int,
    ray_idx: Int,
    width: Int,
    height: Int,
    inv_height: Float32,
) -> Rayf32[.WORLD]:
    var pixels_per_view = width * height
    var view_idx = ray_idx / pixels_per_view
    var local_idx = ray_idx - view_idx * pixels_per_view
    var camera_span = Span(
        unsafe_ptr=camera_params,
        length=ceildiv(ray_count, pixels_per_view) * Camera.STRIDE,
    )
    var camera = Camera(camera_span, view_idx * Camera.STRIDE)
    return camera.make_ray_raster(
        local_idx % width,
        local_idx / width,
        width,
        inv_height,
    )


@always_inline
def _camera_ray_single_view(
    camera_params: ImmPointer[Float32, _],
    ray_idx: Int32,
    width: Int32,
    inv_height: Float32,
) -> Rayf32[.WORLD]:
    """Generate one raster ray without the generic multi-view index math."""
    var px = ray_idx % width
    var py = ray_idx // width
    var screen_x = (2.0 * (Float32(px) + 0.5) - Float32(width)) * inv_height
    var screen_y = 1.0 - 2.0 * (Float32(py) + 0.5) * inv_height
    var fov = camera_params[unsafe_offset=Camera.FOV]
    var sx = screen_x * fov
    var sy = screen_y * fov
    var direction = normalize(
        Vec3f32[.WORLD](
            camera_params[unsafe_offset=Camera.FORWARD + 0]
            + camera_params[unsafe_offset=Camera.RIGHT + 0] * sx
            + camera_params[unsafe_offset=Camera.UP + 0] * sy,
            camera_params[unsafe_offset=Camera.FORWARD + 1]
            + camera_params[unsafe_offset=Camera.RIGHT + 1] * sx
            + camera_params[unsafe_offset=Camera.UP + 1] * sy,
            camera_params[unsafe_offset=Camera.FORWARD + 2]
            + camera_params[unsafe_offset=Camera.RIGHT + 2] * sx
            + camera_params[unsafe_offset=Camera.UP + 2] * sy,
        )
    )
    var origin = Point3f32[.WORLD](
        camera_params[unsafe_offset=Camera.ORIGIN + 0],
        camera_params[unsafe_offset=Camera.ORIGIN + 1],
        camera_params[unsafe_offset=Camera.ORIGIN + 2],
    )
    return Rayf32[.WORLD](origin, direction, 0.0, f32_max)


@always_inline
def _store_camera_hit(
    hit: Hit[.WORLD],
    hits: MutPointer[Float32, _],
    ray_count: Int,
    ray_idx: Int,
):
    hit._store_unchecked(
        Span(unsafe_ptr=hits, length=ray_count * Hit.STRIDE), ray_idx
    )
