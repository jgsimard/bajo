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
    if (
        ray_count <= 0
        or ray_count > Int(Int32.MAX)
        or width <= 0
        or width > Int(Int32.MAX)
        or height <= 0
        or height > Int(Int32.MAX)
    ):
        raise Error("camera launch dimensions must fit positive Int32 values")
    if (
        ceildiv(ray_count, width * height)
        > len(d_camera_params) // Camera.STRIDE
    ):
        raise Error("camera parameter buffer is too short")
    if ray_count > len(d_hits) // Hit.STRIDE:
        raise Error("hit output buffer is too short")


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
    var forward = Vec3f32[.WORLD](camera_params.unsafe_offset(Camera.FORWARD))
    var right = Vec3f32[.WORLD](camera_params.unsafe_offset(Camera.RIGHT))
    var up = Vec3f32[.WORLD](camera_params.unsafe_offset(Camera.UP))
    var direction = normalize(forward + right * sx + up * sy)
    var origin = Point3f32[.WORLD](camera_params.unsafe_offset(Camera.ORIGIN))
    return Rayf32[.WORLD](origin, direction, 0.0, f32_max)
