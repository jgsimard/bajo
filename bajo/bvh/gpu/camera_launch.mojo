"""Shared camera-ray launch ABI for GPU BVH front ends."""

from std.math import ceildiv
from max.gpu.host import DeviceBuffer

from bajo.bvh.camera import Camera
from bajo.bvh.types import Hit
from bajo.core import Frame, Rayf32


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
def _store_camera_hit(
    hit: Hit[.WORLD],
    hits: MutPointer[Float32, _],
    ray_count: Int,
    ray_idx: Int,
):
    hit._store_unchecked(
        Span(unsafe_ptr=hits, length=ray_count * Hit.STRIDE), ray_idx
    )
