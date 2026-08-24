"""Shared camera kernel ABI for standalone GPU BVH owners."""

from std.gpu import global_idx

from bajo.bvh.gpu.camera_launch import (
    _camera_ray,
    _camera_ray_single_view,
    _store_camera_hit,
)
from bajo.bvh.types import Hit
from bajo.core import Rayf32


comptime GpuCameraTraceFn = def(
    Pointer[Float32, ImmutAnyOrigin],
    Pointer[Float32, ImmutAnyOrigin],
    UInt32,
    Rayf32[.WORLD],
) thin -> Hit[.WORLD]


def trace_bvh_camera_kernel[
    trace_ray: GpuCameraTraceFn,
    single_view: Bool = False,
](
    nodes: Pointer[Float32, ImmutAnyOrigin],
    leaves: Pointer[Float32, ImmutAnyOrigin],
    root_idx: UInt32,
    camera_params: Pointer[Float32, ImmutAnyOrigin],
    hits: Pointer[Float32, MutAnyOrigin],
    ray_count: Int32,
    width_px: Int32,
    height_px: Int32,
    inv_height: Float32,
):
    var ray_count_int = Int(ray_count)
    var ray_idx = global_idx.x
    if ray_idx >= ray_count_int:
        return

    var ray: Rayf32[.WORLD]
    comptime if single_view:
        ray = _camera_ray_single_view(
            camera_params,
            Int32(ray_idx),
            width_px,
            inv_height,
        )
    else:
        ray = _camera_ray(
            camera_params,
            ray_count_int,
            ray_idx,
            Int(width_px),
            Int(height_px),
            inv_height,
        )
    var hit = trace_ray(nodes, leaves, root_idx, ray)
    _store_camera_hit(hit, hits, ray_count_int, ray_idx)
