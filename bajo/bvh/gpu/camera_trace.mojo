"""Shared camera kernel ABI for standalone GPU BVH owners."""

from std.math import ceildiv
from max.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh.constants import GPU_BOUNDS_BVH_BLOCK_SIZE

from bajo.bvh.gpu.ray_launch import _store_packed_hit
from bajo.bvh.gpu.camera_launch import (
    _camera_ray,
    _camera_ray_single_view,
    validate_camera_launch,
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
    _store_packed_hit[.WORLD](hit, hits, ray_count_int, ray_idx)


@always_inline
def enqueue_bvh_camera_trace[
    trace_ray: GpuCameraTraceFn,
](
    ctx: DeviceContext,
    nodes: DeviceBuffer[.float32],
    leaves: DeviceBuffer[.float32],
    root_idx: UInt32,
    camera_params: DeviceBuffer[.float32],
    hits: DeviceBuffer[.float32],
    ray_count: Int,
    width: Int,
    height: Int,
) raises:
    """Launch the single-view or multi-view specialization with one ABI."""
    validate_camera_launch(camera_params, hits, ray_count, width, height)

    @always_inline
    def enqueue[single_view: Bool]() raises {imm}:
        ctx.enqueue_function[trace_bvh_camera_kernel[trace_ray, single_view]](
            nodes,
            leaves,
            root_idx,
            camera_params,
            hits,
            Int32(ray_count),
            Int32(width),
            Int32(height),
            1.0 / Float32(height),
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )

    if ray_count == width * height:
        enqueue[True]()
    else:
        enqueue[False]()
