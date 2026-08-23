"""Shared launch sequence for one GPU RT wavefront bounce."""

from std.math import ceildiv
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.rt.types import Integrator
from bajo.rt.gpu.config import GpuRtBvhFormat, GpuRtSceneKind
from bajo.rt.gpu.common_kernels import GPU_RT_BLOCK_SIZE, GPU_RT_MAX_BLOCKS
from bajo.rt.gpu.path_shading import GpuRtMaterials, _enqueue_material_shading
from bajo.rt.gpu.scene_trace import (
    enqueue_gpu_shadows,
    gpu_rt_scene_trace_kernel,
)
from bajo.rt.gpu.views import GpuRtSceneView, gpu_rt_trace_queue_view
from bajo.rt.gpu.wavefront_contract import GpuWavefrontArena


@always_inline
def enqueue_gpu_rt_bounce[
    integrator: Integrator,
    scene_kind: GpuRtSceneKind,
    sphere_format: GpuRtBvhFormat,
    triangle_format: GpuRtBvhFormat,
    tlas_format: GpuRtBvhFormat,
    blas_format: GpuRtBvhFormat,
    MAX_BLOCKS: Int = GPU_RT_MAX_BLOCKS,
    SHADOW_MAX_BLOCKS: Int = MAX_BLOCKS,
](
    ctx: DeviceContext,
    arena: GpuWavefrontArena,
    scene: GpuRtSceneView,
    materials: GpuRtMaterials,
    src_path_ids: DeviceBuffer[.uint32],
    src_path_fields: DeviceBuffer[.float32],
    dst_path_ids: DeviceBuffer[.uint32],
    dst_path_fields: DeviceBuffer[.float32],
    rng_seed: UInt64,
    bounce: UInt32,
) raises:
    """Enqueue trace, shadow, and material stages with one launch contract."""
    var blocks = min(ceildiv(arena.capacity, GPU_RT_BLOCK_SIZE), MAX_BLOCKS)
    var queues = gpu_rt_trace_queue_view(
        arena,
        src_path_ids,
        src_path_fields,
        dst_path_ids,
        dst_path_fields,
    )
    ctx.enqueue_function[
        gpu_rt_scene_trace_kernel[
            integrator,
            scene_kind,
            sphere_format,
            triangle_format,
            tlas_format,
            blas_format,
        ]
    ](
        scene,
        queues,
        rng_seed,
        bounce,
        grid_dim=blocks,
        block_dim=GPU_RT_BLOCK_SIZE,
    )
    enqueue_gpu_shadows[
        integrator,
        scene_kind,
        sphere_format,
        triangle_format,
        tlas_format,
        blas_format,
        SHADOW_MAX_BLOCKS,
    ](ctx, scene, queues, arena.capacity)
    comptime if integrator in (Integrator.PATH, Integrator.NEE, Integrator.MIS):
        _enqueue_material_shading[integrator, MAX_BLOCKS](
            ctx,
            arena,
            materials,
            src_path_ids,
            src_path_fields,
            dst_path_ids,
            dst_path_fields,
            rng_seed,
            bounce,
        )
