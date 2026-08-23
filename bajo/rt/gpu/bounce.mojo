"""Shared launch sequence for one GPU RT wavefront bounce."""

from std.math import ceildiv
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh.gpu import GpuBvhLayout
from bajo.rt.types import Integrator
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
    ALGORITHM: Integrator,
    HAS_SPHERES: Bool,
    HAS_TRIANGLES: Bool,
    HAS_INSTANCES: Bool,
    sphere_node_width: SIMDLength,
    sphere_leaf_width: SIMDLength,
    triangle_node_width: SIMDLength,
    triangle_leaf_width: SIMDLength,
    tlas_node_width: SIMDLength,
    tlas_leaf_width: SIMDLength,
    blas_node_width: SIMDLength,
    blas_leaf_width: SIMDLength,
    MAX_BLOCKS: Int = GPU_RT_MAX_BLOCKS,
    SHADOW_MAX_BLOCKS: Int = MAX_BLOCKS,
    triangle_layout: GpuBvhLayout = .WIDE,
    blas_layout: GpuBvhLayout = .WIDE,
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
            ALGORITHM,
            HAS_SPHERES,
            HAS_TRIANGLES,
            HAS_INSTANCES,
            sphere_node_width,
            sphere_leaf_width,
            triangle_node_width,
            triangle_leaf_width,
            tlas_node_width,
            tlas_leaf_width,
            blas_node_width,
            blas_leaf_width,
            triangle_layout,
            blas_layout,
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
        ALGORITHM,
        HAS_SPHERES,
        HAS_TRIANGLES,
        HAS_INSTANCES,
        sphere_node_width,
        sphere_leaf_width,
        triangle_node_width,
        triangle_leaf_width,
        tlas_node_width,
        tlas_leaf_width,
        blas_node_width,
        blas_leaf_width,
        SHADOW_MAX_BLOCKS,
        triangle_layout,
        blas_layout,
    ](ctx, scene, queues, arena.capacity)
    comptime if ALGORITHM in (Integrator.PATH, Integrator.NEE, Integrator.MIS):
        _enqueue_material_shading[ALGORITHM, MAX_BLOCKS](
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
