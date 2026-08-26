"""Shared launch sequence for one GPU RT wavefront bounce."""

from std.math import ceildiv
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh.constants import PrimitiveKind
from bajo.rt.types import Integrator, SamplingConfig
from bajo.rt.gpu.config import GpuRtBvhFormat, GpuRtSceneKind
from bajo.rt.gpu.common_kernels import GPU_RT_BLOCK_SIZE, GPU_RT_MAX_BLOCKS
from bajo.rt.gpu.path_shading import GpuRtMaterials, _enqueue_material_shading
from bajo.rt.gpu.scene_trace import (
    enqueue_gpu_shadows,
    gpu_rt_primary_scene_trace_kernel,
    gpu_rt_scene_trace_kernel,
)
from bajo.rt.gpu.views import (
    GpuRtSceneView,
    GpuRtTraceQueueView,
    gpu_rt_trace_queue_view,
)
from bajo.rt.gpu.wavefront_contract import GpuWavefrontArena


@always_inline
def _enqueue_primary_trace[
    integrator: Integrator,
    scene_kind: GpuRtSceneKind,
    sphere_format: GpuRtBvhFormat,
    triangle_format: GpuRtBvhFormat,
    tlas_format: GpuRtBvhFormat,
    blas_format: GpuRtBvhFormat,
    store_source_path: Bool,
    light_kind: PrimitiveKind,
](
    ctx: DeviceContext,
    camera: DeviceBuffer[.float32],
    src_path_ids: DeviceBuffer[.uint32],
    src_path_fields: DeviceBuffer[.float32],
    scene: GpuRtSceneView,
    queues: GpuRtTraceQueueView,
    image_width: Int,
    image_height: Int,
    samples_per_pixel: Int,
    sampling: SamplingConfig,
    blocks: Int,
) raises:
    ctx.enqueue_function[
        gpu_rt_primary_scene_trace_kernel[
            integrator,
            scene_kind,
            sphere_format,
            triangle_format,
            tlas_format,
            blas_format,
            store_source_path,
            light_kind,
        ]
    ](
        camera,
        src_path_ids,
        src_path_fields,
        scene,
        queues,
        Int32(image_width),
        Int32(image_height),
        Int32(samples_per_pixel),
        sampling,
        grid_dim=blocks,
        block_dim=GPU_RT_BLOCK_SIZE,
    )


@always_inline
def _enqueue_primary_trace_by_light_kind[
    integrator: Integrator,
    scene_kind: GpuRtSceneKind,
    sphere_format: GpuRtBvhFormat,
    triangle_format: GpuRtBvhFormat,
    tlas_format: GpuRtBvhFormat,
    blas_format: GpuRtBvhFormat,
    store_source_path: Bool,
](
    ctx: DeviceContext,
    camera: DeviceBuffer[.float32],
    src_path_ids: DeviceBuffer[.uint32],
    src_path_fields: DeviceBuffer[.float32],
    scene: GpuRtSceneView,
    queues: GpuRtTraceQueueView,
    uniform_sampling_kind: PrimitiveKind,
    image_width: Int,
    image_height: Int,
    samples_per_pixel: Int,
    sampling: SamplingConfig,
    blocks: Int,
) raises:
    comptime if integrator in (Integrator.NEE, Integrator.MIS):
        if uniform_sampling_kind == .SPHERE:
            _enqueue_primary_trace[
                integrator,
                scene_kind,
                sphere_format,
                triangle_format,
                tlas_format,
                blas_format,
                store_source_path,
                PrimitiveKind.SPHERE,
            ](
                ctx,
                camera,
                src_path_ids,
                src_path_fields,
                scene,
                queues,
                image_width,
                image_height,
                samples_per_pixel,
                sampling,
                blocks,
            )
        elif uniform_sampling_kind == .TRIANGLE:
            _enqueue_primary_trace[
                integrator,
                scene_kind,
                sphere_format,
                triangle_format,
                tlas_format,
                blas_format,
                store_source_path,
                PrimitiveKind.TRIANGLE,
            ](
                ctx,
                camera,
                src_path_ids,
                src_path_fields,
                scene,
                queues,
                image_width,
                image_height,
                samples_per_pixel,
                sampling,
                blocks,
            )
        else:
            _enqueue_primary_trace[
                integrator,
                scene_kind,
                sphere_format,
                triangle_format,
                tlas_format,
                blas_format,
                store_source_path,
                PrimitiveKind.UNKNOWN,
            ](
                ctx,
                camera,
                src_path_ids,
                src_path_fields,
                scene,
                queues,
                image_width,
                image_height,
                samples_per_pixel,
                sampling,
                blocks,
            )
    else:
        _enqueue_primary_trace[
            integrator,
            scene_kind,
            sphere_format,
            triangle_format,
            tlas_format,
            blas_format,
            store_source_path,
            PrimitiveKind.UNKNOWN,
        ](
            ctx,
            camera,
            src_path_ids,
            src_path_fields,
            scene,
            queues,
            image_width,
            image_height,
            samples_per_pixel,
            sampling,
            blocks,
        )


@always_inline
def enqueue_gpu_rt_primary_bounce[
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
    uniform_sampling_kind: PrimitiveKind,
    camera: DeviceBuffer[.float32],
    src_path_ids: DeviceBuffer[.uint32],
    src_path_fields: DeviceBuffer[.float32],
    dst_path_ids: DeviceBuffer[.uint32],
    dst_path_fields: DeviceBuffer[.float32],
    active_count: Int,
    image_width: Int,
    image_height: Int,
    samples_per_pixel: Int,
    sampling: SamplingConfig,
) raises:
    """Generate and trace primary paths without a queue round trip."""
    var queues = gpu_rt_trace_queue_view(
        arena,
        src_path_ids,
        src_path_fields,
        dst_path_ids,
        dst_path_fields,
    )
    var blocks = min(ceildiv(active_count, GPU_RT_BLOCK_SIZE), MAX_BLOCKS)
    if materials.has_non_lambertian:
        _enqueue_primary_trace_by_light_kind[
            integrator,
            scene_kind,
            sphere_format,
            triangle_format,
            tlas_format,
            blas_format,
            True,
        ](
            ctx,
            camera,
            src_path_ids,
            src_path_fields,
            scene,
            queues,
            uniform_sampling_kind,
            image_width,
            image_height,
            samples_per_pixel,
            sampling,
            blocks,
        )
    else:
        _enqueue_primary_trace_by_light_kind[
            integrator,
            scene_kind,
            sphere_format,
            triangle_format,
            tlas_format,
            blas_format,
            False,
        ](
            ctx,
            camera,
            src_path_ids,
            src_path_fields,
            scene,
            queues,
            uniform_sampling_kind,
            image_width,
            image_height,
            samples_per_pixel,
            sampling,
            blocks,
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
            sampling,
            UInt32(0),
        )


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
    uniform_sampling_kind: PrimitiveKind,
    src_path_ids: DeviceBuffer[.uint32],
    src_path_fields: DeviceBuffer[.float32],
    dst_path_ids: DeviceBuffer[.uint32],
    dst_path_fields: DeviceBuffer[.float32],
    sampling: SamplingConfig,
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
    comptime if integrator in (Integrator.NEE, Integrator.MIS):
        if uniform_sampling_kind == .SPHERE:
            ctx.enqueue_function[
                gpu_rt_scene_trace_kernel[
                    integrator,
                    scene_kind,
                    sphere_format,
                    triangle_format,
                    tlas_format,
                    blas_format,
                    PrimitiveKind.SPHERE,
                ]
            ](
                scene,
                queues,
                sampling,
                bounce,
                grid_dim=blocks,
                block_dim=GPU_RT_BLOCK_SIZE,
            )
        elif uniform_sampling_kind == .TRIANGLE:
            ctx.enqueue_function[
                gpu_rt_scene_trace_kernel[
                    integrator,
                    scene_kind,
                    sphere_format,
                    triangle_format,
                    tlas_format,
                    blas_format,
                    PrimitiveKind.TRIANGLE,
                ]
            ](
                scene,
                queues,
                sampling,
                bounce,
                grid_dim=blocks,
                block_dim=GPU_RT_BLOCK_SIZE,
            )
        else:
            ctx.enqueue_function[
                gpu_rt_scene_trace_kernel[
                    integrator,
                    scene_kind,
                    sphere_format,
                    triangle_format,
                    tlas_format,
                    blas_format,
                    PrimitiveKind.UNKNOWN,
                ]
            ](
                scene,
                queues,
                sampling,
                bounce,
                grid_dim=blocks,
                block_dim=GPU_RT_BLOCK_SIZE,
            )
    else:
        ctx.enqueue_function[
            gpu_rt_scene_trace_kernel[
                integrator,
                scene_kind,
                sphere_format,
                triangle_format,
                tlas_format,
                blas_format,
                PrimitiveKind.UNKNOWN,
            ]
        ](
            scene,
            queues,
            sampling,
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
            sampling,
            bounce,
        )
