"""Shared launch sequence for one GPU RT wavefront bounce."""

from std.math import ceildiv
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh.constants import PrimitiveKind
from bajo.rt.render_types import SamplingConfig
from bajo.rt.material_types import Integrator
from bajo.rt.gpu.config import GpuRtBvhFormat, GpuRtSceneKind
from bajo.rt.gpu.common_kernels import GPU_RT_BLOCK_SIZE, GPU_RT_MAX_BLOCKS
from bajo.rt.gpu.materials import GpuRtMaterials
from bajo.rt.gpu.path_shading import _enqueue_material_shading
from bajo.rt.gpu.coated_shading import enqueue_coated_shading
from bajo.rt.gpu.scene_trace import (
    GPU_RT_LARGE_QUEUE_MIN_CAPACITY,
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


comptime GPU_RT_DIRECT_TRACE_MAX_BLOCKS = 2048
comptime GPU_RT_DIRECT_LARGE_TRACE_MAX_BLOCKS = 8192
comptime GPU_RT_DIRECT_LARGE_TRACE_BLOCK_SIZE = 32


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
    comptime if Integrator.uses_direct_lighting[integrator]:
        comptime for light_kind in [
            PrimitiveKind.SPHERE,
            PrimitiveKind.TRIANGLE,
        ]:
            if uniform_sampling_kind == light_kind:
                _enqueue_primary_trace[
                    integrator,
                    scene_kind,
                    sphere_format,
                    triangle_format,
                    tlas_format,
                    blas_format,
                    store_source_path,
                    light_kind,
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
                return
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
    # Fused primary threads each process one sample. Grid caps apply to
    # the grid-stride secondary, material, and shadow stages instead.
    var blocks = ceildiv(active_count, GPU_RT_BLOCK_SIZE)

    @always_inline
    def enqueue_primary[store_source_path: Bool]() raises {imm}:
        _enqueue_primary_trace_by_light_kind[
            integrator,
            scene_kind,
            sphere_format,
            triangle_format,
            tlas_format,
            blas_format,
            store_source_path,
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

    comptime if Integrator.is_path_tracing[integrator]:
        if materials.stages.needs_source_paths():
            enqueue_primary[True]()
        else:
            enqueue_primary[False]()
    else:
        enqueue_primary[False]()
    comptime if Integrator.is_path_tracing[integrator]:
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
        enqueue_coated_shading[integrator](
            ctx,
            materials,
            scene,
            queues,
            sampling,
            UInt32(0),
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
    """Enqueue trace, material, and shadow stages with one launch contract."""
    var trace_block_size = GPU_RT_BLOCK_SIZE
    comptime if Integrator.uses_direct_lighting[integrator] and (
        scene_kind.has_triangles() or scene_kind.has_instances()
    ):
        if arena.capacity >= GPU_RT_LARGE_QUEUE_MIN_CAPACITY:
            trace_block_size = GPU_RT_DIRECT_LARGE_TRACE_BLOCK_SIZE
    var blocks = min(ceildiv(arena.capacity, trace_block_size), MAX_BLOCKS)
    comptime if Integrator.uses_direct_lighting[integrator]:
        # Preserve the tuned thread budget across the two block sizes.
        # Grid-stride tracing covers every path even with an explicit cap.
        var limit = (
            GPU_RT_DIRECT_LARGE_TRACE_MAX_BLOCKS if arena.capacity
            >= GPU_RT_LARGE_QUEUE_MIN_CAPACITY else GPU_RT_DIRECT_TRACE_MAX_BLOCKS
        )
        blocks = min(blocks, limit * GPU_RT_BLOCK_SIZE // trace_block_size)
    var queues = gpu_rt_trace_queue_view(
        arena,
        src_path_ids,
        src_path_fields,
        dst_path_ids,
        dst_path_fields,
    )

    @always_inline
    def enqueue_trace() raises {imm}:
        comptime if Integrator.uses_direct_lighting[integrator]:
            comptime for light_kind in [
                PrimitiveKind.SPHERE,
                PrimitiveKind.TRIANGLE,
            ]:
                if uniform_sampling_kind == light_kind:
                    ctx.enqueue_function[
                        gpu_rt_scene_trace_kernel[
                            integrator,
                            scene_kind,
                            sphere_format,
                            triangle_format,
                            tlas_format,
                            blas_format,
                            light_kind,
                        ]
                    ](
                        scene,
                        queues,
                        sampling,
                        bounce,
                        grid_dim=blocks,
                        block_dim=trace_block_size,
                    )
                    return
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
            block_dim=trace_block_size,
        )

    enqueue_trace()
    comptime if Integrator.is_path_tracing[integrator]:
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
        enqueue_coated_shading[integrator](
            ctx,
            materials,
            scene,
            queues,
            sampling,
            bounce,
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
