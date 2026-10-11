"""Unified prepared and synchronous GPU RT entry points."""

from max.gpu import global_idx
from std.math import ceildiv
from bajo.rt.film import FilmResponse
from bajo.rt.render_types import Color
from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh import Camera
from bajo.bvh.gpu import GpuBvhBuildMethod
from bajo.rt.gpu.bounce import (
    enqueue_gpu_rt_bounce,
    enqueue_gpu_rt_primary_bounce,
)
from bajo.rt.gpu.common_kernels import GPU_RT_MAX_BLOCKS
from bajo.rt.gpu.config import (
    GpuRtBvhFormat,
    GpuRtSceneKind,
    GPU_RT_BVH_CWBVH8,
    GPU_RT_BVH_TLAS2,
    GPU_RT_BVH_WIDE4,
)
from bajo.rt.gpu.resources import (
    GpuRtRenderTarget,
    download_gpu_pixels,
    enqueue_gpu_wavefront,
)
from bajo.rt.gpu.scene import GpuRtScene, prepare_gpu_scene
from bajo.rt.gpu.wavefront_contract import GpuWavefrontArena
from bajo.rt.render_types import (
    Color,
    RenderResult,
    RenderSettings,
    RenderTimings,
    SamplingConfig,
)
from bajo.rt.material_types import Integrator
from bajo.rt.scene_data import SceneData


def _enqueue_scene_bounce[
    integrator: Integrator,
    kind: GpuRtSceneKind,
    sphere_format: GpuRtBvhFormat,
    triangle_format: GpuRtBvhFormat,
    tlas_format: GpuRtBvhFormat,
    blas_format: GpuRtBvhFormat,
    MAX_BLOCKS: Int,
    SHADOW_MAX_BLOCKS: Int,
](
    ctx: DeviceContext,
    arena: GpuWavefrontArena,
    world: GpuRtScene[
        kind, sphere_format, triangle_format, tlas_format, blas_format
    ],
    src_path_ids: DeviceBuffer[.uint32],
    src_path_fields: DeviceBuffer[.float32],
    dst_path_ids: DeviceBuffer[.uint32],
    dst_path_fields: DeviceBuffer[.float32],
    sampling: SamplingConfig,
    bounce: UInt32,
) raises:
    enqueue_gpu_rt_bounce[
        integrator,
        kind,
        sphere_format,
        triangle_format,
        tlas_format,
        blas_format,
        MAX_BLOCKS,
        SHADOW_MAX_BLOCKS,
    ](
        ctx,
        arena,
        world.view(),
        world.materials,
        world.lights.uniform_sampling_kind,
        src_path_ids,
        src_path_fields,
        dst_path_ids,
        dst_path_fields,
        sampling,
        bounce,
    )


def _enqueue_scene_primary_bounce[
    integrator: Integrator,
    kind: GpuRtSceneKind,
    sphere_format: GpuRtBvhFormat,
    triangle_format: GpuRtBvhFormat,
    tlas_format: GpuRtBvhFormat,
    blas_format: GpuRtBvhFormat,
    MAX_BLOCKS: Int,
    SHADOW_MAX_BLOCKS: Int,
](
    ctx: DeviceContext,
    arena: GpuWavefrontArena,
    camera: DeviceBuffer[.float32],
    world: GpuRtScene[
        kind, sphere_format, triangle_format, tlas_format, blas_format
    ],
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
    enqueue_gpu_rt_primary_bounce[
        integrator,
        kind,
        sphere_format,
        triangle_format,
        tlas_format,
        blas_format,
        MAX_BLOCKS,
        SHADOW_MAX_BLOCKS,
    ](
        ctx,
        arena,
        world.view(),
        world.materials,
        world.lights.uniform_sampling_kind,
        camera,
        src_path_ids,
        src_path_fields,
        dst_path_ids,
        dst_path_fields,
        active_count,
        image_width,
        image_height,
        samples_per_pixel,
        sampling,
    )


def enqueue_render_gpu[
    integrator: Integrator = .PATH,
    kind: GpuRtSceneKind = .ALL,
    sphere_format: GpuRtBvhFormat = GPU_RT_BVH_WIDE4,
    triangle_format: GpuRtBvhFormat = GPU_RT_BVH_CWBVH8,
    tlas_format: GpuRtBvhFormat = GPU_RT_BVH_TLAS2,
    blas_format: GpuRtBvhFormat = GPU_RT_BVH_CWBVH8,
    MAX_BLOCKS: Int = GPU_RT_MAX_BLOCKS,
    SHADOW_MAX_BLOCKS: Int = MAX_BLOCKS,
](
    ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    world: GpuRtScene[
        kind, sphere_format, triangle_format, tlas_format, blas_format
    ],
    settings: RenderSettings,
) raises:
    """Submit any prepared scene through one compile-time-specialized path."""
    enqueue_gpu_wavefront[
        integrator,
        _enqueue_scene_primary_bounce[
            integrator,
            kind,
            sphere_format,
            triangle_format,
            tlas_format,
            blas_format,
            MAX_BLOCKS,
            SHADOW_MAX_BLOCKS,
        ],
        _enqueue_scene_bounce[
            integrator,
            kind,
            sphere_format,
            triangle_format,
            tlas_format,
            blas_format,
            MAX_BLOCKS,
            SHADOW_MAX_BLOCKS,
        ],
    ](ctx, target, world, settings)
    if settings.film.is_identity():
        return
    ctx.enqueue_function[gpu_rt_apply_film_kernel](
        target.pixels,
        Int32(target.pixel_count),
        settings.film,
        grid_dim=ceildiv(target.pixel_count, 128),
        block_dim=128,
    )


def render_gpu_configured[
    kind: GpuRtSceneKind,
    integrator: Integrator = .PATH,
    sphere_format: GpuRtBvhFormat = GPU_RT_BVH_WIDE4,
    triangle_format: GpuRtBvhFormat = GPU_RT_BVH_CWBVH8,
    tlas_format: GpuRtBvhFormat = GPU_RT_BVH_TLAS2,
    blas_format: GpuRtBvhFormat = GPU_RT_BVH_CWBVH8,
    triangle_build_method: GpuBvhBuildMethod = .HPLOC,
    tlas_build_method: GpuBvhBuildMethod = .LBVH,
    blas_build_method: GpuBvhBuildMethod = .HPLOC,
](
    settings: RenderSettings,
    camera: Camera,
    world: SceneData,
) raises -> RenderResult:
    """Render one explicitly selected scene shape, format, and builder set."""
    comptime assert integrator.is_valid()
    var total_t0 = perf_counter_ns()
    var pixel_count = settings.image_width * settings.image_height
    var sample_count = pixel_count * settings.samples_per_pixel
    var pixels: List[Color]
    var init_ns: Int
    var render_ns: Int

    with DeviceContext() as ctx:
        var init_t0 = perf_counter_ns()
        var gpu_world = prepare_gpu_scene[
            kind,
            sphere_format,
            triangle_format,
            tlas_format,
            blas_format,
            triangle_build_method,
            tlas_build_method,
            blas_build_method,
        ](ctx, world)
        var target = gpu_world.make_render_target(ctx, settings, camera)
        init_ns = Int(perf_counter_ns() - init_t0)

        var render_t0 = perf_counter_ns()
        enqueue_render_gpu[integrator](ctx, target, gpu_world, settings)
        pixels = download_gpu_pixels(ctx, target)
        render_ns = Int(perf_counter_ns() - render_t0)

    return RenderResult(
        pixels^,
        RenderTimings(
            Int(perf_counter_ns() - total_t0),
            init_ns,
            render_ns,
            pixel_count,
            sample_count,
            settings.max_depth,
        ),
    )


def render_gpu[
    integrator: Integrator = .PATH,
](
    settings: RenderSettings,
    camera: Camera,
    world: SceneData,
) raises -> RenderResult:
    """Select geometry with WIDE4 spheres, CWBVH8 triangles/BLASes and TLAS2."""
    comptime assert integrator.is_valid()
    if len(world.triangle_instances()) > 0:
        if len(world.spheres()) > 0:
            if len(world.triangle_vertices()) > 0:
                return render_gpu_configured[.ALL, integrator](
                    settings, camera, world
                )
            return render_gpu_configured[.SPHERES_INSTANCES, integrator](
                settings, camera, world
            )
        if len(world.triangle_vertices()) > 0:
            return render_gpu_configured[.TRIANGLES_INSTANCES, integrator](
                settings, camera, world
            )
        return render_gpu_configured[.INSTANCES, integrator](
            settings, camera, world
        )

    if len(world.spheres()) > 0:
        if len(world.triangle_vertices()) > 0:
            return render_gpu_configured[.SPHERES_TRIANGLES, integrator](
                settings, camera, world
            )
        return render_gpu_configured[.SPHERES, integrator](
            settings, camera, world
        )
    return render_gpu_configured[.TRIANGLES, integrator](
        settings, camera, world
    )


def render_gpu_viewer[
    integrator: Integrator = .PATH,
](
    settings: RenderSettings,
    camera: Camera,
    world: SceneData,
) raises -> RenderResult:
    """Render the viewer with the fixed default GPU BVH formats."""
    return render_gpu[integrator](settings, camera, world)


def gpu_rt_apply_film_kernel(
    pixels: Pointer[Float32, MutAnyOrigin], count: Int32, film: FilmResponse
):
    var index = global_idx.x
    if index < Int(count):
        var base = 3 * index
        var result = film.apply(
            Color(
                pixels[unsafe_offset=base],
                pixels[unsafe_offset=base + 1],
                pixels[unsafe_offset=base + 2],
            )
        )
        pixels[unsafe_offset=base] = result.x
        pixels[unsafe_offset=base + 1] = result.y
        pixels[unsafe_offset=base + 2] = result.z
