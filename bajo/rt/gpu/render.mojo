"""Unified prepared and synchronous GPU RT entry points."""

from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh import Camera
from bajo.bvh.gpu import GpuBvhBuildMethod
from bajo.rt.gpu.bounce import enqueue_gpu_rt_bounce
from bajo.rt.gpu.common_kernels import GPU_RT_MAX_BLOCKS
from bajo.rt.gpu.policy import (
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
from bajo.rt.types import (
    Color,
    Integrator,
    RenderResult,
    RenderSettings,
    RenderTimings,
    SceneData,
)


comptime GPU_RT_CWBVH8_BLAS_TRIANGLE_THRESHOLD = 32


def _prefer_cwbvh8_triangles(world: SceneData) -> Bool:
    return (
        len(world.triangle_vertices()) / 3
        >= GPU_RT_CWBVH8_BLAS_TRIANGLE_THRESHOLD
    )


def _prefer_cwbvh8_blases(world: SceneData) -> Bool:
    var weighted_triangles = 0
    for instance in world.triangle_instances():
        weighted_triangles += (
            len(world.triangle_meshes()[Int(instance.blas_idx)]) / 3
        )
    return weighted_triangles >= (
        len(world.triangle_instances())
        * GPU_RT_CWBVH8_BLAS_TRIANGLE_THRESHOLD
    )


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
    rng_seed: UInt64,
    bounce: UInt32,
) raises:
    enqueue_gpu_rt_bounce[
        integrator,
        kind,
        sphere_format.node_width,
        sphere_format.leaf_width,
        triangle_format.node_width,
        triangle_format.leaf_width,
        tlas_format.node_width,
        tlas_format.leaf_width,
        blas_format.node_width,
        blas_format.leaf_width,
        MAX_BLOCKS,
        SHADOW_MAX_BLOCKS,
        triangle_format.layout,
        blas_format.layout,
    ](
        ctx,
        arena,
        world.view(),
        world.shading.materials,
        src_path_ids,
        src_path_fields,
        dst_path_ids,
        dst_path_fields,
        rng_seed,
        bounce,
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
        var target = GpuRtRenderTarget(ctx, settings, camera)
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


def _render_gpu_instances_default[
    kind: GpuRtSceneKind,
    integrator: Integrator,
    sphere_format: GpuRtBvhFormat,
](
    settings: RenderSettings,
    camera: Camera,
    world: SceneData,
) raises -> RenderResult:
    comptime assert kind.has_instances()
    if _prefer_cwbvh8_blases(world):
        if kind.has_triangles() and _prefer_cwbvh8_triangles(world):
            return render_gpu_configured[
                kind,
                integrator,
                sphere_format,
                GPU_RT_BVH_CWBVH8,
                GPU_RT_BVH_TLAS2,
                GPU_RT_BVH_CWBVH8,
            ](settings, camera, world)
        return render_gpu_configured[
            kind,
            integrator,
            sphere_format,
            GPU_RT_BVH_WIDE4,
            GPU_RT_BVH_TLAS2,
            GPU_RT_BVH_CWBVH8,
            triangle_build_method=.LBVH,
        ](settings, camera, world)
    if kind.has_triangles() and _prefer_cwbvh8_triangles(world):
        return render_gpu_configured[
            kind,
            integrator,
            sphere_format,
            GPU_RT_BVH_CWBVH8,
            GPU_RT_BVH_TLAS2,
            GPU_RT_BVH_WIDE4,
            blas_build_method=.LBVH,
        ](settings, camera, world)
    return render_gpu_configured[
        kind,
        integrator,
        sphere_format,
        GPU_RT_BVH_WIDE4,
        GPU_RT_BVH_TLAS2,
        GPU_RT_BVH_WIDE4,
        triangle_build_method=.LBVH,
        blas_build_method=.LBVH,
    ](settings, camera, world)


def render_gpu[
    integrator: Integrator = .PATH,
    node_width: SIMDLength = 4,
    leaf_width: SIMDLength = node_width,
](
    settings: RenderSettings,
    camera: Camera,
    world: SceneData,
) raises -> RenderResult:
    """Select a scene shape on the host; all device paths remain specialized."""
    comptime assert integrator.is_valid()
    comptime sphere_format = GpuRtBvhFormat(
        Int(node_width), Int(leaf_width), .WIDE
    )

    if len(world.triangle_instances()) > 0:
        if len(world.spheres()) > 0:
            if len(world.triangle_vertices()) > 0:
                return _render_gpu_instances_default[
                    .ALL, integrator, sphere_format
                ](settings, camera, world)
            return _render_gpu_instances_default[
                .SPHERES_INSTANCES, integrator, sphere_format
            ](settings, camera, world)
        if len(world.triangle_vertices()) > 0:
            return _render_gpu_instances_default[
                .TRIANGLES_INSTANCES, integrator, sphere_format
            ](settings, camera, world)
        return _render_gpu_instances_default[
            .INSTANCES, integrator, sphere_format
        ](settings, camera, world)

    if len(world.spheres()) > 0:
        if len(world.triangle_vertices()) > 0:
            if _prefer_cwbvh8_triangles(world):
                return render_gpu_configured[
                    .SPHERES_TRIANGLES,
                    integrator,
                    sphere_format,
                    GPU_RT_BVH_CWBVH8,
                ](settings, camera, world)
            return render_gpu_configured[
                .SPHERES_TRIANGLES,
                integrator,
                sphere_format,
                GPU_RT_BVH_WIDE4,
                triangle_build_method=.LBVH,
            ](settings, camera, world)
        return render_gpu_configured[.SPHERES, integrator, sphere_format](
            settings, camera, world
        )

    if _prefer_cwbvh8_triangles(world):
        return render_gpu_configured[
            .TRIANGLES,
            integrator,
            sphere_format,
            GPU_RT_BVH_CWBVH8,
        ](settings, camera, world)
    return render_gpu_configured[
        .TRIANGLES,
        integrator,
        sphere_format,
        GPU_RT_BVH_WIDE4,
        triangle_build_method=.LBVH,
    ](settings, camera, world)
