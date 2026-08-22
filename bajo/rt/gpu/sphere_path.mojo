"""End-to-end GPU wavefront renderer for sphere worlds."""

from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh import Camera
from bajo.core import Frame
from bajo.rt.types import (
    Color,
    RENDER,
    RenderResult,
    RenderSettings,
    RenderTimings,
    SceneData,
)
from bajo.rt.gpu.wavefront_contract import GpuWavefrontArena
from bajo.rt.gpu.sphere_geometry import GpuRtSphereGeometry
from bajo.rt.gpu.resources import (
    GpuRtRenderTarget,
    download_gpu_pixels,
    enqueue_gpu_wavefront,
)
from bajo.rt.gpu.path_shading import (
    GpuRtShadingResources,
)
from bajo.rt.gpu.views import (
    GpuRtInstanceView,
    GpuRtSphereView,
    GpuRtTriangleView,
    _immut,
    gpu_rt_scene_view,
)
from bajo.rt.gpu.bounce import enqueue_gpu_rt_bounce


struct GpuRtSphereScene[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
]:
    """Device scene data for the sphere GPU RT specialization."""

    var geometry: GpuRtSphereGeometry[
        Frame.WORLD, Self.node_width, Self.leaf_width
    ]
    var shading: GpuRtShadingResources

    def __init__(
        out self,
        mut ctx: DeviceContext,
        world: SceneData,
    ) raises:
        debug_assert["safe", _use_compiler_assume=True](
            len(world.spheres()) > 0, "GPU sphere RT requires spheres"
        )
        debug_assert["safe", _use_compiler_assume=True](
            len(world.triangle_vertices()) == 0
            and len(world.triangle_instances()) == 0,
            "GPU sphere specialization requires sphere-only geometry",
        )

        self.geometry = GpuRtSphereGeometry[
            Frame.WORLD, Self.node_width, Self.leaf_width
        ](ctx, world.spheres(), world.sphere_surfaces())
        self.shading = GpuRtShadingResources(ctx, world)


def _enqueue_sphere_bounce[
    ALGORITHM: RENDER,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
](
    ctx: DeviceContext,
    arena: GpuWavefrontArena,
    world: GpuRtSphereScene[node_width, leaf_width],
    src_path_ids: DeviceBuffer[DType.uint32],
    src_path_fields: DeviceBuffer[DType.float32],
    dst_path_ids: DeviceBuffer[DType.uint32],
    dst_path_fields: DeviceBuffer[DType.float32],
    rng_seed: UInt64,
    bounce: UInt32,
) raises:
    var scene = gpu_rt_scene_view(
        GpuRtSphereView(
            _immut(world.geometry.bvh.tree.wide_nodes),
            _immut(world.geometry.bvh.leaf_spheres),
            world.geometry.bvh.tree.root_idx,
            _immut(world.geometry.surfaces),
            _immut(world.geometry.signed_radii),
        ),
        None,
        None,
        world.shading.view(),
    )
    enqueue_gpu_rt_bounce[
        ALGORITHM,
        True,
        False,
        False,
        node_width,
        leaf_width,
        node_width,
        leaf_width,
        node_width,
        leaf_width,
        node_width,
        leaf_width,
    ](
        ctx,
        arena,
        scene,
        world.shading.materials,
        src_path_ids,
        src_path_fields,
        dst_path_ids,
        dst_path_fields,
        rng_seed,
        bounce,
    )


def enqueue_render_gpu_spheres[
    ALGORITHM: RENDER = RENDER.PATH,
    node_width: SIMDLength = 4,
    leaf_width: SIMDLength = node_width,
](
    ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    world: GpuRtSphereScene[node_width, leaf_width],
    settings: RenderSettings,
) raises:
    """Submit a sphere render asynchronously into `target.pixels`."""
    enqueue_gpu_wavefront[
        ALGORITHM,
        _enqueue_sphere_bounce[ALGORITHM, node_width, leaf_width],
    ](ctx, target, world, settings)


def render_gpu_spheres[
    ALGORITHM: RENDER = RENDER.PATH,
    node_width: SIMDLength = 4,
    leaf_width: SIMDLength = node_width,
](
    settings: RenderSettings,
    camera: Camera,
    world: SceneData,
) raises -> RenderResult:
    """Render a sphere-only `SceneData` with the shared wavefront contract."""
    comptime assert ALGORITHM in (
        RENDER.PATH,
        RENDER.NORMALS,
        RENDER.AO,
        RENDER.NEE,
        RENDER.MIS,
    )
    var total_t0 = perf_counter_ns()
    var pixel_count = settings.image_width * settings.image_height
    var sample_count = pixel_count * settings.samples_per_pixel
    var pixels: List[Color]
    var init_ns: Int
    var render_ns: Int

    with DeviceContext() as ctx:
        var init_t0 = perf_counter_ns()
        var gpu_world = GpuRtSphereScene[node_width, leaf_width](ctx, world)
        var target = GpuRtRenderTarget(ctx, settings, camera)
        init_ns = Int(perf_counter_ns() - init_t0)

        var render_t0 = perf_counter_ns()
        enqueue_render_gpu_spheres[ALGORITHM, node_width, leaf_width](
            ctx,
            target,
            gpu_world,
            settings,
        )
        pixels = download_gpu_pixels(ctx, target)
        render_ns = Int(perf_counter_ns() - render_t0)

    var total_ns = Int(perf_counter_ns() - total_t0)
    return RenderResult(
        pixels^,
        RenderTimings(
            total_ns,
            init_ns,
            render_ns,
            pixel_count,
            sample_count,
            settings.max_depth,
        ),
    )
