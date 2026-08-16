"""End-to-end GPU wavefront renderer for sphere worlds."""

from std.math import ceildiv
from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh.camera import Camera
from bajo.core import Frame
from bajo.rt.types import (
    Color,
    RENDER,
    RenderResult,
    RenderSettings,
    RenderTimings,
    World,
)
from bajo.rt.gpu.wavefront_contract import GpuWavefrontArena
from bajo.rt.gpu.sphere_geometry import GpuRtSphereGeometry
from bajo.rt.gpu.common_kernels import GPU_RT_BLOCK_SIZE, GPU_RT_MAX_BLOCKS
from bajo.rt.gpu.resources import (
    GpuRtRenderTarget,
    download_gpu_pixels,
    enqueue_gpu_wavefront,
)
from bajo.rt.gpu.path_shading import (
    GpuRtLights,
    GpuRtMaterials,
    _enqueue_material_shading,
)
from bajo.rt.gpu.views import (
    GpuRtSceneView,
    gpu_rt_trace_queue_view,
    _immut,
    _immut,
)
from bajo.rt.gpu.scene_trace import (
    enqueue_gpu_shadows,
    gpu_rt_scene_trace_kernel,
)


struct GpuRtSphereWorld[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
]:
    """Device scene data for the sphere GPU RT specialization."""

    var geometry: GpuRtSphereGeometry[
        Frame.WORLD, Self.node_width, Self.leaf_width
    ]
    var materials: GpuRtMaterials
    var lights: GpuRtLights

    def __init__[
        world_bvh_width: SIMDLength,
        instance_bvh_width: SIMDLength,
    ](
        out self,
        mut ctx: DeviceContext,
        world: World[world_bvh_width, instance_bvh_width],
    ) raises:
        debug_assert["safe", _use_compiler_assume=True](
            len(world.spheres) > 0, "GPU sphere RT requires spheres"
        )
        debug_assert["safe", _use_compiler_assume=True](
            len(world.triangle_vertices) == 0
            and len(world.triangle_instances) == 0,
            "GPU sphere specialization requires sphere-only geometry",
        )

        self.geometry = GpuRtSphereGeometry[
            Frame.WORLD, Self.node_width, Self.leaf_width
        ](ctx, world.spheres, world.sphere_surfaces)
        self.materials = GpuRtMaterials(ctx, world)
        self.lights = GpuRtLights(ctx, world)


def _enqueue_sphere_bounce[
    ALGORITHM: RENDER,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
](
    ctx: DeviceContext,
    arena: GpuWavefrontArena,
    world: GpuRtSphereWorld[node_width, leaf_width],
    src_path_ids: DeviceBuffer[DType.uint32],
    src_path_fields: DeviceBuffer[DType.float32],
    dst_path_ids: DeviceBuffer[DType.uint32],
    dst_path_fields: DeviceBuffer[DType.float32],
    rng_seed: UInt64,
    bounce: UInt32,
) raises:
    var blocks = min(
        ceildiv(arena.capacity, GPU_RT_BLOCK_SIZE), GPU_RT_MAX_BLOCKS
    )
    var dummy_f32 = _immut(world.geometry.bvh.tree.wide_nodes)
    var dummy_u32 = _immut(world.geometry.surfaces)
    var scene = GpuRtSceneView(
        _immut(world.geometry.bvh.tree.wide_nodes),
        _immut(world.geometry.bvh.leaf_spheres),
        world.geometry.bvh.tree.root_idx,
        _immut(world.geometry.surfaces),
        _immut(world.geometry.signed_radii),
        dummy_f32,
        dummy_f32,
        UInt32(0),
        dummy_u32,
        dummy_f32,
        dummy_u32,
        dummy_f32,
        dummy_u32,
        dummy_u32,
        dummy_f32,
        dummy_f32,
        UInt32(0),
        Int32(0),
        dummy_u32,
        _immut(world.materials.emissives),
        _immut(world.materials.lambertians),
        _immut(world.materials.metals),
        _immut(world.materials.dielectrics),
        _immut(world.lights.kinds),
        _immut(world.lights.fields),
        Int32(world.lights.count),
        world.lights.total_weight,
    )
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
    ](ctx, scene, queues, arena.capacity)
    comptime if ALGORITHM in (RENDER.PATH, RENDER.NEE, RENDER.MIS):
        _enqueue_material_shading[ALGORITHM](
            ctx,
            arena,
            world.materials,
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
    world: GpuRtSphereWorld[node_width, leaf_width],
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
    world_bvh_width: SIMDLength = 16,
    instance_bvh_width: SIMDLength = 16,
](
    settings: RenderSettings,
    camera: Camera,
    world: World[world_bvh_width, instance_bvh_width],
) raises -> RenderResult:
    """Render a sphere-only `World` with the shared wavefront contract."""
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
        var gpu_world = GpuRtSphereWorld[node_width, leaf_width](ctx, world)
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
