"""End-to-end GPU wavefront renderer for world-space triangle scenes."""

from std.math import ceildiv
from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh.camera import Camera
from bajo.bvh.gpu.builder import GpuBvhBuildMethod
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
from bajo.rt.gpu.triangle_geometry import GpuRtTriangleGeometry
from bajo.rt.gpu.common_kernels import GPU_RT_BLOCK_SIZE, GPU_RT_MAX_BLOCKS
from bajo.rt.gpu.resources import (
    GpuRtRenderTarget,
    download_gpu_pixels,
    enqueue_gpu_wavefront,
    upload_surface_ids,
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
from bajo.rt.gpu.path_shading import (
    GpuRtMaterials,
    GpuRtLights,
    _enqueue_material_shading,
)


struct GpuRtTriangleWorld[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
    build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.HPLOC,
    compressed: Bool = node_width == 8 and leaf_width == 4,
]:
    """Triangle BVH, surface sidecar, and shared material tables on device."""

    var geometry: GpuRtTriangleGeometry[
        Frame.WORLD,
        Self.node_width,
        Self.leaf_width,
        Self.build_method,
        Self.compressed,
    ]
    var triangle_surfaces: DeviceBuffer[DType.uint32]
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
            len(world.triangle_vertices) > 0,
            "GPU triangle RT requires world-space triangles",
        )
        debug_assert["safe", _use_compiler_assume=True](
            len(world.spheres) == 0 and len(world.triangle_instances) == 0,
            "GPU triangle RT accepts triangle-only worlds",
        )
        self.geometry = GpuRtTriangleGeometry[
            Frame.WORLD,
            Self.node_width,
            Self.leaf_width,
            Self.build_method,
            Self.compressed,
        ](ctx, world.triangle_vertices)
        self.triangle_surfaces = upload_surface_ids(
            ctx, world.triangle_surfaces
        )
        self.materials = GpuRtMaterials(ctx, world)
        self.lights = GpuRtLights(ctx, world)


def _enqueue_triangle_bounce[
    ALGORITHM: RENDER,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    MAX_BLOCKS: Int = GPU_RT_MAX_BLOCKS,
    SHADOW_MAX_BLOCKS: Int = MAX_BLOCKS,
    build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.HPLOC,
    compressed: Bool = node_width == 8 and leaf_width == 4,
](
    ctx: DeviceContext,
    arena: GpuWavefrontArena,
    world: GpuRtTriangleWorld[node_width, leaf_width, build_method, compressed],
    src_path_ids: DeviceBuffer[DType.uint32],
    src_path_fields: DeviceBuffer[DType.float32],
    dst_path_ids: DeviceBuffer[DType.uint32],
    dst_path_fields: DeviceBuffer[DType.float32],
    rng_seed: UInt64,
    bounce: UInt32,
) raises:
    var blocks = min(ceildiv(arena.capacity, GPU_RT_BLOCK_SIZE), MAX_BLOCKS)
    var dummy_f32 = _immut(world.geometry.nodes)
    var dummy_u32 = _immut(world.triangle_surfaces)
    var scene = GpuRtSceneView(
        dummy_f32,
        dummy_f32,
        UInt32(0),
        dummy_u32,
        dummy_f32,
        _immut(world.geometry.nodes),
        _immut(world.geometry.leaves),
        world.geometry.root,
        _immut(world.triangle_surfaces),
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
            False,
            True,
            False,
            node_width,
            leaf_width,
            node_width,
            leaf_width,
            node_width,
            leaf_width,
            node_width,
            leaf_width,
            compressed,
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
        False,
        True,
        False,
        node_width,
        leaf_width,
        node_width,
        leaf_width,
        node_width,
        leaf_width,
        node_width,
        leaf_width,
        SHADOW_MAX_BLOCKS,
        compressed,
    ](ctx, scene, queues, arena.capacity)
    comptime if ALGORITHM in (RENDER.PATH, RENDER.NEE, RENDER.MIS):
        _enqueue_material_shading[ALGORITHM, MAX_BLOCKS](
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


def enqueue_render_gpu_triangles[
    ALGORITHM: RENDER = RENDER.PATH,
    MAX_DEPTH: Int = 8,
    node_width: SIMDLength = 8,
    leaf_width: SIMDLength = 4,
    MAX_BLOCKS: Int = GPU_RT_MAX_BLOCKS,
    SHADOW_MAX_BLOCKS: Int = MAX_BLOCKS,
    build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.HPLOC,
    compressed: Bool = node_width == 8 and leaf_width == 4,
](
    ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    world: GpuRtTriangleWorld[node_width, leaf_width, build_method, compressed],
    settings: RenderSettings,
) raises:
    """Submit a triangle render asynchronously into `target.pixels`."""
    enqueue_gpu_wavefront[
        ALGORITHM,
        MAX_DEPTH,
        _enqueue_triangle_bounce[
            ALGORITHM,
            node_width,
            leaf_width,
            MAX_BLOCKS,
            SHADOW_MAX_BLOCKS,
            build_method,
            compressed,
        ],
    ](ctx, target, world, settings)


def render_gpu_triangles[
    ALGORITHM: RENDER = RENDER.PATH,
    MAX_DEPTH: Int = 8,
    node_width: SIMDLength = 8,
    leaf_width: SIMDLength = 4,
    world_bvh_width: SIMDLength = 16,
    instance_bvh_width: SIMDLength = 16,
    build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.HPLOC,
    compressed: Bool = node_width == 8 and leaf_width == 4,
](
    settings: RenderSettings,
    camera: Camera,
    world: World[world_bvh_width, instance_bvh_width],
) raises -> RenderResult:
    """Render a triangle-only `World` with the shared wavefront stages."""
    comptime assert ALGORITHM in (
        RENDER.PATH,
        RENDER.NORMALS,
        RENDER.AO,
        RENDER.NEE,
        RENDER.MIS,
    )
    comptime assert MAX_DEPTH >= 0
    var total_t0 = perf_counter_ns()
    var pixel_count = settings.image_width * settings.image_height
    var sample_count = pixel_count * settings.samples_per_pixel
    var pixels: List[Color]
    var init_ns: Int
    var render_ns: Int

    with DeviceContext() as ctx:
        var init_t0 = perf_counter_ns()
        var gpu_world = GpuRtTriangleWorld[
            node_width, leaf_width, build_method, compressed
        ](ctx, world)
        var target = GpuRtRenderTarget(ctx, settings, camera)
        init_ns = Int(perf_counter_ns() - init_t0)

        var render_t0 = perf_counter_ns()
        enqueue_render_gpu_triangles[
            ALGORITHM,
            MAX_DEPTH,
            node_width,
            leaf_width,
            GPU_RT_MAX_BLOCKS,
            GPU_RT_MAX_BLOCKS,
            build_method,
            compressed,
        ](
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
            MAX_DEPTH,
        ),
    )
