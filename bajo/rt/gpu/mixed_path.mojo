"""GPU wavefront renderer for mixed world-space sphere/triangle scenes."""

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
from bajo.rt.gpu.sphere_geometry import GpuRtSphereGeometry
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


struct GpuRtMixedWorld[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
    triangle_node_width: SIMDLength = 8,
    triangle_leaf_width: SIMDLength = 4,
    triangle_build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.HPLOC,
    triangle_compressed: Bool = triangle_node_width == 8
    and triangle_leaf_width == 4,
]:
    """Sphere and triangle BVHs plus their compact surface sidecars."""

    var sphere_geometry: GpuRtSphereGeometry[
        Frame.WORLD, Self.node_width, Self.leaf_width
    ]
    var triangle_geometry: GpuRtTriangleGeometry[
        Frame.WORLD,
        Self.triangle_node_width,
        Self.triangle_leaf_width,
        Self.triangle_build_method,
        Self.triangle_compressed,
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
            len(world.spheres) > 0 and len(world.triangle_vertices) > 0,
            "GPU mixed RT requires spheres and world-space triangles",
        )
        debug_assert["safe", _use_compiler_assume=True](
            len(world.triangle_instances) == 0,
            "GPU mixed-static specialization excludes triangle instances",
        )

        self.sphere_geometry = GpuRtSphereGeometry[
            Frame.WORLD, Self.node_width, Self.leaf_width
        ](ctx, world.spheres, world.sphere_surfaces)
        self.triangle_geometry = GpuRtTriangleGeometry[
            Frame.WORLD,
            Self.triangle_node_width,
            Self.triangle_leaf_width,
            Self.triangle_build_method,
            Self.triangle_compressed,
        ](ctx, world.triangle_vertices)
        self.triangle_surfaces = upload_surface_ids(
            ctx, world.triangle_surfaces
        )
        self.materials = GpuRtMaterials(ctx, world)
        self.lights = GpuRtLights(ctx, world)


def _enqueue_mixed_bounce[
    ALGORITHM: RENDER,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    triangle_node_width: SIMDLength,
    triangle_leaf_width: SIMDLength,
    triangle_build_method: GpuBvhBuildMethod,
    triangle_compressed: Bool,
](
    ctx: DeviceContext,
    arena: GpuWavefrontArena,
    world: GpuRtMixedWorld[
        node_width,
        leaf_width,
        triangle_node_width,
        triangle_leaf_width,
        triangle_build_method,
        triangle_compressed,
    ],
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
    var dummy_f32 = _immut(world.sphere_geometry.bvh.tree.wide_nodes)
    var dummy_u32 = _immut(world.sphere_geometry.surfaces)
    var scene = GpuRtSceneView(
        _immut(world.sphere_geometry.bvh.tree.wide_nodes),
        _immut(world.sphere_geometry.bvh.leaf_spheres),
        world.sphere_geometry.bvh.tree.root_idx,
        _immut(world.sphere_geometry.surfaces),
        _immut(world.sphere_geometry.signed_radii),
        _immut(world.triangle_geometry.nodes),
        _immut(world.triangle_geometry.leaves),
        world.triangle_geometry.root,
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
            True,
            True,
            False,
            node_width,
            leaf_width,
            triangle_node_width,
            triangle_leaf_width,
            node_width,
            leaf_width,
            node_width,
            leaf_width,
            triangle_compressed,
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
        True,
        False,
        node_width,
        leaf_width,
        triangle_node_width,
        triangle_leaf_width,
        node_width,
        leaf_width,
        node_width,
        leaf_width,
        GPU_RT_MAX_BLOCKS,
        triangle_compressed,
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


def enqueue_render_gpu_mixed[
    ALGORITHM: RENDER = RENDER.PATH,
    node_width: SIMDLength = 4,
    leaf_width: SIMDLength = node_width,
    triangle_node_width: SIMDLength = 8,
    triangle_leaf_width: SIMDLength = 4,
    triangle_build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.HPLOC,
    triangle_compressed: Bool = triangle_node_width == 8
    and triangle_leaf_width == 4,
](
    ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    world: GpuRtMixedWorld[
        node_width,
        leaf_width,
        triangle_node_width,
        triangle_leaf_width,
        triangle_build_method,
        triangle_compressed,
    ],
    settings: RenderSettings,
) raises:
    """Submit a mixed static-geometry render into `target.pixels`."""
    enqueue_gpu_wavefront[
        ALGORITHM,
        _enqueue_mixed_bounce[
            ALGORITHM,
            node_width,
            leaf_width,
            triangle_node_width,
            triangle_leaf_width,
            triangle_build_method,
            triangle_compressed,
        ],
    ](ctx, target, world, settings)


def render_gpu_mixed[
    ALGORITHM: RENDER = RENDER.PATH,
    node_width: SIMDLength = 4,
    leaf_width: SIMDLength = node_width,
    world_bvh_width: SIMDLength = 16,
    instance_bvh_width: SIMDLength = 16,
    triangle_node_width: SIMDLength = 8,
    triangle_leaf_width: SIMDLength = 4,
    triangle_build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.HPLOC,
    triangle_compressed: Bool = triangle_node_width == 8
    and triangle_leaf_width == 4,
](
    settings: RenderSettings,
    camera: Camera,
    world: World[world_bvh_width, instance_bvh_width],
) raises -> RenderResult:
    """Render a world containing both spheres and world-space triangles."""
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
        var gpu_world = GpuRtMixedWorld[
            node_width,
            leaf_width,
            triangle_node_width,
            triangle_leaf_width,
            triangle_build_method,
            triangle_compressed,
        ](ctx, world)
        var target = GpuRtRenderTarget(ctx, settings, camera)
        init_ns = Int(perf_counter_ns() - init_t0)

        var render_t0 = perf_counter_ns()
        enqueue_render_gpu_mixed[
            ALGORITHM,
            node_width,
            leaf_width,
            triangle_node_width,
            triangle_leaf_width,
            triangle_build_method,
            triangle_compressed,
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
            settings.max_depth,
        ),
    )
