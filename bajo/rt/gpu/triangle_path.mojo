"""End-to-end GPU wavefront renderer for world-space triangle scenes."""

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
    SceneData,
)
from bajo.rt.gpu.wavefront_contract import GpuWavefrontArena
from bajo.rt.gpu.scene import GpuScene
from bajo.rt.gpu.triangle_geometry import GpuRtTriangleGeometry
from bajo.rt.gpu.common_kernels import GPU_RT_MAX_BLOCKS
from bajo.rt.gpu.resources import (
    GpuRtRenderTarget,
    download_gpu_pixels,
    enqueue_gpu_wavefront,
    upload_surface_ids,
)
from bajo.rt.gpu.views import (
    GpuRtInstanceView,
    GpuRtSphereView,
    GpuRtTriangleView,
    _immut,
    gpu_rt_scene_view,
)
from bajo.rt.gpu.path_shading import (
    GpuRtShadingResources,
)
from bajo.rt.gpu.bounce import enqueue_gpu_rt_bounce


struct GpuRtTriangleScene[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
    build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.HPLOC,
    compressed: Bool = node_width == 8 and leaf_width == 4,
](GpuScene):
    """Triangle BVH, surface sidecar, and shared material tables on device."""

    comptime is_prepared_gpu_scene = True

    var geometry: GpuRtTriangleGeometry[
        Frame.WORLD,
        Self.node_width,
        Self.leaf_width,
        Self.build_method,
        Self.compressed,
    ]
    var triangle_surfaces: DeviceBuffer[DType.uint32]
    var shading: GpuRtShadingResources

    def __init__(
        out self,
        mut ctx: DeviceContext,
        world: SceneData,
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
        self.shading = GpuRtShadingResources(ctx, world)


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
    world: GpuRtTriangleScene[node_width, leaf_width, build_method, compressed],
    src_path_ids: DeviceBuffer[DType.uint32],
    src_path_fields: DeviceBuffer[DType.float32],
    dst_path_ids: DeviceBuffer[DType.uint32],
    dst_path_fields: DeviceBuffer[DType.float32],
    rng_seed: UInt64,
    bounce: UInt32,
) raises:
    var scene = gpu_rt_scene_view(
        None,
        GpuRtTriangleView(
            _immut(world.geometry.nodes),
            _immut(world.geometry.leaves),
            world.geometry.root,
            _immut(world.triangle_surfaces),
        ),
        None,
        world.shading.view(),
    )
    enqueue_gpu_rt_bounce[
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
        MAX_BLOCKS,
        SHADOW_MAX_BLOCKS,
        compressed,
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


def enqueue_render_gpu_triangles[
    ALGORITHM: RENDER = RENDER.PATH,
    node_width: SIMDLength = 8,
    leaf_width: SIMDLength = 4,
    MAX_BLOCKS: Int = GPU_RT_MAX_BLOCKS,
    SHADOW_MAX_BLOCKS: Int = MAX_BLOCKS,
    build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.HPLOC,
    compressed: Bool = node_width == 8 and leaf_width == 4,
](
    ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    world: GpuRtTriangleScene[node_width, leaf_width, build_method, compressed],
    settings: RenderSettings,
) raises:
    """Submit a triangle render asynchronously into `target.pixels`."""
    enqueue_gpu_wavefront[
        ALGORITHM,
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
    node_width: SIMDLength = 8,
    leaf_width: SIMDLength = 4,
    build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.HPLOC,
    compressed: Bool = node_width == 8 and leaf_width == 4,
](
    settings: RenderSettings,
    camera: Camera,
    world: SceneData,
) raises -> RenderResult:
    """Render a triangle-only `SceneData` with the shared wavefront stages."""
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
        var gpu_world = GpuRtTriangleScene[
            node_width, leaf_width, build_method, compressed
        ](ctx, world)
        var target = GpuRtRenderTarget(ctx, settings, camera)
        init_ns = Int(perf_counter_ns() - init_t0)

        var render_t0 = perf_counter_ns()
        enqueue_render_gpu_triangles[
            ALGORITHM,
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
            settings.max_depth,
        ),
    )
