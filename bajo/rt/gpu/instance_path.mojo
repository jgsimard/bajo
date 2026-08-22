"""GPU wavefront renderer for instanced triangle BLAS/TLAS scenes."""

from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh import Camera
from bajo.bvh.constants import Primitive
from bajo.bvh.gpu import (
    GpuBlasSet,
    GpuBvhBuildMethod,
    GpuBvhLayout,
    GpuTriangleTlas,
    build_triangle_blas_set,
    build_triangle_tlas,
)
from bajo.rt.types import (
    Color,
    RENDER,
    RenderResult,
    RenderSettings,
    RenderTimings,
    SceneData,
)
from bajo.rt.gpu.wavefront_contract import GpuWavefrontArena
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


struct GpuRtTriangleInstanceScene[
    tlas_node_width: SIMDLength,
    blas_node_width: SIMDLength,
    tlas_leaf_width: SIMDLength = tlas_node_width,
    blas_leaf_width: SIMDLength = blas_node_width,
    blas_build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.HPLOC,
    blas_compressed: Bool = blas_node_width == 8 and blas_leaf_width == 4,
    tlas_build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.LBVH,
]:
    """Packed triangle BLAS set, typed TLAS, surfaces, and materials."""

    var blases: GpuBlasSet[
        Primitive.TRIANGLE,
        GpuBvhLayout(Self.blas_compressed),
        Self.blas_node_width,
        Self.blas_leaf_width,
    ]
    var tlas: GpuTriangleTlas[
        Self.tlas_node_width,
        Self.blas_node_width,
        Self.tlas_leaf_width,
        Self.blas_leaf_width,
        GpuBvhLayout(Self.blas_compressed),
    ]
    var instance_surfaces: DeviceBuffer[DType.uint32]
    var shading: GpuRtShadingResources

    def __init__(
        out self,
        mut ctx: DeviceContext,
        world: SceneData,
    ) raises:
        debug_assert["safe", _use_compiler_assume=True](
            len(world.triangle_instances) > 0,
            "GPU instance RT requires triangle instances",
        )
        debug_assert["safe", _use_compiler_assume=True](
            len(world.spheres) == 0 and len(world.triangle_vertices) == 0,
            "GPU instance RT currently accepts instance-only worlds",
        )
        self.blases = build_triangle_blas_set[
            Self.blas_node_width,
            Self.blas_leaf_width,
            Self.blas_build_method,
            GpuBvhLayout(Self.blas_compressed),
        ](ctx, world.triangle_meshes)
        self.tlas = build_triangle_tlas[
            Self.tlas_node_width,
            Self.blas_node_width,
            Self.tlas_leaf_width,
            Self.blas_leaf_width,
            Self.tlas_build_method,
            GpuBvhLayout(Self.blas_compressed),
        ](ctx, world.triangle_instances)
        self.instance_surfaces = upload_surface_ids(
            ctx, world.triangle_instance_surfaces
        )
        self.shading = GpuRtShadingResources(ctx, world)


def _enqueue_instance_bounce[
    ALGORITHM: RENDER,
    tlas_node_width: SIMDLength,
    tlas_leaf_width: SIMDLength,
    blas_node_width: SIMDLength,
    blas_leaf_width: SIMDLength,
    blas_build_method: GpuBvhBuildMethod,
    blas_compressed: Bool,
    tlas_build_method: GpuBvhBuildMethod,
](
    ctx: DeviceContext,
    arena: GpuWavefrontArena,
    world: GpuRtTriangleInstanceScene[
        tlas_node_width,
        blas_node_width,
        tlas_leaf_width,
        blas_leaf_width,
        blas_build_method,
        blas_compressed,
        tlas_build_method,
    ],
    src_path_ids: DeviceBuffer[DType.uint32],
    src_path_fields: DeviceBuffer[DType.float32],
    dst_path_ids: DeviceBuffer[DType.uint32],
    dst_path_fields: DeviceBuffer[DType.float32],
    rng_seed: UInt64,
    bounce: UInt32,
) raises:
    var scene = gpu_rt_scene_view(
        None,
        None,
        GpuRtInstanceView(
            _immut(world.tlas.core.tree.wide_nodes),
            _immut(world.tlas.core.tree.leaf_block_indices),
            _immut(world.tlas.core.inst_inv_transform),
            _immut(world.tlas.core.inst_blas_indices),
            _immut(world.blases.descs),
            _immut(world.blases.nodes),
            _immut(world.blases.leaves),
            world.tlas.core.tree.root_idx,
            Int32(world.tlas.core.inst_count),
            Int32(world.blases.blas_count),
            _immut(world.instance_surfaces),
        ),
        world.shading.view(),
    )
    enqueue_gpu_rt_bounce[
        ALGORITHM,
        False,
        False,
        True,
        tlas_node_width,
        tlas_leaf_width,
        tlas_node_width,
        tlas_leaf_width,
        tlas_node_width,
        tlas_leaf_width,
        blas_node_width,
        blas_leaf_width,
        blas_compressed=blas_compressed,
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


def enqueue_render_gpu_triangle_instances[
    ALGORITHM: RENDER = RENDER.PATH,
    tlas_node_width: SIMDLength = 2,
    tlas_leaf_width: SIMDLength = tlas_node_width,
    blas_node_width: SIMDLength = 8,
    blas_leaf_width: SIMDLength = 4,
    blas_build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.HPLOC,
    blas_compressed: Bool = blas_node_width == 8 and blas_leaf_width == 4,
    tlas_build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.LBVH,
](
    ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    world: GpuRtTriangleInstanceScene[
        tlas_node_width,
        blas_node_width,
        tlas_leaf_width,
        blas_leaf_width,
        blas_build_method,
        blas_compressed,
        tlas_build_method,
    ],
    settings: RenderSettings,
) raises:
    """Submit an instanced-triangle render into `target.pixels`."""
    enqueue_gpu_wavefront[
        ALGORITHM,
        _enqueue_instance_bounce[
            ALGORITHM,
            tlas_node_width,
            tlas_leaf_width,
            blas_node_width,
            blas_leaf_width,
            blas_build_method,
            blas_compressed,
            tlas_build_method,
        ],
    ](ctx, target, world, settings)


def render_gpu_triangle_instances[
    ALGORITHM: RENDER = RENDER.PATH,
    tlas_node_width: SIMDLength = 2,
    tlas_leaf_width: SIMDLength = tlas_node_width,
    blas_node_width: SIMDLength = 8,
    blas_leaf_width: SIMDLength = 4,
    blas_build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.HPLOC,
    blas_compressed: Bool = blas_node_width == 8 and blas_leaf_width == 4,
    tlas_build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.LBVH,
](
    settings: RenderSettings,
    camera: Camera,
    world: SceneData,
) raises -> RenderResult:
    """Render an instance-only triangle scene through the shared RT stages."""
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
        var gpu_world = GpuRtTriangleInstanceScene[
            tlas_node_width,
            blas_node_width,
            tlas_leaf_width,
            blas_leaf_width,
            blas_build_method,
            blas_compressed,
            tlas_build_method,
        ](ctx, world)
        var target = GpuRtRenderTarget(ctx, settings, camera)
        init_ns = Int(perf_counter_ns() - init_t0)

        var render_t0 = perf_counter_ns()
        enqueue_render_gpu_triangle_instances[
            ALGORITHM,
            tlas_node_width,
            tlas_leaf_width,
            blas_node_width,
            blas_leaf_width,
            blas_build_method,
            blas_compressed,
            tlas_build_method,
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
