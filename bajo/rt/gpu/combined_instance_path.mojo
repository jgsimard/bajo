"""GPU RT specialization combining static geometry with triangle instances."""

from std.math import ceildiv
from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh.camera import Camera
from bajo.bvh.gpu.tlas import GpuTriangleTlas, build_triangle_tlas
from bajo.bvh.gpu.builder import GpuBvhBuildMethod
from bajo.bvh.gpu.triangle_bvh import (
    build_triangle_blas_set,
)
from bajo.bvh.gpu.utils import upload_list
from bajo.bvh.types import BlasSet
from bajo.core import Frame, Point3f32
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
    GpuRtLights,
    GpuRtMaterials,
    _enqueue_material_shading,
)


struct GpuRtCombinedInstanceWorld[
    HAS_SPHERES: Bool,
    HAS_TRIANGLES: Bool,
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
    tlas_node_width: SIMDLength = 2,
    tlas_leaf_width: SIMDLength = 2,
    blas_node_width: SIMDLength = 8,
    blas_leaf_width: SIMDLength = 4,
    blas_build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.HPLOC,
    blas_compressed: Bool = blas_node_width == 8 and blas_leaf_width == 4,
    triangle_node_width: SIMDLength = 8,
    triangle_leaf_width: SIMDLength = 4,
    triangle_build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.HPLOC,
    triangle_compressed: Bool = triangle_node_width == 8
    and triangle_leaf_width == 4,
    tlas_build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.LBVH,
]:
    """Static BVHs plus a triangle TLAS, specialized by geometry presence."""

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
    var blases: BlasSet[Self.blas_node_width, Self.blas_leaf_width]
    var tlas: GpuTriangleTlas[
        Self.tlas_node_width,
        Self.blas_node_width,
        Self.tlas_leaf_width,
        Self.blas_leaf_width,
        Self.blas_compressed,
    ]
    var triangle_surfaces: DeviceBuffer[DType.uint32]
    var instance_surfaces: DeviceBuffer[DType.uint32]
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
            len(world.triangle_instances) > 0,
            "combined GPU RT requires triangle instances",
        )
        debug_assert["safe", _use_compiler_assume=True](
            (len(world.spheres) > 0) == Self.HAS_SPHERES
            and (len(world.triangle_vertices) > 0) == Self.HAS_TRIANGLES,
            "combined GPU RT geometry flags do not match the world",
        )

        var triangle_vertices = List[Point3f32[Frame.WORLD]]()
        comptime if Self.HAS_TRIANGLES:
            triangle_vertices = world.triangle_vertices.copy()
        else:
            triangle_vertices.append(Point3f32[Frame.WORLD](0.0, 0.0, 0.0))
            triangle_vertices.append(Point3f32[Frame.WORLD](1.0, 0.0, 0.0))
            triangle_vertices.append(Point3f32[Frame.WORLD](0.0, 1.0, 0.0))

        self.sphere_geometry = GpuRtSphereGeometry[
            Frame.WORLD, Self.node_width, Self.leaf_width
        ].__init__[Self.HAS_SPHERES](ctx, world.spheres, world.sphere_surfaces)
        self.triangle_geometry = GpuRtTriangleGeometry[
            Frame.WORLD,
            Self.triangle_node_width,
            Self.triangle_leaf_width,
            Self.triangle_build_method,
            Self.triangle_compressed,
        ](ctx, triangle_vertices^)
        self.blases = build_triangle_blas_set[
            Self.blas_node_width,
            Self.blas_leaf_width,
            Self.blas_build_method,
            Self.blas_compressed,
        ](ctx, world.triangle_meshes)
        self.tlas = build_triangle_tlas[
            Self.tlas_node_width,
            Self.blas_node_width,
            Self.tlas_leaf_width,
            Self.blas_leaf_width,
            Self.tlas_build_method,
            Self.blas_compressed,
        ](ctx, world.triangle_instances)
        comptime if Self.HAS_TRIANGLES:
            self.triangle_surfaces = upload_surface_ids(
                ctx, world.triangle_surfaces
            )
        else:
            self.triangle_surfaces = upload_list(ctx, [UInt32(0)])
        self.instance_surfaces = upload_surface_ids(
            ctx, world.triangle_instance_surfaces
        )
        self.materials = GpuRtMaterials(ctx, world)
        self.lights = GpuRtLights(ctx, world)


def _enqueue_combined_instance_bounce[
    ALGORITHM: RENDER,
    HAS_SPHERES: Bool,
    HAS_TRIANGLES: Bool,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    tlas_node_width: SIMDLength,
    tlas_leaf_width: SIMDLength,
    blas_node_width: SIMDLength,
    blas_leaf_width: SIMDLength,
    blas_build_method: GpuBvhBuildMethod,
    blas_compressed: Bool,
    triangle_node_width: SIMDLength,
    triangle_leaf_width: SIMDLength,
    triangle_build_method: GpuBvhBuildMethod,
    triangle_compressed: Bool,
    tlas_build_method: GpuBvhBuildMethod,
](
    ctx: DeviceContext,
    arena: GpuWavefrontArena,
    world: GpuRtCombinedInstanceWorld[
        HAS_SPHERES,
        HAS_TRIANGLES,
        node_width,
        leaf_width,
        tlas_node_width,
        tlas_leaf_width,
        blas_node_width,
        blas_leaf_width,
        blas_build_method,
        blas_compressed,
        triangle_node_width,
        triangle_leaf_width,
        triangle_build_method,
        triangle_compressed,
        tlas_build_method,
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
        _immut(world.tlas.core.tree.wide_nodes),
        _immut(world.tlas.core.tree.leaf_block_indices),
        _immut(world.tlas.core.inst_inv_transform),
        _immut(world.tlas.core.inst_blas_indices),
        _immut(world.blases.descs),
        _immut(world.blases.wide_nodes),
        _immut(world.blases.leaves),
        world.tlas.core.tree.root_idx,
        Int32(world.tlas.core.inst_count),
        _immut(world.instance_surfaces),
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
            HAS_SPHERES,
            HAS_TRIANGLES,
            True,
            node_width,
            leaf_width,
            triangle_node_width,
            triangle_leaf_width,
            tlas_node_width,
            tlas_leaf_width,
            blas_node_width,
            blas_leaf_width,
            triangle_compressed,
            blas_compressed,
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
        HAS_SPHERES,
        HAS_TRIANGLES,
        True,
        node_width,
        leaf_width,
        triangle_node_width,
        triangle_leaf_width,
        tlas_node_width,
        tlas_leaf_width,
        blas_node_width,
        blas_leaf_width,
        GPU_RT_MAX_BLOCKS,
        triangle_compressed,
        blas_compressed,
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


def enqueue_render_gpu_combined_instances[
    ALGORITHM: RENDER,
    HAS_SPHERES: Bool,
    HAS_TRIANGLES: Bool,
    MAX_DEPTH: Int = 8,
    node_width: SIMDLength = 4,
    leaf_width: SIMDLength = node_width,
    tlas_node_width: SIMDLength = 2,
    tlas_leaf_width: SIMDLength = 2,
    blas_node_width: SIMDLength = 8,
    blas_leaf_width: SIMDLength = 4,
    blas_build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.HPLOC,
    blas_compressed: Bool = blas_node_width == 8 and blas_leaf_width == 4,
    triangle_node_width: SIMDLength = 8,
    triangle_leaf_width: SIMDLength = 4,
    triangle_build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.HPLOC,
    triangle_compressed: Bool = triangle_node_width == 8
    and triangle_leaf_width == 4,
    tlas_build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.LBVH,
](
    ctx: DeviceContext,
    mut target: GpuRtRenderTarget,
    world: GpuRtCombinedInstanceWorld[
        HAS_SPHERES,
        HAS_TRIANGLES,
        node_width,
        leaf_width,
        tlas_node_width,
        tlas_leaf_width,
        blas_node_width,
        blas_leaf_width,
        blas_build_method,
        blas_compressed,
        triangle_node_width,
        triangle_leaf_width,
        triangle_build_method,
        triangle_compressed,
        tlas_build_method,
    ],
    settings: RenderSettings,
) raises:
    """Submit a combined static/instanced render into `target.pixels`."""
    enqueue_gpu_wavefront[
        ALGORITHM,
        MAX_DEPTH,
        _enqueue_combined_instance_bounce[
            ALGORITHM,
            HAS_SPHERES,
            HAS_TRIANGLES,
            node_width,
            leaf_width,
            tlas_node_width,
            tlas_leaf_width,
            blas_node_width,
            blas_leaf_width,
            blas_build_method,
            blas_compressed,
            triangle_node_width,
            triangle_leaf_width,
            triangle_build_method,
            triangle_compressed,
            tlas_build_method,
        ],
    ](ctx, target, world, settings)


def render_gpu_combined_instances[
    ALGORITHM: RENDER,
    HAS_SPHERES: Bool,
    HAS_TRIANGLES: Bool,
    MAX_DEPTH: Int = 8,
    node_width: SIMDLength = 4,
    leaf_width: SIMDLength = node_width,
    world_bvh_width: SIMDLength = 16,
    instance_bvh_width: SIMDLength = 16,
    tlas_node_width: SIMDLength = 2,
    tlas_leaf_width: SIMDLength = 2,
    blas_node_width: SIMDLength = 8,
    blas_leaf_width: SIMDLength = 4,
    blas_build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.HPLOC,
    blas_compressed: Bool = blas_node_width == 8 and blas_leaf_width == 4,
    triangle_node_width: SIMDLength = 8,
    triangle_leaf_width: SIMDLength = 4,
    triangle_build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.HPLOC,
    triangle_compressed: Bool = triangle_node_width == 8
    and triangle_leaf_width == 4,
    tlas_build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.LBVH,
](
    settings: RenderSettings,
    camera: Camera,
    world: World[world_bvh_width, instance_bvh_width],
) raises -> RenderResult:
    """Render static geometry and triangle instances in one closest-hit pass."""
    comptime assert HAS_SPHERES or HAS_TRIANGLES
    var total_t0 = perf_counter_ns()
    var pixel_count = settings.image_width * settings.image_height
    var sample_count = pixel_count * settings.samples_per_pixel
    var pixels: List[Color]
    var init_ns: Int
    var render_ns: Int
    with DeviceContext() as ctx:
        var init_t0 = perf_counter_ns()
        var gpu_world = GpuRtCombinedInstanceWorld[
            HAS_SPHERES,
            HAS_TRIANGLES,
            node_width,
            leaf_width,
            tlas_node_width,
            tlas_leaf_width,
            blas_node_width,
            blas_leaf_width,
            blas_build_method,
            blas_compressed,
            triangle_node_width,
            triangle_leaf_width,
            triangle_build_method,
            triangle_compressed,
            tlas_build_method,
        ](ctx, world)
        var target = GpuRtRenderTarget(ctx, settings, camera)
        init_ns = Int(perf_counter_ns() - init_t0)
        var render_t0 = perf_counter_ns()
        enqueue_render_gpu_combined_instances[
            ALGORITHM,
            HAS_SPHERES,
            HAS_TRIANGLES,
            MAX_DEPTH,
            node_width,
            leaf_width,
            tlas_node_width,
            tlas_leaf_width,
            blas_node_width,
            blas_leaf_width,
            blas_build_method,
            blas_compressed,
            triangle_node_width,
            triangle_leaf_width,
            triangle_build_method,
            triangle_compressed,
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
            MAX_DEPTH,
        ),
    )
