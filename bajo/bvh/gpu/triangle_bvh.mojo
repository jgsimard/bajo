from std.math import ceildiv, max
from std.time import perf_counter_ns
from std.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh.gpu.utils import (
    GpuBuildTimings,
    _device_span,
    upload_vertices,
    upload_list,
)
from bajo.core import (
    AABB,
    Vec3f32,
    vmin,
    vmax,
    normalize,
    cross,
    Vec3,
    Point3f32,
    Frame,
    GeoKind,
    Rayf32,
)
from bajo.bvh.types import Hit, BlasSet, TriangleLeafBlock
from bajo.bvh.constants import (
    EMPTY_LANE,
    TRACE,
    TRI_LEAF_VERTEX_STRIDE,
    TRI_LEAF_PACKED_STRIDE,
    GPU_BOUNDS_BVH_BLOCK_SIZE,
    f32_max,
    WideNode,
)
from bajo.bvh.gpu.bounds_bvh import build_bounds_bvh
from bajo.bvh.gpu.wide_layout import GpuWideBoundsBvh
from bajo.bvh.gpu.builder import GpuBvhBuildMethod
from bajo.bvh.gpu.builder.binary_layout import (
    GpuBinaryBoundsBvh,
    GpuBinaryBuildWorkspace,
)
from bajo.bvh.gpu.builder.hploc_binary import (
    enqueue_binary_bvh_with_hploc,
)
from bajo.bvh.gpu.builder.hploc_layout import HPLOC_STATUS_OK
from bajo.bvh.gpu.builder.hploc_multi_wave import GpuHplocBuildState
from bajo.bvh.gpu.builder.lbvh import build_binary_bvh_with_lbvh
from bajo.bvh.gpu.builder.wide_collapse import (
    GpuWideCollapseState,
    GpuWideCollapseWorkspace,
    enqueue_collapse_binary_to_wide,
    enqueue_collapse_binary_to_wide_with_workspace,
)
from bajo.bvh.gpu.camera_launch import (
    validate_camera_launch,
    _camera_ray,
    _store_camera_hit,
)
from bajo.bvh.gpu.ray_launch import (
    validate_ray_launch,
    _load_packed_ray,
    _store_packed_hit,
)

from bajo.core.intersect import (
    intersect_ray_tri_edges,
    intersect_ray_tri_edges_scaled,
)
from bajo.bvh.gpu.trace import (
    GpuTraversalAlgorithm,
    GpuTraversalStats,
    trace_bounds_bvh,
    trace_bounds_bvh_unified_closest,
    trace_bounds_bvh_with_stats,
)


def build_triangle_blas_set[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
](
    mut ctx: DeviceContext,
    vertex_sets: List[List[Point3f32[Frame.LOCAL]]],
) raises -> BlasSet[node_width, leaf_width]:
    debug_assert["safe", _use_compiler_assume=True](len(vertex_sets) > 0)

    var descs = List[UInt32](capacity=len(vertex_sets) * BlasSet.STRIDE)

    var total_wide_nodes = 0
    var total_leaf_vertices = 0

    # First pass: compute final packed offsets without building/downloading.
    for blas_idx in range(len(vertex_sets)):
        var tri_count = len(vertex_sets[blas_idx]) / 3
        debug_assert["safe", _use_compiler_assume=True](tri_count > 0)

        var internal_count = tri_count - 1
        var max_wide_nodes = max(internal_count, 1)
        var max_leaf_blocks = max(tri_count, 1)

        var wide_node_base = UInt32(total_wide_nodes)
        var leaf_f32_base = UInt32(total_leaf_vertices)
        descs.append(wide_node_base)
        descs.append(leaf_f32_base)

        # Filled after the actual GPU BLAS build.
        descs.append(UInt32(0))  # BlasSet.ROOT_IDX

        descs.append(UInt32(max_wide_nodes))
        descs.append(UInt32(max_leaf_blocks))
        descs.append(UInt32(tri_count))

        total_wide_nodes += max_wide_nodes * node_width * WideNode.CHILD_STRIDE
        total_leaf_vertices += (
            max_leaf_blocks * leaf_width * TRI_LEAF_PACKED_STRIDE
        )

    var wide_nodes = ctx.enqueue_create_buffer[DType.float32](total_wide_nodes)
    var leaf_vertices = ctx.enqueue_create_buffer[DType.float32](
        total_leaf_vertices
    )
    # Second pass: build each BLAS, then copy its device buffers into the
    # final packed device buffers. We synchronize inside the loop so the
    # temporary BLAS buffers stay alive until their copy kernels finish.
    for blas_idx in range(len(vertex_sets)):
        var d_vertices = upload_vertices(ctx, vertex_sets[blas_idx])
        var blas = build_triangle_bvh[Frame.LOCAL, node_width, leaf_width](
            ctx, d_vertices
        )

        var desc_base = blas_idx * BlasSet.STRIDE

        descs[desc_base + BlasSet.ROOT_IDX] = blas.tree.root_idx
        descs[desc_base + BlasSet.NODE_COUNT] = UInt32(blas.tree.node_count)
        descs[desc_base + BlasSet.LEAF_BLOCK_COUNT] = UInt32(
            blas.tree.leaf_block_count
        )

        var wide_node_base = Int(descs[desc_base + BlasSet.WIDE_NODE_BASE])
        var leaf_f32_base = Int(descs[desc_base + BlasSet.LEAF_F32_BASE])
        blas.tree.wide_nodes.enqueue_copy_to(
            wide_nodes.unsafe_ptr().unsafe_offset(wide_node_base)
        )
        blas.leaf_vertices.enqueue_copy_to(
            leaf_vertices.unsafe_ptr().unsafe_offset(leaf_f32_base)
        )
        ctx.synchronize()

    return BlasSet[node_width, leaf_width](
        upload_list(ctx, descs),
        wide_nodes,
        leaf_vertices,
        len(vertex_sets),
    )


struct GpuTriangleBvh[
    frame: Frame,
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
]:
    """Ready-to-trace triangle BVH containing no builder configuration."""

    var tree: GpuWideBoundsBvh[
        Self.node_width,
        Self.leaf_width,
        Int(Self.leaf_width),
    ]
    var leaf_vertices: DeviceBuffer[DType.float32]
    var tri_count: Int

    def __init__(
        out self,
        var tree: GpuWideBoundsBvh[
            Self.node_width,
            Self.leaf_width,
            Int(Self.leaf_width),
        ],
        var leaf_vertices: DeviceBuffer[DType.float32],
        tri_count: Int,
    ):
        self.tree = tree^
        self.leaf_vertices = leaf_vertices^
        self.tri_count = tri_count

    def launch_camera[
        algorithm: GpuTraversalAlgorithm = GpuTraversalAlgorithm.STANDARD,
    ](
        self,
        ctx: DeviceContext,
        d_camera_params: DeviceBuffer[DType.float32],
        d_hits: DeviceBuffer[DType.float32],
        ray_count: Int,
        cwidth: Int,
        cheight: Int,
    ) raises:
        comptime assert Self.frame == Frame.WORLD
        validate_camera_launch(
            d_camera_params, d_hits, ray_count, cwidth, cheight
        )

        ctx.enqueue_function[
            trace_triangle_bvh_camera_kernel[
                Self.node_width,
                Self.leaf_width,
                algorithm == GpuTraversalAlgorithm.UNIFIED_TASKS,
            ]
        ](
            self.tree.wide_nodes,
            self.leaf_vertices,
            self.tree.root_idx,
            d_camera_params,
            d_hits,
            Int32(ray_count),
            Int32(cwidth),
            Int32(cheight),
            Float32(1.0) / Float32(cheight),
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )

    def launch_camera_instrumented(
        self,
        ctx: DeviceContext,
        d_camera_params: DeviceBuffer[DType.float32],
        d_hits: DeviceBuffer[DType.float32],
        d_stats: DeviceBuffer[DType.uint32],
        ray_count: Int,
        cwidth: Int,
        cheight: Int,
    ) raises:
        comptime assert Self.frame == Frame.WORLD
        validate_camera_launch(
            d_camera_params, d_hits, ray_count, cwidth, cheight
        )
        debug_assert["safe", _use_compiler_assume=True](
            len(d_stats) >= ray_count * GpuTraversalStats.STRIDE,
            "traversal stats output buffer is too short",
        )
        ctx.enqueue_function[
            trace_triangle_bvh_camera_instrumented_kernel[
                Self.node_width, Self.leaf_width
            ]
        ](
            self.tree.wide_nodes,
            self.leaf_vertices,
            self.tree.root_idx,
            d_camera_params,
            d_hits,
            d_stats,
            Int32(ray_count),
            Int32(cwidth),
            Int32(cheight),
            Float32(1.0) / Float32(cheight),
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )

    def launch_rays[
        mode: TRACE = TRACE.CLOSEST_HIT,
        algorithm: GpuTraversalAlgorithm = GpuTraversalAlgorithm.STANDARD,
    ](
        self,
        ctx: DeviceContext,
        d_rays: DeviceBuffer[DType.float32],
        d_hits: DeviceBuffer[DType.float32],
        ray_count: Int,
    ) raises:
        """Trace a packed ray buffer without camera-generation assumptions."""
        validate_ray_launch(d_rays, d_hits, ray_count)
        ctx.enqueue_function[
            trace_triangle_bvh_rays_kernel[
                Self.frame,
                Self.node_width,
                Self.leaf_width,
                mode,
                algorithm == GpuTraversalAlgorithm.UNIFIED_TASKS,
            ]
        ](
            self.tree.wide_nodes,
            self.leaf_vertices,
            self.tree.root_idx,
            d_rays,
            d_hits,
            Int32(ray_count),
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )


struct GpuTriangleBvhBuildArena:
    """Reusable scratch for serial builds with one triangle count."""

    var triangle_capacity: Int
    var binary: GpuBinaryBuildWorkspace
    var collapse: GpuWideCollapseWorkspace

    def __init__(
        out self, mut ctx: DeviceContext, triangle_capacity: Int
    ) raises:
        debug_assert["safe", _use_compiler_assume=True](
            triangle_capacity > 0,
            "triangle build arena capacity must be positive",
        )
        self.triangle_capacity = triangle_capacity
        self.binary = GpuBinaryBuildWorkspace(ctx, triangle_capacity)
        self.binary.ensure_topology(ctx)
        self.collapse = GpuWideCollapseWorkspace(ctx, triangle_capacity)


@fieldwise_init
struct GpuTriangleBvhBuildTicket[
    frame: Frame,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    build_method: GpuBvhBuildMethod,
]:
    """Owns queued construction inputs and scratch until `wait` completes."""

    var bvh: GpuTriangleBvh[Self.frame, Self.node_width, Self.leaf_width]
    var source_vertices: DeviceBuffer[DType.float32]
    var binary: GpuBinaryBoundsBvh
    var workspace: GpuBinaryBuildWorkspace
    var hploc: Optional[GpuHplocBuildState]
    var collapse: GpuWideCollapseState

    def wait(
        deinit self, ctx: DeviceContext
    ) raises -> GpuTriangleBvh[Self.frame, Self.node_width, Self.leaf_width]:
        """Wait once, validate device status, and release construction state."""
        ctx.synchronize()
        if self.hploc:
            var status = self.hploc.value().result_status()
            if status != UInt32(HPLOC_STATUS_OK):
                raise String(t"H-PLOC build status: {status}")
        self.collapse.finish_synchronized(self.bvh.tree, True)
        return self.bvh^


def enqueue_build_triangle_bvh_with_arena[
    frame: Frame,
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
    build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.LBVH,
](
    mut ctx: DeviceContext,
    vertices: DeviceBuffer[DType.float32],
    arena: GpuTriangleBvhBuildArena,
) raises -> GpuTriangleBvhBuildTicket[
    frame, node_width, leaf_width, build_method
]:
    """Queue a complete triangle build without host synchronization."""
    debug_assert["safe", _use_compiler_assume=True](
        len(vertices) % TRI_LEAF_VERTEX_STRIDE == 0,
        "triangle vertex buffer must contain complete triangle records",
    )
    var tri_count = len(vertices) / TRI_LEAF_VERTEX_STRIDE
    debug_assert["safe", _use_compiler_assume=True](
        tri_count > 0, "passed empty input."
    )
    debug_assert["safe", _use_compiler_assume=True](
        arena.triangle_capacity == tri_count,
        "triangle build arena capacity must match the input",
    )

    var leaf_bounds = ctx.enqueue_create_buffer[DType.float32](
        tri_count * AABB[frame].STRIDE
    )
    var payloads = ctx.enqueue_create_buffer[DType.uint32](tri_count)
    var blocks = ceildiv(tri_count, GPU_BOUNDS_BVH_BLOCK_SIZE)
    ctx.enqueue_function[compute_triangle_bounds_kernel[frame]](
        _device_span[mut=False](vertices),
        _device_span[mut=True](leaf_bounds),
        _device_span[mut=True](payloads),
        grid_dim=blocks,
        block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
    )

    var workspace = GpuBinaryBuildWorkspace(arena.binary)
    var binary = GpuBinaryBoundsBvh(ctx, leaf_bounds^, payloads^, workspace)
    var hploc = Optional[GpuHplocBuildState]()
    comptime if build_method == GpuBvhBuildMethod.LBVH:
        _ = build_binary_bvh_with_lbvh(ctx, binary, workspace)
    elif build_method == GpuBvhBuildMethod.HPLOC:
        hploc = Optional[GpuHplocBuildState](
            enqueue_binary_bvh_with_hploc(ctx, binary, workspace)
        )
    else:
        comptime assert False, "unknown GPU BVH build method"

    var tree = GpuWideBoundsBvh[node_width, leaf_width, Int(leaf_width)](
        ctx, tri_count
    )
    tree.bounds_device = binary.bounds_device.copy()
    var collapse = enqueue_collapse_binary_to_wide_with_workspace[
        node_width,
        leaf_width,
        Int(leaf_width),
        True,
    ](ctx, binary, tree, arena.collapse)

    var leaf_lane_capacity = tree.max_leaf_blocks * leaf_width
    var leaf_vertices = ctx.enqueue_create_buffer[DType.float32](
        leaf_lane_capacity * TRI_LEAF_PACKED_STRIDE
    )
    blocks = ceildiv(leaf_lane_capacity, GPU_BOUNDS_BVH_BLOCK_SIZE)
    ctx.enqueue_function[pack_triangle_leaf_lanes_kernel[leaf_width]](
        vertices,
        tree.leaf_block_indices,
        leaf_vertices,
        tree.leaf_block_count_device,
        Int32(leaf_lane_capacity),
        grid_dim=blocks,
        block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
    )

    return GpuTriangleBvhBuildTicket[
        frame, node_width, leaf_width, build_method
    ](
        GpuTriangleBvh[frame, node_width, leaf_width](
            tree^, leaf_vertices^, tri_count
        ),
        vertices.copy(),
        binary^,
        workspace^,
        hploc^,
        collapse^,
    )


def enqueue_build_triangle_bvh[
    frame: Frame,
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
    build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.LBVH,
](
    mut ctx: DeviceContext,
    vertices: DeviceBuffer[DType.float32],
) raises -> GpuTriangleBvhBuildTicket[
    frame, node_width, leaf_width, build_method
]:
    """Queue a triangle build with a one-shot internal arena."""
    var tri_count = len(vertices) / TRI_LEAF_VERTEX_STRIDE
    var arena = GpuTriangleBvhBuildArena(ctx, tri_count)
    return enqueue_build_triangle_bvh_with_arena[
        frame, node_width, leaf_width, build_method
    ](ctx, vertices, arena)


def build_triangle_bvh[
    frame: Frame,
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
    build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.LBVH,
](
    mut ctx: DeviceContext,
    vertices: DeviceBuffer[DType.float32],
) raises -> GpuTriangleBvh[frame, node_width, leaf_width]:
    """Build and return a ready triangle BVH."""
    var timings = GpuBuildTimings(0, 0, 0, 0, 0, 0, 0)
    return _build_triangle_bvh[frame, node_width, leaf_width, build_method](
        ctx, vertices, timings, False
    )


def build_triangle_bvh_measured[
    frame: Frame,
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
    build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.LBVH,
](
    mut ctx: DeviceContext,
    vertices: DeviceBuffer[DType.float32],
    mut timings: GpuBuildTimings,
) raises -> GpuTriangleBvh[frame, node_width, leaf_width]:
    """Build a ready triangle BVH and populate per-stage timings."""
    return _build_triangle_bvh[frame, node_width, leaf_width, build_method](
        ctx, vertices, timings, True
    )


def _build_triangle_bvh[
    frame: Frame,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    build_method: GpuBvhBuildMethod,
](
    mut ctx: DeviceContext,
    vertices: DeviceBuffer[DType.float32],
    mut timings: GpuBuildTimings,
    measure_build: Bool,
) raises -> GpuTriangleBvh[frame, node_width, leaf_width]:
    """Build and return a ready triangle BVH.

    The final synchronization makes ownership explicit: temporary source
    vertices may be released when this function returns.
    """
    debug_assert["safe", _use_compiler_assume=True](
        len(vertices) % TRI_LEAF_VERTEX_STRIDE == 0,
        "triangle vertex buffer must contain complete triangle records",
    )
    var tri_count = len(vertices) / TRI_LEAF_VERTEX_STRIDE
    var leaf_bounds = ctx.enqueue_create_buffer[DType.float32](
        tri_count * AABB[frame].STRIDE
    )
    var payloads = ctx.enqueue_create_buffer[DType.uint32](tri_count)

    var bounds_pack_start = Int(0)
    if measure_build:
        ctx.synchronize()
        bounds_pack_start = perf_counter_ns()

    var blocks = ceildiv(max(tri_count, 1), GPU_BOUNDS_BVH_BLOCK_SIZE)
    ctx.enqueue_function[compute_triangle_bounds_kernel[frame]](
        _device_span[mut=False](vertices),
        _device_span[mut=True](leaf_bounds),
        _device_span[mut=True](payloads),
        grid_dim=blocks,
        block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
    )
    var bounds_pack_ns = Int(0)
    if measure_build:
        ctx.synchronize()
        bounds_pack_ns = Int(perf_counter_ns() - bounds_pack_start)

    var tree = GpuWideBoundsBvh[node_width, leaf_width, Int(leaf_width)](
        ctx, tri_count
    )
    timings = build_bounds_bvh[
        node_width,
        leaf_width,
        Int(leaf_width),
        build_method,
        True,
    ](
        ctx,
        tree,
        leaf_bounds,
        payloads,
        measure_build=measure_build,
    )
    timings.bounds_pack_ns = bounds_pack_ns

    var leaf_block_capacity = max(tree.leaf_block_count, 1)
    var leaf_vertices = ctx.enqueue_create_buffer[DType.float32](
        leaf_block_capacity * leaf_width * TRI_LEAF_PACKED_STRIDE
    )
    var leaf_pack_start = Int(0)
    if measure_build:
        leaf_pack_start = perf_counter_ns()
    var leaf_lane_count = max(tree.leaf_block_count * leaf_width, 1)
    blocks = ceildiv(leaf_lane_count, GPU_BOUNDS_BVH_BLOCK_SIZE)
    ctx.enqueue_function[pack_triangle_leaf_lanes_kernel[leaf_width]](
        vertices,
        tree.leaf_block_indices,
        leaf_vertices,
        tree.leaf_block_count_device,
        Int32(leaf_lane_count),
        grid_dim=blocks,
        block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
    )
    ctx.synchronize()
    if measure_build:
        timings.leaf_pack_ns = Int(perf_counter_ns() - leaf_pack_start)

    return GpuTriangleBvh[frame, node_width, leaf_width](
        tree^, leaf_vertices^, tri_count
    )


def compute_triangle_bounds_kernel[
    frame: Frame,
](
    vertices: Span[mut=False, Float32, ImmutAnyOrigin],
    leaf_bounds: Span[mut=True, Float32, MutAnyOrigin],
    payloads: Span[mut=True, UInt32, MutAnyOrigin],
):
    var tri_count = len(payloads)
    var tri_idx = global_idx.x
    if tri_idx >= tri_count:
        return

    var vbase = tri_idx * TRI_LEAF_VERTEX_STRIDE
    var bbase = tri_idx * AABB[frame].STRIDE
    debug_assert["safe", _use_compiler_assume=True](
        vbase >= 0
        and vbase <= len(vertices) - TRI_LEAF_VERTEX_STRIDE
        and bbase <= len(leaf_bounds) - AABB[frame].STRIDE
        and tri_idx < len(payloads),
        "triangle bounds record is outside a device span",
    )
    var v0 = Point3f32[frame](
        vertices.unsafe_get(vbase + 0),
        vertices.unsafe_get(vbase + 1),
        vertices.unsafe_get(vbase + 2),
    )
    var v1 = Point3f32[frame](
        vertices.unsafe_get(vbase + 3),
        vertices.unsafe_get(vbase + 4),
        vertices.unsafe_get(vbase + 5),
    )
    var v2 = Point3f32[frame](
        vertices.unsafe_get(vbase + 6),
        vertices.unsafe_get(vbase + 7),
        vertices.unsafe_get(vbase + 8),
    )

    var bmin = vmin(vmin(v0, v1), v2)
    var bmax = vmax(vmax(v0, v1), v2)

    leaf_bounds.unsafe_get(bbase + 0) = bmin.x
    leaf_bounds.unsafe_get(bbase + 1) = bmin.y
    leaf_bounds.unsafe_get(bbase + 2) = bmin.z
    leaf_bounds.unsafe_get(bbase + 3) = bmax.x
    leaf_bounds.unsafe_get(bbase + 4) = bmax.y
    leaf_bounds.unsafe_get(bbase + 5) = bmax.z

    payloads.unsafe_get(tri_idx) = UInt32(tri_idx)


def pack_triangle_leaf_lanes_kernel[
    width: SIMDLength,
](
    vertices: Pointer[Float32, ImmutAnyOrigin],
    leaf_block_indices: Pointer[UInt32, ImmutAnyOrigin],
    leaf_vertices: Pointer[Float32, MutAnyOrigin],
    leaf_block_count: Pointer[UInt32, ImmutAnyOrigin],
    leaf_lane_capacity: Int32,
):
    var leaf_lane_count_int = Int(leaf_block_count[unsafe_offset=0]) * width
    var leaf_lane_capacity_int = Int(leaf_lane_capacity)
    var lane_idx = global_idx.x
    if lane_idx >= leaf_lane_count_int or lane_idx >= leaf_lane_capacity_int:
        return

    var prim = leaf_block_indices[unsafe_offset=lane_idx]

    # AoSoA : [block][field][lane]
    # Packed fields:
    #   0..2   = v0.xyz
    #   3      = prim id bits
    #   4..6   = e1.xyz (v1 - v0)
    #   7      = pad
    #   8..10  = e2.xyz (v2 - v0)
    #   11     = pad
    var lane = lane_idx % width
    var leaf_block_idx = lane_idx / width
    var out_base = leaf_block_idx * TRI_LEAF_PACKED_STRIDE * width
    var leaf_vertices_u32 = leaf_vertices.unsafe_bitcast[UInt32]()

    leaf_vertices_u32[unsafe_offset=out_base + 3 * width + lane] = prim

    # traversal checks packed prim != EMPTY_LANE
    if prim == EMPTY_LANE:
        return

    var in_base = Int(prim) * TRI_LEAF_VERTEX_STRIDE
    var v0x = vertices[unsafe_offset=in_base + 0]
    var v0y = vertices[unsafe_offset=in_base + 1]
    var v0z = vertices[unsafe_offset=in_base + 2]

    leaf_vertices[unsafe_offset=out_base + 0 * width + lane] = v0x
    leaf_vertices[unsafe_offset=out_base + 1 * width + lane] = v0y
    leaf_vertices[unsafe_offset=out_base + 2 * width + lane] = v0z
    leaf_vertices[unsafe_offset=out_base + 4 * width + lane] = (
        vertices[unsafe_offset=in_base + 3] - v0x
    )
    leaf_vertices[unsafe_offset=out_base + 5 * width + lane] = (
        vertices[unsafe_offset=in_base + 4] - v0y
    )
    leaf_vertices[unsafe_offset=out_base + 6 * width + lane] = (
        vertices[unsafe_offset=in_base + 5] - v0z
    )
    leaf_vertices[unsafe_offset=out_base + 7 * width + lane] = 0.0
    leaf_vertices[unsafe_offset=out_base + 8 * width + lane] = (
        vertices[unsafe_offset=in_base + 6] - v0x
    )
    leaf_vertices[unsafe_offset=out_base + 9 * width + lane] = (
        vertices[unsafe_offset=in_base + 7] - v0y
    )
    leaf_vertices[unsafe_offset=out_base + 10 * width + lane] = (
        vertices[unsafe_offset=in_base + 8] - v0z
    )
    leaf_vertices[unsafe_offset=out_base + 11 * width + lane] = 0.0


def trace_triangle_bvh_camera_kernel[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    unified_tasks: Bool = False,
](
    wide_nodes: Pointer[Float32, ImmutAnyOrigin],
    leaf_vertices: Pointer[Float32, ImmutAnyOrigin],
    root_idx: UInt32,
    camera_params: Pointer[Float32, ImmutAnyOrigin],
    hits: Pointer[Float32, MutAnyOrigin],
    ray_count: Int32,
    width_px: Int32,
    height_px: Int32,
    inv_height: Float32,
):
    var ray_count_int = Int(ray_count)
    var width_px_int = Int(width_px)
    var height_px_int = Int(height_px)
    var ray_idx = global_idx.x
    if ray_idx >= ray_count_int:
        return

    var ray = _camera_ray(
        camera_params,
        ray_count_int,
        ray_idx,
        width_px_int,
        height_px_int,
        inv_height,
    )

    var hit = Hit[Frame.WORLD].miss(ray.t_max)
    comptime if (unified_tasks and node_width == 2 and leaf_width == 4):
        hit = trace_bounds_bvh_unified_closest[
            Frame.WORLD,
            node_width,
            _intersect_triangle_leaf[
                Frame.WORLD,
                leaf_width,
                TRACE.CLOSEST_HIT,
                leaf_width > node_width or leaf_width == 8,
            ],
        ](wide_nodes, leaf_vertices, root_idx, ray)
    else:
        # extra distance stack benchmarks positively for triangle BVH4;
        # BVH8 retains the lower-memory stack specialization.
        hit = trace_bounds_bvh[
            Frame.WORLD,
            node_width,
            TRACE.CLOSEST_HIT,
            _intersect_triangle_leaf[
                Frame.WORLD,
                leaf_width,
                TRACE.CLOSEST_HIT,
                leaf_width > node_width or leaf_width == 8,
            ],
            True,
            node_width == 4,
        ](wide_nodes, leaf_vertices, root_idx, ray)
    _store_camera_hit(hit, hits, ray_count_int, ray_idx)


def trace_triangle_bvh_rays_kernel[
    frame: Frame,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    mode: TRACE = TRACE.CLOSEST_HIT,
    unified_tasks: Bool = False,
](
    wide_nodes: Pointer[Float32, ImmutAnyOrigin],
    leaf_vertices: Pointer[Float32, ImmutAnyOrigin],
    root_idx: UInt32,
    rays: Pointer[Float32, ImmutAnyOrigin],
    hits: Pointer[Float32, MutAnyOrigin],
    ray_count: Int32,
):
    var ray_count_int = Int(ray_count)
    var ray_idx = global_idx.x
    if ray_idx >= ray_count_int:
        return

    var ray = _load_packed_ray[frame](rays, ray_count_int, ray_idx)
    var hit = Hit[frame].miss(ray.t_max)
    comptime if (
        mode == TRACE.CLOSEST_HIT
        and unified_tasks
        and node_width == 2
        and leaf_width == 4
    ):
        hit = trace_bounds_bvh_unified_closest[
            frame,
            node_width,
            _intersect_triangle_leaf[
                frame,
                leaf_width,
                TRACE.CLOSEST_HIT,
                leaf_width > node_width or leaf_width == 8,
            ],
        ](wide_nodes, leaf_vertices, root_idx, ray)
    else:
        hit = trace_bounds_bvh[
            frame,
            node_width,
            mode,
            _intersect_triangle_leaf[
                frame,
                leaf_width,
                mode,
                leaf_width > node_width or leaf_width == 8,
            ],
            True,
            node_width == 4,
        ](wide_nodes, leaf_vertices, root_idx, ray)
    _store_packed_hit[frame](hit, hits, ray_count_int, ray_idx)


def trace_triangle_bvh_camera_instrumented_kernel[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
](
    wide_nodes: Pointer[Float32, ImmutAnyOrigin],
    leaf_vertices: Pointer[Float32, ImmutAnyOrigin],
    root_idx: UInt32,
    camera_params: Pointer[Float32, ImmutAnyOrigin],
    hits: Pointer[Float32, MutAnyOrigin],
    stats: Pointer[UInt32, MutAnyOrigin],
    ray_count: Int32,
    width_px: Int32,
    height_px: Int32,
    inv_height: Float32,
):
    var ray_count_int = Int(ray_count)
    var width_px_int = Int(width_px)
    var height_px_int = Int(height_px)
    var ray_idx = global_idx.x
    if ray_idx >= ray_count_int:
        return

    var ray = _camera_ray(
        camera_params,
        ray_count_int,
        ray_idx,
        width_px_int,
        height_px_int,
        inv_height,
    )

    var result = trace_bounds_bvh_with_stats[
        Frame.WORLD,
        node_width,
        TRACE.CLOSEST_HIT,
        _intersect_triangle_leaf[
            Frame.WORLD,
            leaf_width,
            TRACE.CLOSEST_HIT,
            leaf_width > node_width or leaf_width == 8,
        ],
        True,
        node_width == 4,
    ](
        wide_nodes,
        leaf_vertices,
        root_idx,
        ray,
    )
    _store_camera_hit(result.hit, hits, ray_count_int, ray_idx)
    result.stats.store(stats, ray_idx)


# AoSoA :[block][field][lane]
def _intersect_triangle_leaf[
    frame: Frame,
    width: SIMDLength,
    mode: TRACE,
    division_free: Bool = False,
](
    leaf_vertices: Pointer[mut=False, Float32, _],
    leaf_block_idx: UInt32,
    item_count: UInt32,
    ray: Rayf32[frame],
    mut hit: Hit[frame],
) -> Bool:
    var any_hit = False
    var block_base = Int(leaf_block_idx) * TRI_LEAF_PACKED_STRIDE * width
    var leaf_vertices_u32 = leaf_vertices.unsafe_bitcast[UInt32]()

    comptime for lane in range(width):
        var prim = leaf_vertices_u32[
            unsafe_offset=block_base + 3 * width + lane
        ]
        if prim == EMPTY_LANE:
            continue
        var v0 = Point3f32[frame](
            leaf_vertices[unsafe_offset=block_base + 0 * width + lane],
            leaf_vertices[unsafe_offset=block_base + 1 * width + lane],
            leaf_vertices[unsafe_offset=block_base + 2 * width + lane],
        )
        var e1 = Vec3f32[frame](
            leaf_vertices[unsafe_offset=block_base + 4 * width + lane],
            leaf_vertices[unsafe_offset=block_base + 5 * width + lane],
            leaf_vertices[unsafe_offset=block_base + 6 * width + lane],
        )
        var e2 = Vec3f32[frame](
            leaf_vertices[unsafe_offset=block_base + 8 * width + lane],
            leaf_vertices[unsafe_offset=block_base + 9 * width + lane],
            leaf_vertices[unsafe_offset=block_base + 10 * width + lane],
        )

        comptime if division_free:
            # Reject misses and farther candidates in determinant-scaled space.
            # Only a surviving closest-hit candidate pays for a reciprocal.
            var scaled_hit = intersect_ray_tri_edges_scaled(
                ray.o,
                ray.d,
                v0,
                e1,
                e2,
                hit.t,
                ray.t_min,
            )

            if scaled_hit.mask:
                comptime if mode == TRACE.ANY_HIT:
                    return True
                else:
                    var inv_det = 1.0 / scaled_hit.abs_det
                    hit.t = scaled_hit.t_scaled * inv_det
                    hit.u = scaled_hit.u_scaled * inv_det
                    hit.v = scaled_hit.v_scaled * inv_det
                    hit.prim = prim
                    hit.inst = EMPTY_LANE
                    hit.normal = normalize(cross(e1, e2)).unsafe_convert[
                        new_kind=GeoKind.NORMAL
                    ]()
                    any_hit = True
        else:
            var tri_hit = intersect_ray_tri_edges(
                ray.o,
                ray.d,
                v0,
                e1,
                e2,
                hit.t,
                ray.t_min,
            )

            if tri_hit.mask:
                comptime if mode == TRACE.ANY_HIT:
                    return True
                else:
                    hit.t = tri_hit.t
                    hit.u = tri_hit.u
                    hit.v = tri_hit.v
                    hit.prim = prim
                    hit.inst = EMPTY_LANE
                    hit.normal = normalize(cross(e1, e2)).unsafe_convert[
                        new_kind=GeoKind.NORMAL
                    ]()
                    any_hit = True
    return any_hit
