from std.math import ceildiv, max
from std.bit import count_leading_zeros, pop_count
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
    SegmentOffsets,
)
from bajo.bvh.types import GpuBlasSet, Hit, TriangleLeafBlock
from bajo.bvh.constants import (
    EMPTY_LANE,
    TRACE,
    TRI_LEAF_VERTEX_STRIDE,
    TRI_LEAF_PACKED_STRIDE,
    GPU_BOUNDS_BVH_BLOCK_SIZE,
    GPU_STACK_SIZE,
    f32_max,
    WideNode,
)
from bajo.bvh.gpu.wide_layout import (
    GpuCompactWideLayout,
    GpuWideBoundsBvh,
    enqueue_compact_segmented_buffer,
)
from bajo.bvh.gpu.builder import GpuBvhBuildMethod
from bajo.bvh.gpu.compressed_bounds_bvh import (
    CWBVH_NODE_WORDS,
    CWBVH_TRIANGLE_WORDS,
    enqueue_segmented_cwbvh8_representation,
    _intersect_cwbvh8_node_tasks,
)
from bajo.bvh.gpu.builder.binary_layout import _segment_for_item
from bajo.bvh.gpu.builder.segmented_build import (
    GpuSegmentedWideBuildTicket,
    enqueue_segmented_wide_build,
)
from bajo.bvh.gpu.blas_desc import enqueue_segmented_blas_descriptors
from bajo.bvh.gpu.blas_finalize import finalize_ordinary_wide_blas_set
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
    GpuTraversalStats,
    trace_bounds_bvh,
    trace_bounds_bvh_with_stats,
)


def pack_segmented_triangle_leaf_lanes_kernel[
    width: SIMDLength,
](
    vertices: Pointer[Float32, ImmutAnyOrigin],
    primitive_segment_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    leaf_block_segment_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    leaf_block_indices: Pointer[UInt32, ImmutAnyOrigin],
    leaf_block_counts: Pointer[UInt32, ImmutAnyOrigin],
    leaf_vertices: Pointer[Float32, MutAnyOrigin],
    leaf_lane_capacity: Int32,
):
    """Pack every BLAS leaf range while retaining BLAS-local primitive IDs."""
    var lane_idx = global_idx.x
    if lane_idx >= Int(leaf_lane_capacity):
        return

    var physical_block = lane_idx / width
    var segment_idx = _segment_for_item(
        leaf_block_segment_offsets, physical_block
    )
    var segment_block_begin = Int(
        leaf_block_segment_offsets.unsafe_get(segment_idx)
    )
    var local_block = physical_block - segment_block_begin
    if local_block >= Int(leaf_block_counts[unsafe_offset=segment_idx]):
        return

    var prim = leaf_block_indices[unsafe_offset=lane_idx]
    var lane = lane_idx % width
    var out_base = physical_block * TRI_LEAF_PACKED_STRIDE * width
    var leaf_vertices_u32 = leaf_vertices.unsafe_bitcast[UInt32]()

    if prim == EMPTY_LANE:
        leaf_vertices_u32[unsafe_offset=out_base + 3 * width + lane] = prim
        return

    var primitive_begin = primitive_segment_offsets.unsafe_get(segment_idx)
    leaf_vertices_u32[unsafe_offset=out_base + 3 * width + lane] = (
        prim - primitive_begin
    )

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


@fieldwise_init
struct _TriangleHostSegments:
    var segments: SegmentOffsets
    var packed_vertices: List[Float32]


def _flatten_triangle_sets[
    frame: Frame,
](vertex_sets: ImmSpan[List[Point3f32[frame]], _],) -> _TriangleHostSegments:
    """Flatten host triangle sets and retain their primitive segmentation."""
    var primitive_counts = List[Int](capacity=len(vertex_sets))
    var total_vertex_count = 0
    for vertices in vertex_sets:
        debug_assert["safe", _use_compiler_assume=True](
            len(vertices) % 3 == 0,
            "each triangle BLAS must contain complete triangles",
        )
        primitive_counts.append(len(vertices) / 3)
        total_vertex_count += len(vertices)

    var packed_vertices = List[Float32](capacity=total_vertex_count * 3)
    for vertices in vertex_sets:
        for vertex in vertices:
            packed_vertices.append(vertex.x)
            packed_vertices.append(vertex.y)
            packed_vertices.append(vertex.z)
    return _TriangleHostSegments(
        SegmentOffsets.from_counts(primitive_counts^), packed_vertices^
    )


@fieldwise_init
struct _SegmentedTriangleWideBuild[
    frame: Frame,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    build_method: GpuBvhBuildMethod,
]:
    var hierarchy: GpuSegmentedWideBuildTicket[
        Self.node_width,
        Self.leaf_width,
        Int(Self.leaf_width),
        Self.build_method,
        True,
        False,
    ]
    var leaf_vertices: DeviceBuffer[DType.float32]
    var bounds_pack_ns: Int
    var leaf_pack_start_ns: Int

    def into_blas_set(
        deinit self, mut ctx: DeviceContext
    ) raises -> GpuBlasSet[Self.node_width, Self.leaf_width]:
        """Finalize the adapter as a descriptor-backed BLAS set."""
        return finalize_ordinary_wide_blas_set[
            Self.node_width,
            Self.leaf_width,
            Self.build_method,
            TRI_LEAF_PACKED_STRIDE,
        ](ctx, self.hierarchy^, self.leaf_vertices^)

    def into_bvh(
        deinit self,
        mut ctx: DeviceContext,
        mut timings: GpuBuildTimings,
        measure_build: Bool,
    ) raises -> GpuTriangleBvh[Self.frame, Self.node_width, Self.leaf_width]:
        """Finalize the adapter's only segment as a standalone BVH."""
        if measure_build:
            ctx.synchronize()
        else:
            self.hierarchy.wait(ctx)
        ref wide = self.hierarchy.wide
        var layout = GpuCompactWideLayout(
            ctx, wide.node_counts, wide.leaf_block_counts, 1
        )
        var compact_leaves = enqueue_compact_segmented_buffer[
            DType.float32,
            Self.leaf_width * TRI_LEAF_PACKED_STRIDE,
        ](
            ctx,
            self.leaf_vertices,
            wide.leaf_block_segment_offsets,
            layout.leaf_block_segment_offsets,
            layout.leaf_block_segments.item_count(),
            1,
        )
        var tree = self.hierarchy^.take_single_segment_synchronized(
            ctx, timings
        )
        if measure_build:
            timings.bounds_pack_ns = self.bounds_pack_ns
            timings.leaf_pack_ns = Int(
                perf_counter_ns() - self.leaf_pack_start_ns
            )
        return GpuTriangleBvh[Self.frame, Self.node_width, Self.leaf_width](
            tree^, compact_leaves^
        )


def _enqueue_segmented_triangle_wide[
    frame: Frame,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    build_method: GpuBvhBuildMethod,
](
    mut ctx: DeviceContext,
    vertices: DeviceBuffer[DType.float32],
    segments: SegmentOffsets,
    measure_build: Bool = False,
) raises -> _SegmentedTriangleWideBuild[
    frame, node_width, leaf_width, build_method
]:
    """Run the one ordinary-wide triangle adapter for any segment count."""
    var triangle_count = segments.item_count()
    debug_assert["safe", _use_compiler_assume=True](
        triangle_count > 0,
        "triangle wide build requires at least one primitive",
    )
    debug_assert["safe", _use_compiler_assume=True](
        len(vertices) == triangle_count * TRI_LEAF_VERTEX_STRIDE,
        "triangle vertex buffer does not match its segments",
    )

    var leaf_bounds = ctx.enqueue_create_buffer[DType.float32](
        triangle_count * AABB[frame].STRIDE
    )
    var payloads = ctx.enqueue_create_buffer[DType.uint32](triangle_count)
    var bounds_pack_start = Int(0)
    var bounds_pack_ns = Int(0)
    if measure_build:
        ctx.synchronize()
        bounds_pack_start = perf_counter_ns()
    ctx.enqueue_function[compute_triangle_bounds_kernel[frame]](
        _device_span[mut=False](vertices),
        _device_span[mut=True](leaf_bounds),
        _device_span[mut=True](payloads),
        grid_dim=ceildiv(triangle_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
        block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
    )
    if measure_build:
        ctx.synchronize()
        bounds_pack_ns = Int(perf_counter_ns() - bounds_pack_start)

    var hierarchy = enqueue_segmented_wide_build[
        node_width, leaf_width, Int(leaf_width), build_method, True
    ](
        ctx,
        segments,
        leaf_bounds^,
        payloads^,
        measure_build,
    )
    var leaf_lane_capacity = (
        hierarchy.wide.leaf_block_segments.item_count() * leaf_width
    )
    var leaf_vertices = ctx.enqueue_create_buffer[DType.float32](
        leaf_lane_capacity * TRI_LEAF_PACKED_STRIDE
    )
    if measure_build:
        hierarchy.wait(ctx)
    var leaf_pack_start_ns = Int(0)
    if measure_build:
        leaf_pack_start_ns = perf_counter_ns()
    ctx.enqueue_function[pack_segmented_triangle_leaf_lanes_kernel[leaf_width]](
        vertices,
        _device_span[mut=False](hierarchy.binary.segment_offsets),
        _device_span[mut=False](hierarchy.wide.leaf_block_segment_offsets),
        hierarchy.wide.leaf_block_indices,
        hierarchy.wide.leaf_block_counts,
        leaf_vertices,
        Int32(leaf_lane_capacity),
        grid_dim=ceildiv(leaf_lane_capacity, GPU_BOUNDS_BVH_BLOCK_SIZE),
        block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
    )
    return _SegmentedTriangleWideBuild[
        frame, node_width, leaf_width, build_method
    ](
        hierarchy^,
        leaf_vertices^,
        bounds_pack_ns,
        leaf_pack_start_ns,
    )


def _build_segmented_triangle_blas_set[
    frame: Frame,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    build_method: GpuBvhBuildMethod,
](
    mut ctx: DeviceContext,
    vertex_sets: ImmSpan[List[Point3f32[frame]], _],
) raises -> GpuBlasSet[node_width, leaf_width]:
    """Build ordinary wide BLASes as one segmented GPU workload."""
    var inputs = _flatten_triangle_sets(vertex_sets)
    if inputs.segments.item_count() == 0:
        return GpuBlasSet[node_width, leaf_width].empty(ctx, len(vertex_sets))
    var source_vertices = upload_list(ctx, inputs.packed_vertices)
    var adapter = _enqueue_segmented_triangle_wide[
        frame, node_width, leaf_width, build_method
    ](ctx, source_vertices, inputs.segments)
    return adapter^.into_blas_set(ctx)


def _build_segmented_compressed_triangle_blas_set[
    frame: Frame,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    build_method: GpuBvhBuildMethod,
](
    mut ctx: DeviceContext,
    vertex_sets: ImmSpan[List[Point3f32[frame]], _],
) raises -> GpuBlasSet[node_width, leaf_width]:
    """Build and encode every CWBVH8 BLAS as one segmented workload."""
    comptime assert node_width == 8 and leaf_width == 4

    var inputs = _flatten_triangle_sets(vertex_sets)
    if inputs.segments.item_count() == 0:
        return GpuBlasSet[node_width, leaf_width].empty(ctx, len(vertex_sets))
    ref segments = inputs.segments
    var source_vertices = upload_list(ctx, inputs.packed_vertices)
    var triangle_count = segments.item_count()
    var leaf_bounds = ctx.enqueue_create_buffer[DType.float32](
        triangle_count * AABB[frame].STRIDE
    )
    var payloads = ctx.enqueue_create_buffer[DType.uint32](triangle_count)
    ctx.enqueue_function[compute_triangle_bounds_kernel[frame]](
        _device_span[mut=False](source_vertices),
        _device_span[mut=True](leaf_bounds),
        _device_span[mut=True](payloads),
        grid_dim=ceildiv(triangle_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
        block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
    )

    var build = enqueue_segmented_wide_build[
        node_width, leaf_width, 3, build_method, True, True
    ](ctx, segments, leaf_bounds^, payloads^)
    ref binary = build.binary
    ref wide = build.wide

    var nodes = ctx.enqueue_create_buffer[DType.float32](
        wide.node_segments.item_count() * CWBVH_NODE_WORDS
    )
    var triangles = ctx.enqueue_create_buffer[DType.float32](
        triangle_count * CWBVH_TRIANGLE_WORDS
    )
    var triangle_counters = enqueue_segmented_cwbvh8_representation[leaf_width](
        ctx,
        wide.wide_nodes,
        wide.leaf_block_indices,
        wide.node_segment_offsets,
        wide.leaf_block_segment_offsets,
        binary.segment_offsets,
        wide.node_counts,
        source_vertices,
        nodes,
        triangles,
    )

    ctx.synchronize()
    build.finish_synchronized()

    with triangle_counters.map_to_host() as encoded_counts:
        for segment_idx in range(segments.segment_count()):
            if encoded_counts[segment_idx] != segments.count(segment_idx):
                raise "segmented CWBVH8 encoding lost triangle records"

    var layout = GpuCompactWideLayout(
        ctx,
        wide.node_counts,
        wide.leaf_block_counts,
        segments.segment_count(),
    )
    var compact_nodes = enqueue_compact_segmented_buffer[
        DType.float32, CWBVH_NODE_WORDS
    ](
        ctx,
        nodes,
        wide.node_segment_offsets,
        layout.node_segment_offsets,
        layout.node_segments.item_count(),
        segments.segment_count(),
    )
    var descs = enqueue_segmented_blas_descriptors[
        CWBVH_NODE_WORDS, CWBVH_TRIANGLE_WORDS
    ](
        ctx,
        layout.node_segment_offsets,
        binary.segment_offsets,
        binary.segment_offsets,
        wide.node_counts,
        triangle_counters,
        segments.segment_count(),
    )
    ctx.synchronize()

    return GpuBlasSet[node_width, leaf_width](
        descs^,
        compact_nodes^,
        triangles^,
        segments.segment_count(),
    )


def build_triangle_blas_set[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
    build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.HPLOC,
    compressed: Bool = False,
    frame: Frame = Frame.LOCAL,
](
    mut ctx: DeviceContext,
    vertex_sets: ImmSpan[List[Point3f32[frame]], _],
) raises -> GpuBlasSet[node_width, leaf_width]:
    """Select the representation around the shared triangle input adapter."""
    debug_assert["safe", _use_compiler_assume=True](len(vertex_sets) > 0)
    comptime if compressed:
        comptime assert node_width == 8 and leaf_width == 4
        return _build_segmented_compressed_triangle_blas_set[
            frame, node_width, leaf_width, build_method
        ](ctx, vertex_sets)
    else:
        return _build_segmented_triangle_blas_set[
            frame, node_width, leaf_width, build_method
        ](ctx, vertex_sets)


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

    def __init__(
        out self,
        var tree: GpuWideBoundsBvh[
            Self.node_width,
            Self.leaf_width,
            Int(Self.leaf_width),
        ],
        var leaf_vertices: DeviceBuffer[DType.float32],
    ):
        self.tree = tree^
        self.leaf_vertices = leaf_vertices^

    def launch_camera(
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
            trace_triangle_bvh_camera_kernel[Self.node_width, Self.leaf_width]
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


def build_triangle_bvh[
    frame: Frame,
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
    build_method: GpuBvhBuildMethod = GpuBvhBuildMethod.LBVH,
](
    mut ctx: DeviceContext,
    vertices: DeviceBuffer[DType.float32],
) raises -> GpuTriangleBvh[frame, node_width, leaf_width]:
    """Build one triangle segment through the unified segmented driver."""
    var timings = GpuBuildTimings(0, 0, 0, 0, 0, 0, 0)
    return _build_triangle_bvh_segmented[
        frame, node_width, leaf_width, build_method
    ](ctx, vertices, timings, False)


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
    return _build_triangle_bvh_segmented[
        frame, node_width, leaf_width, build_method
    ](ctx, vertices, timings, True)


def _build_triangle_bvh_segmented[
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
    """Build one segment through the same adapter used by BLAS batches."""
    debug_assert["safe", _use_compiler_assume=True](
        len(vertices) % TRI_LEAF_VERTEX_STRIDE == 0,
        "triangle vertex buffer must contain complete triangle records",
    )
    var tri_count = len(vertices) / TRI_LEAF_VERTEX_STRIDE
    debug_assert["safe", _use_compiler_assume=True](
        tri_count > 0, "standalone triangle BVH requires nonempty input"
    )
    var adapter = _enqueue_segmented_triangle_wide[
        frame, node_width, leaf_width, build_method
    ](
        ctx,
        vertices,
        SegmentOffsets.single(tri_count),
        measure_build,
    )
    return adapter^.into_bvh(ctx, timings, measure_build)


def compute_triangle_bounds_kernel[
    frame: Frame,
](
    vertices: ImmSpan[Float32, ImmutAnyOrigin],
    leaf_bounds: MutSpan[Float32, MutAnyOrigin],
    payloads: MutSpan[UInt32, MutAnyOrigin],
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


def trace_triangle_bvh_camera_kernel[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
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

    # extra distance stack benchmarks positively for triangle BVH4;
    # BVH8 retains the lower-memory stack specialization.
    var hit = trace_bounds_bvh[
        Frame.WORLD,
        node_width,
        TRACE.CLOSEST_HIT,
        _intersect_triangle_leaf[
            Frame.WORLD,
            leaf_width,
            TRACE.CLOSEST_HIT,
            leaf_width > node_width or leaf_width == 8,
        ],
        node_width == 4,
        node_width == 2 and leaf_width == 2,
    ](wide_nodes, leaf_vertices, root_idx, ray)
    _store_camera_hit(hit, hits, ray_count_int, ray_idx)


def trace_triangle_bvh_rays_kernel[
    frame: Frame,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    mode: TRACE = TRACE.CLOSEST_HIT,
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
    var hit = trace_bounds_bvh[
        frame,
        node_width,
        mode,
        _intersect_triangle_leaf[
            frame,
            leaf_width,
            mode,
            leaf_width > node_width or leaf_width == 8,
        ],
        node_width == 4,
        mode == TRACE.CLOSEST_HIT and node_width == 2 and leaf_width == 2,
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
    leaf_vertices: ImmPointer[Float32, _],
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

        # Any-hit only needs the candidate mask, so it can always stay in
        # determinant-scaled space and return without paying for a reciprocal.
        # `division_free` remains the layout-tuned closest-hit policy.
        comptime if mode == TRACE.ANY_HIT or division_free:
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


@always_inline
def _intersect_cwbvh_triangle[
    frame: Frame,
    mode: TRACE,
](
    triangles: ImmPointer[Float32, _],
    triangle_idx: UInt32,
    ray: Rayf32[frame],
    mut hit: Hit[frame],
) -> Bool:
    """Intersect one aligned e1/e2/v0 CWBVH triangle record."""
    var base = Int(triangle_idx) * CWBVH_TRIANGLE_WORDS
    var e1 = Vec3f32[frame](
        triangles[unsafe_offset=base + 0],
        triangles[unsafe_offset=base + 1],
        triangles[unsafe_offset=base + 2],
    )
    var e2 = Vec3f32[frame](
        triangles[unsafe_offset=base + 4],
        triangles[unsafe_offset=base + 5],
        triangles[unsafe_offset=base + 6],
    )
    var v0 = Point3f32[frame](
        triangles[unsafe_offset=base + 8],
        triangles[unsafe_offset=base + 9],
        triangles[unsafe_offset=base + 10],
    )
    var scaled_hit = intersect_ray_tri_edges_scaled(
        ray.o, ray.d, v0, e1, e2, hit.t, ray.t_min
    )
    if not scaled_hit.mask:
        return False
    comptime if mode == TRACE.ANY_HIT:
        return True
    else:
        var prim = triangles.unsafe_bitcast[UInt32]()[unsafe_offset=base + 11]
        var inv_det = 1.0 / scaled_hit.abs_det
        hit.t = scaled_hit.t_scaled * inv_det
        hit.u = scaled_hit.u_scaled * inv_det
        hit.v = scaled_hit.v_scaled * inv_det
        hit.prim = prim
        hit.inst = EMPTY_LANE
        hit.normal = normalize(cross(e1, e2)).unsafe_convert[
            new_kind=GeoKind.NORMAL
        ]()
        return True


@always_inline
def trace_cwbvh8_triangles[
    frame: Frame,
    mode: TRACE,
](
    nodes: ImmPointer[Float32, _],
    triangles: ImmPointer[Float32, _],
    root_idx: UInt32,
    ray: Rayf32[frame],
) -> Hit[frame]:
    """Traverse CWBVH8 with compressed node/triangle task masks."""
    var hit = Hit[frame].miss(ray.t_max)
    var stack_base = Array[UInt32, GPU_STACK_SIZE](uninitialized=True)
    var stack_mask = Array[UInt32, GPU_STACK_SIZE](uninitialized=True)
    var stack_ptr = 0

    var sign_code = UInt32(0)
    if ray.d.x < 0.0:
        sign_code |= UInt32(4)
    if ray.d.y < 0.0:
        sign_code |= UInt32(2)
    if ray.d.z < 0.0:
        sign_code |= UInt32(1)
    var octant_inverse = UInt32(7) - sign_code
    var ray_rcp = ray.rcp_direction[1]()

    # The synthetic root task uses the same compact group representation as
    # internal children. Its relative index is always zero.
    var node_group_base = root_idx
    var node_group_mask = UInt32(1) << UInt32(31)
    var triangle_group_base: UInt32
    var triangle_group_mask: UInt32

    while True:
        if node_group_mask > UInt32(0x00FFFFFF):
            var group_imask = node_group_mask
            var child_bit = 31 - Int(count_leading_zeros(node_group_mask))
            node_group_mask &= ~(UInt32(1) << UInt32(child_bit))
            if node_group_mask > UInt32(0x00FFFFFF):
                debug_assert["safe", _use_compiler_assume=True](
                    stack_ptr < GPU_STACK_SIZE,
                    "GPU CWBVH8 traversal stack overflow",
                )
                stack_base[stack_ptr] = node_group_base
                stack_mask[stack_ptr] = node_group_mask
                stack_ptr += 1

            var slot = UInt32(child_bit - 24) ^ octant_inverse
            var slots_before = (UInt32(1) << slot) - UInt32(1)
            var relative = UInt32(pop_count(group_imask & slots_before))
            var node_idx = node_group_base + relative
            var node_t_max = hit.t
            comptime if mode == TRACE.ANY_HIT:
                node_t_max = ray.t_max
            var tasks = _intersect_cwbvh8_node_tasks[frame](
                nodes,
                node_idx,
                ray,
                ray_rcp.x,
                ray_rcp.y,
                ray_rcp.z,
                node_t_max,
                octant_inverse,
            )
            node_group_base = tasks.child_base
            node_group_mask = tasks.node_group_mask
            triangle_group_base = tasks.triangle_base
            triangle_group_mask = tasks.triangle_group_mask
        else:
            triangle_group_base = node_group_base
            triangle_group_mask = node_group_mask
            node_group_mask = UInt32(0)

        while triangle_group_mask != 0:
            var triangle_bit = 31 - Int(
                count_leading_zeros(triangle_group_mask)
            )
            triangle_group_mask &= ~(UInt32(1) << UInt32(triangle_bit))
            var triangle_hit = _intersect_cwbvh_triangle[frame, mode](
                triangles,
                triangle_group_base + UInt32(triangle_bit),
                ray,
                hit,
            )
            comptime if mode == TRACE.ANY_HIT:
                if triangle_hit:
                    return Hit[frame].shadow_hit()

        if node_group_mask <= UInt32(0x00FFFFFF):
            if stack_ptr == 0:
                break
            stack_ptr -= 1
            node_group_base = stack_base[stack_ptr]
            node_group_mask = stack_mask[stack_ptr]

    return hit
