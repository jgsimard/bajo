from std.math import ceildiv, max
from std.time import perf_counter_ns
from max.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.gpu.buffers import _device_span, upload_list
from bajo.bvh.gpu.utils import (
    GpuBuildTimings,
    upload_vertices,
)
from bajo.core import (
    AABB,
    vmin,
    vmax,
    Vec3,
    Point3f32,
    Frame,
    SegmentOffsets,
)
from bajo.bvh.gpu.blas_storage import GpuBlasSet, GpuBvhLayout
from bajo.bvh.constants import (
    EMPTY_LANE,
    TraceMode,
    TRI_LEAF_VERTEX_STRIDE,
    TRI_LEAF_PACKED_STRIDE,
    GPU_BOUNDS_BVH_BLOCK_SIZE,
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
    GpuCwbvh8RepresentationWorkspace,
    enqueue_segmented_cwbvh8_representation,
    pack_segmented_cwbvh_triangles_kernel,
)
from bajo.bvh.gpu.builder.binary_layout import (
    GpuBinaryBoundsBvh,
    GpuBinaryBuildWorkspace,
    _segment_for_item,
)
from bajo.bvh.gpu.builder.hploc_layout import (
    HPLOC_MERGING_THRESHOLD,
    HPLOC_SEARCH_RADIUS,
)
from bajo.bvh.gpu.builder.hploc_multi_wave import GpuHplocBuildState
from bajo.bvh.gpu.builder.lbvh import (
    enqueue_segmented_morton_codes,
    enqueue_segmented_morton_sort,
)
from bajo.bvh.gpu.builder.wide_collapse import (
    CWBVH_COLLAPSE_BLOCK_SIZE,
    GpuWideCollapseWorkspace,
    enqueue_collapse_binary_to_cwbvh8,
)
from bajo.bvh.gpu.builder.segmented_build import (
    GpuSegmentedWideBuildTicket,
    enqueue_segmented_wide_build,
)
from bajo.bvh.gpu.blas_desc import enqueue_segmented_blas_descriptors
from bajo.bvh.gpu.blas_finalize import finalize_ordinary_wide_blas_set
from bajo.bvh.gpu.camera_launch import validate_camera_launch
from bajo.bvh.gpu.camera_trace import enqueue_bvh_camera_trace
from bajo.bvh.gpu.ray_launch import validate_ray_launch
from bajo.bvh.gpu.trace import GpuTraversalStats
from bajo.bvh.gpu.triangle_trace import (
    _trace_triangle_bvh_camera_ray,
    trace_triangle_bvh_camera_instrumented_kernel,
    trace_triangle_bvh_rays_kernel,
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
    comptime for axis in range(3):
        var v0 = vertices[unsafe_offset=in_base + axis]
        leaf_vertices[unsafe_offset=out_base + axis * width + lane] = v0
        leaf_vertices[unsafe_offset=out_base + (4 + axis) * width + lane] = (
            vertices[unsafe_offset=in_base + 3 + axis] - v0
        )
        leaf_vertices[unsafe_offset=out_base + (8 + axis) * width + lane] = (
            vertices[unsafe_offset=in_base + 6 + axis] - v0
        )
    leaf_vertices[unsafe_offset=out_base + 7 * width + lane] = 0.0
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
            packed_vertices.extend([vertex.x, vertex.y, vertex.z])
    return _TriangleHostSegments(
        SegmentOffsets.from_counts(primitive_counts), packed_vertices^
    )


def _flatten_triangle_sets[
    frame: Frame,
](vertices: ImmSpan[Point3f32[frame], _]) -> _TriangleHostSegments:
    """Pack one borrowed mesh directly into the upload representation."""
    debug_assert["safe", _use_compiler_assume=True](
        len(vertices) % 3 == 0,
        "each triangle BLAS must contain complete triangles",
    )
    var packed_vertices = List[Float32](capacity=len(vertices) * 3)
    for vertex in vertices:
        packed_vertices.extend([vertex.x, vertex.y, vertex.z])
    return _TriangleHostSegments(
        SegmentOffsets.single(len(vertices) / 3), packed_vertices^
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
    var leaf_vertices: DeviceBuffer[.float32]
    var bounds_pack_ns: Int
    var leaf_pack_start_ns: Int

    def into_blas_set[
        layout: GpuBvhLayout
    ](deinit self, mut ctx: DeviceContext) raises -> GpuBlasSet[
        .TRIANGLE,
        layout,
        Self.node_width,
        Self.leaf_width,
    ]:
        """Finalize the adapter as a descriptor-backed BLAS set."""
        return finalize_ordinary_wide_blas_set[
            Self.node_width,
            Self.leaf_width,
            Self.build_method,
            .TRIANGLE,
            layout,
            TRI_LEAF_PACKED_STRIDE,
        ](ctx, self.hierarchy^, self.leaf_vertices^)

    def into_bvh(
        deinit self,
        mut ctx: DeviceContext,
        mut timings: GpuBuildTimings,
        measure_build: Bool,
    ) raises -> GpuTriangleBvh[Self.frame, Self.node_width, Self.leaf_width]:
        """Finalize the adapter's only segment as a standalone BVH."""
        var layout = self.hierarchy.finalize_layout(ctx)
        ref wide = self.hierarchy.wide
        var compact_leaves = enqueue_compact_segmented_buffer[
            .float32,
            Self.leaf_width * TRI_LEAF_PACKED_STRIDE,
        ](
            ctx,
            self.leaf_vertices,
            wide.leaf_block_segment_offsets,
            layout.leaf_block_segment_offsets,
            layout.leaf_block_count,
            1,
        )
        var tree = self.hierarchy^.take_single_segment_with_layout(
            ctx, timings, layout
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
    vertices: DeviceBuffer[.float32],
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

    var leaf_bounds = ctx.enqueue_create_buffer[.float32](
        triangle_count * AABB[frame].STRIDE
    )
    var payloads = ctx.enqueue_create_buffer[.uint32](triangle_count)
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
    var leaf_vertices = ctx.enqueue_create_buffer[.float32](
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
    layout: GpuBvhLayout,
](
    mut ctx: DeviceContext,
    var inputs: _TriangleHostSegments,
) raises -> GpuBlasSet[.TRIANGLE, layout, node_width, leaf_width]:
    """Build ordinary wide BLASes as one segmented GPU workload."""
    if inputs.segments.item_count() == 0:
        return GpuBlasSet[.TRIANGLE, layout, node_width, leaf_width].empty(
            ctx, inputs.segments.segment_count()
        )
    var source_vertices = upload_list(ctx, inputs.packed_vertices)
    var adapter = _enqueue_segmented_triangle_wide[
        frame, node_width, leaf_width, build_method
    ](ctx, source_vertices, inputs.segments)
    return adapter^.into_blas_set[layout](ctx)


def _finish_compressed_triangle_blas_set_via_wide[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    build_method: GpuBvhBuildMethod,
    layout: GpuBvhLayout,
    max_leaf_size: Int,
](
    mut ctx: DeviceContext,
    segments: SegmentOffsets,
    var source_vertices: DeviceBuffer[.float32],
    var leaf_bounds: DeviceBuffer[.float32],
    var payloads: DeviceBuffer[.uint32],
) raises -> GpuBlasSet[.TRIANGLE, layout, node_width, leaf_width]:
    """Build CWBVH8 through ordinary wide storage for segmented workloads."""
    var triangle_count = segments.item_count()
    var build = enqueue_segmented_wide_build[
        node_width, leaf_width, max_leaf_size, build_method, True, True
    ](ctx, segments, leaf_bounds^, payloads^)
    ref binary = build.binary
    ref wide = build.wide

    var nodes = ctx.enqueue_create_buffer[.float32](
        wide.node_segments.item_count() * CWBVH_NODE_WORDS
    )
    var triangles = ctx.enqueue_create_buffer[.float32](
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

    var compact_layout = build.finalize_layout(
        ctx, Optional(triangle_counters.copy())
    )
    var compact_nodes = enqueue_compact_segmented_buffer[
        .float32, CWBVH_NODE_WORDS
    ](
        ctx,
        nodes,
        wide.node_segment_offsets,
        compact_layout.node_segment_offsets,
        compact_layout.node_count,
        segments.segment_count(),
    )
    var descs = enqueue_segmented_blas_descriptors[
        CWBVH_NODE_WORDS, CWBVH_TRIANGLE_WORDS
    ](
        ctx,
        compact_layout.node_segment_offsets,
        binary.segment_offsets,
        binary.segment_offsets,
        wide.node_counts,
        triangle_counters,
        segments.segment_count(),
    )
    ctx.synchronize()

    return GpuBlasSet[.TRIANGLE, layout, node_width, leaf_width](
        descs^,
        compact_nodes^,
        triangles^,
        segments.segment_count(),
    )


def _build_segmented_compressed_triangle_blas_set[
    frame: Frame,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    build_method: GpuBvhBuildMethod,
    layout: GpuBvhLayout,
    max_leaf_size: Int,
](
    mut ctx: DeviceContext,
    var inputs: _TriangleHostSegments,
) raises -> GpuBlasSet[.TRIANGLE, layout, node_width, leaf_width]:
    """Build and encode every CWBVH8 BLAS as one segmented workload."""
    comptime assert node_width == 8 and leaf_width == 4
    comptime assert 1 <= max_leaf_size <= 3

    if inputs.segments.item_count() == 0:
        return GpuBlasSet[.TRIANGLE, layout, node_width, leaf_width].empty(
            ctx, inputs.segments.segment_count()
        )
    ref segments = inputs.segments
    var source_vertices = upload_list(ctx, inputs.packed_vertices)
    var triangle_count = segments.item_count()
    var leaf_bounds = ctx.enqueue_create_buffer[.float32](
        triangle_count * AABB[frame].STRIDE
    )
    var payloads = ctx.enqueue_create_buffer[.uint32](triangle_count)
    ctx.enqueue_function[compute_triangle_bounds_kernel[frame]](
        _device_span[mut=False](source_vertices),
        _device_span[mut=True](leaf_bounds),
        _device_span[mut=True](payloads),
        grid_dim=ceildiv(triangle_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
        block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
    )

    # H-PLOC exposes its compact binary topology, so collapse can quantize and
    # emit CWBVH8 nodes directly.  This avoids materializing the ordinary wide
    # nodes and leaf-index blocks only to read them back in a second kernel.
    comptime if build_method == .HPLOC:
        # The segmented collapse has better occupancy for batches of BLASes;
        # reserve the direct, single-segment kernel for the standalone case.
        if segments.segment_count() != 1:
            return _finish_compressed_triangle_blas_set_via_wide[
                node_width, leaf_width, build_method, layout, max_leaf_size
            ](ctx, segments, source_vertices^, leaf_bounds^, payloads^)
        var workspace = GpuBinaryBuildWorkspace(ctx, segments, True)
        workspace.ensure_topology(ctx)
        var binary = GpuBinaryBoundsBvh(ctx, leaf_bounds, payloads, workspace)
        enqueue_segmented_morton_codes(ctx, binary, workspace)
        enqueue_segmented_morton_sort(ctx, binary, workspace)
        var hploc = GpuHplocBuildState[
            HPLOC_SEARCH_RADIUS,
            HPLOC_MERGING_THRESHOLD,
            True,
            True,
            True,
        ].from_binary(ctx, binary, workspace)

        var node_segments = SegmentOffsets.single(max(triangle_count - 1, 1))
        var leaf_block_segments = SegmentOffsets.single(triangle_count)
        var node_segment_offsets = upload_list(ctx, node_segments.offsets)
        var leaf_block_segment_offsets = upload_list(
            ctx, leaf_block_segments.offsets
        )
        var node_counts = ctx.enqueue_create_buffer[.uint32](1)
        var leaf_block_counts = ctx.enqueue_create_buffer[.uint32](1)
        var collapse_workspace = GpuWideCollapseWorkspace(
            ctx, segments, CWBVH_COLLAPSE_BLOCK_SIZE
        )
        var nodes = ctx.enqueue_create_buffer[.float32](
            node_segments.item_count() * CWBVH_NODE_WORDS
        )
        var triangles = ctx.enqueue_create_buffer[.float32](
            triangle_count * CWBVH_TRIANGLE_WORDS
        )
        var representation_workspace = GpuCwbvh8RepresentationWorkspace(
            ctx, triangle_count, segments.segment_count()
        )
        var collapse = enqueue_collapse_binary_to_cwbvh8[
            max_leaf_size, True, True, True
        ](
            ctx,
            binary,
            node_segments,
            node_segment_offsets,
            leaf_block_segments,
            leaf_block_segment_offsets,
            leaf_block_counts,
            node_counts,
            nodes,
            representation_workspace.compact_primitive_ids,
            representation_workspace.triangle_counters,
            hploc.compact_children,
            hploc.scratch_bounds,
            collapse_workspace,
        )
        ctx.enqueue_function[pack_segmented_cwbvh_triangles_kernel[True]](
            source_vertices,
            representation_workspace.compact_primitive_ids,
            _device_span[mut=False](binary.segment_offsets),
            triangles,
            Int32(triangle_count),
            grid_dim=ceildiv(triangle_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
        var triangle_counters = (
            representation_workspace.triangle_counters.copy()
        )

        var compact_layout = GpuCompactWideLayout(
            ctx,
            node_counts,
            leaf_block_counts,
            segments.segment_count(),
            primitive_offsets=Optional(binary.segment_offsets.copy()),
            node_capacity_offsets=Optional(node_segment_offsets.copy()),
            hploc_status=Optional(hploc.status.copy()),
            collapse_status=Optional(collapse.status.copy()),
            encoded_counts=Optional(triangle_counters.copy()),
        )
        var compact_nodes = enqueue_compact_segmented_buffer[
            .float32, CWBVH_NODE_WORDS
        ](
            ctx,
            nodes,
            node_segment_offsets,
            compact_layout.node_segment_offsets,
            compact_layout.node_count,
            segments.segment_count(),
        )
        var descs = enqueue_segmented_blas_descriptors[
            CWBVH_NODE_WORDS, CWBVH_TRIANGLE_WORDS
        ](
            ctx,
            compact_layout.node_segment_offsets,
            binary.segment_offsets,
            binary.segment_offsets,
            node_counts,
            triangle_counters,
            segments.segment_count(),
        )
        ctx.synchronize()
        return GpuBlasSet[.TRIANGLE, layout, node_width, leaf_width](
            descs^,
            compact_nodes^,
            triangles^,
            segments.segment_count(),
        )

    return _finish_compressed_triangle_blas_set_via_wide[
        node_width, leaf_width, build_method, layout, max_leaf_size
    ](ctx, segments, source_vertices^, leaf_bounds^, payloads^)


def _build_gpu_triangle_blas_inputs[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
    build_method: GpuBvhBuildMethod = .HPLOC,
    layout: GpuBvhLayout = GpuBvhLayout.WIDE,
    frame: Frame = .LOCAL,
    max_leaf_size: Int = 3,
](
    mut ctx: DeviceContext,
    var inputs: _TriangleHostSegments,
) raises -> GpuBlasSet[.TRIANGLE, layout, node_width, leaf_width]:
    """Build typed BLAS storage; CWBVH8 supports one to three triangles per leaf.

    ``max_leaf_size`` controls CWBVH8 construction. Its task-mask encoding is
    shared, so selecting smaller leaves does not change the traversal type.
    """
    debug_assert["safe", _use_compiler_assume=True](
        inputs.segments.segment_count() > 0
    )
    comptime match layout:
        case .CWBVH8:
            comptime assert node_width == 8 and leaf_width == 4
            return _build_segmented_compressed_triangle_blas_set[
                frame,
                node_width,
                leaf_width,
                build_method,
                layout,
                max_leaf_size,
            ](ctx, inputs^)
        case .WIDE:
            return _build_segmented_triangle_blas_set[
                frame, node_width, leaf_width, build_method, layout
            ](ctx, inputs^)


def build_gpu_triangle_blas_set[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
    build_method: GpuBvhBuildMethod = .HPLOC,
    layout: GpuBvhLayout = GpuBvhLayout.WIDE,
    frame: Frame = .LOCAL,
    max_leaf_size: Int = 3,
](
    mut ctx: DeviceContext,
    vertex_sets: ImmSpan[List[Point3f32[frame]], _],
) raises -> GpuBlasSet[.TRIANGLE, layout, node_width, leaf_width]:
    """Build borrowed meshes; CWBVH8 allows one to three triangles per leaf."""
    var inputs = _flatten_triangle_sets(vertex_sets)
    return _build_gpu_triangle_blas_inputs[
        node_width, leaf_width, build_method, layout, frame, max_leaf_size
    ](ctx, inputs^)


def build_gpu_triangle_blas_set[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
    build_method: GpuBvhBuildMethod = .HPLOC,
    layout: GpuBvhLayout = GpuBvhLayout.WIDE,
    frame: Frame = .LOCAL,
    max_leaf_size: Int = 3,
](
    mut ctx: DeviceContext,
    vertices: ImmSpan[Point3f32[frame], _],
) raises -> GpuBlasSet[.TRIANGLE, layout, node_width, leaf_width]:
    """Build one borrowed mesh without copying its point list."""
    var inputs = _flatten_triangle_sets(vertices)
    return _build_gpu_triangle_blas_inputs[
        node_width, leaf_width, build_method, layout, frame, max_leaf_size
    ](ctx, inputs^)


@fieldwise_init
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
    var leaf_vertices: DeviceBuffer[.float32]

    def launch_camera(
        self,
        ctx: DeviceContext,
        d_camera_params: DeviceBuffer[.float32],
        d_hits: DeviceBuffer[.float32],
        ray_count: Int,
        cwidth: Int,
        cheight: Int,
    ) raises:
        comptime assert Self.frame == .WORLD
        enqueue_bvh_camera_trace[
            _trace_triangle_bvh_camera_ray[Self.node_width, Self.leaf_width]
        ](
            ctx,
            self.tree.wide_nodes,
            self.leaf_vertices,
            self.tree.root_idx,
            d_camera_params,
            d_hits,
            ray_count,
            cwidth,
            cheight,
        )

    def launch_camera_instrumented(
        self,
        ctx: DeviceContext,
        d_camera_params: DeviceBuffer[.float32],
        d_hits: DeviceBuffer[.float32],
        d_stats: DeviceBuffer[.uint32],
        ray_count: Int,
        cwidth: Int,
        cheight: Int,
    ) raises:
        comptime assert Self.frame == .WORLD
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
            1.0 / Float32(cheight),
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )

    def launch_rays[
        mode: TraceMode = .CLOSEST_HIT,
    ](
        self,
        ctx: DeviceContext,
        d_rays: DeviceBuffer[.float32],
        d_hits: DeviceBuffer[.float32],
        ray_count: Int,
    ) raises:
        """Trace a packed ray buffer without camera-generation assumptions."""
        validate_ray_launch(d_rays, d_hits, ray_count)
        comptime block_size = (
            64 if Self.node_width == 2
            and mode == .CLOSEST_HIT else GPU_BOUNDS_BVH_BLOCK_SIZE
        )
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
            grid_dim=ceildiv(ray_count, block_size),
            block_dim=block_size,
        )


def build_gpu_triangle_bvh[
    frame: Frame,
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
    build_method: GpuBvhBuildMethod = .LBVH,
](
    mut ctx: DeviceContext,
    vertices: DeviceBuffer[.float32],
) raises -> GpuTriangleBvh[frame, node_width, leaf_width]:
    """Build one triangle segment through the unified segmented driver."""
    var timings = GpuBuildTimings(0, 0, 0, 0, 0, 0, 0)
    return _build_gpu_triangle_bvh[frame, node_width, leaf_width, build_method](
        ctx, vertices, timings, False
    )


def build_gpu_triangle_bvh_measured[
    frame: Frame,
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
    build_method: GpuBvhBuildMethod = .LBVH,
](
    mut ctx: DeviceContext,
    vertices: DeviceBuffer[.float32],
    mut timings: GpuBuildTimings,
) raises -> GpuTriangleBvh[frame, node_width, leaf_width]:
    """Build a ready triangle BVH and populate per-stage timings."""
    return _build_gpu_triangle_bvh[frame, node_width, leaf_width, build_method](
        ctx, vertices, timings, True
    )


def _build_gpu_triangle_bvh[
    frame: Frame,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    build_method: GpuBvhBuildMethod,
](
    mut ctx: DeviceContext,
    vertices: DeviceBuffer[.float32],
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
    var v0 = Point3f32[frame](vertices.unsafe_ptr().unsafe_offset(vbase))
    var v1 = Point3f32[frame](vertices.unsafe_ptr().unsafe_offset(vbase + 3))
    var v2 = Point3f32[frame](vertices.unsafe_ptr().unsafe_offset(vbase + 6))

    var bmin = vmin(vmin(v0, v1), v2)
    var bmax = vmax(vmax(v0, v1), v2)

    leaf_bounds.unsafe_get(bbase + 0) = bmin.x
    leaf_bounds.unsafe_get(bbase + 1) = bmin.y
    leaf_bounds.unsafe_get(bbase + 2) = bmin.z
    leaf_bounds.unsafe_get(bbase + 3) = bmax.x
    leaf_bounds.unsafe_get(bbase + 4) = bmax.y
    leaf_bounds.unsafe_get(bbase + 5) = bmax.z

    payloads.unsafe_get(tri_idx) = UInt32(tri_idx)
