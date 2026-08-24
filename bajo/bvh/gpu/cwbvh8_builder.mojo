"""Reusable fixed-capacity CWBVH8 construction."""

from std.math import ceildiv
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh.constants import GPU_BOUNDS_BVH_BLOCK_SIZE
from bajo.bvh.gpu.builder.binary_layout import (
    GpuBinaryBoundsBvh,
    GpuBinaryBuildWorkspace,
)
from bajo.bvh.gpu.builder.hploc_layout import (
    HPLOC_MERGING_THRESHOLD,
    HPLOC_STATUS_OK,
)
from bajo.bvh.gpu.builder.hploc_multi_wave import GpuHplocBuildState
from bajo.bvh.gpu.builder.lbvh import (
    enqueue_segmented_morton_codes,
    enqueue_segmented_morton_sort,
)
from bajo.bvh.gpu.builder.wide_collapse import (
    CWBVH_COLLAPSE_BLOCK_SIZE,
    GpuWideCollapseState,
    GpuWideCollapseWorkspace,
    enqueue_collapse_binary_to_cwbvh8,
    _enqueue_collapse_binary_to_packed,
    _hploc_leaf_block_data,
)
from bajo.bvh.gpu.compressed_bounds_bvh import (
    CWBVH_LEAF_STORAGE_WIDTH,
    CWBVH_NODE_WORDS,
    CWBVH_TRIANGLE_WORDS,
    GpuCwbvh8RepresentationWorkspace,
    enqueue_segmented_cwbvh8_representation_with_workspace,
    pack_segmented_cwbvh_triangles_kernel,
)
from bajo.bvh.gpu.utils import _device_span
from bajo.bvh.gpu.wide_layout import GpuWideBoundsBvhBatch
from bajo.core import SegmentOffsets


comptime CWBVH_HPLOC_SEARCH_RADIUS = 4


struct GpuCwbvh8BuildArena[
    max_leaf_size: Int = 3,
    direct_conversion: Bool = True,
    indexed_triangle_layout: Bool = False,
    spatial_slots: Bool = True,
    search_radius: Int = CWBVH_HPLOC_SEARCH_RADIUS,
    merging_threshold: Int = HPLOC_MERGING_THRESHOLD,
]:
    """All persistent storage and cached offsets for one H-PLOC CWBVH8.

    The arena is intentionally fixed-capacity. Geometry bounds and payloads are
    retained by the binary layout; each rebuild resets and relaunches topology,
    collapse, and compressed representation without allocating device buffers
    or uploading segment metadata.
    """

    var triangle_count: Int
    var workspace: GpuBinaryBuildWorkspace
    var binary: GpuBinaryBoundsBvh
    var hploc: GpuHplocBuildState[
        Self.search_radius,
        Self.merging_threshold,
        Self.direct_conversion,
        True,
        True,
    ]
    var wide: GpuWideBoundsBvhBatch[
        8, CWBVH_LEAF_STORAGE_WIDTH, Self.max_leaf_size
    ]
    var collapse_workspace: GpuWideCollapseWorkspace
    var collapse: GpuWideCollapseState
    var representation_workspace: GpuCwbvh8RepresentationWorkspace
    var nodes: DeviceBuffer[.float32]
    var triangles: DeviceBuffer[.float32]
    var encoded_counts: DeviceBuffer[.uint32]

    def __init__(
        out self,
        mut ctx: DeviceContext,
        var leaf_bounds: DeviceBuffer[.float32],
        var leaf_payloads: DeviceBuffer[.uint32],
        vertices: DeviceBuffer[.float32],
    ) raises:
        comptime assert 1 <= Self.max_leaf_size <= 3
        var triangle_count = len(leaf_payloads)
        var segments = SegmentOffsets.single(triangle_count)
        var workspace = GpuBinaryBuildWorkspace(ctx, segments)
        workspace.ensure_topology(ctx)
        var binary = GpuBinaryBoundsBvh(
            ctx, leaf_bounds^, leaf_payloads^, workspace
        )
        enqueue_segmented_morton_codes(ctx, binary, workspace)
        enqueue_segmented_morton_sort(ctx, binary, workspace)
        ref topology = workspace.topology.value()
        var hploc = GpuHplocBuildState[
            Self.search_radius,
            Self.merging_threshold,
            Self.direct_conversion,
            True,
            True,
        ](
            ctx,
            binary.leaf_bounds.copy(),
            topology.morton_keys.copy(),
            binary.leaf_ids.copy(),
            binary.segments.copy(),
            binary.segment_offsets.copy(),
            binary.internal_segments.copy(),
            binary.internal_segment_offsets.copy(),
            binary.node_meta.copy(),
            topology.leaf_parent.copy(),
            binary.node_bounds.copy(),
            topology.node_flags.copy(),
            binary.node_leaf_counts.copy(),
        )
        binary.roots = hploc.root.copy()

        var wide = GpuWideBoundsBvhBatch[
            8, CWBVH_LEAF_STORAGE_WIDTH, Self.max_leaf_size
        ](ctx, segments)
        wide.bounds_device = binary.bounds_device.copy()
        var collapse_block_size = GPU_BOUNDS_BVH_BLOCK_SIZE
        comptime if Self.direct_conversion:
            collapse_block_size = CWBVH_COLLAPSE_BLOCK_SIZE
        var collapse_workspace = GpuWideCollapseWorkspace(
            ctx, segments, collapse_block_size
        )
        var nodes = ctx.enqueue_create_buffer[.float32](
            wide.node_segments.item_count() * CWBVH_NODE_WORDS
        )
        var triangles = ctx.enqueue_create_buffer[.float32](
            triangle_count * CWBVH_TRIANGLE_WORDS
        )
        var representation_workspace = GpuCwbvh8RepresentationWorkspace(
            ctx, triangle_count, 1
        )
        var collapse: GpuWideCollapseState
        var encoded_counts: DeviceBuffer[.uint32]
        comptime if Self.direct_conversion:
            collapse = enqueue_collapse_binary_to_cwbvh8[
                Self.max_leaf_size,
                True,
                True,
                Self.spatial_slots,
            ](
                ctx,
                binary,
                wide.node_segments,
                wide.node_segment_offsets,
                wide.leaf_block_segments,
                wide.leaf_block_segment_offsets,
                wide.leaf_block_counts,
                wide.node_counts,
                nodes,
                representation_workspace.compact_primitive_ids,
                representation_workspace.triangle_counters,
                hploc.compact_children,
                hploc.scratch_bounds,
                collapse_workspace,
            )
            comptime if not Self.indexed_triangle_layout:
                ctx.enqueue_function[
                    pack_segmented_cwbvh_triangles_kernel[True]
                ](
                    vertices,
                    representation_workspace.compact_primitive_ids,
                    _device_span[mut=False](binary.segment_offsets),
                    triangles,
                    Int32(triangle_count),
                    grid_dim=ceildiv(triangle_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
                    block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
                )
            encoded_counts = representation_workspace.triangle_counters.copy()
        else:
            collapse = _enqueue_collapse_binary_to_packed[
                8,
                CWBVH_LEAF_STORAGE_WIDTH,
                Self.max_leaf_size,
                True,
                True,
                _hploc_leaf_block_data,
            ](
                ctx,
                binary,
                wide.node_segments,
                wide.node_segment_offsets,
                wide.leaf_block_segments,
                wide.leaf_block_segment_offsets,
                wide.wide_nodes,
                wide.leaf_block_indices,
                wide.leaf_block_counts,
                wide.node_counts,
                collapse_workspace,
            )
            encoded_counts = (
                enqueue_segmented_cwbvh8_representation_with_workspace[
                    CWBVH_LEAF_STORAGE_WIDTH
                ](
                    ctx,
                    wide.wide_nodes,
                    wide.leaf_block_indices,
                    wide.node_segment_offsets,
                    wide.leaf_block_segment_offsets,
                    binary.segment_offsets,
                    wide.node_counts,
                    vertices,
                    nodes,
                    triangles,
                    representation_workspace,
                )
            )
        self.triangle_count = triangle_count
        self.workspace = workspace^
        self.binary = binary^
        self.hploc = hploc^
        self.wide = wide^
        self.collapse_workspace = collapse_workspace^
        self.collapse = collapse^
        self.representation_workspace = representation_workspace^
        self.nodes = nodes^
        self.triangles = triangles^
        self.encoded_counts = encoded_counts^

    def enqueue_rebuild(
        mut self,
        mut ctx: DeviceContext,
        vertices: DeviceBuffer[.float32],
    ) raises:
        """Rebuild into retained output and scratch buffers."""
        enqueue_segmented_morton_codes(ctx, self.binary, self.workspace)
        enqueue_segmented_morton_sort(ctx, self.binary, self.workspace)
        ref topology = self.workspace.topology.value()
        self.hploc.enqueue(
            ctx,
            self.binary.leaf_bounds,
            topology.morton_keys,
            self.binary.leaf_ids,
            self.binary.node_meta,
            topology.leaf_parent,
            self.binary.node_bounds,
            topology.node_flags,
            self.binary.node_leaf_counts,
        )
        comptime if Self.direct_conversion:
            self.collapse = enqueue_collapse_binary_to_cwbvh8[
                Self.max_leaf_size,
                True,
                True,
                Self.spatial_slots,
            ](
                ctx,
                self.binary,
                self.wide.node_segments,
                self.wide.node_segment_offsets,
                self.wide.leaf_block_segments,
                self.wide.leaf_block_segment_offsets,
                self.wide.leaf_block_counts,
                self.wide.node_counts,
                self.nodes,
                self.representation_workspace.compact_primitive_ids,
                self.representation_workspace.triangle_counters,
                self.hploc.compact_children,
                self.hploc.scratch_bounds,
                self.collapse_workspace,
            )
            comptime if not Self.indexed_triangle_layout:
                ctx.enqueue_function[
                    pack_segmented_cwbvh_triangles_kernel[True]
                ](
                    vertices,
                    self.representation_workspace.compact_primitive_ids,
                    _device_span[mut=False](self.binary.segment_offsets),
                    self.triangles,
                    Int32(self.triangle_count),
                    grid_dim=ceildiv(
                        self.triangle_count, GPU_BOUNDS_BVH_BLOCK_SIZE
                    ),
                    block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
                )
            self.encoded_counts = (
                self.representation_workspace.triangle_counters.copy()
            )
        else:
            self.collapse = _enqueue_collapse_binary_to_packed[
                8,
                CWBVH_LEAF_STORAGE_WIDTH,
                Self.max_leaf_size,
                True,
                True,
                _hploc_leaf_block_data,
            ](
                ctx,
                self.binary,
                self.wide.node_segments,
                self.wide.node_segment_offsets,
                self.wide.leaf_block_segments,
                self.wide.leaf_block_segment_offsets,
                self.wide.wide_nodes,
                self.wide.leaf_block_indices,
                self.wide.leaf_block_counts,
                self.wide.node_counts,
                self.collapse_workspace,
            )
            self.encoded_counts = (
                enqueue_segmented_cwbvh8_representation_with_workspace[
                    CWBVH_LEAF_STORAGE_WIDTH
                ](
                    ctx,
                    self.wide.wide_nodes,
                    self.wide.leaf_block_indices,
                    self.wide.node_segment_offsets,
                    self.wide.leaf_block_segment_offsets,
                    self.binary.segment_offsets,
                    self.wide.node_counts,
                    vertices,
                    self.nodes,
                    self.triangles,
                    self.representation_workspace,
                )
            )

    def finish_synchronized(self) raises:
        """Validate a rebuild after its stream has synchronized."""
        var status = self.hploc.result_status()
        if status != UInt32(HPLOC_STATUS_OK):
            raise String(t"H-PLOC build status: {status}")
        self.collapse.finish_batch_synchronized(self.wide, True)
        with self.encoded_counts.map_to_host() as counts:
            if counts[0] != UInt32(self.triangle_count):
                raise "reused CWBVH8 encoding lost triangles"
