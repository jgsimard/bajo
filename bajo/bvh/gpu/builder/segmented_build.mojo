"""PrimitiveKind-agnostic segmented GPU hierarchy orchestration."""

from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh.gpu.builder.binary_builder import GpuBvhBuildMethod
from bajo.bvh.gpu.builder.binary_layout import (
    GpuBinaryBoundsBvh,
    GpuBinaryBuildWorkspace,
)
from bajo.bvh.gpu.builder.hploc_binary import (
    build_binary_bvh_with_hploc,
    enqueue_binary_bvh_with_hploc,
)
from bajo.bvh.gpu.builder.hploc_layout import HPLOC_STATUS_OK
from bajo.bvh.gpu.builder.hploc_multi_wave import GpuHplocBuildState
from bajo.bvh.gpu.builder.lbvh import (
    build_binary_bvh_with_lbvh,
    enqueue_segmented_morton_codes,
    enqueue_segmented_morton_sort,
)
from bajo.bvh.gpu.builder.wide_collapse import (
    GpuWideCollapseState,
    GpuWideCollapseWorkspace,
    HplocWideLeafDataFn,
    _hploc_leaf_block_data,
    _hploc_embedded_leaf_payload,
    _enqueue_collapse_binary_to_packed,
    _enqueue_collapse_binary_to_wide_batch,
)
from bajo.bvh.gpu.utils import GpuBuildTimings
from bajo.bvh.gpu.wide_layout import GpuWideBoundsBvh, GpuWideBoundsBvhBatch
from bajo.core import SegmentOffsets


@fieldwise_init
struct GpuSegmentedWideBuildTicket[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    max_leaf_size: Int,
    build_method: GpuBvhBuildMethod,
    fat_leaves: Bool,
    spatial_slots: Bool,
]:
    """Own one queued segmented binary build and packed wide collapse.

    PrimitiveKind adapters may enqueue their leaf-packing work from ``binary`` and
    ``wide`` before calling ``wait``.  This keeps one synchronization boundary
    for the whole BLAS batch without duplicating hierarchy scheduling.
    """

    var workspace: GpuBinaryBuildWorkspace
    var binary: GpuBinaryBoundsBvh
    var wide: GpuWideBoundsBvhBatch[
        Self.node_width, Self.leaf_width, Self.max_leaf_size
    ]
    var hploc: Optional[GpuHplocBuildState[]]
    var collapse: GpuWideCollapseState
    var timings: GpuBuildTimings
    var collapse_start_ns: Int

    def finish_synchronized(mut self) raises:
        if self.hploc:
            var status = self.hploc.value().result_status()
            if status != UInt32(HPLOC_STATUS_OK):
                raise String(t"H-PLOC build status: {status}")
        self.collapse.finish_batch_synchronized(self.wide, Self.fat_leaves)
        if self.collapse_start_ns != 0:
            self.timings.collapse_ns = Int(
                perf_counter_ns() - self.collapse_start_ns
            )

    def wait(mut self, ctx: DeviceContext) raises:
        ctx.synchronize()
        self.finish_synchronized()

    def wait_into_single_segment(
        deinit self, mut ctx: DeviceContext, mut timings: GpuBuildTimings
    ) raises -> GpuWideBoundsBvh[
        Self.node_width, Self.leaf_width, Self.max_leaf_size
    ]:
        """Finish and consume the only segment as a standalone tree owner."""
        ctx.synchronize()
        self.finish_synchronized()
        timings = self.timings
        return self.wide^.into_single_segment(ctx)

    def wait_into_exact_bvh2_leaf1(
        deinit self, mut ctx: DeviceContext, mut timings: GpuBuildTimings
    ) raises -> GpuWideBoundsBvh[
        Self.node_width, Self.leaf_width, Self.max_leaf_size
    ]:
        """Finish an already exact one-segment BVH2/leaf1 allocation."""
        ctx.synchronize()
        self.finish_synchronized()
        timings = self.timings
        return self.wide^.into_exact_bvh2_leaf1()

    def take_single_segment_synchronized(
        deinit self, mut ctx: DeviceContext, mut timings: GpuBuildTimings
    ) raises -> GpuWideBoundsBvh[
        Self.node_width, Self.leaf_width, Self.max_leaf_size
    ]:
        """Consume a ticket already completed by ``wait``."""
        timings = self.timings
        return self.wide^.into_single_segment(ctx)


struct GpuWideBuildArena[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    max_leaf_size: Int = Int(leaf_width),
    build_method: GpuBvhBuildMethod = .HPLOC,
    fat_leaves: Bool = False,
    spatial_slots: Bool = False,
]:
    """Reusable fixed-shape binary and ordinary-wide construction.

    The immutable input buffers and segment layout are retained. Rebuilds reuse
    radix, H-PLOC/LBVH, collapse, and output allocations; callers must not
    change leaf bounds or payloads and must synchronize before starting another
    rebuild or reading the output.
    """

    var workspace: GpuBinaryBuildWorkspace
    var binary: GpuBinaryBoundsBvh
    var hploc: Optional[GpuHplocBuildState[]]
    var wide: GpuWideBoundsBvhBatch[
        Self.node_width, Self.leaf_width, Self.max_leaf_size
    ]
    var collapse_workspace: GpuWideCollapseWorkspace
    var collapse: GpuWideCollapseState

    def __init__(
        out self,
        mut ctx: DeviceContext,
        segments: SegmentOffsets,
        var leaf_bounds: DeviceBuffer[.float32],
        var leaf_payloads: DeviceBuffer[.uint32],
    ) raises:
        comptime assert Self.max_leaf_size > 0
        comptime assert Self.max_leaf_size <= Int(Self.leaf_width)
        var workspace = GpuBinaryBuildWorkspace(ctx, segments)
        workspace.ensure_topology(ctx)
        var binary = GpuBinaryBoundsBvh(
            ctx, leaf_bounds^, leaf_payloads^, workspace
        )
        var hploc = Optional[GpuHplocBuildState[]]()
        comptime if Self.build_method == .HPLOC:
            enqueue_segmented_morton_codes(ctx, binary, workspace)
            enqueue_segmented_morton_sort(ctx, binary, workspace)
            ref topology = workspace.topology.value()
            hploc = Optional(
                GpuHplocBuildState[](
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
            )
            binary.roots = hploc.value().root.copy()
        elif Self.build_method == .LBVH:
            _ = build_binary_bvh_with_lbvh(ctx, binary, workspace)
        else:
            comptime assert False, "unknown GPU BVH build method"
        var wide = GpuWideBoundsBvhBatch[
            Self.node_width, Self.leaf_width, Self.max_leaf_size
        ](ctx, segments)
        wide.bounds_device = binary.bounds_device.copy()
        var collapse_workspace = GpuWideCollapseWorkspace(ctx, segments)
        var collapse = _enqueue_collapse_binary_to_packed[
            Self.node_width,
            Self.leaf_width,
            Self.max_leaf_size,
            Self.fat_leaves,
            Self.spatial_slots,
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
        self.workspace = workspace^
        self.binary = binary^
        self.hploc = hploc^
        self.wide = wide^
        self.collapse_workspace = collapse_workspace^
        self.collapse = collapse^

    def enqueue_rebuild(mut self, mut ctx: DeviceContext) raises:
        """Rebuild the retained inputs without any device allocation."""
        comptime if Self.build_method == .HPLOC:
            enqueue_segmented_morton_codes(ctx, self.binary, self.workspace)
            enqueue_segmented_morton_sort(ctx, self.binary, self.workspace)
            ref topology = self.workspace.topology.value()
            self.hploc.value().enqueue(
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
        else:
            _ = build_binary_bvh_with_lbvh(ctx, self.binary, self.workspace)
        self.collapse = _enqueue_collapse_binary_to_packed[
            Self.node_width,
            Self.leaf_width,
            Self.max_leaf_size,
            Self.fat_leaves,
            Self.spatial_slots,
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

    def finish_synchronized(self) raises:
        comptime if Self.build_method == .HPLOC:
            var status = self.hploc.value().result_status()
            if status != UInt32(HPLOC_STATUS_OK):
                raise String(t"H-PLOC build status: {status}")
        self.collapse.finish_batch_synchronized(self.wide, Self.fat_leaves)


def _enqueue_segmented_wide_build[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    max_leaf_size: Int,
    build_method: GpuBvhBuildMethod,
    fat_leaves: Bool,
    spatial_slots: Bool,
    leaf_data_fn: HplocWideLeafDataFn,
](
    mut ctx: DeviceContext,
    segments: SegmentOffsets,
    var leaf_bounds: DeviceBuffer[.float32],
    var leaf_payloads: DeviceBuffer[.uint32],
    measure_stages: Bool = False,
) raises -> GpuSegmentedWideBuildTicket[
    node_width,
    leaf_width,
    max_leaf_size,
    build_method,
    fat_leaves,
    spatial_slots,
]:
    """Queue the common segmented bounds → binary → packed-wide pipeline."""
    comptime assert max_leaf_size > 0 and max_leaf_size <= Int(leaf_width)
    var workspace = GpuBinaryBuildWorkspace(ctx, segments)
    workspace.ensure_topology(ctx)
    var binary = GpuBinaryBoundsBvh(
        ctx, leaf_bounds^, leaf_payloads^, workspace
    )
    var hploc = Optional[GpuHplocBuildState[]]()
    var timings = GpuBuildTimings(0, 0, 0, 0, 0, 0, 0)
    comptime if build_method == .LBVH:
        timings = build_binary_bvh_with_lbvh(
            ctx, binary, workspace, measure_stages
        )
    elif build_method == .HPLOC:
        if measure_stages:
            timings = build_binary_bvh_with_hploc(ctx, binary, workspace, True)
        else:
            hploc = Optional[GpuHplocBuildState[]](
                enqueue_binary_bvh_with_hploc(ctx, binary, workspace)
            )
    else:
        comptime assert False, "unknown GPU BVH build method"

    var wide = GpuWideBoundsBvhBatch[node_width, leaf_width, max_leaf_size](
        ctx, segments
    )
    var collapse_start_ns = Int(0)
    if measure_stages:
        ctx.synchronize()
        collapse_start_ns = perf_counter_ns()
    var collapse = _enqueue_collapse_binary_to_wide_batch[
        node_width,
        leaf_width,
        max_leaf_size,
        fat_leaves,
        spatial_slots,
        leaf_data_fn,
    ](ctx, binary, wide)
    return GpuSegmentedWideBuildTicket[
        node_width,
        leaf_width,
        max_leaf_size,
        build_method,
        fat_leaves,
        spatial_slots,
    ](
        workspace^,
        binary^,
        wide^,
        hploc^,
        collapse^,
        timings,
        collapse_start_ns,
    )


def enqueue_segmented_wide_build[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    max_leaf_size: Int,
    build_method: GpuBvhBuildMethod,
    fat_leaves: Bool = False,
    spatial_slots: Bool = False,
](
    mut ctx: DeviceContext,
    segments: SegmentOffsets,
    var leaf_bounds: DeviceBuffer[.float32],
    var leaf_payloads: DeviceBuffer[.uint32],
    measure_stages: Bool = False,
) raises -> GpuSegmentedWideBuildTicket[
    node_width,
    leaf_width,
    max_leaf_size,
    build_method,
    fat_leaves,
    spatial_slots,
]:
    return _enqueue_segmented_wide_build[
        node_width,
        leaf_width,
        max_leaf_size,
        build_method,
        fat_leaves,
        spatial_slots,
        _hploc_leaf_block_data,
    ](
        ctx,
        segments,
        leaf_bounds^,
        leaf_payloads^,
        measure_stages,
    )


def enqueue_segmented_wide_build_embedded_leaf1[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    max_leaf_size: Int,
    build_method: GpuBvhBuildMethod,
    fat_leaves: Bool = False,
    spatial_slots: Bool = False,
](
    mut ctx: DeviceContext,
    segments: SegmentOffsets,
    var leaf_bounds: DeviceBuffer[.float32],
    var leaf_payloads: DeviceBuffer[.uint32],
    measure_stages: Bool = False,
) raises -> GpuSegmentedWideBuildTicket[
    node_width,
    leaf_width,
    max_leaf_size,
    build_method,
    fat_leaves,
    spatial_slots,
]:
    """Queue a leaf1 build whose payload is the leaf metadata data field."""
    comptime assert leaf_width == 1 and max_leaf_size == 1
    return _enqueue_segmented_wide_build[
        node_width,
        leaf_width,
        max_leaf_size,
        build_method,
        fat_leaves,
        spatial_slots,
        _hploc_embedded_leaf_payload,
    ](
        ctx,
        segments,
        leaf_bounds^,
        leaf_payloads^,
        measure_stages,
    )


def build_single_segment_wide[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    max_leaf_size: Int,
    build_method: GpuBvhBuildMethod,
    fat_leaves: Bool = False,
    spatial_slots: Bool = False,
](
    mut ctx: DeviceContext,
    leaf_count: Int,
    var leaf_bounds: DeviceBuffer[.float32],
    var leaf_payloads: DeviceBuffer[.uint32],
    mut timings: GpuBuildTimings,
    measure_stages: Bool = False,
) raises -> GpuWideBoundsBvh[node_width, leaf_width, max_leaf_size]:
    """Build one BVH through the exact segmented pipeline used by BLAS sets."""
    debug_assert["safe", _use_compiler_assume=True](
        leaf_count > 0, "standalone BVH requires a nonempty segment"
    )
    var build = enqueue_segmented_wide_build[
        node_width,
        leaf_width,
        max_leaf_size,
        build_method,
        fat_leaves,
        spatial_slots,
    ](
        ctx,
        SegmentOffsets.single(leaf_count),
        leaf_bounds^,
        leaf_payloads^,
        measure_stages,
    )
    return build^.wait_into_single_segment(ctx, timings)


def build_single_segment_wide_embedded_leaf1[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    max_leaf_size: Int,
    build_method: GpuBvhBuildMethod,
](
    mut ctx: DeviceContext,
    leaf_count: Int,
    var leaf_bounds: DeviceBuffer[.float32],
    var leaf_payloads: DeviceBuffer[.uint32],
    mut timings: GpuBuildTimings,
    measure_stages: Bool = False,
) raises -> GpuWideBoundsBvh[node_width, leaf_width, max_leaf_size]:
    """Build a standalone leaf1 BVH with payloads embedded in metadata."""
    comptime assert leaf_width == 1 and max_leaf_size == 1
    debug_assert["safe", _use_compiler_assume=True](
        leaf_count > 0, "standalone BVH requires a nonempty segment"
    )
    var build = enqueue_segmented_wide_build_embedded_leaf1[
        node_width,
        leaf_width,
        max_leaf_size,
        build_method,
        True,
        False,
    ](
        ctx,
        SegmentOffsets.single(leaf_count),
        leaf_bounds^,
        leaf_payloads^,
        measure_stages,
    )
    comptime if node_width == 2:
        return build^.wait_into_exact_bvh2_leaf1(ctx, timings)
    return build^.wait_into_single_segment(ctx, timings)
