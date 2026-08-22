"""Primitive-agnostic segmented GPU hierarchy orchestration."""

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
from bajo.bvh.gpu.builder.lbvh import build_binary_bvh_with_lbvh
from bajo.bvh.gpu.builder.wide_collapse import (
    GpuWideCollapseState,
    enqueue_collapse_binary_to_wide_batch,
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

    Primitive adapters may enqueue their leaf-packing work from ``binary`` and
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

    def take_single_segment_synchronized(
        deinit self, mut ctx: DeviceContext, mut timings: GpuBuildTimings
    ) raises -> GpuWideBoundsBvh[
        Self.node_width, Self.leaf_width, Self.max_leaf_size
    ]:
        """Consume a ticket already completed by ``wait``."""
        timings = self.timings
        return self.wide^.into_single_segment(ctx)


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
    var leaf_bounds: DeviceBuffer[DType.float32],
    var leaf_payloads: DeviceBuffer[DType.uint32],
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
    comptime if build_method == GpuBvhBuildMethod.LBVH:
        timings = build_binary_bvh_with_lbvh(
            ctx, binary, workspace, measure_stages
        )
    elif build_method == GpuBvhBuildMethod.HPLOC:
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
    var collapse = enqueue_collapse_binary_to_wide_batch[
        node_width,
        leaf_width,
        max_leaf_size,
        fat_leaves,
        spatial_slots,
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
    var leaf_bounds: DeviceBuffer[DType.float32],
    var leaf_payloads: DeviceBuffer[DType.uint32],
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
