from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from bajo.bvh.gpu.builder.binary_layout import (
    GpuBinaryBoundsBvh,
    GpuBinaryBuildWorkspace,
)
from bajo.bvh.gpu.builder.hploc_layout import HPLOC_STATUS_OK
from bajo.bvh.gpu.builder.hploc_multi_wave import GpuHplocBuildState
from bajo.bvh.gpu.builder.lbvh import (
    enqueue_segmented_morton_codes,
    enqueue_segmented_morton_sort,
)
from bajo.bvh.gpu.utils import GpuBuildTimings


def enqueue_binary_bvh_with_hploc(
    mut ctx: DeviceContext,
    mut binary: GpuBinaryBoundsBvh,
    mut workspace: GpuBinaryBuildWorkspace,
) raises -> GpuHplocBuildState[]:
    """Queue H-PLOC without synchronizing or reading completion status."""
    if workspace.compact_output:
        raise "full H-PLOC requires a full binary output workspace"
    enqueue_segmented_morton_codes(ctx, binary, workspace)
    enqueue_segmented_morton_sort(ctx, binary, workspace)
    var hploc = GpuHplocBuildState[].from_binary(ctx, binary, workspace)
    return hploc^


def build_binary_bvh_with_hploc(
    mut ctx: DeviceContext,
    mut binary: GpuBinaryBoundsBvh,
    mut workspace: GpuBinaryBuildWorkspace,
    measure_stages: Bool = False,
) raises -> GpuBuildTimings:
    """Build H-PLOC directly into the production binary GPU layout."""

    var timings = GpuBuildTimings(0, 0, 0, 0, 0, 0, 0)
    var stage_start = Int(0)
    if measure_stages:
        ctx.synchronize()
        stage_start = perf_counter_ns()

    enqueue_segmented_morton_codes(ctx, binary, workspace)
    if measure_stages:
        ctx.synchronize()
        var stage_end = perf_counter_ns()
        timings.morton_ns = Int(stage_end - stage_start)
        stage_start = stage_end

    enqueue_segmented_morton_sort(ctx, binary, workspace)
    if measure_stages:
        ctx.synchronize()
        var stage_end = perf_counter_ns()
        timings.sort_ns = Int(stage_end - stage_start)
        stage_start = stage_end

    var hploc = GpuHplocBuildState[].from_binary(ctx, binary, workspace)

    # Keep construction scratch alive until the direct production writes finish
    # and turn a device-side failure into a normal raising API boundary.
    ctx.synchronize()
    var status = hploc.result_status()
    if status != UInt32(HPLOC_STATUS_OK):
        raise String(t"H-PLOC build status: {status}")

    if measure_stages:
        timings.topology_ns = Int(perf_counter_ns() - stage_start)
    return timings
