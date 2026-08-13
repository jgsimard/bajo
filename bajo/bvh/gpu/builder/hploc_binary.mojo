from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from bajo.bvh.constants import GPU_BOUNDS_BVH_BLOCK_SIZE
from bajo.bvh.gpu.builder.binary_layout import (
    GpuBinaryBoundsBvh,
    GpuBinaryBuildWorkspace,
)
from bajo.bvh.gpu.builder.hploc_layout import HPLOC_STATUS_OK
from bajo.bvh.gpu.builder.hploc_multi_wave import GpuHplocBuildState
from bajo.bvh.gpu.builder.lbvh import compute_bounds_morton_codes_kernel
from bajo.bvh.gpu.utils import GpuBuildTimings, _device_span
from bajo.sort.gpu.radix_sort import device_radix_sort_pairs


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

    ctx.enqueue_function[compute_bounds_morton_codes_kernel](
        _device_span[mut=False](binary.leaf_bounds),
        _device_span[mut=False](binary.bounds_device),
        _device_span[mut=True](binary.keys),
        _device_span[mut=True](binary.leaf_ids),
        grid_dim=binary.blocks_leaves(),
        block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
    )
    if measure_stages:
        ctx.synchronize()
        var stage_end = perf_counter_ns()
        timings.morton_ns = Int(stage_end - stage_start)
        stage_start = stage_end

    device_radix_sort_pairs[DType.uint32, DType.uint32](
        ctx,
        workspace.sort,
        binary.keys,
        binary.leaf_ids,
        binary.leaf_count,
    )
    if measure_stages:
        ctx.synchronize()
        var stage_end = perf_counter_ns()
        timings.sort_ns = Int(stage_end - stage_start)
        stage_start = stage_end

    var hploc = GpuHplocBuildState(
        ctx,
        binary.leaf_bounds.copy(),
        binary.keys.copy(),
        binary.leaf_ids.copy(),
        binary.node_meta.copy(),
        binary.leaf_parent.copy(),
        binary.node_bounds.copy(),
        binary.node_flags.copy(),
        binary.node_leaf_counts.copy(),
    )

    # Keep construction scratch alive until the direct production writes finish
    # and turn a device-side failure into a normal raising API boundary.
    ctx.synchronize()
    var status = hploc.result_status()
    if status != UInt32(HPLOC_STATUS_OK):
        raise String(t"H-PLOC build status: {status}")

    if measure_stages:
        timings.topology_ns = Int(perf_counter_ns() - stage_start)
    return timings
