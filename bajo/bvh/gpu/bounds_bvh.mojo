from std.math import max
from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh.gpu.builder.binary_builder import (
    GpuBvhBuildMethod,
    build_binary_bvh,
)
from bajo.bvh.gpu.builder.binary_layout import (
    GpuBinaryBoundsBvh,
    GpuBinaryBuildWorkspace,
)
from bajo.bvh.gpu.builder.wide_collapse import collapse_binary_to_wide
from bajo.bvh.gpu.utils import GpuBuildTimings
from bajo.bvh.gpu.wide_layout import GpuWideBoundsBvh


def build_bounds_bvh[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    max_leaf_size: Int,
    method: GpuBvhBuildMethod = GpuBvhBuildMethod.LBVH,
    pack_subtrees: Bool = False,
    spatial_slots: Bool = False,
](
    mut ctx: DeviceContext,
    mut out: GpuWideBoundsBvh[node_width, leaf_width, max_leaf_size],
    leaf_bounds: DeviceBuffer[DType.float32],
    leaf_payloads: DeviceBuffer[DType.uint32],
    measure_build: Bool = False,
) raises -> GpuBuildTimings:
    """Build final wide bounds data with internally allocated scratch."""
    var workspace = GpuBinaryBuildWorkspace(ctx, max(out.leaf_count, 1))
    return build_bounds_bvh_with_workspace[
        node_width,
        leaf_width,
        max_leaf_size,
        method,
        pack_subtrees,
        spatial_slots,
    ](
        ctx,
        out,
        leaf_bounds,
        leaf_payloads,
        workspace,
        measure_build,
    )


def build_bounds_bvh_with_workspace[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    max_leaf_size: Int,
    method: GpuBvhBuildMethod = GpuBvhBuildMethod.LBVH,
    pack_subtrees: Bool = False,
    spatial_slots: Bool = False,
](
    mut ctx: DeviceContext,
    mut out: GpuWideBoundsBvh[node_width, leaf_width, max_leaf_size],
    leaf_bounds: DeviceBuffer[DType.float32],
    leaf_payloads: DeviceBuffer[DType.uint32],
    mut workspace: GpuBinaryBuildWorkspace,
    measure_build: Bool = False,
) raises -> GpuBuildTimings:
    """Build final wide bounds data with caller-owned reusable scratch."""
    debug_assert["safe", _use_compiler_assume=True](
        out.leaf_count > 0, "passed empty input."
    )
    comptime assert max_leaf_size > 0
    comptime assert max_leaf_size <= Int(leaf_width)
    debug_assert["safe", _use_compiler_assume=True](
        len(leaf_payloads) == out.leaf_count
    )

    var binary = GpuBinaryBoundsBvh(ctx, leaf_bounds, leaf_payloads, workspace)
    out.bounds_device = binary.bounds_device.copy()
    var timings = build_binary_bvh[method](
        ctx,
        binary,
        workspace,
        measure_stages=measure_build,
    )

    var collapse_start = Int(0)
    if measure_build:
        collapse_start = perf_counter_ns()

    collapse_binary_to_wide[
        node_width,
        leaf_width,
        max_leaf_size,
        pack_subtrees,
        spatial_slots,
    ](ctx, binary, out)

    if measure_build:
        timings.collapse_ns = Int(perf_counter_ns() - collapse_start)
    return timings
