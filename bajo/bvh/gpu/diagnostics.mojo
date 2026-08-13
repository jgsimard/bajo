"""Host-side GPU BVH validation helpers; never imported by builders."""

from std.math import max
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
from bajo.bvh.gpu.utils import GpuBVHValidation
from bajo.bvh.gpu.wide_layout import GpuWideBoundsBvh
from bajo.bvh.gpu.validate import (
    validate_sorted_keys,
    validate_topology,
    validate_refit_bounds,
)
from bajo.core import AABB


@fieldwise_init
struct GpuBinaryDiagnosticBuild:
    """Binary artifact plus retained construction state for validation."""

    var binary: GpuBinaryBoundsBvh
    var workspace: GpuBinaryBuildWorkspace


def build_bounds_bvh_for_diagnostics[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    max_leaf_size: Int,
    method: GpuBvhBuildMethod = GpuBvhBuildMethod.LBVH,
    pack_subtrees: Bool = False,
](
    mut ctx: DeviceContext,
    mut out: GpuWideBoundsBvh[node_width, leaf_width, max_leaf_size],
    leaf_bounds: DeviceBuffer[DType.float32],
    leaf_payloads: DeviceBuffer[DType.uint32],
) raises -> GpuBinaryDiagnosticBuild:
    """Build both final wide data and the diagnostic binary intermediate."""
    debug_assert["safe", _use_compiler_assume=True](
        out.leaf_count > 0 and len(leaf_payloads) == out.leaf_count
    )
    var workspace = GpuBinaryBuildWorkspace(ctx, max(out.leaf_count, 1))
    var binary = GpuBinaryBoundsBvh(ctx, leaf_bounds, leaf_payloads, workspace)
    out.bounds_device = binary.bounds_device.copy()
    _ = build_binary_bvh[method](ctx, binary, workspace)
    collapse_binary_to_wide[
        node_width,
        leaf_width,
        max_leaf_size,
        pack_subtrees,
    ](ctx, binary, out)
    return GpuBinaryDiagnosticBuild(binary^, workspace^)


def validate_binary_bvh(
    binary: GpuBinaryBoundsBvh,
    workspace: GpuBinaryBuildWorkspace,
    bounds: AABB,
) raises -> GpuBVHValidation:
    ref topology_scratch = workspace.topology.value()
    var sorted_validation = validate_sorted_keys(
        topology_scratch.morton_keys,
        binary.leaf_ids,
        binary.leaf_count,
    )
    if binary.leaf_count <= 1:
        return GpuBVHValidation(
            sorted_validation.sorted_ok,
            sorted_validation.values_ok,
            True,
            UInt32(1),
            UInt32(0),
            True,
            0.0,
            UInt32(0),
            sorted_validation.guard,
        )

    var topology = validate_topology(
        binary.node_meta,
        topology_scratch.leaf_parent,
        binary.leaf_count,
    )
    var refit = validate_refit_bounds(
        binary.node_bounds,
        topology_scratch.node_flags,
        binary.node_meta,
        binary.leaf_count,
        bounds,
    )
    return GpuBVHValidation(
        sorted_validation.sorted_ok,
        sorted_validation.values_ok,
        topology.ok,
        topology.root_count,
        topology.root_idx,
        refit.ok,
        refit.diff,
        refit.root_idx,
        sorted_validation.guard + topology.guard + refit.guard,
    )
