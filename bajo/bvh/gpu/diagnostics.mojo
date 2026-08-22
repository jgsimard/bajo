"""Host-side GPU BVH validation helpers; never imported by builders."""

from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh.gpu.builder.binary_builder import GpuBvhBuildMethod
from bajo.bvh.gpu.builder.binary_layout import (
    GpuBinaryBoundsBvh,
    GpuBinaryBuildWorkspace,
)
from bajo.bvh.gpu.builder.segmented_build import (
    GpuSegmentedWideBuildTicket,
    enqueue_segmented_wide_build,
)
from bajo.bvh.gpu.utils import GpuBVHValidation
from bajo.bvh.gpu.wide_layout import GpuWideBoundsBvh
from bajo.bvh.gpu.validate import (
    validate_sorted_keys,
    validate_topology,
    validate_refit_bounds,
)
from bajo.core import AABB, SegmentOffsets


@fieldwise_init
struct GpuBinaryDiagnosticBuild[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    max_leaf_size: Int,
    method: GpuBvhBuildMethod,
    pack_subtrees: Bool,
]:
    """Completed segmented build retained for validation."""

    var wide: GpuWideBoundsBvh[
        Self.node_width, Self.leaf_width, Self.max_leaf_size
    ]
    var build: GpuSegmentedWideBuildTicket[
        Self.node_width,
        Self.leaf_width,
        Self.max_leaf_size,
        Self.method,
        Self.pack_subtrees,
        False,
    ]


def build_bounds_bvh_for_diagnostics[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    max_leaf_size: Int,
    method: GpuBvhBuildMethod = GpuBvhBuildMethod.LBVH,
    pack_subtrees: Bool = False,
](
    mut ctx: DeviceContext,
    leaf_bounds: DeviceBuffer[DType.float32],
    leaf_payloads: DeviceBuffer[DType.uint32],
) raises -> GpuBinaryDiagnosticBuild[
    node_width, leaf_width, max_leaf_size, method, pack_subtrees
]:
    """Build both final wide data and the diagnostic binary intermediate."""
    var leaf_count = len(leaf_payloads)
    debug_assert["safe", _use_compiler_assume=True](
        leaf_count > 0 and len(leaf_bounds) == leaf_count * AABB.STRIDE
    )
    var build = enqueue_segmented_wide_build[
        node_width, leaf_width, max_leaf_size, method, pack_subtrees
    ](
        ctx,
        SegmentOffsets.single(leaf_count),
        leaf_bounds.copy(),
        leaf_payloads.copy(),
    )
    build.wait(ctx)
    var wide = build.wide.single_segment_view()
    return GpuBinaryDiagnosticBuild[
        node_width, leaf_width, max_leaf_size, method, pack_subtrees
    ](wide^, build^)


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
