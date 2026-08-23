"""Shared finalization for ordinary-wide segmented GPU BLAS sets."""

from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh.constants import PrimitiveKind, WideNode
from bajo.bvh.gpu.blas_desc import enqueue_segmented_blas_descriptors
from bajo.bvh.gpu.blas_storage import GpuBlasSet, GpuBvhLayout
from bajo.bvh.gpu.builder.binary_builder import GpuBvhBuildMethod
from bajo.bvh.gpu.builder.segmented_build import GpuSegmentedWideBuildTicket
from bajo.bvh.gpu.wide_layout import (
    GpuCompactWideLayout,
    enqueue_compact_segmented_buffer,
)


def finalize_ordinary_wide_blas_set[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    build_method: GpuBvhBuildMethod,
    kind: PrimitiveKind,
    layout: GpuBvhLayout,
    leaf_record_stride: Int,
](
    mut ctx: DeviceContext,
    var hierarchy: GpuSegmentedWideBuildTicket[
        node_width,
        leaf_width,
        Int(leaf_width),
        build_method,
        True,
        False,
    ],
    var leaf_workspace: DeviceBuffer[.float32],
) raises -> GpuBlasSet[kind, layout, node_width, leaf_width]:
    """Compact a completed ordinary-wide build and emit its descriptors."""
    comptime assert leaf_record_stride > 0
    ctx.synchronize()
    hierarchy.finish_synchronized()
    ref binary = hierarchy.binary
    ref wide = hierarchy.wide
    var segment_count = wide.segments.segment_count()
    var compact_layout = GpuCompactWideLayout(
        ctx, wide.node_counts, wide.leaf_block_counts, segment_count
    )
    var compact_nodes = enqueue_compact_segmented_buffer[
        DType.float32, node_width * WideNode.CHILD_STRIDE
    ](
        ctx,
        wide.wide_nodes,
        wide.node_segment_offsets,
        compact_layout.node_segment_offsets,
        compact_layout.node_segments.item_count(),
        segment_count,
    )
    var compact_leaves = enqueue_compact_segmented_buffer[
        DType.float32, leaf_width * leaf_record_stride
    ](
        ctx,
        leaf_workspace,
        wide.leaf_block_segment_offsets,
        compact_layout.leaf_block_segment_offsets,
        compact_layout.leaf_block_segments.item_count(),
        segment_count,
    )
    var descs = enqueue_segmented_blas_descriptors[
        node_width * WideNode.CHILD_STRIDE,
        leaf_width * leaf_record_stride,
    ](
        ctx,
        compact_layout.node_segment_offsets,
        compact_layout.leaf_block_segment_offsets,
        binary.segment_offsets,
        wide.node_counts,
        wide.leaf_block_counts,
        segment_count,
    )
    ctx.synchronize()
    return GpuBlasSet[kind, layout, node_width, leaf_width](
        descs^,
        compact_nodes^,
        compact_leaves^,
        segment_count,
    )
