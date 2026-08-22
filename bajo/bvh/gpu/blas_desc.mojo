"""On-device descriptor emission for segmented packed BLAS storage."""

from std.gpu import global_idx
from std.math import ceildiv
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh.constants import GPU_BOUNDS_BVH_BLOCK_SIZE
from bajo.bvh.gpu.utils import _device_span
from bajo.bvh.types import BlasDescLayout


def emit_segmented_blas_descriptors_kernel[
    node_f32_stride: Int,
    leaf_f32_stride: Int,
](
    node_segment_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    leaf_segment_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    primitive_segment_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    node_counts: ImmSpan[UInt32, ImmutAnyOrigin],
    leaf_counts: ImmSpan[UInt32, ImmutAnyOrigin],
    descs: MutSpan[UInt32, MutAnyOrigin],
    segment_count: Int32,
):
    """Write one descriptor from device-resident segmented build results."""
    var segment_idx = global_idx.x
    if segment_idx >= Int(segment_count):
        return

    var desc_base = BlasDescLayout.base(segment_idx)
    var primitive_begin = primitive_segment_offsets.unsafe_get(segment_idx)
    var primitive_end = primitive_segment_offsets.unsafe_get(segment_idx + 1)
    descs.unsafe_get(
        desc_base + BlasDescLayout.NODE_F32_BASE
    ) = node_segment_offsets.unsafe_get(segment_idx) * UInt32(node_f32_stride)
    descs.unsafe_get(
        desc_base + BlasDescLayout.LEAF_F32_BASE
    ) = leaf_segment_offsets.unsafe_get(segment_idx) * UInt32(leaf_f32_stride)
    descs.unsafe_get(desc_base + BlasDescLayout.ROOT_IDX) = UInt32(0)
    descs.unsafe_get(
        desc_base + BlasDescLayout.NODE_COUNT
    ) = node_counts.unsafe_get(segment_idx)
    descs.unsafe_get(
        desc_base + BlasDescLayout.LEAF_BLOCK_COUNT
    ) = leaf_counts.unsafe_get(segment_idx)
    descs.unsafe_get(desc_base + BlasDescLayout.PRIM_COUNT) = (
        primitive_end - primitive_begin
    )


def enqueue_segmented_blas_descriptors[
    node_f32_stride: Int,
    leaf_f32_stride: Int,
](
    mut ctx: DeviceContext,
    node_segment_offsets: DeviceBuffer[DType.uint32],
    leaf_segment_offsets: DeviceBuffer[DType.uint32],
    primitive_segment_offsets: DeviceBuffer[DType.uint32],
    node_counts: DeviceBuffer[DType.uint32],
    leaf_counts: DeviceBuffer[DType.uint32],
    segment_count: Int,
) raises -> DeviceBuffer[DType.uint32]:
    """Enqueue descriptor generation without a device-to-host round trip."""
    comptime assert node_f32_stride > 0
    comptime assert leaf_f32_stride > 0
    debug_assert["safe", _use_compiler_assume=True](
        segment_count > 0, "descriptor emission requires at least one segment"
    )
    debug_assert["safe", _use_compiler_assume=True](
        len(node_segment_offsets) >= segment_count + 1,
        "node segment offsets are too short",
    )
    debug_assert["safe", _use_compiler_assume=True](
        len(leaf_segment_offsets) >= segment_count + 1,
        "leaf segment offsets are too short",
    )
    debug_assert["safe", _use_compiler_assume=True](
        len(primitive_segment_offsets) >= segment_count + 1,
        "primitive segment offsets are too short",
    )
    debug_assert["safe", _use_compiler_assume=True](
        len(node_counts) >= segment_count and len(leaf_counts) >= segment_count,
        "descriptor count buffers are too short",
    )

    var descs = ctx.enqueue_create_buffer[DType.uint32](
        segment_count * BlasDescLayout.STRIDE
    )
    ctx.enqueue_function[
        emit_segmented_blas_descriptors_kernel[node_f32_stride, leaf_f32_stride]
    ](
        _device_span[mut=False](node_segment_offsets),
        _device_span[mut=False](leaf_segment_offsets),
        _device_span[mut=False](primitive_segment_offsets),
        _device_span[mut=False](node_counts),
        _device_span[mut=False](leaf_counts),
        _device_span[mut=True](descs),
        Int32(segment_count),
        grid_dim=ceildiv(segment_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
        block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
    )
    return descs^
