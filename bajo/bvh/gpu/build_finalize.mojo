"""Device-side validation and compact offsets for completed GPU BVH builds."""

from max.gpu import thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.primitives import block
from max.gpu.sync import barrier
from std.math import min
from std.memory import stack_allocation

from bajo.gpu.buffers import _device_span


comptime _CHECK_SIZES = UInt32(1)
comptime _CHECK_HPLOC = UInt32(2)
comptime _CHECK_COLLAPSE = UInt32(4)
comptime _CHECK_ENCODING = UInt32(8)
comptime _FAT_LEAVES = UInt32(16)


def finalize_wide_layout_kernel[
    block_size: Int
](
    node_counts: ImmSpan[UInt32, ImmutAnyOrigin],
    leaf_counts: ImmSpan[UInt32, ImmutAnyOrigin],
    primitive_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    node_capacity_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    hploc_status: ImmSpan[UInt32, ImmutAnyOrigin],
    collapse_status: ImmSpan[UInt32, ImmutAnyOrigin],
    encoded_counts: ImmSpan[UInt32, ImmutAnyOrigin],
    node_offsets: MutSpan[UInt32, MutAnyOrigin],
    leaf_offsets: MutSpan[UInt32, MutAnyOrigin],
    summary: MutSpan[UInt32, MutAnyOrigin],
    segment_count: Int32,
    flags: UInt32,
):
    """Scan arbitrary segment counts in tiles, reporting the first bad segment.

    One block suffices for this small metadata workload. UInt64 sums detect
    overflow before any compact allocation or copy uses the offsets.
    """
    var lane = thread_idx.x
    var count = Int(segment_count)
    var validate_sizes = (flags & _CHECK_SIZES) != 0
    var validate_hploc = (flags & _CHECK_HPLOC) != 0
    var validate_collapse = (flags & _CHECK_COLLAPSE) != 0
    var validate_encoding = (flags & _CHECK_ENCODING) != 0
    var fat_leaves = (flags & _FAT_LEAVES) != 0
    var errors = stack_allocation[
        block_size, UInt32, address_space=AddressSpace.SHARED
    ]()
    var totals = stack_allocation[
        2, UInt64, address_space=AddressSpace.SHARED
    ]()
    var node_total = UInt64(0)
    var leaf_total = UInt64(0)
    var first_error = UInt32(0)
    var error_segment = 0
    for begin in range(0, count, block_size):
        var segment = begin + lane
        var nodes = UInt32(0)
        var leaves = UInt32(0)
        var error = UInt32(0)
        if segment < count:
            nodes = node_counts.unsafe_get(segment)
            leaves = leaf_counts.unsafe_get(segment)
            if validate_hploc and hploc_status.unsafe_get(segment) != 0:
                error = 1
            elif validate_collapse and collapse_status.unsafe_get(segment) != 0:
                error = 2
            elif validate_sizes:
                var source_count = primitive_offsets.unsafe_get(
                    segment + 1
                ) - primitive_offsets.unsafe_get(segment)
                var node_capacity = node_capacity_offsets.unsafe_get(
                    segment + 1
                ) - node_capacity_offsets.unsafe_get(segment)
                if source_count == 0:
                    if nodes != 0 or leaves != 0:
                        error = 3
                elif nodes == 0 or nodes > node_capacity:
                    error = 4
                elif fat_leaves:
                    if leaves == 0 or leaves > source_count:
                        error = 5
                elif leaves != source_count:
                    error = 5
                if error == 0 and validate_encoding:
                    if encoded_counts.unsafe_get(segment) != source_count:
                        error = 6
        errors[unsafe_offset=lane] = error
        var tile_errors = block.sum[block_size=block_size, broadcast=True](
            UInt32(error != 0)
        )
        var node_prefix = block.prefix_sum[
            block_size=block_size, exclusive=False
        ](UInt64(nodes))
        var leaf_prefix = block.prefix_sum[
            block_size=block_size, exclusive=False
        ](UInt64(leaves))
        if segment < count:
            node_offsets.unsafe_get(segment) = UInt32(
                node_total + node_prefix - UInt64(nodes)
            )
            leaf_offsets.unsafe_get(segment) = UInt32(
                leaf_total + leaf_prefix - UInt64(leaves)
            )
        if lane == block_size - 1:
            totals[unsafe_offset=0] = node_total + node_prefix
            totals[unsafe_offset=1] = leaf_total + leaf_prefix
        barrier()
        if lane == 0 and first_error == 0 and tile_errors != 0:
            for index in range(min(block_size, count - begin)):
                if errors[unsafe_offset=index] != 0:
                    first_error = errors[unsafe_offset=index]
                    error_segment = begin + index
                    break
        node_total = totals[unsafe_offset=0]
        leaf_total = totals[unsafe_offset=1]
        barrier()
    if lane == 0:
        if first_error == 0 and (
            node_total > UInt64(0xFFFFFFFF) or leaf_total > UInt64(0xFFFFFFFF)
        ):
            first_error = 7
        node_offsets.unsafe_get(count) = UInt32(node_total)
        leaf_offsets.unsafe_get(count) = UInt32(leaf_total)
        summary.unsafe_get(0) = UInt32(node_total)
        summary.unsafe_get(1) = UInt32(leaf_total)
        summary.unsafe_get(2) = first_error
        summary.unsafe_get(3) = UInt32(error_segment)
        var detail = UInt32(0)
        if first_error == 1:
            detail = hploc_status.unsafe_get(error_segment)
        elif first_error == 2:
            detail = collapse_status.unsafe_get(error_segment)
        summary.unsafe_get(4) = detail


struct GpuCompactWideLayout:
    """Device offsets and host allocation sizes from one 20-byte readback."""

    var segment_count: Int
    var node_count: Int
    var leaf_block_count: Int
    var node_segment_offsets: DeviceBuffer[.uint32]
    var leaf_block_segment_offsets: DeviceBuffer[.uint32]

    def __init__(
        out self,
        mut ctx: DeviceContext,
        node_counts: DeviceBuffer[.uint32],
        leaf_block_counts: DeviceBuffer[.uint32],
        segment_count: Int,
        *,
        primitive_offsets: Optional[DeviceBuffer[.uint32]] = Optional[
            DeviceBuffer[.uint32]
        ](),
        node_capacity_offsets: Optional[DeviceBuffer[.uint32]] = Optional[
            DeviceBuffer[.uint32]
        ](),
        hploc_status: Optional[DeviceBuffer[.uint32]] = Optional[
            DeviceBuffer[.uint32]
        ](),
        collapse_status: Optional[DeviceBuffer[.uint32]] = Optional[
            DeviceBuffer[.uint32]
        ](),
        encoded_counts: Optional[DeviceBuffer[.uint32]] = Optional[
            DeviceBuffer[.uint32]
        ](),
        fat_leaves: Bool = True,
    ) raises:
        debug_assert["safe", _use_compiler_assume=True](
            segment_count > 0
            and len(node_counts) >= segment_count
            and len(leaf_block_counts) >= segment_count,
            "compact BVH layout requires nonempty segment count buffers",
        )
        debug_assert["safe", _use_compiler_assume=True](
            Bool(primitive_offsets) == Bool(node_capacity_offsets)
            and (not encoded_counts or primitive_offsets),
            "compact BVH validation requires source and capacity offsets",
        )
        self.segment_count = segment_count
        self.node_segment_offsets = ctx.enqueue_create_buffer[.uint32](
            segment_count + 1
        )
        self.leaf_block_segment_offsets = ctx.enqueue_create_buffer[.uint32](
            segment_count + 1
        )
        var summary = ctx.enqueue_create_buffer[.uint32](5)
        # Absent validation inputs are never dereferenced by the kernel.
        var primitives = (
            primitive_offsets.value().copy() if primitive_offsets else node_counts.copy()
        )
        var capacities = (
            node_capacity_offsets.value().copy() if node_capacity_offsets else node_counts.copy()
        )
        var hploc = (
            hploc_status.value().copy() if hploc_status else node_counts.copy()
        )
        var collapse = (
            collapse_status.value().copy() if collapse_status else node_counts.copy()
        )
        var encoded = (
            encoded_counts.value().copy() if encoded_counts else node_counts.copy()
        )
        debug_assert["safe", _use_compiler_assume=True](
            (
                not primitive_offsets
                or (
                    len(primitives) >= segment_count + 1
                    and len(capacities) >= segment_count + 1
                )
            )
            and (not hploc_status or len(hploc) >= segment_count)
            and (not collapse_status or len(collapse) >= segment_count)
            and (not encoded_counts or len(encoded) >= segment_count),
            "compact BVH validation buffers are too short",
        )
        var flags = UInt32(0)
        if primitive_offsets:
            flags |= _CHECK_SIZES
        if hploc_status:
            flags |= _CHECK_HPLOC
        if collapse_status:
            flags |= _CHECK_COLLAPSE
        if encoded_counts:
            flags |= _CHECK_ENCODING
        if fat_leaves:
            flags |= _FAT_LEAVES
        comptime for size in [32, 256]:
            if (segment_count <= 32) == (size == 32):
                ctx.enqueue_function[finalize_wide_layout_kernel[size]](
                    _device_span[mut=False](node_counts),
                    _device_span[mut=False](leaf_block_counts),
                    _device_span[mut=False](primitives),
                    _device_span[mut=False](capacities),
                    _device_span[mut=False](hploc),
                    _device_span[mut=False](collapse),
                    _device_span[mut=False](encoded),
                    _device_span[mut=True](self.node_segment_offsets),
                    _device_span[mut=True](self.leaf_block_segment_offsets),
                    _device_span[mut=True](summary),
                    Int32(segment_count),
                    flags,
                    grid_dim=1,
                    block_dim=size,
                )
        ctx.synchronize()
        with summary.map_to_host() as host:
            if host[2] != 0:
                var reasons = [
                    "",
                    "H-PLOC build failed",
                    "BVH2-to-wide conversion failed",
                    "empty segment emitted wide output",
                    "conversion emitted an invalid node count",
                    "conversion emitted an invalid leaf count",
                    "CWBVH8 encoding lost triangle records",
                    "compact segment offsets exceed UInt32",
                ]
                raise Error(
                    t"Segment {host[3]}: {reasons[Int(host[2])]} "
                    t"(status {host[4]})"
                )
            self.node_count = Int(host[0])
            self.leaf_block_count = Int(host[1])
