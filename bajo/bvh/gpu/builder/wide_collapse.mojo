from std.atomic import Atomic, Ordering
from std.gpu import block_idx, global_idx, thread_idx
from std.math import ceildiv, fma, max, min
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from bajo.bvh.constants import (
    BinaryBvhNode,
    EMPTY_LANE,
    GPU_BOUNDS_BVH_BLOCK_SIZE,
    LBVH_SENTINEL,
    WideNode,
    f32_max,
)
from bajo.bvh.gpu.wide_layout import (
    GpuWideBoundsBvhBatch,
    _wide_node_store_child,
)
from bajo.bvh.gpu.builder.binary_layout import (
    GpuBinaryBoundsBvh,
    _encoded_bounds,
    _node_left,
    _node_right,
    _segment_for_item,
)
from bajo.bvh.tagged_ref import decode_ref_index, is_leaf_ref
from bajo.bvh.wide_meta import _pack_wide_meta
from bajo.bvh.gpu.utils import _device_span, upload_list
from bajo.bvh.gpu.compressed_bounds_bvh import (
    CWBVH_NODE_WORDS,
    CWBVH_PX,
    CWBVH_PY,
    CWBVH_PZ,
    CWBVH_EXP_IMASK,
    CWBVH_CHILD_BASE,
    CWBVH_TRIANGLE_BASE,
    CWBVH_META_BASE,
    CWBVH_QUANTIZED_BASE,
    _power_of_two_scale_exponent,
    _scale_from_exponent,
    _quantize_lower,
    _quantize_upper,
    _unary_triangle_count,
)
from bajo.core import AABB, Point3f32, SegmentOffsets


comptime HPLOC_WIDE_STATUS_OK = UInt32(0)
comptime HPLOC_WIDE_STATUS_OUT_OF_WORK = UInt32(1)
comptime HPLOC_WIDE_STATUS_OUT_OF_NODES = UInt32(2)
comptime HPLOC_WIDE_STATUS_OUT_OF_LEAVES = UInt32(3)
comptime HPLOC_WIDE_STATUS_TIMEOUT = UInt32(4)
comptime HPLOC_WIDE_NOOP_ENCODED = LBVH_SENTINEL


@always_inline
def _collapse_node_left[compact_hploc: Bool](
    node_meta: ImmPointer[UInt32, _],
    compact_children: ImmPointer[UInt32, _],
    node_idx: UInt32,
) -> UInt32:
    comptime if compact_hploc:
        return compact_children[unsafe_offset=Int(node_idx) * 2]
    return _node_left(node_meta, node_idx)


@always_inline
def _collapse_node_right[compact_hploc: Bool](
    node_meta: ImmPointer[UInt32, _],
    compact_children: ImmPointer[UInt32, _],
    node_idx: UInt32,
) -> UInt32:
    comptime if compact_hploc:
        return compact_children[unsafe_offset=Int(node_idx) * 2 + 1]
    return _node_right(node_meta, node_idx)


@always_inline
def _collapse_encoded_bounds[compact_hploc: Bool](
    encoded: UInt32,
    leaf_bounds: ImmSpan[Float32, _],
    leaf_ids: ImmSpan[UInt32, _],
    node_bounds: ImmSpan[Float32, _],
    compact_bounds: ImmPointer[Float32, _],
) -> AABB[.WORLD]:
    comptime if compact_hploc:
        if is_leaf_ref(encoded):
            var sorted_leaf_idx = decode_ref_index(encoded)
            var item_idx = leaf_ids.unsafe_get(Int(sorted_leaf_idx))
            return AABB[.WORLD].load6(
                leaf_bounds, Int(item_idx) * AABB.STRIDE
            )
        var base = Int(decode_ref_index(encoded)) * AABB.STRIDE
        return AABB[.WORLD](
            Point3f32[.WORLD](
                compact_bounds[unsafe_offset=base],
                compact_bounds[unsafe_offset=base + 1],
                compact_bounds[unsafe_offset=base + 2],
            ),
            Point3f32[.WORLD](
                compact_bounds[unsafe_offset=base + 3],
                compact_bounds[unsafe_offset=base + 4],
                compact_bounds[unsafe_offset=base + 5],
            ),
        )
    return _encoded_bounds(encoded, leaf_bounds, leaf_ids, node_bounds)


def _write_hploc_terminal_primitive_ids[
    compact_hploc: Bool,
](
    encoded: UInt32,
    leaf_payloads: ImmPointer[UInt32, _],
    leaf_ids: ImmPointer[UInt32, _],
    node_meta: ImmPointer[UInt32, _],
    compact_children: ImmPointer[UInt32, _],
    output: MutPointer[UInt32, _],
):
    """Write one terminal subtree's primitive remap."""
    var stack = Array[UInt32, 4](uninitialized=True)
    var stack_size = 1
    var out_count = 0
    stack[0] = encoded
    while stack_size > 0:
        stack_size -= 1
        var current = stack[stack_size]
        if is_leaf_ref(current):
            var sorted_leaf_idx = decode_ref_index(current)
            var item_idx = leaf_ids[unsafe_offset=Int(sorted_leaf_idx)]
            var prim = leaf_payloads[unsafe_offset=Int(item_idx)]
            output[unsafe_offset=out_count] = prim
            out_count += 1
        else:
            var node_idx = decode_ref_index(current)
            stack[stack_size] = _collapse_node_right[compact_hploc](
                node_meta, compact_children, node_idx
            )
            stack_size += 1
            stack[stack_size] = _collapse_node_left[compact_hploc](
                node_meta, compact_children, node_idx
            )
            stack_size += 1


def _write_direct_cwbvh8_node[
    compact_hploc: Bool,
](
    cwbvh_nodes: MutPointer[Float32, _],
    compact_primitive_ids: MutPointer[UInt32, _],
    triangle_counter: MutPointer[UInt32, _],
    leaf_payloads: ImmPointer[UInt32, _],
    leaf_ids: ImmPointer[UInt32, _],
    node_meta: ImmPointer[UInt32, _],
    compact_children: ImmPointer[UInt32, _],
    out_idx: UInt32,
    child_bounds: Array[AABB[.WORLD], 8],
    child_encoded: Array[UInt32, 8],
    child_data: Array[UInt32, 8],
    child_counts: Array[UInt32, 8],
    internal_mask: UInt32,
):
    """Encode one collapsed BVH2 frontier directly as an 80-byte CWBVH8 node."""
    var lo_x = f32_max
    var lo_y = f32_max
    var lo_z = f32_max
    var hi_x = -f32_max
    var hi_y = -f32_max
    var hi_z = -f32_max
    var triangle_count = UInt32(0)
    var child_base = UInt32(0)
    var internal_count = UInt32(0)
    var valid_count = 0
    comptime for lane in range(8):
        var count = child_counts[lane]
        if count == EMPTY_LANE:
            continue
        valid_count += 1
        var bounds = child_bounds[lane]
        lo_x = min(lo_x, bounds._min.x)
        lo_y = min(lo_y, bounds._min.y)
        lo_z = min(lo_z, bounds._min.z)
        hi_x = max(hi_x, bounds._max.x)
        hi_y = max(hi_y, bounds._max.y)
        hi_z = max(hi_z, bounds._max.z)
        if (internal_mask & (UInt32(1) << UInt32(lane))) != 0:
            if internal_count == 0:
                child_base = child_data[lane]
            debug_assert["safe", _use_compiler_assume=True](
                child_data[lane] == child_base + internal_count,
                "direct CWBVH8 children must be contiguous",
            )
            internal_count += 1
        else:
            triangle_count += count
    debug_assert["safe", _use_compiler_assume=True](valid_count > 0)
    var triangle_base = Atomic.fetch_add[ordering=Ordering.RELAXED](
        triangle_counter, triangle_count
    )
    var exponent_x = _power_of_two_scale_exponent(hi_x - lo_x)
    var exponent_y = _power_of_two_scale_exponent(hi_y - lo_y)
    var exponent_z = _power_of_two_scale_exponent(hi_z - lo_z)
    var scale_x = _scale_from_exponent(exponent_x)
    var scale_y = _scale_from_exponent(exponent_y)
    var scale_z = _scale_from_exponent(exponent_z)
    if fma(Float32(255), scale_x, lo_x) < hi_x and exponent_x < UInt32(254):
        exponent_x += 1
        scale_x = _scale_from_exponent(exponent_x)
    if fma(Float32(255), scale_y, lo_y) < hi_y and exponent_y < UInt32(254):
        exponent_y += 1
        scale_y = _scale_from_exponent(exponent_y)
    if fma(Float32(255), scale_z, lo_z) < hi_z and exponent_z < UInt32(254):
        exponent_z += 1
        scale_z = _scale_from_exponent(exponent_z)

    var base = Int(out_idx) * CWBVH_NODE_WORDS
    cwbvh_nodes[unsafe_offset=base + CWBVH_PX] = lo_x
    cwbvh_nodes[unsafe_offset=base + CWBVH_PY] = lo_y
    cwbvh_nodes[unsafe_offset=base + CWBVH_PZ] = lo_z
    var words = cwbvh_nodes.unsafe_bitcast[UInt32]()
    words[unsafe_offset=base + CWBVH_EXP_IMASK] = (
        exponent_x
        | (exponent_y << UInt32(8))
        | (exponent_z << UInt32(16))
        | (internal_mask << UInt32(24))
    )
    words[unsafe_offset=base + CWBVH_CHILD_BASE] = child_base
    words[unsafe_offset=base + CWBVH_TRIANGLE_BASE] = triangle_base

    var leaf_offset = UInt32(0)
    var packed_meta0 = UInt32(0)
    var packed_meta1 = UInt32(0)
    comptime for lane in range(8):
        var count = child_counts[lane]
        var byte = UInt32(0)
        if (internal_mask & (UInt32(1) << UInt32(lane))) != 0:
            byte = UInt32(0x20) | (UInt32(24) + UInt32(lane))
        elif count != EMPTY_LANE:
            byte = _unary_triangle_count(count) | leaf_offset
            _write_hploc_terminal_primitive_ids[compact_hploc](
                child_encoded[lane],
                leaf_payloads,
                leaf_ids,
                node_meta,
                compact_children,
                compact_primitive_ids.unsafe_offset(
                    Int(triangle_base + leaf_offset)
                ),
            )
            leaf_offset += count
        comptime if lane < 4:
            packed_meta0 |= byte << UInt32(lane * 8)
        else:
            packed_meta1 |= byte << UInt32((lane - 4) * 8)
    words[unsafe_offset=base + CWBVH_META_BASE] = packed_meta0
    words[unsafe_offset=base + CWBVH_META_BASE + 1] = packed_meta1

    comptime for plane in range(6):
        comptime for group in range(2):
            var packed = UInt32(0)
            comptime for byte_i in range(4):
                comptime lane = group * 4 + byte_i
                var q = UInt32(0)
                if child_counts[lane] != EMPTY_LANE:
                    var bounds = child_bounds[lane]
                    comptime if plane == 0:
                        q = _quantize_lower(bounds._min.x, lo_x, scale_x)
                    elif plane == 1:
                        q = _quantize_lower(bounds._min.y, lo_y, scale_y)
                    elif plane == 2:
                        q = _quantize_lower(bounds._min.z, lo_z, scale_z)
                    elif plane == 3:
                        q = _quantize_upper(bounds._max.x, lo_x, scale_x)
                    elif plane == 4:
                        q = _quantize_upper(bounds._max.y, lo_y, scale_y)
                    else:
                        q = _quantize_upper(bounds._max.z, lo_z, scale_z)
                packed |= q << UInt32(byte_i * 8)
            words[
                unsafe_offset=base + CWBVH_QUANTIZED_BASE + plane * 2 + group
            ] = packed


@always_inline
def _pack_hploc_wide_pair(encoded: UInt32, out_idx: UInt32) -> UInt64:
    return (UInt64(encoded) << 32) | UInt64(out_idx)


@always_inline
def _hploc_wide_pair_encoded(pair: UInt64) -> UInt32:
    return UInt32(pair >> 32)


@always_inline
def _hploc_wide_pair_out_idx(pair: UInt64) -> UInt32:
    return UInt32(pair)


@always_inline
def _hploc_encoded_leaf_count(
    encoded: UInt32,
    node_leaf_counts: ImmPointer[UInt32, _],
) -> UInt32:
    if is_leaf_ref(encoded):
        return UInt32(1)
    return node_leaf_counts[unsafe_offset=Int(decode_ref_index(encoded))]


comptime HplocWideLeafDataFn = def(
    UInt32,
    UInt32,
    ImmPointer[UInt32, _],
    ImmPointer[UInt32, _],
) thin -> UInt32


@always_inline
def _hploc_leaf_block_data(
    encoded: UInt32,
    leaf_block_idx: UInt32,
    leaf_payloads: ImmPointer[UInt32, _],
    leaf_ids: ImmPointer[UInt32, _],
) -> UInt32:
    return leaf_block_idx


@always_inline
def _hploc_embedded_leaf_payload(
    encoded: UInt32,
    leaf_block_idx: UInt32,
    leaf_payloads: ImmPointer[UInt32, _],
    leaf_ids: ImmPointer[UInt32, _],
) -> UInt32:
    var sorted_leaf_idx = decode_ref_index(encoded)
    var item_idx = leaf_ids[unsafe_offset=Int(sorted_leaf_idx)]
    return leaf_payloads[unsafe_offset=Int(item_idx)]


def _write_hploc_terminal_leaf_block[
    leaf_width: SIMDLength,
](
    encoded: UInt32,
    leaf_payloads: ImmPointer[UInt32, _],
    leaf_ids: ImmPointer[UInt32, _],
    node_meta: ImmPointer[UInt32, _],
    leaf_block_indices: MutPointer[UInt32, _],
    leaf_block_idx: UInt32,
):
    var block_base = Int(leaf_block_idx) * leaf_width
    leaf_block_indices.unsafe_offset(block_base).unsafe_store[width=leaf_width](
        EMPTY_LANE
    )

    # HIPRT retains binary subtrees of at most four triangles as fat leaves.
    # A terminal subtree therefore needs at most leaf_width stack entries.
    var stack = Array[UInt32, leaf_width](uninitialized=True)
    var stack_size = 1
    var out_count = 0
    stack[0] = encoded
    while stack_size > 0:
        stack_size -= 1
        var current = stack[stack_size]
        if is_leaf_ref(current):
            var sorted_leaf_idx = decode_ref_index(current)
            var item_idx = leaf_ids[unsafe_offset=Int(sorted_leaf_idx)]
            leaf_block_indices[
                unsafe_offset=block_base + out_count
            ] = leaf_payloads[unsafe_offset=Int(item_idx)]
            out_count += 1
        else:
            var node_idx = decode_ref_index(current)
            # Push right first so payload order remains left-to-right.
            stack[stack_size] = _node_right(node_meta, node_idx)
            stack_size += 1
            stack[stack_size] = _node_left(node_meta, node_idx)
            stack_size += 1


def init_hploc_literature_wide_kernel(
    segment_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    index_pairs: MutSpan[UInt64, MutAnyOrigin],
    work_alloc_counter: MutSpan[UInt32, MutAnyOrigin],
    work_group_counter: MutSpan[UInt32, MutAnyOrigin],
    leaf_block_counter: MutSpan[UInt32, MutAnyOrigin],
    wide_node_counter: MutSpan[UInt32, MutAnyOrigin],
    status: MutSpan[UInt32, MutAnyOrigin],
):
    var i = global_idx.x
    if i < len(index_pairs):
        index_pairs.unsafe_get(i) = UInt64.MAX
    if i < len(segment_offsets) - 1:
        var leaf_count = segment_offsets.unsafe_get(
            i + 1
        ) - segment_offsets.unsafe_get(i)
        work_alloc_counter.unsafe_get(i) = UInt32(1)
        work_group_counter.unsafe_get(i) = UInt32(0)
        leaf_block_counter.unsafe_get(i) = UInt32(0)
        wide_node_counter.unsafe_get(i) = UInt32(
            1
        ) if leaf_count > 0 else UInt32(0)
        status.unsafe_get(i) = HPLOC_WIDE_STATUS_OK


def publish_hploc_literature_wide_roots_kernel(
    roots: ImmSpan[UInt32, ImmutAnyOrigin],
    segment_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    index_pairs: MutSpan[UInt64, MutAnyOrigin],
):
    var segment_idx = global_idx.x
    if segment_idx >= len(roots):
        return
    var work_base = Int(segment_offsets.unsafe_get(segment_idx))
    if segment_offsets.unsafe_get(segment_idx + 1) == UInt32(work_base):
        return
    index_pairs.unsafe_get(work_base) = _pack_hploc_wide_pair(
        roots.unsafe_get(segment_idx), UInt32(0)
    )


def hploc_literature_to_wide_kernel[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    max_leaf_size: Int,
    fat_leaves: Bool,
    spatial_slots: Bool,
    direct_cwbvh8: Bool,
    compact_hploc: Bool,
    leaf_data_fn: HplocWideLeafDataFn,
    block_size: Int,
](
    leaf_bounds: Pointer[Float32, MutAnyOrigin],
    leaf_payloads: Pointer[UInt32, MutAnyOrigin],
    leaf_ids: Pointer[UInt32, MutAnyOrigin],
    node_meta: Pointer[UInt32, MutAnyOrigin],
    node_bounds: Pointer[Float32, MutAnyOrigin],
    node_leaf_counts: Pointer[UInt32, MutAnyOrigin],
    compact_children: Pointer[UInt32, MutAnyOrigin],
    compact_bounds: Pointer[Float32, MutAnyOrigin],
    segment_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    internal_segment_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    block_segment_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    node_output_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    leaf_output_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    all_index_pairs: Pointer[UInt64, MutAnyOrigin],
    all_work_alloc_counters: Pointer[UInt32, MutAnyOrigin],
    all_work_group_counters: Pointer[UInt32, MutAnyOrigin],
    all_leaf_block_counters: Pointer[UInt32, MutAnyOrigin],
    all_wide_node_counters: Pointer[UInt32, MutAnyOrigin],
    all_status: Pointer[UInt32, MutAnyOrigin],
    all_wide_nodes: Pointer[Float32, MutAnyOrigin],
    all_leaf_block_indices: Pointer[UInt32, MutAnyOrigin],
    all_cwbvh_nodes: Pointer[Float32, MutAnyOrigin],
    all_compact_primitive_ids: Pointer[UInt32, MutAnyOrigin],
    all_triangle_counters: Pointer[UInt32, MutAnyOrigin],
):
    """Paper §3.4 for every segment in one top-down GPU dispatch."""

    comptime assert block_size == GPU_BOUNDS_BVH_BLOCK_SIZE
    comptime if spatial_slots:
        comptime assert node_width == 8
    comptime if direct_cwbvh8:
        comptime assert node_width == 8 and leaf_width == 4
    comptime fat_leaf_limit = min(max_leaf_size, 4)
    var physical_block = block_idx.x
    var segment_idx = _segment_for_item(block_segment_offsets, physical_block)
    var leaf_begin = Int(segment_offsets.unsafe_get(segment_idx))
    var leaf_end = Int(segment_offsets.unsafe_get(segment_idx + 1))
    var leaf_count_int = leaf_end - leaf_begin
    var internal_count = Int(
        internal_segment_offsets.unsafe_get(len(internal_segment_offsets) - 1)
    )
    var total_leaf_count = Int(
        segment_offsets.unsafe_get(len(segment_offsets) - 1)
    )
    var node_output_base = Int(node_output_offsets.unsafe_get(segment_idx))
    var leaf_output_base = Int(leaf_output_offsets.unsafe_get(segment_idx))
    var max_wide_nodes_int = (
        Int(node_output_offsets.unsafe_get(segment_idx + 1)) - node_output_base
    )
    var max_leaf_blocks_int = (
        Int(leaf_output_offsets.unsafe_get(segment_idx + 1)) - leaf_output_base
    )
    var index_pairs = all_index_pairs.unsafe_offset(leaf_begin)
    var work_alloc_counter = all_work_alloc_counters.unsafe_offset(segment_idx)
    var work_group_counter = all_work_group_counters.unsafe_offset(segment_idx)
    var leaf_block_counter = all_leaf_block_counters.unsafe_offset(segment_idx)
    var wide_node_counter = all_wide_node_counters.unsafe_offset(segment_idx)
    var status = all_status.unsafe_offset(segment_idx)
    var wide_nodes = all_wide_nodes.unsafe_offset(
        node_output_base * node_width * WideNode.CHILD_STRIDE
    )
    var leaf_block_indices = all_leaf_block_indices.unsafe_offset(
        leaf_output_base * leaf_width
    )
    var cwbvh_nodes = all_cwbvh_nodes.unsafe_offset(
        node_output_base * CWBVH_NODE_WORDS
    )
    var compact_primitive_ids = all_compact_primitive_ids.unsafe_offset(
        leaf_begin
    )
    var triangle_counter = all_triangle_counters.unsafe_offset(segment_idx)
    var logical_block = stack_allocation[
        1, UInt32, address_space=AddressSpace.SHARED
    ]()
    if thread_idx.x == 0:
        logical_block[unsafe_offset=0] = Atomic.fetch_add[
            ordering=Ordering.RELAXED
        ](work_group_counter, UInt32(1))
    barrier()

    var work_id = (
        Int(logical_block[unsafe_offset=0]) * block_size + thread_idx.x
    )
    if work_id >= leaf_count_int:
        return

    var leaf_bounds_span = Span(
        unsafe_ptr=leaf_bounds, length=total_leaf_count * AABB.STRIDE
    )
    var leaf_ids_span = Span(unsafe_ptr=leaf_ids, length=total_leaf_count)
    var node_bounds_span = Span(
        unsafe_ptr=node_bounds,
        length=max(internal_count, 1) * BinaryBvhNode.BOUNDS_STRIDE,
    )

    def load_encoded_bounds(encoded_ref: UInt32) {imm} -> AABB[.WORLD]:
        return _collapse_encoded_bounds[compact_hploc](
            encoded_ref,
            leaf_bounds_span,
            leaf_ids_span,
            node_bounds_span,
            compact_bounds,
        )

    var failsafe = 10000000

    while True:
        if Atomic.load[ordering=Ordering.ACQUIRE](status) != (
            HPLOC_WIDE_STATUS_OK
        ):
            return

        var pair = Atomic.load[ordering=Ordering.ACQUIRE](
            index_pairs.unsafe_offset(work_id)
        )
        if pair == UInt64.MAX:
            failsafe -= 1
            if failsafe == 0:
                Atomic.store[ordering=Ordering.RELEASE](
                    status, HPLOC_WIDE_STATUS_TIMEOUT
                )
                return
            continue

        var encoded = _hploc_wide_pair_encoded(pair)
        var out_idx = _hploc_wide_pair_out_idx(pair)
        if encoded == HPLOC_WIDE_NOOP_ENCODED:
            return

        var encoded_leaf_count = _hploc_encoded_leaf_count(
            encoded, node_leaf_counts
        )
        var terminal = is_leaf_ref(encoded)
        comptime if fat_leaves:
            terminal = encoded_leaf_count <= UInt32(fat_leaf_limit)
        if terminal:
            if Int(out_idx) >= max_leaf_blocks_int:
                Atomic.store[ordering=Ordering.RELEASE](
                    status, HPLOC_WIDE_STATUS_OUT_OF_LEAVES
                )
                return
            comptime if not direct_cwbvh8:
                _write_hploc_terminal_leaf_block[leaf_width](
                    encoded,
                    leaf_payloads,
                    leaf_ids,
                    node_meta,
                    leaf_block_indices,
                    out_idx,
                )
            # The root has no parent task to publish the no-op jobs that make
            # one fixed paper work slot available per primitive.
            if work_id == 0 and encoded_leaf_count == UInt32(leaf_count_int):
                Atomic.store[ordering=Ordering.RELEASE](
                    leaf_block_counter, UInt32(1)
                )
                var root_leaf_data = leaf_data_fn(
                    encoded, UInt32(0), leaf_payloads, leaf_ids
                )
                comptime if direct_cwbvh8:
                    var direct_bounds = Array[AABB[.WORLD], 8](
                        fill=AABB[.WORLD].invalid()
                    )
                    var direct_encoded = Array[UInt32, 8](fill=UInt32(0))
                    var direct_data = Array[UInt32, 8](fill=UInt32(0))
                    var direct_counts = Array[UInt32, 8](fill=EMPTY_LANE)
                    direct_bounds[0] = _collapse_encoded_bounds[compact_hploc](
                        encoded,
                        leaf_bounds_span,
                        leaf_ids_span,
                        node_bounds_span,
                        compact_bounds,
                    )
                    direct_encoded[0] = encoded
                    direct_data[0] = root_leaf_data
                    direct_counts[0] = encoded_leaf_count
                    _write_direct_cwbvh8_node[compact_hploc](
                        cwbvh_nodes,
                        compact_primitive_ids,
                        triangle_counter,
                        leaf_payloads,
                        leaf_ids,
                        node_meta,
                        compact_children,
                        UInt32(0),
                        direct_bounds,
                        direct_encoded,
                        direct_data,
                        direct_counts,
                        UInt32(0),
                    )
                else:
                    _wide_node_store_child[node_width](
                        wide_nodes,
                        UInt32(0),
                        0,
                        _collapse_encoded_bounds[compact_hploc](
                            encoded,
                            leaf_bounds_span,
                            leaf_ids_span,
                            node_bounds_span,
                            compact_bounds,
                        ),
                        _pack_wide_meta(root_leaf_data, encoded_leaf_count),
                    )
                    comptime for lane in range(1, node_width):
                        _wide_node_store_child[node_width](
                            wide_nodes,
                            UInt32(0),
                            lane,
                            AABB[.WORLD].invalid(),
                            _pack_wide_meta(UInt32(0), EMPTY_LANE),
                        )
                for noop_id in range(1, leaf_count_int):
                    Atomic.store[ordering=Ordering.RELEASE](
                        index_pairs.unsafe_offset(noop_id),
                        _pack_hploc_wide_pair(
                            HPLOC_WIDE_NOOP_ENCODED, UInt32(0)
                        ),
                    )
            return

        if Int(out_idx) >= max_wide_nodes_int:
            Atomic.store[ordering=Ordering.RELEASE](
                status, HPLOC_WIDE_STATUS_OUT_OF_NODES
            )
            return

        var candidates = Array[UInt32, node_width](uninitialized=True)
        var candidate_leaf_counts = Array[UInt32, node_width](
            uninitialized=True
        )
        var candidate_areas = Array[Float32, node_width](uninitialized=True)
        var node_idx = decode_ref_index(encoded)
        candidates[0] = _collapse_node_left[compact_hploc](
            node_meta, compact_children, node_idx
        )
        candidates[1] = _collapse_node_right[compact_hploc](
            node_meta, compact_children, node_idx
        )
        comptime if fat_leaves:
            candidate_leaf_counts[0] = _hploc_encoded_leaf_count(
                candidates[0], node_leaf_counts
            )
            candidate_leaf_counts[1] = _hploc_encoded_leaf_count(
                candidates[1], node_leaf_counts
            )
        comptime if spatial_slots:
            candidate_areas[0] = load_encoded_bounds(
                candidates[0]
            ).surface_area()[0]
            candidate_areas[1] = load_encoded_bounds(
                candidates[1]
            ).surface_area()[0]
        var child_count = 2

        # HIPRT first opens non-fat internal nodes using the paper's
        # largest-area rule. This prevents an oversized subtree from becoming
        # a leaf merely because the wide node filled.
        while child_count < node_width:
            var open_pos = -1
            var largest_area = Float32(-1.0)
            for candidate_pos in range(child_count):
                var candidate = candidates[candidate_pos]
                if is_leaf_ref(candidate):
                    continue
                comptime if fat_leaves:
                    if candidate_leaf_counts[candidate_pos] <= UInt32(
                        fat_leaf_limit
                    ):
                        continue
                var area: Float32
                comptime if spatial_slots:
                    area = candidate_areas[candidate_pos]
                else:
                    area = load_encoded_bounds(candidate).surface_area()[0]
                if area > largest_area:
                    largest_area = area
                    open_pos = candidate_pos

            if open_pos < 0:
                break
            var opened_idx = decode_ref_index(candidates[open_pos])
            candidates[open_pos] = _collapse_node_left[compact_hploc](
                node_meta, compact_children, opened_idx
            )
            candidates[child_count] = _collapse_node_right[compact_hploc](
                node_meta, compact_children, opened_idx
            )
            comptime if fat_leaves:
                candidate_leaf_counts[open_pos] = _hploc_encoded_leaf_count(
                    candidates[open_pos], node_leaf_counts
                )
                candidate_leaf_counts[child_count] = _hploc_encoded_leaf_count(
                    candidates[child_count], node_leaf_counts
                )
            comptime if spatial_slots:
                candidate_areas[open_pos] = load_encoded_bounds(
                    candidates[open_pos]
                ).surface_area()[0]
                candidate_areas[child_count] = load_encoded_bounds(
                    candidates[child_count]
                ).surface_area()[0]
            child_count += 1

        # Once only legal fat leaves remain, open the largest ones until the
        # node is full. This is HIPRT's second collapse pass.
        comptime if fat_leaves:
            while child_count < node_width:
                var open_pos = -1
                var largest_area = Float32(-1.0)
                for candidate_pos in range(child_count):
                    var candidate = candidates[candidate_pos]
                    if is_leaf_ref(candidate):
                        continue
                    var area: Float32
                    comptime if spatial_slots:
                        area = candidate_areas[candidate_pos]
                    else:
                        area = load_encoded_bounds(candidate).surface_area()[0]
                    if area > largest_area:
                        largest_area = area
                        open_pos = candidate_pos

                if open_pos < 0:
                    break
                var opened_idx = decode_ref_index(candidates[open_pos])
                candidates[open_pos] = _collapse_node_left[compact_hploc](
                    node_meta, compact_children, opened_idx
                )
                candidates[child_count] = _collapse_node_right[compact_hploc](
                    node_meta, compact_children, opened_idx
                )
                candidate_leaf_counts[open_pos] = _hploc_encoded_leaf_count(
                    candidates[open_pos], node_leaf_counts
                )
                candidate_leaf_counts[child_count] = _hploc_encoded_leaf_count(
                    candidates[child_count], node_leaf_counts
                )
                comptime if spatial_slots:
                    candidate_areas[open_pos] = load_encoded_bounds(
                        candidates[open_pos]
                    ).surface_area()[0]
                    candidate_areas[child_count] = load_encoded_bounds(
                        candidates[child_count]
                    ).surface_area()[0]
                child_count += 1

        # CWBVH traversal encodes ray-octant order in physical child slots.
        # Greedily assign candidates to the eight spatial slots exactly as in
        # the reference converter. Node ids are allocated later in ascending
        # slot order, preserving CWBVH's compact child-base+imask addressing.
        var slot_candidate = Array[Int, node_width](fill=-1)
        comptime if spatial_slots:
            var candidate_assigned_mask = UInt32(0)
            var occupied_slot_mask = UInt32(0)
            var candidate_centers = Array[SIMD[.float32, 4], node_width](
                uninitialized=True
            )
            var parent_center = load_encoded_bounds(encoded).centroid()
            for candidate_pos in range(child_count):
                var center = load_encoded_bounds(
                    candidates[candidate_pos]
                ).centroid()
                candidate_centers[candidate_pos] = SIMD[.float32, 4](
                    center.x, center.y, center.z, 0.0
                )
            for _ in range(child_count):
                var best_cost = f32_max
                var best_candidate = -1
                var best_slot = -1
                for candidate_pos in range(child_count):
                    var candidate_bit = UInt32(1) << UInt32(candidate_pos)
                    if (candidate_assigned_mask & candidate_bit) != 0:
                        continue
                    var candidate_center = candidate_centers[candidate_pos]
                    comptime for slot in range(node_width):
                        var slot_bit = UInt32(1) << UInt32(slot)
                        if (occupied_slot_mask & slot_bit) != 0:
                            continue
                        var sx = Float32(1.0)
                        var sy = Float32(1.0)
                        var sz = Float32(1.0)
                        if (slot & 4) != 0:
                            sx = -1.0
                        if (slot & 2) != 0:
                            sy = -1.0
                        if (slot & 1) != 0:
                            sz = -1.0
                        var cost = (
                            (candidate_center[0] - parent_center.x) * sx
                            + (candidate_center[1] - parent_center.y) * sy
                            + (candidate_center[2] - parent_center.z) * sz
                        )
                        if cost < best_cost:
                            best_cost = cost
                            best_candidate = candidate_pos
                            best_slot = slot
                debug_assert["safe", _use_compiler_assume=True](
                    best_candidate >= 0 and best_slot >= 0,
                    "CWBVH spatial child assignment failed",
                )
                candidate_assigned_mask |= UInt32(1) << UInt32(best_candidate)
                occupied_slot_mask |= UInt32(1) << UInt32(best_slot)
                slot_candidate[best_slot] = best_candidate
        else:
            for candidate_pos in range(child_count):
                slot_candidate[candidate_pos] = candidate_pos

        var inner_count = 0
        var leaf_count_in_node = 0
        var published_work_count = 0
        for child_pos in range(child_count):
            var candidate = candidates[child_pos]
            var candidate_leaf_count: UInt32
            comptime if fat_leaves:
                candidate_leaf_count = candidate_leaf_counts[child_pos]
            else:
                candidate_leaf_count = _hploc_encoded_leaf_count(
                    candidate, node_leaf_counts
                )
            candidate_leaf_counts[child_pos] = candidate_leaf_count
            var child_is_leaf = is_leaf_ref(candidate)
            comptime if fat_leaves:
                child_is_leaf = candidate_leaf_count <= UInt32(fat_leaf_limit)
            if child_is_leaf:
                leaf_count_in_node += 1
                published_work_count += Int(candidate_leaf_count)
            else:
                inner_count += 1
                published_work_count += 1

        var child_node_base = Atomic.fetch_add[ordering=Ordering.RELAXED](
            wide_node_counter, UInt32(inner_count)
        )
        var leaf_block_base = Atomic.fetch_add[ordering=Ordering.RELAXED](
            leaf_block_counter, UInt32(leaf_count_in_node)
        )
        var work_base = Atomic.fetch_add[ordering=Ordering.RELAXED](
            work_alloc_counter, UInt32(published_work_count - 1)
        )
        if (
            Int(child_node_base) + inner_count > max_wide_nodes_int
            or Int(leaf_block_base) + leaf_count_in_node > max_leaf_blocks_int
            or Int(work_base) + published_work_count - 1 > leaf_count_int
        ):
            var error = HPLOC_WIDE_STATUS_OUT_OF_WORK
            if Int(child_node_base) + inner_count > max_wide_nodes_int:
                error = HPLOC_WIDE_STATUS_OUT_OF_NODES
            elif (
                Int(leaf_block_base) + leaf_count_in_node > max_leaf_blocks_int
            ):
                error = HPLOC_WIDE_STATUS_OUT_OF_LEAVES
            Atomic.store[ordering=Ordering.RELEASE](status, error)
            return

        var inner_rank = UInt32(0)
        var leaf_rank = UInt32(0)
        var publish_pos = 0
        var direct_bounds = Array[AABB[.WORLD], 8](
            fill=AABB[.WORLD].invalid()
        )
        var direct_encoded = Array[UInt32, 8](fill=UInt32(0))
        var direct_data = Array[UInt32, 8](fill=UInt32(0))
        var direct_counts = Array[UInt32, 8](fill=EMPTY_LANE)
        var direct_internal_mask = UInt32(0)
        comptime for child_slot in range(node_width):
            var child_pos = slot_candidate[child_slot]
            if child_pos < 0:
                comptime if not direct_cwbvh8:
                    _wide_node_store_child[node_width](
                        wide_nodes,
                        out_idx,
                        child_slot,
                        AABB[.WORLD].invalid(),
                        _pack_wide_meta(UInt32(0), EMPTY_LANE),
                    )
            else:
                var child = candidates[child_pos]
                var child_is_leaf = is_leaf_ref(child)
                comptime if fat_leaves:
                    child_is_leaf = candidate_leaf_counts[child_pos] <= UInt32(
                        fat_leaf_limit
                    )
                var child_out_idx: UInt32
                var meta: UInt32
                if child_is_leaf:
                    child_out_idx = leaf_block_base + leaf_rank
                    leaf_rank += 1
                    var leaf_data = leaf_data_fn(
                        child, child_out_idx, leaf_payloads, leaf_ids
                    )
                    meta = _pack_wide_meta(
                        leaf_data, candidate_leaf_counts[child_pos]
                    )
                else:
                    child_out_idx = child_node_base + inner_rank
                    inner_rank += 1
                    meta = _pack_wide_meta(child_out_idx, UInt32(0))

                var bounds = load_encoded_bounds(child)
                comptime if direct_cwbvh8:
                    direct_bounds[child_slot] = bounds
                    direct_encoded[child_slot] = child
                    direct_data[child_slot] = child_out_idx
                    direct_counts[child_slot] = candidate_leaf_counts[
                        child_pos
                    ]
                    if not child_is_leaf:
                        direct_internal_mask |= UInt32(1) << UInt32(
                            child_slot
                        )
                else:
                    _wide_node_store_child[node_width](
                        wide_nodes,
                        out_idx,
                        child_slot,
                        bounds,
                        meta,
                    )

                var child_work_id = work_id
                if publish_pos > 0:
                    child_work_id = Int(work_base) + publish_pos - 1
                Atomic.store[ordering=Ordering.RELEASE](
                    index_pairs.unsafe_offset(child_work_id),
                    _pack_hploc_wide_pair(child, child_out_idx),
                )
                publish_pos += 1

                # The fixed paper work array still owns one slot per primitive.
                # A packed k-triangle leaf therefore publishes k-1 no-op jobs.
                if child_is_leaf:
                    var noops = Int(candidate_leaf_counts[child_pos]) - 1
                    for _ in range(noops):
                        var noop_work_id = Int(work_base) + publish_pos - 1
                        Atomic.store[ordering=Ordering.RELEASE](
                            index_pairs.unsafe_offset(noop_work_id),
                            _pack_hploc_wide_pair(
                                HPLOC_WIDE_NOOP_ENCODED, UInt32(0)
                            ),
                        )
                        publish_pos += 1

        comptime if direct_cwbvh8:
            _write_direct_cwbvh8_node[compact_hploc](
                cwbvh_nodes,
                compact_primitive_ids,
                triangle_counter,
                leaf_payloads,
                leaf_ids,
                node_meta,
                compact_children,
                out_idx,
                direct_bounds,
                direct_encoded,
                direct_data,
                direct_counts,
                direct_internal_mask,
            )


@fieldwise_init
struct GpuWideCollapseState:
    """Owns device work queues until asynchronous wide conversion completes."""

    var segments: SegmentOffsets
    var index_pairs: DeviceBuffer[.uint64]
    var work_alloc_counter: DeviceBuffer[.uint32]
    var work_group_counter: DeviceBuffer[.uint32]
    var leaf_block_counter: DeviceBuffer[.uint32]
    var wide_node_counter: DeviceBuffer[.uint32]
    var status: DeviceBuffer[.uint32]

    def finish_batch_synchronized[
        node_width: SIMDLength,
        leaf_width: SIMDLength,
        max_leaf_size: Int,
    ](
        self,
        tree: GpuWideBoundsBvhBatch[node_width, leaf_width, max_leaf_size],
        fat_leaves: Bool,
    ) raises:
        with self.leaf_block_counter.map_to_host() as leaves, self.wide_node_counter.map_to_host() as nodes, self.status.map_to_host() as build_status:
            for segment_idx in range(self.segments.segment_count()):
                if build_status[segment_idx] != HPLOC_WIDE_STATUS_OK:
                    raise String(
                        t"Segment {segment_idx} BVH2-to-wide status:"
                        t" {build_status[segment_idx]}"
                    )
                var source_count = Int(self.segments.count(segment_idx))
                if source_count == 0:
                    if (
                        Int(nodes[segment_idx]) != 0
                        or Int(leaves[segment_idx]) != 0
                    ):
                        raise "empty segment emitted wide output"
                    continue
                if Int(nodes[segment_idx]) <= 0 or Int(
                    nodes[segment_idx]
                ) > Int(tree.node_segments.count(segment_idx)):
                    raise "segmented conversion emitted an invalid node count"
                if fat_leaves:
                    if (
                        Int(leaves[segment_idx]) <= 0
                        or Int(leaves[segment_idx]) > source_count
                    ):
                        raise "fat-leaf conversion emitted an invalid leaf count"
                elif Int(leaves[segment_idx]) != source_count:
                    raise "§3.4 conversion did not emit one leaf per primitive"

    def wait_batch[
        node_width: SIMDLength,
        leaf_width: SIMDLength,
        max_leaf_size: Int,
    ](
        self,
        ctx: DeviceContext,
        tree: GpuWideBoundsBvhBatch[node_width, leaf_width, max_leaf_size],
        fat_leaves: Bool,
    ) raises:
        ctx.synchronize()
        self.finish_batch_synchronized(tree, fat_leaves)


struct GpuWideCollapseWorkspace:
    """Reusable queues and counters for one fixed segmented workload."""

    var leaf_capacity: Int
    var segments: SegmentOffsets
    var block_segments: SegmentOffsets
    var block_segment_offsets: DeviceBuffer[.uint32]
    var index_pairs: DeviceBuffer[.uint64]
    var work_alloc_counter: DeviceBuffer[.uint32]
    var work_group_counter: DeviceBuffer[.uint32]
    var wide_node_counter: DeviceBuffer[.uint32]
    var status: DeviceBuffer[.uint32]

    def __init__(
        out self,
        mut ctx: DeviceContext,
        segments: SegmentOffsets,
    ) raises:
        var leaf_capacity = segments.item_count()
        debug_assert["safe", _use_compiler_assume=True](
            leaf_capacity > 0, "wide workspace capacity must be positive"
        )
        var block_counts = List[Int](capacity=segments.segment_count())
        for segment_idx in range(segments.segment_count()):
            var leaf_count = Int(segments.count(segment_idx))
            block_counts.append(ceildiv(leaf_count, GPU_BOUNDS_BVH_BLOCK_SIZE))
        self.leaf_capacity = leaf_capacity
        self.segments = segments.copy()
        self.block_segments = SegmentOffsets.from_counts(block_counts^)
        self.block_segment_offsets = upload_list(
            ctx, self.block_segments.offsets
        )
        self.index_pairs = ctx.enqueue_create_buffer[.uint64](
            leaf_capacity
        )
        self.work_alloc_counter = ctx.enqueue_create_buffer[.uint32](
            segments.segment_count()
        )
        self.work_group_counter = ctx.enqueue_create_buffer[.uint32](
            segments.segment_count()
        )
        self.wide_node_counter = ctx.enqueue_create_buffer[.uint32](
            segments.segment_count()
        )
        self.status = ctx.enqueue_create_buffer[.uint32](
            segments.segment_count()
        )


def _enqueue_collapse_binary_to_packed[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    max_leaf_size: Int,
    fat_leaves: Bool,
    spatial_slots: Bool,
    leaf_data_fn: HplocWideLeafDataFn,
](
    mut ctx: DeviceContext,
    binary: GpuBinaryBoundsBvh,
    node_segments: SegmentOffsets,
    node_segment_offsets: DeviceBuffer[.uint32],
    leaf_block_segments: SegmentOffsets,
    leaf_block_segment_offsets: DeviceBuffer[.uint32],
    wide_nodes: DeviceBuffer[.float32],
    leaf_block_indices: DeviceBuffer[.uint32],
    leaf_block_counter_buffer: DeviceBuffer[.uint32],
    wide_node_counter_buffer: DeviceBuffer[.uint32],
    workspace: GpuWideCollapseWorkspace,
) raises -> GpuWideCollapseState:
    debug_assert["safe", _use_compiler_assume=True](
        workspace.leaf_capacity == binary.leaf_count
        and workspace.segments.segment_count()
        == binary.segments.segment_count(),
        "wide workspace does not match the segmented input",
    )
    debug_assert["safe", _use_compiler_assume=True](
        node_segments.segment_count() == binary.segments.segment_count()
        and leaf_block_segments.segment_count()
        == binary.segments.segment_count(),
        "wide output ranges do not match the segmented input",
    )
    var index_pairs = workspace.index_pairs.copy()
    var work_alloc_counter = workspace.work_alloc_counter.copy()
    var work_group_counter = workspace.work_group_counter.copy()
    var leaf_block_counter = leaf_block_counter_buffer.copy()
    var wide_node_counter = wide_node_counter_buffer.copy()
    var status = workspace.status.copy()
    var init_items = max(binary.leaf_count, binary.segments.segment_count())

    ctx.enqueue_function[init_hploc_literature_wide_kernel](
        _device_span[mut=False](binary.segment_offsets),
        _device_span[mut=True](index_pairs),
        _device_span[mut=True](work_alloc_counter),
        _device_span[mut=True](work_group_counter),
        _device_span[mut=True](leaf_block_counter),
        _device_span[mut=True](wide_node_counter),
        _device_span[mut=True](status),
        grid_dim=ceildiv(init_items, GPU_BOUNDS_BVH_BLOCK_SIZE),
        block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
    )
    ctx.enqueue_function[publish_hploc_literature_wide_roots_kernel](
        _device_span[mut=False](binary.roots),
        _device_span[mut=False](binary.segment_offsets),
        _device_span[mut=True](index_pairs),
        grid_dim=ceildiv(
            binary.segments.segment_count(), GPU_BOUNDS_BVH_BLOCK_SIZE
        ),
        block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
    )
    ctx.enqueue_function[
        hploc_literature_to_wide_kernel[
            node_width,
            leaf_width,
            max_leaf_size,
            fat_leaves,
            spatial_slots,
            False,
            False,
            leaf_data_fn,
            GPU_BOUNDS_BVH_BLOCK_SIZE,
        ]
    ](
        binary.leaf_bounds,
        binary.leaf_payloads,
        binary.leaf_ids,
        binary.node_meta,
        binary.node_bounds,
        binary.node_leaf_counts,
        binary.node_meta,
        binary.node_bounds,
        _device_span[mut=False](binary.segment_offsets),
        _device_span[mut=False](binary.internal_segment_offsets),
        _device_span[mut=False](workspace.block_segment_offsets),
        _device_span[mut=False](node_segment_offsets),
        _device_span[mut=False](leaf_block_segment_offsets),
        index_pairs,
        work_alloc_counter,
        work_group_counter,
        leaf_block_counter,
        wide_node_counter,
        status,
        wide_nodes,
        leaf_block_indices,
        wide_nodes,
        leaf_block_indices,
        leaf_block_counter,
        grid_dim=workspace.block_segments.item_count(),
        block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
    )
    return GpuWideCollapseState(
        binary.segments.copy(),
        index_pairs^,
        work_alloc_counter^,
        work_group_counter^,
        leaf_block_counter^,
        wide_node_counter^,
        status^,
    )


def enqueue_collapse_binary_to_wide_batch[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    max_leaf_size: Int,
    fat_leaves: Bool = False,
    spatial_slots: Bool = False,
](
    mut ctx: DeviceContext,
    binary: GpuBinaryBoundsBvh,
    mut out: GpuWideBoundsBvhBatch[node_width, leaf_width, max_leaf_size],
) raises -> GpuWideCollapseState:
    return _enqueue_collapse_binary_to_wide_batch[
        node_width,
        leaf_width,
        max_leaf_size,
        fat_leaves,
        spatial_slots,
        _hploc_leaf_block_data,
    ](ctx, binary, out)


def enqueue_collapse_binary_to_cwbvh8[
    max_leaf_size: Int,
    compact_hploc: Bool = False,
](
    mut ctx: DeviceContext,
    binary: GpuBinaryBoundsBvh,
    node_segments: SegmentOffsets,
    node_segment_offsets: DeviceBuffer[.uint32],
    leaf_block_segments: SegmentOffsets,
    leaf_block_segment_offsets: DeviceBuffer[.uint32],
    leaf_block_counter_buffer: DeviceBuffer[.uint32],
    wide_node_counter_buffer: DeviceBuffer[.uint32],
    cwbvh_nodes: DeviceBuffer[.float32],
    compact_primitive_ids: DeviceBuffer[.uint32],
    triangle_counters: DeviceBuffer[.uint32],
    compact_children: DeviceBuffer[.uint32],
    compact_bounds: DeviceBuffer[.float32],
    workspace: GpuWideCollapseWorkspace,
) raises -> GpuWideCollapseState:
    """Collapse BVH2 frontiers directly into compressed CWBVH8 nodes."""
    comptime assert 1 <= max_leaf_size <= 3
    var index_pairs = workspace.index_pairs.copy()
    var work_alloc_counter = workspace.work_alloc_counter.copy()
    var work_group_counter = workspace.work_group_counter.copy()
    var leaf_block_counter = leaf_block_counter_buffer.copy()
    var wide_node_counter = wide_node_counter_buffer.copy()
    var status = workspace.status.copy()
    ctx.enqueue_memset(triangle_counters, 0)
    var init_items = max(binary.leaf_count, binary.segments.segment_count())
    ctx.enqueue_function[init_hploc_literature_wide_kernel](
        _device_span[mut=False](binary.segment_offsets),
        _device_span[mut=True](index_pairs),
        _device_span[mut=True](work_alloc_counter),
        _device_span[mut=True](work_group_counter),
        _device_span[mut=True](leaf_block_counter),
        _device_span[mut=True](wide_node_counter),
        _device_span[mut=True](status),
        grid_dim=ceildiv(init_items, GPU_BOUNDS_BVH_BLOCK_SIZE),
        block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
    )
    ctx.enqueue_function[publish_hploc_literature_wide_roots_kernel](
        _device_span[mut=False](binary.roots),
        _device_span[mut=False](binary.segment_offsets),
        _device_span[mut=True](index_pairs),
        grid_dim=ceildiv(
            binary.segments.segment_count(), GPU_BOUNDS_BVH_BLOCK_SIZE
        ),
        block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
    )
    ctx.enqueue_function[
        hploc_literature_to_wide_kernel[
            8,
            4,
            max_leaf_size,
            True,
            True,
            True,
            compact_hploc,
            _hploc_leaf_block_data,
            GPU_BOUNDS_BVH_BLOCK_SIZE,
        ]
    ](
        binary.leaf_bounds,
        binary.leaf_payloads,
        binary.leaf_ids,
        binary.node_meta,
        binary.node_bounds,
        binary.node_leaf_counts,
        compact_children,
        compact_bounds,
        _device_span[mut=False](binary.segment_offsets),
        _device_span[mut=False](binary.internal_segment_offsets),
        _device_span[mut=False](workspace.block_segment_offsets),
        _device_span[mut=False](node_segment_offsets),
        _device_span[mut=False](leaf_block_segment_offsets),
        index_pairs,
        work_alloc_counter,
        work_group_counter,
        leaf_block_counter,
        wide_node_counter,
        status,
        cwbvh_nodes,
        compact_primitive_ids,
        cwbvh_nodes,
        compact_primitive_ids,
        triangle_counters,
        grid_dim=workspace.block_segments.item_count(),
        block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
    )
    return GpuWideCollapseState(
        binary.segments.copy(),
        index_pairs^,
        work_alloc_counter^,
        work_group_counter^,
        leaf_block_counter^,
        wide_node_counter^,
        status^,
    )


def _enqueue_collapse_binary_to_wide_batch[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    max_leaf_size: Int,
    fat_leaves: Bool,
    spatial_slots: Bool,
    leaf_data_fn: HplocWideLeafDataFn,
](
    mut ctx: DeviceContext,
    binary: GpuBinaryBoundsBvh,
    mut out: GpuWideBoundsBvhBatch[node_width, leaf_width, max_leaf_size],
) raises -> GpuWideCollapseState:
    var workspace = GpuWideCollapseWorkspace(ctx, binary.segments)
    out.bounds_device = binary.bounds_device.copy()
    return _enqueue_collapse_binary_to_packed[
        node_width,
        leaf_width,
        max_leaf_size,
        fat_leaves,
        spatial_slots,
        leaf_data_fn,
    ](
        ctx,
        binary,
        out.node_segments,
        out.node_segment_offsets,
        out.leaf_block_segments,
        out.leaf_block_segment_offsets,
        out.wide_nodes,
        out.leaf_block_indices,
        out.leaf_block_counts,
        out.node_counts,
        workspace,
    )


def collapse_binary_to_wide_batch[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    max_leaf_size: Int,
    fat_leaves: Bool = False,
    spatial_slots: Bool = False,
](
    mut ctx: DeviceContext,
    binary: GpuBinaryBoundsBvh,
    mut out: GpuWideBoundsBvhBatch[node_width, leaf_width, max_leaf_size],
) raises:
    var pending = enqueue_collapse_binary_to_wide_batch[
        node_width, leaf_width, max_leaf_size, fat_leaves, spatial_slots
    ](ctx, binary, out)
    pending.wait_batch(ctx, out, fat_leaves)
