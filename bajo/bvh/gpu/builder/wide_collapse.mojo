from std.atomic import Atomic, Ordering
from std.gpu import global_idx, thread_idx
from std.math import ceildiv, max, min
from std.memory import stack_allocation
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from bajo.bvh.constants import (
    BinaryBvhNode,
    EMPTY_LANE,
    GPU_BOUNDS_BVH_BLOCK_SIZE,
    LBVH_SENTINEL,
    WideNode,
)
from bajo.bvh.gpu.wide_layout import (
    GpuWideBoundsBvh,
    _wide_node_store_child,
)
from bajo.bvh.gpu.builder.binary_layout import (
    GpuBinaryBoundsBvh,
    _encoded_bounds,
    _node_left,
    _node_parent_index,
    _node_right,
)
from bajo.bvh.tagged_ref import decode_ref_index, is_leaf_ref
from bajo.bvh.gpu.wide_meta import _pack_wide_meta
from bajo.core import AABB, Frame


comptime HPLOC_WIDE_STATUS_OK = UInt32(0)
comptime HPLOC_WIDE_STATUS_OUT_OF_WORK = UInt32(1)
comptime HPLOC_WIDE_STATUS_OUT_OF_NODES = UInt32(2)
comptime HPLOC_WIDE_STATUS_OUT_OF_LEAVES = UInt32(3)
comptime HPLOC_WIDE_STATUS_TIMEOUT = UInt32(4)
comptime HPLOC_WIDE_NOOP_ENCODED = LBVH_SENTINEL


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
    node_leaf_counts: Pointer[mut=False, UInt32, _],
) -> UInt32:
    if is_leaf_ref(encoded):
        return UInt32(1)
    return node_leaf_counts[unsafe_offset=Int(decode_ref_index(encoded))]


def _write_hploc_terminal_leaf_block[
    leaf_width: SIMDLength,
](
    encoded: UInt32,
    leaf_payloads: Pointer[mut=False, UInt32, _],
    leaf_ids: Pointer[mut=False, UInt32, _],
    node_meta: Pointer[mut=False, UInt32, _],
    leaf_block_indices: Pointer[mut=True, UInt32, _],
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
    node_meta: Pointer[UInt32, MutAnyOrigin],
    index_pairs: Pointer[UInt64, MutAnyOrigin],
    work_alloc_counter: Pointer[UInt32, MutAnyOrigin],
    work_group_counter: Pointer[UInt32, MutAnyOrigin],
    leaf_block_counter: Pointer[UInt32, MutAnyOrigin],
    wide_node_counter: Pointer[UInt32, MutAnyOrigin],
    status: Pointer[UInt32, MutAnyOrigin],
    internal_count: Int32,
    slot_count: Int32,
):
    var i = global_idx.x
    var internal_count_int = Int(internal_count)
    var slot_count_int = Int(slot_count)
    # Slot zero is exclusively published by the thread that finds the root.
    # Clearing it here as well would race that publication across the grid.
    if i > 0 and i < slot_count_int:
        index_pairs[unsafe_offset=i] = UInt64.MAX

    if i == 0:
        work_alloc_counter[unsafe_offset=0] = UInt32(1)
        work_group_counter[unsafe_offset=0] = UInt32(0)
        leaf_block_counter[unsafe_offset=0] = UInt32(0)
        wide_node_counter[unsafe_offset=0] = UInt32(1)
        status[unsafe_offset=0] = HPLOC_WIDE_STATUS_OK

    if i >= internal_count_int:
        return
    var node_idx = UInt32(i)
    if node_meta[unsafe_offset=_node_parent_index(node_idx)] == LBVH_SENTINEL:
        index_pairs[unsafe_offset=0] = _pack_hploc_wide_pair(
            node_idx, UInt32(0)
        )


def hploc_literature_wide_single_leaf_kernel[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
](
    leaf_bounds: Pointer[Float32, ImmutAnyOrigin],
    leaf_payloads: Pointer[UInt32, ImmutAnyOrigin],
    leaf_ids: Pointer[UInt32, ImmutAnyOrigin],
    wide_nodes: Pointer[Float32, MutAnyOrigin],
    leaf_block_indices: Pointer[UInt32, MutAnyOrigin],
):
    if global_idx.x != 0:
        return
    var item_idx = leaf_ids[unsafe_offset=0]
    var bounds_span = Span(unsafe_ptr=leaf_bounds, length=AABB.STRIDE)
    var bounds = AABB[Frame.WORLD].load6(
        bounds_span, Int(item_idx) * AABB.STRIDE
    )
    _wide_node_store_child[node_width](
        wide_nodes,
        UInt32(0),
        0,
        bounds,
        _pack_wide_meta(UInt32(0), UInt32(1)),
    )
    comptime for lane in range(1, node_width):
        _wide_node_store_child[node_width](
            wide_nodes,
            UInt32(0),
            lane,
            AABB[Frame.WORLD].invalid(),
            _pack_wide_meta(UInt32(0), EMPTY_LANE),
        )

    leaf_block_indices.unsafe_offset(0).unsafe_store[width=leaf_width](
        EMPTY_LANE
    )
    leaf_block_indices[unsafe_offset=0] = leaf_payloads[
        unsafe_offset=Int(item_idx)
    ]


def hploc_literature_to_wide_kernel[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    max_leaf_size: Int,
    fat_leaves: Bool,
    block_size: Int,
](
    leaf_bounds: Pointer[Float32, MutAnyOrigin],
    leaf_payloads: Pointer[UInt32, MutAnyOrigin],
    leaf_ids: Pointer[UInt32, MutAnyOrigin],
    node_meta: Pointer[UInt32, MutAnyOrigin],
    node_bounds: Pointer[Float32, MutAnyOrigin],
    node_leaf_counts: Pointer[UInt32, MutAnyOrigin],
    index_pairs: Pointer[UInt64, MutAnyOrigin],
    work_alloc_counter: Pointer[UInt32, MutAnyOrigin],
    work_group_counter: Pointer[UInt32, MutAnyOrigin],
    leaf_block_counter: Pointer[UInt32, MutAnyOrigin],
    wide_node_counter: Pointer[UInt32, MutAnyOrigin],
    status: Pointer[UInt32, MutAnyOrigin],
    wide_nodes: Pointer[Float32, MutAnyOrigin],
    leaf_block_indices: Pointer[UInt32, MutAnyOrigin],
    leaf_count: Int32,
    max_wide_nodes: Int32,
    max_leaf_blocks: Int32,
):
    """Paper §3.4: single-dispatch top-down BVH2-to-N-wide conversion."""

    comptime assert block_size == GPU_BOUNDS_BVH_BLOCK_SIZE
    comptime fat_leaf_limit = min(max_leaf_size, 4)
    var logical_block = stack_allocation[
        1, UInt32, address_space=AddressSpace.SHARED
    ]()
    if thread_idx.x == 0:
        logical_block[unsafe_offset=0] = Atomic.fetch_add[
            ordering=Ordering.RELAXED
        ](work_group_counter, UInt32(1))
    barrier()

    var leaf_count_int = Int(leaf_count)
    var work_id = (
        Int(logical_block[unsafe_offset=0]) * block_size + thread_idx.x
    )
    if work_id >= leaf_count_int:
        return

    var internal_count = leaf_count_int - 1
    var leaf_bounds_span = Span(
        unsafe_ptr=leaf_bounds, length=leaf_count_int * AABB.STRIDE
    )
    var leaf_ids_span = Span(unsafe_ptr=leaf_ids, length=leaf_count_int)
    var node_bounds_span = Span(
        unsafe_ptr=node_bounds,
        length=max(internal_count, 1) * BinaryBvhNode.BOUNDS_STRIDE,
    )
    var max_wide_nodes_int = Int(max_wide_nodes)
    var max_leaf_blocks_int = Int(max_leaf_blocks)
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
                _wide_node_store_child[node_width](
                    wide_nodes,
                    UInt32(0),
                    0,
                    _encoded_bounds(
                        encoded,
                        leaf_bounds_span,
                        leaf_ids_span,
                        node_bounds_span,
                    ),
                    _pack_wide_meta(UInt32(0), encoded_leaf_count),
                )
                comptime for lane in range(1, node_width):
                    _wide_node_store_child[node_width](
                        wide_nodes,
                        UInt32(0),
                        lane,
                        AABB[Frame.WORLD].invalid(),
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
        var node_idx = decode_ref_index(encoded)
        candidates[0] = _node_left(node_meta, node_idx)
        candidates[1] = _node_right(node_meta, node_idx)
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
                    if _hploc_encoded_leaf_count(
                        candidate, node_leaf_counts
                    ) <= UInt32(fat_leaf_limit):
                        continue
                var area = _encoded_bounds(
                    candidate,
                    leaf_bounds_span,
                    leaf_ids_span,
                    node_bounds_span,
                ).surface_area()[0]
                if area > largest_area:
                    largest_area = area
                    open_pos = candidate_pos

            if open_pos < 0:
                break
            var opened_idx = decode_ref_index(candidates[open_pos])
            candidates[open_pos] = _node_left(node_meta, opened_idx)
            candidates[child_count] = _node_right(node_meta, opened_idx)
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
                    var area = _encoded_bounds(
                        candidate,
                        leaf_bounds_span,
                        leaf_ids_span,
                        node_bounds_span,
                    ).surface_area()[0]
                    if area > largest_area:
                        largest_area = area
                        open_pos = candidate_pos

                if open_pos < 0:
                    break
                var opened_idx = decode_ref_index(candidates[open_pos])
                candidates[open_pos] = _node_left(node_meta, opened_idx)
                candidates[child_count] = _node_right(node_meta, opened_idx)
                child_count += 1

        var inner_count = 0
        var leaf_count_in_node = 0
        var published_work_count = 0
        var candidate_leaf_counts = Array[UInt32, node_width](
            uninitialized=True
        )
        for child_pos in range(child_count):
            var candidate = candidates[child_pos]
            var candidate_leaf_count = _hploc_encoded_leaf_count(
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
        for child_pos in range(child_count):
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
                meta = _pack_wide_meta(
                    child_out_idx, candidate_leaf_counts[child_pos]
                )
            else:
                child_out_idx = child_node_base + inner_rank
                inner_rank += 1
                meta = _pack_wide_meta(child_out_idx, UInt32(0))

            var bounds = _encoded_bounds(
                child,
                leaf_bounds_span,
                leaf_ids_span,
                node_bounds_span,
            )
            _wide_node_store_child[node_width](
                wide_nodes, out_idx, child_pos, bounds, meta
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

        comptime for lane in range(node_width):
            if lane >= child_count:
                _wide_node_store_child[node_width](
                    wide_nodes,
                    out_idx,
                    lane,
                    AABB[Frame.WORLD].invalid(),
                    _pack_wide_meta(UInt32(0), EMPTY_LANE),
                )


def collapse_binary_to_wide[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    max_leaf_size: Int,
    fat_leaves: Bool = False,
](
    mut ctx: DeviceContext,
    binary: GpuBinaryBoundsBvh,
    mut out: GpuWideBoundsBvh[node_width, leaf_width, max_leaf_size],
) raises:
    """Convert any supported BVH2 using the paper's §3.4 GPU method.

    The optional triangle path follows HIPRT by retaining binary subtrees of
    at most four primitives as packed leaves. The default remains the paper's
    one-primitive-per-leaf baseline.
    """

    if binary.leaf_count == 1:
        ctx.enqueue_function[
            hploc_literature_wide_single_leaf_kernel[node_width, leaf_width]
        ](
            binary.leaf_bounds,
            binary.leaf_payloads,
            binary.leaf_ids,
            out.wide_nodes,
            out.leaf_block_indices,
            grid_dim=1,
            block_dim=1,
        )
        ctx.synchronize()
        out.root_idx = UInt32(0)
        out.node_count = 1
        out.leaf_block_count = 1
        return

    var slot_count = binary.leaf_count
    var blocks = ceildiv(slot_count, GPU_BOUNDS_BVH_BLOCK_SIZE)
    var index_pairs = ctx.enqueue_create_buffer[DType.uint64](slot_count)
    var work_alloc_counter = ctx.enqueue_create_buffer[DType.uint32](1)
    var work_group_counter = ctx.enqueue_create_buffer[DType.uint32](1)
    var leaf_block_counter = ctx.enqueue_create_buffer[DType.uint32](1)
    var wide_node_counter = ctx.enqueue_create_buffer[DType.uint32](1)
    var status = ctx.enqueue_create_buffer[DType.uint32](1)

    ctx.enqueue_function[init_hploc_literature_wide_kernel](
        binary.node_meta,
        index_pairs,
        work_alloc_counter,
        work_group_counter,
        leaf_block_counter,
        wide_node_counter,
        status,
        Int32(binary.internal_count),
        Int32(slot_count),
        grid_dim=blocks,
        block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
    )
    ctx.enqueue_function[
        hploc_literature_to_wide_kernel[
            node_width,
            leaf_width,
            max_leaf_size,
            fat_leaves,
            GPU_BOUNDS_BVH_BLOCK_SIZE,
        ]
    ](
        binary.leaf_bounds,
        binary.leaf_payloads,
        binary.leaf_ids,
        binary.node_meta,
        binary.node_bounds,
        binary.node_leaf_counts,
        index_pairs,
        work_alloc_counter,
        work_group_counter,
        leaf_block_counter,
        wide_node_counter,
        status,
        out.wide_nodes,
        out.leaf_block_indices,
        Int32(slot_count),
        Int32(out.max_wide_nodes),
        Int32(out.max_leaf_blocks),
        grid_dim=blocks,
        block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
    )
    ctx.synchronize()

    with leaf_block_counter.map_to_host() as leaves, wide_node_counter.map_to_host() as nodes, status.map_to_host() as build_status:
        if build_status[0] != HPLOC_WIDE_STATUS_OK:
            raise String(t"BVH2-to-wide conversion status: {build_status[0]}")
        comptime if fat_leaves:
            if Int(leaves[0]) <= 0 or Int(leaves[0]) > binary.leaf_count:
                raise "fat-leaf conversion emitted an invalid leaf count"
        else:
            if Int(leaves[0]) != binary.leaf_count:
                raise "§3.4 conversion did not emit one leaf per primitive"
        out.root_idx = UInt32(0)
        out.node_count = Int(nodes[0])
        out.leaf_block_count = Int(leaves[0])
