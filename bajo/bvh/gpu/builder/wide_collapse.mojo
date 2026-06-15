from std.math import ceildiv
from std.atomic import Atomic, Ordering
from std.gpu import DeviceContext, DeviceBuffer, global_idx

from bajo.core import AABB
from bajo.bvh.constants import (
    LBVH_SENTINEL,
    GPU_STACK_SIZE,
    EMPTY_LANE,
    GPU_BOUNDS_BVH_BLOCK_SIZE,
)
from bajo.bvh.gpu.bounds_bvh import GpuBoundsBvh, _wide_node_store_child
from bajo.bvh.gpu.wide_meta import _pack_wide_meta
from bajo.bvh.gpu.builder.binary_layout import (
    GpuBinaryBoundsBvh,
    _node_parent_index,
    _node_left,
    _node_right,
    _is_encoded_leaf,
    _encoded_index,
    _load_and_union_node_bounds,
)


def _encoded_bounds(
    encoded: UInt32,
    leaf_bounds: UnsafePointer[mut=True, Float32, _],
    leaf_ids: UnsafePointer[mut=True, UInt32, _],
    node_bounds: UnsafePointer[mut=True, Float32, _],
) -> AABB:
    if _is_encoded_leaf(encoded):
        var sorted_leaf_idx = _encoded_index(encoded)
        var item_idx = UInt32(leaf_ids[Int(sorted_leaf_idx)])
        var b = Int(item_idx) * AABB.STRIDE
        return AABB.load6(leaf_bounds, b)

    return _load_and_union_node_bounds(node_bounds, _encoded_index(encoded))


def collapse_terminal_root_to_wide_kernel[
    width: SIMDSize,
](
    leaf_bounds: UnsafePointer[Float32, MutAnyOrigin],
    leaf_payloads: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_ids: UnsafePointer[UInt32, MutAnyOrigin],
    node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    node_bounds: UnsafePointer[Float32, MutAnyOrigin],
    wide_nodes: UnsafePointer[Float32, MutAnyOrigin],
    leaf_block_indices: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_block_counter: UnsafePointer[UInt32, MutAnyOrigin],
    wide_node_counter: UnsafePointer[UInt32, MutAnyOrigin],
    wide_root: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_count: Int,
    internal_count: Int,
):
    var i = global_idx.x
    if i != 0:
        return

    wide_root[0] = UInt32(0)
    wide_node_counter[0] = UInt32(1)
    leaf_block_counter[0] = UInt32(1)

    var root_bounds = AABB.invalid()

    if leaf_count == 1:
        # Single primitive: no binary internal node exists.
        root_bounds = AABB.load6(leaf_bounds, 0)

        (leaf_block_indices + 0).store[width=width](EMPTY_LANE)
        leaf_block_indices[0] = leaf_payloads[0]

    else:
        # Small tree: find the binary root and pack the whole subtree into
        # leaf block 0.
        var root = UInt32(0)

        for n in range(internal_count):
            var node_idx = UInt32(n)
            if UInt32(node_meta[_node_parent_index(node_idx)]) == LBVH_SENTINEL:
                root = node_idx

        _write_terminal_leaf_block[width](
            root,
            leaf_payloads,
            leaf_ids,
            node_meta,
            leaf_block_indices,
            UInt32(0),
        )

        root_bounds = _encoded_bounds(
            root,
            leaf_bounds,
            leaf_ids,
            node_bounds,
        )

    # Root wide node: lane 0 is one packed leaf block.
    _wide_node_store_child[width](
        wide_nodes,
        UInt32(0),
        0,
        root_bounds,
        _pack_wide_meta(UInt32(0), UInt32(leaf_count)),
    )

    # Remaining lanes are empty.
    comptime for lane in range(1, width):
        _wide_node_store_child[width](
            wide_nodes,
            UInt32(0),
            lane,
            AABB.invalid(),
            _pack_wide_meta(UInt32(0), EMPTY_LANE),
        )


def collapse[
    width: SIMDSize
](
    mut ctx: DeviceContext,
    binary: GpuBinaryBoundsBvh,
    mut out: GpuBoundsBvh[width],
) raises:
    var leaf_block_counter = ctx.enqueue_create_buffer[DType.uint32](1)
    var wide_node_counter = ctx.enqueue_create_buffer[DType.uint32](1)
    var wide_root = ctx.enqueue_create_buffer[DType.uint32](1)

    if binary.leaf_count <= width:
        ctx.enqueue_function[collapse_terminal_root_to_wide_kernel[width]](
            binary.leaf_bounds,
            binary.leaf_payloads,
            binary.leaf_ids,
            binary.node_meta,
            binary.node_bounds,
            out.wide_nodes,
            out.leaf_block_indices,
            leaf_block_counter,
            wide_node_counter,
            wide_root,
            binary.leaf_count,
            binary.internal_count,
            grid_dim=1,
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )

        ctx.synchronize()

        out.root_idx = 0
        out.node_count = 1
        out.leaf_block_count = 1
        return

    comptime if width == 2:
        # Converts a sorted binary bounds BVH into a precomputed wide layout.

        # Every binary internal node gets a possible wide node with the same index.
        # This is not a dense reachable layout: some wide nodes may be unreachable
        # because an ancestor wide node opened through their binary subtree.

        # The benefit is that conversion is deterministic, non-polling, and does not
        # require a host-side frontier loop.

        # Every binary internal node is precomputed as a possible wide node.
        # Child links use the binary internal node index directly as the wide
        # node index. Some precomputed wide nodes are unreachable because
        # their binary subtree was opened into an ancestor wide node,
        # but this keeps the conversion single-pass, deterministic, and
        # free of host frontier loops.
        var leaf_blocks = ceildiv(binary.leaf_count, GPU_BOUNDS_BVH_BLOCK_SIZE)
        ctx.enqueue_function[init_precomputed_wide_leaf_blocks_kernel[width]](
            binary.leaf_payloads,
            binary.leaf_ids,
            out.leaf_block_indices,
            leaf_block_counter,
            wide_node_counter,
            binary.leaf_count,
            binary.internal_count,
            grid_dim=leaf_blocks,
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )

        ctx.enqueue_function[collapse_precomputed_wide_kernel[width]](
            binary.leaf_bounds,
            binary.leaf_ids,
            binary.node_meta,
            binary.node_bounds,
            out.wide_nodes,
            wide_root,
            binary.internal_count,
            grid_dim=binary.blocks_internal,
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )

        ctx.synchronize()

        with wide_root.map_to_host() as wr:
            out.root_idx = UInt32(wr[0])

        out.node_count = binary.internal_count
        out.leaf_block_count = binary.leaf_count

    else:  # HPLOC
        var slot_count = binary.leaf_count
        var slot_blocks = ceildiv(slot_count, GPU_BOUNDS_BVH_BLOCK_SIZE)

        var index_pairs = ctx.enqueue_create_buffer[DType.uint64](slot_count)
        var slot_counter = ctx.enqueue_create_buffer[DType.uint32](1)
        var hploc_status = ctx.enqueue_create_buffer[DType.uint32](1)

        ctx.enqueue_function[init_hploc_index_pairs_kernel](
            binary.node_meta,
            index_pairs,
            slot_counter,
            leaf_block_counter,
            wide_node_counter,
            hploc_status,
            wide_root,
            binary.internal_count,
            slot_count,
            grid_dim=slot_blocks,
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )

        ctx.enqueue_function[hploc_to_wide_kernel[width]](
            binary.leaf_bounds,
            binary.leaf_payloads,
            binary.leaf_ids,
            binary.node_meta,
            binary.node_bounds,
            binary.node_leaf_counts,
            index_pairs,
            slot_counter,
            leaf_block_counter,
            wide_node_counter,
            hploc_status,
            out.wide_nodes,
            out.leaf_block_indices,
            slot_count,
            out.max_wide_nodes,
            out.max_leaf_blocks,
            grid_dim=slot_blocks,
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )

        ctx.synchronize()

        with wide_root.map_to_host() as wr, leaf_block_counter.map_to_host() as lbc, wide_node_counter.map_to_host() as wnc, hploc_status.map_to_host() as hs:
            out.root_idx = UInt32(wr[0])
            out.node_count = Int(wnc[0])
            out.leaf_block_count = Int(lbc[0])

            if UInt32(hs[0]) != HPLOC_STATUS_OK:
                raise String(t"HPLOC collapse status: {UInt32(hs[0])}")


def init_precomputed_wide_leaf_blocks_kernel[
    width: SIMDSize,
](
    leaf_payloads: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_ids: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_block_indices: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_block_counter: UnsafePointer[UInt32, MutAnyOrigin],
    wide_node_counter: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_count: Int,
    internal_count: Int,
):
    var i = global_idx.x
    if i >= leaf_count:
        return

    if i == 0:
        leaf_block_counter[0] = UInt32(leaf_count)
        wide_node_counter[0] = UInt32(internal_count)

    var sorted_leaf_idx = UInt32(i)
    var item_idx = UInt32(leaf_ids[i])
    var payload = UInt32(leaf_payloads[Int(item_idx)])

    var base = i * width
    (leaf_block_indices + base).store[width=width](EMPTY_LANE)
    leaf_block_indices[base] = payload


def collapse_precomputed_wide_kernel[
    width: SIMDSize,
](
    leaf_bounds: UnsafePointer[Float32, MutAnyOrigin],
    leaf_ids: UnsafePointer[UInt32, MutAnyOrigin],
    node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    node_bounds: UnsafePointer[Float32, MutAnyOrigin],
    wide_nodes: UnsafePointer[Float32, MutAnyOrigin],
    wide_root: UnsafePointer[UInt32, MutAnyOrigin],
    internal_count: Int,
):
    var node_i = global_idx.x
    if node_i >= internal_count:
        return

    var node_idx = UInt32(node_i)

    if UInt32(node_meta[_node_parent_index(node_idx)]) == UInt32(LBVH_SENTINEL):
        wide_root[0] = node_idx

    var pool = InlineArray[UInt32, width](uninitialized=True)
    pool[0] = node_idx
    var p_size = 1

    # repeatedly open the largest-area internal subtree until
    # the wide node has width entries, or only leaves remain
    while p_size < width:
        var best_area = Float32(-1.0)
        var best_lane = -1

        comptime for lane in range(width):
            if lane < p_size:
                var e = pool[lane]

                if not _is_encoded_leaf(e):
                    var b = _encoded_bounds(
                        e,
                        leaf_bounds,
                        leaf_ids,
                        node_bounds,
                    )
                    var area = b.surface_area()

                    if area > best_area:
                        best_area = area
                        best_lane = lane

        if best_lane < 0:
            break

        var n_idx = _encoded_index(pool[best_lane])
        pool[best_lane] = _node_left(node_meta, n_idx)
        pool[p_size] = _node_right(node_meta, n_idx)
        p_size += 1

    comptime for lane in range(width):
        if lane >= p_size:
            _wide_node_store_child[width](
                wide_nodes,
                node_idx,
                lane,
                AABB.invalid(),
                _pack_wide_meta(UInt32(0), EMPTY_LANE),
            )
            continue

        var e = pool[lane]
        var b = _encoded_bounds(e, leaf_bounds, leaf_ids, node_bounds)

        var meta = _pack_wide_meta(_encoded_index(e), UInt32(0))
        if _is_encoded_leaf(e):
            var sorted_leaf_idx = _encoded_index(e)
            meta = _pack_wide_meta(sorted_leaf_idx, UInt32(1))

        _wide_node_store_child[width](
            wide_nodes,
            node_idx,
            lane,
            b,
            meta,
        )


# hploc

comptime HPLOC_STATUS_OK = UInt32(0)
comptime HPLOC_STATUS_OUT_OF_WORK_SLOTS = UInt32(1)
comptime HPLOC_STATUS_OUT_OF_WIDE_NODES = UInt32(2)
comptime HPLOC_STATUS_OUT_OF_LEAF_BLOCKS = UInt32(3)
comptime HPLOC_STATUS_TIMEOUT = UInt32(4)
comptime HPLOC_NOOP_ENCODED = UInt32(LBVH_SENTINEL)


def _pack_hploc_pair(encoded: UInt32, out_idx: UInt32) -> UInt64:
    return (UInt64(encoded) << 32) | UInt64(out_idx)


def _hploc_pair_encoded(pair: UInt64) -> UInt32:
    return UInt32(pair >> 32)


def _hploc_pair_out_idx(pair: UInt64) -> UInt32:
    return UInt32(pair & UInt64(0xFFFFFFFF))


def _encoded_leaf_count[
    origin: ImmutOrigin
](encoded: UInt32, node_leaf_counts: UnsafePointer[UInt32, origin]) -> UInt32:
    if _is_encoded_leaf(encoded):
        return UInt32(1)
    return node_leaf_counts[Int(_encoded_index(encoded))]


def _write_terminal_leaf_block[
    width: SIMDSize,
](
    encoded: UInt32,
    leaf_payloads: UnsafePointer[mut=False, UInt32, _],
    leaf_ids: UnsafePointer[mut=False, UInt32, _],
    node_meta: UnsafePointer[mut=False, UInt32, _],
    leaf_block_indices: UnsafePointer[mut=True, UInt32, _],
    leaf_block_idx: UInt32,
):
    var stack = InlineArray[UInt32, GPU_STACK_SIZE](uninitialized=True)
    var sp = 0
    stack[sp] = encoded
    sp += 1

    var out_count = 0

    while sp > 0:
        sp -= 1
        var e = stack[sp]

        if _is_encoded_leaf(e):
            if out_count < width:
                var sorted_leaf_idx = _encoded_index(e)
                var item_idx = leaf_ids[Int(sorted_leaf_idx)]
                var payload = leaf_payloads[Int(item_idx)]

                leaf_block_indices[
                    Int(leaf_block_idx) * width + out_count
                ] = payload
                out_count += 1
        else:
            var node_idx = _encoded_index(e)
            var left = _node_left(node_meta, node_idx)
            var right = _node_right(node_meta, node_idx)

            if sp + 2 <= GPU_STACK_SIZE:
                stack[sp] = right
                sp += 1
                stack[sp] = left
                sp += 1

    comptime for lane in range(width):
        if lane >= out_count:
            leaf_block_indices[Int(leaf_block_idx) * width + lane] = EMPTY_LANE


def _write_one_leaf_block[
    width: SIMDSize,
](
    encoded_leaf: UInt32,
    leaf_payloads: UnsafePointer[mut=False, UInt32, _],
    leaf_ids: UnsafePointer[mut=False, UInt32, _],
    leaf_block_indices: UnsafePointer[mut=True, UInt32, _],
    leaf_block_idx: UInt32,
):
    var sorted_leaf_idx = _encoded_index(encoded_leaf)
    var item_idx = UInt32(leaf_ids[Int(sorted_leaf_idx)])
    var payload = UInt32(leaf_payloads[Int(item_idx)])

    var base = Int(leaf_block_idx) * width
    (leaf_block_indices + base).store[width=width](EMPTY_LANE)
    leaf_block_indices[base] = payload


def init_hploc_index_pairs_kernel(
    node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    index_pairs: UnsafePointer[UInt64, MutAnyOrigin],
    slot_counter: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_block_counter: UnsafePointer[UInt32, MutAnyOrigin],
    wide_node_counter: UnsafePointer[UInt32, MutAnyOrigin],
    status: UnsafePointer[UInt32, MutAnyOrigin],
    wide_root: UnsafePointer[UInt32, MutAnyOrigin],
    internal_count: Int,
    slot_count: Int,
):
    # slot 0 receives the pair (binary root cluster, wide root index 0) all other slots are invalid
    # worker-slot counter starts at 0, first child reuses the current worker and the remaining children use base + i
    var i = global_idx.x

    if i < slot_count:
        index_pairs[i] = UInt64.MAX

    if i == 0:
        slot_counter[0] = UInt32(0)
        leaf_block_counter[0] = UInt32(0)
        wide_node_counter[0] = UInt32(1)
        status[0] = HPLOC_STATUS_OK
        wide_root[0] = UInt32(0)

    if i >= internal_count:
        return

    var node_idx = UInt32(i)
    if UInt32(node_meta[_node_parent_index(node_idx)]) == LBVH_SENTINEL:
        index_pairs[0] = _pack_hploc_pair(node_idx, UInt32(0))


def hploc_to_wide_kernel[
    width: SIMDSize,
](
    leaf_bounds: UnsafePointer[Float32, MutAnyOrigin],
    leaf_payloads: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_ids: UnsafePointer[UInt32, MutAnyOrigin],
    node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    node_bounds: UnsafePointer[Float32, MutAnyOrigin],
    node_leaf_counts: UnsafePointer[UInt32, MutAnyOrigin],
    index_pairs: UnsafePointer[UInt64, MutAnyOrigin],
    slot_counter: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_block_counter: UnsafePointer[UInt32, MutAnyOrigin],
    wide_node_counter: UnsafePointer[UInt32, MutAnyOrigin],
    status: UnsafePointer[UInt32, MutAnyOrigin],
    wide_nodes: UnsafePointer[Float32, MutAnyOrigin],
    leaf_block_indices: UnsafePointer[UInt32, MutAnyOrigin],
    slot_count: Int,
    max_wide_nodes: Int,
    max_leaf_blocks: Int,
):
    # algo:
    #   - each GPU thread owns one slot in index_pairs
    #   - invalid slots spin until another worker writes a valid pair
    #   - an internal task collapses one binary subtree into one wide node
    #   - child leaf tasks and child internal-node tasks are assigned to slots
    #   - the current worker slot is reused for the first child, and the remaining
    #     children are assigned to atomically allocated slots.
    var thread_id = global_idx.x
    if thread_id >= slot_count:
        return

    var failsafe = 1000000

    while True:
        if status[0] != HPLOC_STATUS_OK:
            return

        var pair = (index_pairs + thread_id).load[volatile=True]()

        if pair == UInt64.MAX:
            failsafe -= 1
            if failsafe <= 0:
                status[0] = HPLOC_STATUS_TIMEOUT
                return
            continue

        var encoded = _hploc_pair_encoded(pair)
        var out_idx = _hploc_pair_out_idx(pair)

        if encoded == HPLOC_NOOP_ENCODED:
            return

        # terminal task: write one packed leaf block and terminate
        # terminal = subtree_leaves <= width
        var encoded_leaf_count = _encoded_leaf_count(
            encoded,
            node_leaf_counts,
        )
        if encoded_leaf_count <= UInt32(width):
            if Int(out_idx) >= max_leaf_blocks:
                status[0] = HPLOC_STATUS_OUT_OF_LEAF_BLOCKS
                return

            _write_terminal_leaf_block[width](
                encoded,
                leaf_payloads,
                leaf_ids,
                node_meta,
                leaf_block_indices,
                out_idx,
            )
            return

        # internal task: collapse one binary subtree into one wide node
        if Int(out_idx) >= max_wide_nodes:
            status[0] = HPLOC_STATUS_OUT_OF_WIDE_NODES
            return

        var leaves = InlineArray[UInt32, width](uninitialized=True)
        var leaf_counts = InlineArray[UInt32, width](uninitialized=True)
        var nodes = InlineArray[UInt32, width](uninitialized=True)
        var num_leaves = 0
        var num_nodes = 0

        # starts from the two children of the assigned BVH2 node,ordered by surface area,
        # loop:
        #   pop an internal node
        #   pushes its ordered children
        #   stop: stack is full or only leaves remain
        var node_idx = _encoded_index(encoded)
        var left = _node_left(node_meta, node_idx)
        var right = _node_right(node_meta, node_idx)
        var left_bounds = _encoded_bounds(
            left,
            leaf_bounds,
            leaf_ids,
            node_bounds,
        )
        var right_bounds = _encoded_bounds(
            right,
            leaf_bounds,
            leaf_ids,
            node_bounds,
        )

        var first = left
        var second = right
        if right_bounds.surface_area() > left_bounds.surface_area():
            first = right
            second = left

        var first_leaf_count = _encoded_leaf_count(
            first,
            node_leaf_counts,
        )
        if first_leaf_count <= UInt32(width):
            leaves[num_leaves] = first
            leaf_counts[num_leaves] = first_leaf_count
            num_leaves += 1
        else:
            nodes[num_nodes] = first
            num_nodes += 1

        var second_leaf_count = _encoded_leaf_count(
            second,
            node_leaf_counts,
        )
        if second_leaf_count <= UInt32(width):
            leaves[num_leaves] = second
            leaf_counts[num_leaves] = second_leaf_count
            num_leaves += 1
        else:
            nodes[num_nodes] = second
            num_nodes += 1

        while num_leaves + num_nodes < width and num_nodes > 0:
            # pop internal node
            num_nodes -= 1
            var open_encoded = nodes[num_nodes]
            var open_idx = _encoded_index(open_encoded)

            var c0 = _node_left(node_meta, open_idx)
            var c1 = _node_right(node_meta, open_idx)
            var c0_bounds = _encoded_bounds(
                c0,
                leaf_bounds,
                leaf_ids,
                node_bounds,
            )
            var c1_bounds = _encoded_bounds(
                c1,
                leaf_bounds,
                leaf_ids,
                node_bounds,
            )

            var ordered0 = c0
            var ordered1 = c1
            if c1_bounds.surface_area() > c0_bounds.surface_area():
                ordered0 = c1
                ordered1 = c0

            var ordered0_leaf_count = _encoded_leaf_count(
                ordered0,
                node_leaf_counts,
            )
            if ordered0_leaf_count <= UInt32(width):
                leaves[num_leaves] = ordered0
                leaf_counts[num_leaves] = ordered0_leaf_count
                num_leaves += 1
            else:
                nodes[num_nodes] = ordered0
                num_nodes += 1

            var ordered1_leaf_count = _encoded_leaf_count(
                ordered1,
                node_leaf_counts,
            )
            if ordered1_leaf_count <= UInt32(width):
                leaves[num_leaves] = ordered1
                leaf_counts[num_leaves] = ordered1_leaf_count
                num_leaves += 1
            else:
                nodes[num_nodes] = ordered1
                num_nodes += 1

        var child_count = num_leaves + num_nodes

        # one worker per primitive leaf
        # if terminal subtree with k <= width leaves = packed
        # publish k - 1 no-op tasks so unused workers can exit instead of timing out
        var noop_count = UInt32(0)
        comptime for i in range(width):
            if i < num_leaves:
                if leaf_counts[i] > UInt32(1):
                    noop_count += leaf_counts[i] - UInt32(1)

        var total_published = UInt32(child_count) + noop_count

        # Allocate worker slots for every published item except the one that
        # reuses the current thread's slot.
        var extra_workers = UInt32(0)
        if total_published > UInt32(0):
            extra_workers = total_published - UInt32(1)

        var base_worker_offset = Atomic.fetch_add[ordering=Ordering.RELAXED](
            slot_counter,
            extra_workers,
        )

        var child_node_base = Atomic.fetch_add[ordering=Ordering.RELAXED](
            wide_node_counter,
            UInt32(num_nodes),
        )
        var leaf_block_base = Atomic.fetch_add[ordering=Ordering.RELAXED](
            leaf_block_counter,
            UInt32(num_leaves),
        )

        if Int(child_node_base) + num_nodes > max_wide_nodes:
            status[0] = HPLOC_STATUS_OUT_OF_WIDE_NODES
            return

        if Int(leaf_block_base) + num_leaves > max_leaf_blocks:
            status[0] = HPLOC_STATUS_OUT_OF_LEAF_BLOCKS
            return

        # schedule children: 1. terminal subtrees, 2. internal nodes
        # current slot is reused by published item 0
        # others are written to allocated slots
        var publish_i = 0
        comptime for i in range(width):
            if i < child_count:
                var child_encoded: UInt32
                var child_out_idx: UInt32

                if i < num_leaves:
                    child_encoded = leaves[i]
                    child_out_idx = leaf_block_base + UInt32(i)
                else:
                    var j = i - num_leaves
                    child_encoded = nodes[j]
                    child_out_idx = child_node_base + UInt32(j)

                var worker_addr = thread_id
                if publish_i > 0:
                    worker_addr = Int(base_worker_offset) + publish_i

                if worker_addr >= slot_count:
                    status[0] = HPLOC_STATUS_OUT_OF_WORK_SLOTS
                    return

                index_pairs[worker_addr] = _pack_hploc_pair(
                    child_encoded,
                    child_out_idx,
                )
                publish_i += 1

        comptime for i in range(width):
            if i < num_leaves:
                var extra_noops = Int(leaf_counts[i]) - 1
                for _ in range(extra_noops):
                    var worker_addr = thread_id
                    if publish_i > 0:
                        worker_addr = Int(base_worker_offset) + publish_i

                    if worker_addr >= slot_count:
                        status[0] = HPLOC_STATUS_OUT_OF_WORK_SLOTS
                        return

                    index_pairs[worker_addr] = _pack_hploc_pair(
                        HPLOC_NOOP_ENCODED,
                        UInt32(0),
                    )
                    publish_i += 1

        # emit this wide node
        # lane order = scheduled child order: leaves first,then internal nodes
        comptime for lane in range(width):
            if lane >= child_count:
                _wide_node_store_child[width](
                    wide_nodes,
                    out_idx,
                    lane,
                    AABB.invalid(),
                    _pack_wide_meta(UInt32(0), EMPTY_LANE),
                )
                continue

            var child_encoded: UInt32
            var child_ref_idx: UInt32
            var child_is_leaf = False

            if lane < num_leaves:
                child_encoded = leaves[lane]
                child_ref_idx = leaf_block_base + UInt32(lane)
                child_is_leaf = True
            else:
                var j = lane - num_leaves
                child_encoded = nodes[j]
                child_ref_idx = child_node_base + UInt32(j)

            var b = _encoded_bounds(
                child_encoded,
                leaf_bounds,
                leaf_ids,
                node_bounds,
            )
            var meta = _pack_wide_meta(child_ref_idx, UInt32(0))
            if child_is_leaf:
                meta = _pack_wide_meta(child_ref_idx, leaf_counts[lane])

            _wide_node_store_child[width](
                wide_nodes,
                out_idx,
                lane,
                b,
                meta,
            )

        # continue with the child assigned to the current slot
