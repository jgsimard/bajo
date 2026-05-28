from std.math import max, ceildiv
from std.atomic import Atomic
from std.gpu import DeviceContext, global_idx

from bajo.core.aabb import AABB
from bajo.bvh.constants import (
    LBVH_SENTINEL,
    GPU_STACK_SIZE,
    EMPTY_LANE,
    GPU_BOUNDS_BVH_BLOCK_SIZE,
)
from bajo.bvh.gpu.bounds_bvh import GpuBoundsBvh
from bajo.bvh.gpu.builder.binary_layout import (
    GpuBinaryBoundsBvh,
    _node_parent_index,
    _node_left,
    _node_right,
    _is_encoded_leaf,
    _encoded_index,
    _load_and_union_node_bounds,
)


def _encoded_leaf_count_gpu(
    encoded: UInt32,
    node_leaf_counts: UnsafePointer[UInt32, MutAnyOrigin],
) -> UInt32:
    if _is_encoded_leaf(encoded):
        return UInt32(1)
    return UInt32(node_leaf_counts[Int(_encoded_index(encoded))])


def _encoded_bounds_gpu(
    encoded: UInt32,
    leaf_bounds: UnsafePointer[Float32, MutAnyOrigin],
    leaf_ids: UnsafePointer[UInt32, MutAnyOrigin],
    node_bounds: UnsafePointer[Float32, MutAnyOrigin],
) -> AABB:
    if _is_encoded_leaf(encoded):
        var sorted_leaf_idx = _encoded_index(encoded)
        var item_idx = UInt32(leaf_ids[Int(sorted_leaf_idx)])
        var b = Int(item_idx) * AABB.STRIDE
        return AABB.load6(leaf_bounds, b)

    return _load_and_union_node_bounds(node_bounds, _encoded_index(encoded))


def _write_wide_lane_bounds[
    width: Int,
](
    wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    wide_node_idx: UInt32,
    lane: Int,
    bounds: AABB,
):
    var b = (Int(wide_node_idx) * width + lane) * AABB.STRIDE
    bounds.store6(wide_bounds, b)


def _collect_encoded_leaf_payloads_gpu[
    width: Int,
](
    encoded: UInt32,
    leaf_ids: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_payloads: UnsafePointer[UInt32, MutAnyOrigin],
    node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    out_leaf_block_indices: UnsafePointer[UInt32, MutAnyOrigin],
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
                var item_idx = UInt32(leaf_ids[Int(sorted_leaf_idx)])
                out_leaf_block_indices[
                    Int(leaf_block_idx) * width + out_count
                ] = UInt32(leaf_payloads[Int(item_idx)])
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
            out_leaf_block_indices[
                Int(leaf_block_idx) * width + lane
            ] = EMPTY_LANE


def init_gpu_wide_collapse_kernel[
    width: Int,
](
    wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_block_indices: UnsafePointer[UInt32, MutAnyOrigin],
    node_leaf_counts: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_block_counter: UnsafePointer[UInt32, MutAnyOrigin],
    wide_root: UnsafePointer[UInt32, MutAnyOrigin],
    max_wide_lanes: Int,
    max_leaf_slots: Int,
    internal_count: Int,
):
    var i = global_idx.x

    if i < max_wide_lanes:
        wide_data[i] = UInt32(0)
        wide_counts[i] = EMPTY_LANE

        var b = i * AABB.STRIDE
        AABB.invalid().store6(wide_bounds, b)

    if i < max_leaf_slots:
        leaf_block_indices[i] = EMPTY_LANE

    if i < internal_count:
        node_leaf_counts[i] = UInt32(0)

    if i == 0:
        leaf_block_counter[0] = UInt32(0)
        wide_root[0] = UInt32(0)


def compute_binary_subtree_leaf_counts_kernel(
    node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_parent: UnsafePointer[UInt32, MutAnyOrigin],
    node_leaf_counts: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_count: Int,
):
    var leaf_idx = global_idx.x
    if leaf_idx >= leaf_count:
        return

    var parent = UInt32(leaf_parent[leaf_idx])
    while parent != LBVH_SENTINEL:
        _ = Atomic.fetch_add(node_leaf_counts + Int(parent), UInt32(1))
        parent = UInt32(node_meta[_node_parent_index(parent)])


def find_binary_root_kernel(
    node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    wide_root: UnsafePointer[UInt32, MutAnyOrigin],
    internal_count: Int,
):
    var i = global_idx.x
    if i >= internal_count:
        return

    if UInt32(node_meta[_node_parent_index(UInt32(i))]) == LBVH_SENTINEL:
        wide_root[0] = UInt32(i)


def collapse_single_leaf_to_wide_kernel[
    width: Int,
](
    leaf_bounds: UnsafePointer[Float32, MutAnyOrigin],
    leaf_payloads: UnsafePointer[UInt32, MutAnyOrigin],
    wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_block_indices: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_block_counter: UnsafePointer[UInt32, MutAnyOrigin],
    wide_root: UnsafePointer[UInt32, MutAnyOrigin],
):
    var i = global_idx.x
    if i != 0:
        return

    wide_root[0] = UInt32(0)
    leaf_block_counter[0] = UInt32(1)

    wide_bounds[0] = leaf_bounds[0]
    wide_bounds[1] = leaf_bounds[1]
    wide_bounds[2] = leaf_bounds[2]
    wide_bounds[3] = leaf_bounds[3]
    wide_bounds[4] = leaf_bounds[4]
    wide_bounds[5] = leaf_bounds[5]

    wide_data[0] = UInt32(0)
    wide_counts[0] = UInt32(1)
    leaf_block_indices[0] = UInt32(leaf_payloads[0])


def collapse_binary_to_wide_kernel[
    width: Int,
](
    leaf_bounds: UnsafePointer[Float32, MutAnyOrigin],
    leaf_payloads: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_ids: UnsafePointer[UInt32, MutAnyOrigin],
    node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    node_bounds: UnsafePointer[Float32, MutAnyOrigin],
    node_leaf_counts: UnsafePointer[UInt32, MutAnyOrigin],
    wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_block_indices: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_block_counter: UnsafePointer[UInt32, MutAnyOrigin],
    internal_count: Int,
):
    var node_i = global_idx.x
    if node_i >= internal_count:
        return

    var wide_idx = UInt32(node_i)
    var encoded = UInt32(node_i)

    var pool = InlineArray[UInt32, width](fill=0)
    pool[0] = encoded
    var p_size = 1

    # Greedy CPU-like collapse: expand the largest internal subtree while it
    # still contains more leaves than can fit into one packed leaf block.
    while p_size < width:
        var best_area = Float32(-1.0)
        var best_lane = -1

        comptime for lane in range(width):
            if lane < p_size:
                var e = pool[lane]
                if not _is_encoded_leaf(e):
                    var subtree_leaves = _encoded_leaf_count_gpu(
                        e, node_leaf_counts
                    )
                    if subtree_leaves > UInt32(width):
                        var b = _encoded_bounds_gpu(
                            e, leaf_bounds, leaf_ids, node_bounds
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
        var lane_base = Int(wide_idx) * width + lane

        if lane >= p_size:
            wide_counts[lane_base] = EMPTY_LANE
            continue

        var e = pool[lane]
        var b = _encoded_bounds_gpu(e, leaf_bounds, leaf_ids, node_bounds)
        _write_wide_lane_bounds[width](wide_bounds, wide_idx, lane, b)

        var subtree_leaves = _encoded_leaf_count_gpu(e, node_leaf_counts)
        if subtree_leaves <= UInt32(width):
            var block_idx = Atomic.fetch_add(leaf_block_counter, UInt32(1))
            _collect_encoded_leaf_payloads_gpu[width](
                e,
                leaf_ids,
                leaf_payloads,
                node_meta,
                leaf_block_indices,
                block_idx,
            )
            wide_data[lane_base] = block_idx
            wide_counts[lane_base] = subtree_leaves
        else:
            wide_data[lane_base] = _encoded_index(e)
            wide_counts[lane_base] = UInt32(0)


def collapse[
    width: Int
](
    mut ctx: DeviceContext,
    binary: GpuBinaryBoundsBvh,
    mut out: GpuBoundsBvh[width],
) raises:
    """Converts a sorted binary bounds BVH into the final wide layout."""
    var n_internal = max(binary.internal_count, 1)
    var max_wide_lanes = out.max_wide_nodes * width
    var max_leaf_slots = out.max_leaf_blocks * width
    var init_n = max(
        max(max_wide_lanes, max_leaf_slots),
        n_internal,
    )
    var init_blocks = ceildiv(init_n, GPU_BOUNDS_BVH_BLOCK_SIZE)

    # Collapse scratch. These buffers are not traversal data.
    var node_leaf_counts = ctx.enqueue_create_buffer[DType.uint32](n_internal)
    var leaf_block_counter = ctx.enqueue_create_buffer[DType.uint32](1)
    var wide_root = ctx.enqueue_create_buffer[DType.uint32](1)

    ctx.enqueue_function[init_gpu_wide_collapse_kernel[width]](
        out.wide_bounds,
        out.wide_data,
        out.wide_counts,
        out.leaf_block_indices,
        node_leaf_counts,
        leaf_block_counter,
        wide_root,
        max_wide_lanes,
        max_leaf_slots,
        binary.internal_count,
        grid_dim=init_blocks,
        block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
    )

    if binary.leaf_count == 1:
        ctx.enqueue_function[collapse_single_leaf_to_wide_kernel[width]](
            binary.leaf_bounds,
            binary.leaf_payloads,
            out.wide_bounds,
            out.wide_data,
            out.wide_counts,
            out.leaf_block_indices,
            leaf_block_counter,
            wide_root,
            grid_dim=1,
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
    elif binary.leaf_count > 1:
        ctx.enqueue_function[compute_binary_subtree_leaf_counts_kernel](
            binary.node_meta,
            binary.leaf_parent,
            node_leaf_counts,
            binary.leaf_count,
            grid_dim=binary.blocks_leaves,
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
        ctx.enqueue_function[find_binary_root_kernel](
            binary.node_meta,
            wide_root,
            binary.internal_count,
            grid_dim=binary.blocks_internal,
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
        ctx.enqueue_function[collapse_binary_to_wide_kernel[width]](
            binary.leaf_bounds,
            binary.leaf_payloads,
            binary.leaf_ids,
            binary.node_meta,
            binary.node_bounds,
            node_leaf_counts,
            out.wide_bounds,
            out.wide_data,
            out.wide_counts,
            out.leaf_block_indices,
            leaf_block_counter,
            binary.internal_count,
            grid_dim=binary.blocks_internal,
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )

    ctx.synchronize()

    leaf_block_count: Int
    with leaf_block_counter.map_to_host() as h:
        leaf_block_count = Int(h[0])

    with wide_root.map_to_host() as h:
        root_idx = UInt32(h[0])

    out.root_idx = root_idx
    out.node_count = max(binary.internal_count, 1)
    out.leaf_block_count = leaf_block_count
