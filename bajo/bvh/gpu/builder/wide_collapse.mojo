from std.math import max, ceildiv
from std.atomic import Atomic
from std.gpu import DeviceContext, global_idx

from bajo.core import AABB
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
    if i >= width:
        return

    wide_root[0] = 0
    leaf_block_counter[0] = 1

    if i == 0:
        wide_bounds[0] = leaf_bounds[0]
        wide_bounds[1] = leaf_bounds[1]
        wide_bounds[2] = leaf_bounds[2]
        wide_bounds[3] = leaf_bounds[3]
        wide_bounds[4] = leaf_bounds[4]
        wide_bounds[5] = leaf_bounds[5]

        wide_data[i] = 0
        wide_counts[i] = 1
        leaf_block_indices[0] = leaf_payloads[0]
    else:
        wide_data[i] = 0
        wide_counts[i] = EMPTY_LANE

        var b = i * AABB.STRIDE
        AABB.invalid().store6(wide_bounds, b)
        leaf_block_indices[i] = EMPTY_LANE


def collapse[
    width: Int
](
    mut ctx: DeviceContext,
    binary: GpuBinaryBoundsBvh,
    mut out: GpuBoundsBvh[width],
) raises:
    """Converts a sorted binary bounds BVH into a dense reachable wide layout.
    """
    if binary.leaf_count == 1:
        ctx.enqueue_function[collapse_single_leaf_to_wide_kernel[width]](
            binary.leaf_bounds,
            binary.leaf_payloads,
            out.wide_bounds,
            out.wide_data,
            out.wide_counts,
            out.leaf_block_indices,
            out.leaf_block_counter,
            out.wide_root,
            grid_dim=width,
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )

        ctx.synchronize()

        out.root_idx = 0
        out.node_count = 1
        out.leaf_block_count = 1
        return

    comptime if width == 2:
        ctx.enqueue_function[collapse_binary_to_wide2_direct_kernel](
            binary.leaf_bounds,
            binary.leaf_payloads,
            binary.leaf_ids,
            binary.node_meta,
            binary.node_bounds,
            out.wide_bounds,
            out.wide_data,
            out.wide_counts,
            out.leaf_block_indices,
            out.leaf_block_counter,
            out.wide_node_counter,
            out.wide_root,
            binary.internal_count,
            binary.leaf_count,
            grid_dim=binary.blocks_internal,
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )

        ctx.synchronize()

        with out.wide_root.map_to_host() as wr:
            out.root_idx = UInt32(wr[0])

        out.node_count = binary.internal_count
        out.leaf_block_count = binary.leaf_count
        return

    if binary.leaf_count > 1:
        ctx.enqueue_function[init_dense_wide_root_kernel](
            binary.node_meta,
            out.leaf_block_counter,
            out.wide_node_counter,
            out.wide_root,
            out.work_encoded_in,
            out.work_wide_in,
            binary.internal_count,
            grid_dim=binary.blocks_internal,
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )

        var frontier_count = 1
        while frontier_count > 0:
            out.frontier_count.enqueue_fill(0)

            var frontier_blocks = ceildiv(
                frontier_count,
                GPU_BOUNDS_BVH_BLOCK_SIZE,
            )

            ctx.enqueue_function[collapse_dense_frontier_kernel[width]](
                binary.leaf_bounds,
                binary.leaf_payloads,
                binary.leaf_ids,
                binary.node_meta,
                binary.node_bounds,
                binary.node_leaf_counts,
                out.work_encoded_in,
                out.work_wide_in,
                out.work_encoded_out,
                out.work_wide_out,
                out.frontier_count,
                out.wide_node_counter,
                out.leaf_block_counter,
                out.wide_bounds,
                out.wide_data,
                out.wide_counts,
                out.leaf_block_indices,
                frontier_count,
                out.max_wide_nodes,
                grid_dim=frontier_blocks,
                block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
            )

            ctx.synchronize()

            with out.frontier_count.map_to_host() as h:
                frontier_count = Int(h[0])

            swap(out.work_encoded_in, out.work_encoded_out)
            swap(out.work_wide_in, out.work_wide_out)

    with out.leaf_block_counter.map_to_host() as lbc, out.wide_node_counter.map_to_host() as wnc:
        out.root_idx = UInt32(0)
        out.node_count = Int(wnc[0])
        out.leaf_block_count = Int(lbc[0])


def init_dense_wide_root_kernel(
    node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_block_counter: UnsafePointer[UInt32, MutAnyOrigin],
    wide_node_counter: UnsafePointer[UInt32, MutAnyOrigin],
    wide_root: UnsafePointer[UInt32, MutAnyOrigin],
    work_encoded: UnsafePointer[UInt32, MutAnyOrigin],
    work_wide: UnsafePointer[UInt32, MutAnyOrigin],
    internal_count: Int,
):
    var i = global_idx.x

    if i == 0:
        leaf_block_counter[0] = 0
        wide_node_counter[0] = 1
        wide_root[0] = 0
        work_wide[0] = 0

    if i >= internal_count:
        return

    if UInt32(node_meta[_node_parent_index(UInt32(i))]) == LBVH_SENTINEL:
        work_encoded[0] = UInt32(i)
        work_wide[0] = 0


def collapse_dense_frontier_kernel[
    width: Int,
](
    leaf_bounds: UnsafePointer[Float32, MutAnyOrigin],
    leaf_payloads: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_ids: UnsafePointer[UInt32, MutAnyOrigin],
    node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    node_bounds: UnsafePointer[Float32, MutAnyOrigin],
    node_leaf_counts: UnsafePointer[UInt32, MutAnyOrigin],
    work_encoded_in: UnsafePointer[UInt32, MutAnyOrigin],
    work_wide_in: UnsafePointer[UInt32, MutAnyOrigin],
    work_encoded_out: UnsafePointer[UInt32, MutAnyOrigin],
    work_wide_out: UnsafePointer[UInt32, MutAnyOrigin],
    next_count: UnsafePointer[UInt32, MutAnyOrigin],
    wide_node_counter: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_block_counter: UnsafePointer[UInt32, MutAnyOrigin],
    wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_block_indices: UnsafePointer[UInt32, MutAnyOrigin],
    work_count: Int,
    max_wide_nodes: Int,
):
    var work_i = global_idx.x
    if work_i >= work_count:
        return

    var encoded = work_encoded_in[work_i]
    var wide_idx = work_wide_in[work_i]

    var pool = InlineArray[UInt32, width](uninitialized=True)
    pool[0] = encoded
    var p_size = 1

    # open the largest-area internal subtree while it cannot fit into one leaf block
    while p_size < width:
        var best_area = Float32(-1.0)
        var best_lane = -1

        for lane in range(width):
            if lane < p_size:
                var e = pool[lane]

                if not _is_encoded_leaf(e):
                    var subtree_leaves = _encoded_leaf_count_gpu(
                        e,
                        node_leaf_counts,
                    )

                    if subtree_leaves > UInt32(width):
                        var b = _encoded_bounds_gpu(
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
        var lane_base = Int(wide_idx) * width + lane

        if lane >= p_size:
            wide_data[lane_base] = UInt32(0)
            wide_counts[lane_base] = EMPTY_LANE
            _write_wide_lane_bounds[width](
                wide_bounds,
                wide_idx,
                lane,
                AABB.invalid(),
            )
            continue

        var e = pool[lane]
        var b = _encoded_bounds_gpu(e, leaf_bounds, leaf_ids, node_bounds)

        _write_wide_lane_bounds[width](
            wide_bounds,
            wide_idx,
            lane,
            b,
        )

        var subtree_leaves = _encoded_leaf_count_gpu(e, node_leaf_counts)

        if subtree_leaves <= UInt32(width):
            var block_idx = Atomic.fetch_add(
                leaf_block_counter,
                UInt32(1),
            )

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
            var child_wide_idx = Atomic.fetch_add(
                wide_node_counter,
                UInt32(1),
            )

            wide_data[lane_base] = child_wide_idx
            wide_counts[lane_base] = UInt32(0)

            if Int(child_wide_idx) < max_wide_nodes:
                var out_i = Atomic.fetch_add(next_count, UInt32(1))
                work_encoded_out[Int(out_i)] = e
                work_wide_out[Int(out_i)] = child_wide_idx


def collapse_binary_to_wide2_direct_kernel(
    leaf_bounds: UnsafePointer[Float32, MutAnyOrigin],
    leaf_payloads: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_ids: UnsafePointer[UInt32, MutAnyOrigin],
    node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    node_bounds: UnsafePointer[Float32, MutAnyOrigin],
    wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_block_indices: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_block_counter: UnsafePointer[UInt32, MutAnyOrigin],
    wide_node_counter: UnsafePointer[UInt32, MutAnyOrigin],
    wide_root: UnsafePointer[UInt32, MutAnyOrigin],
    internal_count: Int,
    leaf_count: Int,
):
    var node_i = global_idx.x

    if node_i == 0:
        leaf_block_counter[0] = UInt32(leaf_count)
        wide_node_counter[0] = UInt32(internal_count)

    if node_i >= internal_count:
        return

    var node_idx = UInt32(node_i)

    if UInt32(node_meta[_node_parent_index(node_idx)]) == LBVH_SENTINEL:
        wide_root[0] = node_idx

    var left = _node_left(node_meta, node_idx)
    var right = _node_right(node_meta, node_idx)

    # Lane 0 = left child.
    var left_bounds = _encoded_bounds_gpu(
        left,
        leaf_bounds,
        leaf_ids,
        node_bounds,
    )
    _write_wide_lane_bounds[2](
        wide_bounds,
        node_idx,
        0,
        left_bounds,
    )

    if _is_encoded_leaf(left):
        var sorted_leaf_idx = _encoded_index(left)
        var item_idx = UInt32(leaf_ids[Int(sorted_leaf_idx)])
        var payload = UInt32(leaf_payloads[Int(item_idx)])

        wide_data[Int(node_idx) * 2 + 0] = sorted_leaf_idx
        wide_counts[Int(node_idx) * 2 + 0] = UInt32(1)

        leaf_block_indices[Int(sorted_leaf_idx) * 2 + 0] = payload
        leaf_block_indices[Int(sorted_leaf_idx) * 2 + 1] = EMPTY_LANE
    else:
        wide_data[Int(node_idx) * 2 + 0] = _encoded_index(left)
        wide_counts[Int(node_idx) * 2 + 0] = UInt32(0)

    # Lane 1 = right child.
    var right_bounds = _encoded_bounds_gpu(
        right,
        leaf_bounds,
        leaf_ids,
        node_bounds,
    )
    _write_wide_lane_bounds[2](
        wide_bounds,
        node_idx,
        1,
        right_bounds,
    )

    if _is_encoded_leaf(right):
        var sorted_leaf_idx = _encoded_index(right)
        var item_idx = UInt32(leaf_ids[Int(sorted_leaf_idx)])
        var payload = UInt32(leaf_payloads[Int(item_idx)])

        wide_data[Int(node_idx) * 2 + 1] = sorted_leaf_idx
        wide_counts[Int(node_idx) * 2 + 1] = UInt32(1)

        leaf_block_indices[Int(sorted_leaf_idx) * 2 + 0] = payload
        leaf_block_indices[Int(sorted_leaf_idx) * 2 + 1] = EMPTY_LANE
    else:
        wide_data[Int(node_idx) * 2 + 1] = _encoded_index(right)
        wide_counts[Int(node_idx) * 2 + 1] = UInt32(0)
