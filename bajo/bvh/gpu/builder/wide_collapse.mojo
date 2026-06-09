from std.math import ceildiv
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
    origin: MutOrigin,
    //,
    width: Int,
](
    wide_bounds: UnsafePointer[Float32, origin],
    wide_node_idx: UInt32,
    lane: Int,
    bounds: AABB,
):
    var b = (Int(wide_node_idx) * width + lane) * AABB.STRIDE
    bounds.store6(wide_bounds, b)


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
    """Converts a sorted binary bounds BVH into a precomputed wide layout.

    Every binary internal node gets a possible wide node with the same index.
    This is not a dense reachable layout: some wide nodes may be unreachable
    because an ancestor wide node opened through their binary subtree.

    The benefit is that conversion is deterministic, non-polling, and does not
    require a host-side frontier loop.
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
        out.leaf_block_counter,
        out.wide_node_counter,
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
        out.wide_bounds,
        out.wide_data,
        out.wide_counts,
        out.wide_root,
        binary.internal_count,
        grid_dim=binary.blocks_internal,
        block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
    )

    ctx.synchronize()

    with out.wide_root.map_to_host() as wr:
        out.root_idx = UInt32(wr[0])

    out.node_count = binary.internal_count
    out.leaf_block_count = binary.leaf_count


def init_precomputed_wide_leaf_blocks_kernel[
    width: Int,
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

    if i == 0:
        leaf_block_counter[0] = UInt32(leaf_count)
        wide_node_counter[0] = UInt32(internal_count)

    if i >= leaf_count:
        return

    var sorted_leaf_idx = UInt32(i)
    var item_idx = UInt32(leaf_ids[i])
    var payload = UInt32(leaf_payloads[Int(item_idx)])

    var base = i * width
    leaf_block_indices[base] = payload

    comptime for lane in range(width):
        if lane > 0:
            leaf_block_indices[base + lane] = EMPTY_LANE


def collapse_precomputed_wide_kernel[
    width: Int,
](
    leaf_bounds: UnsafePointer[Float32, MutAnyOrigin],
    leaf_ids: UnsafePointer[UInt32, MutAnyOrigin],
    node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    node_bounds: UnsafePointer[Float32, MutAnyOrigin],
    wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
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
    # the wide node has width entries, or only leaves remain.
    while p_size < width:
        var best_area = Float32(-1.0)
        var best_lane = -1

        for lane in range(width):
            if lane < p_size:
                var e = pool[lane]

                if not _is_encoded_leaf(e):
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
        var lane_base = Int(node_idx) * width + lane

        if lane >= p_size:
            wide_data[lane_base] = UInt32(0)
            wide_counts[lane_base] = EMPTY_LANE
            _write_wide_lane_bounds[width](
                wide_bounds,
                node_idx,
                lane,
                AABB.invalid(),
            )
            continue

        var e = pool[lane]
        var b = _encoded_bounds_gpu(e, leaf_bounds, leaf_ids, node_bounds)

        _write_wide_lane_bounds[width](
            wide_bounds,
            node_idx,
            lane,
            b,
        )

        if _is_encoded_leaf(e):
            var sorted_leaf_idx = _encoded_index(e)
            wide_data[lane_base] = sorted_leaf_idx
            wide_counts[lane_base] = UInt32(1)
        else:
            wide_data[lane_base] = _encoded_index(e)
            wide_counts[lane_base] = UInt32(0)
