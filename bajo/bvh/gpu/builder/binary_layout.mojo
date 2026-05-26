from bajo.core.aabb import AABB
from bajo.bvh.constants import (
    LBVH_LEAF_FLAG,
    LBVH_INDEX_MASK,
    BOUNDS_STRIDE,
)


comptime LBVH_NODE_META_STRIDE = 4
comptime LBVH_NODE_PARENT = 0
comptime LBVH_NODE_LEFT = 1
comptime LBVH_NODE_RIGHT = 2
comptime LBVH_NODE_FENCE = 3
comptime LBVH_NODE_BOUNDS_STRIDE = 12

comptime GPU_BOUNDS_BVH_BLOCK_SIZE = 128


def _node_meta_base(node_idx: UInt32) -> Int:
    return Int(node_idx) * LBVH_NODE_META_STRIDE


def _node_bounds_base(node_idx: UInt32) -> Int:
    return Int(node_idx) * LBVH_NODE_BOUNDS_STRIDE


def _node_parent_index(node_idx: UInt32) -> Int:
    return _node_meta_base(node_idx) + LBVH_NODE_PARENT


def _node_left_index(node_idx: UInt32) -> Int:
    return _node_meta_base(node_idx) + LBVH_NODE_LEFT


def _node_right_index(node_idx: UInt32) -> Int:
    return _node_meta_base(node_idx) + LBVH_NODE_RIGHT


def _node_left(
    node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    node_idx: UInt32,
) -> UInt32:
    return UInt32(node_meta[_node_left_index(node_idx)])


def _node_right(
    node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    node_idx: UInt32,
) -> UInt32:
    return UInt32(node_meta[_node_right_index(node_idx)])


def _load_leaf_bounds_host(
    leaf_bounds: List[Float32],
    leaf_idx: UInt32,
) -> AABB:
    var b = Int(leaf_idx) * BOUNDS_STRIDE
    return AABB.load6(leaf_bounds.unsafe_ptr(), b)


def _load_internal_bounds_host(
    node_bounds: List[Float32],
    node_idx: UInt32,
) -> AABB:
    var b = Int(node_idx) * LBVH_NODE_BOUNDS_STRIDE
    b1 = AABB.load6(node_bounds.unsafe_ptr(), b)
    b2 = AABB.load6(node_bounds.unsafe_ptr(), b + 6)
    return AABB.merge(b1, b2)


def _is_encoded_leaf(encoded: UInt32) -> Bool:
    return (encoded & LBVH_LEAF_FLAG) != 0


def _encoded_index(encoded: UInt32) -> UInt32:
    return encoded & LBVH_INDEX_MASK


def _write_child_bounds(
    node_bounds: UnsafePointer[Float32, MutAnyOrigin],
    parent: UInt32,
    write_left: Bool,
    bounds: AABB,
):
    var b = _node_bounds_base(parent)
    if not write_left:
        b += 6
    bounds.store6(node_bounds, b)


def _load_and_union_node_bounds(
    node_bounds: UnsafePointer[Float32, MutAnyOrigin],
    parent: UInt32,
) -> AABB:
    var b = _node_bounds_base(parent)
    b1 = AABB.load6(node_bounds, b)
    b2 = AABB.load6(node_bounds, b + 6)
    return AABB.merge(b1, b2)
