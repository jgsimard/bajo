from bajo.core import AABB, Frame
from std.gpu import WARP_SIZE


comptime HPLOC_SEARCH_RADIUS = 8
comptime HPLOC_MERGING_THRESHOLD = WARP_SIZE / 2


comptime HPLOC_NODE_META_STRIDE = 4
comptime HPLOC_NODE_PARENT = 0
comptime HPLOC_NODE_LEFT = 1
comptime HPLOC_NODE_RIGHT = 2
comptime HPLOC_NODE_LEAF_ID = 3

comptime HPLOC_STATUS_OK = 0
comptime HPLOC_STATUS_NO_PROGRESS = 1
comptime HPLOC_STATUS_INVALID_RESULT = 2


@always_inline
def _hploc_meta_base(node_idx: UInt32) -> Int:
    return Int(node_idx) * HPLOC_NODE_META_STRIDE


@always_inline
def _hploc_load_bounds(
    node_bounds: ImmSpan[Float32, _], node_idx: UInt32
) -> AABB[Frame.WORLD]:
    return AABB[Frame.WORLD].load6(node_bounds, Int(node_idx) * AABB.STRIDE)


@always_inline
def _hploc_store_bounds(
    node_bounds: MutSpan[Float32, _],
    node_idx: UInt32,
    bounds: AABB[Frame.WORLD],
):
    bounds.store6(node_bounds, Int(node_idx) * AABB.STRIDE)
