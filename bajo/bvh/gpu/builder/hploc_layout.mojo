from bajo.core import AABB
from max.gpu import WARP_SIZE


comptime HPLOC_SEARCH_RADIUS = 8
comptime HPLOC_MERGING_THRESHOLD = WARP_SIZE / 2


comptime HPLOC_STATUS_OK = 0
comptime HPLOC_STATUS_NO_PROGRESS = 1
comptime HPLOC_STATUS_INVALID_RESULT = 2


@always_inline
def _hploc_load_bounds(
    node_bounds: ImmSpan[Float32, _], node_idx: UInt32
) -> AABB[.WORLD]:
    return AABB[.WORLD].load6(node_bounds, Int(node_idx) * AABB.STRIDE)


@always_inline
def _hploc_store_bounds(
    node_bounds: MutSpan[Float32, _],
    node_idx: UInt32,
    bounds: AABB[.WORLD],
):
    bounds.store6(node_bounds, Int(node_idx) * AABB.STRIDE)
