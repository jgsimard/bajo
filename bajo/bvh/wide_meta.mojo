from bajo.bvh.constants import EMPTY_LANE, WideNode


# Five count bits support independent leaf packets through width 16 while
# retaining 27 bits (134 million entries) for node and leaf-block indices.
comptime WIDE_META_INDEX_MASK = UInt32(0x07FFFFFF)
comptime WIDE_META_COUNT_SHIFT = 27


@always_inline
def _wide_node_base[width: SIMDLength](node_idx: UInt32) -> Int:
    return Int(node_idx) * width * WideNode.CHILD_STRIDE


@always_inline
def _wide_node_index[
    width: SIMDLength
](node_idx: UInt32, field: Int, lane: Int) -> Int:
    return _wide_node_base[width](node_idx) + field * width + lane


@always_inline
def _pack_wide_meta(data: UInt32, count: UInt32) -> UInt32:
    if count == EMPTY_LANE:
        return EMPTY_LANE
    return (count << WIDE_META_COUNT_SHIFT) | (data & WIDE_META_INDEX_MASK)


@always_inline
def _wide_meta_data(meta: UInt32) -> UInt32:
    return meta & WIDE_META_INDEX_MASK


@always_inline
def _wide_meta_count(meta: UInt32) -> UInt32:
    if meta == EMPTY_LANE:
        return EMPTY_LANE
    return meta >> WIDE_META_COUNT_SHIFT
