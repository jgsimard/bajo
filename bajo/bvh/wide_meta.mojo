from bajo.bvh.constants import EMPTY_LANE, WideNode


# Six count bits represent internal nodes (count zero) and leaf packets through
# width 32 while retaining 26 bits (67 million entries) for local indices.
comptime WIDE_META_INDEX_MASK = UInt32(0x03FFFFFF)
comptime WIDE_META_COUNT_SHIFT = 26


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
    debug_assert["safe", _use_compiler_assume=True](
        data <= WIDE_META_INDEX_MASK, "wide BVH metadata index exceeds 26 bits"
    )
    debug_assert["safe", _use_compiler_assume=True](
        count <= UInt32(32), "wide BVH metadata count exceeds leaf width 32"
    )
    return (count << WIDE_META_COUNT_SHIFT) | (data & WIDE_META_INDEX_MASK)


@always_inline
def _wide_meta_data(meta: UInt32) -> UInt32:
    return meta & WIDE_META_INDEX_MASK


@always_inline
def _wide_meta_count(meta: UInt32) -> UInt32:
    if meta == EMPTY_LANE:
        return EMPTY_LANE
    return meta >> WIDE_META_COUNT_SHIFT
