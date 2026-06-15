from bajo.bvh.constants import EMPTY_LANE


comptime WIDE_META_INDEX_MASK = UInt32(0x0FFFFFFF)
comptime WIDE_META_COUNT_SHIFT = 28


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
