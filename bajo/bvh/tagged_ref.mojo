"""Shared high-bit tagged references for binary and wide BVH children."""

comptime _BVH_LEAF_FLAG = UInt32(0x80000000)
comptime _BVH_INDEX_MASK = UInt32(0x7FFFFFFF)


def encode_internal_ref(index: UInt32) -> UInt32:
    debug_assert["safe", _use_compiler_assume=True](
        index < _BVH_LEAF_FLAG,
        "BVH internal node index exceeds tagged-reference capacity",
    )
    return index


def encode_leaf_ref(index: UInt32) -> UInt32:
    debug_assert["safe", _use_compiler_assume=True](
        index < _BVH_INDEX_MASK,
        "BVH leaf index exceeds tagged-reference capacity",
    )
    return _BVH_LEAF_FLAG | index


def is_leaf_ref(value: UInt32) -> Bool:
    return (value & _BVH_LEAF_FLAG) != 0


def decode_ref_index(value: UInt32) -> UInt32:
    return value & _BVH_INDEX_MASK
