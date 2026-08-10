from bajo.core import AABB, AxisAlignedBoundingBox, Point3f32, Frame
from bajo.bvh.constants import EMPTY_LANE
from bajo.bvh.cpu.builder import BoundsBvhBuilder, BoundsItem

comptime BVH_LEAF_REF_BIT = UInt32(0x80000000)
comptime BVH_REF_INDEX_MASK = UInt32(0x7FFFFFFF)


def encode_internal_ref(index: UInt32) -> UInt32:
    debug_assert["safe", _use_compiler_assume=True](
        index < BVH_LEAF_REF_BIT,
        "BVH internal node index exceeds tagged-reference capacity",
    )
    return index


def encode_leaf_ref(index: UInt32) -> UInt32:
    # BVH_LEAF_REF_BIT | BVH_REF_INDEX_MASK equals EMPTY_LANE
    debug_assert["safe", _use_compiler_assume=True](
        index < BVH_REF_INDEX_MASK,
        "BVH leaf index exceeds tagged-reference capacity",
    )
    return BVH_LEAF_REF_BIT | index


def is_leaf_ref(x: UInt32) -> Bool:
    return (x & BVH_LEAF_REF_BIT) != 0


def decode_ref_index(x: UInt32) -> UInt32:
    return x & BVH_REF_INDEX_MASK


@fieldwise_init
struct WideLeafRange(Copyable):
    """Construction-time item range referenced by a tagged leaf."""

    var first_item: UInt32
    var item_count: UInt32


@fieldwise_init
struct WideBvhNode[frame: Frame, width: SIMDLength](Copyable):
    """Compact lane node used by BoundsBvh traversal.

    Lane encoding in data:
        data[i] == EMPTY_LANE             -> unused lane
        data[i] & BVH_LEAF_REF_BIT == 0  -> child node index
        data[i] & BVH_LEAF_REF_BIT != 0  -> leaf payload index
    """

    var aabb: AxisAlignedBoundingBox[DType.float32, Self.frame, Self.width]
    var data: SIMD[DType.uint32, Self.width]

    def __init__(out self):
        self.aabb = AxisAlignedBoundingBox[
            DType.float32, Self.frame, Self.width
        ].invalid()
        self.data = SIMD[DType.uint32, Self.width](EMPTY_LANE)


@always_inline
def _checked_typed_leaf_range[
    leaf_width: SIMDLength
](first_item: UInt32, item_count: UInt32, item_index_count: Int) -> Tuple[
    Int, Int
]:
    """Validate and convert a builder leaf range for typed payload packing."""
    var first = Int(first_item)
    var count = Int(item_count)

    debug_assert["safe", _use_compiler_assume=True](
        count <= Int(leaf_width),
        "typed BVH leaf exceeds SIMD width",
    )
    debug_assert["safe", _use_compiler_assume=True](
        first <= item_index_count and count <= item_index_count - first,
        "typed BVH leaf range is outside item indices",
    )
    return (first, count)


struct BoundsBvh[frame: Frame, width: SIMDLength](Copyable):
    """Generic compact wide/lane BVH.

    The generic constructor stores leaf ranges and builder item arrays. Typed
    BVHs use the packer constructor so collapse emits their final leaf-block
    references directly and leaves those construction-only arrays empty.
    """

    var nodes: List[WideBvhNode[Self.frame, Self.width]]
    var leaf_ranges: List[WideLeafRange]
    var item_indices: List[UInt32]
    var item_payloads: List[UInt32]

    def __init__(out self, bvh: BoundsBvhBuilder):
        self.nodes = List[WideBvhNode[Self.frame, Self.width]]()
        self.item_indices = bvh.item_indices.copy()
        self.item_payloads = [item.payload for item in bvh.items]
        var leaf_ranges = List[WideLeafRange]()

        @always_inline
        def pack_leaf_range(
            first_item: UInt32, item_count: UInt32
        ) capturing -> UInt32:
            var leaf_range_idx = UInt32(len(leaf_ranges))
            leaf_ranges.append(WideLeafRange(first_item, item_count))
            return leaf_range_idx

        self.leaf_ranges = List[WideLeafRange]()

        if bvh.nodes_used > 0:
            _ = self._collapse[
                pack_leaf_range, pack_leaves_before_children=False
            ](bvh, 0)

        self.leaf_ranges = leaf_ranges^

    def __init__[
        pack_leaf_fn: def(UInt32, UInt32) capturing -> UInt32
    ](out self, bvh: BoundsBvhBuilder):
        """Collapse a binary BVH while packing its typed leaf payloads.

        Unlike the generic constructor, this path does not materialize
        construction-time leaf ranges or copy the builder's item arrays. The
        packer is called exactly when a binary leaf is written into a wide
        node and its returned index becomes the tagged leaf payload.
        """
        self.nodes = List[WideBvhNode[Self.frame, Self.width]]()
        self.leaf_ranges = List[WideLeafRange]()
        self.item_indices = List[UInt32]()
        self.item_payloads = List[UInt32]()

        if bvh.nodes_used > 0:
            _ = self._collapse[pack_leaf_fn, pack_leaves_before_children=True](
                bvh, 0
            )

    def _collapse[
        pack_leaf_fn: def(UInt32, UInt32) capturing -> UInt32,
        pack_leaves_before_children: Bool,
    ](mut self, bvh: BoundsBvhBuilder, bin_idx: UInt32) -> UInt32:
        var wide_idx = len(self.nodes)
        self.nodes.append(WideBvhNode[Self.frame, Self.width]())

        var pool = Array[UInt32, Self.width](fill=bin_idx)
        var p_size = 1

        # Pull up the largest internal nodes until the wide node is full or no
        # internal node remains.
        while p_size < Self.width:
            var best_a: Float32 = -1.0
            var best_i: Int = -1

            for i in range(p_size):
                ref candidate = bvh.nodes[Int(pool[i])]

                if not candidate.is_leaf():
                    var a = candidate.surface_area()

                    if a > best_a:
                        best_a = a
                        best_i = i

            if best_i == -1:
                break

            ref n = bvh.nodes[Int(pool[best_i])]
            pool[best_i] = n.left_child()
            pool[p_size] = n.right_child()
            p_size += 1

        var node = WideBvhNode[Self.frame, Self.width]()

        comptime for i in range(Self.width):
            if i < p_size:
                ref n = bvh.nodes[Int(pool[i])]

                node.aabb._min.x[i] = n.aabb._min.x
                node.aabb._min.y[i] = n.aabb._min.y
                node.aabb._min.z[i] = n.aabb._min.z

                node.aabb._max.x[i] = n.aabb._max.x
                node.aabb._max.y[i] = n.aabb._max.y
                node.aabb._max.z[i] = n.aabb._max.z

        comptime if pack_leaves_before_children:
            # Typed construction packs every leaf in this node before
            # descending, preserving node-order typed-block indices.
            comptime for i in range(Self.width):
                if i < p_size:
                    ref n = bvh.nodes[Int(pool[i])]
                    if n.is_leaf():
                        node.data[i] = encode_leaf_ref(
                            pack_leaf_fn(n.first_item(), n.item_count)
                        )

            comptime for i in range(Self.width):
                if i < p_size:
                    ref n = bvh.nodes[Int(pool[i])]
                    if not n.is_leaf():
                        node.data[i] = encode_internal_ref(
                            self._collapse[
                                pack_leaf_fn,
                                pack_leaves_before_children=True,
                            ](bvh, pool[i])
                        )
        else:
            # Generic construction retains its depth-first leaf-range order.
            comptime for i in range(Self.width):
                if i < p_size:
                    ref n = bvh.nodes[Int(pool[i])]
                    if n.is_leaf():
                        node.data[i] = encode_leaf_ref(
                            pack_leaf_fn(n.first_item(), n.item_count)
                        )
                    else:
                        node.data[i] = encode_internal_ref(
                            self._collapse[
                                pack_leaf_fn,
                                pack_leaves_before_children=False,
                            ](bvh, pool[i])
                        )

        self.nodes[wide_idx] = node^
        return UInt32(wide_idx)

    def root_bounds(self) -> AABB[Self.frame]:
        var out = AABB[Self.frame].invalid()

        if len(self.nodes) > 0:
            ref root = self.nodes[0]

            comptime for lane in range(Self.width):
                if root.data[lane] != EMPTY_LANE:
                    out.grow(
                        Point3f32[Self.frame](
                            root.aabb._min.x[lane],
                            root.aabb._min.y[lane],
                            root.aabb._min.z[lane],
                        ),
                        Point3f32[Self.frame](
                            root.aabb._max.x[lane],
                            root.aabb._max.y[lane],
                            root.aabb._max.z[lane],
                        ),
                    )
        return out
