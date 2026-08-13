from bajo.core import AABB, AxisAlignedBoundingBox, Point3f32, Frame
from bajo.bvh.constants import EMPTY_LANE, f32_max
from bajo.bvh.cpu.builder import BoundsBvhBuilder, BoundsItem
from bajo.bvh.tagged_ref import (
    encode_internal_ref,
    encode_leaf_ref,
    is_leaf_ref,
    decode_ref_index,
)


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
        is_leaf_ref(data[i]) == False  -> child node index
        is_leaf_ref(data[i]) == True   -> leaf payload index
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


struct _WideCollapseDp:
    """Temporary DP storage for binary -> wide BVH collapse.

    For binary node n and k in [1, width]:

        costs[n * width + k - 1]

    is the minimum downstream wide-node SAH numerator when subtree n
    occupies exactly k lanes in its parent's wide node.

    k == 1 means n remains as one lane and, if internal, becomes the root
    of another wide node.

    k > 1 means n is flattened into k lanes of the current wide node.

    Leaf primitive-intersection cost is omitted because the binary leaves
    are unchanged by collapse, so it is constant across all alternatives.
    """

    var costs: List[Float32]
    var splits: List[UInt8]
    var best_slots: List[UInt8]
    var max_slots: List[UInt8]

    def __init__(out self, node_count: Int, width: Int):
        self.costs = [f32_max for _ in range(node_count * width)]
        self.splits = [UInt8(0) for _ in range(node_count * width)]
        self.best_slots = [UInt8(1) for _ in range(node_count)]
        self.max_slots = [UInt8(1) for _ in range(node_count)]


struct BoundsBvh[frame: Frame, width: SIMDLength](Copyable):
    """Generic compact wide/lane BVH.

    The generic constructor stores leaf ranges and builder item arrays. Typed
    BVHs use the packer constructor so collapse emits their final leaf-block
    references directly and leaves those construction-only arrays empty.
    """

    var nodes: List[WideBvhNode[Self.frame, Self.width]]
    var child_masks: List[UInt32]
    var leaf_ranges: List[WideLeafRange]
    var item_indices: List[UInt32]
    var item_payloads: List[UInt32]

    def __init__(out self, bvh: BoundsBvhBuilder):
        self.nodes = List[WideBvhNode[Self.frame, Self.width]]()
        self.child_masks = List[UInt32]()
        self.item_indices = bvh.item_indices.copy()
        self.item_payloads = [item.payload for item in bvh.items]
        var leaf_ranges = List[WideLeafRange]()

        @always_inline
        def pack_leaf_range(
            first_item: UInt32, item_count: UInt32
        ) {imm, mut leaf_ranges} -> UInt32:
            var leaf_range_idx = UInt32(len(leaf_ranges))
            leaf_ranges.append(WideLeafRange(first_item, item_count))
            return leaf_range_idx

        self.leaf_ranges = List[WideLeafRange]()

        if bvh.nodes_used > 0:
            self._collapse_root[pack_leaves_before_children=False](
                bvh, pack_leaf_range
            )

        self.leaf_ranges = leaf_ranges^

    def __init__[
        PackLeafFn: def(UInt32, UInt32) -> UInt32
    ](out self, bvh: BoundsBvhBuilder, ref pack_leaf_fn: PackLeafFn):
        """Collapse a binary BVH while packing its typed leaf payloads.

        Unlike the generic constructor, this path does not materialize
        construction-time leaf ranges or copy the builder's item arrays. The
        packer is called exactly when a binary leaf is written into a wide
        node and its returned index becomes the tagged leaf payload.
        """
        self.nodes = List[WideBvhNode[Self.frame, Self.width]]()
        self.child_masks = List[UInt32]()
        self.leaf_ranges = List[WideLeafRange]()
        self.item_indices = List[UInt32]()
        self.item_payloads = List[UInt32]()

        if bvh.nodes_used > 0:
            self._collapse_root[pack_leaves_before_children=True](
                bvh, pack_leaf_fn
            )

    def _collapse_root[
        PackLeafFn: def(UInt32, UInt32) -> UInt32,
        pack_leaves_before_children: Bool,
    ](mut self, bvh: BoundsBvhBuilder, ref pack_leaf_fn: PackLeafFn):
        comptime if Self.width > 2:
            var dp = _WideCollapseDp(Int(bvh.nodes_used), Int(Self.width))
            self._compute_collapse_dp(bvh, 0, dp)
            _ = self._collapse_dp[
                pack_leaves_before_children=pack_leaves_before_children
            ](bvh, 0, dp, pack_leaf_fn)

        else:
            _ = self._collapse_greedy[
                pack_leaves_before_children=pack_leaves_before_children
            ](bvh, 0, pack_leaf_fn)

    def _compute_collapse_dp(
        self,
        bvh: BoundsBvhBuilder,
        bin_idx: UInt32,
        mut dp: _WideCollapseDp,
    ):
        """Bottom-up SAH-optimal collapse constrained to the binary topology.

        The cost is proportional to the sum of surface areas of wide internal
        nodes. This is sufficient for comparing collapse alternatives because:

        * Self.width is fixed, so one wide-node traversal has fixed cost.
        * Binary leaves and primitive ranges are unchanged by collapse, so leaf
          intersection cost is constant across alternatives.
        """
        var node_i = Int(bin_idx)
        var base = node_i * Int(Self.width)
        ref node = bvh.nodes[node_i]

        if node.is_leaf():
            dp.costs[base] = 0.0
            dp.best_slots[node_i] = UInt8(1)
            dp.max_slots[node_i] = UInt8(1)
            return

        var left_idx = node.left_child()
        var right_idx = node.right_child()

        self._compute_collapse_dp(bvh, left_idx, dp)
        self._compute_collapse_dp(bvh, right_idx, dp)

        var left_i = Int(left_idx)
        var right_i = Int(right_idx)
        var left_base = left_i * Int(Self.width)
        var right_base = right_i * Int(Self.width)

        var left_max = Int(dp.max_slots[left_i])
        var right_max = Int(dp.max_slots[right_i])

        var subtree_max = left_max + right_max
        if subtree_max > Int(Self.width):
            subtree_max = Int(Self.width)
        dp.max_slots[node_i] = UInt8(subtree_max)

        # k > 1: flatten this binary node into exactly k lanes of the current
        # wide node. Split those k lanes optimally between left and right.
        for k in range(2, subtree_max + 1):
            var first_left_slots = 1
            var required_left_slots = k - right_max
            if required_left_slots > first_left_slots:
                first_left_slots = required_left_slots

            var last_left_slots = k - 1
            if last_left_slots > left_max:
                last_left_slots = left_max

            var best_cost = f32_max
            var best_left_slots = 0

            for left_slots in range(first_left_slots, last_left_slots + 1):
                var right_slots = k - left_slots
                var candidate_cost = (
                    dp.costs[left_base + left_slots - 1]
                    + dp.costs[right_base + right_slots - 1]
                )

                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_left_slots = left_slots

            debug_assert["safe", _use_compiler_assume=True](
                best_left_slots > 0,
                "wide-collapse DP could not find a valid slot split",
            )
            dp.costs[base + k - 1] = best_cost
            dp.splits[base + k - 1] = UInt8(best_left_slots)

        # k == 1: keep this binary node as one parent lane. It therefore
        # becomes a wide internal node itself. Choose the best arity for that
        # wide node, and add its traversal probability numerator: area(node).
        #
        # Start from the largest feasible arity so exact ties prefer fuller
        # nodes rather than introducing unnecessary under-filled nodes.
        var best_k = subtree_max
        var best_children_cost = dp.costs[base + best_k - 1]

        for k in range(2, subtree_max):
            var candidate_cost = dp.costs[base + k - 1]
            if candidate_cost < best_children_cost:
                best_children_cost = candidate_cost
                best_k = k

        dp.best_slots[node_i] = UInt8(best_k)
        dp.costs[base] = node.surface_area() + best_children_cost

    def _dp_frontier(
        self,
        bvh: BoundsBvhBuilder,
        bin_idx: UInt32,
        ref dp: _WideCollapseDp,
    ) -> Tuple[Array[UInt32, Self.width], Int]:
        """Reconstruct the DP-selected frontier for one wide node."""
        var pool = Array[UInt32, Self.width](fill=bin_idx)
        var slot_budget = Array[UInt8, Self.width](fill=UInt8(1))
        var p_size = 1

        ref root = bvh.nodes[Int(bin_idx)]
        if not root.is_leaf():
            slot_budget[0] = dp.best_slots[Int(bin_idx)]

        while True:
            var expand_i = -1

            # Any lane with budget > 1 still needs to be flattened.
            for i in range(p_size):
                if slot_budget[i] > UInt8(1):
                    expand_i = i
                    break

            if expand_i == -1:
                break

            var candidate_idx = pool[expand_i]
            var candidate_i = Int(candidate_idx)
            var slots = Int(slot_budget[expand_i])
            ref candidate = bvh.nodes[candidate_i]

            debug_assert["safe", _use_compiler_assume=True](
                not candidate.is_leaf(),
                "wide-collapse DP assigned multiple slots to a leaf",
            )

            var split_idx = candidate_i * Int(Self.width) + slots - 1
            var left_slots = Int(dp.splits[split_idx])
            var right_slots = slots - left_slots

            debug_assert["safe", _use_compiler_assume=True](
                left_slots > 0 and right_slots > 0,
                "wide-collapse DP reconstructed an invalid slot split",
            )
            debug_assert["safe", _use_compiler_assume=True](
                p_size < Int(Self.width),
                "wide-collapse DP frontier exceeded BVH width",
            )

            pool[expand_i] = candidate.left_child()
            slot_budget[expand_i] = UInt8(left_slots)

            pool[p_size] = candidate.right_child()
            slot_budget[p_size] = UInt8(right_slots)
            p_size += 1

        return (pool^, p_size)

    def _collapse_dp[
        PackLeafFn: def(UInt32, UInt32) -> UInt32,
        pack_leaves_before_children: Bool,
    ](
        mut self,
        bvh: BoundsBvhBuilder,
        bin_idx: UInt32,
        ref dp: _WideCollapseDp,
        ref pack_leaf_fn: PackLeafFn,
    ) -> UInt32:
        var wide_idx = len(self.nodes)
        self.nodes.append(WideBvhNode[Self.frame, Self.width]())
        self.child_masks.append(0)

        # var pool, p_size = self._dp_frontier(bvh, bin_idx, dp)
        var frontier = self._dp_frontier(bvh, bin_idx, dp)
        var pool = frontier[0].copy()
        var p_size = frontier[1]

        var child_mask = UInt32(0)
        for i in range(p_size):
            child_mask |= UInt32(1) << UInt32(i)
        self.child_masks[wide_idx] = child_mask

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
                            self._collapse_dp[pack_leaves_before_children=True](
                                bvh, pool[i], dp, pack_leaf_fn
                            )
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
                            self._collapse_dp[
                                pack_leaves_before_children=False
                            ](bvh, pool[i], dp, pack_leaf_fn)
                        )

        self.nodes[wide_idx] = node^
        return UInt32(wide_idx)

    def _collapse_greedy[
        PackLeafFn: def(UInt32, UInt32) -> UInt32,
        pack_leaves_before_children: Bool,
    ](
        mut self,
        bvh: BoundsBvhBuilder,
        bin_idx: UInt32,
        ref pack_leaf_fn: PackLeafFn,
    ) -> UInt32:
        """Original largest-area greedy collapse, retained for A/B tests."""
        var wide_idx = len(self.nodes)
        self.nodes.append(WideBvhNode[Self.frame, Self.width]())
        self.child_masks.append(0)

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

        var child_mask = UInt32(0)
        for i in range(p_size):
            child_mask |= UInt32(1) << UInt32(i)
        self.child_masks[wide_idx] = child_mask

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
                            self._collapse_greedy[
                                pack_leaves_before_children=True
                            ](bvh, pool[i], pack_leaf_fn)
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
                            self._collapse_greedy[
                                pack_leaves_before_children=False
                            ](bvh, pool[i], pack_leaf_fn)
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
