from max.algorithm import parallelize
from std.atomic import Atomic, Ordering
from std.sys import num_performance_cores

from bajo.core import AABB, vmin, vmax, longest_axis, Frame
from bajo.sort.cpu.nth_element import nth_element

from .sah import (
    BoundsPartitionResult,
    _calculate_partition_bounds,
    _find_sah_split,
    _partition_items_by_bin,
)
from .lbvh import _build_lbvh


comptime PARALLEL_SAH_MIN_ITEMS = UInt32(4096)
comptime PARALLEL_SAH_FRONTIER_DEPTH = 3


struct BoundsBvhBuilder[frame: Frame, leaf_size: Int](Copyable):
    """Generic binary BVH builder over AABBs/items.

    Build modes:
        "median" -> top-down spatial median
        "sah"    -> top-down binned SAH
        "lbvh"   -> sorted-Morton recursive LBVH
    """

    var nodes: List[BoundsBvhNode[Self.frame]]
    var item_indices: List[UInt32]
    var items: List[BoundsItem[Self.frame]]
    var item_count: UInt32
    var nodes_used: UInt32

    def __init__(out self, var items: List[BoundsItem[Self.frame]]):
        self.items = items^
        self.item_count = UInt32(len(self.items))
        debug_assert["safe", _use_compiler_assume=True](self.item_count > 0)

        self.item_indices = [i for i in range(self.item_count)]

        var max_nodes = Int(self.item_count * 2 - 1)
        self.nodes = List[BoundsBvhNode[Self.frame]](capacity=max_nodes)
        self.nodes.append(BoundsBvhNode[Self.frame]())

        # build() initializes the root for its selected topology.
        self.nodes_used = 1

    def allocate_children(mut self) -> UInt32:
        var left_child_idx = self.nodes_used
        self.nodes_used += 2

        if Int(self.nodes_used) > len(self.nodes):
            debug_assert["safe", _use_compiler_assume=True](
                Int(left_child_idx) == len(self.nodes),
                "BVH nodes must be allocated in index order",
            )
            self.nodes.append(BoundsBvhNode[Self.frame]())
            self.nodes.append(BoundsBvhNode[Self.frame]())

        return left_child_idx

    def update_node_bounds(mut self, node_idx: UInt32):
        ref node = self.nodes[Int(node_idx)]
        node.aabb = AABB[Self.frame].invalid()

        var first = Int(node.first_item())
        for i in range(Int(node.item_count)):
            var item_idx = Int(self.item_indices[first + i])
            self.items[item_idx].grow_into(node.aabb)

    def update_node_bounds_and_centroid_bounds(
        mut self, node_idx: UInt32
    ) -> AABB[Self.frame]:
        ref node = self.nodes[Int(node_idx)]
        node.aabb = AABB[Self.frame].invalid()
        var centroid_bounds = AABB[Self.frame].invalid()

        var first = Int(node.first_item())
        for i in range(Int(node.item_count)):
            var item_idx = Int(self.item_indices[first + i])
            ref item = self.items[item_idx]
            item.grow_into(node.aabb)
            centroid_bounds.grow(item.bounds.centroid())

        return centroid_bounds

    def build[split_method: String = "median"](mut self):
        comptime assert split_method in ["median", "sah", "lbvh"]
        debug_assert["safe", _use_compiler_assume=True](
            self.item_count > 0, "passed empty input."
        )

        comptime if split_method == "lbvh":
            _build_lbvh[Self.frame, Self.leaf_size](self)
        else:
            self.nodes_used = 1
            self.nodes[0].set_leaf(0, self.item_count)
            comptime if split_method == "sah":
                var centroid_bounds = (
                    self.update_node_bounds_and_centroid_bounds(0)
                )
                if self.item_count >= PARALLEL_SAH_MIN_ITEMS:
                    self._build_parallel_sah(centroid_bounds)
                else:
                    self._subdivide[split_method](0, centroid_bounds)
            else:
                self.update_node_bounds(0)
                self._subdivide[split_method](0, AABB[Self.frame].invalid())

    def _build_parallel_sah(mut self, root_centroid_bounds: AABB[Self.frame]):
        # Parallel workers write disjoint item ranges and uniquely allocated
        # node pairs. Extending the node list without initialization keeps the
        # old no-worst-case-initialization property while making its storage
        # stable for concurrent indexed writes.
        var max_nodes = Int(self.item_count * 2 - 1)
        self.nodes.resize(unsafe_uninit_length=max_nodes)

        var frontier_nodes = List[UInt32](capacity=8)
        var frontier_centroid_bounds = List[AABB[Self.frame]](capacity=8)
        self._collect_sah_frontier(
            0,
            root_centroid_bounds,
            PARALLEL_SAH_FRONTIER_DEPTH,
            frontier_nodes,
            frontier_centroid_bounds,
        )

        var next_node = [self.nodes_used]

        def worker(task_idx: Int) {imm, mut self, mut next_node}:
            self._subdivide_parallel_sah(
                frontier_nodes[task_idx],
                frontier_centroid_bounds[task_idx],
                next_node.unsafe_ptr(),
            )

        var task_count = len(frontier_nodes)
        if task_count > 0:
            var thread_count = num_performance_cores()
            if thread_count < 1:
                thread_count = 1
            if thread_count > task_count:
                thread_count = task_count
            parallelize(worker, task_count, thread_count)

        self.nodes_used = next_node[0]
        self.nodes.shrink(Int(self.nodes_used))

    def _collect_sah_frontier(
        mut self,
        node_idx: UInt32,
        centroid_bounds: AABB[Self.frame],
        depth: Int,
        mut frontier_nodes: List[UInt32],
        mut frontier_centroid_bounds: List[AABB[Self.frame]],
    ):
        var source_node = self.nodes[Int(node_idx)]
        if source_node.item_count <= UInt32(Self.leaf_size):
            return

        if depth == 0:
            frontier_nodes.append(node_idx)
            frontier_centroid_bounds.append(centroid_bounds)
            return

        var left_child_idx = self.allocate_children()
        var child_centroid_bounds = self._split_node["sah"](
            node_idx, centroid_bounds, left_child_idx
        )
        self._collect_sah_frontier(
            left_child_idx,
            child_centroid_bounds[0],
            depth - 1,
            frontier_nodes,
            frontier_centroid_bounds,
        )
        self._collect_sah_frontier(
            left_child_idx + 1,
            child_centroid_bounds[1],
            depth - 1,
            frontier_nodes,
            frontier_centroid_bounds,
        )

    def _subdivide_parallel_sah[
        next_node_origin: MutOrigin
    ](
        mut self,
        node_idx: UInt32,
        centroid_bounds: AABB[Self.frame],
        next_node: Pointer[UInt32, next_node_origin],
    ):
        var source_node = self.nodes[Int(node_idx)]
        if source_node.item_count <= UInt32(Self.leaf_size):
            return

        var left_child_idx = Atomic.fetch_add[ordering=Ordering.RELAXED](
            next_node, UInt32(2)
        )
        var child_centroid_bounds = self._split_node["sah"](
            node_idx, centroid_bounds, left_child_idx
        )
        self._subdivide_parallel_sah(
            left_child_idx,
            child_centroid_bounds[0],
            next_node,
        )
        self._subdivide_parallel_sah(
            left_child_idx + 1,
            child_centroid_bounds[1],
            next_node,
        )

    @always_inline
    def _split_node[
        split_method: String
    ](
        mut self,
        node_idx: UInt32,
        centroid_bounds: AABB[Self.frame],
        left_child_idx: UInt32,
    ) -> Tuple[AABB[Self.frame], AABB[Self.frame]]:
        comptime assert split_method in ["median", "sah"]
        var source_node = self.nodes[Int(node_idx)]
        var first = Int(source_node.first_item())
        var first_item = source_node.first_item()
        var item_count = source_node.item_count
        var partition = self._partition_node[split_method](
            source_node, centroid_bounds
        )
        var left_count = UInt32(partition.split_idx - first)

        if left_count == 0 or left_count == item_count:
            partition = self._partition_node_by_median(source_node)
            left_count = UInt32(partition.split_idx - first)

        ref left_child = self.nodes[Int(left_child_idx)]
        ref right_child = self.nodes[Int(left_child_idx + 1)]
        left_child.set_leaf(first_item, left_count)
        right_child.set_leaf(
            UInt32(partition.split_idx), item_count - left_count
        )
        left_child.aabb = partition.left_bounds
        right_child.aabb = partition.right_bounds

        ref node = self.nodes[Int(node_idx)]
        node.set_internal(left_child_idx)

        return (
            partition.left_centroid_bounds,
            partition.right_centroid_bounds,
        )

    def _subdivide[
        split_method: String
    ](mut self, node_idx: UInt32, centroid_bounds: AABB[Self.frame],):
        comptime assert split_method in ["median", "sah"]

        var source_node = self.nodes[Int(node_idx)]
        if source_node.item_count <= UInt32(Self.leaf_size):
            return

        var left_child_idx = self.allocate_children()
        var child_centroid_bounds = self._split_node[split_method](
            node_idx, centroid_bounds, left_child_idx
        )

        self._subdivide[split_method](left_child_idx, child_centroid_bounds[0])
        self._subdivide[split_method](
            left_child_idx + 1, child_centroid_bounds[1]
        )

    def _partition_node[
        split_method: String
    ](
        mut self,
        node: BoundsBvhNode[Self.frame],
        centroid_bounds: AABB[Self.frame],
    ) -> BoundsPartitionResult[Self.frame]:
        comptime assert split_method in ["median", "sah"]
        var first = Int(node.first_item())
        var count = Int(node.item_count)

        comptime if split_method == "median":
            return self._partition_node_by_median(node)

        else:
            comptime BVH_BINS = 16
            var split = _find_sah_split[Self.frame, BVH_BINS](
                node,
                centroid_bounds,
                Span(self.item_indices),
                Span(self.items),
            )

            if split.valid():
                return _partition_items_by_bin[Self.frame, BVH_BINS](
                    Span(self.item_indices),
                    Span(self.items),
                    first,
                    count,
                    split.axis,
                    split.bin,
                    split.bin_min,
                    split.bin_scale,
                )

            return self._partition_node_by_median(node)

    @always_inline
    def _partition_node_by_median(
        mut self, node: BoundsBvhNode[Self.frame]
    ) -> BoundsPartitionResult[Self.frame]:
        var first = Int(node.first_item())
        var count = Int(node.item_count)
        var axis = longest_axis(node.aabb.extent())
        var split_idx = _partition_items_by_median_center(
            Span(self.item_indices),
            Span(self.items),
            first,
            count,
            axis,
        )
        return _calculate_partition_bounds(
            Span(self.item_indices),
            Span(self.items),
            first,
            count,
            split_idx,
        )

    def tree_quality(self) -> Float32:
        debug_assert["safe", _use_compiler_assume=True](self.nodes_used > 0)

        ref root = self.nodes[0]
        var root_area = root.surface_area()
        if root_area <= 0.0:
            return 0.0

        var q = Float32(0.0)
        for i in range(Int(self.nodes_used)):
            ref n = self.nodes[i]
            q += n.surface_area() / root_area

        return q


@fieldwise_init
struct BoundsItem[frame: Frame](TrivialRegisterPassable):
    """Generic build item.

    `bounds` is what the BVH builder sees.

    `payload` is owned by the caller:
    - TriangleBvh: triangle id
    - TLAS: instance id
    - Broadphase: index into an ItemRef array
    """

    var bounds: AABB[Self.frame]
    var payload: UInt32

    def center_axis(self, axis: Int) -> Float32:
        return (self.bounds._min[axis] + self.bounds._max[axis]) * 0.5

    def grow_into(self, mut aabb: AABB[Self.frame]):
        aabb._min = vmin(aabb._min, self.bounds._min)
        aabb._max = vmax(aabb._max, self.bounds._max)


@fieldwise_init
struct BoundsBvhNode[frame: Frame](TrivialRegisterPassable):
    """Binary builder node over generic item ranges.

    Leaf:
        first_or_left = first item in item_indices
        item_count    = number of items

    Internal:
        first_or_left = left child node index
        item_count    = 0
        right child   = left child + 1
    """

    var aabb: AABB[Self.frame]
    var first_or_left: UInt32
    var item_count: UInt32

    def __init__(out self):
        self.aabb = AABB[Self.frame].invalid()
        self.first_or_left = 0
        self.item_count = 0

    def is_leaf(self) -> Bool:
        return self.item_count > 0

    def is_internal(self) -> Bool:
        return self.item_count == 0

    def first_item(self) -> UInt32:
        return self.first_or_left

    def left_child(self) -> UInt32:
        return self.first_or_left

    def right_child(self) -> UInt32:
        return self.first_or_left + 1

    def set_leaf(mut self, first_item: UInt32, item_count: UInt32):
        self.first_or_left = first_item
        self.item_count = item_count

    def set_internal(mut self, left_child: UInt32):
        self.first_or_left = left_child
        self.item_count = 0

    def surface_area(self) -> Float32:
        return self.aabb.surface_area()[0]


def _partition_items_by_median_center[
    frame: Frame
](
    indices: Span[mut=True, UInt32, _],
    items: Span[mut=False, BoundsItem[frame], _],
    first: Int,
    count: Int,
    axis: Int,
) -> Int:
    debug_assert["safe", _use_compiler_assume=True](
        first >= 0
        and count > 0
        and first <= len(indices)
        and count <= len(indices) - first,
        "median partition range is outside item indices",
    )
    debug_assert["safe", _use_compiler_assume=True](
        len(indices) == len(items),
        "BVH item indices and items have different lengths",
    )
    debug_assert["safe", _use_compiler_assume=True](
        axis >= 0 and axis < 3,
        "median partition axis is outside [0, 3)",
    )

    var mid = count / 2

    def cmp(a_idx: UInt32, b_idx: UInt32) {items, axis} -> Bool:
        var a = items.unsafe_get(Int(a_idx)).center_axis(axis)
        var b = items.unsafe_get(Int(b_idx)).center_axis(axis)

        if a == b:
            return a_idx < b_idx

        return a < b

    var range = indices[first : first + count]
    nth_element(range, mid, cmp)

    return first + mid
