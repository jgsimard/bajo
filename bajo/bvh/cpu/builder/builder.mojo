from max.algorithm import parallelize
from std.atomic import Atomic, Ordering

from bajo.core import AABB, longest_axis, Frame
from bajo.bvh.cpu.build_method import CpuBvhBuildMethod
from bajo.sort.cpu.nth_element import nth_element
from ..parallel import _worker_count
from .hploc import _build_hploc
from .lbvh import _build_lbvh
from .types import BoundsBvhNode, BoundsItem

from .sah import (
    BoundsPartitionResult,
    _calculate_partition_bounds,
    _find_sah_split,
    _partition_items_by_bin,
)


comptime PARALLEL_SAH_MIN_ITEMS = UInt32(4096)
comptime PARALLEL_MEDIAN_MIN_ITEMS = UInt32(1024)
comptime PARALLEL_FRONTIER_DEPTH = 3
comptime PARALLEL_FRONTIER_CAPACITY = 1 << PARALLEL_FRONTIER_DEPTH


struct BinaryBoundsBvh[
    frame: Frame,
    leaf_size: Int,
    method: CpuBvhBuildMethod = .MEDIAN,
](Copyable):
    """Generic binary BVH builder over AABBs/items.

    Build methods:
        MEDIAN -> top-down spatial median
        SAH    -> top-down binned SAH
        LBVH   -> sorted-Morton recursive LBVH
        HPLOC  -> Morton-guided hierarchical PLOC
    """

    var nodes: List[BoundsBvhNode[Self.frame]]
    var item_indices: List[UInt32]
    var items: List[BoundsItem[Self.frame]]
    var item_count: UInt32
    var nodes_used: UInt32

    def __init__(
        out self,
        var items: List[BoundsItem[Self.frame]],
        root_bounds: AABB[Self.frame] = AABB[Self.frame].invalid(),
        centroid_bounds: AABB[Self.frame] = AABB[Self.frame].invalid(),
    ):
        self.items = items^
        self.item_count = UInt32(len(self.items))
        debug_assert["safe", _use_compiler_assume=True](self.item_count > 0)

        self.item_indices = [i for i in range(self.item_count)]

        var max_nodes = Int(self.item_count * 2 - 1)
        self.nodes = List[BoundsBvhNode[Self.frame]](capacity=max_nodes)
        self.nodes.append(BoundsBvhNode[Self.frame]())

        self.nodes_used = 1
        comptime if Self.method == .LBVH:
            _build_lbvh[
                Self.frame,
                Self.leaf_size,
                Self.method,
            ](self, centroid_bounds)

        elif Self.method == .HPLOC:
            _build_hploc[
                Self.frame,
                Self.leaf_size,
                Self.method,
            ](self, centroid_bounds)

        elif Self.method == .SAH:
            self.nodes[0].set_leaf(0, self.item_count)
            var have_precomputed_bounds = (
                root_bounds._min.x[0] <= root_bounds._max.x[0]
                and centroid_bounds._min.x[0] <= centroid_bounds._max.x[0]
            )
            var build_centroid_bounds: AABB[Self.frame]
            if have_precomputed_bounds:
                self.nodes[0].aabb = root_bounds
                build_centroid_bounds = centroid_bounds
            else:
                build_centroid_bounds = (
                    self.update_node_bounds_and_centroid_bounds(0)
                )
            if self.item_count >= PARALLEL_SAH_MIN_ITEMS:
                self._build_parallel_top_down(build_centroid_bounds)
            else:
                self._subdivide(0, build_centroid_bounds)

        elif Self.method == .MEDIAN:
            self.nodes[0].set_leaf(0, self.item_count)
            if root_bounds._min.x[0] <= root_bounds._max.x[0]:
                self.nodes[0].aabb = root_bounds
            else:
                self.update_node_bounds(0)
            if self.item_count >= PARALLEL_MEDIAN_MIN_ITEMS:
                self._build_parallel_top_down(AABB[Self.frame].invalid())
            else:
                self._subdivide(0, AABB[Self.frame].invalid())

        else:
            comptime assert False, "unknown CPU BVH build method"

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

    def _build_parallel_top_down(
        mut self, root_centroid_bounds: AABB[Self.frame]
    ) where Self.method == .SAH or Self.method == .MEDIAN:
        # Parallel workers write disjoint item ranges and uniquely allocated
        # node pairs. Extending the node list without initialization keeps the
        # old no-worst-case-initialization property while making its storage
        # stable for concurrent indexed writes.
        var max_nodes = Int(self.item_count * 2 - 1)
        self.nodes.resize(unsafe_uninit_length=max_nodes)

        var frontier_nodes = List[UInt32](capacity=PARALLEL_FRONTIER_CAPACITY)
        var frontier_centroid_bounds = List[AABB[Self.frame]](
            capacity=PARALLEL_FRONTIER_CAPACITY
        )
        self._collect_top_down_frontier(
            0,
            root_centroid_bounds,
            PARALLEL_FRONTIER_DEPTH,
            frontier_nodes,
            frontier_centroid_bounds,
        )

        var next_node = [self.nodes_used]

        def worker(task_idx: Int) {imm, mut self, mut next_node}:
            self._subdivide_parallel_top_down(
                frontier_nodes[task_idx],
                frontier_centroid_bounds[task_idx],
                next_node.unsafe_ptr(),
            )

        var task_count = len(frontier_nodes)
        if task_count > 0:
            parallelize(worker, task_count, _worker_count(task_count))

        self.nodes_used = next_node[0]
        self.nodes.shrink(Int(self.nodes_used))

    def _collect_top_down_frontier(
        mut self,
        node_idx: UInt32,
        centroid_bounds: AABB[Self.frame],
        depth: Int,
        mut frontier_nodes: List[UInt32],
        mut frontier_centroid_bounds: List[AABB[Self.frame]],
    ) where Self.method == .SAH or Self.method == .MEDIAN:
        var source_node = self.nodes[Int(node_idx)]
        if source_node.item_count <= UInt32(Self.leaf_size):
            return

        if depth == 0:
            frontier_nodes.append(node_idx)
            frontier_centroid_bounds.append(centroid_bounds)
            return

        var left_child_idx = self.allocate_children()
        var child_centroid_bounds = self._split_node(
            node_idx, centroid_bounds, left_child_idx
        )
        self._collect_top_down_frontier(
            left_child_idx,
            child_centroid_bounds[0],
            depth - 1,
            frontier_nodes,
            frontier_centroid_bounds,
        )
        self._collect_top_down_frontier(
            left_child_idx + 1,
            child_centroid_bounds[1],
            depth - 1,
            frontier_nodes,
            frontier_centroid_bounds,
        )

    def _subdivide_parallel_top_down[
        next_node_origin: MutOrigin
    ](
        mut self,
        node_idx: UInt32,
        centroid_bounds: AABB[Self.frame],
        next_node: Pointer[UInt32, next_node_origin],
    ) where (Self.method == .SAH or Self.method == .MEDIAN):
        var source_node = self.nodes[Int(node_idx)]
        if source_node.item_count <= UInt32(Self.leaf_size):
            return

        var left_child_idx = Atomic.fetch_add[ordering=Ordering.RELAXED](
            next_node, UInt32(2)
        )
        var child_centroid_bounds = self._split_node(
            node_idx, centroid_bounds, left_child_idx
        )
        self._subdivide_parallel_top_down(
            left_child_idx,
            child_centroid_bounds[0],
            next_node,
        )
        self._subdivide_parallel_top_down(
            left_child_idx + 1,
            child_centroid_bounds[1],
            next_node,
        )

    @always_inline
    def _split_node(
        mut self,
        node_idx: UInt32,
        centroid_bounds: AABB[Self.frame],
        left_child_idx: UInt32,
    ) -> Tuple[AABB[Self.frame], AABB[Self.frame]]:
        var source_node = self.nodes[Int(node_idx)]
        var first = Int(source_node.first_item())
        var first_item = source_node.first_item()
        var item_count = source_node.item_count
        var partition = self._partition_node(source_node, centroid_bounds)
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

    def _subdivide(
        mut self, node_idx: UInt32, centroid_bounds: AABB[Self.frame]
    ):
        var source_node = self.nodes[Int(node_idx)]
        if source_node.item_count <= UInt32(Self.leaf_size):
            return

        var left_child_idx = self.allocate_children()
        var child_centroid_bounds = self._split_node(
            node_idx, centroid_bounds, left_child_idx
        )

        self._subdivide(left_child_idx, child_centroid_bounds[0])
        self._subdivide(left_child_idx + 1, child_centroid_bounds[1])

    def _partition_node(
        mut self,
        node: BoundsBvhNode[Self.frame],
        centroid_bounds: AABB[Self.frame],
    ) -> BoundsPartitionResult[Self.frame]:
        comptime if Self.method == .MEDIAN:
            return self._partition_node_by_median(node)

        elif Self.method == .SAH:
            var first = Int(node.first_item())
            var count = Int(node.item_count)
            comptime BVH_BINS = 16
            var split = _find_sah_split[Self.frame, BVH_BINS](
                node,
                centroid_bounds,
                self.item_indices,
                self.items,
            )

            if split.valid():
                return _partition_items_by_bin[
                    Self.frame,
                    BVH_BINS,
                ](
                    self.item_indices,
                    self.items,
                    first,
                    count,
                    split.axis,
                    split.bin,
                    split.bin_min,
                    split.bin_scale,
                )

            return self._partition_node_by_median(node)
        else:
            comptime assert False

    @always_inline
    def _partition_node_by_median(
        mut self, node: BoundsBvhNode[Self.frame]
    ) -> BoundsPartitionResult[Self.frame]:
        var first = Int(node.first_item())
        var count = Int(node.item_count)
        var axis = longest_axis(node.aabb.extent())
        var split_idx = _partition_items_by_median_center(
            self.item_indices,
            self.items,
            first,
            count,
            axis,
        )
        return _calculate_partition_bounds(
            self.item_indices,
            self.items,
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


def _partition_items_by_median_center[
    frame: Frame
](
    indices: MutSpan[UInt32, _],
    items: ImmSpan[BoundsItem[frame], _],
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
        ref a_item = items.unsafe_get(Int(a_idx))
        ref b_item = items.unsafe_get(Int(b_idx))
        var a = a_item.bounds._min[axis] + a_item.bounds._max[axis]
        var b = b_item.bounds._min[axis] + b_item.bounds._max[axis]

        if a == b:
            return a_idx < b_idx

        return a < b

    var range = indices[first : first + count]
    nth_element(range, mid, cmp)

    return first + mid
