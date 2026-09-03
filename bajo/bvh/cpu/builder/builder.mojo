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
    BoundsSplitResult,
    BoundsPartitionResult,
    _calculate_partition_bounds,
    _find_sah_split,
    _find_sah_split_parallel,
    _partition_items_by_bin,
    _partition_items_by_bin_parallel,
)


comptime PARALLEL_SAH_MIN_ITEMS = UInt32(4096)
comptime PARALLEL_MEDIAN_MIN_ITEMS = UInt32(1024)
comptime PARALLEL_FRONTIER_DEPTH = 3
comptime PARALLEL_FRONTIER_CAPACITY = 1 << PARALLEL_FRONTIER_DEPTH
comptime PARALLEL_SAH_PARTITION_MIN_ITEMS = UInt32(32768)
comptime PARALLEL_TOP_DOWN_DONATE_MIN_ITEMS = UInt32(4096)


@always_inline
def _acquire_top_down_queue_lock[
    origin: MutOrigin
](lock: Pointer[UInt32, origin],):
    var expected = UInt32(0)
    while not Atomic.compare_exchange[
        success_ordering=Ordering.ACQUIRE,
        failure_ordering=Ordering.RELAXED,
    ](lock, expected, UInt32(1)):
        expected = UInt32(0)


@always_inline
def _release_top_down_queue_lock[
    origin: MutOrigin
](lock: Pointer[UInt32, origin],):
    Atomic.store[ordering=Ordering.RELEASE](lock, UInt32(0))


struct BinaryBoundsBvh[
    frame: Frame,
    leaf_size: Int,
    method: CpuBvhBuildMethod = .MEDIAN,
    hploc_microleaf_size: Int = 1,
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

    def __init__[
        hploc_balance_tasks: Int = 32
    ](
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
                Self.hploc_microleaf_size,
            ](self, centroid_bounds)
        elif Self.method == .HPLOC:
            comptime assert (
                Self.hploc_microleaf_size > 0
                and Self.hploc_microleaf_size <= Self.leaf_size
            ), "H-PLOC microleaf size must fit the final leaf"
            _build_hploc[
                Self.frame,
                Self.leaf_size,
                Self.method,
                Self.hploc_microleaf_size,
                hploc_balance_tasks,
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
            centroid_bounds.grow(item.centroid)

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
        self._collect_top_down_frontier_parallel(
            0,
            root_centroid_bounds,
            PARALLEL_FRONTIER_DEPTH,
            frontier_nodes,
            frontier_centroid_bounds,
        )

        var task_count = len(frontier_nodes)
        var next_node = [self.nodes_used]
        var worker_count = _worker_count(Int(self.item_count))
        if task_count > 0 and worker_count > 1:
            # Keep a small shared stack of coarse subtrees. Workers recurse
            # locally, donating only a large sibling so the tail can be stolen
            # without paying synchronization at every BVH node.
            var queue_capacity = max(
                64,
                Int(self.item_count / PARALLEL_TOP_DOWN_DONATE_MIN_ITEMS) * 4
                + PARALLEL_FRONTIER_CAPACITY,
            )
            var work_nodes = List[UInt32](capacity=queue_capacity)
            var work_centroid_bounds = List[AABB[Self.frame]](
                capacity=queue_capacity
            )
            work_nodes.resize(unsafe_uninit_length=queue_capacity)
            work_centroid_bounds.resize(unsafe_uninit_length=queue_capacity)
            for task_idx in range(task_count):
                work_nodes[task_idx] = frontier_nodes[task_idx]
                work_centroid_bounds[task_idx] = frontier_centroid_bounds[
                    task_idx
                ]

            var queue_size = [UInt32(task_count)]
            var outstanding = [UInt32(task_count)]
            var queue_lock = [UInt32(0)]

            def queue_worker(
                _task_idx: Int,
            ) {
                imm,
                mut self,
                mut next_node,
                mut work_nodes,
                mut work_centroid_bounds,
                mut queue_size,
                mut outstanding,
                mut queue_lock,
            }:
                while True:
                    _acquire_top_down_queue_lock(queue_lock.unsafe_ptr())
                    var size = queue_size[0]
                    if size > 0:
                        size -= 1
                        queue_size[0] = size
                        var node_idx = work_nodes[Int(size)]
                        var bounds = work_centroid_bounds[Int(size)]
                        _release_top_down_queue_lock(queue_lock.unsafe_ptr())

                        self._subdivide_queued_top_down(
                            node_idx,
                            bounds,
                            next_node.unsafe_ptr(),
                            work_nodes,
                            work_centroid_bounds,
                            queue_size.unsafe_ptr(),
                            outstanding.unsafe_ptr(),
                            queue_lock.unsafe_ptr(),
                        )

                        _acquire_top_down_queue_lock(queue_lock.unsafe_ptr())
                        outstanding[0] -= 1
                        var finished = outstanding[0] == 0
                        _release_top_down_queue_lock(queue_lock.unsafe_ptr())
                        if finished:
                            return
                    else:
                        var finished = outstanding[0] == 0
                        _release_top_down_queue_lock(queue_lock.unsafe_ptr())
                        if finished:
                            return

            parallelize(queue_worker, worker_count, worker_count)
        elif task_count > 0:
            for task_idx in range(task_count):
                self._subdivide(
                    frontier_nodes[task_idx],
                    frontier_centroid_bounds[task_idx],
                )
            next_node[0] = self.nodes_used

        self.nodes_used = next_node[0]
        self.nodes.shrink(Int(self.nodes_used))

    def _subdivide_queued_top_down[
        next_node_origin: MutOrigin,
        queue_size_origin: MutOrigin,
        outstanding_origin: MutOrigin,
        queue_lock_origin: MutOrigin,
    ](
        mut self,
        node_idx: UInt32,
        centroid_bounds: AABB[Self.frame],
        next_node: Pointer[UInt32, next_node_origin],
        work_nodes: MutSpan[UInt32, _],
        work_centroid_bounds: MutSpan[AABB[Self.frame], _],
        queue_size: Pointer[UInt32, queue_size_origin],
        outstanding: Pointer[UInt32, outstanding_origin],
        queue_lock: Pointer[UInt32, queue_lock_origin],
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
        var left_count = self.nodes[Int(left_child_idx)].item_count
        var right_count = self.nodes[Int(left_child_idx + 1)].item_count
        var left_internal = left_count > UInt32(Self.leaf_size)
        var right_internal = right_count > UInt32(Self.leaf_size)

        if (
            left_internal
            and right_internal
            and max(left_count, right_count)
            >= PARALLEL_TOP_DOWN_DONATE_MIN_ITEMS
        ):
            var donated_node: UInt32
            var donated_bounds: AABB[Self.frame]
            var local_node: UInt32
            var local_bounds: AABB[Self.frame]
            if left_count >= right_count:
                donated_node = left_child_idx
                donated_bounds = child_centroid_bounds[0]
                local_node = left_child_idx + 1
                local_bounds = child_centroid_bounds[1]
            else:
                donated_node = left_child_idx + 1
                donated_bounds = child_centroid_bounds[1]
                local_node = left_child_idx
                local_bounds = child_centroid_bounds[0]

            _acquire_top_down_queue_lock(queue_lock)
            var slot = queue_size[unsafe_offset=0]
            debug_assert["safe", _use_compiler_assume=True](
                Int(slot) < len(work_nodes), "top-down work queue overflow"
            )
            work_nodes[Int(slot)] = donated_node
            work_centroid_bounds[Int(slot)] = donated_bounds
            queue_size[unsafe_offset=0] = slot + 1
            outstanding[unsafe_offset=0] += 1
            _release_top_down_queue_lock(queue_lock)

            self._subdivide_queued_top_down(
                local_node,
                local_bounds,
                next_node,
                work_nodes,
                work_centroid_bounds,
                queue_size,
                outstanding,
                queue_lock,
            )
            return

        if left_internal:
            self._subdivide_queued_top_down(
                left_child_idx,
                child_centroid_bounds[0],
                next_node,
                work_nodes,
                work_centroid_bounds,
                queue_size,
                outstanding,
                queue_lock,
            )
        if right_internal:
            self._subdivide_queued_top_down(
                left_child_idx + 1,
                child_centroid_bounds[1],
                next_node,
                work_nodes,
                work_centroid_bounds,
                queue_size,
                outstanding,
                queue_lock,
            )

    def _collect_top_down_frontier_parallel(
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

        # Split the root first. SAH uses its parallel histogram/partition path
        # here, while subsequent levels consist of independent item ranges.
        var left_child_idx = self.allocate_children()
        var child_centroid_bounds = self._split_node(
            node_idx, centroid_bounds, left_child_idx, True
        )
        if self.nodes[Int(left_child_idx)].item_count > UInt32(Self.leaf_size):
            frontier_nodes.append(left_child_idx)
            frontier_centroid_bounds.append(child_centroid_bounds[0])
        if self.nodes[Int(left_child_idx + 1)].item_count > UInt32(
            Self.leaf_size
        ):
            frontier_nodes.append(left_child_idx + 1)
            frontier_centroid_bounds.append(child_centroid_bounds[1])

        # Build the remaining upper levels breadth-first. Node and item ranges
        # are disjoint, so each level can split concurrently. Child indices are
        # reserved before launching workers to keep allocation race-free.
        for _ in range(1, depth):
            var task_count = len(frontier_nodes)
            if task_count == 0:
                return

            var child_indices = List[UInt32](capacity=task_count)
            var left_centroid_bounds = List[AABB[Self.frame]](
                capacity=task_count
            )
            var right_centroid_bounds = List[AABB[Self.frame]](
                capacity=task_count
            )
            for _ in range(task_count):
                child_indices.append(self.allocate_children())
                left_centroid_bounds.append(AABB[Self.frame].invalid())
                right_centroid_bounds.append(AABB[Self.frame].invalid())

            def split_worker(
                task_idx: Int,
            ) {
                imm,
                mut self,
                mut left_centroid_bounds,
                mut right_centroid_bounds,
            }:
                var result = self._split_node(
                    frontier_nodes[task_idx],
                    frontier_centroid_bounds[task_idx],
                    child_indices[task_idx],
                )
                left_centroid_bounds[task_idx] = result[0]
                right_centroid_bounds[task_idx] = result[1]

            var use_parallel_ranges = task_count <= 4
            for task_idx in range(task_count):
                use_parallel_ranges &= (
                    self.nodes[Int(frontier_nodes[task_idx])].item_count
                    >= PARALLEL_SAH_PARTITION_MIN_ITEMS
                )
            if use_parallel_ranges:
                # A shallow SAH level has too few nodes to occupy the machine.
                # Split those large ranges one at a time using their internal
                # parallel histogram and partition instead.
                for task_idx in range(task_count):
                    var result = self._split_node(
                        frontier_nodes[task_idx],
                        frontier_centroid_bounds[task_idx],
                        child_indices[task_idx],
                        True,
                    )
                    left_centroid_bounds[task_idx] = result[0]
                    right_centroid_bounds[task_idx] = result[1]
            else:
                parallelize(split_worker, task_count, _worker_count(task_count))

            var next_frontier_nodes = List[UInt32](capacity=task_count * 2)
            var next_frontier_centroid_bounds = List[AABB[Self.frame]](
                capacity=task_count * 2
            )
            for task_idx in range(task_count):
                var child_idx = child_indices[task_idx]
                if self.nodes[Int(child_idx)].item_count > UInt32(
                    Self.leaf_size
                ):
                    next_frontier_nodes.append(child_idx)
                    next_frontier_centroid_bounds.append(
                        left_centroid_bounds[task_idx]
                    )
                if self.nodes[Int(child_idx + 1)].item_count > UInt32(
                    Self.leaf_size
                ):
                    next_frontier_nodes.append(child_idx + 1)
                    next_frontier_centroid_bounds.append(
                        right_centroid_bounds[task_idx]
                    )

            frontier_nodes = next_frontier_nodes^
            frontier_centroid_bounds = next_frontier_centroid_bounds^

    @always_inline
    def _split_node(
        mut self,
        node_idx: UInt32,
        centroid_bounds: AABB[Self.frame],
        left_child_idx: UInt32,
        allow_parallel_partition: Bool = False,
    ) -> Tuple[AABB[Self.frame], AABB[Self.frame]]:
        var source_node = self.nodes[Int(node_idx)]
        var first = Int(source_node.first_item())
        var first_item = source_node.first_item()
        var item_count = source_node.item_count
        var partition = self._partition_node(
            source_node,
            centroid_bounds,
            allow_parallel_partition
            and source_node.item_count >= PARALLEL_SAH_PARTITION_MIN_ITEMS
            and _worker_count(Int(source_node.item_count)) > 1,
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
        parallel_root: Bool = False,
    ) -> BoundsPartitionResult[Self.frame]:
        comptime if Self.method == .MEDIAN:
            return self._partition_node_by_median(node)

        elif Self.method == .SAH:
            var count = Int(node.item_count)
            if count < 128:
                return self._partition_node_sah[8](
                    node, centroid_bounds, parallel_root
                )
            if count < 512:
                return self._partition_node_sah[16](
                    node, centroid_bounds, parallel_root
                )
            return self._partition_node_sah[32](
                node, centroid_bounds, parallel_root
            )
        else:
            comptime assert False

    def _partition_node_sah[
        BVH_BINS: Int
    ](
        mut self,
        node: BoundsBvhNode[Self.frame],
        centroid_bounds: AABB[Self.frame],
        parallel_root: Bool,
    ) -> BoundsPartitionResult[Self.frame] where (Self.method == .SAH):
        var first = Int(node.first_item())
        var count = Int(node.item_count)
        var split: BoundsSplitResult[Self.frame]
        if parallel_root:
            split = _find_sah_split_parallel[Self.frame, BVH_BINS](
                node,
                centroid_bounds,
                self.item_indices,
                self.items,
                _worker_count(count),
            )
        else:
            split = _find_sah_split[Self.frame, BVH_BINS](
                node,
                centroid_bounds,
                self.item_indices,
                self.items,
            )

        if split.valid():
            if parallel_root:
                var worker_count = _worker_count(count)
                return _partition_items_by_bin_parallel[
                    Self.frame,
                    BVH_BINS,
                    True,
                ](
                    self.item_indices,
                    self.items,
                    first,
                    count,
                    split.axis,
                    split.bin,
                    split.bin_min,
                    split.bin_scale,
                    worker_count,
                    split.left_bounds,
                    split.right_bounds,
                )
            return _partition_items_by_bin[
                Self.frame,
                BVH_BINS,
                True,
            ](
                self.item_indices,
                self.items,
                first,
                count,
                split.axis,
                split.bin,
                split.bin_min,
                split.bin_scale,
                split.left_bounds,
                split.right_bounds,
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
        var a = a_item.centroid[axis]
        var b = b_item.centroid[axis]

        if a == b:
            return a_idx < b_idx

        return a < b

    var range = indices[first : first + count]
    nth_element(range, mid, cmp)

    return first + mid
