"""Owning spatial BVH with original item IDs and independent group roots."""

from max.algorithm import parallelize
from bajo.bvh.cpu.parallel import _worker_count
from bajo.core import AABB, Frame, Point3f32, longest_axis
from std.math import min, max
from std.memory import Pointer, bitcast
from std.sys.intrinsics import masked_load
from std.sys.info import size_of
from bajo.bvh.constants import EMPTY_LANE
from bajo.bvh.cpu.builder import BoundsItem
from bajo.bvh.cpu.builder.builder import _partition_items_by_median_center
from .query import BoundsQueryView
from .radix import sort_morton30
from .policy import REFIT_WIDTH, BUILD_WIDTH, MORTON_BUILD_MIN_ITEMS
from bajo.core.morton import morton3
from bajo.bvh.cpu.builder.lbvh import (
    MortonItem,
    LbvhFrontierTask,
)


# A bounded label scan avoids hashing for the common few-group case. Larger
# forests discover labels with a directory and partition all groups together.
comptime _SMALL_GROUP_COUNT = 8


@always_inline
def _finite_key[
    width: SIMDLength
](bits: SIMD[.uint32, width]) -> SIMD[.uint32, width]:
    # IEEE finite values have integer total order after sign reversal.
    return (bits >> 31).ne(0).select(~bits, bits ^ UInt32(0x80000000))


@always_inline
def _finite_min[
    width: SIMDLength
](a: SIMD[.float32, width], b: SIMD[.float32, width]) -> SIMD[.float32, width]:
    var ua = bitcast[.uint32](a)
    var ub = bitcast[.uint32](b)
    return bitcast[.float32](_finite_key(ua).lt(_finite_key(ub)).select(ua, ub))


@always_inline
def _finite_max[
    width: SIMDLength
](a: SIMD[.float32, width], b: SIMD[.float32, width]) -> SIMD[.float32, width]:
    var ua = bitcast[.uint32](a)
    var ub = bitcast[.uint32](b)
    return bitcast[.float32](_finite_key(ua).gt(_finite_key(ub)).select(ua, ub))


@always_inline
def _finite_box_properties[frame: Frame](box: AABB[frame]) -> Tuple[Bool, Bool]:
    var values = SIMD[.float32, 8](
        box._min.x,
        box._min.y,
        box._min.z,
        box._max.x,
        box._max.y,
        box._max.z,
        0.0,
        0.0,
    )
    var bits = bitcast[.uint32](values)
    var tiny = (
        (bits & UInt32(0x7F800000)).eq(0) & (bits & UInt32(0x007FFFFF)).ne(0)
    ).reduce_or()
    if not tiny:
        return (box.is_valid()[0], False)
    var valid = True
    var subnormal = False
    comptime for axis in range(3):
        var a = bitcast[.uint32](box._min[axis])
        var b = bitcast[.uint32](box._max[axis])
        valid &= (a & UInt32(0x7F800000)) != UInt32(0x7F800000)
        valid &= (b & UInt32(0x7F800000)) != UInt32(0x7F800000)
        subnormal |= (a & UInt32(0x7F800000)) == 0 and (
            a & UInt32(0x007FFFFF)
        ) != 0
        subnormal |= (b & UInt32(0x7F800000)) == 0 and (
            b & UInt32(0x007FFFFF)
        ) != 0
        # Both representations of zero compare equal for validation.
        var ka = _finite_key(a)
        var kb = _finite_key(b)
        if (a & UInt32(0x7FFFFFFF)) == 0:
            ka = UInt32(0x80000000)
        if (b & UInt32(0x7FFFFFFF)) == 0:
            kb = UInt32(0x80000000)
        valid &= ka <= kb
    return (valid, subnormal)


@always_inline
def _has_subnormal[frame: Frame](boxes: ImmSpan[AABB[frame], _]) -> Bool:
    comptime assert size_of[AABB[frame]]() == 6 * size_of[Float32]()
    var bits = boxes.unsafe_ptr().unsafe_bitcast[UInt32]()
    var count = len(boxes) * 6
    var end = count // 8 * 8
    for first in range(0, end, 8):
        var packet = bits.unsafe_load[width=8](first)
        if (
            (packet & UInt32(0x7F800000)).eq(0)
            & (packet & UInt32(0x007FFFFF)).ne(0)
        ).reduce_or():
            return True
    for first in range(end, count):
        var word = bits.unsafe_load(first)
        if (word & UInt32(0x7F800000)) == 0 and (
            word & UInt32(0x007FFFFF)
        ) != 0:
            return True
    return False


struct SpatialBoundsBvh[frame: Frame = .WORLD](Copyable):
    """CPU owner of the shared spatial-query representation.

    Construction uses median partitions for small trees and stable Morton order
    with balanced splits for large trees. Both emit stackless preorder nodes;
    global and group trees reference original item IDs.
    Empty inputs are supported. Groups are optional nonnegative UInt32 labels.
    """

    var item_bounds: List[AABB[Self.frame]]
    var groups: List[UInt32]
    var group_ids: List[UInt32]
    var group_roots: List[UInt32]
    var bounds: List[Float32]
    var nodes: List[UInt32]

    def __init__(out self, boxes: ImmSpan[AABB[Self.frame], _]) raises:
        var groups = List[UInt32]()
        self = Self(boxes, groups)

    def __init__(
        out self,
        boxes: ImmSpan[AABB[Self.frame], _],
        groups: ImmSpan[UInt32, _],
    ) raises:
        if len(groups) != 0 and len(groups) != len(boxes):
            raise Error("BVH group count must match item count")
        self.item_bounds = List[AABB[Self.frame]](capacity=len(boxes))
        self.groups = List[UInt32](capacity=len(groups))
        self.group_ids = List[UInt32]()
        self.group_roots = List[UInt32]()
        self.bounds = List[Float32]()
        self.nodes = List[UInt32]()
        var precise = _has_subnormal(boxes)
        if precise:
            self._append_validated_boxes[True](boxes)
        else:
            self._append_validated_boxes[False](boxes)
        var group_index = self._collect_groups(groups)
        var item_count = len(self.item_bounds)
        var node_count = 2 * item_count - 1 if item_count > 0 else 0
        if len(groups) > 0:
            node_count += 2 * item_count - len(self.group_ids)
        self.bounds.reserve(6 * node_count)
        self.nodes.reserve(2 * node_count)
        self.group_roots.reserve(len(self.group_ids))
        var items = List[BoundsItem[Self.frame]](capacity=item_count)
        for i in range(item_count):
            items.append(BoundsItem(self.item_bounds[i], UInt32(i)))
        self.bounds.resize(unsafe_uninit_length=6 * node_count)
        self.nodes.resize(unsafe_uninit_length=2 * node_count)
        if precise:
            self._build_all_trees[True](items, group_index)
        else:
            self._build_all_trees[False](items, group_index)

    def _collect_groups(
        mut self, groups: ImmSpan[UInt32, _]
    ) -> Dict[UInt32, Int]:
        var index = Dict[UInt32, Int]()
        for group in groups:
            self.groups.append(group)
            var count = len(self.group_ids)
            if count > 0 and self.group_ids[count - 1] == group:
                continue
            if count <= _SMALL_GROUP_COUNT:
                var found = False
                for previous in self.group_ids:
                    if previous == group:
                        found = True
                        break
                if found:
                    continue
                if count == _SMALL_GROUP_COUNT:
                    for slot, label in enumerate(self.group_ids):
                        index[label] = slot
                    index[group] = count
            else:
                if index.get(group, -1) >= 0:
                    continue
                index[group] = count
            self.group_ids.append(group)
        return index^

    def _append_validated_boxes[
        precise: Bool
    ](mut self, boxes: ImmSpan[AABB[Self.frame], _]) raises:
        for box in boxes:
            comptime if precise:
                if not _finite_box_properties(box)[0]:
                    raise Error("BVH item bounds must be finite and ordered")
            else:
                if not box.is_valid()[0]:
                    raise Error("BVH item bounds must be finite and ordered")
            self.item_bounds.append(box)

    def _check_boxes[
        precise: Bool
    ](self, boxes: ImmSpan[AABB[Self.frame], _]) raises:
        for box in boxes:
            comptime if precise:
                if not _finite_box_properties(box)[0]:
                    raise Error("BVH item bounds must be finite and ordered")
            else:
                if not box.is_valid()[0]:
                    raise Error("BVH item bounds must be finite and ordered")

    def _build_all_trees[
        precise: Bool
    ](
        mut self,
        items: ImmSpan[BoundsItem[Self.frame], _],
        group_index: Dict[UInt32, Int],
    ):
        var shared_pairs = List[MortonItem]()
        var output = self._build_tree[precise](
            items, EMPTY_LANE, False, 0, shared_pairs
        )
        if len(self.group_ids) > _SMALL_GROUP_COUNT:
            self._build_partitioned_groups[precise](
                items, shared_pairs, group_index, output
            )
            return
        for group in self.group_ids:
            self.group_roots.append(UInt32(output))
            if len(shared_pairs) > 0:
                output = self._build_shared_group[precise](
                    items, group, output, shared_pairs
                )
            else:
                output = self._build_tree[precise](
                    items, group, True, output, shared_pairs
                )

    def _build_partitioned_groups[
        precise: Bool
    ](
        mut self,
        items: ImmSpan[BoundsItem[Self.frame], _],
        shared_pairs: ImmSpan[MortonItem, _],
        group_index: Dict[UInt32, Int],
        output: Int,
    ):
        # Prefix offsets give each group a disjoint slice of one flat buffer.
        # Cache each item's dense group slot so scattering the global Morton
        # order needs no further label lookups or per-group input scans.
        var group_count = len(self.group_ids)
        var offsets = List[Int](length=group_count + 1, fill=0)
        var slots = List[UInt32](capacity=len(items))
        slots.resize(unsafe_uninit_length=len(items))
        for item in range(len(items)):
            var slot = group_index.get(self.groups[item], -1)
            debug_assert["safe", _use_compiler_assume=True](
                slot >= 0 and slot < group_count,
                "spatial group label is missing from the build directory",
            )
            slots.unsafe_get(item) = UInt32(slot)
            offsets[slot + 1] += 1
        for group in range(group_count):
            offsets[group + 1] += offsets[group]
        var cursors = offsets.copy()
        var next_node = output
        if len(shared_pairs) > 0:
            var pairs = List[MortonItem](capacity=len(items))
            pairs.resize(unsafe_uninit_length=len(items))
            for pair in shared_pairs:
                var slot = Int(slots.unsafe_get(Int(pair.item_idx)))
                pairs.unsafe_get(cursors[slot]) = pair
                cursors[slot] += 1
            for group in range(group_count):
                self.group_roots.append(UInt32(next_node))
                next_node = self._emit_morton[precise](
                    items,
                    pairs,
                    offsets[group],
                    offsets[group + 1] - offsets[group],
                    next_node,
                )
        else:
            var order = List[UInt32](capacity=len(items))
            order.resize(unsafe_uninit_length=len(items))
            for item in range(len(items)):
                var slot = Int(slots.unsafe_get(item))
                order.unsafe_get(cursors[slot]) = UInt32(item)
                cursors[slot] += 1
            for group in range(group_count):
                self.group_roots.append(UInt32(next_node))
                next_node = self._emit_median[precise](
                    items,
                    order,
                    offsets[group],
                    offsets[group + 1] - offsets[group],
                    next_node,
                )

    def _build_shared_group[
        precise: Bool
    ](
        mut self,
        items: ImmSpan[BoundsItem[Self.frame], _],
        group: UInt32,
        output: Int,
        shared_pairs: ImmSpan[MortonItem, _],
    ) -> Int:
        var count = 0
        # Validated snapshot labels and canonical item IDs bound every load.
        # The first scan counts precisely the records selected by the second.
        var labels = self.groups.unsafe_ptr()
        for i in range(len(self.groups)):
            count += Int(labels.unsafe_load(i) == group)
        var pairs = List[MortonItem](capacity=count)
        pairs.resize(unsafe_uninit_length=count)
        var cursor = 0
        for i in range(len(shared_pairs)):
            var pair = shared_pairs.unsafe_get(i)
            if labels.unsafe_load(Int(pair.item_idx)) == group:
                pairs.unsafe_get(cursor) = pair
                cursor += 1
        return self._emit_morton[precise](items, pairs, 0, count, output)

    def _build_tree[
        precise: Bool = False
    ](
        mut self,
        items: ImmSpan[BoundsItem[Self.frame], _],
        group: UInt32,
        grouped: Bool,
        output: Int,
        mut shared_pairs: List[MortonItem],
    ) -> Int:
        var order = List[UInt32](capacity=len(items))
        for i in range(len(items)):
            if not grouped or self.groups[i] == group:
                order.append(UInt32(i))
        if len(order) >= MORTON_BUILD_MIN_ITEMS:
            if grouped and len(shared_pairs) > 0:
                var pairs = List[MortonItem](capacity=len(order))
                for pair in shared_pairs:
                    if self.groups[Int(pair.item_idx)] == group:
                        pairs.append(pair)
                return self._emit_morton[precise](
                    items, pairs, 0, len(pairs), output
                )
            var box = self._range_bounds[BUILD_WIDTH, precise](
                items, order, 0, len(order)
            )
            var pairs = List[MortonItem](capacity=len(order))
            var shared_extent = Float64(0.0)
            comptime for axis in range(3):
                shared_extent = max(
                    shared_extent,
                    Float64(box._max[axis]) - Float64(box._min[axis]),
                )
            # Keep scalar Float64 normalization and stable codes, batching
            # independent items into SIMD lanes. The tail never reads padding.
            comptime width = 8
            pairs.resize(unsafe_uninit_length=len(order))
            var batch_end = len(order) // width * width
            for first in range(0, batch_end, width):
                var normalized = Array[SIMD[.float32, width], 3](
                    fill=SIMD[.float32, width](0.0)
                )
                if shared_extent > 0.0:
                    comptime for axis in range(3):
                        var lower = SIMD[.float64, width](0.0)
                        var upper = SIMD[.float64, width](0.0)
                        comptime for lane in range(width):
                            ref bounds = items.unsafe_get(
                                Int(order.unsafe_get(first + lane))
                            ).bounds
                            lower[lane] = Float64(bounds._min[axis])
                            upper[lane] = Float64(bounds._max[axis])
                        var center = (lower + upper) * 0.5
                        normalized[axis] = max(
                            SIMD[.float64, width](0.0),
                            min(
                                SIMD[.float64, width](1.0),
                                (center - Float64(box._min[axis]))
                                / shared_extent,
                            ),
                        ).cast[.float32]()
                var codes = morton3(normalized[0], normalized[1], normalized[2])
                comptime for lane in range(width):
                    pairs.unsafe_get(first + lane) = MortonItem(
                        codes[lane], order.unsafe_get(first + lane)
                    )
            for tail in range(batch_end, len(order)):
                var id = order.unsafe_get(tail)
                var normalized = Point3f32[Self.frame](0.0)
                ref bounds = items.unsafe_get(Int(id)).bounds
                comptime for axis in range(3):
                    if shared_extent > 0.0:
                        var center = (
                            Float64(bounds._min[axis])
                            + Float64(bounds._max[axis])
                        ) * 0.5
                        normalized[axis] = Float32(
                            max(
                                Float64(0.0),
                                min(
                                    Float64(1.0),
                                    (center - Float64(box._min[axis]))
                                    / shared_extent,
                                ),
                            )
                        )
                pairs.unsafe_get(tail) = MortonItem(
                    morton3(normalized.x, normalized.y, normalized.z), id
                )
            sort_morton30(pairs)
            var end = self._emit_morton[precise](
                items, pairs, 0, len(pairs), output
            )
            if not grouped and len(self.group_ids) > 0:
                shared_pairs = pairs^
            return end
        if len(order) > 0:
            return self._emit_median[precise](
                items, order, 0, len(order), output
            )
        return output

    def _collect_morton_frontier(
        mut self,
        first: Int,
        count: Int,
        node: Int,
        depth: Int,
        mut tasks: List[LbvhFrontierTask],
        mut parents: List[Int],
    ):
        if depth == 0 or count == 1:
            tasks.append(LbvhFrontierTask(UInt32(node), first, count))
            return
        self.nodes[node * 2] = UInt32(node + 2 * count - 1)
        self.nodes[node * 2 + 1] = EMPTY_LANE
        parents.append(node)
        var split = count // 2
        self._collect_morton_frontier(
            first, split, node + 1, depth - 1, tasks, parents
        )
        self._collect_morton_frontier(
            first + split,
            count - split,
            node + 2 * split,
            depth - 1,
            tasks,
            parents,
        )

    def _emit_morton[
        precise: Bool = False
    ](
        mut self,
        items: ImmSpan[BoundsItem[Self.frame], _],
        pairs: ImmSpan[MortonItem, _],
        first: Int,
        count: Int,
        node: Int,
    ) -> Int:
        if count < 16384:
            return self._emit_morton_serial[precise](
                items, pairs, first, count, node
            )
        var tasks = List[LbvhFrontierTask](capacity=8)
        var parents = List[Int](capacity=7)
        self._collect_morton_frontier(first, count, node, 3, tasks, parents)

        def worker(task_idx: Int) {imm, mut self}:
            var task = tasks[task_idx]
            _ = self._emit_morton_serial[precise](
                items, pairs, task.first, task.count, Int(task.node_idx)
            )

        parallelize(worker, len(tasks), _worker_count(len(tasks)))
        for index in range(len(parents) - 1, -1, -1):
            var parent = parents[index]
            var left = parent + 1
            var right = Int(self.nodes[left * 2])
            self._store_bounds(
                parent,
                self._merge_node_bounds[precise](left, right),
            )
        return node + 2 * count - 1

    def _emit_morton_serial[
        precise: Bool = False
    ](
        mut self,
        items: ImmSpan[BoundsItem[Self.frame], _],
        pairs: ImmSpan[MortonItem, _],
        first: Int,
        count: Int,
        node: Int,
    ) -> Int:
        # Balanced preorder: subtree length is exactly 2 * primitive_count - 1.
        # Emit topology/leaves iteratively, then union children in reverse order.
        var pending_first = Array[Int, 64](fill=0)
        var pending_count = Array[Int, 64](fill=0)
        var pending_node = Array[Int, 64](fill=0)
        var depth = 0
        var cursor = node
        var begin = first
        var size = count
        var nodes = self.nodes.unsafe_ptr()
        var end = node + 2 * count - 1
        while True:
            nodes.unsafe_store(cursor * 2, UInt32(cursor + 2 * size - 1))
            nodes.unsafe_store(cursor * 2 + 1, EMPTY_LANE)
            if size == 1:
                ref item = items.unsafe_get(
                    Int(pairs.unsafe_get(begin).item_idx)
                )
                self._store_bounds(cursor, item.bounds)
                nodes.unsafe_store(cursor * 2 + 1, item.payload)
                if depth == 0:
                    break
                depth -= 1
                begin = pending_first[depth]
                size = pending_count[depth]
                cursor = pending_node[depth]
            else:
                var split = size // 2
                pending_first[depth] = begin + split
                pending_count[depth] = size - split
                pending_node[depth] = cursor + 2 * split
                depth += 1
                cursor += 1
                size = split
        for current in range(end - 1, node - 1, -1):
            if nodes.unsafe_load(current * 2 + 1) == EMPTY_LANE:
                var left = current + 1
                var right = Int(nodes.unsafe_load(left * 2))
                self._store_bounds(
                    current,
                    self._merge_node_bounds[precise](left, right),
                )
        return end

    @always_inline
    def _range_bounds[
        width: SIMDLength = BUILD_WIDTH, precise: Bool = True
    ](
        self,
        items: ImmSpan[BoundsItem[Self.frame], _],
        order: ImmSpan[UInt32, _],
        first: Int,
        count: Int,
    ) -> AABB[Self.frame]:
        comptime assert size_of[AABB[Self.frame]]() == 6 * size_of[Float32]()
        var box = items.unsafe_get(Int(order.unsafe_get(first))).bounds
        if count == 1:
            return box
        comptime for chunk in range((6 + width - 1) // width):
            var valid = SIMD[.bool, width](fill=False)
            var lower = SIMD[.bool, width](fill=False)
            comptime for lane in range(width):
                valid[lane] = chunk * width + lane < 6
                lower[lane] = chunk * width + lane < 3
            var start = (
                Pointer(
                    to=items.unsafe_get(Int(order.unsafe_get(first))).bounds
                )
                .unsafe_bitcast[Float32]()
                .unsafe_offset(chunk * width)
            )
            var merged = masked_load(start, valid, SIMD[.float32, width](0.0))
            for i in range(first + 1, first + count):
                var ptr = (
                    Pointer(
                        to=items.unsafe_get(Int(order.unsafe_get(i))).bounds
                    )
                    .unsafe_bitcast[Float32]()
                    .unsafe_offset(chunk * width)
                )
                var value = masked_load(ptr, valid, SIMD[.float32, width](0.0))
                comptime if precise:
                    merged = lower.select(
                        _finite_min(merged, value), _finite_max(merged, value)
                    )
                else:
                    merged = lower.select(
                        min(merged, value), max(merged, value)
                    )
            comptime for lane in range(width):
                comptime index = chunk * width + lane
                comptime if index < 3:
                    box._min[index] = merged[lane]
                elif index < 6:
                    box._max[index - 3] = merged[lane]
        return box

    def _emit_median[
        precise: Bool = False
    ](
        mut self,
        items: ImmSpan[BoundsItem[Self.frame], _],
        order: MutSpan[UInt32, _],
        first: Int,
        count: Int,
        node: Int,
    ) -> Int:
        var box = self._range_bounds[BUILD_WIDTH, precise](
            items, order, first, count
        )
        var output = node * 2
        var next_node = node + 1
        self._store_bounds(node, box)
        self.nodes[output] = 0
        self.nodes[output + 1] = EMPTY_LANE
        if count == 1:
            self.nodes[output + 1] = items.unsafe_get(
                Int(order.unsafe_get(first))
            ).payload
        else:
            var axis = longest_axis(box.extent())
            var split: Int
            if count == 2:
                split = first + 1
            elif axis == 0:
                split = _partition_items_by_median_center[Self.frame, 0](
                    order, items, first, count, axis
                )
            elif axis == 1:
                split = _partition_items_by_median_center[Self.frame, 1](
                    order, items, first, count, axis
                )
            else:
                split = _partition_items_by_median_center[Self.frame, 2](
                    order, items, first, count, axis
                )
            next_node = self._emit_median[precise](
                items, order, first, split - first, next_node
            )
            next_node = self._emit_median[precise](
                items, order, split, first + count - split, next_node
            )
        self.nodes[output] = UInt32(next_node)
        return next_node

    def view(self) -> BoundsQueryView[origin_of(self), Self.frame]:
        # Both buffers are owned by this object; widen their field origins to
        # the owner's borrow so a view can hold them together.
        return BoundsQueryView[origin_of(self), Self.frame](
            ImmSpan(
                unsafe_ptr=self.bounds.unsafe_ptr().unsafe_origin_cast[
                    origin_of(self)
                ](),
                length=len(self.bounds),
            ),
            ImmSpan(
                unsafe_ptr=self.nodes.unsafe_ptr().unsafe_origin_cast[
                    origin_of(self)
                ](),
                length=len(self.nodes),
            ),
        )

    def group_root(self, group: UInt32) -> UInt32:
        for i in range(len(self.group_ids)):
            if self.group_ids[i] == group:
                return self.group_roots[i]
        return EMPTY_LANE

    @always_inline
    def _merge_node_bounds[
        precise: Bool
    ](self, left: Int, right: Int) -> AABB[Self.frame]:
        comptime if not precise:
            return AABB[Self.frame].merge(
                self._node_bounds(left), self._node_bounds(right)
            )
        else:
            var a = self._node_bounds(left)
            var b = self._node_bounds(right)
            var result = a
            comptime for axis in range(3):
                result._min[axis] = _finite_min(a._min[axis], b._min[axis])
                result._max[axis] = _finite_max(a._max[axis], b._max[axis])
            return result

    @always_inline
    def _node_bounds(self, index: Int) -> AABB[Self.frame]:
        var base = index * 6
        var bounds = self.bounds.unsafe_ptr()
        return AABB[Self.frame].load6(bounds, base)

    @always_inline
    def _store_bounds(mut self, index: Int, box: AABB[Self.frame]):
        var base = index * 6
        var bounds = self.bounds.unsafe_ptr()
        comptime for axis in range(3):
            bounds.unsafe_store(base + axis, box._min[axis])
            bounds.unsafe_store(base + axis + 3, box._max[axis])

    def _validate_update(
        self, boxes: ImmSpan[AABB[Self.frame], _]
    ) raises -> Bool:
        if len(boxes) != len(self.item_bounds):
            raise Error("BVH update requires unchanged item count")
        var precise = _has_subnormal(boxes)
        if precise:
            self._check_boxes[True](boxes)
        else:
            self._check_boxes[False](boxes)
        return precise

    def refit[
        width: SIMDLength = REFIT_WIDTH
    ](mut self, boxes: ImmSpan[AABB[Self.frame], _]) raises:
        """Validate first, then refit retained global/group buffers in place."""
        comptime assert size_of[AABB[Self.frame]]() == 6 * size_of[Float32]()
        var precise = self._validate_update(boxes)
        var source = boxes.unsafe_ptr().unsafe_bitcast[Float32]()
        var item_floats = self.item_bounds.unsafe_ptr().unsafe_bitcast[
            Float32
        ]()
        var total = len(boxes) * 6
        for i in range(total // width):
            item_floats.unsafe_store[width=width](
                i * width, source.unsafe_load[width=width](i * width)
            )
        for i in range((total // width) * width, total):
            item_floats.unsafe_store(i, source.unsafe_load(i))
        if precise:
            self._refit_nodes[width, True]()
        else:
            self._refit_nodes[width, False]()

    def _refit_nodes[
        width: SIMDLength = REFIT_WIDTH, precise: Bool = True
    ](mut self):
        """Refit from already validated owned item bounds; no allocation."""
        var item_floats = self.item_bounds.unsafe_ptr().unsafe_bitcast[
            Float32
        ]()
        var bounds = self.bounds.unsafe_ptr()
        var nodes = self.nodes.unsafe_ptr()
        for node in range(len(self.nodes) // 2 - 1, -1, -1):
            var primitive = nodes.unsafe_load(node * 2 + 1)
            var output = node * 6
            if primitive != EMPTY_LANE:
                var input = Int(primitive) * 6
                bounds.unsafe_store[width=4](
                    output, item_floats.unsafe_load[width=4](input)
                )
                bounds.unsafe_store[width=2](
                    output + 4, item_floats.unsafe_load[width=2](input + 4)
                )
            else:
                var left = node + 1
                var right = Int(nodes.unsafe_load(left * 2))
                comptime for chunk in range((6 + width - 1) // width):
                    var a = SIMD[.float32, width](0.0)
                    var b = SIMD[.float32, width](0.0)
                    var lower_mask = SIMD[.bool, width](fill=False)
                    comptime for lane in range(width):
                        comptime index = chunk * width + lane
                        comptime if index < 6:
                            a[lane] = bounds.unsafe_load(left * 6 + index)
                            b[lane] = bounds.unsafe_load(right * 6 + index)
                            lower_mask[lane] = index < 3
                    var merged: SIMD[.float32, width]
                    comptime if precise:
                        merged = lower_mask.select(
                            _finite_min(a, b), _finite_max(a, b)
                        )
                    else:
                        merged = lower_mask.select(min(a, b), max(a, b))
                    comptime if (chunk + 1) * width <= 6:
                        bounds.unsafe_store[width=width](
                            output + chunk * width, merged
                        )
                    else:
                        comptime for lane in range(width):
                            comptime if chunk * width + lane < 6:
                                bounds.unsafe_store(
                                    output + chunk * width + lane, merged[lane]
                                )

    def rebuild(mut self, boxes: ImmSpan[AABB[Self.frame], _]) raises:
        """Rebuild topology while retaining published buffer addresses.

        Temporary builder storage is allowed. Item count and groups stay fixed.
        Iterators must be discarded before any update or rebuild.
        """
        _ = self._validate_update(boxes)
        var fresh = Self(boxes, self.groups)
        debug_assert["safe"](len(fresh.nodes) == len(self.nodes))
        for i in range(len(self.nodes)):
            self.nodes[i] = fresh.nodes[i]
        for i in range(len(self.bounds)):
            self.bounds[i] = fresh.bounds[i]
        for i in range(len(boxes)):
            self.item_bounds[i] = boxes[i]
