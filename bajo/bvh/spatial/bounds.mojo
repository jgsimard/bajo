"""Owning spatial BVH with original item IDs and independent group roots."""

from bajo.core import AABB, Frame, Point3f32
from bajo.bvh.constants import EMPTY_LANE
from bajo.bvh.cpu.builder import BinaryBoundsBvh, BoundsItem
from .query import BoundsQueryView


struct SpatialBoundsBvh[frame: Frame = .WORLD](Copyable):
    """CPU owner of the shared spatial-query representation.

    Construction reuses Bajo's median builder, then emits stackless preorder
    nodes. The global tree and each group tree reference original item IDs.
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
        for box in boxes:
            if not box.is_valid()[0]:
                raise Error("BVH item bounds must be finite and ordered")
            self.item_bounds.append(box)
        for group in groups:
            self.groups.append(group)
            var found = False
            for previous in self.group_ids:
                found |= previous == group
            if not found:
                self.group_ids.append(group)
        self._build_tree(EMPTY_LANE, False)
        for group in self.group_ids:
            self.group_roots.append(UInt32(len(self.nodes) // 2))
            self._build_tree(group, True)

    def _build_tree(mut self, group: UInt32, grouped: Bool):
        var items = List[BoundsItem[Self.frame]]()
        for i in range(len(self.item_bounds)):
            if not grouped or self.groups[i] == group:
                items.append(BoundsItem(self.item_bounds[i], UInt32(i)))
        if len(items) > 0:
            var binary = BinaryBoundsBvh[Self.frame, 1](items^)
            self._emit(binary, 0)

    def _emit(mut self, binary: BinaryBoundsBvh[Self.frame, 1], index: UInt32):
        var output = len(self.nodes)
        ref node = binary.nodes[Int(index)]
        self.bounds.extend(
            [
                node.aabb._min.x,
                node.aabb._min.y,
                node.aabb._min.z,
                node.aabb._max.x,
                node.aabb._max.y,
                node.aabb._max.z,
            ]
        )
        self.nodes.extend([UInt32(0), EMPTY_LANE])
        if node.is_leaf():
            var item = binary.item_indices[Int(node.first_item())]
            self.nodes[output + 1] = binary.items[Int(item)].payload
        else:
            self._emit(binary, node.left_child())
            self._emit(binary, node.right_child())
        self.nodes[output] = UInt32(len(self.nodes) // 2)

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

    def _node_bounds(self, index: Int) -> AABB[Self.frame]:
        var base = index * 6
        return AABB(
            Point3f32[Self.frame](
                self.bounds[base], self.bounds[base + 1], self.bounds[base + 2]
            ),
            Point3f32[Self.frame](
                self.bounds[base + 3],
                self.bounds[base + 4],
                self.bounds[base + 5],
            ),
        )

    def _store_bounds(mut self, index: Int, box: AABB[Self.frame]):
        var base = index * 6
        for axis in range(3):
            self.bounds[base + axis] = box._min[axis]
            self.bounds[base + axis + 3] = box._max[axis]

    def _validate_update(self, boxes: ImmSpan[AABB[Self.frame], _]) raises:
        if len(boxes) != len(self.item_bounds):
            raise Error("BVH update requires unchanged item count")
        for box in boxes:
            if not box.is_valid()[0]:
                raise Error("BVH item bounds must be finite and ordered")

    def refit(mut self, boxes: ImmSpan[AABB[Self.frame], _]) raises:
        """Update all global/group bounds in place, preserving node topology.

        Item count, original IDs and group membership stay fixed. Validation
        precedes mutation; failed updates preserve the existing BVH.
        """
        self._validate_update(boxes)
        for i in range(len(boxes)):
            self.item_bounds[i] = boxes[i]
        for i in range(len(self.nodes) // 2 - 1, -1, -1):
            var primitive = self.nodes[i * 2 + 1]
            var box: AABB[Self.frame]
            if primitive != EMPTY_LANE:
                box = self.item_bounds[Int(primitive)]
            else:
                var left = i + 1
                var right = Int(self.nodes[left * 2])
                box = AABB[Self.frame].merge(
                    self._node_bounds(left), self._node_bounds(right)
                )
            self._store_bounds(i, box)

    def rebuild(mut self, boxes: ImmSpan[AABB[Self.frame], _]) raises:
        """Rebuild topology while retaining published buffer addresses.

        Temporary builder storage is allowed. Item count and groups stay fixed.
        Iterators must be discarded before any update or rebuild.
        """
        self._validate_update(boxes)
        var fresh = Self(boxes, self.groups)
        debug_assert["safe"](len(fresh.nodes) == len(self.nodes))
        for i in range(len(self.nodes)):
            self.nodes[i] = fresh.nodes[i]
        for i in range(len(self.bounds)):
            self.bounds[i] = fresh.bounds[i]
        for i in range(len(boxes)):
            self.item_bounds[i] = boxes[i]
