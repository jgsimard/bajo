from bajo.core import AABB, Frame, vmin, vmax


@fieldwise_init
struct BoundsItem[frame: Frame](TrivialRegisterPassable):
    """Generic build item pairing bounds with a caller-owned payload."""

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
