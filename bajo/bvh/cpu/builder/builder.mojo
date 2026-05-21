from std.math import min, max
from std.utils.numerics import max_finite, min_finite

from bajo.core.aabb import AABB, AxisAlignedBoundingBox
from bajo.core.vec import Vec3f32, vmin, vmax, longest_axis
from bajo.bvh.constants import EMPTY_LANE

from .sah import _sah_items, _partition_items_by_bin
from .lbvh import _build_lbvh


comptime f32_max = max_finite[DType.float32]()
comptime f32_min = min_finite[DType.float32]()


struct BoundsBvhBuilder[leaf_size: Int](Copyable):
    """Generic binary BVH builder over AABBs/items.

    Build modes:
        "median" -> top-down spatial median
        "sah"    -> top-down binned SAH
        "lbvh"   -> sorted-Morton recursive LBVH
    """

    var nodes: List[BoundsBvhNode]
    var item_indices: List[UInt32]
    var items: List[BoundsItem]
    var item_count: UInt32
    var nodes_used: UInt32

    def __init__(out self, items: List[BoundsItem]):
        self.items = items.copy()
        self.item_count = UInt32(len(self.items))

        self.item_indices = List[UInt32](capacity=Int(self.item_count))
        for i in range(Int(self.item_count)):
            self.item_indices.append(UInt32(i))

        var max_nodes = 1
        if self.item_count > 0:
            max_nodes = Int(self.item_count * 2 - 1)

        self.nodes = List[BoundsBvhNode](capacity=max_nodes)
        for _ in range(max_nodes):
            self.nodes.append(BoundsBvhNode())

        if self.item_count > 0:
            self.nodes_used = 1
            self.nodes[0].set_leaf(0, self.item_count)
            self.update_node_bounds(0)
        else:
            self.nodes_used = 0

    def update_node_bounds(mut self, node_idx: UInt32):
        ref node = self.nodes[Int(node_idx)]
        node.aabb = AABB.invalid()

        var first = Int(node.first_item())
        for i in range(Int(node.item_count)):
            var item_idx = Int(self.item_indices[first + i])
            self.items[item_idx].grow_into(node.aabb)

    def build[split_method: String = "median"](mut self):
        comptime assert split_method in ["median", "sah", "lbvh"]
        if self.item_count == 0:
            self.nodes_used = 0
            return

        comptime if split_method == "lbvh":
            _build_lbvh[Self.leaf_size](self)
        else:
            self.nodes_used = 1
            self.nodes[0].set_leaf(0, self.item_count)
            self.update_node_bounds(0)
            self._subdivide[split_method](0)

    def _subdivide[split_method: String](mut self, node_idx: UInt32):
        comptime assert split_method in ["median", "sah"]

        ref node = self.nodes[Int(node_idx)]
        if node.item_count <= UInt32(Self.leaf_size):
            return

        var axis: Int
        var pos: Float32
        var use_sah_bounds = False
        var split_bin = -1
        var split_bin_min = Float32(0.0)
        var split_bin_scale = Float32(0.0)
        var cached_left_bounds = AABB.invalid()
        var cached_right_bounds = AABB.invalid()

        comptime if split_method == "median":
            var extent = node.aabb._max - node.aabb._min
            axis = longest_axis(extent)
            pos = node.aabb._min[axis] + extent[axis] * 0.5

        elif split_method == "sah":
            var split = _sah_items(
                node,
                self.item_indices.unsafe_ptr(),
                self.items.unsafe_ptr(),
            )

            var leaf_cost = node.surface_area() * Float32(node.item_count)
            var split_cost = split.cost + node.surface_area()

            if (not split.valid()) or split_cost >= leaf_cost:
                if node.item_count > UInt32(Self.leaf_size):
                    var extent = node.aabb._max - node.aabb._min
                    axis = longest_axis(extent)
                    pos = node.aabb._min[axis] + extent[axis] * 0.5
                else:
                    return
            else:
                axis = split.axis
                pos = split.pos
                split_bin = split.bin
                split_bin_min = split.bin_min
                split_bin_scale = split.bin_scale
                cached_left_bounds = split.left_bounds
                cached_right_bounds = split.right_bounds
                use_sah_bounds = True
        else:
            comptime assert False, "Unknown BoundsBvh split method"

        var split_idx: Int
        comptime if split_method == "sah":
            if use_sah_bounds:
                split_idx = _partition_items_by_bin(
                    self.item_indices.unsafe_ptr(),
                    self.items.unsafe_ptr(),
                    Int(node.first_item()),
                    Int(node.item_count),
                    axis,
                    split_bin,
                    split_bin_min,
                    split_bin_scale,
                )
            else:
                split_idx = _partition_items(
                    self.item_indices.unsafe_ptr(),
                    self.items.unsafe_ptr(),
                    Int(node.first_item()),
                    Int(node.item_count),
                    axis,
                    pos,
                )
        else:
            split_idx = _partition_items(
                self.item_indices.unsafe_ptr(),
                self.items.unsafe_ptr(),
                Int(node.first_item()),
                Int(node.item_count),
                axis,
                pos,
            )

        var left_count = UInt32(split_idx - Int(node.first_item()))

        if left_count == 0 or left_count == node.item_count:
            left_count = node.item_count / 2
            split_idx = Int(node.first_item()) + Int(left_count)
            use_sah_bounds = False

        var left_child_idx = self.nodes_used
        self.nodes_used += 2

        ref left_child = self.nodes[Int(left_child_idx)]
        ref right_child = self.nodes[Int(left_child_idx + 1)]

        left_child.set_leaf(node.first_item(), left_count)
        right_child.set_leaf(UInt32(split_idx), node.item_count - left_count)

        comptime if split_method == "sah":
            if use_sah_bounds:
                left_child.aabb = cached_left_bounds
                right_child.aabb = cached_right_bounds
            else:
                self.update_node_bounds(left_child_idx)
                self.update_node_bounds(left_child_idx + 1)
        else:
            self.update_node_bounds(left_child_idx)
            self.update_node_bounds(left_child_idx + 1)

        node.set_internal(left_child_idx)

        self._subdivide[split_method](left_child_idx)
        self._subdivide[split_method](left_child_idx + 1)

    def tree_quality(self) -> Float32:
        if self.nodes_used == 0:
            return 0.0

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
struct BoundsItem(TrivialRegisterPassable):
    """Generic build item.

    `bounds` is what the BVH builder sees.

    `payload` is owned by the caller:
    - TriangleBvh: triangle id
    - TLAS: instance id
    - Broadphase: index into an ItemRef array
    """

    var bounds: AABB
    var payload: UInt32

    def center_axis(self, axis: Int) -> Float32:
        return self.bounds.centroid()[axis]

    def grow_into(self, mut aabb: AABB):
        aabb._min = vmin(aabb._min, self.bounds._min)
        aabb._max = vmax(aabb._max, self.bounds._max)


@fieldwise_init
struct BoundsBvhNode(TrivialRegisterPassable):
    """Binary builder node over generic item ranges.

    Leaf:
        first_or_left = first item in item_indices
        item_count    = number of items

    Internal:
        first_or_left = left child node index
        item_count    = 0
        right child   = left child + 1
    """

    var aabb: AABB
    var first_or_left: UInt32
    var item_count: UInt32

    def __init__(out self):
        self.aabb = AABB.invalid()
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


def _partition_items(
    indices: UnsafePointer[UInt32, MutAnyOrigin],
    items: UnsafePointer[BoundsItem, ImmutAnyOrigin],
    first: Int,
    count: Int,
    axis: Int,
    pos: Float32,
) -> Int:
    var i = first
    var j = first + count - 1

    while i <= j:
        var item_idx = Int(indices[i])
        var c = items[item_idx].center_axis(axis)

        if c < pos:
            i += 1
        else:
            indices[i], indices[j] = indices[j], indices[i]
            j -= 1

    return i
