from std.math import min, max
from std.utils.numerics import max_finite, min_finite
from std.bit import count_leading_zeros

from bajo.core.aabb import AABB, AxisAlignedBoundingBox
from bajo.core.intersect import (
    intersect_ray_tri,
    intersect_ray_aabb,
    intersect_ray_sphere,
)
from bajo.core.mat import Mat44f32, transform_point, transform_vector
from bajo.core.morton import morton3
from bajo.core.vec import Vec3, Vec3f32, vmin, vmax, longest_axis, dot
from bajo.bvh.constants import EMPTY_LANE
from bajo.bvh.types import Ray


comptime f32_max = max_finite[DType.float32]()
comptime f32_min = min_finite[DType.float32]()
comptime BVH_BINS = 16


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

    @always_inline
    def __init__(out self):
        self.bounds = AABB.invalid()
        self.payload = 0

    @always_inline
    def center_axis(self, axis: Int) -> Float32:
        return self.bounds.centroid()[axis]

    @always_inline
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

    @always_inline
    def __init__(out self):
        self.aabb = AABB.invalid()
        self.first_or_left = 0
        self.item_count = 0

    @always_inline
    def is_leaf(self) -> Bool:
        return self.item_count > 0

    @always_inline
    def is_internal(self) -> Bool:
        return self.item_count == 0

    @always_inline
    def first_item(self) -> UInt32:
        return self.first_or_left

    @always_inline
    def left_child(self) -> UInt32:
        return self.first_or_left

    @always_inline
    def right_child(self) -> UInt32:
        return self.first_or_left + 1

    @always_inline
    def set_leaf(mut self, first_item: UInt32, item_count: UInt32):
        self.first_or_left = first_item
        self.item_count = item_count

    @always_inline
    def set_internal(mut self, left_child: UInt32):
        self.first_or_left = left_child
        self.item_count = 0

    @always_inline
    def surface_area(self) -> Float32:
        return self.aabb.surface_area()[0]


@fieldwise_init
struct BoundsBin(TrivialRegisterPassable):
    var bounds: AABB
    var item_count: UInt32

    @always_inline
    def __init__(out self):
        self.bounds = AABB.invalid()
        self.item_count = 0


@fieldwise_init
struct BoundsSplitResult(Copyable):
    var axis: Int
    var bin: Int
    var pos: Float32
    var cost: Float32
    var bin_min: Float32
    var bin_scale: Float32
    var left_bounds: AABB
    var right_bounds: AABB

    @always_inline
    def __init__(out self):
        self.axis = -1
        self.bin = -1
        self.pos = 0.0
        self.cost = f32_max
        self.bin_min = 0.0
        self.bin_scale = 0.0
        self.left_bounds = AABB.invalid()
        self.right_bounds = AABB.invalid()

    @always_inline
    def valid(self) -> Bool:
        return self.axis >= 0 and self.bin >= 0


@always_inline
def _item_bin(
    items: UnsafePointer[BoundsItem, ImmutAnyOrigin],
    item_idx: Int,
    axis: Int,
    bin_min: Float32,
    bin_scale: Float32,
) -> Int:
    var c = items[item_idx].center_axis(axis)
    var b_idx = Int((c - bin_min) * bin_scale)

    if b_idx < 0:
        return 0
    if b_idx >= BVH_BINS:
        return BVH_BINS - 1

    return b_idx


@always_inline
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


@always_inline
def _partition_items_by_bin(
    indices: UnsafePointer[UInt32, MutAnyOrigin],
    items: UnsafePointer[BoundsItem, ImmutAnyOrigin],
    first: Int,
    count: Int,
    axis: Int,
    split_bin: Int,
    bin_min: Float32,
    bin_scale: Float32,
) -> Int:
    var i = first
    var j = first + count - 1

    while i <= j:
        var item_idx = Int(indices[i])
        var b_idx = _item_bin(items, item_idx, axis, bin_min, bin_scale)

        if b_idx <= split_bin:
            i += 1
        else:
            indices[i], indices[j] = indices[j], indices[i]
            j -= 1

    return i


@always_inline
def _sah_items(
    node: BoundsBvhNode,
    indices: UnsafePointer[UInt32, ImmutAnyOrigin],
    items: UnsafePointer[BoundsItem, ImmutAnyOrigin],
) -> BoundsSplitResult:
    var best = BoundsSplitResult()

    for axis in range(3):
        var min_c = f32_max
        var max_c = f32_min

        for i in range(Int(node.item_count)):
            var item_idx = Int(indices[Int(node.first_item()) + i])
            var c = items[item_idx].center_axis(axis)
            min_c = min(min_c, c)
            max_c = max(max_c, c)

        if min_c == max_c:
            continue

        var bins = InlineArray[BoundsBin, BVH_BINS](fill=BoundsBin())
        var scale = Float32(BVH_BINS) / (max_c - min_c)

        for i in range(Int(node.item_count)):
            var item_idx = Int(indices[Int(node.first_item()) + i])
            var b_idx = _item_bin(items, item_idx, axis, min_c, scale)
            bins[b_idx].item_count += 1
            items[item_idx].grow_into(bins[b_idx].bounds)

        var left_areas = InlineArray[Float32, BVH_BINS](fill=0.0)
        var left_counts = InlineArray[UInt32, BVH_BINS](fill=0)
        var left_bounds = InlineArray[AABB, BVH_BINS](fill=AABB.invalid())

        var left_box = AABB.invalid()
        var left_sum = UInt32(0)

        for i in range(BVH_BINS - 1):
            left_sum += bins[i].item_count
            left_counts[i] = left_sum
            left_box.grow(bins[i].bounds)
            left_bounds[i] = left_box
            left_areas[i] = left_box.surface_area()[0]

        var right_box = AABB.invalid()
        var right_sum = UInt32(0)

        for i in range(BVH_BINS - 1, 0, -1):
            right_sum += bins[i].item_count
            right_box.grow(bins[i].bounds)

            var left_count = left_counts[i - 1]
            var right_count = right_sum

            if left_count == 0 or right_count == 0:
                continue

            var left_cost = left_areas[i - 1] * Float32(left_count)
            var right_cost = right_box.surface_area()[0] * Float32(right_count)
            var cost = left_cost + right_cost

            if cost < best.cost:
                best.axis = axis
                best.bin = i - 1
                best.pos = min_c + Float32(i) / scale
                best.cost = cost
                best.bin_min = min_c
                best.bin_scale = scale
                best.left_bounds = left_bounds[i - 1]
                best.right_bounds = right_box

    return best^


@fieldwise_init
struct BoundsMortonItem(TrivialRegisterPassable):
    var code: UInt32
    var item_idx: UInt32

    @always_inline
    def __init__(out self):
        self.code = 0
        self.item_idx = 0


@always_inline
def _bounds_morton_item_less(
    a: BoundsMortonItem,
    b: BoundsMortonItem,
) capturing -> Bool:
    if a.code < b.code:
        return True

    if a.code > b.code:
        return False

    return a.item_idx < b.item_idx


@always_inline
def _highest_set_bit(v: UInt32) -> Int:
    if v == 0:
        return -1

    return 31 - Int(count_leading_zeros(v))


@always_inline
def _find_lbvh_split(
    pairs: UnsafePointer[BoundsMortonItem, ImmutAnyOrigin],
    first: Int,
    last: Int,
) -> Int:
    var first_code = pairs[first].code
    var last_code = pairs[last - 1].code

    if first_code == last_code:
        return (first + last) / 2

    var bit = _highest_set_bit(first_code ^ last_code)
    if bit < 0:
        return (first + last) / 2

    var mask = UInt32(1) << UInt32(bit)
    var left_bit = first_code & mask

    for i in range(first + 1, last):
        if (pairs[i].code & mask) != left_bit:
            return i

    return (first + last) / 2


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

    def __init__(out self):
        self.nodes = List[BoundsBvhNode]()
        self.item_indices = List[UInt32]()
        self.items = List[BoundsItem]()
        self.item_count = 0
        self.nodes_used = 0

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

    @always_inline
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
            self.build_lbvh()
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

    def _build_lbvh_recursive(
        mut self,
        pairs: UnsafePointer[BoundsMortonItem, ImmutAnyOrigin],
        node_idx: UInt32,
        first: Int,
        count: Int,
    ) -> AABB:
        if count <= Self.leaf_size:
            ref leaf = self.nodes[Int(node_idx)]
            leaf.set_leaf(UInt32(first), UInt32(count))
            leaf.aabb = AABB.invalid()

            for i in range(count):
                var item_idx = Int(self.item_indices[first + i])
                self.items[item_idx].grow_into(leaf.aabb)

            return leaf.aabb

        var split = _find_lbvh_split(pairs, first, first + count)
        var left_count = split - first

        if left_count <= 0 or left_count >= count:
            split = first + count / 2
            left_count = split - first

        var left_child_idx = self.nodes_used
        self.nodes_used += 2

        var left_bounds = self._build_lbvh_recursive(
            pairs,
            left_child_idx,
            first,
            left_count,
        )

        var right_bounds = self._build_lbvh_recursive(
            pairs,
            left_child_idx + 1,
            split,
            count - left_count,
        )

        ref node = self.nodes[Int(node_idx)]
        node.set_internal(left_child_idx)
        node.aabb = AABB.invalid()
        node.aabb.grow(left_bounds)
        node.aabb.grow(right_bounds)

        return node.aabb

    def build_lbvh(mut self):
        """Build a binary LBVH using sorted Morton codes over BoundsItem centers.

        This is a CPU reference builder.
        """
        self.nodes_used = 1

        if self.item_count == 0:
            self.nodes_used = 0
            return

        self.nodes[0].set_leaf(0, self.item_count)

        var centroid_bounds = AABB.invalid()
        for i in range(Int(self.item_count)):
            centroid_bounds.grow(self.items[i].bounds.centroid())

        var extent = centroid_bounds.extent()
        var inv = extent.safe_inv()

        var pairs = List[BoundsMortonItem](capacity=Int(self.item_count))

        for i in range(Int(self.item_count)):
            var centroid = self.items[i].bounds.centroid()
            var c = (centroid - centroid_bounds._min) * inv
            var code = morton3(c.x, c.y, c.z)
            pairs.append(BoundsMortonItem(code, UInt32(i)))

        var span = Span(pairs)
        sort[_bounds_morton_item_less](span)

        for i in range(len(pairs)):
            self.item_indices[i] = pairs[i].item_idx

        _ = self._build_lbvh_recursive(
            pairs.unsafe_ptr(),
            0,
            0,
            Int(self.item_count),
        )

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
struct WideBvhNode[width: Int](Copyable):
    """Lane node used by BoundsBvh.

    Lane encoding:
        counts[i] == EMPTY_LANE -> unused lane
        counts[i] == 0          -> child node, data[i] = child node index
        counts[i] > 0           -> leaf range, data[i] = first item
    """

    var aabb: AxisAlignedBoundingBox[DType.float32, Self.width]
    var data: SIMD[DType.uint32, Self.width]
    var counts: SIMD[DType.uint32, Self.width]

    @always_inline
    def __init__(out self):
        self.aabb = AxisAlignedBoundingBox[DType.float32, Self.width].invalid()
        self.data = SIMD[DType.uint32, Self.width](0)
        self.counts = SIMD[DType.uint32, Self.width](EMPTY_LANE)


struct BoundsBvh[width: Int](Copyable):
    """Generic wide/lane BVH layout with range leaves."""

    var nodes: List[WideBvhNode[Self.width]]
    var item_indices: List[UInt32]
    var item_payloads: List[UInt32]

    def __init__(out self):
        self.nodes = List[WideBvhNode[Self.width]]()
        self.item_indices = List[UInt32]()
        self.item_payloads = List[UInt32]()

    def __init__(out self, bvh: BoundsBvhBuilder):
        self.nodes = List[WideBvhNode[Self.width]]()
        self.item_indices = bvh.item_indices.copy()
        self.item_payloads = List[UInt32](capacity=len(bvh.items))

        for i in range(len(bvh.items)):
            self.item_payloads.append(bvh.items[i].payload)

        if bvh.nodes_used > 0:
            _ = self._collapse(bvh, 0)

    def _collapse(mut self, bvh: BoundsBvhBuilder, bin_idx: UInt32) -> UInt32:
        var wide_idx = UInt32(len(self.nodes))
        self.nodes.append(WideBvhNode[Self.width]())

        var pool = InlineArray[UInt32, Self.width](fill=bin_idx)
        var p_size = 1

        # Pull up the largest internal nodes until we fill the wide node or run
        # out of internal nodes.
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

        var node = WideBvhNode[Self.width]()

        comptime for i in range(Self.width):
            if i < p_size:
                ref n = bvh.nodes[Int(pool[i])]

                node.aabb._min.x[i] = n.aabb._min.x
                node.aabb._min.y[i] = n.aabb._min.y
                node.aabb._min.z[i] = n.aabb._min.z

                node.aabb._max.x[i] = n.aabb._max.x
                node.aabb._max.y[i] = n.aabb._max.y
                node.aabb._max.z[i] = n.aabb._max.z

                if n.is_leaf():
                    node.data[i] = n.first_item()
                    node.counts[i] = n.item_count
                else:
                    node.data[i] = self._collapse(bvh, pool[i])
                    node.counts[i] = 0
            else:
                node.counts[i] = EMPTY_LANE

        self.nodes[Int(wide_idx)] = node^
        return wide_idx

    def root_bounds(self) -> AABB:
        if len(self.nodes) == 0:
            return AABB.invalid()

        ref root = self.nodes[0]
        var out = AABB.invalid()

        comptime for lane in range(Self.width):
            if root.counts[lane] != EMPTY_LANE:
                out.grow(
                    Vec3f32(
                        root.aabb._min.x[lane],
                        root.aabb._min.y[lane],
                        root.aabb._min.z[lane],
                    )
                )
                out.grow(
                    Vec3f32(
                        root.aabb._max.x[lane],
                        root.aabb._max.y[lane],
                        root.aabb._max.z[lane],
                    )
                )

        return out
