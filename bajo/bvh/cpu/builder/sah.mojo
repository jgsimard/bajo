from bajo.core import AABB
from bajo.bvh.constants import f32_max, f32_min, BVH_BINS
from .builder import BoundsItem, BoundsBvhNode


@fieldwise_init
struct BoundsSplitResult(Movable):
    var axis: Int
    var bin: Int
    var pos: Float32
    var cost: Float32
    var bin_min: Float32
    var bin_scale: Float32
    var left_bounds: AABB
    var right_bounds: AABB

    def __init__(out self):
        self.axis = -1
        self.bin = -1
        self.pos = 0.0
        self.cost = f32_max
        self.bin_min = 0.0
        self.bin_scale = 0.0
        self.left_bounds = AABB.invalid()
        self.right_bounds = AABB.invalid()

    def valid(self) -> Bool:
        return self.axis >= 0 and self.bin >= 0


def _find_sah_split(
    node: BoundsBvhNode,
    indices: UnsafePointer[mut=False, UInt32, _],
    items: UnsafePointer[mut=False, BoundsItem, _],
) -> BoundsSplitResult:
    var best = BoundsSplitResult()
    var first = Int(node.first_item())
    var count = Int(node.item_count)

    for axis in range(3):
        var min_c = f32_max
        var max_c = f32_min

        # centroid range
        for i in range(count):
            var item_idx = Int(indices[first + i])
            var c = items[item_idx].center_axis(axis)
            min_c = min(min_c, c)
            max_c = max(max_c, c)

        if min_c == max_c:
            continue

        var bins = InlineArray[BoundsBin, BVH_BINS](fill=BoundsBin())
        var scale = Float32(BVH_BINS) / (max_c - min_c)

        for i in range(count):
            var item_idx = Int(indices[first + i])
            var b_idx = _item_bin(items, item_idx, axis, min_c, scale)
            bins[b_idx].item_count += 1
            items[item_idx].grow_into(bins[b_idx].bounds)

        # from the left
        var left_prefix = InlineArray[BoundsBin, BVH_BINS](fill=BoundsBin())
        var left_box = AABB.invalid()
        var left_count = UInt32(0)

        for i in range(BVH_BINS - 1):
            left_count += bins[i].item_count
            left_box.grow(bins[i].bounds)
            left_prefix[i].item_count = left_count
            left_prefix[i].bounds = left_box

        # from the right
        var right_box = AABB.invalid()
        var right_count = UInt32(0)

        for i in range(BVH_BINS - 1, 0, -1):
            right_count += bins[i].item_count
            right_box.grow(bins[i].bounds)

            var split_bin = i - 1
            var left = left_prefix[split_bin]

            if left.item_count == 0 or right_count == 0:
                continue

            var left_cost = left.bounds.surface_area()[0] * Float32(
                left.item_count
            )
            var right_cost = right_box.surface_area()[0] * Float32(right_count)
            var cost = left_cost + right_cost

            if cost < best.cost:
                best.axis = axis
                best.bin = split_bin
                best.pos = min_c + Float32(i) / scale
                best.cost = cost
                best.bin_min = min_c
                best.bin_scale = scale
                best.left_bounds = left.bounds
                best.right_bounds = right_box

    return best^


@fieldwise_init
struct BoundsBin(TrivialRegisterPassable):
    var bounds: AABB
    var item_count: UInt32

    def __init__(out self):
        self.bounds = AABB.invalid()
        self.item_count = 0


def _item_bin[
    origin: ImmutOrigin
](
    items: UnsafePointer[BoundsItem, origin],
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


def _partition_items_by_bin(
    indices: UnsafePointer[mut=True, UInt32, _],
    items: UnsafePointer[mut=False, BoundsItem, _],
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
