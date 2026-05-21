from std.utils.numerics import max_finite, min_finite

from bajo.core.aabb import AABB
from .builder import BoundsItem, BoundsBvhNode


comptime f32_max = max_finite[DType.float32]()
comptime f32_min = min_finite[DType.float32]()
comptime BVH_BINS = 16


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
struct BoundsBin(TrivialRegisterPassable):
    var bounds: AABB
    var item_count: UInt32

    def __init__(out self):
        self.bounds = AABB.invalid()
        self.item_count = 0


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
