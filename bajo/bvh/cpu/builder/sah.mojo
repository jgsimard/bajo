from std.math import clamp

from bajo.core import AABB, Frame
from bajo.bvh.constants import f32_max
from .builder import BoundsItem, BoundsBvhNode


@fieldwise_init
struct BoundsSplitResult[frame: Frame]:
    var axis: Int
    var bin: Int
    var pos: Float32
    var cost: Float32
    var bin_min: Float32
    var bin_scale: Float32

    def __init__(out self):
        self.axis = -1
        self.bin = -1
        self.pos = 0.0
        self.cost = f32_max
        self.bin_min = 0.0
        self.bin_scale = 0.0

    def valid(self) -> Bool:
        return self.axis >= 0 and self.bin >= 0


@fieldwise_init
struct BoundsPartitionResult[frame: Frame]:
    var split_idx: Int
    var left_bounds: AABB[Self.frame]
    var right_bounds: AABB[Self.frame]
    var left_centroid_bounds: AABB[Self.frame]
    var right_centroid_bounds: AABB[Self.frame]

    def __init__(out self, split_idx: Int):
        self.split_idx = split_idx
        self.left_bounds = AABB[Self.frame].invalid()
        self.right_bounds = AABB[Self.frame].invalid()
        self.left_centroid_bounds = AABB[Self.frame].invalid()
        self.right_centroid_bounds = AABB[Self.frame].invalid()


@fieldwise_init
struct BoundsBin[frame: Frame](TrivialRegisterPassable):
    var bounds: AABB[Self.frame]
    var item_count: UInt32

    def __init__(out self):
        self.bounds = AABB[Self.frame].invalid()
        self.item_count = 0


def _find_sah_split[
    frame: Frame, BVH_BINS: Int
](
    node: BoundsBvhNode,
    centroid_bounds: AABB[frame],
    indices: ImmSpan[UInt32, _],
    items: ImmSpan[BoundsItem[frame], _],
) -> BoundsSplitResult[frame]:
    var best = BoundsSplitResult[frame]()
    var first = Int(node.first_item())
    var count = Int(node.item_count)
    debug_assert["safe", _use_compiler_assume=True](
        first >= 0
        and count > 0
        and first <= len(indices)
        and count <= len(indices) - first,
        "SAH node range is outside item indices",
    )
    debug_assert["safe", _use_compiler_assume=True](
        len(indices) == len(items),
        "BVH item indices and items have different lengths",
    )

    var bin_min = centroid_bounds._min
    var centroid_extent = centroid_bounds.extent()
    var bin_scale = centroid_extent
    comptime for axis in range(3):
        if centroid_extent[axis] > 0.0:
            bin_scale.set_axis[axis](Float32(BVH_BINS) / centroid_extent[axis])
        else:
            bin_scale.set_axis[axis](0.0)
    var bins = Array[BoundsBin[frame], 3 * BVH_BINS](fill=BoundsBin[frame]())

    # Bin all three axes in one primitive-range pass.
    var node_indices = indices[first : first + count]
    for item_idx_u32 in node_indices:
        var item_idx = Int(item_idx_u32)
        ref item = items.unsafe_get(item_idx)
        var centroid = item.bounds.centroid()

        comptime for axis in range(3):
            if centroid_extent[axis] > 0.0:
                var b_idx = _centroid_bin[BVH_BINS](
                    centroid[axis], bin_min[axis], bin_scale[axis]
                )
                var flat_idx = axis * BVH_BINS + b_idx
                bins[flat_idx].item_count += 1
                item.grow_into(bins[flat_idx].bounds)

    # Prefix/suffix evaluation is constant-sized and performs no primitive
    # range scans.
    comptime for axis in range(3):
        if centroid_extent[axis] == 0.0:
            continue

        var axis_base = axis * BVH_BINS
        var left_prefix = Array[BoundsBin[frame], BVH_BINS](
            fill=BoundsBin[frame]()
        )
        var left_box = AABB[frame].invalid()
        var left_count = UInt32(0)

        for i in range(BVH_BINS - 1):
            ref bin = bins[axis_base + i]
            left_count += bin.item_count
            left_box.grow(bin.bounds)
            left_prefix[i].item_count = left_count
            left_prefix[i].bounds = left_box

        var right_box = AABB[frame].invalid()
        var right_count = UInt32(0)

        for i in range(BVH_BINS - 1, 0, -1):
            ref bin = bins[axis_base + i]
            right_count += bin.item_count
            right_box.grow(bin.bounds)

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
                best.pos = bin_min[axis] + Float32(i) / bin_scale[axis]
                best.cost = cost
                best.bin_min = bin_min[axis]
                best.bin_scale = bin_scale[axis]

    return best^


def _centroid_bin[
    BVH_BINS: Int
](centroid: Float32, bin_min: Float32, bin_scale: Float32) -> Int:
    var b_idx = Int((centroid - bin_min) * bin_scale)
    return clamp(b_idx, 0, BVH_BINS - 1)


def _grow_partition_side[
    frame: Frame
](
    item: BoundsItem[frame],
    mut bounds: AABB[frame],
    mut centroid_bounds: AABB[frame],
):
    item.grow_into(bounds)
    centroid_bounds.grow(item.bounds.centroid())


def _partition_items_by_bin[
    frame: Frame, BVH_BINS: Int
](
    indices: MutSpan[UInt32, _],
    items: ImmSpan[BoundsItem[frame], _],
    first: Int,
    count: Int,
    axis: Int,
    split_bin: Int,
    bin_min: Float32,
    bin_scale: Float32,
) -> BoundsPartitionResult[frame]:
    debug_assert["safe", _use_compiler_assume=True](
        first >= 0
        and count > 0
        and first <= len(indices)
        and count <= len(indices) - first,
        "SAH partition range is outside item indices",
    )
    debug_assert["safe", _use_compiler_assume=True](
        len(indices) == len(items),
        "BVH item indices and items have different lengths",
    )
    debug_assert["safe", _use_compiler_assume=True](
        axis >= 0 and axis < 3,
        "SAH partition axis is outside [0, 3)",
    )
    debug_assert["safe", _use_compiler_assume=True](
        split_bin >= 0 and split_bin < BVH_BINS,
        "SAH split bin is outside the bin array",
    )

    var out = BoundsPartitionResult[frame](first)
    var node_indices = indices[first : first + count]
    var i = 0
    var j = len(node_indices) - 1

    # Each primitive is classified once. Bounds are accumulated before a
    # right-side item is swapped out of the unclassified range.
    while i <= j:
        var item_idx = Int(node_indices.unsafe_get(i))
        ref item = items.unsafe_get(item_idx)
        var centroid = item.bounds.centroid()
        var b_idx = _centroid_bin[BVH_BINS](centroid[axis], bin_min, bin_scale)

        if b_idx <= split_bin:
            _grow_partition_side(
                item,
                out.left_bounds,
                out.left_centroid_bounds,
            )
            i += 1
        else:
            _grow_partition_side(
                item,
                out.right_bounds,
                out.right_centroid_bounds,
            )
            node_indices.unsafe_swap_elements(i, j)
            j -= 1

    out.split_idx = first + i
    return out^


def _calculate_partition_bounds[
    frame: Frame
](
    indices: ImmSpan[UInt32, _],
    items: ImmSpan[BoundsItem[frame], _],
    first: Int,
    count: Int,
    split_idx: Int,
) -> BoundsPartitionResult[frame]:
    var out = BoundsPartitionResult[frame](split_idx)

    for offset in range(count):
        var item_idx = Int(indices.unsafe_get(first + offset))
        ref item = items.unsafe_get(item_idx)

        if first + offset < split_idx:
            _grow_partition_side(
                item,
                out.left_bounds,
                out.left_centroid_bounds,
            )
        else:
            _grow_partition_side(
                item,
                out.right_bounds,
                out.right_centroid_bounds,
            )

    return out^
