from std.bit import count_leading_zeros
from std.math import abs, min, max, clamp
from std.utils.numerics import max_finite, min_finite

from bajo.core.aabb import AABB
from bajo.core.vec import Vec3f32, vmin, vmax, longest_axis, InlineArray
from bajo.core.bvh.types import (
    Ray,
    BvhNode,
    Bin,
    Fragment,
    SplitResult,
    MortonPrim,
)

comptime f32_max = max_finite[DType.float32]()
comptime f32_min = min_finite[DType.float32]()
comptime BVH_BINS = 16


@always_inline
def _partition_fragments(
    prims: UnsafePointer[UInt32, MutAnyOrigin],
    fragments: UnsafePointer[Fragment, ImmutAnyOrigin],
    first: Int,
    count: Int,
    axis: Int,
    pos: Float32,
) -> Int:
    var i = first
    var j = first + count - 1

    while i <= j:
        var frag_idx = Int(prims[i])
        var c = fragments[frag_idx].center_axis(axis)

        if c < pos:
            i += 1
        else:
            prims[i], prims[j] = prims[j], prims[i]
            j -= 1

    return i


@always_inline
def _fragment_bin(
    fragments: UnsafePointer[Fragment, ImmutAnyOrigin],
    frag_idx: Int,
    axis: Int,
    bin_min: Float32,
    bin_scale: Float32,
) -> Int:
    var c = fragments[frag_idx].center_axis(axis)
    var b_idx = Int((c - bin_min) * bin_scale)
    if b_idx < 0:
        return 0
    if b_idx >= BVH_BINS:
        return BVH_BINS - 1
    return b_idx


@always_inline
def _partition_fragments_by_bin(
    prims: UnsafePointer[UInt32, MutAnyOrigin],
    fragments: UnsafePointer[Fragment, ImmutAnyOrigin],
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
        var frag_idx = Int(prims[i])
        var b_idx = _fragment_bin(fragments, frag_idx, axis, bin_min, bin_scale)

        if b_idx <= split_bin:
            i += 1
        else:
            prims[i], prims[j] = prims[j], prims[i]
            j -= 1

    return i


@always_inline
def _sah(
    node: BvhNode,
    prims: UnsafePointer[UInt32, ImmutAnyOrigin],
    fragments: UnsafePointer[Fragment, ImmutAnyOrigin],
) -> SplitResult:
    var best = SplitResult()

    for axis in range(3):
        var min_c = f32_max
        var max_c = f32_min

        # 1. Find centroid range for this node/axis.
        for i in range(Int(node.tri_count)):
            var frag_idx = Int(prims[Int(node.left_first) + i])
            var c = fragments[frag_idx].center_axis(axis)
            min_c = min(min_c, c)
            max_c = max(max_c, c)

        if min_c == max_c:
            continue

        # 2. Bin cached primitive bounds.
        var bins = InlineArray[Bin, BVH_BINS](fill=Bin())
        var scale = Float32(BVH_BINS) / (max_c - min_c)

        for i in range(Int(node.tri_count)):
            var frag_idx = Int(prims[Int(node.left_first) + i])
            ref frag = fragments[frag_idx]

            var b_idx = _fragment_bin(fragments, frag_idx, axis, min_c, scale)
            bins[b_idx].tri_count += 1
            frag.grow_into(bins[b_idx].bounds)

        # 3. Left sweep: store both area/count and exact bounds at each split.
        var left_areas = InlineArray[Float32, BVH_BINS](fill=0.0)
        var left_counts = InlineArray[UInt32, BVH_BINS](fill=0)
        var left_bounds = InlineArray[AABB, BVH_BINS](fill=AABB.invalid())

        var left_box = AABB.invalid()
        var left_sum = UInt32(0)

        for i in range(BVH_BINS - 1):
            left_sum += bins[i].tri_count
            left_counts[i] = left_sum
            left_box.grow(bins[i].bounds)
            left_bounds[i] = left_box.copy()
            left_areas[i] = left_box.surface_area()

        # 4. Right sweep + split cost.
        var right_box = AABB.invalid()
        var right_sum = UInt32(0)

        for i in range(BVH_BINS - 1, 0, -1):
            right_sum += bins[i].tri_count
            right_box.grow(bins[i].bounds)

            var left_count = left_counts[i - 1]
            var right_count = right_sum

            if left_count == 0 or right_count == 0:
                continue

            var left_cost = left_areas[i - 1] * Float32(left_count)
            var right_cost = right_box.surface_area() * Float32(right_count)
            var cost = left_cost + right_cost

            if cost < best.cost:
                best.axis = axis
                best.bin = i - 1
                best.pos = min_c + Float32(i) / scale
                best.cost = cost
                best.bin_min = min_c
                best.bin_scale = scale
                best.left_bounds = left_bounds[i - 1].copy()
                best.right_bounds = right_box.copy()

    return best^


@always_inline
def _morton_pair_less(a: MortonPrim, b: MortonPrim) capturing -> Bool:
    if a.code < b.code:
        return True
    if a.code > b.code:
        return False
    return a.frag_idx < b.frag_idx


@always_inline
def _highest_set_bit(v: UInt32) -> Int:
    if v == 0:
        return -1
    return 31 - Int(count_leading_zeros(v))


@always_inline
def _find_lbvh_split(
    pairs: UnsafePointer[MortonPrim, ImmutAnyOrigin],
    first: Int,
    last: Int,
) -> Int:
    var first_code = pairs[first].code
    var last_code = pairs[last - 1].code

    if first_code == last_code:
        return (first + last) // 2

    var bit = _highest_set_bit(first_code ^ last_code)
    if bit < 0:
        return (first + last) // 2

    var mask = UInt32(1) << UInt32(bit)
    var left_bit = first_code & mask

    for i in range(first + 1, last):
        if (pairs[i].code & mask) != left_bit:
            return i

    return (first + last) // 2
