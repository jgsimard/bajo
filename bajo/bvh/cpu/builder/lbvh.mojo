from std.bit import count_leading_zeros

from bajo.core.aabb import AABB
from bajo.core.morton import morton3


@fieldwise_init
struct MortonItem(TrivialRegisterPassable):
    var code: UInt32
    var item_idx: UInt32


def _bounds_morton_item_less(
    a: MortonItem,
    b: MortonItem,
) capturing -> Bool:
    if a.code < b.code:
        return True

    if a.code > b.code:
        return False

    return a.item_idx < b.item_idx


def _common_prefix(
    pairs: UnsafePointer[MortonItem, ImmutAnyOrigin],
    i: Int,
    j: Int,
    n: Int,
) -> Int:
    if j < 0 or j >= n:
        return -1

    var a = pairs[i].code
    var b = pairs[j].code

    if a != b:
        return Int(count_leading_zeros(a ^ b))

    # duplicate Morton codes are ordered by sorted position
    var x = UInt32(i) ^ UInt32(j)
    if x == 0:
        return 64

    return 32 + Int(count_leading_zeros(x))


def _lbvh_find_split(
    pairs: UnsafePointer[MortonItem, ImmutAnyOrigin],
    first: Int,
    last: Int,
    n: Int,
) -> Int:
    var node_prefix = _common_prefix(pairs, first, last, n)

    var split = first
    var step = last - first

    while step > 1:
        step = (step + 1) >> 1

        var new_split = split + step
        if new_split < last:
            var split_prefix = _common_prefix(
                pairs,
                first,
                new_split,
                n,
            )

            if split_prefix > node_prefix:
                split = new_split

    return split


def _build_lbvh[leaf_size: Int](mut builder: BoundsBvhBuilder[leaf_size]):
    """Build a binary LBVH using sorted Morton codes over BoundsItem centers."""

    if builder.item_count == 0:
        builder.nodes_used = 0
        return

    builder.nodes_used = 1
    var item_count = Int(builder.item_count)
    var centroid_bounds = AABB.invalid()
    for item in builder.items:
        centroid_bounds.grow(item.bounds.centroid())

    var extent = centroid_bounds.extent()
    var inv = extent.safe_inv()

    var pairs = List[MortonItem](capacity=item_count)

    for i, item in enumerate(builder.items):
        var centroid = item.bounds.centroid()
        var c = (centroid - centroid_bounds._min) * inv
        var code = morton3(c.x, c.y, c.z)
        pairs.append(MortonItem(code, UInt32(i)))

    sort[_bounds_morton_item_less](Span(pairs))

    for i in range(len(pairs)):
        builder.item_indices[i] = pairs[i].item_idx

    _ = _build_lbvh_recursive[leaf_size](
        builder,
        pairs.unsafe_ptr(),
        0,
        0,
        item_count,
    )


def _build_lbvh_recursive[
    leaf_size: Int
](
    mut builder: BoundsBvhBuilder[leaf_size],
    pairs: UnsafePointer[MortonItem, ImmutAnyOrigin],
    node_idx: UInt32,
    first: Int,
    count: Int,
) -> AABB:
    if count <= leaf_size:
        ref leaf = builder.nodes[Int(node_idx)]
        leaf.set_leaf(UInt32(first), UInt32(count))
        leaf.aabb = AABB.invalid()

        for i in range(count):
            var item_idx = Int(builder.item_indices[first + i])
            builder.items[item_idx].grow_into(leaf.aabb)

        return leaf.aabb

    var last = first + count - 1
    var split = _lbvh_find_split(
        pairs,
        first,
        last,
        Int(builder.item_count),
    )

    var left_count = split - first + 1
    var right_count = count - left_count

    var left_child_idx = builder.nodes_used
    builder.nodes_used += 2

    var left_bounds = _build_lbvh_recursive[leaf_size](
        builder,
        pairs,
        left_child_idx,
        first,
        left_count,
    )

    var right_bounds = _build_lbvh_recursive[leaf_size](
        builder,
        pairs,
        left_child_idx + 1,
        split + 1,
        right_count,
    )

    ref node = builder.nodes[Int(node_idx)]
    node.set_internal(left_child_idx)
    node.aabb = AABB.merge(left_bounds, right_bounds)
    return node.aabb
