from std.bit import count_leading_zeros

from bajo.core.aabb import AABB
from bajo.core.morton import morton3


@fieldwise_init
struct BoundsMortonItem(TrivialRegisterPassable):
    var code: UInt32
    var item_idx: UInt32


def _bounds_morton_item_less(
    a: BoundsMortonItem,
    b: BoundsMortonItem,
) capturing -> Bool:
    if a.code < b.code:
        return True

    if a.code > b.code:
        return False

    return a.item_idx < b.item_idx


def _find_lbvh_split(
    pairs: UnsafePointer[BoundsMortonItem, ImmutAnyOrigin],
    first: Int,
    last: Int,
) -> Int:
    var first_code = pairs[first].code
    var last_code = pairs[last - 1].code
    var diff = first_code ^ last_code

    if diff == 0:
        return (first + last) / 2

    var bit = 31 - Int(count_leading_zeros(diff))
    var mask = UInt32(1) << UInt32(bit)
    var left_bit = first_code & mask

    # binary search
    var lo = first + 1
    var hi = last - 1

    while lo < hi:
        var mid = (lo + hi) / 2
        if (pairs[mid].code & mask) == left_bit:
            lo = mid + 1
        else:
            hi = mid

    return lo


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

    var pairs = List[BoundsMortonItem](capacity=item_count)

    for i, item in enumerate(builder.items):
        var centroid = item.bounds.centroid()
        var c = (centroid - centroid_bounds._min) * inv
        var code = morton3(c.x, c.y, c.z)
        pairs.append(BoundsMortonItem(code, UInt32(i)))

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
    pairs: UnsafePointer[BoundsMortonItem, ImmutAnyOrigin],
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

    var split = _find_lbvh_split(pairs, first, first + count)
    var left_count = split - first

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
        split,
        count - left_count,
    )

    ref node = builder.nodes[Int(node_idx)]
    node.set_internal(left_child_idx)
    node.aabb = AABB.merge(left_bounds, right_bounds)
    return node.aabb
