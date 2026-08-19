from max.algorithm import parallelize
from std.atomic import Atomic, Ordering
from std.sys import num_logical_cores

from bajo.core import AABB, Frame
from bajo.core.morton import morton3, morton_common_prefix
from ..parallel import _worker_count
from .builder import BinaryBoundsBvh


comptime PARALLEL_LBVH_MIN_ITEMS = UInt32(4096)
comptime RADIX_LBVH_MIN_ITEMS = UInt32(16384)
comptime PARALLEL_RADIX_LBVH_MIN_ITEMS = UInt32(32768)
comptime PARALLEL_RADIX_LBVH_MIN_LOGICAL_CORES = 16
comptime PARALLEL_FRONTIER_DEPTH = 3
comptime PARALLEL_FRONTIER_CAPACITY = 1 << PARALLEL_FRONTIER_DEPTH


@fieldwise_init
struct MortonItem(Comparable, TrivialRegisterPassable):
    var code: UInt32
    var item_idx: UInt32

    def __lt__(self, rhs: Self) -> Bool:
        return self.code < rhs.code


@fieldwise_init
struct LbvhFrontierTask(TrivialRegisterPassable):
    var node_idx: UInt32
    var first: Int
    var count: Int


def _radix_sort_morton_pass[
    shift: Int
](src: ImmSpan[MortonItem, _], dst: MutSpan[MortonItem, _]):
    var counts = Array[UInt32, 256](fill=0)
    for item in src:
        var bucket = Int((item.code >> UInt32(shift)) & UInt32(0xFF))
        counts[bucket] += 1

    var offsets = Array[Int, 256](fill=0)
    var offset = 0
    for bucket in range(256):
        offsets[bucket] = offset
        offset += Int(counts[bucket])

    for item in src:
        var bucket = Int((item.code >> UInt32(shift)) & UInt32(0xFF))
        dst[offsets[bucket]] = item
        offsets[bucket] += 1


def _radix_sort_morton_pairs(mut pairs: List[MortonItem]):
    var scratch = List[MortonItem](capacity=len(pairs))
    scratch.resize(unsafe_uninit_length=len(pairs))
    _radix_sort_morton_pass[0](Span(pairs), Span(scratch))
    _radix_sort_morton_pass[8](Span(scratch), Span(pairs))
    _radix_sort_morton_pass[16](Span(pairs), Span(scratch))
    _radix_sort_morton_pass[24](Span(scratch), Span(pairs))


def _radix_sort_morton_pass_parallel[
    shift: Int
](
    src: ImmSpan[MortonItem, _],
    dst: MutSpan[MortonItem, _],
    mut counts: List[UInt32],
    mut offsets: List[Int],
    worker_count: Int,
):
    def histogram_worker(task_idx: Int) {imm, mut counts}:
        var counts_base = task_idx * 256
        for bucket in range(256):
            counts[counts_base + bucket] = 0

        var first = len(src) * task_idx // worker_count
        var end = len(src) * (task_idx + 1) // worker_count
        for i in range(first, end):
            var bucket = Int(
                (src.unsafe_get(i).code >> UInt32(shift)) & UInt32(0xFF)
            )
            counts[counts_base + bucket] += 1

    parallelize(histogram_worker, worker_count, worker_count)

    # Workers own contiguous input chunks. Assign each bucket's destinations
    # in worker order to retain the serial radix pass's stable ordering.
    var output_offset = 0
    for bucket in range(256):
        for worker_idx in range(worker_count):
            var count_idx = worker_idx * 256 + bucket
            offsets[count_idx] = output_offset
            output_offset += Int(counts[count_idx])

    def scatter_worker(task_idx: Int) {imm}:
        var offsets_base = task_idx * 256
        var worker_offsets = Array[Int, 256](fill=0)
        for bucket in range(256):
            worker_offsets[bucket] = offsets[offsets_base + bucket]

        var first = len(src) * task_idx // worker_count
        var end = len(src) * (task_idx + 1) // worker_count
        for i in range(first, end):
            var item = src.unsafe_get(i)
            var bucket = Int((item.code >> UInt32(shift)) & UInt32(0xFF))
            dst[worker_offsets[bucket]] = item
            worker_offsets[bucket] += 1

    parallelize(scatter_worker, worker_count, worker_count)


def _radix_sort_morton_pairs_parallel(
    mut pairs: List[MortonItem], worker_count: Int
):
    var scratch = List[MortonItem](capacity=len(pairs))
    scratch.resize(unsafe_uninit_length=len(pairs))
    var counts = [UInt32(0) for _ in range(worker_count * 256)]
    var offsets = [Int(0) for _ in range(worker_count * 256)]
    _radix_sort_morton_pass_parallel[0](
        pairs, scratch, counts, offsets, worker_count
    )
    _radix_sort_morton_pass_parallel[8](
        scratch, pairs, counts, offsets, worker_count
    )
    _radix_sort_morton_pass_parallel[16](
        pairs, scratch, counts, offsets, worker_count
    )
    _radix_sort_morton_pass_parallel[24](
        scratch, pairs, counts, offsets, worker_count
    )


def _common_prefix(pairs: ImmSpan[MortonItem, _], i: Int, j: Int) -> Int:
    if j < 0 or j >= len(pairs):
        return -1
    var a = pairs.unsafe_get(i).code
    var b = pairs.unsafe_get(j).code
    return morton_common_prefix(a, UInt32(i), b, UInt32(j))


def _lbvh_find_split(
    pairs: ImmSpan[MortonItem, _],
    first: Int,
    last: Int,
) -> Int:
    debug_assert["safe", _use_compiler_assume=True](
        0 <= first <= last < len(pairs),
        "LBVH split range is invalid",
    )
    var node_prefix = _common_prefix(pairs, first, last)
    var split = first
    var step = last - first
    while step > 1:
        step = (step + 1) >> 1
        var new_split = split + step
        if new_split < last:
            var split_prefix = _common_prefix(pairs, first, new_split)
            if split_prefix > node_prefix:
                split = new_split
    return split


def _build_lbvh[
    frame: Frame,
    leaf_size: Int,
    method: String,
](
    mut builder: BinaryBoundsBvh[frame, leaf_size, method],
    precomputed_centroid_bounds: AABB[frame],
) where (method == "lbvh"):
    """Build a binary LBVH using sorted Morton codes over item centers."""
    debug_assert["safe", _use_compiler_assume=True](builder.item_count > 0)
    var item_count = Int(builder.item_count)
    var centroid_bounds = precomputed_centroid_bounds
    if centroid_bounds._min.x[0] > centroid_bounds._max.x[0]:
        for item in builder.items:
            centroid_bounds.grow(item.bounds.centroid())

    var extent = centroid_bounds.extent()
    var inv = extent.safe_inv()
    var pairs = List[MortonItem](capacity=item_count)
    var use_parallel_build = builder.item_count >= PARALLEL_LBVH_MIN_ITEMS
    if use_parallel_build:
        pairs.resize(unsafe_uninit_length=item_count)
        var worker_count = _worker_count(item_count)

        def morton_worker(task_idx: Int) {imm, mut pairs}:
            var first = item_count * task_idx // worker_count
            var end = item_count * (task_idx + 1) // worker_count
            for i in range(first, end):
                ref item = builder.items[i]
                var centroid = item.bounds.centroid()
                var c = (centroid - centroid_bounds._min) * inv
                pairs[i] = MortonItem(morton3(c.x, c.y, c.z), UInt32(i))

        parallelize(morton_worker, worker_count, worker_count)
    else:
        for i, item in enumerate(builder.items):
            var centroid = item.bounds.centroid()
            var c = (centroid - centroid_bounds._min) * inv
            pairs.append(MortonItem(morton3(c.x, c.y, c.z), UInt32(i)))
    if builder.item_count >= RADIX_LBVH_MIN_ITEMS:
        var radix_worker_count = _worker_count(item_count)
        if (
            builder.item_count >= PARALLEL_RADIX_LBVH_MIN_ITEMS
            and num_logical_cores() >= PARALLEL_RADIX_LBVH_MIN_LOGICAL_CORES
        ):
            _radix_sort_morton_pairs_parallel(pairs, radix_worker_count)
        else:
            _radix_sort_morton_pairs(pairs)
    else:
        sort(Span(pairs))
    if not use_parallel_build:
        for i in range(len(pairs)):
            builder.item_indices[i] = pairs[i].item_idx

    debug_assert["safe", _use_compiler_assume=True](
        len(pairs) == item_count,
        "LBVH Morton pair count does not match item count",
    )
    debug_assert["safe", _use_compiler_assume=True](
        len(builder.item_indices) == item_count
        and len(builder.items) == item_count,
        "LBVH builder arrays have inconsistent lengths",
    )
    if use_parallel_build:
        _build_lbvh_parallel(builder, Span(pairs))
    else:
        _ = _build_lbvh_recursive[frame, leaf_size, method](
            builder, Span(pairs), 0, 0, item_count
        )


@always_inline
def _build_lbvh_leaf[
    frame: Frame, leaf_size: Int, method: String
](
    mut builder: BinaryBoundsBvh[frame, leaf_size, method],
    node_idx: UInt32,
    first: Int,
    count: Int,
) -> AABB[frame] where (method == "lbvh"):
    ref leaf = builder.nodes[Int(node_idx)]
    leaf.set_leaf(UInt32(first), UInt32(count))
    leaf.aabb = AABB[frame].invalid()
    for i in range(count):
        var item_idx = Int(builder.item_indices[first + i])
        builder.items[item_idx].grow_into(leaf.aabb)
    return leaf.aabb


def _build_lbvh_recursive[
    frame: Frame, leaf_size: Int, method: String
](
    mut builder: BinaryBoundsBvh[frame, leaf_size, method],
    pairs: ImmSpan[MortonItem, _],
    node_idx: UInt32,
    first: Int,
    count: Int,
) -> AABB[frame] where (method == "lbvh"):
    debug_assert["safe", _use_compiler_assume=True](
        first >= 0
        and count > 0
        and first <= len(pairs)
        and count <= len(pairs) - first,
        "LBVH recursive range is outside Morton pairs",
    )
    debug_assert["safe", _use_compiler_assume=True](
        Int(node_idx) < len(builder.nodes),
        "LBVH node index is outside builder nodes",
    )
    if count <= leaf_size:
        return _build_lbvh_leaf(builder, node_idx, first, count)

    var last = first + count - 1
    var split = _lbvh_find_split(pairs, first, last)
    var left_count = split - first + 1
    var right_count = count - left_count
    var left_child_idx = builder.allocate_children()
    var left_bounds = _build_lbvh_recursive(
        builder,
        pairs,
        left_child_idx,
        first,
        left_count,
    )
    var right_bounds = _build_lbvh_recursive(
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


def _build_lbvh_parallel[
    frame: Frame, leaf_size: Int, method: String
](
    mut builder: BinaryBoundsBvh[frame, leaf_size, method],
    pairs: ImmSpan[MortonItem, _],
) where (method == "lbvh"):
    var max_nodes = Int(builder.item_count * 2 - 1)
    builder.nodes.resize(unsafe_uninit_length=max_nodes)
    var frontier = List[LbvhFrontierTask](capacity=PARALLEL_FRONTIER_CAPACITY)
    _collect_lbvh_frontier(
        builder,
        pairs,
        0,
        0,
        len(pairs),
        PARALLEL_FRONTIER_DEPTH,
        frontier,
    )

    var next_node = [builder.nodes_used]

    def worker(task_idx: Int) {imm, mut builder, mut next_node}:
        var task = frontier[task_idx]
        for i in range(task.first, task.first + task.count):
            builder.item_indices[i] = pairs[i].item_idx
        _ = _build_lbvh_recursive_parallel(
            builder,
            pairs,
            task.node_idx,
            task.first,
            task.count,
            next_node.unsafe_ptr(),
        )

    var task_count = len(frontier)
    if task_count > 0:
        parallelize(worker, task_count, _worker_count(task_count))

    builder.nodes_used = next_node[0]
    _ = _refit_lbvh_frontier(builder, 0, PARALLEL_FRONTIER_DEPTH)
    builder.nodes.shrink(Int(builder.nodes_used))


def _collect_lbvh_frontier[
    frame: Frame, leaf_size: Int, method: String
](
    mut builder: BinaryBoundsBvh[frame, leaf_size, method],
    pairs: ImmSpan[MortonItem, _],
    node_idx: UInt32,
    first: Int,
    count: Int,
    depth: Int,
    mut frontier: List[LbvhFrontierTask],
) where (method == "lbvh"):
    if count <= leaf_size or depth == 0:
        frontier.append(LbvhFrontierTask(node_idx, first, count))
        return

    var last = first + count - 1
    var split = _lbvh_find_split(pairs, first, last)
    var left_count = split - first + 1
    var left_child_idx = builder.allocate_children()
    builder.nodes[Int(node_idx)].set_internal(left_child_idx)
    _collect_lbvh_frontier(
        builder,
        pairs,
        left_child_idx,
        first,
        left_count,
        depth - 1,
        frontier,
    )
    _collect_lbvh_frontier(
        builder,
        pairs,
        left_child_idx + 1,
        split + 1,
        count - left_count,
        depth - 1,
        frontier,
    )


def _build_lbvh_recursive_parallel[
    frame: Frame,
    leaf_size: Int,
    method: String,
    next_node_origin: MutOrigin,
](
    mut builder: BinaryBoundsBvh[frame, leaf_size, method],
    pairs: ImmSpan[MortonItem, _],
    node_idx: UInt32,
    first: Int,
    count: Int,
    next_node: Pointer[UInt32, next_node_origin],
) -> AABB[frame] where (method == "lbvh"):
    if count <= leaf_size:
        return _build_lbvh_leaf(builder, node_idx, first, count)

    var last = first + count - 1
    var split = _lbvh_find_split(pairs, first, last)
    var left_count = split - first + 1
    var left_child_idx = Atomic.fetch_add[ordering=Ordering.RELAXED](
        next_node, UInt32(2)
    )
    var left_bounds = _build_lbvh_recursive_parallel(
        builder, pairs, left_child_idx, first, left_count, next_node
    )
    var right_bounds = _build_lbvh_recursive_parallel(
        builder,
        pairs,
        left_child_idx + 1,
        split + 1,
        count - left_count,
        next_node,
    )
    ref node = builder.nodes[Int(node_idx)]
    node.set_internal(left_child_idx)
    node.aabb = AABB.merge(left_bounds, right_bounds)
    return node.aabb


def _refit_lbvh_frontier[
    frame: Frame, leaf_size: Int, method: String
](
    mut builder: BinaryBoundsBvh[frame, leaf_size, method],
    node_idx: UInt32,
    depth: Int,
) -> AABB[frame] where (method == "lbvh"):
    var source_node = builder.nodes[Int(node_idx)]
    if depth == 0 or source_node.is_leaf():
        return source_node.aabb
    var left_idx = source_node.left_child()
    var left_bounds = _refit_lbvh_frontier(builder, left_idx, depth - 1)
    var right_bounds = _refit_lbvh_frontier(builder, left_idx + 1, depth - 1)
    ref node = builder.nodes[Int(node_idx)]
    node.aabb = AABB.merge(left_bounds, right_bounds)
    return node.aabb
