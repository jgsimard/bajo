"""Recursive sibling-pair versus indexed-linear LBVH topology benchmark."""

from max.algorithm import parallelize
from std.benchmark import keep
from std.math import round
from std.time import perf_counter_ns

from bajo.benchmark.bvh_fixtures import make_grid_triangles
from bajo.bvh.cpu.builder.lbvh import MortonItem
from bajo.bvh.cpu.parallel import _worker_count
from bajo.core import AABB, Point3f32
from bajo.core.morton import morton3, morton_common_prefix
from bajo.core.utils import ns_to_ms
from bajo.parser.obj.pack import pack_obj_triangles


comptime OBJ_PATH = "./assets/dragon/dragon.obj"
comptime REPEATS = 7


def _median(values: List[Int]) -> Int:
    var ordered = values.copy()
    sort(ordered)
    return ordered[(len(ordered) - 1) >> 1]


@always_inline
def _prefix(pairs: List[MortonItem], i: Int, j: Int) -> Int:
    if j < 0 or j >= len(pairs):
        return -1
    var a = pairs[i].code
    var b = pairs[j].code
    return morton_common_prefix(a, UInt32(i), b, UInt32(j))


def _find_split(pairs: List[MortonItem], first: Int, last: Int) -> Int:
    var node_prefix = _prefix(pairs, first, last)
    var split = first
    var step = last - first
    while step > 1:
        step = (step + 1) >> 1
        var candidate = split + step
        if candidate < last and _prefix(pairs, first, candidate) > node_prefix:
            split = candidate
    return split


def _build_recursive(
    pairs: List[MortonItem],
    mut first_or_left: List[Int],
    mut right_or_count: List[Int],
    node_idx: Int,
    first: Int,
    last: Int,
    mut next_node: Array[Int, 1],
):
    if first == last:
        first_or_left[node_idx] = first
        right_or_count[node_idx] = 1
        return

    var split = _find_split(pairs, first, last)
    var left = next_node[0]
    next_node[0] += 2
    first_or_left[node_idx] = left
    right_or_count[node_idx] = 0
    _build_recursive(
        pairs, first_or_left, right_or_count, left, first, split, next_node
    )
    _build_recursive(
        pairs,
        first_or_left,
        right_or_count,
        left + 1,
        split + 1,
        last,
        next_node,
    )


def _build_indexed_parallel(
    pairs: List[MortonItem],
    mut left: List[Int],
    mut right: List[Int],
):
    var leaf_base = len(pairs) - 1
    var internal_count = len(pairs) - 1
    var worker_count = _worker_count(internal_count)

    def worker(task_idx: Int) {imm, mut left, mut right}:
        var begin = internal_count * task_idx // worker_count
        var end = internal_count * (task_idx + 1) // worker_count
        for i in range(begin, end):
            var direction = 1
            if _prefix(pairs, i, i + 1) < _prefix(pairs, i, i - 1):
                direction = -1
            var minimum_prefix = _prefix(pairs, i, i - direction)

            var maximum_length = 2
            while (
                _prefix(pairs, i, i + maximum_length * direction)
                > minimum_prefix
            ):
                maximum_length *= 2

            var length = 0
            var step = maximum_length >> 1
            while step > 0:
                if (
                    _prefix(pairs, i, i + (length + step) * direction)
                    > minimum_prefix
                ):
                    length += step
                step >>= 1

            var other = i + length * direction
            var first = min(i, other)
            var last = max(i, other)
            var split = _find_split(pairs, first, last)
            if split == first:
                left[i] = leaf_base + split
            else:
                left[i] = split
            if split + 1 == last:
                right[i] = leaf_base + split + 1
            else:
                right[i] = split + 1

    parallelize(worker, worker_count, worker_count)


def _remap_indexed(
    left: List[Int],
    right: List[Int],
    indexed_node: Int,
    indexed_leaf_base: Int,
    mut first_or_left: List[Int],
    mut right_or_count: List[Int],
    output_node: Int,
    mut next_output: Array[Int, 1],
):
    if indexed_node >= indexed_leaf_base:
        first_or_left[output_node] = indexed_node - indexed_leaf_base
        right_or_count[output_node] = 1
        return

    var output_left = next_output[0]
    next_output[0] += 2
    first_or_left[output_node] = output_left
    right_or_count[output_node] = 0
    _remap_indexed(
        left,
        right,
        left[indexed_node],
        indexed_leaf_base,
        first_or_left,
        right_or_count,
        output_left,
        next_output,
    )
    _remap_indexed(
        left,
        right,
        right[indexed_node],
        indexed_leaf_base,
        first_or_left,
        right_or_count,
        output_left + 1,
        next_output,
    )


def _make_pairs(vertices: List[Point3f32[.WORLD]]) -> List[MortonItem]:
    var count = len(vertices) // 3
    var centroids = List[Point3f32[.WORLD]](capacity=count)
    var centroid_bounds = AABB[.WORLD].invalid()
    for i in range(count):
        var bounds = AABB(
            vertices[i * 3], vertices[i * 3 + 1], vertices[i * 3 + 2]
        )
        var centroid = bounds.centroid()
        centroids.append(centroid)
        centroid_bounds.grow(centroid)
    var inverse_extent = centroid_bounds.extent().safe_inv()
    var pairs = List[MortonItem](capacity=count)
    for i in range(count):
        var c = (centroids[i] - centroid_bounds._min) * inverse_extent
        pairs.append(MortonItem(morton3(c.x, c.y, c.z), UInt32(i)))
    sort(pairs)
    return pairs^


def _bench(label: String, vertices: List[Point3f32[.WORLD]]):
    var pairs = _make_pairs(vertices)
    var node_count = len(pairs) * 2 - 1
    var recursive_times = List[Int](capacity=REPEATS)
    var indexed_times = List[Int](capacity=REPEATS)
    var indexed_only_times = List[Int](capacity=REPEATS)
    var recursive_checksum = 0
    var indexed_checksum = 0

    for _ in range(REPEATS):
        var first_or_left = List[Int](length=node_count, fill=-1)
        var right_or_count = List[Int](length=node_count, fill=-1)
        var next_node = [1]
        var start = perf_counter_ns()
        _build_recursive(
            pairs,
            first_or_left,
            right_or_count,
            0,
            0,
            len(pairs) - 1,
            next_node,
        )
        recursive_times.append(Int(perf_counter_ns() - start))
        recursive_checksum = next_node[0]

        var indexed_left = List[Int](length=len(pairs) - 1, fill=-1)
        var indexed_right = List[Int](length=len(pairs) - 1, fill=-1)
        start = perf_counter_ns()
        _build_indexed_parallel(pairs, indexed_left, indexed_right)
        indexed_only_times.append(Int(perf_counter_ns() - start))
        var indexed_first_or_left = List[Int](length=node_count, fill=-1)
        var indexed_right_or_count = List[Int](length=node_count, fill=-1)
        var next_output = [1]
        _remap_indexed(
            indexed_left,
            indexed_right,
            0,
            len(pairs) - 1,
            indexed_first_or_left,
            indexed_right_or_count,
            0,
            next_output,
        )
        indexed_times.append(Int(perf_counter_ns() - start))
        indexed_checksum = next_output[0]

    keep(recursive_checksum)
    keep(indexed_checksum)
    print(
        t"{label}\t{len(pairs)}\t"
        t"{round(ns_to_ms(_median(recursive_times)), 3)}\t"
        t"{round(ns_to_ms(_median(indexed_only_times)), 3)}\t"
        t"{round(ns_to_ms(_median(indexed_times)), 3)}\t"
        t"{recursive_checksum}\t{indexed_checksum}"
    )


def main() raises:
    print("LBVH leaf1 topology benchmark; median of 7")
    print(
        "Scene\tLeaves\tRecursive sibling-pair ms\tIndexed-only ms\t"
        "Indexed + sibling remap ms\tRecursive nodes\tIndexed nodes"
    )
    _bench("grid", make_grid_triangles())
    _bench("dragon", pack_obj_triangles[.WORLD](OBJ_PATH))
