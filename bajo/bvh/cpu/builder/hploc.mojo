from max.algorithm import parallelize
from std.atomic import Atomic, Ordering
from std.math import abs, max, min
from std.sys import simd_width_of

from bajo.bvh.constants import LBVH_SENTINEL
from bajo.core import AABB, Frame
from bajo.bvh.cpu.build_method import CpuBvhBuildMethod
from bajo.core.utils import fmax, fmin
from ..parallel import _worker_count
from .builder import BinaryBoundsBvh
from .lbvh import MortonItem, _lbvh_find_split, _sorted_morton_pairs


comptime HPLOC_SEARCH_RADIUS = 8
comptime HPLOC_MERGING_THRESHOLD = 16
comptime _HPLOC_CLUSTER_CAPACITY = HPLOC_MERGING_THRESHOLD * 2
comptime _HPLOC_SIMD_WIDTH = simd_width_of[DType.float32]()
comptime _HPLOC_BOUNDS_PAD = HPLOC_SEARCH_RADIUS
comptime _HPLOC_PADDED_BOUNDS_CAPACITY = (
    _HPLOC_CLUSTER_CAPACITY + 2 * _HPLOC_BOUNDS_PAD
)
comptime PARALLEL_HPLOC_MIN_ITEMS = 4096
comptime PARALLEL_HPLOC_FRONTIER_DEPTH = 4
comptime _PARALLEL_HPLOC_FRONTIER_CAPACITY = 1 << PARALLEL_HPLOC_FRONTIER_DEPTH
comptime PARALLEL_HPLOC_EMIT_FRONTIER_DEPTH = 4
comptime _PARALLEL_HPLOC_EMIT_FRONTIER_CAPACITY = (
    1 << PARALLEL_HPLOC_EMIT_FRONTIER_DEPTH
)


@fieldwise_init
struct _HplocFrontierTask(TrivialRegisterPassable):
    var first: Int
    var last: Int


@fieldwise_init
struct _HplocEmitTask(TrivialRegisterPassable):
    var topology_idx: UInt32
    var node_idx: UInt32
    var first_item: UInt32


@fieldwise_init
struct _HplocMicroleafRange(TrivialRegisterPassable):
    var first: Int
    var count: Int


def _collect_hploc_microleaf_ranges(
    pairs: ImmSpan[MortonItem, _],
    first: Int,
    last: Int,
    max_count: Int,
    mut ranges: List[_HplocMicroleafRange],
):
    var count = last - first + 1
    if count <= max_count:
        ranges.append(_HplocMicroleafRange(first, count))
        return
    var split = _lbvh_find_split(pairs, first, last)
    _collect_hploc_microleaf_ranges(pairs, first, split, max_count, ranges)
    _collect_hploc_microleaf_ranges(pairs, split + 1, last, max_count, ranges)


@fieldwise_init
struct _HplocBoundsPacket[width: SIMDLength](TrivialRegisterPassable):
    var min_x: SIMD[.float32, Self.width]
    var min_y: SIMD[.float32, Self.width]
    var min_z: SIMD[.float32, Self.width]
    var max_x: SIMD[.float32, Self.width]
    var max_y: SIMD[.float32, Self.width]
    var max_z: SIMD[.float32, Self.width]

    @always_inline
    def merged_area(self, rhs: Self) -> SIMD[.float32, Self.width]:
        var dx = fmax(self.max_x, rhs.max_x) - fmin(self.min_x, rhs.min_x)
        var dy = fmax(self.max_y, rhs.max_y) - fmin(self.min_y, rhs.min_y)
        var dz = fmax(self.max_z, rhs.max_z) - fmin(self.min_z, rhs.min_z)
        return 2.0 * (dx * dy + dx * dz + dy * dz)


struct _HplocClusterBounds(Copyable):
    """Padded SoA bounds cache for scalar-reference and CPU SIMD merging."""

    var min_x: Array[Float32, _HPLOC_PADDED_BOUNDS_CAPACITY]
    var min_y: Array[Float32, _HPLOC_PADDED_BOUNDS_CAPACITY]
    var min_z: Array[Float32, _HPLOC_PADDED_BOUNDS_CAPACITY]
    var max_x: Array[Float32, _HPLOC_PADDED_BOUNDS_CAPACITY]
    var max_y: Array[Float32, _HPLOC_PADDED_BOUNDS_CAPACITY]
    var max_z: Array[Float32, _HPLOC_PADDED_BOUNDS_CAPACITY]

    def __init__(out self):
        self.min_x = Array[Float32, _HPLOC_PADDED_BOUNDS_CAPACITY](
            uninitialized=True
        )
        self.min_y = Array[Float32, _HPLOC_PADDED_BOUNDS_CAPACITY](
            uninitialized=True
        )
        self.min_z = Array[Float32, _HPLOC_PADDED_BOUNDS_CAPACITY](
            uninitialized=True
        )
        self.max_x = Array[Float32, _HPLOC_PADDED_BOUNDS_CAPACITY](
            uninitialized=True
        )
        self.max_y = Array[Float32, _HPLOC_PADDED_BOUNDS_CAPACITY](
            uninitialized=True
        )
        self.max_z = Array[Float32, _HPLOC_PADDED_BOUNDS_CAPACITY](
            uninitialized=True
        )

    @always_inline
    def set[frame: Frame](mut self, pos: Int, bounds: AABB[frame]):
        var storage_pos = _HPLOC_BOUNDS_PAD + pos
        self.min_x[storage_pos] = bounds._min.x[0]
        self.min_y[storage_pos] = bounds._min.y[0]
        self.min_z[storage_pos] = bounds._min.z[0]
        self.max_x[storage_pos] = bounds._max.x[0]
        self.max_y[storage_pos] = bounds._max.y[0]
        self.max_z[storage_pos] = bounds._max.z[0]

    @always_inline
    def copy(mut self, dst: Int, src: Int):
        var dst_pos = _HPLOC_BOUNDS_PAD + dst
        var src_pos = _HPLOC_BOUNDS_PAD + src
        self.min_x[dst_pos] = self.min_x[src_pos]
        self.min_y[dst_pos] = self.min_y[src_pos]
        self.min_z[dst_pos] = self.min_z[src_pos]
        self.max_x[dst_pos] = self.max_x[src_pos]
        self.max_y[dst_pos] = self.max_y[src_pos]
        self.max_z[dst_pos] = self.max_z[src_pos]

    @always_inline
    def merged_area(self, a: Int, b: Int) -> Float32:
        var a_pos = _HPLOC_BOUNDS_PAD + a
        var b_pos = _HPLOC_BOUNDS_PAD + b
        var dx = max(self.max_x[a_pos], self.max_x[b_pos]) - min(
            self.min_x[a_pos], self.min_x[b_pos]
        )
        var dy = max(self.max_y[a_pos], self.max_y[b_pos]) - min(
            self.min_y[a_pos], self.min_y[b_pos]
        )
        var dz = max(self.max_z[a_pos], self.max_z[b_pos]) - min(
            self.min_z[a_pos], self.min_z[b_pos]
        )
        return 2.0 * (dx * dy + dx * dz + dy * dz)

    @always_inline
    def load[
        width: SIMDLength
    ](self, storage_pos: Int) -> _HplocBoundsPacket[width]:
        return _HplocBoundsPacket[width](
            self.min_x.unsafe_ptr()
            .unsafe_offset(storage_pos)
            .unsafe_load[width=width](),
            self.min_y.unsafe_ptr()
            .unsafe_offset(storage_pos)
            .unsafe_load[width=width](),
            self.min_z.unsafe_ptr()
            .unsafe_offset(storage_pos)
            .unsafe_load[width=width](),
            self.max_x.unsafe_ptr()
            .unsafe_offset(storage_pos)
            .unsafe_load[width=width](),
            self.max_y.unsafe_ptr()
            .unsafe_offset(storage_pos)
            .unsafe_load[width=width](),
            self.max_z.unsafe_ptr()
            .unsafe_offset(storage_pos)
            .unsafe_load[width=width](),
        )

    def initialize_padding(mut self, cluster_count: Int):
        for storage_pos in range(_HPLOC_BOUNDS_PAD):
            self._clear_storage(storage_pos)
        for storage_pos in range(
            _HPLOC_BOUNDS_PAD + cluster_count,
            _HPLOC_PADDED_BOUNDS_CAPACITY,
        ):
            self._clear_storage(storage_pos)

    def clear_positions(mut self, first: Int, last: Int):
        for pos in range(first, last):
            self._clear_storage(_HPLOC_BOUNDS_PAD + pos)

    @always_inline
    def _clear_storage(mut self, storage_pos: Int):
        self.min_x[storage_pos] = 0.0
        self.min_y[storage_pos] = 0.0
        self.min_z[storage_pos] = 0.0
        self.max_x[storage_pos] = 0.0
        self.max_y[storage_pos] = 0.0
        self.max_z[storage_pos] = 0.0


@fieldwise_init
struct HplocNode[frame: Frame](TrivialRegisterPassable, Writable):
    var bounds: AABB[Self.frame]
    var parent: UInt32
    var left: UInt32
    var right: UInt32
    var leaf_id: UInt32
    var leaf_count: UInt32

    def is_leaf(self) -> Bool:
        return self.left == LBVH_SENTINEL


@fieldwise_init
struct HplocStats(Copyable, Writable):
    var guide_nodes: Int
    var merge_calls: Int
    var merge_rounds: Int
    var hierarchical_rounds: Int
    var final_rounds: Int
    var max_cluster_count: Int


struct HplocTopology[frame: Frame]:
    var leaf_count: Int
    var root: UInt32
    var nodes: List[HplocNode[Self.frame]]
    var stats: HplocStats

    def __init__(
        out self,
        leaf_count: Int,
        root: UInt32,
        var nodes: List[HplocNode[Self.frame]],
        stats: HplocStats,
    ):
        self.leaf_count = leaf_count
        self.root = root
        self.nodes = nodes^
        self.stats = stats.copy()

    def root_bounds(self) -> AABB[Self.frame]:
        return self.nodes[Int(self.root)].bounds

    def quality(self) -> Float64:
        var root_area = Float64(self.root_bounds().surface_area()[0])
        if root_area <= 0.0:
            return 0.0

        var area = Float64(0.0)
        for node in self.nodes:
            area += Float64(node.bounds.surface_area()[0])
        return area / root_area

    def topology_checksum(self) -> UInt64:
        var checksum = UInt64(1469598103934665603)
        for node in self.nodes:
            checksum = (checksum ^ UInt64(node.parent)) * UInt64(1099511628211)
            checksum = (checksum ^ UInt64(node.left)) * UInt64(1099511628211)
            checksum = (checksum ^ UInt64(node.right)) * UInt64(1099511628211)
            checksum = (checksum ^ UInt64(node.leaf_id)) * UInt64(1099511628211)
        return checksum

    def validate(self, tolerance: Float64 = 1.0e-5) -> Bool:
        if self.leaf_count <= 0:
            return False
        if len(self.nodes) != self.leaf_count * 2 - 1:
            return False
        if self.root >= UInt32(len(self.nodes)):
            return False
        if self.nodes[Int(self.root)].parent != LBVH_SENTINEL:
            return False
        if self.nodes[Int(self.root)].leaf_count != UInt32(self.leaf_count):
            return False

        var visited = List[Bool](length=len(self.nodes), fill=False)
        var seen_leaves = List[Bool](length=self.leaf_count, fill=False)
        var pending = List[UInt32]()
        pending.append(self.root)
        var cursor = 0
        var visited_count = 0

        while cursor < len(pending):
            var node_idx = pending[cursor]
            cursor += 1
            if node_idx >= UInt32(len(self.nodes)):
                return False
            if visited[Int(node_idx)]:
                return False
            visited[Int(node_idx)] = True
            visited_count += 1

            var node = self.nodes[Int(node_idx)]
            if node.is_leaf():
                if node.right != LBVH_SENTINEL or node.leaf_count != 1:
                    return False
                if node.leaf_id >= UInt32(self.leaf_count):
                    return False
                if seen_leaves[Int(node.leaf_id)]:
                    return False
                seen_leaves[Int(node.leaf_id)] = True
                continue

            if node.leaf_id != LBVH_SENTINEL:
                return False
            if (
                node.left >= UInt32(len(self.nodes))
                or node.right >= UInt32(len(self.nodes))
                or node.left == node.right
            ):
                return False
            if (
                self.nodes[Int(node.left)].parent != node_idx
                or self.nodes[Int(node.right)].parent != node_idx
            ):
                return False
            if node.leaf_count != (
                self.nodes[Int(node.left)].leaf_count
                + self.nodes[Int(node.right)].leaf_count
            ):
                return False

            var merged = AABB[Self.frame].merge(
                self.nodes[Int(node.left)].bounds,
                self.nodes[Int(node.right)].bounds,
            )
            if _bounds_difference(node.bounds, merged) > tolerance:
                return False
            pending.append(node.left)
            pending.append(node.right)

        if visited_count != len(self.nodes):
            return False
        for seen in seen_leaves:
            if not seen:
                return False
        return True


def _bounds_difference[frame: Frame](a: AABB[frame], b: AABB[frame]) -> Float64:
    return max(
        max(
            max(
                abs(Float64(a._min.x - b._min.x)),
                abs(Float64(a._min.y - b._min.y)),
            ),
            max(
                abs(Float64(a._min.z - b._min.z)),
                abs(Float64(a._max.x - b._max.x)),
            ),
        ),
        max(
            abs(Float64(a._max.y - b._max.y)),
            abs(Float64(a._max.z - b._max.z)),
        ),
    )


def _nearest_neighbors_scalar(
    cluster_bounds: _HplocClusterBounds,
    cluster_count: Int,
    search_radius: Int,
) -> Array[Int, _HPLOC_CLUSTER_CAPACITY]:
    var nearest = Array[Int, _HPLOC_CLUSTER_CAPACITY](uninitialized=True)
    for cluster_pos in range(cluster_count):
        var best_area = Float32.MAX
        var best_neighbor = -1
        for radius in range(1, search_radius + 1):
            var right = cluster_pos + radius
            if right < cluster_count:
                var area = cluster_bounds.merged_area(cluster_pos, right)
                if area < best_area:
                    best_area = area
                    best_neighbor = right

            var left = cluster_pos - radius
            if left >= 0:
                var area = cluster_bounds.merged_area(cluster_pos, left)
                if area < best_area:
                    best_area = area
                    best_neighbor = left

        debug_assert["safe", _use_compiler_assume=True](
            best_neighbor >= 0,
            "H-PLOC cluster has no neighbor in the search radius",
        )
        nearest[cluster_pos] = best_neighbor

    return nearest^


def _nearest_neighbors_simd(
    mut cluster_bounds: _HplocClusterBounds,
    cluster_count: Int,
    search_radius: Int,
) -> Array[Int, _HPLOC_CLUSTER_CAPACITY]:
    """Evaluate one radius across a native SIMD packet of CPU clusters."""
    comptime assert _HPLOC_SIMD_WIDTH <= _HPLOC_CLUSTER_CAPACITY
    debug_assert["safe", _use_compiler_assume=True](
        search_radius <= HPLOC_SEARCH_RADIUS,
        "H-PLOC SIMD search radius exceeds its padded bounds cache",
    )
    var nearest = Array[Int, _HPLOC_CLUSTER_CAPACITY](uninitialized=True)

    for cluster_base in range(0, cluster_count, _HPLOC_SIMD_WIDTH):
        var positions = SIMD[.int32, _HPLOC_SIMD_WIDTH](0)
        comptime for lane in range(_HPLOC_SIMD_WIDTH):
            positions[lane] = Int32(cluster_base + lane)

        var active = positions.lt(Int32(cluster_count))
        var own = cluster_bounds.load[_HPLOC_SIMD_WIDTH](
            _HPLOC_BOUNDS_PAD + cluster_base
        )
        var best_areas = SIMD[.float32, _HPLOC_SIMD_WIDTH](Float32.MAX)
        var best_neighbors = SIMD[.int32, _HPLOC_SIMD_WIDTH](-1)

        for radius in range(1, search_radius + 1):
            var right_positions = positions + Int32(radius)
            var right = cluster_bounds.load[_HPLOC_SIMD_WIDTH](
                _HPLOC_BOUNDS_PAD + cluster_base + radius
            )
            var right_areas = own.merged_area(right)
            var take_right = (
                active
                & right_positions.lt(Int32(cluster_count))
                & right_areas.lt(best_areas)
            )
            best_areas = take_right.select(right_areas, best_areas)
            best_neighbors = take_right.select(right_positions, best_neighbors)

            var left_positions = positions - Int32(radius)
            var left = cluster_bounds.load[_HPLOC_SIMD_WIDTH](
                _HPLOC_BOUNDS_PAD + cluster_base - radius
            )
            var left_areas = own.merged_area(left)
            var take_left = (
                active & left_positions.ge(Int32(0)) & left_areas.lt(best_areas)
            )
            best_areas = take_left.select(left_areas, best_areas)
            best_neighbors = take_left.select(left_positions, best_neighbors)

        comptime for lane in range(_HPLOC_SIMD_WIDTH):
            if cluster_base + lane < cluster_count:
                var best_neighbor = Int(best_neighbors[lane])
                debug_assert["safe", _use_compiler_assume=True](
                    best_neighbor >= 0,
                    "H-PLOC SIMD cluster has no neighbor in the search radius",
                )
                nearest[cluster_base + lane] = best_neighbor

    return nearest^


def _write_merged_node[
    frame: Frame,
    build_metadata: Bool,
](
    mut nodes: List[HplocNode[frame]],
    node_idx: UInt32,
    left: UInt32,
    right: UInt32,
):
    var bounds = AABB[frame].merge(
        nodes[Int(left)].bounds, nodes[Int(right)].bounds
    )
    var leaf_count = nodes[Int(left)].leaf_count + nodes[Int(right)].leaf_count
    nodes[Int(node_idx)] = HplocNode[frame](
        bounds,
        LBVH_SENTINEL,
        left,
        right,
        LBVH_SENTINEL,
        leaf_count,
    )
    comptime if build_metadata:
        nodes[Int(left)].parent = node_idx
        nodes[Int(right)].parent = node_idx


def _merge_round[
    frame: Frame,
    build_metadata: Bool,
    parallel_build: Bool,
](
    mut nodes: List[HplocNode[frame]],
    cluster_indices: MutSpan[UInt32, _],
    first: Int,
    cluster_count: Int,
    mut cluster_bounds: _HplocClusterBounds,
    search_radius: Int,
    mut next_node: List[UInt32],
) -> Int:
    var nearest: Array[Int, _HPLOC_CLUSTER_CAPACITY]
    comptime if build_metadata:
        nearest = _nearest_neighbors_scalar(
            cluster_bounds, cluster_count, search_radius
        )
    else:
        nearest = _nearest_neighbors_simd(
            cluster_bounds, cluster_count, search_radius
        )
    var merge_count = 0
    for cluster_pos in range(cluster_count):
        var neighbor = nearest[cluster_pos]
        if nearest[neighbor] == cluster_pos and cluster_pos < neighbor:
            merge_count += 1

    # A symmetric nearest-neighbor graph should always contain a mutual pair.
    # Retain a deterministic adjacent fallback so a release build cannot spin
    # forever if malformed bounds violate that invariant.
    if merge_count == 0:
        debug_assert["safe"](
            False, "H-PLOC mutual-nearest merging made no progress"
        )
        var node_idx: UInt32
        comptime if parallel_build:
            node_idx = Atomic.fetch_add[ordering=Ordering.RELAXED](
                next_node.unsafe_ptr(), UInt32(1)
            )
        else:
            node_idx = next_node[0]
            next_node[0] += 1
        _write_merged_node[frame, build_metadata](
            nodes,
            node_idx,
            cluster_indices.unsafe_get(first),
            cluster_indices.unsafe_get(first + 1),
        )
        cluster_indices.unsafe_get(first) = node_idx
        cluster_bounds.set(0, nodes[Int(node_idx)].bounds)
        for i in range(2, cluster_count):
            cluster_indices.unsafe_get(
                first + i - 1
            ) = cluster_indices.unsafe_get(first + i)
            cluster_bounds.copy(i - 1, i)
        cluster_bounds.clear_positions(cluster_count - 1, cluster_count)
        return cluster_count - 1

    var allocation_base: UInt32
    comptime if parallel_build:
        allocation_base = Atomic.fetch_add[ordering=Ordering.RELAXED](
            next_node.unsafe_ptr(), UInt32(merge_count)
        )
    else:
        allocation_base = next_node[0]
        next_node[0] += UInt32(merge_count)

    var merge_rank = 0
    var compacted_count = 0

    for cluster_pos in range(cluster_count):
        var neighbor = nearest[cluster_pos]
        var mutual = nearest[neighbor] == cluster_pos
        if mutual and cluster_pos < neighbor:
            var node_idx = allocation_base + UInt32(merge_rank)
            merge_rank += 1
            _write_merged_node[frame, build_metadata](
                nodes,
                node_idx,
                cluster_indices.unsafe_get(first + cluster_pos),
                cluster_indices.unsafe_get(first + neighbor),
            )
            cluster_indices.unsafe_get(first + compacted_count) = node_idx
            cluster_bounds.set(compacted_count, nodes[Int(node_idx)].bounds)
            compacted_count += 1
        elif not mutual:
            var survivor_idx = cluster_indices.unsafe_get(first + cluster_pos)
            cluster_indices.unsafe_get(first + compacted_count) = survivor_idx
            cluster_bounds.copy(compacted_count, cluster_pos)
            compacted_count += 1

    cluster_bounds.clear_positions(compacted_count, cluster_count)
    return compacted_count


def _reduce_clusters[
    frame: Frame,
    build_metadata: Bool,
    parallel_build: Bool,
](
    mut nodes: List[HplocNode[frame]],
    cluster_indices: MutSpan[UInt32, _],
    first: Int,
    cluster_count: Int,
    threshold: Int,
    search_radius: Int,
    final: Bool,
    mut stats: HplocStats,
    mut next_node: List[UInt32],
) -> Int:
    if cluster_count <= threshold:
        return cluster_count

    comptime if build_metadata:
        stats.merge_calls += 1
        stats.max_cluster_count = max(stats.max_cluster_count, cluster_count)
    var cluster_bounds = _HplocClusterBounds()
    for cluster_pos in range(cluster_count):
        cluster_bounds.set(
            cluster_pos,
            nodes[Int(cluster_indices.unsafe_get(first + cluster_pos))].bounds,
        )
    comptime if not build_metadata:
        cluster_bounds.initialize_padding(cluster_count)
    var reduced_count = cluster_count
    while reduced_count > threshold:
        reduced_count = _merge_round[frame, build_metadata, parallel_build](
            nodes,
            cluster_indices,
            first,
            reduced_count,
            cluster_bounds,
            search_radius,
            next_node,
        )
        comptime if build_metadata:
            stats.merge_rounds += 1
            if final:
                stats.final_rounds += 1
            else:
                stats.hierarchical_rounds += 1
    return reduced_count


def _build_hploc_range[
    frame: Frame,
    build_metadata: Bool,
    parallel_build: Bool,
](
    mut nodes: List[HplocNode[frame]],
    pairs: ImmSpan[MortonItem, _],
    cluster_indices: MutSpan[UInt32, _],
    first: Int,
    last: Int,
    merging_threshold: Int,
    search_radius: Int,
    mut stats: HplocStats,
    mut next_node: List[UInt32],
) -> Int:
    if first == last:
        cluster_indices.unsafe_get(first) = pairs.unsafe_get(first).item_idx
        return 1

    comptime if build_metadata:
        stats.guide_nodes += 1
    var split = _lbvh_find_split(pairs, first, last)
    var left_count = _build_hploc_range[frame, build_metadata, parallel_build](
        nodes,
        pairs,
        cluster_indices,
        first,
        split,
        merging_threshold,
        search_radius,
        stats,
        next_node,
    )
    var right_count = _build_hploc_range[frame, build_metadata, parallel_build](
        nodes,
        pairs,
        cluster_indices,
        split + 1,
        last,
        merging_threshold,
        search_radius,
        stats,
        next_node,
    )

    # Match the GPU range workspace: each child writes a packed list at its
    # Morton-range start. Move only the right list to concatenate both children
    # at the parent range start; the left list is already in place.
    for i in range(right_count):
        cluster_indices.unsafe_get(
            first + left_count + i
        ) = cluster_indices.unsafe_get(split + 1 + i)

    var final = first == 0 and last == len(pairs) - 1
    var threshold = 1 if final else merging_threshold
    return _reduce_clusters[frame, build_metadata, parallel_build](
        nodes,
        cluster_indices,
        first,
        left_count + right_count,
        threshold,
        search_radius,
        final,
        stats,
        next_node,
    )


def _collect_hploc_frontier(
    pairs: ImmSpan[MortonItem, _],
    first: Int,
    last: Int,
    depth: Int,
    mut frontier: List[_HplocFrontierTask],
):
    if first == last or depth == 0:
        frontier.append(_HplocFrontierTask(first, last))
        return

    var split = _lbvh_find_split(pairs, first, last)
    _collect_hploc_frontier(pairs, first, split, depth - 1, frontier)
    _collect_hploc_frontier(pairs, split + 1, last, depth - 1, frontier)


def _collect_hploc_count_balanced_frontier(
    pairs: ImmSpan[MortonItem, _],
    first: Int,
    last: Int,
    target_count: Int,
    mut frontier: List[_HplocFrontierTask],
):
    """Cut the Morton guide tree when a task reaches the target item count."""
    if last - first + 1 <= target_count:
        frontier.append(_HplocFrontierTask(first, last))
        return
    var split = _lbvh_find_split(pairs, first, last)
    _collect_hploc_count_balanced_frontier(
        pairs, first, split, target_count, frontier
    )
    _collect_hploc_count_balanced_frontier(
        pairs, split + 1, last, target_count, frontier
    )


def _finish_hploc_frontier_ancestors[
    frame: Frame
](
    mut nodes: List[HplocNode[frame]],
    pairs: ImmSpan[MortonItem, _],
    cluster_indices: MutSpan[UInt32, _],
    first: Int,
    last: Int,
    depth: Int,
    frontier_counts: ImmSpan[Int, _],
    mut frontier_cursor: Int,
    mut stats: HplocStats,
    mut next_node: List[UInt32],
) -> Int:
    if first == last or depth == 0:
        var count = frontier_counts.unsafe_get(frontier_cursor)
        frontier_cursor += 1
        return count

    var split = _lbvh_find_split(pairs, first, last)
    var left_count = _finish_hploc_frontier_ancestors(
        nodes,
        pairs,
        cluster_indices,
        first,
        split,
        depth - 1,
        frontier_counts,
        frontier_cursor,
        stats,
        next_node,
    )
    var right_count = _finish_hploc_frontier_ancestors(
        nodes,
        pairs,
        cluster_indices,
        split + 1,
        last,
        depth - 1,
        frontier_counts,
        frontier_cursor,
        stats,
        next_node,
    )

    for i in range(right_count):
        cluster_indices.unsafe_get(
            first + left_count + i
        ) = cluster_indices.unsafe_get(split + 1 + i)

    var final = first == 0 and last == len(pairs) - 1
    var threshold = 1 if final else HPLOC_MERGING_THRESHOLD
    return _reduce_clusters[frame, False, False](
        nodes,
        cluster_indices,
        first,
        left_count + right_count,
        threshold,
        HPLOC_SEARCH_RADIUS,
        final,
        stats,
        next_node,
    )


def _finish_hploc_count_balanced_ancestors[
    frame: Frame
](
    mut nodes: List[HplocNode[frame]],
    pairs: ImmSpan[MortonItem, _],
    cluster_indices: MutSpan[UInt32, _],
    first: Int,
    last: Int,
    frontier: ImmSpan[_HplocFrontierTask, _],
    frontier_counts: ImmSpan[Int, _],
    mut frontier_cursor: Int,
    mut stats: HplocStats,
    mut next_node: List[UInt32],
) -> Int:
    ref next_task = frontier.unsafe_get(frontier_cursor)
    if next_task.first == first and next_task.last == last:
        var count = frontier_counts.unsafe_get(frontier_cursor)
        frontier_cursor += 1
        return count

    var split = _lbvh_find_split(pairs, first, last)
    var left_count = _finish_hploc_count_balanced_ancestors(
        nodes,
        pairs,
        cluster_indices,
        first,
        split,
        frontier,
        frontier_counts,
        frontier_cursor,
        stats,
        next_node,
    )
    var right_count = _finish_hploc_count_balanced_ancestors(
        nodes,
        pairs,
        cluster_indices,
        split + 1,
        last,
        frontier,
        frontier_counts,
        frontier_cursor,
        stats,
        next_node,
    )

    for i in range(right_count):
        cluster_indices.unsafe_get(
            first + left_count + i
        ) = cluster_indices.unsafe_get(split + 1 + i)

    var final = first == 0 and last == len(pairs) - 1
    var threshold = 1 if final else HPLOC_MERGING_THRESHOLD
    return _reduce_clusters[frame, False, False](
        nodes,
        cluster_indices,
        first,
        left_count + right_count,
        threshold,
        HPLOC_SEARCH_RADIUS,
        final,
        stats,
        next_node,
    )


def _finish_hploc_topology[
    frame: Frame,
    build_metadata: Bool,
    balance_tasks: Int = 0,
](
    var nodes: List[HplocNode[frame]],
    pairs: ImmSpan[MortonItem, _],
    search_radius: Int,
    merging_threshold: Int,
) -> HplocTopology[frame]:
    var leaf_count = len(pairs)
    var stats = HplocStats(0, 0, 0, 0, 0, 0)
    nodes.resize(unsafe_uninit_length=leaf_count * 2 - 1)
    var next_node: List[UInt32] = [UInt32(leaf_count)]
    var cluster_indices = List[UInt32](capacity=leaf_count)
    cluster_indices.resize(unsafe_uninit_length=leaf_count)
    var root_count: Int
    comptime if build_metadata:
        root_count = _build_hploc_range[frame, True, False](
            nodes,
            pairs,
            cluster_indices,
            0,
            leaf_count - 1,
            merging_threshold,
            search_radius,
            stats,
            next_node,
        )
    else:
        if leaf_count >= PARALLEL_HPLOC_MIN_ITEMS:
            var frontier = List[_HplocFrontierTask](
                capacity=(
                    _PARALLEL_HPLOC_FRONTIER_CAPACITY if balance_tasks
                    == 0 else balance_tasks * 2
                )
            )
            comptime if balance_tasks > 0:
                var target_count = (
                    leaf_count + balance_tasks - 1
                ) // balance_tasks
                _collect_hploc_count_balanced_frontier(
                    pairs, 0, leaf_count - 1, target_count, frontier
                )
            else:
                _collect_hploc_frontier(
                    pairs,
                    0,
                    leaf_count - 1,
                    PARALLEL_HPLOC_FRONTIER_DEPTH,
                    frontier,
                )
            var frontier_counts = List[Int](capacity=len(frontier))
            frontier_counts.resize(unsafe_uninit_length=len(frontier))

            def worker(
                task_idx: Int,
            ) {
                imm,
                mut nodes,
                mut cluster_indices,
                mut frontier_counts,
                mut next_node,
            }:
                var task = frontier[task_idx]
                var local_stats = HplocStats(0, 0, 0, 0, 0, 0)
                frontier_counts[task_idx] = _build_hploc_range[
                    frame, False, True
                ](
                    nodes,
                    pairs,
                    cluster_indices,
                    task.first,
                    task.last,
                    HPLOC_MERGING_THRESHOLD,
                    HPLOC_SEARCH_RADIUS,
                    local_stats,
                    next_node,
                )

            var task_count = len(frontier)
            parallelize(worker, task_count, _worker_count(task_count))
            var frontier_cursor = 0
            comptime if balance_tasks > 0:
                root_count = _finish_hploc_count_balanced_ancestors(
                    nodes,
                    pairs,
                    cluster_indices,
                    0,
                    leaf_count - 1,
                    frontier,
                    frontier_counts,
                    frontier_cursor,
                    stats,
                    next_node,
                )
            else:
                root_count = _finish_hploc_frontier_ancestors(
                    nodes,
                    pairs,
                    cluster_indices,
                    0,
                    leaf_count - 1,
                    PARALLEL_HPLOC_FRONTIER_DEPTH,
                    frontier_counts,
                    frontier_cursor,
                    stats,
                    next_node,
                )
        else:
            root_count = _build_hploc_range[frame, False, False](
                nodes,
                pairs,
                cluster_indices,
                0,
                leaf_count - 1,
                merging_threshold,
                search_radius,
                stats,
                next_node,
            )
    debug_assert["safe", _use_compiler_assume=True](
        root_count == 1,
        "H-PLOC final reduction did not produce one root",
    )
    debug_assert["safe", _use_compiler_assume=True](
        next_node[0] == UInt32(len(nodes)),
        "H-PLOC did not initialize every topology node",
    )
    return HplocTopology(leaf_count, cluster_indices[0], nodes^, stats)


def build_hploc_topology[
    frame: Frame
](
    leaf_bounds: ImmSpan[AABB[frame], _],
    sorted_morton_codes: ImmSpan[UInt32, _],
    sorted_leaf_ids: ImmSpan[UInt32, _],
    search_radius: Int = HPLOC_SEARCH_RADIUS,
    merging_threshold: Int = HPLOC_MERGING_THRESHOLD,
) -> HplocTopology[frame]:
    """Build the deterministic H-PLOC binary topology on the CPU."""
    var leaf_count = len(leaf_bounds)
    debug_assert["safe", _use_compiler_assume=True](
        leaf_count > 0,
        "H-PLOC requires at least one leaf",
    )
    debug_assert["safe", _use_compiler_assume=True](
        len(sorted_morton_codes) == leaf_count
        and len(sorted_leaf_ids) == leaf_count,
        "H-PLOC input lengths do not match",
    )
    debug_assert["safe", _use_compiler_assume=True](
        search_radius > 0
        and merging_threshold > 0
        and merging_threshold <= HPLOC_MERGING_THRESHOLD,
        "H-PLOC search parameters are outside the supported range",
    )

    var seen = List[Bool](length=leaf_count, fill=False)
    var pairs = List[MortonItem](capacity=leaf_count)
    for sorted_pos in range(leaf_count):
        var code = sorted_morton_codes.unsafe_get(sorted_pos)
        if sorted_pos > 0:
            debug_assert["safe", _use_compiler_assume=True](
                sorted_morton_codes.unsafe_get(sorted_pos - 1) <= code,
                "H-PLOC Morton codes must be sorted",
            )
        var leaf_id = sorted_leaf_ids.unsafe_get(sorted_pos)
        debug_assert["safe", _use_compiler_assume=True](
            leaf_id < UInt32(leaf_count) and not seen[Int(leaf_id)],
            "H-PLOC leaf IDs must be a permutation",
        )
        seen[Int(leaf_id)] = True
        pairs.append(MortonItem(code, leaf_id))

    var nodes = List[HplocNode[frame]](capacity=leaf_count * 2 - 1)
    for leaf_id in range(leaf_count):
        nodes.append(
            HplocNode[frame](
                leaf_bounds.unsafe_get(leaf_id),
                LBVH_SENTINEL,
                LBVH_SENTINEL,
                LBVH_SENTINEL,
                UInt32(leaf_id),
                UInt32(1),
            )
        )

    return _finish_hploc_topology[frame, True](
        nodes^,
        pairs,
        search_radius,
        merging_threshold,
    )


def _write_hploc_leaf_indices[
    frame: Frame,
    microleaf_size: Int,
](
    topology: HplocTopology[frame],
    sorted_pairs: ImmSpan[MortonItem, _],
    topology_idx: UInt32,
    item_indices: MutSpan[UInt32, _],
    mut cursor: UInt32,
):
    ref node = topology.nodes[Int(topology_idx)]
    if node.is_leaf():
        comptime if microleaf_size == 1:
            item_indices[Int(cursor)] = node.leaf_id
            cursor += 1
        else:
            for leaf_offset in range(Int(node.leaf_count)):
                item_indices[Int(cursor)] = sorted_pairs.unsafe_get(
                    Int(node.leaf_id) + leaf_offset
                ).item_idx
                cursor += 1
        return
    _write_hploc_leaf_indices[frame, microleaf_size](
        topology, sorted_pairs, node.left, item_indices, cursor
    )
    _write_hploc_leaf_indices[frame, microleaf_size](
        topology, sorted_pairs, node.right, item_indices, cursor
    )


def _emit_hploc_node[
    frame: Frame,
    leaf_size: Int,
    method: CpuBvhBuildMethod,
    microleaf_size: Int,
](
    mut builder: BinaryBoundsBvh[frame, leaf_size, method, microleaf_size],
    topology: HplocTopology[frame],
    sorted_pairs: ImmSpan[MortonItem, _],
    topology_idx: UInt32,
    node_idx: UInt32,
    mut item_cursor: UInt32,
):
    var source = topology.nodes[Int(topology_idx)]
    builder.nodes[Int(node_idx)].aabb = source.bounds

    if source.leaf_count <= UInt32(leaf_size):
        var first_item = item_cursor
        _write_hploc_leaf_indices[frame, microleaf_size](
            topology,
            sorted_pairs,
            topology_idx,
            builder.item_indices,
            item_cursor,
        )
        builder.nodes[Int(node_idx)].set_leaf(first_item, source.leaf_count)
        return

    var left_child = builder.allocate_children()
    builder.nodes[Int(node_idx)].set_internal(left_child)
    _emit_hploc_node(
        builder,
        topology,
        sorted_pairs,
        source.left,
        left_child,
        item_cursor,
    )
    _emit_hploc_node(
        builder,
        topology,
        sorted_pairs,
        source.right,
        left_child + 1,
        item_cursor,
    )


def _emit_hploc_node_parallel[
    frame: Frame,
    leaf_size: Int,
    method: CpuBvhBuildMethod,
    microleaf_size: Int,
](
    mut builder: BinaryBoundsBvh[frame, leaf_size, method, microleaf_size],
    topology: HplocTopology[frame],
    sorted_pairs: ImmSpan[MortonItem, _],
    topology_idx: UInt32,
    node_idx: UInt32,
    first_item: UInt32,
    mut next_node: List[UInt32],
):
    var source = topology.nodes[Int(topology_idx)]
    builder.nodes[Int(node_idx)].aabb = source.bounds

    if source.leaf_count <= UInt32(leaf_size):
        var item_cursor = first_item
        _write_hploc_leaf_indices[frame, microleaf_size](
            topology,
            sorted_pairs,
            topology_idx,
            builder.item_indices,
            item_cursor,
        )
        builder.nodes[Int(node_idx)].set_leaf(first_item, source.leaf_count)
        return

    var left_child = Atomic.fetch_add[ordering=Ordering.RELAXED](
        next_node.unsafe_ptr(), UInt32(2)
    )
    builder.nodes[Int(node_idx)].set_internal(left_child)
    var left_item_count = topology.nodes[Int(source.left)].leaf_count
    _emit_hploc_node_parallel(
        builder,
        topology,
        sorted_pairs,
        source.left,
        left_child,
        first_item,
        next_node,
    )
    _emit_hploc_node_parallel(
        builder,
        topology,
        sorted_pairs,
        source.right,
        left_child + 1,
        first_item + left_item_count,
        next_node,
    )


def _collect_hploc_emit_frontier[
    frame: Frame,
    leaf_size: Int,
    method: CpuBvhBuildMethod,
    microleaf_size: Int,
](
    mut builder: BinaryBoundsBvh[frame, leaf_size, method, microleaf_size],
    topology: HplocTopology[frame],
    sorted_pairs: ImmSpan[MortonItem, _],
    topology_idx: UInt32,
    node_idx: UInt32,
    first_item: UInt32,
    depth: Int,
    mut frontier: List[_HplocEmitTask],
):
    var source = topology.nodes[Int(topology_idx)]
    if source.leaf_count <= UInt32(leaf_size):
        builder.nodes[Int(node_idx)].aabb = source.bounds
        var item_cursor = first_item
        _write_hploc_leaf_indices[frame, microleaf_size](
            topology,
            sorted_pairs,
            topology_idx,
            builder.item_indices,
            item_cursor,
        )
        builder.nodes[Int(node_idx)].set_leaf(first_item, source.leaf_count)
        return

    if depth == 0:
        frontier.append(_HplocEmitTask(topology_idx, node_idx, first_item))
        return

    builder.nodes[Int(node_idx)].aabb = source.bounds
    var left_child = builder.allocate_children()
    builder.nodes[Int(node_idx)].set_internal(left_child)
    var left_item_count = topology.nodes[Int(source.left)].leaf_count
    _collect_hploc_emit_frontier(
        builder,
        topology,
        sorted_pairs,
        source.left,
        left_child,
        first_item,
        depth - 1,
        frontier,
    )
    _collect_hploc_emit_frontier(
        builder,
        topology,
        sorted_pairs,
        source.right,
        left_child + 1,
        first_item + left_item_count,
        depth - 1,
        frontier,
    )


def _build_hploc[
    frame: Frame,
    leaf_size: Int,
    method: CpuBvhBuildMethod,
    microleaf_size: Int,
    balance_tasks: Int,
](
    mut builder: BinaryBoundsBvh[frame, leaf_size, method, microleaf_size],
    precomputed_centroid_bounds: AABB[frame],
):
    """Build H-PLOC and emit the standard CPU binary BVH with fat leaves."""
    var pairs = _sorted_morton_pairs(builder, precomputed_centroid_bounds)
    var leaf_count = len(pairs)
    var topology: HplocTopology[frame]
    comptime if microleaf_size == 1:
        var topology_nodes = List[HplocNode[frame]](capacity=leaf_count * 2 - 1)
        for item_idx in range(leaf_count):
            topology_nodes.append(
                HplocNode[frame](
                    builder.items[item_idx].bounds,
                    LBVH_SENTINEL,
                    LBVH_SENTINEL,
                    LBVH_SENTINEL,
                    UInt32(item_idx),
                    UInt32(1),
                )
            )
        topology = _finish_hploc_topology[frame, False, balance_tasks](
            topology_nodes^,
            pairs,
            HPLOC_SEARCH_RADIUS,
            HPLOC_MERGING_THRESHOLD,
        )
    else:
        var microleaf_ranges = List[_HplocMicroleafRange](capacity=leaf_count)
        _collect_hploc_microleaf_ranges(
            pairs, 0, leaf_count - 1, microleaf_size, microleaf_ranges
        )
        var microleaf_count = len(microleaf_ranges)
        var topology_nodes = List[HplocNode[frame]](
            capacity=microleaf_count * 2 - 1
        )
        var guide_pairs = List[MortonItem](capacity=microleaf_count)
        for microleaf_idx in range(microleaf_count):
            var first = microleaf_ranges[microleaf_idx].first
            var count = microleaf_ranges[microleaf_idx].count
            var bounds = AABB[frame].invalid()
            for leaf_offset in range(count):
                var item_idx = Int(pairs[first + leaf_offset].item_idx)
                bounds.grow(builder.items[item_idx].bounds)
            topology_nodes.append(
                HplocNode[frame](
                    bounds,
                    LBVH_SENTINEL,
                    LBVH_SENTINEL,
                    LBVH_SENTINEL,
                    UInt32(first),
                    UInt32(count),
                )
            )
            var representative = first + count // 2
            guide_pairs.append(
                MortonItem(pairs[representative].code, UInt32(microleaf_idx))
            )
        topology = _finish_hploc_topology[frame, False, balance_tasks](
            topology_nodes^,
            guide_pairs,
            HPLOC_SEARCH_RADIUS,
            HPLOC_MERGING_THRESHOLD,
        )
    if leaf_count >= PARALLEL_HPLOC_MIN_ITEMS:
        var max_builder_nodes = leaf_count * 2 - 1
        builder.nodes.resize(unsafe_uninit_length=max_builder_nodes)
        var emit_frontier = List[_HplocEmitTask](
            capacity=_PARALLEL_HPLOC_EMIT_FRONTIER_CAPACITY
        )
        _collect_hploc_emit_frontier(
            builder,
            topology,
            pairs,
            topology.root,
            UInt32(0),
            UInt32(0),
            PARALLEL_HPLOC_EMIT_FRONTIER_DEPTH,
            emit_frontier,
        )
        var next_builder_node: List[UInt32] = [builder.nodes_used]

        def emit_worker(
            task_idx: Int,
        ) {imm, mut builder, mut next_builder_node}:
            var task = emit_frontier[task_idx]
            _emit_hploc_node_parallel(
                builder,
                topology,
                pairs,
                task.topology_idx,
                task.node_idx,
                task.first_item,
                next_builder_node,
            )

        var task_count = len(emit_frontier)
        parallelize(emit_worker, task_count, _worker_count(task_count))
        builder.nodes_used = next_builder_node[0]
        builder.nodes.shrink(Int(builder.nodes_used))
    else:
        var item_cursor = UInt32(0)
        _emit_hploc_node(
            builder,
            topology,
            pairs,
            topology.root,
            UInt32(0),
            item_cursor,
        )
        debug_assert["safe", _use_compiler_assume=True](
            item_cursor == builder.item_count,
            "H-PLOC emission did not write every item",
        )
