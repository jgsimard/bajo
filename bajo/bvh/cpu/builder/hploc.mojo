from std.math import abs, max

from bajo.bvh.constants import LBVH_SENTINEL
from bajo.core import AABB, Frame
from .builder import BinaryBoundsBvh
from .lbvh import MortonItem, _lbvh_find_split, _sorted_morton_pairs


comptime HPLOC_SEARCH_RADIUS = 8
comptime HPLOC_MERGING_THRESHOLD = 16
comptime _HPLOC_CLUSTER_CAPACITY = HPLOC_MERGING_THRESHOLD * 2


struct _HplocClusters(Copyable):
    """Fixed storage for two reduced H-PLOC guide children."""

    var values: Array[UInt32, _HPLOC_CLUSTER_CAPACITY]
    var count: Int

    def __init__(out self):
        self.values = Array[UInt32, _HPLOC_CLUSTER_CAPACITY](fill=0)
        self.count = 0

    @always_inline
    def append(mut self, value: UInt32):
        debug_assert["safe", _use_compiler_assume=True](
            self.count < _HPLOC_CLUSTER_CAPACITY,
            "H-PLOC cluster scratch overflow",
        )
        self.values[self.count] = value
        self.count += 1


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


def _nearest_neighbors[
    frame: Frame
](
    nodes: ImmSpan[HplocNode[frame], _],
    clusters: _HplocClusters,
    search_radius: Int,
) -> Array[Int, _HPLOC_CLUSTER_CAPACITY]:
    var nearest = Array[Int, _HPLOC_CLUSTER_CAPACITY](fill=-1)

    for cluster_pos in range(clusters.count):
        var cluster_idx = clusters.values[cluster_pos]
        var best_area = Float32.MAX
        var best_neighbor = -1

        # Match the paper/GPU ordering: test right before left at each radius,
        # and use strict comparison as the deterministic tie-breaker.
        for radius in range(1, search_radius + 1):
            var right = cluster_pos + radius
            if right < clusters.count:
                var bounds = AABB[frame].merge(
                    nodes[Int(cluster_idx)].bounds,
                    nodes[Int(clusters.values[right])].bounds,
                )
                var area = bounds.surface_area()[0]
                if area < best_area:
                    best_area = area
                    best_neighbor = right

            var left = cluster_pos - radius
            if left >= 0:
                var bounds = AABB[frame].merge(
                    nodes[Int(cluster_idx)].bounds,
                    nodes[Int(clusters.values[left])].bounds,
                )
                var area = bounds.surface_area()[0]
                if area < best_area:
                    best_area = area
                    best_neighbor = left

        debug_assert["safe", _use_compiler_assume=True](
            best_neighbor >= 0,
            "H-PLOC cluster has no neighbor in the search radius",
        )
        nearest[cluster_pos] = best_neighbor

    return nearest^


def _append_merged_node[
    frame: Frame
](mut nodes: List[HplocNode[frame]], left: UInt32, right: UInt32,) -> UInt32:
    var node_idx = UInt32(len(nodes))
    var bounds = AABB[frame].merge(
        nodes[Int(left)].bounds, nodes[Int(right)].bounds
    )
    var leaf_count = nodes[Int(left)].leaf_count + nodes[Int(right)].leaf_count
    nodes.append(
        HplocNode[frame](
            bounds,
            LBVH_SENTINEL,
            left,
            right,
            LBVH_SENTINEL,
            leaf_count,
        )
    )
    nodes[Int(left)].parent = node_idx
    nodes[Int(right)].parent = node_idx
    return node_idx


def _merge_round[
    frame: Frame
](
    mut nodes: List[HplocNode[frame]],
    mut clusters: _HplocClusters,
    search_radius: Int,
) -> Int:
    var nearest = _nearest_neighbors(nodes, clusters, search_radius)
    var compacted = _HplocClusters()
    var merge_count = 0

    for cluster_pos in range(clusters.count):
        var neighbor = nearest[cluster_pos]
        var mutual = nearest[neighbor] == cluster_pos
        if mutual and cluster_pos < neighbor:
            compacted.append(
                _append_merged_node(
                    nodes,
                    clusters.values[cluster_pos],
                    clusters.values[neighbor],
                )
            )
            merge_count += 1
        elif not mutual:
            compacted.append(clusters.values[cluster_pos])

    # A symmetric nearest-neighbor graph should always contain a mutual pair.
    # Retain a deterministic adjacent fallback so a release build cannot spin
    # forever if malformed bounds violate that invariant.
    if merge_count == 0:
        debug_assert["safe"](
            False, "H-PLOC mutual-nearest merging made no progress"
        )
        compacted = _HplocClusters()
        compacted.append(
            _append_merged_node(nodes, clusters.values[0], clusters.values[1])
        )
        for i in range(2, clusters.count):
            compacted.append(clusters.values[i])
        merge_count = 1

    clusters = compacted^
    return merge_count


def _reduce_clusters[
    frame: Frame
](
    mut nodes: List[HplocNode[frame]],
    mut clusters: _HplocClusters,
    threshold: Int,
    search_radius: Int,
    final: Bool,
    mut stats: HplocStats,
):
    if clusters.count <= threshold:
        return

    stats.merge_calls += 1
    stats.max_cluster_count = max(stats.max_cluster_count, clusters.count)
    while clusters.count > threshold:
        _ = _merge_round(nodes, clusters, search_radius)
        stats.merge_rounds += 1
        if final:
            stats.final_rounds += 1
        else:
            stats.hierarchical_rounds += 1


def _build_hploc_range[
    frame: Frame
](
    mut nodes: List[HplocNode[frame]],
    pairs: ImmSpan[MortonItem, _],
    first: Int,
    last: Int,
    merging_threshold: Int,
    search_radius: Int,
    mut stats: HplocStats,
) -> _HplocClusters:
    if first == last:
        var leaf = _HplocClusters()
        leaf.append(pairs.unsafe_get(first).item_idx)
        return leaf^

    stats.guide_nodes += 1
    var split = _lbvh_find_split(pairs, first, last)
    var left = _build_hploc_range(
        nodes,
        pairs,
        first,
        split,
        merging_threshold,
        search_radius,
        stats,
    )
    var right = _build_hploc_range(
        nodes,
        pairs,
        split + 1,
        last,
        merging_threshold,
        search_radius,
        stats,
    )

    var clusters = _HplocClusters()
    for i in range(left.count):
        clusters.append(left.values[i])
    for i in range(right.count):
        clusters.append(right.values[i])

    var final = first == 0 and last == len(pairs) - 1
    var threshold = 1 if final else merging_threshold
    _reduce_clusters(
        nodes,
        clusters,
        threshold,
        search_radius,
        final,
        stats,
    )
    return clusters^


def _finish_hploc_topology[
    frame: Frame
](
    var nodes: List[HplocNode[frame]],
    pairs: ImmSpan[MortonItem, _],
    search_radius: Int,
    merging_threshold: Int,
) -> HplocTopology[frame]:
    var leaf_count = len(pairs)
    var stats = HplocStats(0, 0, 0, 0, 0, 0)
    var roots = _build_hploc_range(
        nodes,
        pairs,
        0,
        leaf_count - 1,
        merging_threshold,
        search_radius,
        stats,
    )
    debug_assert["safe", _use_compiler_assume=True](
        roots.count == 1,
        "H-PLOC final reduction did not produce one root",
    )
    return HplocTopology(leaf_count, roots.values[0], nodes^, stats)


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

    return _finish_hploc_topology(
        nodes^,
        pairs,
        search_radius,
        merging_threshold,
    )


def _write_hploc_leaf_indices[
    frame: Frame
](
    topology: HplocTopology[frame],
    topology_idx: UInt32,
    item_indices: MutSpan[UInt32, _],
    mut cursor: UInt32,
):
    ref node = topology.nodes[Int(topology_idx)]
    if node.is_leaf():
        item_indices[Int(cursor)] = node.leaf_id
        cursor += 1
        return
    _write_hploc_leaf_indices(topology, node.left, item_indices, cursor)
    _write_hploc_leaf_indices(topology, node.right, item_indices, cursor)


def _emit_hploc_node[
    frame: Frame,
    leaf_size: Int,
    method: String,
](
    mut builder: BinaryBoundsBvh[frame, leaf_size, method],
    topology: HplocTopology[frame],
    topology_idx: UInt32,
    node_idx: UInt32,
    mut item_cursor: UInt32,
) where (method == "hploc"):
    var source = topology.nodes[Int(topology_idx)]
    builder.nodes[Int(node_idx)].aabb = source.bounds

    if source.leaf_count <= UInt32(leaf_size):
        var first_item = item_cursor
        _write_hploc_leaf_indices(
            topology,
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
        source.left,
        left_child,
        item_cursor,
    )
    _emit_hploc_node(
        builder,
        topology,
        source.right,
        left_child + 1,
        item_cursor,
    )


def _build_hploc[
    frame: Frame,
    leaf_size: Int,
    method: String,
](
    mut builder: BinaryBoundsBvh[frame, leaf_size, method],
    precomputed_centroid_bounds: AABB[frame],
) where (method == "hploc"):
    """Build H-PLOC and emit the standard CPU binary BVH with fat leaves."""
    var pairs = _sorted_morton_pairs(builder, precomputed_centroid_bounds)
    var leaf_count = len(pairs)
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

    var topology = _finish_hploc_topology(
        topology_nodes^,
        pairs,
        HPLOC_SEARCH_RADIUS,
        HPLOC_MERGING_THRESHOLD,
    )
    var item_cursor = UInt32(0)
    _emit_hploc_node(
        builder,
        topology,
        topology.root,
        UInt32(0),
        item_cursor,
    )
    debug_assert["safe", _use_compiler_assume=True](
        item_cursor == builder.item_count,
        "H-PLOC emission did not write every item",
    )
