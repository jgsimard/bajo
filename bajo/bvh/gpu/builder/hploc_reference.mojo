from std.math import abs, max

from bajo.bvh.constants import LBVH_SENTINEL
from bajo.bvh.gpu.builder.lbvh import _lbvh_find_split
from bajo.bvh.gpu.builder.hploc_layout import (
    HPLOC_MERGING_THRESHOLD,
    HPLOC_SEARCH_RADIUS,
)
from bajo.core import AABB, Frame


@fieldwise_init
struct HplocReferenceNode(TrivialRegisterPassable, Writable):
    var bounds: AABB[Frame.WORLD]
    var parent: UInt32
    var left: UInt32
    var right: UInt32
    var leaf_id: UInt32

    def is_leaf(self) -> Bool:
        return self.left == LBVH_SENTINEL


@fieldwise_init
struct HplocReferenceStats(Copyable, Writable):
    var guide_nodes: Int
    var merge_calls: Int
    var merge_rounds: Int
    var hierarchical_rounds: Int
    var final_rounds: Int
    var max_cluster_count: Int


struct HplocReferenceBvh:
    var leaf_count: Int
    var root: UInt32
    var nodes: List[HplocReferenceNode]
    var stats: HplocReferenceStats

    def __init__(
        out self,
        leaf_count: Int,
        root: UInt32,
        var nodes: List[HplocReferenceNode],
        stats: HplocReferenceStats,
    ):
        self.leaf_count = leaf_count
        self.root = root
        self.nodes = nodes^
        self.stats = stats.copy()

    def root_bounds(self) -> AABB[Frame.WORLD]:
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
                if node.right != LBVH_SENTINEL:
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

            var merged = AABB[Frame.WORLD].merge(
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


def _bounds_difference(a: AABB[Frame.WORLD], b: AABB[Frame.WORLD]) -> Float64:
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


def _nearest_neighbors(
    nodes: List[HplocReferenceNode],
    clusters: List[UInt32],
    search_radius: Int,
) raises -> List[Int]:
    var count = len(clusters)
    var nearest = List[Int](length=count, fill=-1)

    for cluster_pos in range(count):
        var cluster_idx = clusters[cluster_pos]
        var best_area = Float32.MAX
        var best_neighbor = -1

        # This is the PLOC++ ordering used by H-PLOC: for each increasing
        # radius, test right first and left second. Strict comparison makes
        # that order the deterministic tie-breaker.
        for radius in range(1, search_radius + 1):
            var right = cluster_pos + radius
            if right < count:
                var bounds = AABB[Frame.WORLD].merge(
                    nodes[Int(cluster_idx)].bounds,
                    nodes[Int(clusters[right])].bounds,
                )
                var area = bounds.surface_area()[0]
                if area < best_area:
                    best_area = area
                    best_neighbor = right

            var left = cluster_pos - radius
            if left >= 0:
                var bounds = AABB[Frame.WORLD].merge(
                    nodes[Int(cluster_idx)].bounds,
                    nodes[Int(clusters[left])].bounds,
                )
                var area = bounds.surface_area()[0]
                if area < best_area:
                    best_area = area
                    best_neighbor = left

        if best_neighbor < 0:
            raise "H-PLOC reference cluster has no neighbor in search radius"
        nearest[cluster_pos] = best_neighbor

    return nearest^


def _append_merged_node(
    mut nodes: List[HplocReferenceNode],
    left: UInt32,
    right: UInt32,
) -> UInt32:
    var node_idx = UInt32(len(nodes))
    var bounds = AABB[Frame.WORLD].merge(
        nodes[Int(left)].bounds, nodes[Int(right)].bounds
    )
    nodes.append(
        HplocReferenceNode(
            bounds,
            LBVH_SENTINEL,
            left,
            right,
            LBVH_SENTINEL,
        )
    )
    nodes[Int(left)].parent = node_idx
    nodes[Int(right)].parent = node_idx
    return node_idx


def _merge_round(
    mut nodes: List[HplocReferenceNode],
    mut clusters: List[UInt32],
    search_radius: Int,
) raises -> Int:
    var nearest = _nearest_neighbors(nodes, clusters, search_radius)
    var compacted = List[UInt32](capacity=len(clusters))
    var merge_count = 0

    for cluster_pos in range(len(clusters)):
        var neighbor = nearest[cluster_pos]
        var mutual = nearest[neighbor] == cluster_pos
        if mutual and cluster_pos < neighbor:
            compacted.append(
                _append_merged_node(
                    nodes,
                    clusters[cluster_pos],
                    clusters[neighbor],
                )
            )
            merge_count += 1
        elif not mutual:
            compacted.append(clusters[cluster_pos])

    if merge_count == 0:
        raise "H-PLOC reference made no progress during mutual merging"
    clusters = compacted^
    return merge_count


def _reduce_clusters(
    mut nodes: List[HplocReferenceNode],
    mut clusters: List[UInt32],
    threshold: Int,
    search_radius: Int,
    final: Bool,
    mut stats: HplocReferenceStats,
) raises:
    if len(clusters) <= threshold:
        return

    stats.merge_calls += 1
    stats.max_cluster_count = max(stats.max_cluster_count, len(clusters))
    while len(clusters) > threshold:
        _ = _merge_round(nodes, clusters, search_radius)
        stats.merge_rounds += 1
        if final:
            stats.final_rounds += 1
        else:
            stats.hierarchical_rounds += 1


def _build_hploc_range(
    mut nodes: List[HplocReferenceNode],
    sorted_morton_codes: ImmSpan[UInt32, _],
    sorted_leaf_ids: ImmSpan[UInt32, _],
    first: Int,
    last: Int,
    merging_threshold: Int,
    search_radius: Int,
    mut stats: HplocReferenceStats,
) raises -> List[UInt32]:
    if first == last:
        var leaf = List[UInt32](capacity=1)
        leaf.append(sorted_leaf_ids.unsafe_get(first))
        return leaf^

    stats.guide_nodes += 1
    var split = _lbvh_find_split(sorted_morton_codes, first, last)
    var left = _build_hploc_range(
        nodes,
        sorted_morton_codes,
        sorted_leaf_ids,
        first,
        split,
        merging_threshold,
        search_radius,
        stats,
    )
    var right = _build_hploc_range(
        nodes,
        sorted_morton_codes,
        sorted_leaf_ids,
        split + 1,
        last,
        merging_threshold,
        search_radius,
        stats,
    )

    var clusters = List[UInt32](capacity=len(left) + len(right))
    for node_idx in left:
        clusters.append(node_idx)
    for node_idx in right:
        clusters.append(node_idx)

    var final = first == 0 and last == len(sorted_morton_codes) - 1
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


def build_hploc_reference(
    leaf_bounds: ImmSpan[AABB[Frame.WORLD], _],
    sorted_morton_codes: ImmSpan[UInt32, _],
    sorted_leaf_ids: ImmSpan[UInt32, _],
    search_radius: Int = HPLOC_SEARCH_RADIUS,
    merging_threshold: Int = HPLOC_MERGING_THRESHOLD,
) raises -> HplocReferenceBvh:
    """Build the paper's H-PLOC BVH2 deterministically on the host.

    Leaves occupy [0, N), inner nodes are appended in stable left-to-right
    merge order, and the sorted Morton hierarchy is used only as a range guide.
    """

    var leaf_count = len(leaf_bounds)
    if leaf_count <= 0:
        raise "H-PLOC reference requires at least one leaf"
    if (
        len(sorted_morton_codes) != leaf_count
        or len(sorted_leaf_ids) != leaf_count
    ):
        raise "H-PLOC reference input lengths do not match"
    if search_radius <= 0:
        raise "H-PLOC search radius must be positive"
    if merging_threshold <= 0:
        raise "H-PLOC merging threshold must be positive"

    var seen = List[Bool](length=leaf_count, fill=False)
    for sorted_pos in range(leaf_count):
        if sorted_pos > 0 and sorted_morton_codes.unsafe_get(
            sorted_pos - 1
        ) > sorted_morton_codes.unsafe_get(sorted_pos):
            raise "H-PLOC Morton codes must be sorted"
        var leaf_id = sorted_leaf_ids.unsafe_get(sorted_pos)
        if leaf_id >= UInt32(leaf_count) or seen[Int(leaf_id)]:
            raise "H-PLOC leaf IDs must be a permutation"
        seen[Int(leaf_id)] = True

    var nodes = List[HplocReferenceNode](capacity=leaf_count * 2 - 1)
    for leaf_id in range(leaf_count):
        nodes.append(
            HplocReferenceNode(
                leaf_bounds.unsafe_get(leaf_id),
                LBVH_SENTINEL,
                LBVH_SENTINEL,
                LBVH_SENTINEL,
                UInt32(leaf_id),
            )
        )

    var stats = HplocReferenceStats(0, 0, 0, 0, 0, 0)
    var roots = _build_hploc_range(
        nodes,
        sorted_morton_codes,
        sorted_leaf_ids,
        0,
        leaf_count - 1,
        merging_threshold,
        search_radius,
        stats,
    )
    if len(roots) != 1:
        raise "H-PLOC final reduction did not produce one root"
    return HplocReferenceBvh(leaf_count, roots[0], nodes^, stats)
