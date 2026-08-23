from bajo.bvh.constants import (
    BinaryBvhNode,
    EMPTY_LANE,
    WideNode,
)
from bajo.bvh.gpu.wide_layout import GpuWideBoundsBvh
from bajo.bvh.gpu.builder.binary_layout import GpuBinaryBoundsBvh
from bajo.bvh.wide_meta import (
    _wide_meta_count,
    _wide_meta_data,
    _wide_node_index,
)
from bajo.core import AABB, Frame


@fieldwise_init
struct GpuBvhQuality(Writable):
    """Host-side SAH baseline for GPU builder comparisons.

    quality = (sum internal-node areas + sum leaf area * primitive count)
              / root area

    This matches the existing CPU builder's normalized surface-area metric
    when the binary tree has one primitive per leaf.
    """

    var quality: Float64
    var internal_area_ratio: Float64
    var primitive_area_ratio: Float64
    var internal_nodes: Int
    var leaf_references: Int
    var primitives: Int


def _normalized_quality(
    root_area: Float64,
    internal_area: Float64,
    primitive_area: Float64,
    internal_nodes: Int,
    leaf_references: Int,
    primitives: Int,
) -> GpuBvhQuality:
    if root_area <= 0.0:
        return GpuBvhQuality(
            0.0,
            0.0,
            0.0,
            internal_nodes,
            leaf_references,
            primitives,
        )

    var internal_ratio = internal_area / root_area
    var primitive_ratio = primitive_area / root_area
    return GpuBvhQuality(
        internal_ratio + primitive_ratio,
        internal_ratio,
        primitive_ratio,
        internal_nodes,
        leaf_references,
        primitives,
    )


def measure_binary_bvh_quality(
    binary: GpuBinaryBoundsBvh,
) raises -> GpuBvhQuality:
    """Measure the one-primitive-per-leaf binary BVH after refit."""

    var root_area = Float64(binary.root_bounds().surface_area()[0])
    var internal_area = Float64(0.0)
    var primitive_area = Float64(0.0)

    if binary.internal_count > 0:
        with binary.node_bounds.map_to_host() as node_bounds:
            var bounds_span = Span(
                unsafe_ptr=node_bounds.unsafe_ptr(), length=len(node_bounds)
            )
            for node_idx in range(binary.internal_count):
                var base = node_idx * BinaryBvhNode.BOUNDS_STRIDE
                var left = AABB[.WORLD].load6(bounds_span, base)
                var right = AABB[.WORLD].load6(
                    bounds_span, base + AABB.STRIDE
                )
                internal_area += Float64(
                    AABB[.WORLD].merge(left, right).surface_area()[0]
                )

    with binary.leaf_bounds.map_to_host() as leaf_bounds:
        var leaf_span = Span(
            unsafe_ptr=leaf_bounds.unsafe_ptr(), length=len(leaf_bounds)
        )
        for leaf_idx in range(binary.leaf_count):
            primitive_area += Float64(
                AABB[.WORLD]
                .load6(leaf_span, leaf_idx * AABB.STRIDE)
                .surface_area()[0]
            )

    return _normalized_quality(
        root_area,
        internal_area,
        primitive_area,
        binary.internal_count,
        binary.leaf_count,
        binary.leaf_count,
    )


def measure_wide_bvh_quality[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    max_leaf_size: Int,
](
    tree: GpuWideBoundsBvh[node_width, leaf_width, max_leaf_size],
) raises -> GpuBvhQuality:
    """Measure reachable wide nodes and primitive-weighted leaf bounds."""

    var root_area = Float64(tree.root_bounds().surface_area()[0])
    var internal_area = Float64(0.0)
    var primitive_area = Float64(0.0)
    var internal_nodes = 0
    var leaf_references = 0
    var primitives = 0

    var pending = List[UInt32]()
    pending.append(tree.root_idx)
    var visited = List[Bool](length=tree.node_count, fill=False)
    var cursor = 0

    with tree.wide_nodes.map_to_host() as wide_nodes:
        var bounds_span = Span(
            unsafe_ptr=wide_nodes.unsafe_ptr(), length=len(wide_nodes)
        )
        var nodes_u32 = wide_nodes.unsafe_ptr().unsafe_bitcast[UInt32]()

        while cursor < len(pending):
            var node_idx = pending[cursor]
            cursor += 1
            debug_assert["safe", _use_compiler_assume=True](
                Int(node_idx) < tree.node_count,
                "wide quality traversal references an invalid node",
            )
            if visited[Int(node_idx)]:
                continue
            visited[Int(node_idx)] = True
            internal_nodes += 1

            var node_bounds = AABB[.WORLD].invalid()
            var live_lanes = 0
            comptime for lane in range(node_width):
                var meta = nodes_u32[
                    unsafe_offset=_wide_node_index[node_width](
                        node_idx, WideNode.META, lane
                    )
                ]
                var count = _wide_meta_count(meta)
                if count == EMPTY_LANE:
                    continue

                live_lanes += 1
                var child_bounds = AABB[.WORLD].invalid()
                child_bounds._min.x = bounds_span[
                    _wide_node_index[node_width](node_idx, WideNode.MIN_X, lane)
                ]
                child_bounds._min.y = bounds_span[
                    _wide_node_index[node_width](node_idx, WideNode.MIN_Y, lane)
                ]
                child_bounds._min.z = bounds_span[
                    _wide_node_index[node_width](node_idx, WideNode.MIN_Z, lane)
                ]
                child_bounds._max.x = bounds_span[
                    _wide_node_index[node_width](node_idx, WideNode.MAX_X, lane)
                ]
                child_bounds._max.y = bounds_span[
                    _wide_node_index[node_width](node_idx, WideNode.MAX_Y, lane)
                ]
                child_bounds._max.z = bounds_span[
                    _wide_node_index[node_width](node_idx, WideNode.MAX_Z, lane)
                ]
                node_bounds.grow(child_bounds)
                if count == 0:
                    pending.append(_wide_meta_data(meta))
                else:
                    leaf_references += 1
                    primitives += Int(count)
                    primitive_area += Float64(
                        child_bounds.surface_area()[0]
                    ) * Float64(count)

            debug_assert["safe", _use_compiler_assume=True](
                live_lanes > 0, "reachable wide node has no live lanes"
            )
            internal_area += Float64(node_bounds.surface_area()[0])

    return _normalized_quality(
        root_area,
        internal_area,
        primitive_area,
        internal_nodes,
        leaf_references,
        primitives,
    )
