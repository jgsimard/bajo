from std.math import abs, max, min
from std.sys import has_accelerator
from std.testing import TestSuite, assert_equal, assert_false, assert_true
from max.gpu.host import DeviceContext

from bajo.bvh.constants import (
    EMPTY_LANE,
    TRACE,
    WideNode,
    f32_max,
)
from bajo.bvh.cpu.blas_set import (
    build_triangle_blases,
    trace_blas_set,
)
from bajo.bvh.cpu import CpuBvhBuildMethod
from bajo.bvh.gpu.diagnostics import build_bounds_bvh_for_diagnostics
from bajo.bvh.gpu.wide_layout import (
    GpuWideBoundsBvh,
    GpuWideBoundsBvhBatch,
)
from bajo.bvh.gpu.builder import GpuBvhBuildMethod, build_binary_bvh
from bajo.bvh.gpu.builder.binary_layout import (
    GpuBinaryBoundsBvh,
    GpuBinaryBuildWorkspace,
    _node_left,
    _node_right,
)
from bajo.bvh.gpu.builder.wide_collapse import collapse_binary_to_wide_batch
from bajo.bvh.gpu.builder.binary_layout import _encoded_bounds
from bajo.bvh.tagged_ref import is_leaf_ref
from bajo.bvh.gpu.triangle_bvh import build_triangle_bvh
from bajo.bvh.gpu.utils import upload_list, upload_vertices
from bajo.bvh.wide_meta import (
    _wide_meta_count,
    _wide_meta_data,
    _wide_node_index,
)
from bajo.bvh.types import Hit
from bajo.core import AABB, Frame, Point3f32, SegmentOffsets

from test.bvh.fixtures import (
    _append_tri,
    _make_camera_rays_and_params,
    _make_duplicate_centroid_scene,
    _make_single_triangle_scene,
    _make_small_scene,
)


def _bounds_match(
    actual: AABB[.WORLD], expected: AABB[.WORLD]
) -> Bool:
    return (
        max(
            max(
                max(
                    abs(Float64(actual._min.x - expected._min.x)),
                    abs(Float64(actual._min.y - expected._min.y)),
                ),
                max(
                    abs(Float64(actual._min.z - expected._min.z)),
                    abs(Float64(actual._max.x - expected._max.x)),
                ),
            ),
            max(
                abs(Float64(actual._max.y - expected._max.y)),
                abs(Float64(actual._max.z - expected._max.z)),
            ),
        )
        <= 1.0e-4
    )


def _load_wide_lane_bounds[
    width: SIMDLength
](nodes: Span[Float32, _], node_idx: UInt32, lane: Int) -> AABB[.WORLD]:
    var bounds = AABB[.WORLD].invalid()
    bounds._min.x = nodes[
        _wide_node_index[width](node_idx, WideNode.MIN_X, lane)
    ]
    bounds._min.y = nodes[
        _wide_node_index[width](node_idx, WideNode.MIN_Y, lane)
    ]
    bounds._min.z = nodes[
        _wide_node_index[width](node_idx, WideNode.MIN_Z, lane)
    ]
    bounds._max.x = nodes[
        _wide_node_index[width](node_idx, WideNode.MAX_X, lane)
    ]
    bounds._max.y = nodes[
        _wide_node_index[width](node_idx, WideNode.MAX_Y, lane)
    ]
    bounds._max.z = nodes[
        _wide_node_index[width](node_idx, WideNode.MAX_Z, lane)
    ]
    return bounds


def _make_irregular_scene(
    count: Int,
) -> List[Point3f32[.WORLD]]:
    var verts = List[Point3f32[.WORLD]](capacity=count * 3)
    for i in range(count):
        var x = Float32((i % 19) * 4 - 36)
        var y = Float32(((i / 19) % 14) * 4 - 26)
        var z = Float32((i * 7) % 11 + 2)
        var scale = Float32(2.5) if i % 17 == 0 else Float32(0.7)
        verts.append(Point3f32[.WORLD](x - scale, y - scale, z))
        verts.append(Point3f32[.WORLD](x + scale, y - scale, z))
        verts.append(Point3f32[.WORLD](x, y + scale, z + 0.1))
    return verts^


def _triangle_bounds(verts: List[Point3f32[.WORLD]]) -> AABB[.WORLD]:
    var bounds = AABB[.WORLD].invalid()
    for vertex in verts:
        bounds.grow(vertex)
    return bounds


def _flatten_triangle_bounds(
    verts: List[Point3f32[.WORLD]],
) -> Tuple[List[Float32], List[UInt32]]:
    var tri_count = len(verts) / 3
    var flat = List[Float32](capacity=tri_count * AABB.STRIDE)
    var payloads = List[UInt32](capacity=tri_count)
    for i in range(tri_count):
        var bounds = AABB(verts[i * 3], verts[i * 3 + 1], verts[i * 3 + 2])
        flat.append(bounds._min.x)
        flat.append(bounds._min.y)
        flat.append(bounds._min.z)
        flat.append(bounds._max.x)
        flat.append(bounds._max.y)
        flat.append(bounds._max.z)
        payloads.append(UInt32(i))
    return (flat^, payloads^)


def _assert_literature_wide_invariants[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    max_leaf_size: Int,
](
    tree: GpuWideBoundsBvh[node_width, leaf_width, max_leaf_size],
    verts: List[Point3f32[.WORLD]],
) raises:
    var tri_count = len(verts) / 3
    comptime fat_leaf_limit = min(max_leaf_size, 4)
    assert_true(tree.leaf_block_count > 0)
    assert_true(tree.leaf_block_count <= tri_count)
    assert_true(tree.node_count > 0)

    var triangle_bounds_list = List[AABB[.WORLD]](capacity=tri_count)
    for i in range(tri_count):
        triangle_bounds_list.append(
            AABB(verts[i * 3], verts[i * 3 + 1], verts[i * 3 + 2])
        )

    with tree.wide_nodes.map_to_host() as wide_nodes, tree.leaf_block_indices.map_to_host() as leaf_blocks:
        var nodes_span = Span(
            unsafe_ptr=wide_nodes.unsafe_ptr(), length=len(wide_nodes)
        )
        var nodes_u32 = wide_nodes.unsafe_ptr().unsafe_bitcast[UInt32]()
        var node_unions = List[AABB[.WORLD]](
            length=tree.node_count, fill=AABB[.WORLD].invalid()
        )

        for node_idx in range(tree.node_count):
            var node_union = AABB[.WORLD].invalid()
            var live_lanes = 0
            comptime for lane in range(node_width):
                var meta = nodes_u32[
                    unsafe_offset=_wide_node_index[node_width](
                        UInt32(node_idx), WideNode.META, lane
                    )
                ]
                if _wide_meta_count(meta) == EMPTY_LANE:
                    continue
                live_lanes += 1
                node_union.grow(
                    _load_wide_lane_bounds[node_width](
                        nodes_span, UInt32(node_idx), lane
                    )
                )
            assert_true(live_lanes > 0)
            assert_true(live_lanes <= node_width)
            if tri_count > 1:
                assert_true(live_lanes >= 2)
            node_unions[node_idx] = node_union

        var seen_nodes = List[Bool](length=tree.node_count, fill=False)
        var seen_primitives = List[Bool](length=tri_count, fill=False)
        var pending = List[UInt32]()
        pending.append(tree.root_idx)
        var cursor = 0
        var visited_nodes = 0
        var visited_leaves = 0
        while cursor < len(pending):
            var node_idx = pending[cursor]
            cursor += 1
            assert_true(node_idx < UInt32(tree.node_count))
            assert_false(seen_nodes[Int(node_idx)])
            seen_nodes[Int(node_idx)] = True
            visited_nodes += 1

            comptime for lane in range(node_width):
                var meta = nodes_u32[
                    unsafe_offset=_wide_node_index[node_width](
                        node_idx, WideNode.META, lane
                    )
                ]
                var count = _wide_meta_count(meta)
                if count == EMPTY_LANE:
                    continue
                var data = _wide_meta_data(meta)
                var lane_bounds = _load_wide_lane_bounds[node_width](
                    nodes_span, node_idx, lane
                )
                if count == 0:
                    assert_true(data < UInt32(tree.node_count))
                    assert_true(
                        _bounds_match(lane_bounds, node_unions[Int(data)])
                    )
                    pending.append(data)
                else:
                    assert_true(count >= UInt32(1))
                    assert_true(count <= UInt32(fat_leaf_limit))
                    assert_true(data < UInt32(tree.leaf_block_count))
                    var block_base = Int(data) * leaf_width
                    var leaf_union = AABB[.WORLD].invalid()
                    for leaf_lane in range(Int(count)):
                        var payload = leaf_blocks[block_base + leaf_lane]
                        assert_true(payload < UInt32(tri_count))
                        assert_false(seen_primitives[Int(payload)])
                        seen_primitives[Int(payload)] = True
                        visited_leaves += 1
                        leaf_union.grow(triangle_bounds_list[Int(payload)])
                    assert_true(_bounds_match(lane_bounds, leaf_union))
                    comptime for leaf_lane in range(leaf_width):
                        if leaf_lane >= Int(count):
                            assert_equal(
                                leaf_blocks[block_base + leaf_lane], EMPTY_LANE
                            )

        assert_equal(visited_nodes, tree.node_count)
        assert_equal(visited_leaves, tri_count)
        for seen in seen_primitives:
            assert_true(seen)


def _assert_root_uses_largest_area_opening[
    node_width: SIMDLength,
](verts: List[Point3f32[.WORLD]]) raises:
    var build = _flatten_triangle_bounds(verts)
    with DeviceContext() as ctx:
        var diagnostic = build_bounds_bvh_for_diagnostics[
            node_width,
            node_width,
            Int(node_width),
            .HPLOC,
        ](
            ctx,
            upload_list(ctx, build[0]),
            upload_list(ctx, build[1]),
        )
        ref tree = diagnostic.wide
        ctx.synchronize()
        ref binary = diagnostic.build.binary
        assert_equal(tree.leaf_block_count, len(verts) / 3)

        with binary.node_meta.map_to_host() as meta, binary.node_bounds.map_to_host() as node_bounds, binary.leaf_bounds.map_to_host() as leaf_bounds, binary.leaf_ids.map_to_host() as leaf_ids, tree.wide_nodes.map_to_host() as wide_nodes:
            var root = UInt32(0)
            for i in range(binary.internal_count):
                if meta[i * 4] == UInt32.MAX:
                    root = UInt32(i)

            var meta_span = Span(unsafe_ptr=meta.unsafe_ptr(), length=len(meta))
            var node_bounds_span = Span(
                unsafe_ptr=node_bounds.unsafe_ptr(), length=len(node_bounds)
            )
            var leaf_bounds_span = Span(
                unsafe_ptr=leaf_bounds.unsafe_ptr(), length=len(leaf_bounds)
            )
            var leaf_ids_span = Span(
                unsafe_ptr=leaf_ids.unsafe_ptr(), length=len(leaf_ids)
            )
            var candidates = List[UInt32](capacity=node_width)
            candidates.append(_node_left(meta_span, root))
            candidates.append(_node_right(meta_span, root))

            while len(candidates) < node_width:
                var open_pos = -1
                var largest_area = Float32(-1.0)
                for i, candidate in enumerate(candidates):
                    if is_leaf_ref(candidate):
                        continue
                    var area = _encoded_bounds(
                        candidate,
                        leaf_bounds_span,
                        leaf_ids_span,
                        node_bounds_span,
                    ).surface_area()[0]
                    if area > largest_area:
                        largest_area = area
                        open_pos = i
                if open_pos < 0:
                    break
                var opened = candidates[open_pos]
                candidates[open_pos] = _node_left(meta_span, opened)
                candidates.append(_node_right(meta_span, opened))

            var wide_span = Span(
                unsafe_ptr=wide_nodes.unsafe_ptr(), length=len(wide_nodes)
            )
            var wide_u32 = wide_nodes.unsafe_ptr().unsafe_bitcast[UInt32]()
            for lane, candidate in enumerate(candidates):
                var actual = _load_wide_lane_bounds[node_width](
                    wide_span, UInt32(0), lane
                )
                var expected = _encoded_bounds(
                    candidate,
                    leaf_bounds_span,
                    leaf_ids_span,
                    node_bounds_span,
                )
                assert_true(_bounds_match(actual, expected))
                var count = _wide_meta_count(
                    wide_u32[
                        unsafe_offset=_wide_node_index[node_width](
                            UInt32(0), WideNode.META, lane
                        )
                    ]
                )
                if is_leaf_ref(candidates[lane]):
                    assert_equal(count, UInt32(1))
                else:
                    assert_equal(count, UInt32(0))


def _assert_hploc_triangle_matches_cpu[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    allow_identical_primitive_ties: Bool = False,
    expect_packing: Bool = False,
](verts: List[Point3f32[.WORLD]]) raises:
    comptime width = 32
    comptime height = 24
    comptime views = 2
    var cpu = build_triangle_blases[
        node_width, leaf_width, .LBVH, .WORLD
    ]([verts.copy()])
    var camera_data = _make_camera_rays_and_params(
        _triangle_bounds(verts), width, height, views
    )
    var rays = camera_data[0].copy()
    var camera_params = camera_data[1].copy()

    with DeviceContext() as ctx:
        var gpu = build_triangle_bvh[
            .WORLD,
            node_width,
            leaf_width,
            .HPLOC,
        ](ctx, upload_vertices(ctx, verts))
        _assert_literature_wide_invariants(gpu.tree, verts)
        comptime if expect_packing:
            assert_true(gpu.tree.leaf_block_count < len(verts) / 3)

        var hits = ctx.enqueue_create_buffer[.float32](
            len(rays) * Hit.STRIDE
        )
        gpu.launch_camera(
            ctx,
            upload_list(ctx, camera_params),
            hits,
            len(rays),
            width,
            height,
        )
        ctx.synchronize()

        with hits.map_to_host() as host_hits:
            var hit_span = Span(
                unsafe_ptr=host_hits.unsafe_ptr(), length=len(host_hits)
            )
            for i, ray in enumerate(rays):
                var actual = Hit[.WORLD].load(hit_span, i)
                var expected = trace_blas_set[
                    node_width, leaf_width, .CLOSEST_HIT, .WORLD
                ](cpu, UInt32(0), ray)
                var both_miss = actual.t >= f32_max and expected.t >= f32_max
                if not both_miss:
                    assert_true(abs(Float64(actual.t - expected.t)) <= 1.0e-4)
                    comptime if not allow_identical_primitive_ties:
                        assert_equal(actual.prim, expected.prim)


def test_hploc_section_3_4_opens_largest_area_candidate() raises:
    var scene = _make_irregular_scene(64)
    _assert_root_uses_largest_area_opening[4](scene)
    _assert_root_uses_largest_area_opening[8](scene)


def test_hploc_section_3_4_cross_block_invariants() raises:
    var scene = _make_irregular_scene(257)
    _assert_hploc_triangle_matches_cpu[8, 4, False, True](scene)


def test_hploc_wide_collapse_is_segmented_and_packed() raises:
    var segments = SegmentOffsets.from_counts([3, 1, 257])
    var verts = _make_irregular_scene(segments.item_count())
    var build = _flatten_triangle_bounds(verts)

    with DeviceContext() as ctx:
        var workspace = GpuBinaryBuildWorkspace(ctx, segments)
        var binary = GpuBinaryBoundsBvh(
            ctx,
            upload_list(ctx, build[0]),
            upload_list(ctx, build[1]),
            workspace,
        )
        _ = build_binary_bvh[.HPLOC](ctx, binary, workspace)
        var wide = GpuWideBoundsBvhBatch[8, 4, 4](ctx, segments)
        collapse_binary_to_wide_batch[8, 4, 4, True](ctx, binary, wide)

        with wide.wide_nodes.map_to_host() as nodes, wide.leaf_block_indices.map_to_host() as leaf_blocks, wide.node_counts.map_to_host() as node_counts, wide.leaf_block_counts.map_to_host() as leaf_counts:
            var nodes_u32 = nodes.unsafe_ptr().unsafe_bitcast[UInt32]()
            for segment_idx in range(segments.segment_count()):
                var node_base = Int(wide.node_segments.begin(segment_idx))
                var leaf_base = Int(wide.leaf_block_segments.begin(segment_idx))
                var node_count = Int(node_counts[segment_idx])
                var leaf_count = Int(leaf_counts[segment_idx])
                assert_true(node_count > 0)
                assert_true(leaf_count > 0)

                var seen_nodes = List[Bool](length=node_count, fill=False)
                var pending: List[UInt32] = [UInt32(0)]
                var cursor = 0
                var primitive_count = 0
                while cursor < len(pending):
                    var local_node = pending[cursor]
                    cursor += 1
                    assert_true(local_node < UInt32(node_count))
                    assert_false(seen_nodes[Int(local_node)])
                    seen_nodes[Int(local_node)] = True
                    comptime for lane in range(8):
                        var physical_node = UInt32(node_base) + local_node
                        var meta = nodes_u32[
                            unsafe_offset=_wide_node_index[8](
                                physical_node, WideNode.META, lane
                            )
                        ]
                        var count = _wide_meta_count(meta)
                        if count == EMPTY_LANE:
                            continue
                        var data = _wide_meta_data(meta)
                        if count == 0:
                            assert_true(data < UInt32(node_count))
                            pending.append(data)
                        else:
                            assert_true(data < UInt32(leaf_count))
                            var block_base = (leaf_base + Int(data)) * 4
                            for leaf_lane in range(Int(count)):
                                var payload = Int(
                                    leaf_blocks[block_base + leaf_lane]
                                )
                                assert_true(
                                    payload >= Int(segments.begin(segment_idx))
                                    and payload < Int(segments.end(segment_idx))
                                )
                                primitive_count += 1

                assert_equal(primitive_count, Int(segments.count(segment_idx)))
                for seen in seen_nodes:
                    assert_true(seen)


def test_hploc_triangle_opt_in_widths_match_cpu() raises:
    var scene = _make_small_scene[.WORLD]()
    _assert_hploc_triangle_matches_cpu[2, 2](scene)
    _assert_hploc_triangle_matches_cpu[2, 4](scene)
    _assert_hploc_triangle_matches_cpu[4, 4](scene)
    _assert_hploc_triangle_matches_cpu[8, 8](scene)
    _assert_hploc_triangle_matches_cpu[16, 4](scene)


def test_hploc_triangle_opt_in_edge_cases() raises:
    _assert_hploc_triangle_matches_cpu[4, 4](
        _make_single_triangle_scene[.WORLD]()
    )
    _assert_hploc_triangle_matches_cpu[8, 4, True](
        _make_duplicate_centroid_scene[.WORLD]()
    )


def main() raises:
    comptime if not has_accelerator():
        raise "No Accelerator found"
    TestSuite.discover_tests[__functions_in_module()]().run()
