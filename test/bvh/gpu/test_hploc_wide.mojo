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
from bajo.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.bvh.gpu.diagnostics import build_bounds_bvh_for_diagnostics
from bajo.bvh.gpu.wide_layout import GpuWideBoundsBvh, _wide_node_base
from bajo.bvh.gpu.builder import GpuBvhBuildMethod
from bajo.bvh.gpu.builder.binary_layout import (
    _is_encoded_leaf,
    _node_left,
    _node_right,
)
from bajo.bvh.gpu.builder.binary_layout import _encoded_bounds
from bajo.bvh.gpu.triangle_bvh import build_triangle_bvh
from bajo.bvh.gpu.utils import upload_list, upload_vertices
from bajo.bvh.gpu.trace import GpuTraversalAlgorithm
from bajo.bvh.gpu.wide_meta import _wide_meta_count, _wide_meta_data
from bajo.bvh.host_utils import triangle_bounds
from bajo.bvh.types import Hit
from bajo.core import AABB, Frame, Point3f32

from test.bvh.fixtures import (
    _append_tri,
    _make_camera_rays_and_params,
    _make_duplicate_centroid_scene,
    _make_single_triangle_scene,
    _make_small_scene,
)


def _bounds_match(
    actual: AABB[Frame.WORLD], expected: AABB[Frame.WORLD]
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


def _make_irregular_scene(
    count: Int,
) -> List[Point3f32[Frame.WORLD]]:
    var verts = List[Point3f32[Frame.WORLD]](capacity=count * 3)
    for i in range(count):
        var x = Float32((i % 19) * 4 - 36)
        var y = Float32(((i / 19) % 14) * 4 - 26)
        var z = Float32((i * 7) % 11 + 2)
        var scale = Float32(2.5) if i % 17 == 0 else Float32(0.7)
        verts.append(Point3f32[Frame.WORLD](x - scale, y - scale, z))
        verts.append(Point3f32[Frame.WORLD](x + scale, y - scale, z))
        verts.append(Point3f32[Frame.WORLD](x, y + scale, z + 0.1))
    return verts^


def _flatten_triangle_bounds(
    verts: List[Point3f32[Frame.WORLD]],
) -> Tuple[List[Float32], List[UInt32]]:
    var tri_count = len(verts) / 3
    var flat = List[Float32](capacity=tri_count * AABB.STRIDE)
    var payloads = List[UInt32](capacity=tri_count)
    for i in range(tri_count):
        var bounds = triangle_bounds(
            verts[i * 3], verts[i * 3 + 1], verts[i * 3 + 2]
        )
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
    verts: List[Point3f32[Frame.WORLD]],
) raises:
    var tri_count = len(verts) / 3
    comptime fat_leaf_limit = min(max_leaf_size, 4)
    assert_true(tree.leaf_block_count > 0)
    assert_true(tree.leaf_block_count <= tri_count)
    assert_true(tree.node_count > 0)

    var triangle_bounds_list = List[AABB[Frame.WORLD]](capacity=tri_count)
    for i in range(tri_count):
        triangle_bounds_list.append(
            triangle_bounds(verts[i * 3], verts[i * 3 + 1], verts[i * 3 + 2])
        )

    with tree.wide_nodes.map_to_host() as nodes, tree.leaf_block_indices.map_to_host() as leaf_blocks:
        var nodes_span = Span(unsafe_ptr=nodes.unsafe_ptr(), length=len(nodes))
        var nodes_u32 = nodes.unsafe_ptr().unsafe_bitcast[UInt32]()
        var node_unions = List[AABB[Frame.WORLD]](
            length=tree.node_count, fill=AABB[Frame.WORLD].invalid()
        )

        for node_idx in range(tree.node_count):
            var node_union = AABB[Frame.WORLD].invalid()
            var live_lanes = 0
            comptime for lane in range(node_width):
                var base = _wide_node_base[node_width](UInt32(node_idx), lane)
                var meta = nodes_u32[unsafe_offset=base + WideNode.META]
                if _wide_meta_count(meta) == EMPTY_LANE:
                    continue
                live_lanes += 1
                node_union.grow(AABB[Frame.WORLD].load6(nodes_span, base))
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
                var base = _wide_node_base[node_width](node_idx, lane)
                var meta = nodes_u32[unsafe_offset=base + WideNode.META]
                var count = _wide_meta_count(meta)
                if count == EMPTY_LANE:
                    continue
                var data = _wide_meta_data(meta)
                var lane_bounds = AABB[Frame.WORLD].load6(nodes_span, base)
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
                    var leaf_union = AABB[Frame.WORLD].invalid()
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
](verts: List[Point3f32[Frame.WORLD]]) raises:
    var build = _flatten_triangle_bounds(verts)
    with DeviceContext() as ctx:
        var tree = GpuWideBoundsBvh[node_width, node_width](ctx, len(build[1]))
        var binary = build_bounds_bvh_for_diagnostics[
            node_width,
            node_width,
            Int(node_width),
            GpuBvhBuildMethod.HPLOC,
        ](
            ctx,
            tree,
            upload_list(ctx, build[0]),
            upload_list(ctx, build[1]),
        )
        ctx.synchronize()
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
                for i in range(len(candidates)):
                    if _is_encoded_leaf(candidates[i]):
                        continue
                    var area = _encoded_bounds(
                        candidates[i],
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
            for lane in range(len(candidates)):
                var base = _wide_node_base[node_width](UInt32(0), lane)
                var actual = AABB[Frame.WORLD].load6(wide_span, base)
                var expected = _encoded_bounds(
                    candidates[lane],
                    leaf_bounds_span,
                    leaf_ids_span,
                    node_bounds_span,
                )
                assert_true(_bounds_match(actual, expected))
                var count = _wide_meta_count(
                    wide_u32[unsafe_offset=base + WideNode.META]
                )
                if _is_encoded_leaf(candidates[lane]):
                    assert_equal(count, UInt32(1))
                else:
                    assert_equal(count, UInt32(0))


def _assert_hploc_triangle_matches_cpu[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    allow_identical_primitive_ties: Bool = False,
    expect_packing: Bool = False,
](verts: List[Point3f32[Frame.WORLD]]) raises:
    comptime width = 32
    comptime height = 24
    comptime views = 2
    var cpu = TriangleBvh[Frame.WORLD, node_width, leaf_width].__init__["lbvh"](
        verts
    )
    var camera_data = _make_camera_rays_and_params(
        cpu.bounds(), width, height, views
    )
    var rays = camera_data[0].copy()
    var camera_params = camera_data[1].copy()

    with DeviceContext() as ctx:
        var gpu = build_triangle_bvh[
            Frame.WORLD,
            node_width,
            leaf_width,
            GpuBvhBuildMethod.HPLOC,
        ](ctx, upload_vertices(ctx, verts))
        _assert_literature_wide_invariants(gpu.tree, verts)
        comptime if expect_packing:
            assert_true(gpu.tree.leaf_block_count < len(verts) / 3)

        var hits = ctx.enqueue_create_buffer[DType.float32](
            len(rays) * Hit.STRIDE
        )
        gpu.launch_camera[algorithm=GpuTraversalAlgorithm.UNIFIED_TASKS](
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
            for i in range(len(rays)):
                var actual = Hit[Frame.WORLD].load(hit_span, i)
                var expected = cpu.trace[TRACE.CLOSEST_HIT](rays[i])
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


def test_hploc_triangle_opt_in_widths_match_cpu() raises:
    var scene = _make_small_scene[Frame.WORLD]()
    _assert_hploc_triangle_matches_cpu[2, 2](scene)
    _assert_hploc_triangle_matches_cpu[2, 4](scene)
    _assert_hploc_triangle_matches_cpu[4, 4](scene)
    _assert_hploc_triangle_matches_cpu[8, 8](scene)
    _assert_hploc_triangle_matches_cpu[16, 4](scene)


def test_hploc_triangle_opt_in_edge_cases() raises:
    _assert_hploc_triangle_matches_cpu[4, 4](
        _make_single_triangle_scene[Frame.WORLD]()
    )
    _assert_hploc_triangle_matches_cpu[8, 4, True](
        _make_duplicate_centroid_scene[Frame.WORLD]()
    )


def main() raises:
    comptime if not has_accelerator():
        raise "No Accelerator found"
    TestSuite.discover_tests[__functions_in_module()]().run()
