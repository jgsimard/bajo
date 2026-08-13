from std.math import abs
from std.sys import has_accelerator
from std.testing import TestSuite, assert_equal, assert_true
from max.gpu.host import DeviceContext

from bajo.bvh.constants import LBVH_SENTINEL
from bajo.bvh.gpu.diagnostics import build_bounds_bvh_for_diagnostics
from bajo.bvh.gpu.wide_layout import GpuWideBoundsBvh
from bajo.bvh.gpu.builder.hploc_reference import build_hploc_reference
from bajo.bvh.gpu.quality import measure_binary_bvh_quality
from bajo.bvh.gpu.utils import upload_list
from bajo.bvh.host_utils import triangle_bounds
from bajo.core import AABB, Frame, Point3f32


def _box(center_x: Float32) -> AABB[Frame.WORLD]:
    return AABB[Frame.WORLD](
        Point3f32[Frame.WORLD](center_x - 0.1, -0.1, -0.1),
        Point3f32[Frame.WORLD](center_x + 0.1, 0.1, 0.1),
    )


def _identity_ids(count: Int) -> List[UInt32]:
    var ids = List[UInt32](capacity=count)
    for i in range(count):
        ids.append(UInt32(i))
    return ids^


def test_hploc_reference_hierarchical_merging_exact_topology() raises:
    var centers: List[Float32] = [
        0.0,
        1.0,
        10.0,
        11.0,
        100.0,
        101.0,
        110.0,
        111.0,
    ]
    var bounds = List[AABB[Frame.WORLD]](capacity=len(centers))
    for center in centers:
        bounds.append(_box(center))

    var codes: List[UInt32] = [0, 1, 2, 3, 256, 257, 258, 259]
    var ids = _identity_ids(len(bounds))
    var tree = build_hploc_reference(
        Span(bounds),
        Span(codes),
        Span(ids),
        search_radius=8,
        merging_threshold=2,
    )

    assert_true(tree.validate())
    assert_equal(tree.root, UInt32(14))
    assert_equal(len(tree.nodes), 15)
    assert_equal(tree.stats.guide_nodes, 7)
    assert_equal(tree.stats.merge_calls, 3)
    assert_equal(tree.stats.hierarchical_rounds, 2)
    assert_equal(tree.stats.final_rounds, 2)

    var expected_left: List[UInt32] = [0, 2, 4, 6, 8, 10, 12]
    var expected_right: List[UInt32] = [1, 3, 5, 7, 9, 11, 13]
    for internal in range(7):
        var node = tree.nodes[8 + internal]
        assert_equal(node.left, expected_left[internal])
        assert_equal(node.right, expected_right[internal])


def test_hploc_reference_strict_ties_prefer_right_then_near() raises:
    var bounds = List[AABB[Frame.WORLD]](capacity=4)
    for _ in range(4):
        bounds.append(_box(0.0))
    var codes: List[UInt32] = [7, 7, 7, 7]
    var ids = _identity_ids(4)

    var tree = build_hploc_reference(
        Span(bounds), Span(codes), Span(ids), merging_threshold=16
    )
    assert_true(tree.validate())

    # Equal areas exercise the paper implementation's strict '<' update:
    # nearest right wins before nearest left at the same radius.
    assert_equal(tree.nodes[4].left, UInt32(2))
    assert_equal(tree.nodes[4].right, UInt32(3))
    assert_equal(tree.nodes[5].left, UInt32(1))
    assert_equal(tree.nodes[5].right, UInt32(4))
    assert_equal(tree.nodes[6].left, UInt32(0))
    assert_equal(tree.nodes[6].right, UInt32(5))
    assert_equal(tree.root, UInt32(6))


def test_hploc_reference_duplicate_codes_and_permutation_are_deterministic() raises:
    var bounds: List[AABB[Frame.WORLD]] = [
        _box(-6.0),
        _box(-2.0),
        _box(2.0),
        _box(6.0),
    ]
    var codes: List[UInt32] = [11, 11, 11, 11]
    var ids: List[UInt32] = [3, 1, 2, 0]

    var first = build_hploc_reference(
        Span(bounds), Span(codes), Span(ids), merging_threshold=2
    )
    var second = build_hploc_reference(
        Span(bounds), Span(codes), Span(ids), merging_threshold=2
    )
    assert_true(first.validate())
    assert_true(second.validate())
    assert_equal(first.topology_checksum(), second.topology_checksum())
    assert_equal(first.nodes[Int(first.root)].parent, LBVH_SENTINEL)


def _make_quality_triangles() -> List[AABB[Frame.WORLD]]:
    var bounds = List[AABB[Frame.WORLD]](capacity=64)
    for i in range(64):
        var x = Float32((i % 8) * 4 - 14)
        var y = Float32(((i / 8) % 8) * 4 - 14)
        var z = Float32((i * 7) % 5)
        var scale = Float32(2.5) if i % 9 == 0 else Float32(0.7)
        var p0 = Point3f32[Frame.WORLD](x - scale, y - scale, z)
        var p1 = Point3f32[Frame.WORLD](x + scale, y - scale, z)
        var p2 = Point3f32[Frame.WORLD](x, y + scale, z + 0.1)
        bounds.append(triangle_bounds(p0, p1, p2))
    return bounds^


def test_hploc_reference_matches_gpu_lbvh_inputs_and_quality_gate() raises:
    var bounds = _make_quality_triangles()
    var flat_bounds = List[Float32](capacity=len(bounds) * AABB.STRIDE)
    var payloads = _identity_ids(len(bounds))
    for bound in bounds:
        flat_bounds.append(bound._min.x)
        flat_bounds.append(bound._min.y)
        flat_bounds.append(bound._min.z)
        flat_bounds.append(bound._max.x)
        flat_bounds.append(bound._max.y)
        flat_bounds.append(bound._max.z)

    with DeviceContext() as ctx:
        var device_bounds = upload_list(ctx, flat_bounds)
        var device_payloads = upload_list(ctx, payloads)
        var wide = GpuWideBoundsBvh[2, 2](ctx, len(bounds))
        var diagnostic = build_bounds_bvh_for_diagnostics(
            ctx, wide, device_bounds, device_payloads
        )
        ref binary = diagnostic.binary
        ctx.synchronize()

        var codes = List[UInt32](capacity=len(bounds))
        var sorted_ids = List[UInt32](capacity=len(bounds))
        with diagnostic.workspace.topology.value().morton_keys.map_to_host() as host_codes:
            for i in range(len(host_codes)):
                codes.append(host_codes[i])
        with binary.leaf_ids.map_to_host() as host_ids:
            for i in range(len(host_ids)):
                sorted_ids.append(host_ids[i])

        var hploc = build_hploc_reference(
            Span(bounds), Span(codes), Span(sorted_ids)
        )
        var lbvh_quality = measure_binary_bvh_quality(binary)

        assert_true(hploc.validate())
        assert_equal(hploc.leaf_count, binary.leaf_count)
        assert_equal(len(hploc.nodes), binary.leaf_count * 2 - 1)
        assert_true(
            abs(
                Float64(hploc.root_bounds().surface_area()[0])
                - Float64(binary.root_bounds().surface_area()[0])
            )
            <= 1.0e-4
        )
        assert_true(hploc.quality() <= lbvh_quality.quality + 1.0e-5)


def main() raises:
    comptime if not has_accelerator():
        raise "No Accelerator found"
    TestSuite.discover_tests[__functions_in_module()]().run()
