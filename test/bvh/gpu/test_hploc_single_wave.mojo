from std.math import abs
from std.sys import has_accelerator
from std.testing import TestSuite, assert_equal, assert_true
from max.gpu.host import DeviceContext

from bajo.bvh.constants import LBVH_SENTINEL
from bajo.bvh.gpu.diagnostics import build_bounds_bvh_for_diagnostics
from bajo.bvh.gpu.wide_layout import GpuWideBoundsBvh
from bajo.bvh.cpu.builder.hploc import (
    HplocTopology,
    build_hploc_topology,
)
from bajo.bvh.gpu.builder.hploc_layout import (
    HPLOC_NODE_LEAF_ID,
    HPLOC_NODE_LEFT,
    HPLOC_NODE_META_STRIDE,
    HPLOC_NODE_PARENT,
    HPLOC_NODE_RIGHT,
    HPLOC_STATUS_OK,
)
from bajo.bvh.gpu.builder.hploc_single_wave import GpuHplocSingleWaveBvh
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


def _flatten_bounds(
    bounds: List[AABB[Frame.WORLD]],
) -> List[Float32]:
    var flat = List[Float32](capacity=len(bounds) * AABB.STRIDE)
    for bound in bounds:
        flat.append(bound._min.x)
        flat.append(bound._min.y)
        flat.append(bound._min.z)
        flat.append(bound._max.x)
        flat.append(bound._max.y)
        flat.append(bound._max.z)
    return flat^


def _bounds_match(
    actual: AABB[Frame.WORLD],
    expected: AABB[Frame.WORLD],
    tolerance: Float64 = 1.0e-5,
) -> Bool:
    return (
        abs(Float64(actual._min.x - expected._min.x)) <= tolerance
        and abs(Float64(actual._min.y - expected._min.y)) <= tolerance
        and abs(Float64(actual._min.z - expected._min.z)) <= tolerance
        and abs(Float64(actual._max.x - expected._max.x)) <= tolerance
        and abs(Float64(actual._max.y - expected._max.y)) <= tolerance
        and abs(Float64(actual._max.z - expected._max.z)) <= tolerance
    )


def _assert_exact_gpu_tree(
    gpu: GpuHplocSingleWaveBvh,
    reference: HplocTopology[Frame.WORLD],
) raises:
    assert_equal(gpu.result_status(), UInt32(HPLOC_STATUS_OK))
    assert_equal(gpu.result_root(), reference.root)
    assert_equal(gpu.result_node_count(), UInt32(len(reference.nodes)))

    with gpu.node_meta.map_to_host() as meta, gpu.node_bounds.map_to_host() as bounds:
        for node_idx in range(len(reference.nodes)):
            var expected = reference.nodes[node_idx]
            var base = node_idx * HPLOC_NODE_META_STRIDE
            assert_equal(meta[base + HPLOC_NODE_PARENT], expected.parent)
            assert_equal(meta[base + HPLOC_NODE_LEFT], expected.left)
            assert_equal(meta[base + HPLOC_NODE_RIGHT], expected.right)
            assert_equal(meta[base + HPLOC_NODE_LEAF_ID], expected.leaf_id)

            var bounds_base = node_idx * AABB.STRIDE
            var actual_bounds = AABB[Frame.WORLD](
                Point3f32[Frame.WORLD](
                    bounds[bounds_base],
                    bounds[bounds_base + 1],
                    bounds[bounds_base + 2],
                ),
                Point3f32[Frame.WORLD](
                    bounds[bounds_base + 3],
                    bounds[bounds_base + 4],
                    bounds[bounds_base + 5],
                ),
            )
            assert_true(_bounds_match(actual_bounds, expected.bounds))


def test_hploc_single_wave_matches_balanced_reference_exactly() raises:
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
    var leaf_bounds = List[AABB[Frame.WORLD]](capacity=len(centers))
    for center in centers:
        leaf_bounds.append(_box(center))
    var codes: List[UInt32] = [0, 1, 2, 3, 256, 257, 258, 259]
    var ids = _identity_ids(len(leaf_bounds))
    var reference = build_hploc_topology(
        Span(leaf_bounds),
        Span(codes),
        Span(ids),
        search_radius=8,
        merging_threshold=2,
    )
    var flat_bounds = _flatten_bounds(leaf_bounds)

    with DeviceContext() as ctx:
        var gpu = GpuHplocSingleWaveBvh(
            ctx,
            upload_list(ctx, flat_bounds),
            upload_list(ctx, codes),
            upload_list(ctx, ids),
            search_radius=8,
            merging_threshold=2,
        )
        ctx.synchronize()
        _assert_exact_gpu_tree(gpu, reference)


def test_hploc_single_wave_matches_strict_tie_reference_exactly() raises:
    var leaf_bounds = List[AABB[Frame.WORLD]](capacity=4)
    for _ in range(4):
        leaf_bounds.append(_box(0.0))
    var codes: List[UInt32] = [7, 7, 7, 7]
    var ids = _identity_ids(4)
    var reference = build_hploc_topology(
        Span(leaf_bounds), Span(codes), Span(ids)
    )
    var flat_bounds = _flatten_bounds(leaf_bounds)

    with DeviceContext() as ctx:
        var gpu = GpuHplocSingleWaveBvh(
            ctx,
            upload_list(ctx, flat_bounds),
            upload_list(ctx, codes),
            upload_list(ctx, ids),
        )
        ctx.synchronize()
        _assert_exact_gpu_tree(gpu, reference)


def test_hploc_single_wave_duplicate_codes_and_permutation_repeat() raises:
    var leaf_bounds: List[AABB[Frame.WORLD]] = [
        _box(-6.0),
        _box(-2.0),
        _box(2.0),
        _box(6.0),
    ]
    var codes: List[UInt32] = [11, 11, 11, 11]
    var ids: List[UInt32] = [3, 1, 2, 0]
    var reference = build_hploc_topology(
        Span(leaf_bounds), Span(codes), Span(ids), merging_threshold=2
    )
    var flat_bounds = _flatten_bounds(leaf_bounds)

    with DeviceContext() as ctx:
        var first = GpuHplocSingleWaveBvh(
            ctx,
            upload_list(ctx, flat_bounds),
            upload_list(ctx, codes),
            upload_list(ctx, ids),
            merging_threshold=2,
        )
        var second = GpuHplocSingleWaveBvh(
            ctx,
            upload_list(ctx, flat_bounds),
            upload_list(ctx, codes),
            upload_list(ctx, ids),
            merging_threshold=2,
        )
        ctx.synchronize()
        _assert_exact_gpu_tree(first, reference)
        _assert_exact_gpu_tree(second, reference)

        with first.node_meta.map_to_host() as first_meta, second.node_meta.map_to_host() as second_meta:
            for i in range(len(first_meta)):
                assert_equal(first_meta[i], second_meta[i])


def test_hploc_single_wave_handles_one_leaf() raises:
    var leaf_bounds: List[AABB[Frame.WORLD]] = [_box(3.0)]
    var codes: List[UInt32] = [42]
    var ids: List[UInt32] = [0]
    var reference = build_hploc_topology(
        Span(leaf_bounds), Span(codes), Span(ids)
    )
    var flat_bounds = _flatten_bounds(leaf_bounds)

    with DeviceContext() as ctx:
        var gpu = GpuHplocSingleWaveBvh(
            ctx,
            upload_list(ctx, flat_bounds),
            upload_list(ctx, codes),
            upload_list(ctx, ids),
        )
        ctx.synchronize()
        _assert_exact_gpu_tree(gpu, reference)


def _make_triangles(count: Int) -> List[AABB[Frame.WORLD]]:
    var bounds = List[AABB[Frame.WORLD]](capacity=count)
    for i in range(count):
        var x = Float32((i % 7) * 4 - 12)
        var y = Float32(((i / 7) % 5) * 5 - 9)
        var z = Float32((i * 7) % 5)
        var scale = Float32(2.0) if i % 8 == 0 else Float32(0.6)
        bounds.append(
            triangle_bounds(
                Point3f32[Frame.WORLD](x - scale, y - scale, z),
                Point3f32[Frame.WORLD](x + scale, y - scale, z),
                Point3f32[Frame.WORLD](x, y + scale, z + 0.1),
            )
        )
    return bounds^


def _find_mask(mask: UInt32, masks: List[UInt32], first: Int) -> Int:
    for node_idx in range(first, len(masks)):
        if masks[node_idx] == mask:
            return node_idx
    return -1


def test_hploc_single_wave_matches_reference_from_gpu_lbvh_inputs() raises:
    comptime leaf_count = 31
    var leaf_bounds = _make_triangles(leaf_count)
    var flat_bounds = _flatten_bounds(leaf_bounds)
    var payloads = _identity_ids(leaf_count)

    with DeviceContext() as ctx:
        var device_bounds = upload_list(ctx, flat_bounds)
        var device_payloads = upload_list(ctx, payloads)
        var wide = GpuWideBoundsBvh[2, 2](ctx, leaf_count)
        var diagnostic = build_bounds_bvh_for_diagnostics(
            ctx, wide, device_bounds, device_payloads
        )
        ref binary = diagnostic.binary
        ctx.synchronize()

        var codes = List[UInt32](capacity=leaf_count)
        var sorted_ids = List[UInt32](capacity=leaf_count)
        with diagnostic.workspace.topology.value().morton_keys.map_to_host() as host_codes:
            for i in range(leaf_count):
                codes.append(host_codes[i])
        with binary.leaf_ids.map_to_host() as host_ids:
            for i in range(leaf_count):
                sorted_ids.append(host_ids[i])

        var reference = build_hploc_topology(
            Span(leaf_bounds), Span(codes), Span(sorted_ids)
        )
        var gpu = GpuHplocSingleWaveBvh(
            ctx,
            binary.leaf_bounds.copy(),
            diagnostic.workspace.topology.value().morton_keys.copy(),
            binary.leaf_ids.copy(),
        )
        ctx.synchronize()
        assert_equal(gpu.result_status(), UInt32(HPLOC_STATUS_OK))
        assert_equal(gpu.result_node_count(), UInt32(leaf_count * 2 - 1))

        var reference_masks = List[UInt32](
            length=leaf_count * 2 - 1, fill=UInt32(0)
        )
        for node_idx in range(len(reference.nodes)):
            var node = reference.nodes[node_idx]
            if node.left == LBVH_SENTINEL:
                reference_masks[node_idx] = UInt32(1) << node.leaf_id
            else:
                reference_masks[node_idx] = (
                    reference_masks[Int(node.left)]
                    | reference_masks[Int(node.right)]
                )

        with gpu.node_meta.map_to_host() as meta, gpu.node_bounds.map_to_host() as gpu_bounds:
            var gpu_masks = List[UInt32](
                length=leaf_count * 2 - 1, fill=UInt32(0)
            )
            for node_idx in range(leaf_count * 2 - 1):
                var base = node_idx * HPLOC_NODE_META_STRIDE
                var left = meta[base + HPLOC_NODE_LEFT]
                var right = meta[base + HPLOC_NODE_RIGHT]
                if left == LBVH_SENTINEL:
                    gpu_masks[node_idx] = (
                        UInt32(1) << meta[base + HPLOC_NODE_LEAF_ID]
                    )
                else:
                    gpu_masks[node_idx] = (
                        gpu_masks[Int(left)] | gpu_masks[Int(right)]
                    )

            for gpu_idx in range(leaf_count, leaf_count * 2 - 1):
                var reference_idx = _find_mask(
                    gpu_masks[gpu_idx], reference_masks, leaf_count
                )
                assert_true(reference_idx >= leaf_count)
                var gpu_base = gpu_idx * HPLOC_NODE_META_STRIDE
                var expected = reference.nodes[reference_idx]
                assert_equal(
                    gpu_masks[Int(meta[gpu_base + HPLOC_NODE_LEFT])],
                    reference_masks[Int(expected.left)],
                )
                assert_equal(
                    gpu_masks[Int(meta[gpu_base + HPLOC_NODE_RIGHT])],
                    reference_masks[Int(expected.right)],
                )
                var bounds_base = gpu_idx * AABB.STRIDE
                var actual_bounds = AABB[Frame.WORLD](
                    Point3f32[Frame.WORLD](
                        gpu_bounds[bounds_base],
                        gpu_bounds[bounds_base + 1],
                        gpu_bounds[bounds_base + 2],
                    ),
                    Point3f32[Frame.WORLD](
                        gpu_bounds[bounds_base + 3],
                        gpu_bounds[bounds_base + 4],
                        gpu_bounds[bounds_base + 5],
                    ),
                )
                assert_true(
                    _bounds_match(actual_bounds, expected.bounds, 1.0e-4)
                )


def main() raises:
    comptime if not has_accelerator():
        raise "No Accelerator found"
    TestSuite.discover_tests[__functions_in_module()]().run()
