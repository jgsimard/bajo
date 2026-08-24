from std.math import abs, max
from std.sys import has_accelerator
from std.testing import TestSuite, assert_equal, assert_false, assert_true
from max.gpu.host import DeviceContext

from bajo.bvh.constants import (
    BinaryBvhNode,
    LBVH_SENTINEL,
)
from bajo.bvh.tagged_ref import decode_ref_index, is_leaf_ref
from bajo.bvh.gpu.builder import build_binary_bvh
from bajo.bvh.gpu.builder.binary_layout import (
    GpuBinaryBoundsBvh,
    GpuBinaryBuildWorkspace,
)
from bajo.bvh.gpu.builder.lbvh import (
    enqueue_segmented_morton_codes,
    enqueue_segmented_morton_sort,
)
from bajo.bvh.gpu.builder.hploc_binary import enqueue_binary_bvh_with_hploc
from bajo.bvh.cpu.builder.hploc import (
    HplocTopology,
    build_hploc_topology,
)
from bajo.bvh.gpu.quality import measure_binary_bvh_quality
from bajo.bvh.gpu.diagnostics import validate_binary_bvh
from bajo.bvh.gpu.utils import upload_list
from bajo.core import AABB, Point3f32, SegmentOffsets


@fieldwise_init
struct _TestBinaryBuild:
    var binary: GpuBinaryBoundsBvh
    var workspace: GpuBinaryBuildWorkspace


def _identity_ids(count: Int) -> List[UInt32]:
    return [UInt32(i) for i in range(count)]


def _flatten_bounds(
    bounds: List[AABB[.WORLD]],
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


def _rotate_left(value: UInt64, shift: Int) -> UInt64:
    return (value << UInt64(shift)) | (value >> UInt64(64 - shift))


def _leaf_hash(leaf_id: UInt32) -> UInt64:
    return (
        UInt64(leaf_id + 1)
        ^ (UInt64(leaf_id) << 32)
        ^ UInt64(0x9E3779B97F4A7C15)
    )


def _inner_hash(left: UInt64, right: UInt64) -> UInt64:
    return (
        _rotate_left(left, 13)
        ^ _rotate_left(right, 37)
        ^ UInt64(0xD6E8FEB86659FD93)
    )


def _reference_root_hash(reference: HplocTopology[.WORLD]) -> UInt64:
    var hashes = List[UInt64](length=len(reference.nodes), fill=UInt64(0))
    for i, node in enumerate(reference.nodes):
        if node.left == LBVH_SENTINEL:
            hashes[i] = _leaf_hash(node.leaf_id)
        else:
            hashes[i] = _inner_hash(
                hashes[Int(node.left)], hashes[Int(node.right)]
            )
    return hashes[Int(reference.root)]


def _assert_binary_matches_reference(
    build: _TestBinaryBuild,
    reference: HplocTopology[.WORLD],
) raises:
    ref binary = build.binary
    var validation = validate_binary_bvh(
        binary, build.workspace, binary.root_bounds()
    )
    assert_true(validation.sorted_ok)
    assert_true(validation.values_ok)
    assert_true(validation.topology_ok)
    assert_true(validation.bounds_ok)

    if binary.leaf_count == 1:
        assert_equal(binary.internal_count, 0)
        return

    var root = UInt32(LBVH_SENTINEL)
    var hashes = List[UInt64](length=binary.internal_count, fill=UInt64(0))
    var counts = List[UInt32](length=binary.internal_count, fill=UInt32(0))
    var subtree_bounds = List[AABB[.WORLD]](
        length=binary.internal_count, fill=AABB[.WORLD].invalid()
    )

    with binary.node_meta.map_to_host() as meta, binary.node_bounds.map_to_host() as node_bounds, binary.node_leaf_counts.map_to_host() as leaf_counts, binary.leaf_ids.map_to_host() as leaf_ids, binary.leaf_bounds.map_to_host() as leaf_bounds:
        var node_bounds_span = Span(
            unsafe_ptr=node_bounds.unsafe_ptr(), length=len(node_bounds)
        )
        var leaf_bounds_span = Span(
            unsafe_ptr=leaf_bounds.unsafe_ptr(), length=len(leaf_bounds)
        )

        for node_idx in range(binary.internal_count):
            var base = node_idx * BinaryBvhNode.META_STRIDE
            var parent = meta[base + BinaryBvhNode.PARENT]
            var left = meta[base + BinaryBvhNode.LEFT]
            var right = meta[base + BinaryBvhNode.RIGHT]
            assert_equal(meta[base + BinaryBvhNode.FENCE], LBVH_SENTINEL)
            if parent == LBVH_SENTINEL:
                assert_equal(root, LBVH_SENTINEL)
                root = UInt32(node_idx)

            var left_hash: UInt64
            var left_count: UInt32
            var left_bounds: AABB[.WORLD]
            if is_leaf_ref(left):
                var sorted_pos = Int(decode_ref_index(left))
                var leaf_id = leaf_ids[sorted_pos]
                left_hash = _leaf_hash(leaf_id)
                left_count = UInt32(1)
                left_bounds = AABB[.WORLD].load6(
                    leaf_bounds_span, Int(leaf_id) * AABB.STRIDE
                )
            else:
                var child = Int(decode_ref_index(left))
                assert_true(child < node_idx)
                left_hash = hashes[child]
                left_count = counts[child]
                left_bounds = subtree_bounds[child]

            var right_hash: UInt64
            var right_count: UInt32
            var right_bounds: AABB[.WORLD]
            if is_leaf_ref(right):
                var sorted_pos = Int(decode_ref_index(right))
                var leaf_id = leaf_ids[sorted_pos]
                right_hash = _leaf_hash(leaf_id)
                right_count = UInt32(1)
                right_bounds = AABB[.WORLD].load6(
                    leaf_bounds_span, Int(leaf_id) * AABB.STRIDE
                )
            else:
                var child = Int(decode_ref_index(right))
                assert_true(child < node_idx)
                right_hash = hashes[child]
                right_count = counts[child]
                right_bounds = subtree_bounds[child]

            var stored_left = AABB[.WORLD].load6(
                node_bounds_span, node_idx * BinaryBvhNode.BOUNDS_STRIDE
            )
            var stored_right = AABB[.WORLD].load6(
                node_bounds_span,
                node_idx * BinaryBvhNode.BOUNDS_STRIDE + AABB.STRIDE,
            )
            assert_true(_bounds_match(stored_left, left_bounds))
            assert_true(_bounds_match(stored_right, right_bounds))

            hashes[node_idx] = _inner_hash(left_hash, right_hash)
            counts[node_idx] = left_count + right_count
            subtree_bounds[node_idx] = AABB[.WORLD].merge(
                left_bounds, right_bounds
            )
            assert_equal(leaf_counts[node_idx], counts[node_idx])

        assert_true(root != LBVH_SENTINEL)
        assert_equal(hashes[Int(root)], _reference_root_hash(reference))
        assert_equal(counts[Int(root)], UInt32(binary.leaf_count))

        var visited = List[Bool](length=binary.internal_count, fill=False)
        var pending = List[UInt32]()
        pending.append(root)
        var cursor = 0
        var visited_count = 0
        while cursor < len(pending):
            var node_idx = pending[cursor]
            cursor += 1
            assert_false(visited[Int(node_idx)])
            visited[Int(node_idx)] = True
            visited_count += 1
            var base = Int(node_idx) * BinaryBvhNode.META_STRIDE
            var left = meta[base + BinaryBvhNode.LEFT]
            var right = meta[base + BinaryBvhNode.RIGHT]
            if not is_leaf_ref(left):
                pending.append(left)
            if not is_leaf_ref(right):
                pending.append(right)
        assert_equal(visited_count, binary.internal_count)

    var quality = measure_binary_bvh_quality(binary)
    assert_true(abs(quality.quality - reference.quality()) <= 1.0e-4)


def _make_triangles(count: Int) -> List[AABB[.WORLD]]:
    var bounds = List[AABB[.WORLD]](capacity=count)
    for i in range(count):
        var x = Float32((i % 19) * 4 - 36)
        var y = Float32(((i / 19) % 14) * 4 - 26)
        var z = Float32((i * 7) % 11)
        var scale = Float32(2.2) if i % 17 == 0 else Float32(0.7)
        bounds.append(
            AABB(
                Point3f32[.WORLD](x - scale, y - scale, z),
                Point3f32[.WORLD](x + scale, y - scale, z),
                Point3f32[.WORLD](x, y + scale, z + 0.1),
            )
        )
    return bounds^


def _build_hploc_binary(
    mut ctx: DeviceContext,
    bounds: List[AABB[.WORLD]],
) raises -> _TestBinaryBuild:
    var flat = _flatten_bounds(bounds)
    var payloads = _identity_ids(len(bounds))
    var workspace = GpuBinaryBuildWorkspace(
        ctx, SegmentOffsets.single(len(bounds))
    )
    var binary = GpuBinaryBoundsBvh(
        ctx, upload_list(ctx, flat), upload_list(ctx, payloads), workspace
    )
    _ = build_binary_bvh[.HPLOC](ctx, binary, workspace)
    return _TestBinaryBuild(binary^, workspace^)


def _build_lbvh_binary(
    mut ctx: DeviceContext,
    bounds: List[AABB[.WORLD]],
) raises -> _TestBinaryBuild:
    var flat = _flatten_bounds(bounds)
    var payloads = _identity_ids(len(bounds))
    var workspace = GpuBinaryBuildWorkspace(
        ctx, SegmentOffsets.single(len(bounds))
    )
    var binary = GpuBinaryBoundsBvh(
        ctx, upload_list(ctx, flat), upload_list(ctx, payloads), workspace
    )
    _ = build_binary_bvh[.LBVH](ctx, binary, workspace)
    return _TestBinaryBuild(binary^, workspace^)


def _reference_from_binary(
    build: _TestBinaryBuild,
    bounds: List[AABB[.WORLD]],
) raises -> HplocTopology[.WORLD]:
    ref binary = build.binary
    var codes = List[UInt32](capacity=binary.leaf_count)
    var ids = List[UInt32](capacity=binary.leaf_count)
    with build.workspace.topology.value().morton_keys.map_to_host() as host_codes:
        for i in range(binary.leaf_count):
            codes.append(host_codes[i])
    with binary.leaf_ids.map_to_host() as host_ids:
        for i in range(binary.leaf_count):
            ids.append(host_ids[i])
    return build_hploc_topology(bounds, codes, ids)


def test_hploc_binary_layout_one_leaf() raises:
    var bounds: List[AABB[.WORLD]] = [
        AABB(
            Point3f32[.WORLD](0.0, 0.0, 0.0),
            Point3f32[.WORLD](1.0, 0.0, 0.0),
            Point3f32[.WORLD](0.0, 1.0, 0.0),
        )
    ]
    with DeviceContext() as ctx:
        var build = _build_hploc_binary(ctx, bounds)
        var reference = _reference_from_binary(build, bounds)
        _assert_binary_matches_reference(build, reference)


def test_hploc_binary_layout_cross_block_matches_reference() raises:
    var bounds = _make_triangles(257)
    with DeviceContext() as ctx:
        var build = _build_hploc_binary(ctx, bounds)
        var reference = _reference_from_binary(build, bounds)
        _assert_binary_matches_reference(build, reference)


def test_lbvh_fuses_topology_and_bounds() raises:
    var bounds = _make_triangles(257)
    with DeviceContext() as ctx:
        var build = _build_lbvh_binary(ctx, bounds)
        ctx.synchronize()
        var validation = validate_binary_bvh(
            build.binary, build.workspace, build.binary.root_bounds()
        )
        assert_true(validation.sorted_ok)
        assert_true(validation.values_ok)
        assert_true(validation.topology_ok)
        assert_true(validation.bounds_ok)
        with build.binary.roots.map_to_host() as roots:
            assert_false(is_leaf_ref(roots[0]))
            assert_equal(
                decode_ref_index(roots[0]), validation.topology_root_idx
            )


def test_lbvh_duplicate_morton_codes() raises:
    var bounds = List[AABB[.WORLD]](capacity=65)
    for i in range(65):
        var scale = Float32(i + 1) * 0.01
        bounds.append(
            AABB(
                Point3f32[.WORLD](-scale, -scale, 0.0),
                Point3f32[.WORLD](scale, -scale, 0.0),
                Point3f32[.WORLD](0.0, scale, 0.1),
            )
        )
    with DeviceContext() as ctx:
        var build = _build_lbvh_binary(ctx, bounds)
        ctx.synchronize()
        var validation = validate_binary_bvh(
            build.binary, build.workspace, build.binary.root_bounds()
        )
        assert_true(validation.sorted_ok)
        assert_true(validation.topology_ok)
        assert_true(validation.bounds_ok)


def test_binary_layout_reduces_bounds_per_segment() raises:
    var bounds = _make_triangles(6)
    var flat = _flatten_bounds(bounds)
    var payloads = _identity_ids(len(bounds))
    var segments = SegmentOffsets.from_counts([2, 1, 3])

    with DeviceContext() as ctx:
        var workspace = GpuBinaryBuildWorkspace(ctx, segments)
        var binary = GpuBinaryBoundsBvh(
            ctx,
            upload_list(ctx, flat),
            upload_list(ctx, payloads),
            workspace,
        )
        ctx.synchronize()

        assert_equal(binary.segments.segment_count(), 3)
        assert_equal(binary.internal_count, 3)
        assert_equal(binary.internal_segments.begin(0), UInt32(0))
        assert_equal(binary.internal_segments.end(0), UInt32(1))
        assert_equal(binary.internal_segments.end(1), UInt32(1))
        assert_equal(binary.internal_segments.end(2), UInt32(3))

        for segment_idx in range(segments.segment_count()):
            var expected_bounds = AABB[.WORLD].invalid()
            var expected_centroids = AABB[.WORLD].invalid()
            for leaf_idx in range(
                Int(segments.begin(segment_idx)),
                Int(segments.end(segment_idx)),
            ):
                expected_bounds.grow(bounds[leaf_idx])
                expected_centroids.grow(bounds[leaf_idx].centroid())
            assert_true(
                _bounds_match(binary.root_bounds(segment_idx), expected_bounds)
            )
            assert_true(
                _bounds_match(
                    binary.centroid_bounds(segment_idx), expected_centroids
                )
            )


def test_segmented_morton_sort_groups_without_losing_precision() raises:
    var bounds = _make_triangles(9)
    var flat = _flatten_bounds(bounds)
    var payloads = _identity_ids(len(bounds))
    var segments = SegmentOffsets.from_counts([3, 1, 5])

    with DeviceContext() as ctx:
        var workspace = GpuBinaryBuildWorkspace(ctx, segments)
        var binary = GpuBinaryBoundsBvh(
            ctx,
            upload_list(ctx, flat),
            upload_list(ctx, payloads),
            workspace,
        )
        workspace.ensure_topology(ctx)
        enqueue_segmented_morton_codes(ctx, binary, workspace)
        enqueue_segmented_morton_sort(ctx, binary, workspace)
        ctx.synchronize()

        with binary.leaf_ids.map_to_host() as ids, workspace.topology.value().morton_keys.map_to_host() as codes:
            for segment_idx in range(segments.segment_count()):
                var begin = Int(segments.begin(segment_idx))
                var end = Int(segments.end(segment_idx))
                for sorted_idx in range(begin, end):
                    var leaf_idx = Int(ids[sorted_idx])
                    assert_true(leaf_idx >= begin and leaf_idx < end)
                    if sorted_idx > begin:
                        assert_true(codes[sorted_idx - 1] <= codes[sorted_idx])


def test_hploc_builds_independent_segment_topologies() raises:
    var bounds = _make_triangles(9)
    var flat = _flatten_bounds(bounds)
    var payloads = _identity_ids(len(bounds))
    var segments = SegmentOffsets.from_counts([3, 1, 5])

    with DeviceContext() as ctx:
        var workspace = GpuBinaryBuildWorkspace(ctx, segments)
        var binary = GpuBinaryBoundsBvh(
            ctx,
            upload_list(ctx, flat),
            upload_list(ctx, payloads),
            workspace,
        )
        var state = enqueue_binary_bvh_with_hploc(ctx, binary, workspace)
        ctx.synchronize()

        assert_equal(state.result_status(), UInt32(0))
        assert_equal(state.result_node_count(), UInt32(6))
        with binary.node_meta.map_to_host() as meta, workspace.topology.value().leaf_parent.map_to_host() as leaf_parent, binary.node_leaf_counts.map_to_host() as leaf_counts:
            for segment_idx in range(segments.segment_count()):
                var leaf_begin = Int(segments.begin(segment_idx))
                var leaf_end = Int(segments.end(segment_idx))
                var node_begin = Int(
                    binary.internal_segments.begin(segment_idx)
                )
                var node_end = Int(binary.internal_segments.end(segment_idx))
                var root = state.result_root(segment_idx)
                var root_idx = Int(decode_ref_index(root))
                if leaf_end - leaf_begin == 1:
                    assert_true(is_leaf_ref(root))
                    assert_equal(root_idx, leaf_begin)
                    assert_equal(leaf_parent[leaf_begin], LBVH_SENTINEL)
                    continue

                assert_false(is_leaf_ref(root))
                assert_true(root_idx >= node_begin and root_idx < node_end)
                assert_equal(
                    leaf_counts[root_idx], UInt32(leaf_end - leaf_begin)
                )

                for sorted_idx in range(leaf_begin, leaf_end):
                    var parent = Int(leaf_parent[sorted_idx])
                    assert_true(parent >= node_begin and parent < node_end)

                for node_idx in range(node_begin, node_end):
                    var base = node_idx * BinaryBvhNode.META_STRIDE
                    var parent = meta[base + BinaryBvhNode.PARENT]
                    if node_idx == root_idx:
                        assert_equal(parent, LBVH_SENTINEL)
                    else:
                        assert_true(
                            Int(parent) >= node_begin and Int(parent) < node_end
                        )
                    for child_slot in range(
                        BinaryBvhNode.LEFT, BinaryBvhNode.RIGHT + 1
                    ):
                        var child = meta[base + child_slot]
                        var child_idx = Int(decode_ref_index(child))
                        if is_leaf_ref(child):
                            assert_true(
                                child_idx >= leaf_begin and child_idx < leaf_end
                            )
                        else:
                            assert_true(
                                child_idx >= node_begin and child_idx < node_end
                            )


def test_lbvh_builds_independent_segment_topologies() raises:
    var bounds = _make_triangles(9)
    var segments = SegmentOffsets.from_counts([0, 3, 1, 0, 5])
    with DeviceContext() as ctx:
        var workspace = GpuBinaryBuildWorkspace(ctx, segments)
        var binary = GpuBinaryBoundsBvh(
            ctx,
            upload_list(ctx, _flatten_bounds(bounds)),
            upload_list(ctx, _identity_ids(len(bounds))),
            workspace,
        )
        _ = build_binary_bvh[.LBVH](ctx, binary, workspace)
        ctx.synchronize()

        with binary.roots.map_to_host() as roots, binary.node_meta.map_to_host() as meta, binary.node_bounds.map_to_host() as node_bounds:
            with workspace.topology.value().leaf_parent.map_to_host() as leaf_parent, workspace.topology.value().node_flags.map_to_host() as node_flags:
                with binary.node_leaf_counts.map_to_host() as leaf_counts, binary.leaf_ids.map_to_host() as leaf_ids:
                    var node_bounds_span = Span(
                        unsafe_ptr=node_bounds.unsafe_ptr(),
                        length=len(node_bounds),
                    )
                    for segment_idx in range(segments.segment_count()):
                        var leaf_begin = Int(segments.begin(segment_idx))
                        var leaf_end = Int(segments.end(segment_idx))
                        var node_begin = Int(
                            binary.internal_segments.begin(segment_idx)
                        )
                        var node_end = Int(
                            binary.internal_segments.end(segment_idx)
                        )
                        var root = roots[segment_idx]
                        if leaf_end == leaf_begin:
                            assert_equal(root, LBVH_SENTINEL)
                            continue

                        var root_idx = Int(decode_ref_index(root))
                        for sorted_idx in range(leaf_begin, leaf_end):
                            var item_idx = Int(leaf_ids[sorted_idx])
                            assert_true(
                                item_idx >= leaf_begin and item_idx < leaf_end
                            )

                        if leaf_end - leaf_begin == 1:
                            assert_true(is_leaf_ref(root))
                            assert_equal(root_idx, leaf_begin)
                            assert_equal(
                                leaf_parent[leaf_begin], LBVH_SENTINEL
                            )
                            continue

                        assert_false(is_leaf_ref(root))
                        assert_true(
                            root_idx >= node_begin and root_idx < node_end
                        )
                        assert_equal(
                            meta[root_idx * BinaryBvhNode.META_STRIDE],
                            LBVH_SENTINEL,
                        )
                        assert_equal(
                            leaf_counts[root_idx],
                            UInt32(leaf_end - leaf_begin),
                        )
                        var root_bounds = AABB[.WORLD].merge(
                            AABB[.WORLD].load6(
                                node_bounds_span,
                                root_idx * BinaryBvhNode.BOUNDS_STRIDE,
                            ),
                            AABB[.WORLD].load6(
                                node_bounds_span,
                                root_idx * BinaryBvhNode.BOUNDS_STRIDE
                                + AABB.STRIDE,
                            ),
                        )
                        assert_true(
                            _bounds_match(
                                root_bounds,
                                binary.root_bounds(segment_idx),
                            )
                        )
                        for sorted_idx in range(leaf_begin, leaf_end):
                            var parent = Int(leaf_parent[sorted_idx])
                            assert_true(
                                parent >= node_begin and parent < node_end
                            )
                        for node_idx in range(node_begin, node_end):
                            assert_equal(node_flags[node_idx], UInt32(2))
                            var base = (
                                node_idx * BinaryBvhNode.META_STRIDE
                            )
                            var parent = meta[base + BinaryBvhNode.PARENT]
                            if node_idx != root_idx:
                                assert_true(
                                    Int(parent) >= node_begin
                                    and Int(parent) < node_end
                                )
                            for child_slot in range(
                                BinaryBvhNode.LEFT,
                                BinaryBvhNode.RIGHT + 1,
                            ):
                                var child = meta[base + child_slot]
                                var child_idx = Int(decode_ref_index(child))
                                if is_leaf_ref(child):
                                    assert_true(
                                        child_idx >= leaf_begin
                                        and child_idx < leaf_end
                                    )
                                else:
                                    assert_true(
                                        child_idx >= node_begin
                                        and child_idx < node_end
                                    )


def test_hploc_binary_selector_default_remains_lbvh_and_quality_improves() raises:
    var bounds = _make_triangles(64)
    var flat = _flatten_bounds(bounds)
    var payloads = _identity_ids(len(bounds))

    with DeviceContext() as ctx:
        var hploc = _build_hploc_binary(ctx, bounds)
        var reference = _reference_from_binary(hploc, bounds)
        _assert_binary_matches_reference(hploc, reference)

        var workspace = GpuBinaryBuildWorkspace(
            ctx, SegmentOffsets.single(len(bounds))
        )
        var lbvh = GpuBinaryBoundsBvh(
            ctx,
            upload_list(ctx, flat),
            upload_list(ctx, payloads),
            workspace,
        )
        _ = build_binary_bvh(ctx, lbvh, workspace)
        ctx.synchronize()
        var validation = validate_binary_bvh(
            lbvh, workspace, lbvh.root_bounds()
        )
        assert_true(validation.sorted_ok)
        assert_true(validation.topology_ok)
        assert_true(validation.bounds_ok)
        assert_true(
            measure_binary_bvh_quality(hploc.binary).quality
            <= measure_binary_bvh_quality(lbvh).quality + 1.0e-5
        )


def main() raises:
    comptime if not has_accelerator():
        raise "No Accelerator found"
    TestSuite.discover_tests[__functions_in_module()]().run()
