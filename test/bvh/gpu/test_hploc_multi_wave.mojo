from std.math import abs, max
from std.sys import has_accelerator
from std.testing import TestSuite, assert_equal, assert_false, assert_true
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh.constants import (
    BinaryBvhNode,
    LBVH_SENTINEL,
)
from bajo.bvh.tagged_ref import decode_ref_index, encode_leaf_ref, is_leaf_ref
from bajo.bvh.gpu.diagnostics import build_bounds_bvh_for_diagnostics
from bajo.bvh.gpu.builder.hploc_layout import (
    HPLOC_MERGING_THRESHOLD,
    HPLOC_SEARCH_RADIUS,
    HPLOC_STATUS_OK,
)
from bajo.bvh.gpu.builder.hploc_multi_wave import GpuHplocBuildState
from bajo.bvh.cpu.builder.hploc import (
    HplocTopology,
    build_hploc_topology,
)
from bajo.bvh.gpu.utils import upload_list
from bajo.core import AABB, Point3f32


struct GpuHplocMultiWaveBvh[
    search_radius: Int = HPLOC_SEARCH_RADIUS,
    merging_threshold: Int = HPLOC_MERGING_THRESHOLD,
]:
    """Test fixture owning the direct-layout H-PLOC buffers."""

    var leaf_count: Int
    var leaf_bounds: DeviceBuffer[.float32]
    var sorted_morton_codes: DeviceBuffer[.uint32]
    var sorted_leaf_ids: DeviceBuffer[.uint32]
    var node_meta: DeviceBuffer[.uint32]
    var leaf_parent: DeviceBuffer[.uint32]
    var node_bounds: DeviceBuffer[.float32]
    var node_flags: DeviceBuffer[.uint32]
    var node_leaf_counts: DeviceBuffer[.uint32]
    var state: GpuHplocBuildState[Self.search_radius, Self.merging_threshold]

    def __init__(
        out self,
        mut ctx: DeviceContext,
        leaf_bounds: DeviceBuffer[.float32],
        sorted_morton_codes: DeviceBuffer[.uint32],
        sorted_leaf_ids: DeviceBuffer[.uint32],
    ) raises:
        self.leaf_count = len(sorted_leaf_ids)
        self.leaf_bounds = leaf_bounds
        self.sorted_morton_codes = sorted_morton_codes
        self.sorted_leaf_ids = sorted_leaf_ids

        var internal_capacity = max(self.leaf_count - 1, 1)
        self.node_meta = ctx.enqueue_create_buffer[.uint32](
            internal_capacity * BinaryBvhNode.META_STRIDE
        )
        self.leaf_parent = ctx.enqueue_create_buffer[.uint32](self.leaf_count)
        self.node_bounds = ctx.enqueue_create_buffer[.float32](
            internal_capacity * BinaryBvhNode.BOUNDS_STRIDE
        )
        self.node_flags = ctx.enqueue_create_buffer[.uint32](internal_capacity)
        self.node_leaf_counts = ctx.enqueue_create_buffer[.uint32](
            internal_capacity
        )
        self.state = GpuHplocBuildState[
            Self.search_radius, Self.merging_threshold
        ](
            ctx,
            self.leaf_bounds.copy(),
            self.sorted_morton_codes.copy(),
            self.sorted_leaf_ids.copy(),
            self.node_meta.copy(),
            self.leaf_parent.copy(),
            self.node_bounds.copy(),
            self.node_flags.copy(),
            self.node_leaf_counts.copy(),
        )

    def result_status(self) raises -> UInt32:
        return self.state.result_status()

    def result_root(self) raises -> UInt32:
        return self.state.result_root()

    def result_node_count(self) raises -> UInt32:
        return self.state.result_node_count()


def _box(center_x: Float32) -> AABB[.WORLD]:
    return AABB[.WORLD](
        Point3f32[.WORLD](center_x - 0.1, -0.1, -0.1),
        Point3f32[.WORLD](center_x + 0.1, 0.1, 0.1),
    )


def _identity_ids(count: Int) -> List[UInt32]:
    var ids = List[UInt32](capacity=count)
    for i in range(count):
        ids.append(UInt32(i))
    return ids^


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
    actual: AABB[.WORLD],
    expected: AABB[.WORLD],
    tolerance: Float64 = 1.0e-4,
) -> Bool:
    return (
        abs(Float64(actual._min.x - expected._min.x)) <= tolerance
        and abs(Float64(actual._min.y - expected._min.y)) <= tolerance
        and abs(Float64(actual._min.z - expected._min.z)) <= tolerance
        and abs(Float64(actual._max.x - expected._max.x)) <= tolerance
        and abs(Float64(actual._max.y - expected._max.y)) <= tolerance
        and abs(Float64(actual._max.z - expected._max.z)) <= tolerance
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
    for node_idx, node in enumerate(reference.nodes):
        if node.left == LBVH_SENTINEL:
            hashes[node_idx] = _leaf_hash(node.leaf_id)
        else:
            hashes[node_idx] = _inner_hash(
                hashes[Int(node.left)], hashes[Int(node.right)]
            )
    return hashes[Int(reference.root)]


def _assert_gpu_matches_reference[
    search_radius: Int,
    merging_threshold: Int,
](
    gpu: GpuHplocMultiWaveBvh[search_radius, merging_threshold],
    reference: HplocTopology[.WORLD],
) raises -> UInt64:
    var internal_count = max(reference.leaf_count - 1, 0)
    assert_equal(gpu.result_status(), UInt32(HPLOC_STATUS_OK))
    assert_equal(gpu.result_node_count(), UInt32(internal_count))
    var root = gpu.result_root()
    if reference.leaf_count == 1:
        assert_equal(root, encode_leaf_ref(UInt32(0)))
        with gpu.sorted_leaf_ids.map_to_host() as leaf_ids:
            var result = _leaf_hash(leaf_ids[0])
            assert_equal(result, _reference_root_hash(reference))
            return result

    assert_true(root < UInt32(internal_count))
    with gpu.node_meta.map_to_host() as meta, gpu.leaf_parent.map_to_host() as leaf_parent, gpu.node_bounds.map_to_host() as flat_bounds, gpu.node_flags.map_to_host() as flags, gpu.node_leaf_counts.map_to_host() as leaf_counts, gpu.sorted_leaf_ids.map_to_host() as leaf_ids, gpu.leaf_bounds.map_to_host() as leaf_bounds:
        var hashes = List[UInt64](length=internal_count, fill=UInt64(0))
        var counts = List[UInt32](length=internal_count, fill=UInt32(0))
        var subtree_bounds = List[AABB[.WORLD]](
            length=internal_count, fill=AABB[.WORLD].invalid()
        )
        var seen_leaves = List[Bool](length=reference.leaf_count, fill=False)
        var node_area_sum = Float64(0.0)
        var leaf_bounds_span = Span(
            unsafe_ptr=leaf_bounds.unsafe_ptr(), length=len(leaf_bounds)
        )
        var node_bounds_span = Span(
            unsafe_ptr=flat_bounds.unsafe_ptr(), length=len(flat_bounds)
        )

        for node_idx in range(internal_count):
            var base = node_idx * BinaryBvhNode.META_STRIDE
            var left = meta[base + BinaryBvhNode.LEFT]
            var right = meta[base + BinaryBvhNode.RIGHT]
            assert_equal(meta[base + BinaryBvhNode.FENCE], LBVH_SENTINEL)
            assert_equal(flags[node_idx], UInt32(2))

            var left_hash: UInt64
            var left_count: UInt32
            var left_bounds: AABB[.WORLD]
            if is_leaf_ref(left):
                var sorted_pos = Int(decode_ref_index(left))
                var leaf_id = leaf_ids[sorted_pos]
                assert_equal(leaf_parent[sorted_pos], UInt32(node_idx))
                assert_false(seen_leaves[Int(leaf_id)])
                seen_leaves[Int(leaf_id)] = True
                left_hash = _leaf_hash(leaf_id)
                left_count = UInt32(1)
                left_bounds = AABB[.WORLD].load6(
                    leaf_bounds_span, Int(leaf_id) * AABB.STRIDE
                )
                node_area_sum += Float64(left_bounds.surface_area()[0])
            else:
                var child = Int(decode_ref_index(left))
                assert_true(child < node_idx)
                assert_equal(
                    meta[
                        child * BinaryBvhNode.META_STRIDE + BinaryBvhNode.PARENT
                    ],
                    UInt32(node_idx),
                )
                left_hash = hashes[child]
                left_count = counts[child]
                left_bounds = subtree_bounds[child]

            var right_hash: UInt64
            var right_count: UInt32
            var right_bounds: AABB[.WORLD]
            if is_leaf_ref(right):
                var sorted_pos = Int(decode_ref_index(right))
                var leaf_id = leaf_ids[sorted_pos]
                assert_equal(leaf_parent[sorted_pos], UInt32(node_idx))
                assert_false(seen_leaves[Int(leaf_id)])
                seen_leaves[Int(leaf_id)] = True
                right_hash = _leaf_hash(leaf_id)
                right_count = UInt32(1)
                right_bounds = AABB[.WORLD].load6(
                    leaf_bounds_span, Int(leaf_id) * AABB.STRIDE
                )
                node_area_sum += Float64(right_bounds.surface_area()[0])
            else:
                var child = Int(decode_ref_index(right))
                assert_true(child < node_idx)
                assert_equal(
                    meta[
                        child * BinaryBvhNode.META_STRIDE + BinaryBvhNode.PARENT
                    ],
                    UInt32(node_idx),
                )
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
            node_area_sum += Float64(subtree_bounds[node_idx].surface_area()[0])
            assert_equal(leaf_counts[node_idx], counts[node_idx])

        assert_equal(
            meta[Int(root) * BinaryBvhNode.META_STRIDE + BinaryBvhNode.PARENT],
            LBVH_SENTINEL,
        )
        assert_equal(hashes[Int(root)], _reference_root_hash(reference))
        for seen in seen_leaves:
            assert_true(seen)

        var visited = List[Bool](length=internal_count, fill=False)
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

        assert_equal(visited_count, internal_count)
        var root_area = Float64(subtree_bounds[Int(root)].surface_area()[0])
        var gpu_quality = node_area_sum / root_area
        assert_true(abs(gpu_quality - reference.quality()) <= 1.0e-4)
        return hashes[Int(root)]


def test_hploc_multi_wave_one_leaf() raises:
    var bounds: List[AABB[.WORLD]] = [_box(2.0)]
    var flat = _flatten_bounds(bounds)
    var codes: List[UInt32] = [9]
    var ids: List[UInt32] = [0]
    var reference = build_hploc_topology(bounds, codes, ids)

    with DeviceContext() as ctx:
        var gpu = GpuHplocMultiWaveBvh[](
            ctx,
            upload_list(ctx, flat),
            upload_list(ctx, codes),
            upload_list(ctx, ids),
        )
        ctx.synchronize()
        _ = _assert_gpu_matches_reference(gpu, reference)


def test_hploc_multi_wave_strict_ties_match_reference() raises:
    comptime leaf_count = 4
    var bounds = List[AABB[.WORLD]](capacity=leaf_count)
    for _ in range(leaf_count):
        bounds.append(_box(0.0))
    var flat = _flatten_bounds(bounds)
    var codes: List[UInt32] = [7, 7, 7, 7]
    var ids = _identity_ids(leaf_count)
    var reference = build_hploc_topology(bounds, codes, ids)

    with DeviceContext() as ctx:
        var gpu = GpuHplocMultiWaveBvh[](
            ctx,
            upload_list(ctx, flat),
            upload_list(ctx, codes),
            upload_list(ctx, ids),
        )
        ctx.synchronize()
        _ = _assert_gpu_matches_reference(gpu, reference)


def test_hploc_multi_wave_crosses_waves_at_64_leaves() raises:
    comptime leaf_count = 64
    var bounds = List[AABB[.WORLD]](capacity=leaf_count)
    var codes = List[UInt32](capacity=leaf_count)
    for i in range(leaf_count):
        bounds.append(_box(Float32(i * 3)))
        codes.append(UInt32(i))
    var ids = _identity_ids(leaf_count)
    var reference = build_hploc_topology(bounds, codes, ids)
    var flat = _flatten_bounds(bounds)

    with DeviceContext() as ctx:
        var gpu = GpuHplocMultiWaveBvh[](
            ctx,
            upload_list(ctx, flat),
            upload_list(ctx, codes),
            upload_list(ctx, ids),
        )
        ctx.synchronize()
        _ = _assert_gpu_matches_reference(gpu, reference)


def test_hploc_multi_wave_parameterized_policy() raises:
    comptime leaf_count = 48
    comptime search_radius = 4
    comptime merging_threshold = 8
    var bounds = List[AABB[.WORLD]](capacity=leaf_count)
    var codes = List[UInt32](capacity=leaf_count)
    for i in range(leaf_count):
        bounds.append(_box(Float32(i * 3)))
        codes.append(UInt32(i))
    var ids = _identity_ids(leaf_count)
    var reference = build_hploc_topology(
        bounds,
        codes,
        ids,
        search_radius,
        merging_threshold,
    )
    var flat = _flatten_bounds(bounds)

    with DeviceContext() as ctx:
        var gpu = GpuHplocMultiWaveBvh[search_radius, merging_threshold](
            ctx,
            upload_list(ctx, flat),
            upload_list(ctx, codes),
            upload_list(ctx, ids),
        )
        ctx.synchronize()
        _ = _assert_gpu_matches_reference(gpu, reference)


def test_hploc_multi_wave_duplicate_codes_repeat_deterministically() raises:
    comptime leaf_count = 96
    var bounds = List[AABB[.WORLD]](capacity=leaf_count)
    var codes = List[UInt32](capacity=leaf_count)
    var ids = List[UInt32](capacity=leaf_count)
    for i in range(leaf_count):
        bounds.append(_box(Float32(i - 48)))
        codes.append(UInt32(11))
        ids.append(UInt32((i * 37) % leaf_count))
    var reference = build_hploc_topology(bounds, codes, ids)
    var flat = _flatten_bounds(bounds)

    with DeviceContext() as ctx:
        var first = GpuHplocMultiWaveBvh[](
            ctx,
            upload_list(ctx, flat),
            upload_list(ctx, codes),
            upload_list(ctx, ids),
        )
        var second = GpuHplocMultiWaveBvh[](
            ctx,
            upload_list(ctx, flat),
            upload_list(ctx, codes),
            upload_list(ctx, ids),
        )
        ctx.synchronize()
        var first_hash = _assert_gpu_matches_reference(first, reference)
        var second_hash = _assert_gpu_matches_reference(second, reference)
        assert_equal(first_hash, second_hash)


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


def test_hploc_multi_wave_crosses_blocks_with_gpu_lbvh_inputs() raises:
    comptime leaf_count = 257
    var bounds = _make_triangles(leaf_count)
    var flat = _flatten_bounds(bounds)
    var payloads = _identity_ids(leaf_count)

    with DeviceContext() as ctx:
        var device_bounds = upload_list(ctx, flat)
        var device_payloads = upload_list(ctx, payloads)
        var diagnostic = build_bounds_bvh_for_diagnostics[2, 2, 2](
            ctx, device_bounds, device_payloads
        )
        ref binary = diagnostic.build.binary
        ctx.synchronize()

        var codes = List[UInt32](capacity=leaf_count)
        var sorted_ids = List[UInt32](capacity=leaf_count)
        with diagnostic.build.workspace.topology.value().morton_keys.map_to_host() as host_codes:
            for i in range(leaf_count):
                codes.append(host_codes[i])
        with binary.leaf_ids.map_to_host() as host_ids:
            for i in range(leaf_count):
                sorted_ids.append(host_ids[i])

        var reference = build_hploc_topology(bounds, codes, sorted_ids)
        var gpu = GpuHplocMultiWaveBvh[](
            ctx,
            binary.leaf_bounds.copy(),
            diagnostic.build.workspace.topology.value().morton_keys.copy(),
            binary.leaf_ids.copy(),
        )
        ctx.synchronize()
        _ = _assert_gpu_matches_reference(gpu, reference)


def test_hploc_multi_wave_stress_4097_triangles() raises:
    comptime leaf_count = 4097
    var bounds = _make_triangles(leaf_count)
    var flat = _flatten_bounds(bounds)
    var payloads = _identity_ids(leaf_count)

    with DeviceContext() as ctx:
        var device_bounds = upload_list(ctx, flat)
        var device_payloads = upload_list(ctx, payloads)
        var diagnostic = build_bounds_bvh_for_diagnostics[2, 2, 2](
            ctx, device_bounds, device_payloads
        )
        ref binary = diagnostic.build.binary
        ctx.synchronize()

        var codes = List[UInt32](capacity=leaf_count)
        var sorted_ids = List[UInt32](capacity=leaf_count)
        with diagnostic.build.workspace.topology.value().morton_keys.map_to_host() as host_codes:
            for i in range(leaf_count):
                codes.append(host_codes[i])
        with binary.leaf_ids.map_to_host() as host_ids:
            for i in range(leaf_count):
                sorted_ids.append(host_ids[i])

        var reference = build_hploc_topology(bounds, codes, sorted_ids)
        var gpu = GpuHplocMultiWaveBvh[](
            ctx,
            binary.leaf_bounds.copy(),
            diagnostic.build.workspace.topology.value().morton_keys.copy(),
            binary.leaf_ids.copy(),
        )
        ctx.synchronize()
        _ = _assert_gpu_matches_reference(gpu, reference)


def main() raises:
    comptime if not has_accelerator():
        raise "No Accelerator found"
    TestSuite.discover_tests[__functions_in_module()]().run()
