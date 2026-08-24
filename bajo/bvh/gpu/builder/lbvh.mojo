"""Apetrei 2014 leaf-driven agglomerative LBVH builder.
source: https://diglib.eg.org/items/3aca7692-f2be-4b5d-a7f0-b7a865be6e5b.
"""

from std.math import max
from std.time import perf_counter_ns
from std.atomic import Atomic, Ordering, fence
from max.gpu.host import DeviceContext
from std.gpu import global_idx

from bajo.core import AABB
from bajo.core.morton import morton3, morton_key_delta
from bajo.sort.gpu.radix_sort import device_radix_sort_pairs
from bajo.bvh.constants import (
    LBVH_SENTINEL,
    BinaryBvhNode,
    GPU_BOUNDS_BVH_BLOCK_SIZE,
)
from bajo.bvh.tagged_ref import (
    decode_ref_index,
    encode_internal_ref,
    encode_leaf_ref,
    is_leaf_ref,
)
from bajo.bvh.gpu.utils import GpuBuildTimings, _device_span
from bajo.bvh.gpu.builder.binary_layout import (
    _node_parent_index,
    _write_child_bounds,
    _load_and_union_node_bounds,
    _segment_for_item,
    GpuBinaryBoundsBvh,
    GpuBinaryBuildWorkspace,
)


def compute_bounds_morton_codes_kernel(
    leaf_bounds: ImmSpan[Float32, ImmutAnyOrigin],
    bounds_device: ImmSpan[Float32, ImmutAnyOrigin],
    segment_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    composite_keys: MutSpan[UInt64, MutAnyOrigin],
    values: MutSpan[UInt32, MutAnyOrigin],
):
    var leaf_count = len(values)
    var i = global_idx.x
    if i >= leaf_count:
        return

    debug_assert["safe", _use_compiler_assume=True](
        i < len(composite_keys) and i < len(values),
        "Morton output is outside a device span",
    )
    var segment_idx = _segment_for_item(segment_offsets, i)
    var centroid_bounds = AABB[.WORLD].load6(
        bounds_device,
        segment_idx * 2 * AABB.STRIDE + AABB.STRIDE,
    )
    var cmin = centroid_bounds._min
    var inv_extent = centroid_bounds.extent().safe_inv()

    var b = i * AABB.STRIDE
    var bounds = AABB[.WORLD].load6(leaf_bounds, b)
    var c = (bounds.centroid() - cmin) * inv_extent

    var morton_code = morton3(c.x, c.y, c.z)
    composite_keys.unsafe_get(i) = (UInt64(UInt32(segment_idx)) << 32) | UInt64(
        morton_code
    )
    values.unsafe_get(i) = UInt32(i)


def gather_sorted_morton_codes_kernel(
    sorted_composite_keys: ImmSpan[UInt64, ImmutAnyOrigin],
    sorted_morton_codes: MutSpan[UInt32, MutAnyOrigin],
):
    var sorted_idx = global_idx.x
    if sorted_idx >= len(sorted_composite_keys):
        return
    sorted_morton_codes.unsafe_get(sorted_idx) = UInt32(
        sorted_composite_keys.unsafe_get(sorted_idx) & UInt64(0xFFFFFFFF)
    )


def enqueue_segmented_morton_codes(
    ctx: DeviceContext,
    mut binary: GpuBinaryBoundsBvh,
    mut workspace: GpuBinaryBuildWorkspace,
) raises:
    ref topology = workspace.topology.value()
    ctx.enqueue_function[compute_bounds_morton_codes_kernel](
        _device_span[mut=False](binary.leaf_bounds),
        _device_span[mut=False](binary.bounds_device),
        _device_span[mut=False](binary.segment_offsets),
        _device_span[mut=True](topology.sort_keys),
        _device_span[mut=True](binary.leaf_ids),
        grid_dim=binary.blocks_leaves(),
        block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
    )


def enqueue_segmented_morton_sort(
    ctx: DeviceContext,
    mut binary: GpuBinaryBoundsBvh,
    mut workspace: GpuBinaryBuildWorkspace,
) raises:
    """Sort one UInt64 ``(segment_id, morton_code)`` composite key."""
    ref topology = workspace.topology.value()
    device_radix_sort_pairs[.uint64, .uint32](
        ctx,
        workspace.sort,
        topology.sort_keys,
        binary.leaf_ids,
        binary.leaf_count,
    )
    ctx.enqueue_function[gather_sorted_morton_codes_kernel](
        _device_span[mut=False](topology.sort_keys),
        _device_span[mut=True](topology.morton_keys),
        grid_dim=binary.blocks_leaves(),
        block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
    )


@always_inline
def _lbvh_delta(
    sorted_morton_codes: ImmSpan[UInt32, _],
    boundary: Int,
    segment_begin: Int,
    segment_end: Int,
) -> UInt64:
    """XOR distance across one boundary, or infinity outside the segment."""

    if boundary < segment_begin or boundary + 1 >= segment_end:
        return UInt64.MAX
    return morton_key_delta(
        sorted_morton_codes.unsafe_get(boundary),
        UInt32(boundary),
        sorted_morton_codes.unsafe_get(boundary + 1),
        UInt32(boundary + 1),
    )


@always_inline
def _lbvh_atomic_exchange(
    ptr: MutPointer[UInt32, _], desired: UInt32
) -> UInt32:
    var expected = Atomic.load[ordering=Ordering.ACQUIRE](ptr)
    while not Atomic.compare_exchange[
        success_ordering=Ordering.ACQUIRE_RELEASE,
        failure_ordering=Ordering.ACQUIRE,
    ](ptr, expected, desired):
        pass
    return expected


@always_inline
def _lbvh_set_parent(
    encoded: UInt32,
    parent: UInt32,
    node_meta: MutSpan[UInt32, _],
    leaf_parent: MutSpan[UInt32, _],
):
    var child_idx = decode_ref_index(encoded)
    if is_leaf_ref(encoded):
        leaf_parent.unsafe_get(Int(child_idx)) = parent
    else:
        node_meta.unsafe_get(_node_parent_index(child_idx)) = parent


def init_segmented_lbvh_kernel(
    segment_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    node_flags: MutSpan[UInt32, MutAnyOrigin],
    roots: MutSpan[UInt32, MutAnyOrigin],
    leaf_parent: MutSpan[UInt32, MutAnyOrigin],
):
    var i = global_idx.x
    if i < len(node_flags):
        node_flags.unsafe_get(i) = LBVH_SENTINEL

    if i < len(roots):
        var leaf_begin = segment_offsets.unsafe_get(i)
        var leaf_end = segment_offsets.unsafe_get(i + 1)
        if leaf_end == leaf_begin:
            roots.unsafe_get(i) = LBVH_SENTINEL
        elif leaf_end - leaf_begin == UInt32(1):
            roots.unsafe_get(i) = encode_leaf_ref(leaf_begin)
            leaf_parent.unsafe_get(Int(leaf_begin)) = LBVH_SENTINEL
        else:
            roots.unsafe_get(i) = LBVH_SENTINEL


def build_lbvh_topology_and_bounds_kernel(
    leaf_bounds: ImmSpan[Float32, ImmutAnyOrigin],
    sorted_leaf_ids: ImmSpan[UInt32, ImmutAnyOrigin],
    sorted_morton_codes: ImmSpan[UInt32, ImmutAnyOrigin],
    segment_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    internal_segment_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    node_meta: MutSpan[UInt32, MutAnyOrigin],
    leaf_parent: MutSpan[UInt32, MutAnyOrigin],
    node_bounds: MutSpan[Float32, MutAnyOrigin],
    arrival_slots: MutSpan[UInt32, MutAnyOrigin],
    node_leaf_counts: MutSpan[UInt32, MutAnyOrigin],
    roots: MutSpan[UInt32, MutAnyOrigin],
):
    """Apetrei topology construction and AABB propagation in one kernel."""

    var sorted_idx = global_idx.x
    if sorted_idx >= len(sorted_leaf_ids):
        return

    var segment_idx = _segment_for_item(segment_offsets, sorted_idx)
    var leaf_begin = Int(segment_offsets.unsafe_get(segment_idx))
    var leaf_end = Int(segment_offsets.unsafe_get(segment_idx + 1))
    if leaf_end - leaf_begin <= 1:
        return
    var internal_begin = Int(internal_segment_offsets.unsafe_get(segment_idx))

    var item_idx = Int(sorted_leaf_ids.unsafe_get(sorted_idx))
    var bounds = AABB[.WORLD].load6(leaf_bounds, item_idx * AABB.STRIDE)
    var current_encoded = encode_leaf_ref(UInt32(sorted_idx))
    var range_left = sorted_idx
    var range_right = sorted_idx

    while True:
        var delta_left = _lbvh_delta(
            sorted_morton_codes, range_left - 1, leaf_begin, leaf_end
        )
        var delta_right = _lbvh_delta(
            sorted_morton_codes, range_right, leaf_begin, leaf_end
        )
        var attach_on_left = delta_right < delta_left
        var parent_boundary = range_right if attach_on_left else range_left - 1
        var parent = UInt32(internal_begin + parent_boundary - leaf_begin)
        var parent_base = Int(parent) * BinaryBvhNode.META_STRIDE

        if attach_on_left:
            node_meta.unsafe_get(
                parent_base + BinaryBvhNode.LEFT
            ) = current_encoded
        else:
            node_meta.unsafe_get(
                parent_base + BinaryBvhNode.RIGHT
            ) = current_encoded
        _write_child_bounds(node_bounds, parent, attach_on_left, bounds)
        _lbvh_set_parent(current_encoded, parent, node_meta, leaf_parent)

        fence[ordering=Ordering.SEQUENTIAL]()
        var endpoint = UInt32(range_left if attach_on_left else range_right)
        var other_endpoint = _lbvh_atomic_exchange(
            arrival_slots.unsafe_ptr().unsafe_offset(Int(parent)), endpoint
        )
        if other_endpoint == LBVH_SENTINEL:
            return

        if attach_on_left:
            range_right = Int(other_endpoint)
        else:
            range_left = Int(other_endpoint)

        node_leaf_counts.unsafe_get(Int(parent)) = UInt32(
            range_right - range_left + 1
        )
        node_meta.unsafe_get(parent_base + BinaryBvhNode.FENCE) = UInt32(
            range_right
        )
        arrival_slots.unsafe_get(Int(parent)) = UInt32(2)
        bounds = _load_and_union_node_bounds(node_bounds, parent)
        current_encoded = encode_internal_ref(parent)

        if range_left == leaf_begin and range_right + 1 == leaf_end:
            node_meta.unsafe_get(_node_parent_index(parent)) = LBVH_SENTINEL
            roots.unsafe_get(segment_idx) = current_encoded
            return


def build_binary_bvh_with_lbvh(
    ctx: DeviceContext,
    mut binary: GpuBinaryBoundsBvh,
    mut workspace: GpuBinaryBuildWorkspace,
    measure_stages: Bool = False,
) raises -> GpuBuildTimings:
    """Build Apetrei's 2014 LBVH into the shared binary layout."""
    var timings = GpuBuildTimings(0, 0, 0, 0, 0, 0, 0)
    var stage_start = Int(0)
    ref topology = workspace.topology.value()

    if measure_stages:
        ctx.synchronize()
        stage_start = perf_counter_ns()

    enqueue_segmented_morton_codes(ctx, binary, workspace)
    if measure_stages:
        ctx.synchronize()
        var stage_end = perf_counter_ns()
        timings.morton_ns = Int(stage_end - stage_start)
        stage_start = stage_end

    enqueue_segmented_morton_sort(ctx, binary, workspace)

    if measure_stages:
        ctx.synchronize()
        var stage_end = perf_counter_ns()
        timings.sort_ns = Int(stage_end - stage_start)
        stage_start = stage_end

    var init_count = max(
        max(binary.internal_count, 1), binary.segments.segment_count()
    )
    ctx.enqueue_function[init_segmented_lbvh_kernel](
        _device_span[mut=False](binary.segment_offsets),
        _device_span[mut=True](topology.node_flags),
        _device_span[mut=True](binary.roots),
        _device_span[mut=True](topology.leaf_parent),
        grid_dim=(init_count + GPU_BOUNDS_BVH_BLOCK_SIZE - 1)
        // GPU_BOUNDS_BVH_BLOCK_SIZE,
        block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
    )

    if binary.internal_count > 0:
        ctx.enqueue_function[build_lbvh_topology_and_bounds_kernel](
            _device_span[mut=False](binary.leaf_bounds),
            _device_span[mut=False](binary.leaf_ids),
            _device_span[mut=False](topology.morton_keys),
            _device_span[mut=False](binary.segment_offsets),
            _device_span[mut=False](binary.internal_segment_offsets),
            _device_span[mut=True](binary.node_meta),
            _device_span[mut=True](topology.leaf_parent),
            _device_span[mut=True](binary.node_bounds),
            _device_span[mut=True](topology.node_flags),
            _device_span[mut=True](binary.node_leaf_counts),
            _device_span[mut=True](binary.roots),
            grid_dim=binary.blocks_leaves(),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
    if measure_stages:
        ctx.synchronize()
        var stage_end = perf_counter_ns()
        timings.topology_ns = Int(stage_end - stage_start)

    # Topology and refit are the same kernel; refit_ns intentionally stays 0.
    return timings
