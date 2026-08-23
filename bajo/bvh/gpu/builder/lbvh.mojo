from std.math import min, max
from std.time import perf_counter_ns
from std.atomic import Atomic
from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import global_idx

from bajo.core import AABB, Vec3f32, Frame
from bajo.core.morton import morton3, morton_common_prefix
from bajo.sort.gpu.radix_sort import device_radix_sort_pairs
from bajo.bvh.constants import (
    LBVH_SENTINEL,
    BinaryBvhNode,
    GPU_BOUNDS_BVH_BLOCK_SIZE,
)
from bajo.bvh.tagged_ref import encode_internal_ref, encode_leaf_ref
from bajo.bvh.gpu.utils import GpuBuildTimings, _device_span
from bajo.bvh.gpu.builder.binary_layout import (
    _node_parent_index,
    _node_left,
    _node_right,
    _write_child_bounds,
    _load_and_union_node_bounds,
    _segment_for_item,
    GpuBinaryBoundsBvh,
    GpuBinaryBuildWorkspace,
    init_empty_bounds_kernel,
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
    device_radix_sort_pairs[DType.uint64, DType.uint32](
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


def refit_lbvh_bounds_from_leaves_kernel(
    leaf_bounds: ImmSpan[Float32, ImmutAnyOrigin],
    leaf_ids: ImmSpan[UInt32, ImmutAnyOrigin],
    node_meta: ImmSpan[UInt32, ImmutAnyOrigin],
    leaf_parent: ImmSpan[UInt32, ImmutAnyOrigin],
    node_bounds: MutSpan[Float32, MutAnyOrigin],
    node_flags: MutSpan[UInt32, MutAnyOrigin],
):
    var leaf_count = len(leaf_ids)
    var leaf_idx = global_idx.x
    if leaf_idx >= leaf_count:
        return

    debug_assert["safe", _use_compiler_assume=True](
        leaf_idx < len(leaf_ids) and leaf_idx < len(leaf_parent),
        "LBVH leaf record is outside a device span",
    )
    var item_idx = UInt32(leaf_ids.unsafe_get(leaf_idx))
    var b = Int(item_idx) * AABB.STRIDE
    var bounds = AABB[.WORLD].load6(leaf_bounds, b)

    var current_encoded = encode_leaf_ref(UInt32(leaf_idx))
    var parent = UInt32(leaf_parent.unsafe_get(leaf_idx))

    while parent != LBVH_SENTINEL:
        var left = _node_left(node_meta, parent)
        var right = _node_right(node_meta, parent)

        var is_left = current_encoded == left
        var is_right = current_encoded == right
        if not is_left and not is_right:
            break

        _write_child_bounds(node_bounds, parent, is_left, bounds)

        var old = Atomic.fetch_add(
            node_flags.unsafe_ptr().unsafe_offset(Int(parent)), 1
        )
        if old == 0:
            break

        bounds = _load_and_union_node_bounds(node_bounds, parent)
        current_encoded = encode_internal_ref(parent)
        parent = UInt32(
            node_meta.unsafe_get(_node_parent_index(current_encoded))
        )


def _lbvh_find_range(
    morton_codes: ImmSpan[UInt32, _],
    i: Int,
) -> Tuple[Int, Int]:
    var d_next = _common_prefix(morton_codes, i, i + 1)
    var d_prev = _common_prefix(morton_codes, i, i - 1)

    var d = 1
    if d_next < d_prev:
        d = -1

    var delta_min = _common_prefix(morton_codes, i, i - d)

    var lmax = 2
    while _common_prefix(morton_codes, i, i + lmax * d) > delta_min:
        lmax <<= 1
        if lmax > len(morton_codes) * 2:
            break

    var l = 0
    var t = lmax >> 1
    while t > 0:
        if _common_prefix(morton_codes, i, i + (l + t) * d) > delta_min:
            l += t
        t >>= 1

    var j = i + l * d
    return (min(i, j), max(i, j))


def _lbvh_find_split(
    morton_codes: ImmSpan[UInt32, _],
    first: Int,
    last: Int,
) -> Int:
    var node_prefix = _common_prefix(morton_codes, first, last)

    var split = first
    var step = last - first

    while step > 1:
        step = (step + 1) >> 1
        var new_split = split + step

        if new_split < last:
            var split_prefix = _common_prefix(
                morton_codes,
                first,
                new_split,
            )

            if split_prefix > node_prefix:
                split = new_split

    return split


def _common_prefix(
    morton_codes: ImmSpan[UInt32, _],
    i: Int,
    j: Int,
) -> Int:
    if j < 0 or j >= len(morton_codes):
        return -1

    var a = UInt32(morton_codes.unsafe_get(i))
    var b = UInt32(morton_codes.unsafe_get(j))

    return morton_common_prefix(a, UInt32(i), b, UInt32(j))


def init_segmented_lbvh_roots_kernel(
    segment_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    internal_segment_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    roots: MutSpan[UInt32, MutAnyOrigin],
    leaf_parent: MutSpan[UInt32, MutAnyOrigin],
):
    var segment_idx = global_idx.x
    if segment_idx >= len(roots):
        return
    var leaf_begin = segment_offsets.unsafe_get(segment_idx)
    var leaf_end = segment_offsets.unsafe_get(segment_idx + 1)
    if leaf_end == leaf_begin:
        roots.unsafe_get(segment_idx) = LBVH_SENTINEL
    elif leaf_end - leaf_begin == UInt32(1):
        roots.unsafe_get(segment_idx) = encode_leaf_ref(leaf_begin)
        leaf_parent.unsafe_get(Int(leaf_begin)) = LBVH_SENTINEL
    else:
        roots.unsafe_get(segment_idx) = encode_internal_ref(
            internal_segment_offsets.unsafe_get(segment_idx)
        )


def build_lbvh_topology_kernel(
    sorted_morton_codes: ImmSpan[UInt32, ImmutAnyOrigin],
    segment_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    internal_segment_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    node_meta: MutSpan[UInt32, MutAnyOrigin],
    leaf_parent: MutSpan[UInt32, MutAnyOrigin],
    node_bounds: MutSpan[Float32, MutAnyOrigin],
    node_flags: MutSpan[UInt32, MutAnyOrigin],
    node_leaf_counts: MutSpan[UInt32, MutAnyOrigin],
):
    var i = global_idx.x
    var internal_count = Int(
        internal_segment_offsets.unsafe_get(len(internal_segment_offsets) - 1)
    )
    if i >= internal_count:
        return

    var segment_idx = _segment_for_item(internal_segment_offsets, i)
    var internal_begin = Int(internal_segment_offsets.unsafe_get(segment_idx))
    var leaf_begin = Int(segment_offsets.unsafe_get(segment_idx))
    var leaf_end = Int(segment_offsets.unsafe_get(segment_idx + 1))
    var leaf_count = leaf_end - leaf_begin
    var local_i = i - internal_begin
    var morton_codes = Span(
        unsafe_ptr=sorted_morton_codes.unsafe_ptr().unsafe_offset(leaf_begin),
        length=leaf_count,
    )

    debug_assert["safe", _use_compiler_assume=True](
        i < len(node_flags) and i < len(node_leaf_counts),
        "LBVH topology record is outside a device span",
    )
    node_flags.unsafe_get(i) = UInt32(0)

    var invalid = AABB[.WORLD].invalid()
    var bounds_base = i * BinaryBvhNode.BOUNDS_STRIDE
    invalid.store6(node_bounds, bounds_base)
    invalid.store6(node_bounds, bounds_base + AABB.STRIDE)

    var first, last = _lbvh_find_range(morton_codes, local_i)
    node_leaf_counts.unsafe_get(i) = UInt32(last - first + 1)

    # only root parent is sentinel
    if first == 0 and last == leaf_count - 1:
        node_meta.unsafe_get(_node_parent_index(UInt32(i))) = LBVH_SENTINEL

    var split = _lbvh_find_split(morton_codes, first, last)

    var left_encoded = encode_internal_ref(UInt32(internal_begin + split))
    if split == first:
        left_encoded = encode_leaf_ref(UInt32(leaf_begin + split))
        leaf_parent.unsafe_get(leaf_begin + split) = UInt32(i)
    else:
        node_meta.unsafe_get(
            _node_parent_index(UInt32(internal_begin + split))
        ) = UInt32(i)

    var right_child = split + 1
    var right_encoded = encode_internal_ref(
        UInt32(internal_begin + right_child)
    )
    if right_child == last:
        right_encoded = encode_leaf_ref(UInt32(leaf_begin + right_child))
        leaf_parent.unsafe_get(leaf_begin + right_child) = UInt32(i)
    else:
        node_meta.unsafe_get(
            _node_parent_index(UInt32(internal_begin + right_child))
        ) = UInt32(i)

    var base = i * BinaryBvhNode.META_STRIDE
    node_meta.unsafe_get(base + BinaryBvhNode.LEFT) = left_encoded
    node_meta.unsafe_get(base + BinaryBvhNode.RIGHT) = right_encoded
    node_meta.unsafe_get(base + BinaryBvhNode.FENCE) = UInt32(leaf_begin + last)


def build_binary_bvh_with_lbvh(
    ctx: DeviceContext,
    mut binary: GpuBinaryBoundsBvh,
    mut workspace: GpuBinaryBuildWorkspace,
    measure_stages: Bool = False,
) raises -> GpuBuildTimings:
    """Builds the temporary sorted binary LBVH.

    This stage owns:
    1. leaf AABBs
    2. Morton keys
    3. sorted leaf ids
    4. binary topology
    5. refit bounds
    """
    var timings = GpuBuildTimings(0, 0, 0, 0, 0, 0, 0)
    var stage_start = Int(0)
    ref topology = workspace.topology.value()

    if measure_stages:
        ctx.synchronize()
        stage_start = perf_counter_ns()

    # leaf AABB
    # for now: inside binary_bvh __init__

    # morton codes

    enqueue_segmented_morton_codes(ctx, binary, workspace)
    if measure_stages:
        ctx.synchronize()
        var stage_end = perf_counter_ns()
        timings.morton_ns = Int(stage_end - stage_start)
        stage_start = stage_end

    # sort by morton codes
    enqueue_segmented_morton_sort(ctx, binary, workspace)

    if measure_stages:
        ctx.synchronize()
        var stage_end = perf_counter_ns()
        timings.sort_ns = Int(stage_end - stage_start)
        stage_start = stage_end

    ctx.enqueue_function[init_segmented_lbvh_roots_kernel](
        _device_span[mut=False](binary.segment_offsets),
        _device_span[mut=False](binary.internal_segment_offsets),
        _device_span[mut=True](binary.roots),
        _device_span[mut=True](topology.leaf_parent),
        grid_dim=(
            binary.segments.segment_count() + GPU_BOUNDS_BVH_BLOCK_SIZE - 1
        )
        // GPU_BOUNDS_BVH_BLOCK_SIZE,
        block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
    )

    # merge nodes
    if binary.internal_count > 0:
        ctx.enqueue_function[build_lbvh_topology_kernel](
            _device_span[mut=False](topology.morton_keys),
            _device_span[mut=False](binary.segment_offsets),
            _device_span[mut=False](binary.internal_segment_offsets),
            _device_span[mut=True](binary.node_meta),
            _device_span[mut=True](topology.leaf_parent),
            _device_span[mut=True](binary.node_bounds),
            _device_span[mut=True](topology.node_flags),
            _device_span[mut=True](binary.node_leaf_counts),
            grid_dim=binary.blocks_internal(),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
    if measure_stages:
        ctx.synchronize()
        var stage_end = perf_counter_ns()
        timings.topology_ns = Int(stage_end - stage_start)
        stage_start = stage_end

    # compute aabb over merged nodes
    if binary.internal_count > 0:
        ctx.enqueue_function[refit_lbvh_bounds_from_leaves_kernel](
            _device_span[mut=False](binary.leaf_bounds),
            _device_span[mut=False](binary.leaf_ids),
            _device_span[mut=False](binary.node_meta),
            _device_span[mut=False](topology.leaf_parent),
            _device_span[mut=True](binary.node_bounds),
            _device_span[mut=True](topology.node_flags),
            grid_dim=binary.blocks_leaves(),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
    if measure_stages:
        ctx.synchronize()
        timings.refit_ns = Int(perf_counter_ns() - stage_start)

    return timings
