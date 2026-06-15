from std.bit import count_leading_zeros
from std.math import min, max
from std.time import perf_counter_ns
from std.atomic import Atomic
from std.gpu import DeviceBuffer, DeviceContext, global_idx

from bajo.core import AABB, Vec3f32
from bajo.core.morton import morton3
from bajo.sort.gpu.radix_sort import device_radix_sort_pairs, RadixSortWorkspace
from bajo.bvh.constants import (
    LBVH_LEAF_FLAG,
    LBVH_SENTINEL,
    BinaryBvhNode,
    GPU_BOUNDS_BVH_BLOCK_SIZE,
)
from bajo.bvh.gpu.utils import GpuBuildTimings
from bajo.bvh.gpu.builder.binary_layout import (
    _node_parent_index,
    _node_left,
    _node_right,
    _write_child_bounds,
    _load_and_union_node_bounds,
    GpuBinaryBoundsBvh,
    init_empty_bounds_kernel,
)


def compute_bounds_morton_codes_kernel(
    leaf_bounds: UnsafePointer[Float32, ImmutAnyOrigin],
    bounds_device: UnsafePointer[Float32, ImmutAnyOrigin],
    morton_codes: UnsafePointer[UInt32, MutAnyOrigin],
    values: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_count: Int,
):
    var i = global_idx.x
    if i >= leaf_count:
        return

    var centroid_bounds = AABB.load6(bounds_device, AABB.STRIDE)
    var cmin = centroid_bounds._min
    var inv_extent = centroid_bounds.extent().safe_inv()

    var b = i * AABB.STRIDE
    var bounds = AABB.load6(leaf_bounds, b)
    var c = (bounds.centroid() - cmin) * inv_extent

    morton_codes[i] = morton3(c.x, c.y, c.z)
    values[i] = UInt32(i)


def refit_lbvh_bounds_from_leaves_kernel(
    leaf_bounds: UnsafePointer[Float32, MutAnyOrigin],
    leaf_ids: UnsafePointer[UInt32, MutAnyOrigin],
    node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_parent: UnsafePointer[UInt32, MutAnyOrigin],
    node_bounds: UnsafePointer[Float32, MutAnyOrigin],
    node_flags: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_count: Int,
):
    var leaf_idx = global_idx.x
    if leaf_idx >= leaf_count:
        return

    var item_idx = UInt32(leaf_ids[leaf_idx])
    var b = Int(item_idx) * AABB.STRIDE
    var bounds = AABB.load6(leaf_bounds, b)

    var current_encoded = UInt32(leaf_idx) | LBVH_LEAF_FLAG
    var parent = UInt32(leaf_parent[leaf_idx])

    while parent != LBVH_SENTINEL:
        var left = _node_left(node_meta, parent)
        var right = _node_right(node_meta, parent)

        var is_left = current_encoded == left
        var is_right = current_encoded == right
        if not is_left and not is_right:
            break

        _write_child_bounds(node_bounds, parent, is_left, bounds)

        var old = Atomic.fetch_add(node_flags + Int(parent), 1)
        if old == 0:
            break

        bounds = _load_and_union_node_bounds(node_bounds, parent)
        current_encoded = parent
        parent = UInt32(node_meta[_node_parent_index(current_encoded)])


def _lbvh_find_range[
    origin: ImmutOrigin
](
    morton_codes: UnsafePointer[UInt32, origin],
    i: Int,
    n: Int,
) -> Tuple[
    Int, Int
]:
    var d_next = _common_prefix(morton_codes, i, i + 1, n)
    var d_prev = _common_prefix(morton_codes, i, i - 1, n)

    var d = 1
    if d_next < d_prev:
        d = -1

    var delta_min = _common_prefix(morton_codes, i, i - d, n)

    var lmax = 2
    while _common_prefix(morton_codes, i, i + lmax * d, n) > delta_min:
        lmax <<= 1
        if lmax > n * 2:
            break

    var l = 0
    var t = lmax >> 1
    while t > 0:
        if _common_prefix(morton_codes, i, i + (l + t) * d, n) > delta_min:
            l += t
        t >>= 1

    var j = i + l * d
    return (min(i, j), max(i, j))


def _lbvh_find_split[
    origin: ImmutOrigin
](
    morton_codes: UnsafePointer[UInt32, origin],
    first: Int,
    last: Int,
    n: Int,
) -> Int:
    var node_prefix = _common_prefix(morton_codes, first, last, n)

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
                n,
            )

            if split_prefix > node_prefix:
                split = new_split

    return split


def _common_prefix(
    morton_codes: UnsafePointer[mut=False, UInt32, _],
    i: Int,
    j: Int,
    n: Int,
) -> Int:
    if j < 0 or j >= n:
        return -1

    var a = UInt32(morton_codes[i])
    var b = UInt32(morton_codes[j])

    if a != b:
        return Int(count_leading_zeros(a ^ b))

    # duplicate Morton codes are ordered by sorted position
    var x = UInt32(i) ^ UInt32(j)
    if x == 0:
        return 64
    return 32 + Int(count_leading_zeros(x))


def init_lbvh_topology_kernel(
    node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_parent: UnsafePointer[UInt32, MutAnyOrigin],
    internal_count: Int,
    leaf_count: Int,
):
    var i = global_idx.x

    if i < internal_count:
        var base = i * BinaryBvhNode.META_STRIDE
        node_meta[base + BinaryBvhNode.PARENT] = LBVH_SENTINEL
        node_meta[base + BinaryBvhNode.LEFT] = 0
        node_meta[base + BinaryBvhNode.RIGHT] = 0
        node_meta[base + BinaryBvhNode.FENCE] = 0

    if i < leaf_count:
        leaf_parent[i] = LBVH_SENTINEL


def build_lbvh_topology_kernel(
    sorted_morton_codes: UnsafePointer[UInt32, ImmutAnyOrigin],
    node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_parent: UnsafePointer[UInt32, MutAnyOrigin],
    node_bounds: UnsafePointer[Float32, MutAnyOrigin],
    node_flags: UnsafePointer[UInt32, MutAnyOrigin],
    node_leaf_counts: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_count: Int,
):
    var i = global_idx.x
    var internal_count = leaf_count - 1
    if i >= internal_count:
        return

    # each internal node initializes its own refit state
    node_flags[i] = UInt32(0)

    var invalid = AABB.invalid()
    var bounds_base = i * BinaryBvhNode.BOUNDS_STRIDE
    invalid.store6(node_bounds, bounds_base)
    invalid.store6(node_bounds, bounds_base + AABB.STRIDE)

    # Karras LBVH range owned by this internal node
    var r = _lbvh_find_range(sorted_morton_codes, i, leaf_count)
    var first = r[0]
    var last = r[1]

    node_leaf_counts[i] = UInt32(last - first + 1)

    # only the root has no parent
    if first == 0 and last == leaf_count - 1:
        node_meta[_node_parent_index(UInt32(i))] = LBVH_SENTINEL

    var split = _lbvh_find_split(sorted_morton_codes, first, last, leaf_count)

    var left_encoded = UInt32(split)
    if split == first:
        left_encoded |= LBVH_LEAF_FLAG
        leaf_parent[split] = UInt32(i)
    else:
        node_meta[_node_parent_index(UInt32(split))] = UInt32(i)

    var right_child = split + 1
    var right_encoded = UInt32(right_child)
    if right_child == last:
        right_encoded |= LBVH_LEAF_FLAG
        leaf_parent[right_child] = UInt32(i)
    else:
        node_meta[_node_parent_index(UInt32(right_child))] = UInt32(i)

    var meta_base = i * BinaryBvhNode.META_STRIDE
    node_meta[meta_base + BinaryBvhNode.LEFT] = left_encoded
    node_meta[meta_base + BinaryBvhNode.RIGHT] = right_encoded
    node_meta[meta_base + BinaryBvhNode.FENCE] = UInt32(last)


def build_binary_bvh_with_lbvh(
    ctx: DeviceContext,
    mut binary: GpuBinaryBoundsBvh,
    mut workspace: RadixSortWorkspace[DType.uint32, DType.uint32],
) raises -> GpuBuildTimings:
    """Builds the temporary sorted binary LBVH.

    This stage owns:
    1. leaf AABBs
    2. Morton keys
    3. sorted leaf ids
    4. binary topology
    5. refit bounds
    """
    var t_start = perf_counter_ns()

    # leaf AABB
    # for now: inside binary_bvh __init__

    # morton codes

    ctx.enqueue_function[compute_bounds_morton_codes_kernel](
        binary.leaf_bounds,
        binary.bounds_device,
        binary.keys,
        binary.leaf_ids,
        binary.leaf_count,
        grid_dim=binary.blocks_leaves,
        block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
    )
    ctx.synchronize()
    var t_morton = perf_counter_ns()

    # sort by morton codes
    device_radix_sort_pairs[DType.uint32, DType.uint32](
        ctx,
        workspace,
        binary.keys,
        binary.leaf_ids,
        binary.leaf_count,
    )

    ctx.synchronize()
    var t_sort = perf_counter_ns()

    # merge nodes
    if binary.internal_count > 0:
        ctx.enqueue_function[build_lbvh_topology_kernel](
            binary.keys,
            binary.node_meta,
            binary.leaf_parent,
            binary.node_bounds,
            binary.node_flags,
            binary.node_leaf_counts,
            binary.leaf_count,
            grid_dim=binary.blocks_internal,
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
        ctx.synchronize()
    var t_topology = perf_counter_ns()

    # compute aabb over merged nodes
    if binary.internal_count > 0:
        binary.node_flags.enqueue_fill(0)
        ctx.enqueue_function[refit_lbvh_bounds_from_leaves_kernel](
            binary.leaf_bounds,
            binary.leaf_ids,
            binary.node_meta,
            binary.leaf_parent,
            binary.node_bounds,
            binary.node_flags,
            binary.leaf_count,
            grid_dim=binary.blocks_leaves,
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
        ctx.synchronize()
    var t_refit = perf_counter_ns()

    return GpuBuildTimings(
        Int(t_morton - t_start),
        Int(t_sort - t_morton),
        Int(t_topology - t_sort),
        Int(t_refit - t_topology),
        0,
        0,
        0,
    )
