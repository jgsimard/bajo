from std.bit import count_leading_zeros
from std.math import min, max
from std.time import perf_counter_ns
from std.atomic import Atomic
from std.gpu import DeviceBuffer, DeviceContext, global_idx

from bajo.core.aabb import AABB
from bajo.core.vec import Vec3f32
from bajo.core.morton import morton3
from bajo.sort.gpu.radix_sort import device_radix_sort_pairs, RadixSortWorkspace
from bajo.bvh.constants import (
    LBVH_LEAF_FLAG,
    LBVH_SENTINEL,
    BOUNDS_STRIDE,
)
from bajo.bvh.gpu.utils import GpuBuildTimings
from bajo.bvh.gpu.builder.binary_layout import (
    BINARY_BVH_NODE_META_STRIDE,
    BINARY_BVH_NODE_PARENT,
    BINARY_BVH_NODE_LEFT,
    BINARY_BVH_NODE_RIGHT,
    BINARY_BVH_NODE_FENCE,
    BINARY_BVH_NODE_BOUNDS_STRIDE,
    GPU_BOUNDS_BVH_BLOCK_SIZE,
    _node_parent_index,
    _node_left,
    _node_right,
    _write_child_bounds,
    _load_and_union_node_bounds,
    GpuBinaryBoundsBvh,
)


def compute_bounds_morton_codes_kernel(
    leaf_bounds: UnsafePointer[Float32, MutAnyOrigin],
    keys: UnsafePointer[UInt32, MutAnyOrigin],
    values: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_count: Int,
    cmin: Vec3f32,
    inv_extent: Vec3f32,
):
    var i = global_idx.x
    if i >= leaf_count:
        return

    var b = i * BOUNDS_STRIDE
    var bounds = AABB.load6(leaf_bounds, b)
    var c = (bounds.centroid() - cmin) * inv_extent

    keys[i] = morton3(c.x, c.y, c.z)
    values[i] = UInt32(i)


def refit_lbvh_bounds_from_leaves_kernel(
    leaf_bounds: UnsafePointer[Float32, MutAnyOrigin],
    sorted_leaf_ids: UnsafePointer[UInt32, MutAnyOrigin],
    node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_parent: UnsafePointer[UInt32, MutAnyOrigin],
    node_bounds: UnsafePointer[Float32, MutAnyOrigin],
    node_flags: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_count: Int,
):
    var leaf_idx = global_idx.x
    if leaf_idx >= leaf_count:
        return

    var item_idx = UInt32(sorted_leaf_ids[leaf_idx])
    var b = Int(item_idx) * BOUNDS_STRIDE
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


def _common_prefix_gpu(
    keys: UnsafePointer[UInt32, MutAnyOrigin],
    i: Int,
    j: Int,
    n: Int,
) -> Int:
    if j < 0 or j >= n:
        return -1

    var a = UInt32(keys[i])
    var b = UInt32(keys[j])

    if a != b:
        return Int(count_leading_zeros(a ^ b))

    # Tie-break equal Morton codes with the sorted leaf index. This makes the
    # prefix order total and keeps degenerate duplicate-code cases deterministic.
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
        var base = i * BINARY_BVH_NODE_META_STRIDE
        node_meta[base + BINARY_BVH_NODE_PARENT] = LBVH_SENTINEL  # parent
        node_meta[base + BINARY_BVH_NODE_LEFT] = 0  # left child, encoded
        node_meta[base + BINARY_BVH_NODE_RIGHT] = 0  # right child, encoded
        # fence/debug: rightmost leaf in range
        node_meta[base + BINARY_BVH_NODE_FENCE] = 0

    if i < leaf_count:
        leaf_parent[i] = LBVH_SENTINEL


def build_lbvh_topology_kernel(
    sorted_keys: UnsafePointer[UInt32, MutAnyOrigin],
    node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_parent: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_count: Int,
):
    var i = global_idx.x
    var internal_count = leaf_count - 1
    if i >= internal_count:
        return

    # Determine direction of the range for this internal node.
    var d_next = _common_prefix_gpu(sorted_keys, i, i + 1, leaf_count)
    var d_prev = _common_prefix_gpu(sorted_keys, i, i - 1, leaf_count)
    var d = 1
    if d_next < d_prev:
        d = -1

    # Minimum prefix outside the range.
    var delta_min = _common_prefix_gpu(sorted_keys, i, i - d, leaf_count)

    # Find an upper bound on the range length.
    var lmax = 2
    while (
        _common_prefix_gpu(sorted_keys, i, i + lmax * d, leaf_count) > delta_min
    ):
        lmax <<= 1
        if lmax > leaf_count * 2:
            break

    # Binary search for exact range length.
    var l = 0
    var t = lmax >> 1
    while t > 0:
        if (
            _common_prefix_gpu(sorted_keys, i, i + (l + t) * d, leaf_count)
            > delta_min
        ):
            l += t
        t >>= 1

    var j = i + l * d
    var first = min(i, j)
    var last = max(i, j)

    # Find split inside [first, last].
    var node_prefix = _common_prefix_gpu(sorted_keys, first, last, leaf_count)
    var split = first
    var step = last - first
    while step > 1:
        step = (step + 1) >> 1
        var new_split = split + step
        if new_split < last:
            var split_prefix = _common_prefix_gpu(
                sorted_keys, first, new_split, leaf_count
            )
            if split_prefix > node_prefix:
                split = new_split

    var left_encoded: UInt32
    var right_encoded: UInt32

    if split == first:
        left_encoded = UInt32(split) | LBVH_LEAF_FLAG
        if split >= 0 and split < leaf_count:
            leaf_parent[split] = UInt32(i)
    else:
        left_encoded = UInt32(split)
        if split >= 0 and split < internal_count:
            node_meta[_node_parent_index(UInt32(split))] = UInt32(i)

    var right_child = split + 1
    if right_child == last:
        right_encoded = UInt32(right_child) | LBVH_LEAF_FLAG
        if right_child >= 0 and right_child < leaf_count:
            leaf_parent[right_child] = UInt32(i)
    else:
        right_encoded = UInt32(right_child)
        if right_child >= 0 and right_child < internal_count:
            node_meta[_node_parent_index(UInt32(right_child))] = UInt32(i)

    var base = i * BINARY_BVH_NODE_META_STRIDE
    node_meta[base + BINARY_BVH_NODE_LEFT] = left_encoded
    node_meta[base + BINARY_BVH_NODE_RIGHT] = right_encoded
    node_meta[base + BINARY_BVH_NODE_FENCE] = UInt32(last)


def init_lbvh_bounds_kernel(
    node_bounds: UnsafePointer[Float32, MutAnyOrigin],
    node_flags: UnsafePointer[UInt32, MutAnyOrigin],
    internal_count: Int,
):
    var i = global_idx.x
    if i >= internal_count:
        return

    var b = i * BINARY_BVH_NODE_BOUNDS_STRIDE
    invalid = AABB.invalid()
    invalid.store6(node_bounds, b)
    invalid.store6(node_bounds, b + 6)
    node_flags[i] = UInt32(0)


@fieldwise_init
struct GpuBoundsLbvhBuilder[width: Int]:
    """Builds the temporary sorted binary LBVH.

    This stage owns:
    1. leaf AABBs
    2. Morton keys
    3. sorted leaf ids
    4. binary topology
    5. refit bounds
    """

    def build(
        mut self,
        ctx: DeviceContext,
        mut binary: GpuBinaryBoundsBvh,
        mut workspace: RadixSortWorkspace[DType.uint32, DType.uint32],
    ) raises -> GpuBuildTimings:
        var start_ns = perf_counter_ns()
        var centroid_bounds = self._compute_centroid_bounds(
            binary.leaf_count,
            binary.leaf_bounds_host,
        )
        var extent = centroid_bounds.extent()
        var inv = extent.safe_inv()

        ctx.enqueue_function[compute_bounds_morton_codes_kernel](
            binary.leaf_bounds,
            binary.keys,
            binary.sorted_leaf_ids,
            binary.leaf_count,
            centroid_bounds._min,
            inv,
            grid_dim=binary.blocks_leaves,
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
        ctx.synchronize()
        var m = perf_counter_ns()

        device_radix_sort_pairs[DType.uint32, DType.uint32](
            ctx,
            workspace,
            binary.keys,
            binary.sorted_leaf_ids,
            binary.leaf_count,
        )

        ctx.synchronize()
        var s = perf_counter_ns()

        if binary.internal_count > 0:
            ctx.enqueue_function[init_lbvh_topology_kernel](
                binary.node_meta,
                binary.leaf_parent,
                binary.internal_count,
                binary.leaf_count,
                grid_dim=binary.blocks_init,
                block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
            )
            ctx.enqueue_function[build_lbvh_topology_kernel](
                binary.keys,
                binary.node_meta,
                binary.leaf_parent,
                binary.leaf_count,
                grid_dim=binary.blocks_internal,
                block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
            )
            ctx.synchronize()
        var t = perf_counter_ns()

        if binary.internal_count > 0:
            ctx.enqueue_function[init_lbvh_bounds_kernel](
                binary.node_bounds,
                binary.node_flags,
                binary.internal_count,
                grid_dim=binary.blocks_internal,
                block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
            )
            ctx.enqueue_function[refit_lbvh_bounds_from_leaves_kernel](
                binary.leaf_bounds,
                binary.sorted_leaf_ids,
                binary.node_meta,
                binary.leaf_parent,
                binary.node_bounds,
                binary.node_flags,
                binary.leaf_count,
                grid_dim=binary.blocks_leaves,
                block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
            )
            ctx.synchronize()
        var r = perf_counter_ns()

        return GpuBuildTimings(
            Int(m - start_ns),
            Int(s - m),
            Int(t - s),
            Int(r - t),
            0,
        )

    def _compute_centroid_bounds(
        self,
        leaf_count: Int,
        leaf_bounds_host: List[Float32],
    ) -> AABB:
        var out = AABB.invalid()
        for i in range(leaf_count):
            var b = i * BOUNDS_STRIDE
            aabb = AABB.load6(leaf_bounds_host.unsafe_ptr(), b)
            out.grow(aabb.centroid())
        return out
