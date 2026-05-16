from std.math import min, max, ceildiv, sqrt
from std.time import perf_counter_ns
from std.atomic import Atomic
from std.gpu import DeviceBuffer, global_idx
from std.gpu.host import DeviceContext

from bajo.core.aabb import AABB
from bajo.core.vec import Vec3f32, vmin, vmax
from bajo.core.mat import Mat44f32, transform_point, transform_vector
from bajo.core.intersect import intersect_ray_aabb, intersect_ray_tri
from bajo.core.morton import morton3
from bajo.core.bvh.types import RayFlat, Hit, Sphere
from bajo.core.bvh.gpu.constants import (
    LBVH_LEAF_FLAG,
    LBVH_INDEX_MASK,
    LBVH_SENTINEL,
)
from bajo.core.bvh.gpu.kernels import (
    GPU_TRAVERSAL_STACK_SIZE,
    LBVH_NODE_META_STRIDE,
    LBVH_NODE_PARENT,
    LBVH_NODE_LEFT,
    LBVH_NODE_RIGHT,
    LBVH_NODE_BOUNDS_STRIDE,
    LBVH_BOUNDS_LEFT,
    LBVH_BOUNDS_RIGHT,
    TRACE_PRIMARY_FULL,
    TRACE_PRIMARY_T,
    TRACE_SHADOW,
    init_lbvh_topology_kernel,
    init_lbvh_bounds_kernel,
    build_lbvh_topology_kernel,
)
from bajo.core.bvh.gpu.validate import (
    validate_sorted_keys,
    validate_topology,
    validate_refit_bounds,
)
from bajo.core.bvh.gpu.utils import GpuBuildTimings, GpuBVHValidation
from bajo.sort.gpu.radix_sort import device_radix_sort_pairs, RadixSortWorkspace


comptime GPU_BOUNDS_BVH_BLOCK_SIZE = 128
comptime GPU_WIDE_EMPTY_LANE = UInt32(0xFFFFFFFF)
comptime GPU_WIDE_BOUNDS_STRIDE = 6
comptime GPU_TRI_LEAF_VERTEX_STRIDE = 9
comptime GPU_SPHERE_STRIDE = 4
comptime _gpu_inf_t = Float32(3.4028234663852886e38)
comptime _gpu_miss_prim = UInt32(0xFFFFFFFF)


@always_inline
def _node_meta_base(node_idx: UInt32) -> Int:
    return Int(node_idx) * LBVH_NODE_META_STRIDE


@always_inline
def _node_bounds_base(node_idx: UInt32) -> Int:
    return Int(node_idx) * LBVH_NODE_BOUNDS_STRIDE


@always_inline
def _node_parent_index(node_idx: UInt32) -> Int:
    return _node_meta_base(node_idx) + LBVH_NODE_PARENT


@always_inline
def _node_left_index(node_idx: UInt32) -> Int:
    return _node_meta_base(node_idx) + LBVH_NODE_LEFT


@always_inline
def _node_right_index(node_idx: UInt32) -> Int:
    return _node_meta_base(node_idx) + LBVH_NODE_RIGHT


@always_inline
def _node_left(
    node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    node_idx: UInt32,
) -> UInt32:
    return UInt32(node_meta[_node_left_index(node_idx)])


@always_inline
def _node_right(
    node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    node_idx: UInt32,
) -> UInt32:
    return UInt32(node_meta[_node_right_index(node_idx)])


@always_inline
def _load_leaf_bounds_host(
    leaf_bounds: List[Float32],
    leaf_idx: UInt32,
) -> AABB:
    var b = Int(leaf_idx) * GPU_WIDE_BOUNDS_STRIDE
    return AABB.load6(leaf_bounds.unsafe_ptr(), b)


@always_inline
def _load_internal_bounds_host(
    node_bounds: List[Float32],
    node_idx: UInt32,
) -> AABB:
    var b = Int(node_idx) * LBVH_NODE_BOUNDS_STRIDE
    b1 = AABB.load6(node_bounds.unsafe_ptr(), b)
    b2 = AABB.load6(node_bounds.unsafe_ptr(), b + 6)
    return AABB.merge(b1, b2)


@always_inline
def _is_encoded_leaf(encoded: UInt32) -> Bool:
    return (encoded & LBVH_LEAF_FLAG) != 0


@always_inline
def _encoded_index(encoded: UInt32) -> UInt32:
    return encoded & LBVH_INDEX_MASK


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

    var b = i * GPU_WIDE_BOUNDS_STRIDE
    var bounds = AABB.load6(leaf_bounds, b)
    var c = (bounds.centroid() - cmin) * inv_extent

    keys[i] = morton3(c.x, c.y, c.z)
    values[i] = UInt32(i)


@always_inline
def _write_child_bounds(
    node_bounds: UnsafePointer[Float32, MutAnyOrigin],
    parent: UInt32,
    write_left: Bool,
    bounds: AABB,
):
    var b = _node_bounds_base(parent)
    if not write_left:
        b += 6
    bounds.store6(node_bounds, b)


@always_inline
def _load_and_union_node_bounds(
    node_bounds: UnsafePointer[Float32, MutAnyOrigin],
    parent: UInt32,
) -> AABB:
    var b = _node_bounds_base(parent)
    b1 = AABB.load6(node_bounds, b)
    b2 = AABB.load6(node_bounds, b + 6)
    return AABB.merge(b1, b2)


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
    var b = Int(item_idx) * GPU_WIDE_BOUNDS_STRIDE
    var bounds = AABB(
        Vec3f32(leaf_bounds[b + 0], leaf_bounds[b + 1], leaf_bounds[b + 2]),
        Vec3f32(leaf_bounds[b + 3], leaf_bounds[b + 4], leaf_bounds[b + 5]),
    )

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


# -----------------------------------------------------------------------------
# GPU-resident wide collapse helpers
# -----------------------------------------------------------------------------


@always_inline
def _encoded_leaf_count_gpu(
    encoded: UInt32,
    node_leaf_counts: UnsafePointer[UInt32, MutAnyOrigin],
) -> UInt32:
    if _is_encoded_leaf(encoded):
        return UInt32(1)
    return UInt32(node_leaf_counts[Int(_encoded_index(encoded))])


@always_inline
def _encoded_bounds_gpu(
    encoded: UInt32,
    leaf_bounds: UnsafePointer[Float32, MutAnyOrigin],
    sorted_leaf_ids: UnsafePointer[UInt32, MutAnyOrigin],
    node_bounds: UnsafePointer[Float32, MutAnyOrigin],
) -> AABB:
    if _is_encoded_leaf(encoded):
        var sorted_leaf_idx = _encoded_index(encoded)
        var item_idx = UInt32(sorted_leaf_ids[Int(sorted_leaf_idx)])
        var b = Int(item_idx) * GPU_WIDE_BOUNDS_STRIDE
        return AABB.load6(leaf_bounds, b)

    return _load_and_union_node_bounds(node_bounds, _encoded_index(encoded))


@always_inline
def _write_wide_lane_bounds[
    width: Int,
](
    wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    wide_node_idx: UInt32,
    lane: Int,
    bounds: AABB,
):
    var b = (Int(wide_node_idx) * width + lane) * GPU_WIDE_BOUNDS_STRIDE
    bounds.store6(wide_bounds, b)


@always_inline
def _collect_encoded_leaf_payloads_gpu[
    width: Int,
](
    encoded: UInt32,
    sorted_leaf_ids: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_payloads: UnsafePointer[UInt32, MutAnyOrigin],
    node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    out_leaf_block_indices: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_block_idx: UInt32,
):
    var stack = InlineArray[UInt32, GPU_TRAVERSAL_STACK_SIZE](fill=0)
    var sp = 0
    stack[sp] = encoded
    sp += 1

    var out_count = 0
    while sp > 0:
        sp -= 1
        var e = stack[sp]

        if _is_encoded_leaf(e):
            if out_count < width:
                var sorted_leaf_idx = _encoded_index(e)
                var item_idx = UInt32(sorted_leaf_ids[Int(sorted_leaf_idx)])
                out_leaf_block_indices[
                    Int(leaf_block_idx) * width + out_count
                ] = UInt32(leaf_payloads[Int(item_idx)])
                out_count += 1
        else:
            var node_idx = _encoded_index(e)
            var left = _node_left(node_meta, node_idx)
            var right = _node_right(node_meta, node_idx)

            if sp + 2 <= GPU_TRAVERSAL_STACK_SIZE:
                stack[sp] = right
                sp += 1
                stack[sp] = left
                sp += 1

    comptime for lane in range(width):
        if lane >= out_count:
            out_leaf_block_indices[
                Int(leaf_block_idx) * width + lane
            ] = GPU_WIDE_EMPTY_LANE


def init_gpu_wide_collapse_kernel[
    width: Int,
](
    wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_block_indices: UnsafePointer[UInt32, MutAnyOrigin],
    node_leaf_counts: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_block_counter: UnsafePointer[UInt32, MutAnyOrigin],
    wide_root: UnsafePointer[UInt32, MutAnyOrigin],
    max_wide_lanes: Int,
    max_leaf_slots: Int,
    internal_count: Int,
):
    var i = global_idx.x

    if i < max_wide_lanes:
        wide_data[i] = UInt32(0)
        wide_counts[i] = GPU_WIDE_EMPTY_LANE

        var b = i * GPU_WIDE_BOUNDS_STRIDE
        wide_bounds[b + 0] = 0.0
        wide_bounds[b + 1] = 0.0
        wide_bounds[b + 2] = 0.0
        wide_bounds[b + 3] = 0.0
        wide_bounds[b + 4] = 0.0
        wide_bounds[b + 5] = 0.0

    if i < max_leaf_slots:
        leaf_block_indices[i] = GPU_WIDE_EMPTY_LANE

    if i < internal_count:
        node_leaf_counts[i] = UInt32(0)

    if i == 0:
        leaf_block_counter[0] = UInt32(0)
        wide_root[0] = UInt32(0)


def compute_lbvh_subtree_leaf_counts_kernel(
    node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_parent: UnsafePointer[UInt32, MutAnyOrigin],
    node_leaf_counts: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_count: Int,
):
    var leaf_idx = global_idx.x
    if leaf_idx >= leaf_count:
        return

    var parent = UInt32(leaf_parent[leaf_idx])
    while parent != LBVH_SENTINEL:
        _ = Atomic.fetch_add(node_leaf_counts + Int(parent), UInt32(1))
        parent = UInt32(node_meta[_node_parent_index(parent)])


def find_lbvh_root_kernel(
    node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    wide_root: UnsafePointer[UInt32, MutAnyOrigin],
    internal_count: Int,
):
    var i = global_idx.x
    if i >= internal_count:
        return

    if UInt32(node_meta[_node_parent_index(UInt32(i))]) == LBVH_SENTINEL:
        wide_root[0] = UInt32(i)


def collapse_single_leaf_to_wide_kernel[
    width: Int,
](
    leaf_bounds: UnsafePointer[Float32, MutAnyOrigin],
    leaf_payloads: UnsafePointer[UInt32, MutAnyOrigin],
    wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_block_indices: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_block_counter: UnsafePointer[UInt32, MutAnyOrigin],
    wide_root: UnsafePointer[UInt32, MutAnyOrigin],
):
    var i = global_idx.x
    if i != 0:
        return

    wide_root[0] = UInt32(0)
    leaf_block_counter[0] = UInt32(1)

    wide_bounds[0] = leaf_bounds[0]
    wide_bounds[1] = leaf_bounds[1]
    wide_bounds[2] = leaf_bounds[2]
    wide_bounds[3] = leaf_bounds[3]
    wide_bounds[4] = leaf_bounds[4]
    wide_bounds[5] = leaf_bounds[5]

    wide_data[0] = UInt32(0)
    wide_counts[0] = UInt32(1)
    leaf_block_indices[0] = UInt32(leaf_payloads[0])


def collapse_lbvh_to_wide_kernel[
    width: Int,
](
    leaf_bounds: UnsafePointer[Float32, MutAnyOrigin],
    leaf_payloads: UnsafePointer[UInt32, MutAnyOrigin],
    sorted_leaf_ids: UnsafePointer[UInt32, MutAnyOrigin],
    node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    node_bounds: UnsafePointer[Float32, MutAnyOrigin],
    node_leaf_counts: UnsafePointer[UInt32, MutAnyOrigin],
    wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_block_indices: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_block_counter: UnsafePointer[UInt32, MutAnyOrigin],
    internal_count: Int,
):
    var node_i = global_idx.x
    if node_i >= internal_count:
        return

    var wide_idx = UInt32(node_i)
    var encoded = UInt32(node_i)

    var pool = InlineArray[UInt32, width](fill=0)
    pool[0] = encoded
    var p_size = 1

    # Greedy CPU-like collapse: expand the largest internal subtree while it
    # still contains more leaves than can fit into one packed leaf block.
    while p_size < width:
        var best_area = Float32(-1.0)
        var best_lane = -1

        comptime for lane in range(width):
            if lane < p_size:
                var e = pool[lane]
                if not _is_encoded_leaf(e):
                    var subtree_leaves = _encoded_leaf_count_gpu(
                        e, node_leaf_counts
                    )
                    if subtree_leaves > UInt32(width):
                        var b = _encoded_bounds_gpu(
                            e, leaf_bounds, sorted_leaf_ids, node_bounds
                        )
                        var area = b.surface_area()
                        if area > best_area:
                            best_area = area
                            best_lane = lane

        if best_lane < 0:
            break

        var n_idx = _encoded_index(pool[best_lane])
        pool[best_lane] = _node_left(node_meta, n_idx)
        pool[p_size] = _node_right(node_meta, n_idx)
        p_size += 1

    comptime for lane in range(width):
        var lane_base = Int(wide_idx) * width + lane

        if lane >= p_size:
            wide_counts[lane_base] = GPU_WIDE_EMPTY_LANE
            continue

        var e = pool[lane]
        var b = _encoded_bounds_gpu(
            e, leaf_bounds, sorted_leaf_ids, node_bounds
        )
        _write_wide_lane_bounds[width](wide_bounds, wide_idx, lane, b)

        var subtree_leaves = _encoded_leaf_count_gpu(e, node_leaf_counts)
        if subtree_leaves <= UInt32(width):
            var block_idx = Atomic.fetch_add(leaf_block_counter, UInt32(1))
            _collect_encoded_leaf_payloads_gpu[width](
                e,
                sorted_leaf_ids,
                leaf_payloads,
                node_meta,
                leaf_block_indices,
                block_idx,
            )
            wide_data[lane_base] = block_idx
            wide_counts[lane_base] = subtree_leaves
        else:
            wide_data[lane_base] = _encoded_index(e)
            wide_counts[lane_base] = UInt32(0)


def pack_triangle_leaf_blocks_kernel[
    width: Int,
](
    vertices: UnsafePointer[Float32, MutAnyOrigin],
    leaf_block_indices: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_vertices: UnsafePointer[Float32, MutAnyOrigin],
    leaf_prims: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_block_count: Int,
):
    var block_idx = global_idx.x
    if block_idx >= leaf_block_count:
        return

    comptime for lane in range(width):
        var idx = block_idx * width + lane
        var prim = UInt32(leaf_block_indices[idx])
        leaf_prims[idx] = prim

        var out_base = idx * GPU_TRI_LEAF_VERTEX_STRIDE
        if prim == GPU_WIDE_EMPTY_LANE:
            for k in range(GPU_TRI_LEAF_VERTEX_STRIDE):
                leaf_vertices[out_base + k] = 0.0
        else:
            var in_base = Int(prim) * GPU_TRI_LEAF_VERTEX_STRIDE
            for k in range(GPU_TRI_LEAF_VERTEX_STRIDE):
                leaf_vertices[out_base + k] = vertices[in_base + k]


def pack_sphere_leaf_blocks_kernel[
    width: Int,
](
    spheres: UnsafePointer[Float32, MutAnyOrigin],
    leaf_block_indices: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_spheres: UnsafePointer[Float32, MutAnyOrigin],
    leaf_prims: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_block_count: Int,
):
    var block_idx = global_idx.x
    if block_idx >= leaf_block_count:
        return

    comptime for lane in range(width):
        var idx = block_idx * width + lane
        var prim = UInt32(leaf_block_indices[idx])
        leaf_prims[idx] = prim

        var out_base = idx * GPU_SPHERE_STRIDE
        if prim == GPU_WIDE_EMPTY_LANE:
            for k in range(GPU_SPHERE_STRIDE):
                leaf_spheres[out_base + k] = 0.0
        else:
            var in_base = Int(prim) * GPU_SPHERE_STRIDE
            for k in range(GPU_SPHERE_STRIDE):
                leaf_spheres[out_base + k] = spheres[in_base + k]


struct GpuBoundsBvh[width: Int]:
    """Generic GPU LBVH + host-side binary-to-wide collapse.

    Build input is only leaf AABBs plus payload ids.  The LBVH topology and
    bounds refit are generic.  The width-parametric wide layout uses flat SoA
    buffers so device code can be specialized by `width` without requiring
    DeviceBuffer[GpuWideBvhNode[width]].

    Wide lane encoding mirrors the CPU BVH:
        count == GPU_WIDE_EMPTY_LANE -> unused lane
        count == 0                   -> child node, data = wide child index
        count > 0                    -> leaf block, data = leaf block index
    """

    var leaf_count: Int
    var internal_count: Int
    var root_idx: UInt32
    var node_count: Int
    var leaf_block_count: Int
    var max_wide_nodes: Int
    var max_leaf_blocks: Int
    var collapse_ns: Int

    var blocks_leaves: Int
    var blocks_internal: Int
    var blocks_init: Int

    var leaf_bounds_host: List[Float32]
    var leaf_payloads_host: List[UInt32]

    var leaf_bounds: DeviceBuffer[DType.float32]
    var leaf_payloads: DeviceBuffer[DType.uint32]
    var keys: DeviceBuffer[DType.uint32]
    var values: DeviceBuffer[DType.uint32]
    var node_meta: DeviceBuffer[DType.uint32]
    var leaf_parent: DeviceBuffer[DType.uint32]
    var node_bounds: DeviceBuffer[DType.float32]
    var node_flags: DeviceBuffer[DType.uint32]

    var wide_bounds: DeviceBuffer[DType.float32]
    var wide_data: DeviceBuffer[DType.uint32]
    var wide_counts: DeviceBuffer[DType.uint32]
    var leaf_block_indices: DeviceBuffer[DType.uint32]
    var node_leaf_counts: DeviceBuffer[DType.uint32]
    var leaf_block_counter: DeviceBuffer[DType.uint32]
    var wide_root: DeviceBuffer[DType.uint32]

    var workspace: RadixSortWorkspace[DType.uint32, DType.uint32]

    def __init__(
        out self,
        mut ctx: DeviceContext,
        leaf_bounds: List[Float32],
        leaf_payloads: List[UInt32],
    ) raises:
        self.leaf_count = len(leaf_payloads)
        self.internal_count = max(self.leaf_count - 1, 0)
        self.root_idx = UInt32(0)
        self.node_count = 0
        self.leaf_block_count = 0
        self.max_wide_nodes = max(self.internal_count, 1)
        self.max_leaf_blocks = max(self.internal_count * Self.width, 1)
        self.collapse_ns = 0

        self.blocks_leaves = ceildiv(
            max(self.leaf_count, 1), GPU_BOUNDS_BVH_BLOCK_SIZE
        )
        self.blocks_internal = ceildiv(
            max(self.internal_count, 1), GPU_BOUNDS_BVH_BLOCK_SIZE
        )
        self.blocks_init = ceildiv(
            max(max(self.leaf_count, self.internal_count), 1),
            GPU_BOUNDS_BVH_BLOCK_SIZE,
        )

        self.leaf_bounds_host = leaf_bounds.copy()
        self.leaf_payloads_host = leaf_payloads.copy()

        self.leaf_bounds = _copy_f32_to_device(ctx, self.leaf_bounds_host)
        self.leaf_payloads = _copy_u32_to_device(ctx, self.leaf_payloads_host)
        self.keys = ctx.enqueue_create_buffer[DType.uint32](
            max(self.leaf_count, 1)
        )
        self.values = ctx.enqueue_create_buffer[DType.uint32](
            max(self.leaf_count, 1)
        )
        self.node_meta = ctx.enqueue_create_buffer[DType.uint32](
            max(self.internal_count, 1) * LBVH_NODE_META_STRIDE
        )
        self.leaf_parent = ctx.enqueue_create_buffer[DType.uint32](
            max(self.leaf_count, 1)
        )
        self.node_bounds = ctx.enqueue_create_buffer[DType.float32](
            max(self.internal_count, 1) * LBVH_NODE_BOUNDS_STRIDE
        )
        self.node_flags = ctx.enqueue_create_buffer[DType.uint32](
            max(self.internal_count, 1)
        )

        # Wide node index is the binary internal node index. Leaf blocks are
        # allocated on the GPU during collapse with an atomic counter.
        self.wide_bounds = ctx.enqueue_create_buffer[DType.float32](
            self.max_wide_nodes * Self.width * GPU_WIDE_BOUNDS_STRIDE
        )
        self.wide_data = ctx.enqueue_create_buffer[DType.uint32](
            self.max_wide_nodes * Self.width
        )
        self.wide_counts = ctx.enqueue_create_buffer[DType.uint32](
            self.max_wide_nodes * Self.width
        )
        self.leaf_block_indices = ctx.enqueue_create_buffer[DType.uint32](
            self.max_leaf_blocks * Self.width
        )
        self.node_leaf_counts = ctx.enqueue_create_buffer[DType.uint32](
            max(self.internal_count, 1)
        )
        self.leaf_block_counter = ctx.enqueue_create_buffer[DType.uint32](1)
        self.wide_root = ctx.enqueue_create_buffer[DType.uint32](1)

        self.workspace = RadixSortWorkspace[DType.uint32, DType.uint32](
            ctx, max(self.leaf_count, 1)
        )

    def build(mut self, ctx: DeviceContext) raises -> GpuBuildTimings:
        var start = perf_counter_ns()

        if self.leaf_count == 0:
            return GpuBuildTimings(0, 0, 0, 0, 0, 0)

        var centroid_bounds = self._compute_centroid_bounds()
        var extent = centroid_bounds._max - centroid_bounds._min
        var inv = Vec3f32(0.0)
        if extent.x > 0.0:
            inv.x = 1.0 / extent.x
        if extent.y > 0.0:
            inv.y = 1.0 / extent.y
        if extent.z > 0.0:
            inv.z = 1.0 / extent.z

        ctx.enqueue_function[compute_bounds_morton_codes_kernel](
            self.leaf_bounds.unsafe_ptr(),
            self.keys.unsafe_ptr(),
            self.values.unsafe_ptr(),
            self.leaf_count,
            centroid_bounds._min,
            inv,
            grid_dim=self.blocks_leaves,
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
        ctx.synchronize()
        var m = perf_counter_ns()

        device_radix_sort_pairs[DType.uint32, DType.uint32](
            ctx, self.workspace, self.keys, self.values, self.leaf_count
        )
        ctx.synchronize()
        var s = perf_counter_ns()

        if self.internal_count > 0:
            ctx.enqueue_function[init_lbvh_topology_kernel](
                self.node_meta.unsafe_ptr(),
                self.leaf_parent.unsafe_ptr(),
                self.internal_count,
                self.leaf_count,
                grid_dim=self.blocks_init,
                block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
            )
            ctx.enqueue_function[build_lbvh_topology_kernel](
                self.keys.unsafe_ptr(),
                self.node_meta.unsafe_ptr(),
                self.leaf_parent.unsafe_ptr(),
                self.leaf_count,
                grid_dim=self.blocks_internal,
                block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
            )
            ctx.synchronize()
        var t = perf_counter_ns()

        if self.internal_count > 0:
            ctx.enqueue_function[init_lbvh_bounds_kernel](
                self.node_bounds.unsafe_ptr(),
                self.node_flags.unsafe_ptr(),
                self.internal_count,
                grid_dim=self.blocks_internal,
                block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
            )
            ctx.enqueue_function[refit_lbvh_bounds_from_leaves_kernel](
                self.leaf_bounds.unsafe_ptr(),
                self.values.unsafe_ptr(),
                self.node_meta.unsafe_ptr(),
                self.leaf_parent.unsafe_ptr(),
                self.node_bounds.unsafe_ptr(),
                self.node_flags.unsafe_ptr(),
                self.leaf_count,
                grid_dim=self.blocks_leaves,
                block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
            )
            ctx.synchronize()
        var r = perf_counter_ns()

        self._collapse_to_wide(ctx)
        ctx.synchronize()
        var c = perf_counter_ns()
        self.collapse_ns = Int(c - r)

        return GpuBuildTimings(
            0,
            Int(m - start),
            Int(s - m),
            Int(t - s),
            Int(r - t),
            Int(c - start),
        )

    def validate(mut self, scene_bounds: AABB) raises -> GpuBVHValidation:
        var sorted_validation = validate_sorted_keys(
            self.keys, self.values, self.leaf_count
        )

        if self.leaf_count <= 1:
            return GpuBVHValidation(
                sorted_validation.sorted_ok,
                sorted_validation.values_ok,
                True,
                UInt32(1),
                UInt32(0),
                True,
                0.0,
                UInt32(0),
                sorted_validation.guard,
            )

        var topo_validation = validate_topology(
            self.node_meta, self.leaf_parent, self.leaf_count
        )
        var refit_validation = validate_refit_bounds(
            self.node_bounds,
            self.node_flags,
            self.node_meta,
            self.leaf_count,
            scene_bounds,
        )
        self.root_idx = refit_validation.root_idx
        var guard = (
            sorted_validation.guard
            + topo_validation.guard
            + refit_validation.guard
        )

        return GpuBVHValidation(
            sorted_validation.sorted_ok,
            sorted_validation.values_ok,
            topo_validation.ok,
            topo_validation.root_count,
            topo_validation.root_idx,
            refit_validation.ok,
            refit_validation.diff,
            refit_validation.root_idx,
            guard,
        )

    def root_bounds(self) -> AABB:
        var out = AABB.invalid()
        for i in range(self.leaf_count):
            var b = i * GPU_WIDE_BOUNDS_STRIDE
            out.grow(Vec3f32.load(self.leaf_bounds_host.unsafe_ptr(), b))
            out.grow(Vec3f32.load(self.leaf_bounds_host.unsafe_ptr(), b + 3))
        return out

    def _compute_centroid_bounds(self) -> AABB:
        var out = AABB.invalid()
        for i in range(self.leaf_count):
            var b = i * GPU_WIDE_BOUNDS_STRIDE
            v0 = Vec3f32.load(self.leaf_bounds_host.unsafe_ptr(), b)
            v1 = Vec3f32.load(self.leaf_bounds_host.unsafe_ptr(), b + 3)
            out.grow((v0 + v1) * 0.5)
        return out

    def _collapse_to_wide(mut self, ctx: DeviceContext) raises:
        self.node_count = max(self.internal_count, 1)
        self.leaf_block_count = 0

        var max_wide_lanes = self.max_wide_nodes * Self.width
        var max_leaf_slots = self.max_leaf_blocks * Self.width
        var init_n = max(
            max(max_wide_lanes, max_leaf_slots), max(self.internal_count, 1)
        )
        var init_blocks = ceildiv(init_n, GPU_BOUNDS_BVH_BLOCK_SIZE)

        ctx.enqueue_function[init_gpu_wide_collapse_kernel[Self.width]](
            self.wide_bounds.unsafe_ptr(),
            self.wide_data.unsafe_ptr(),
            self.wide_counts.unsafe_ptr(),
            self.leaf_block_indices.unsafe_ptr(),
            self.node_leaf_counts.unsafe_ptr(),
            self.leaf_block_counter.unsafe_ptr(),
            self.wide_root.unsafe_ptr(),
            max_wide_lanes,
            max_leaf_slots,
            self.internal_count,
            grid_dim=init_blocks,
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )

        if self.leaf_count == 1:
            ctx.enqueue_function[
                collapse_single_leaf_to_wide_kernel[Self.width]
            ](
                self.leaf_bounds.unsafe_ptr(),
                self.leaf_payloads.unsafe_ptr(),
                self.wide_bounds.unsafe_ptr(),
                self.wide_data.unsafe_ptr(),
                self.wide_counts.unsafe_ptr(),
                self.leaf_block_indices.unsafe_ptr(),
                self.leaf_block_counter.unsafe_ptr(),
                self.wide_root.unsafe_ptr(),
                grid_dim=1,
                block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
            )
        elif self.leaf_count > 1:
            ctx.enqueue_function[compute_lbvh_subtree_leaf_counts_kernel](
                self.node_meta.unsafe_ptr(),
                self.leaf_parent.unsafe_ptr(),
                self.node_leaf_counts.unsafe_ptr(),
                self.leaf_count,
                grid_dim=self.blocks_leaves,
                block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
            )
            ctx.enqueue_function[find_lbvh_root_kernel](
                self.node_meta.unsafe_ptr(),
                self.wide_root.unsafe_ptr(),
                self.internal_count,
                grid_dim=self.blocks_internal,
                block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
            )
            ctx.enqueue_function[collapse_lbvh_to_wide_kernel[Self.width]](
                self.leaf_bounds.unsafe_ptr(),
                self.leaf_payloads.unsafe_ptr(),
                self.values.unsafe_ptr(),
                self.node_meta.unsafe_ptr(),
                self.node_bounds.unsafe_ptr(),
                self.node_leaf_counts.unsafe_ptr(),
                self.wide_bounds.unsafe_ptr(),
                self.wide_data.unsafe_ptr(),
                self.wide_counts.unsafe_ptr(),
                self.leaf_block_indices.unsafe_ptr(),
                self.leaf_block_counter.unsafe_ptr(),
                self.internal_count,
                grid_dim=self.blocks_internal,
                block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
            )

        ctx.synchronize()

        with self.leaf_block_counter.map_to_host() as h:
            self.leaf_block_count = Int(h[0])

        with self.wide_root.map_to_host() as h:
            self.root_idx = UInt32(h[0])


struct GpuTriangleBvh[width: Int]:
    var tree: GpuBoundsBvh[Self.width]
    var vertices: DeviceBuffer[DType.float32]
    var leaf_vertices: DeviceBuffer[DType.float32]
    var leaf_prims: DeviceBuffer[DType.uint32]
    var tri_count: Int
    var leaf_pack_ns: Int

    def __init__(
        out self,
        mut ctx: DeviceContext,
        tri_vertices: List[Vec3f32],
    ) raises:
        self.tri_count = len(tri_vertices) / 3
        self.leaf_pack_ns = 0

        var flat_vertices = _flatten_vertices(tri_vertices)
        self.vertices = _copy_f32_to_device(ctx, flat_vertices)

        var leaf_bounds = List[Float32](
            capacity=max(self.tri_count, 1) * GPU_WIDE_BOUNDS_STRIDE
        )
        var payloads = List[UInt32](capacity=max(self.tri_count, 1))

        for i in range(self.tri_count):
            ref v0 = tri_vertices[i * 3 + 0]
            ref v1 = tri_vertices[i * 3 + 1]
            ref v2 = tri_vertices[i * 3 + 2]
            var bmin = vmin(vmin(v0, v1), v2)
            var bmax = vmax(vmax(v0, v1), v2)
            leaf_bounds.append(bmin.x)
            leaf_bounds.append(bmin.y)
            leaf_bounds.append(bmin.z)
            leaf_bounds.append(bmax.x)
            leaf_bounds.append(bmax.y)
            leaf_bounds.append(bmax.z)
            payloads.append(UInt32(i))

        self.tree = GpuBoundsBvh[Self.width](ctx, leaf_bounds, payloads)
        _ = self.tree.build(ctx)

        self.leaf_vertices = ctx.enqueue_create_buffer[DType.float32](
            self.tree.max_leaf_blocks * Self.width * GPU_TRI_LEAF_VERTEX_STRIDE
        )
        self.leaf_prims = ctx.enqueue_create_buffer[DType.uint32](
            self.tree.max_leaf_blocks * Self.width
        )
        self._pack_leaf_blocks(ctx)

    def _pack_leaf_blocks(
        mut self,
        ctx: DeviceContext,
    ) raises:
        var start = perf_counter_ns()
        var blocks = ceildiv(
            max(self.tree.leaf_block_count, 1), GPU_BOUNDS_BVH_BLOCK_SIZE
        )
        ctx.enqueue_function[pack_triangle_leaf_blocks_kernel[Self.width]](
            self.vertices.unsafe_ptr(),
            self.tree.leaf_block_indices.unsafe_ptr(),
            self.leaf_vertices.unsafe_ptr(),
            self.leaf_prims.unsafe_ptr(),
            self.tree.leaf_block_count,
            grid_dim=blocks,
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
        ctx.synchronize()
        self.leaf_pack_ns = Int(perf_counter_ns() - start)

    def launch_uploaded_primary(
        self,
        ctx: DeviceContext,
        d_rays: DeviceBuffer[DType.float32],
        d_hits_f32: DeviceBuffer[DType.float32],
        d_hits_u32: DeviceBuffer[DType.uint32],
        ray_count: Int,
    ) raises:
        ctx.enqueue_function[trace_gpu_triangle_bvh_primary_kernel[Self.width]](
            self.tree.wide_bounds.unsafe_ptr(),
            self.tree.wide_data.unsafe_ptr(),
            self.tree.wide_counts.unsafe_ptr(),
            self.leaf_vertices.unsafe_ptr(),
            self.leaf_prims.unsafe_ptr(),
            self.tree.root_idx,
            d_rays.unsafe_ptr(),
            d_hits_f32.unsafe_ptr(),
            d_hits_u32.unsafe_ptr(),
            ray_count,
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )

    def launch_uploaded_shadow(
        self,
        ctx: DeviceContext,
        d_rays: DeviceBuffer[DType.float32],
        d_flags: DeviceBuffer[DType.uint32],
        ray_count: Int,
    ) raises:
        ctx.enqueue_function[trace_gpu_triangle_bvh_shadow_kernel[Self.width]](
            self.tree.wide_bounds.unsafe_ptr(),
            self.tree.wide_data.unsafe_ptr(),
            self.tree.wide_counts.unsafe_ptr(),
            self.leaf_vertices.unsafe_ptr(),
            self.leaf_prims.unsafe_ptr(),
            self.tree.root_idx,
            d_rays.unsafe_ptr(),
            d_flags.unsafe_ptr(),
            ray_count,
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )


struct GpuSphereBvh[width: Int]:
    var tree: GpuBoundsBvh[Self.width]
    var spheres: DeviceBuffer[DType.float32]
    var leaf_spheres: DeviceBuffer[DType.float32]
    var leaf_prims: DeviceBuffer[DType.uint32]
    var sphere_count: Int
    var leaf_pack_ns: Int

    def __init__(
        out self,
        mut ctx: DeviceContext,
        spheres: List[Sphere],
    ) raises:
        self.sphere_count = len(spheres)
        self.leaf_pack_ns = 0

        var flat_spheres = _flatten_spheres(spheres)
        self.spheres = _copy_f32_to_device(ctx, flat_spheres)

        var leaf_bounds = List[Float32](
            capacity=max(self.sphere_count, 1) * GPU_WIDE_BOUNDS_STRIDE
        )
        var payloads = List[UInt32](capacity=max(self.sphere_count, 1))

        for i in range(self.sphere_count):
            ref s = spheres[i]
            var r = s.radius

            leaf_bounds.append(s.center.x - r)
            leaf_bounds.append(s.center.y - r)
            leaf_bounds.append(s.center.z - r)
            leaf_bounds.append(s.center.x + r)
            leaf_bounds.append(s.center.y + r)
            leaf_bounds.append(s.center.z + r)
            payloads.append(UInt32(i))

        self.tree = GpuBoundsBvh[Self.width](ctx, leaf_bounds, payloads)
        _ = self.tree.build(ctx)

        self.leaf_spheres = ctx.enqueue_create_buffer[DType.float32](
            self.tree.max_leaf_blocks * Self.width * GPU_SPHERE_STRIDE
        )
        self.leaf_prims = ctx.enqueue_create_buffer[DType.uint32](
            self.tree.max_leaf_blocks * Self.width
        )
        self._pack_leaf_blocks(ctx)

    def _pack_leaf_blocks(
        mut self,
        ctx: DeviceContext,
    ) raises:
        var start = perf_counter_ns()
        var blocks = ceildiv(
            max(self.tree.leaf_block_count, 1), GPU_BOUNDS_BVH_BLOCK_SIZE
        )
        ctx.enqueue_function[pack_sphere_leaf_blocks_kernel[Self.width]](
            self.spheres.unsafe_ptr(),
            self.tree.leaf_block_indices.unsafe_ptr(),
            self.leaf_spheres.unsafe_ptr(),
            self.leaf_prims.unsafe_ptr(),
            self.tree.leaf_block_count,
            grid_dim=blocks,
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
        ctx.synchronize()
        self.leaf_pack_ns = Int(perf_counter_ns() - start)

    def launch_uploaded_primary(
        self,
        ctx: DeviceContext,
        d_rays: DeviceBuffer[DType.float32],
        d_hits_f32: DeviceBuffer[DType.float32],
        d_hits_u32: DeviceBuffer[DType.uint32],
        ray_count: Int,
    ) raises:
        ctx.enqueue_function[trace_gpu_sphere_bvh_primary_kernel[Self.width]](
            self.tree.wide_bounds.unsafe_ptr(),
            self.tree.wide_data.unsafe_ptr(),
            self.tree.wide_counts.unsafe_ptr(),
            self.leaf_spheres.unsafe_ptr(),
            self.leaf_prims.unsafe_ptr(),
            self.tree.root_idx,
            d_rays.unsafe_ptr(),
            d_hits_f32.unsafe_ptr(),
            d_hits_u32.unsafe_ptr(),
            ray_count,
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )

    def launch_uploaded_shadow(
        self,
        ctx: DeviceContext,
        d_rays: DeviceBuffer[DType.float32],
        d_flags: DeviceBuffer[DType.uint32],
        ray_count: Int,
    ) raises:
        ctx.enqueue_function[trace_gpu_sphere_bvh_shadow_kernel[Self.width]](
            self.tree.wide_bounds.unsafe_ptr(),
            self.tree.wide_data.unsafe_ptr(),
            self.tree.wide_counts.unsafe_ptr(),
            self.leaf_spheres.unsafe_ptr(),
            self.leaf_prims.unsafe_ptr(),
            self.tree.root_idx,
            d_rays.unsafe_ptr(),
            d_flags.unsafe_ptr(),
            ray_count,
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )


@always_inline
def _wide_lane_base[width: Int](node_idx: UInt32, lane: Int) -> Int:
    return Int(node_idx) * width + lane


@always_inline
def _wide_bounds_base[width: Int](node_idx: UInt32, lane: Int) -> Int:
    return _wide_lane_base[width](node_idx, lane) * GPU_WIDE_BOUNDS_STRIDE


@always_inline
def _intersect_wide_lane_bounds[
    width: Int
](
    wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    node_idx: UInt32,
    lane: Int,
    ray: RayFlat,
    t_max: Float32,
) -> Bool:
    var b = _wide_bounds_base[width](node_idx, lane)
    var h = intersect_ray_aabb(
        ray.o,
        ray.rd,
        Vec3f32.load(wide_bounds, b + 0),
        Vec3f32.load(wide_bounds, b + 3),
        t_max,
    )
    return h.mask


@always_inline
def _intersect_triangle_leaf_block[
    width: Int,
    mode: String,
](
    leaf_vertices: UnsafePointer[Float32, MutAnyOrigin],
    leaf_prims: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_block_idx: UInt32,
    item_count: UInt32,
    ray: RayFlat,
    mut best_t: Float32,
    mut best_u: Float32,
    mut best_v: Float32,
    mut best_prim: UInt32,
) -> Bool:
    comptime assert mode in [TRACE_PRIMARY_FULL, TRACE_PRIMARY_T, TRACE_SHADOW]

    var hit_any = False

    comptime for lane in range(width):
        if lane < Int(item_count):
            var idx = Int(leaf_block_idx) * width + lane
            var prim = UInt32(leaf_prims[idx])

            if prim != GPU_WIDE_EMPTY_LANE:
                var base = idx * GPU_TRI_LEAF_VERTEX_STRIDE
                var v0 = Vec3f32.load(leaf_vertices, base + 0)
                var v1 = Vec3f32.load(leaf_vertices, base + 3)
                var v2 = Vec3f32.load(leaf_vertices, base + 6)

                var tri_hit = intersect_ray_tri(
                    ray.o,
                    ray.d,
                    v0,
                    v1,
                    v2,
                    best_t,
                )

                if tri_hit.mask:
                    comptime if mode == TRACE_SHADOW:
                        return True
                    else:
                        hit_any = True
                        best_t = tri_hit.t
                        comptime if mode == TRACE_PRIMARY_FULL:
                            best_u = tri_hit.u
                            best_v = tri_hit.v
                            best_prim = prim

    return hit_any


@always_inline
def _intersect_sphere_leaf_block[
    width: Int,
    mode: String,
](
    leaf_spheres: UnsafePointer[Float32, MutAnyOrigin],
    leaf_prims: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_block_idx: UInt32,
    item_count: UInt32,
    ray: RayFlat,
    mut best_t: Float32,
    mut best_u: Float32,
    mut best_v: Float32,
    mut best_prim: UInt32,
) -> Bool:
    comptime assert mode in [TRACE_PRIMARY_FULL, TRACE_PRIMARY_T, TRACE_SHADOW]

    var hit_any = False

    comptime for lane in range(width):
        if lane < Int(item_count):
            var idx = Int(leaf_block_idx) * width + lane
            var prim = UInt32(leaf_prims[idx])

            if prim != GPU_WIDE_EMPTY_LANE:
                var base = idx * GPU_SPHERE_STRIDE
                var center = Vec3f32(
                    leaf_spheres[base + 0],
                    leaf_spheres[base + 1],
                    leaf_spheres[base + 2],
                )
                var radius = leaf_spheres[base + 3]

                var oc = ray.o - center
                var a = (
                    ray.d.x * ray.d.x + ray.d.y * ray.d.y + ray.d.z * ray.d.z
                )
                var half_b = oc.x * ray.d.x + oc.y * ray.d.y + oc.z * ray.d.z
                var c = (
                    oc.x * oc.x + oc.y * oc.y + oc.z * oc.z - radius * radius
                )

                var det = half_b * half_b - a * c
                if det >= 0.0:
                    var sqrt_det = sqrt(det)
                    var inv_a = 1.0 / a

                    var t = (-half_b - sqrt_det) * inv_a
                    if not (t > 1.0e-4 and t < best_t):
                        t = (-half_b + sqrt_det) * inv_a

                    if t > 1.0e-4 and t < best_t:
                        comptime if mode == TRACE_SHADOW:
                            return True
                        else:
                            hit_any = True
                            best_t = t
                            comptime if mode == TRACE_PRIMARY_FULL:
                                best_u = 0.0
                                best_v = 0.0
                                best_prim = prim

    return hit_any


@always_inline
def trace_gpu_wide_triangle_ray[
    width: Int,
    mode: String = TRACE_PRIMARY_FULL,
](
    wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_vertices: UnsafePointer[Float32, MutAnyOrigin],
    leaf_prims: UnsafePointer[UInt32, MutAnyOrigin],
    root_idx: UInt32,
    ray: RayFlat,
) -> Hit:
    comptime assert mode in [TRACE_PRIMARY_FULL, TRACE_PRIMARY_T, TRACE_SHADOW]

    var best_t = ray.t_max
    var best_u = Float32(0.0)
    var best_v = Float32(0.0)
    var best_prim = _gpu_miss_prim

    var stack = InlineArray[UInt32, GPU_TRAVERSAL_STACK_SIZE](fill=0)
    var stack_ptr = 0
    var current = root_idx

    while True:
        comptime for lane in range(width):
            var lane_base = _wide_lane_base[width](current, lane)
            var count = UInt32(wide_counts[lane_base])

            if count != GPU_WIDE_EMPTY_LANE:
                if _intersect_wide_lane_bounds[width](
                    wide_bounds, current, lane, ray, best_t
                ):
                    var data = UInt32(wide_data[lane_base])
                    if count == 0:
                        if stack_ptr < GPU_TRAVERSAL_STACK_SIZE:
                            stack[stack_ptr] = data
                            stack_ptr += 1
                    else:
                        var leaf_hit = _intersect_triangle_leaf_block[
                            width, mode
                        ](
                            leaf_vertices,
                            leaf_prims,
                            data,
                            count,
                            ray,
                            best_t,
                            best_u,
                            best_v,
                            best_prim,
                        )
                        comptime if mode == TRACE_SHADOW:
                            if leaf_hit:
                                return Hit(0.0, 0.0, 0.0, best_prim, UInt32(1))

        if stack_ptr == 0:
            break

        stack_ptr -= 1
        current = stack[stack_ptr]

    return Hit(best_t, best_u, best_v, best_prim, UInt32(0))


@always_inline
def trace_gpu_wide_sphere_ray[
    width: Int,
    mode: String = TRACE_PRIMARY_FULL,
](
    wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_spheres: UnsafePointer[Float32, MutAnyOrigin],
    leaf_prims: UnsafePointer[UInt32, MutAnyOrigin],
    root_idx: UInt32,
    ray: RayFlat,
) -> Hit:
    comptime assert mode in [TRACE_PRIMARY_FULL, TRACE_PRIMARY_T, TRACE_SHADOW]

    var best_t = ray.t_max
    var best_u = Float32(0.0)
    var best_v = Float32(0.0)
    var best_prim = _gpu_miss_prim

    var stack = InlineArray[UInt32, GPU_TRAVERSAL_STACK_SIZE](fill=0)
    var stack_ptr = 0
    var current = root_idx

    while True:
        comptime for lane in range(width):
            var lane_base = _wide_lane_base[width](current, lane)
            var count = UInt32(wide_counts[lane_base])

            if count != GPU_WIDE_EMPTY_LANE:
                if _intersect_wide_lane_bounds[width](
                    wide_bounds, current, lane, ray, best_t
                ):
                    var data = UInt32(wide_data[lane_base])
                    if count == 0:
                        if stack_ptr < GPU_TRAVERSAL_STACK_SIZE:
                            stack[stack_ptr] = data
                            stack_ptr += 1
                    else:
                        var leaf_hit = _intersect_sphere_leaf_block[
                            width, mode
                        ](
                            leaf_spheres,
                            leaf_prims,
                            data,
                            count,
                            ray,
                            best_t,
                            best_u,
                            best_v,
                            best_prim,
                        )
                        comptime if mode == TRACE_SHADOW:
                            if leaf_hit:
                                return Hit(0.0, 0.0, 0.0, best_prim, UInt32(1))

        if stack_ptr == 0:
            break

        stack_ptr -= 1
        current = stack[stack_ptr]

    return Hit(best_t, best_u, best_v, best_prim, UInt32(0))


def trace_gpu_triangle_bvh_primary_kernel[
    width: Int,
](
    wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_vertices: UnsafePointer[Float32, MutAnyOrigin],
    leaf_prims: UnsafePointer[UInt32, MutAnyOrigin],
    root_idx: UInt32,
    rays: UnsafePointer[Float32, MutAnyOrigin],
    hits_f32: UnsafePointer[Float32, MutAnyOrigin],
    hits_u32: UnsafePointer[UInt32, MutAnyOrigin],
    ray_count: Int,
):
    var ray_idx = global_idx.x
    if ray_idx >= ray_count:
        return

    var ray = RayFlat(rays, ray_idx)
    var hit = trace_gpu_wide_triangle_ray[width, TRACE_PRIMARY_FULL](
        wide_bounds,
        wide_data,
        wide_counts,
        leaf_vertices,
        leaf_prims,
        root_idx,
        ray,
    )

    var hit_base = ray_idx * 3
    hits_f32[hit_base + 0] = hit.t
    hits_f32[hit_base + 1] = hit.u
    hits_f32[hit_base + 2] = hit.v
    hits_u32[ray_idx] = hit.prim


def trace_gpu_sphere_bvh_primary_kernel[
    width: Int,
](
    wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_spheres: UnsafePointer[Float32, MutAnyOrigin],
    leaf_prims: UnsafePointer[UInt32, MutAnyOrigin],
    root_idx: UInt32,
    rays: UnsafePointer[Float32, MutAnyOrigin],
    hits_f32: UnsafePointer[Float32, MutAnyOrigin],
    hits_u32: UnsafePointer[UInt32, MutAnyOrigin],
    ray_count: Int,
):
    var ray_idx = global_idx.x
    if ray_idx >= ray_count:
        return

    var ray = RayFlat(rays, ray_idx)
    var hit = trace_gpu_wide_sphere_ray[width, TRACE_PRIMARY_FULL](
        wide_bounds,
        wide_data,
        wide_counts,
        leaf_spheres,
        leaf_prims,
        root_idx,
        ray,
    )

    var hit_base = ray_idx * 3
    hits_f32[hit_base + 0] = hit.t
    hits_f32[hit_base + 1] = hit.u
    hits_f32[hit_base + 2] = hit.v
    hits_u32[ray_idx] = hit.prim


def trace_gpu_triangle_bvh_shadow_kernel[
    width: Int,
](
    wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_vertices: UnsafePointer[Float32, MutAnyOrigin],
    leaf_prims: UnsafePointer[UInt32, MutAnyOrigin],
    root_idx: UInt32,
    rays: UnsafePointer[Float32, MutAnyOrigin],
    flags: UnsafePointer[UInt32, MutAnyOrigin],
    ray_count: Int,
):
    var ray_idx = global_idx.x
    if ray_idx >= ray_count:
        return

    var ray = RayFlat(rays, ray_idx)
    var hit = trace_gpu_wide_triangle_ray[width, TRACE_SHADOW](
        wide_bounds,
        wide_data,
        wide_counts,
        leaf_vertices,
        leaf_prims,
        root_idx,
        ray,
    )

    flags[ray_idx] = hit.occluded


def trace_gpu_sphere_bvh_shadow_kernel[
    width: Int,
](
    wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    leaf_spheres: UnsafePointer[Float32, MutAnyOrigin],
    leaf_prims: UnsafePointer[UInt32, MutAnyOrigin],
    root_idx: UInt32,
    rays: UnsafePointer[Float32, MutAnyOrigin],
    flags: UnsafePointer[UInt32, MutAnyOrigin],
    ray_count: Int,
):
    var ray_idx = global_idx.x
    if ray_idx >= ray_count:
        return

    var ray = RayFlat(rays, ray_idx)
    var hit = trace_gpu_wide_sphere_ray[width, TRACE_SHADOW](
        wide_bounds,
        wide_data,
        wide_counts,
        leaf_spheres,
        leaf_prims,
        root_idx,
        ray,
    )

    flags[ray_idx] = hit.occluded


# Host utility copies local to this module to avoid depending on old triangle-only
# GPU host helpers.
def _copy_f32_to_device(
    mut ctx: DeviceContext,
    values: List[Float32],
) raises -> DeviceBuffer[DType.float32]:
    var buf = ctx.enqueue_create_buffer[DType.float32](max(len(values), 1))
    with buf.map_to_host() as h:
        for i in range(len(values)):
            h[i] = values[i]
    return buf^


def _copy_u32_to_device(
    mut ctx: DeviceContext,
    values: List[UInt32],
) raises -> DeviceBuffer[DType.uint32]:
    var buf = ctx.enqueue_create_buffer[DType.uint32](max(len(values), 1))
    with buf.map_to_host() as h:
        for i in range(len(values)):
            h[i] = values[i]
    return buf^


def _flatten_vertices(verts: List[Vec3f32]) -> List[Float32]:
    var out = List[Float32](capacity=max(len(verts), 1) * 3)
    for i in range(len(verts)):
        out.append(verts[i].x)
        out.append(verts[i].y)
        out.append(verts[i].z)
    return out^


def _flatten_spheres(spheres: List[Sphere]) -> List[Float32]:
    var out = List[Float32](capacity=max(len(spheres), 1) * GPU_SPHERE_STRIDE)
    for i in range(len(spheres)):
        out.append(spheres[i].center.x)
        out.append(spheres[i].center.y)
        out.append(spheres[i].center.z)
        out.append(spheres[i].radius)
    return out^


# -----------------------------------------------------------------------------
# GPU TLAS over GpuTriangleBvh[blas_width]
# -----------------------------------------------------------------------------

comptime GPU_TLAS_TRANSFORM_STRIDE = 16


@fieldwise_init
struct GpuTlasInstance(Copyable):
    """Host-side TLAS instance descriptor for the new GPU wide BVH path.

    The first implementation supports many instances of one shared BLAS.
    `blas_idx` is kept in the layout so the API does not need to change when
    same-type multi-BLAS support is added later.
    """

    var bounds: AABB
    var inv_transform: Mat44f32
    var blas_idx: UInt32

    @always_inline
    def __init__(out self):
        self.bounds = AABB.invalid()
        self.inv_transform = Mat44f32.identity()
        self.blas_idx = UInt32(0)


def _flatten_instance_inv_transforms(
    instances: List[GpuTlasInstance],
) -> List[Float32]:
    var out = List[Float32](
        capacity=max(len(instances), 1) * GPU_TLAS_TRANSFORM_STRIDE
    )
    for i in range(len(instances)):
        ref m = instances[i].inv_transform
        comptime for j in range(GPU_TLAS_TRANSFORM_STRIDE):
            comptime row = j / 4
            comptime col = j - row * 4
            out.append(m[row][col])
    return out^


def _flatten_instance_blas_indices(
    instances: List[GpuTlasInstance],
) -> List[UInt32]:
    return [instance.blas_idx for instance in instances]


@always_inline
def _safe_rcp_gpu_tlas(x: Float32) -> Float32:
    if x == 0.0:
        return _gpu_inf_t
    return 1.0 / x


@always_inline
def _make_tlas_local_ray(
    inst_inv_transform: UnsafePointer[Float32, MutAnyOrigin],
    inst_idx: UInt32,
    ray: RayFlat,
    t_max: Float32,
) -> RayFlat:
    var base = Int(inst_idx) * GPU_TLAS_TRANSFORM_STRIDE
    var o = transform_point(inst_inv_transform, base, ray.o)
    var d = transform_vector(inst_inv_transform, base, ray.d)
    return RayFlat(
        o,
        d,
        Vec3f32(
            _safe_rcp_gpu_tlas(d.x),
            _safe_rcp_gpu_tlas(d.y),
            _safe_rcp_gpu_tlas(d.z),
        ),
        t_max,
    )


@always_inline
def _intersect_tlas_triangle_instance_block[
    tlas_width: Int,
    blas_width: Int,
    mode: String,
](
    tlas_leaf_instances: UnsafePointer[UInt32, MutAnyOrigin],
    inst_inv_transform: UnsafePointer[Float32, MutAnyOrigin],
    inst_blas_indices: UnsafePointer[UInt32, MutAnyOrigin],
    blas_wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    blas_wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    blas_wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    blas_leaf_vertices: UnsafePointer[Float32, MutAnyOrigin],
    blas_leaf_prims: UnsafePointer[UInt32, MutAnyOrigin],
    blas_root_idx: UInt32,
    leaf_block_idx: UInt32,
    item_count: UInt32,
    ray: RayFlat,
    mut best_hit: Hit,
    mut best_inst: UInt32,
) -> Bool:
    comptime assert mode in [TRACE_PRIMARY_FULL, TRACE_PRIMARY_T, TRACE_SHADOW]

    var hit_any = False

    comptime for lane in range(tlas_width):
        if lane < Int(item_count):
            var idx = Int(leaf_block_idx) * tlas_width + lane
            var inst_idx = UInt32(tlas_leaf_instances[idx])

            if inst_idx != GPU_WIDE_EMPTY_LANE:
                # Current implementation supports one shared BLAS. Keep the
                # blas index buffer in place for the later multi-BLAS table.
                var blas_idx = UInt32(inst_blas_indices[Int(inst_idx)])
                if blas_idx == UInt32(0):
                    var local_ray = _make_tlas_local_ray(
                        inst_inv_transform,
                        inst_idx,
                        ray,
                        best_hit.t,
                    )
                    var local_hit = trace_gpu_wide_triangle_ray[
                        blas_width, mode
                    ](
                        blas_wide_bounds,
                        blas_wide_data,
                        blas_wide_counts,
                        blas_leaf_vertices,
                        blas_leaf_prims,
                        blas_root_idx,
                        local_ray,
                    )

                    if local_hit.t < best_hit.t:
                        comptime if mode == TRACE_SHADOW:
                            return True
                        else:
                            best_hit = local_hit
                            best_inst = inst_idx
                            hit_any = True

    return hit_any


@always_inline
def trace_gpu_wide_tlas_triangle_ray[
    tlas_width: Int,
    blas_width: Int,
    mode: String = TRACE_PRIMARY_FULL,
](
    tlas_wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    tlas_wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_leaf_instances: UnsafePointer[UInt32, MutAnyOrigin],
    inst_inv_transform: UnsafePointer[Float32, MutAnyOrigin],
    inst_blas_indices: UnsafePointer[UInt32, MutAnyOrigin],
    blas_wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    blas_wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    blas_wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    blas_leaf_vertices: UnsafePointer[Float32, MutAnyOrigin],
    blas_leaf_prims: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_root_idx: UInt32,
    blas_root_idx: UInt32,
    ray: RayFlat,
) -> Tuple[Hit, UInt32]:
    comptime assert mode in [TRACE_PRIMARY_FULL, TRACE_PRIMARY_T, TRACE_SHADOW]

    var best_hit = Hit(ray.t_max, 0.0, 0.0, _gpu_miss_prim, UInt32(0))
    var best_inst = _gpu_miss_prim

    var stack = InlineArray[UInt32, GPU_TRAVERSAL_STACK_SIZE](fill=0)
    var stack_ptr = 0
    var current = tlas_root_idx

    while True:
        comptime for lane in range(tlas_width):
            var lane_base = _wide_lane_base[tlas_width](current, lane)
            var count = UInt32(tlas_wide_counts[lane_base])

            if count != GPU_WIDE_EMPTY_LANE:
                if _intersect_wide_lane_bounds[tlas_width](
                    tlas_wide_bounds,
                    current,
                    lane,
                    ray,
                    best_hit.t,
                ):
                    var data = UInt32(tlas_wide_data[lane_base])
                    if count == 0:
                        if stack_ptr < GPU_TRAVERSAL_STACK_SIZE:
                            stack[stack_ptr] = data
                            stack_ptr += 1
                    else:
                        var leaf_hit = _intersect_tlas_triangle_instance_block[
                            tlas_width,
                            blas_width,
                            mode,
                        ](
                            tlas_leaf_instances,
                            inst_inv_transform,
                            inst_blas_indices,
                            blas_wide_bounds,
                            blas_wide_data,
                            blas_wide_counts,
                            blas_leaf_vertices,
                            blas_leaf_prims,
                            blas_root_idx,
                            data,
                            count,
                            ray,
                            best_hit,
                            best_inst,
                        )
                        comptime if mode == TRACE_SHADOW:
                            if leaf_hit:
                                return (
                                    Hit(
                                        0.0, 0.0, 0.0, _gpu_miss_prim, UInt32(1)
                                    ),
                                    best_inst,
                                )

        if stack_ptr == 0:
            break

        stack_ptr -= 1
        current = stack[stack_ptr]

    return (best_hit, best_inst)


@always_inline
def _intersect_tlas_sphere_instance_block[
    tlas_width: Int,
    blas_width: Int,
    mode: String,
](
    tlas_leaf_instances: UnsafePointer[UInt32, MutAnyOrigin],
    inst_inv_transform: UnsafePointer[Float32, MutAnyOrigin],
    inst_blas_indices: UnsafePointer[UInt32, MutAnyOrigin],
    blas_wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    blas_wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    blas_wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    blas_leaf_spheres: UnsafePointer[Float32, MutAnyOrigin],
    blas_leaf_prims: UnsafePointer[UInt32, MutAnyOrigin],
    blas_root_idx: UInt32,
    leaf_block_idx: UInt32,
    item_count: UInt32,
    ray: RayFlat,
    mut best_hit: Hit,
    mut best_inst: UInt32,
) -> Bool:
    comptime assert mode in [TRACE_PRIMARY_FULL, TRACE_PRIMARY_T, TRACE_SHADOW]

    var hit_any = False

    comptime for lane in range(tlas_width):
        if lane < Int(item_count):
            var idx = Int(leaf_block_idx) * tlas_width + lane
            var inst_idx = UInt32(tlas_leaf_instances[idx])

            if inst_idx != GPU_WIDE_EMPTY_LANE:
                var blas_idx = UInt32(inst_blas_indices[Int(inst_idx)])
                if blas_idx == UInt32(0):
                    var local_ray = _make_tlas_local_ray(
                        inst_inv_transform,
                        inst_idx,
                        ray,
                        best_hit.t,
                    )
                    var local_hit = trace_gpu_wide_sphere_ray[blas_width, mode](
                        blas_wide_bounds,
                        blas_wide_data,
                        blas_wide_counts,
                        blas_leaf_spheres,
                        blas_leaf_prims,
                        blas_root_idx,
                        local_ray,
                    )

                    if local_hit.t < best_hit.t:
                        comptime if mode == TRACE_SHADOW:
                            return True
                        else:
                            best_hit = local_hit
                            best_inst = inst_idx
                            hit_any = True

    return hit_any


@always_inline
def trace_gpu_wide_tlas_sphere_ray[
    tlas_width: Int,
    blas_width: Int,
    mode: String = TRACE_PRIMARY_FULL,
](
    tlas_wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    tlas_wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_leaf_instances: UnsafePointer[UInt32, MutAnyOrigin],
    inst_inv_transform: UnsafePointer[Float32, MutAnyOrigin],
    inst_blas_indices: UnsafePointer[UInt32, MutAnyOrigin],
    blas_wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    blas_wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    blas_wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    blas_leaf_spheres: UnsafePointer[Float32, MutAnyOrigin],
    blas_leaf_prims: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_root_idx: UInt32,
    blas_root_idx: UInt32,
    ray: RayFlat,
) -> Tuple[Hit, UInt32]:
    comptime assert mode in [TRACE_PRIMARY_FULL, TRACE_PRIMARY_T, TRACE_SHADOW]

    var best_hit = Hit(ray.t_max, 0.0, 0.0, _gpu_miss_prim, UInt32(0))
    var best_inst = _gpu_miss_prim

    var stack = InlineArray[UInt32, GPU_TRAVERSAL_STACK_SIZE](fill=0)
    var stack_ptr = 0
    var current = tlas_root_idx

    while True:
        comptime for lane in range(tlas_width):
            var lane_base = _wide_lane_base[tlas_width](current, lane)
            var count = UInt32(tlas_wide_counts[lane_base])

            if count != GPU_WIDE_EMPTY_LANE:
                if _intersect_wide_lane_bounds[tlas_width](
                    tlas_wide_bounds,
                    current,
                    lane,
                    ray,
                    best_hit.t,
                ):
                    var data = UInt32(tlas_wide_data[lane_base])
                    if count == 0:
                        if stack_ptr < GPU_TRAVERSAL_STACK_SIZE:
                            stack[stack_ptr] = data
                            stack_ptr += 1
                    else:
                        var leaf_hit = _intersect_tlas_sphere_instance_block[
                            tlas_width,
                            blas_width,
                            mode,
                        ](
                            tlas_leaf_instances,
                            inst_inv_transform,
                            inst_blas_indices,
                            blas_wide_bounds,
                            blas_wide_data,
                            blas_wide_counts,
                            blas_leaf_spheres,
                            blas_leaf_prims,
                            blas_root_idx,
                            data,
                            count,
                            ray,
                            best_hit,
                            best_inst,
                        )
                        comptime if mode == TRACE_SHADOW:
                            if leaf_hit:
                                return (
                                    Hit(
                                        0.0, 0.0, 0.0, _gpu_miss_prim, UInt32(1)
                                    ),
                                    best_inst,
                                )

        if stack_ptr == 0:
            break

        stack_ptr -= 1
        current = stack[stack_ptr]

    return (best_hit, best_inst)


struct GpuTlas[width: Int]:
    """GPU TLAS built with the same generic GpuBoundsBvh[width] as BLAS.

    This first implementation supports many transformed instances of one shared
    triangle BLAS.  Instance leaves are packed by the generic wide collapse:
    `tree.leaf_block_indices[leaf_block * width + lane]` stores the instance id.
    """

    var tree: GpuBoundsBvh[Self.width]
    var inst_inv_transform: DeviceBuffer[DType.float32]
    var inst_blas_indices: DeviceBuffer[DType.uint32]
    var inst_count: Int

    def __init__(
        out self,
        mut ctx: DeviceContext,
        instances: List[GpuTlasInstance],
    ) raises:
        self.inst_count = len(instances)

        var leaf_bounds = List[Float32](
            capacity=max(self.inst_count, 1) * GPU_WIDE_BOUNDS_STRIDE
        )
        var payloads = List[UInt32](capacity=max(self.inst_count, 1))

        for i in range(self.inst_count):
            ref inst = instances[i]
            leaf_bounds.append(inst.bounds._min.x)
            leaf_bounds.append(inst.bounds._min.y)
            leaf_bounds.append(inst.bounds._min.z)
            leaf_bounds.append(inst.bounds._max.x)
            leaf_bounds.append(inst.bounds._max.y)
            leaf_bounds.append(inst.bounds._max.z)
            payloads.append(UInt32(i))

        self.tree = GpuBoundsBvh[Self.width](ctx, leaf_bounds, payloads)
        _ = self.tree.build(ctx)

        self.inst_inv_transform = _copy_f32_to_device(
            ctx, _flatten_instance_inv_transforms(instances)
        )
        self.inst_blas_indices = _copy_u32_to_device(
            ctx, _flatten_instance_blas_indices(instances)
        )

    def launch_uploaded_triangle_primary[
        blas_width: Int,
    ](
        self,
        ctx: DeviceContext,
        blas: GpuTriangleBvh[blas_width],
        d_rays: DeviceBuffer[DType.float32],
        d_hits_f32: DeviceBuffer[DType.float32],
        d_hits_u32: DeviceBuffer[DType.uint32],
        ray_count: Int,
    ) raises:
        ctx.enqueue_function[
            trace_gpu_tlas_triangle_primary_kernel[Self.width, blas_width]
        ](
            self.tree.wide_bounds.unsafe_ptr(),
            self.tree.wide_data.unsafe_ptr(),
            self.tree.wide_counts.unsafe_ptr(),
            self.tree.leaf_block_indices.unsafe_ptr(),
            self.inst_inv_transform.unsafe_ptr(),
            self.inst_blas_indices.unsafe_ptr(),
            blas.tree.wide_bounds.unsafe_ptr(),
            blas.tree.wide_data.unsafe_ptr(),
            blas.tree.wide_counts.unsafe_ptr(),
            blas.leaf_vertices.unsafe_ptr(),
            blas.leaf_prims.unsafe_ptr(),
            self.tree.root_idx,
            blas.tree.root_idx,
            d_rays.unsafe_ptr(),
            d_hits_f32.unsafe_ptr(),
            d_hits_u32.unsafe_ptr(),
            ray_count,
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )

    def launch_uploaded_triangle_shadow[
        blas_width: Int,
    ](
        self,
        ctx: DeviceContext,
        blas: GpuTriangleBvh[blas_width],
        d_rays: DeviceBuffer[DType.float32],
        d_flags: DeviceBuffer[DType.uint32],
        ray_count: Int,
    ) raises:
        ctx.enqueue_function[
            trace_gpu_tlas_triangle_shadow_kernel[Self.width, blas_width]
        ](
            self.tree.wide_bounds.unsafe_ptr(),
            self.tree.wide_data.unsafe_ptr(),
            self.tree.wide_counts.unsafe_ptr(),
            self.tree.leaf_block_indices.unsafe_ptr(),
            self.inst_inv_transform.unsafe_ptr(),
            self.inst_blas_indices.unsafe_ptr(),
            blas.tree.wide_bounds.unsafe_ptr(),
            blas.tree.wide_data.unsafe_ptr(),
            blas.tree.wide_counts.unsafe_ptr(),
            blas.leaf_vertices.unsafe_ptr(),
            blas.leaf_prims.unsafe_ptr(),
            self.tree.root_idx,
            blas.tree.root_idx,
            d_rays.unsafe_ptr(),
            d_flags.unsafe_ptr(),
            ray_count,
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )

    def launch_uploaded_sphere_primary[
        blas_width: Int,
    ](
        self,
        ctx: DeviceContext,
        blas: GpuSphereBvh[blas_width],
        d_rays: DeviceBuffer[DType.float32],
        d_hits_f32: DeviceBuffer[DType.float32],
        d_hits_u32: DeviceBuffer[DType.uint32],
        ray_count: Int,
    ) raises:
        ctx.enqueue_function[
            trace_gpu_tlas_sphere_primary_kernel[Self.width, blas_width]
        ](
            self.tree.wide_bounds.unsafe_ptr(),
            self.tree.wide_data.unsafe_ptr(),
            self.tree.wide_counts.unsafe_ptr(),
            self.tree.leaf_block_indices.unsafe_ptr(),
            self.inst_inv_transform.unsafe_ptr(),
            self.inst_blas_indices.unsafe_ptr(),
            blas.tree.wide_bounds.unsafe_ptr(),
            blas.tree.wide_data.unsafe_ptr(),
            blas.tree.wide_counts.unsafe_ptr(),
            blas.leaf_spheres.unsafe_ptr(),
            blas.leaf_prims.unsafe_ptr(),
            self.tree.root_idx,
            blas.tree.root_idx,
            d_rays.unsafe_ptr(),
            d_hits_f32.unsafe_ptr(),
            d_hits_u32.unsafe_ptr(),
            ray_count,
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )

    def launch_uploaded_sphere_shadow[
        blas_width: Int,
    ](
        self,
        ctx: DeviceContext,
        blas: GpuSphereBvh[blas_width],
        d_rays: DeviceBuffer[DType.float32],
        d_flags: DeviceBuffer[DType.uint32],
        ray_count: Int,
    ) raises:
        ctx.enqueue_function[
            trace_gpu_tlas_sphere_shadow_kernel[Self.width, blas_width]
        ](
            self.tree.wide_bounds.unsafe_ptr(),
            self.tree.wide_data.unsafe_ptr(),
            self.tree.wide_counts.unsafe_ptr(),
            self.tree.leaf_block_indices.unsafe_ptr(),
            self.inst_inv_transform.unsafe_ptr(),
            self.inst_blas_indices.unsafe_ptr(),
            blas.tree.wide_bounds.unsafe_ptr(),
            blas.tree.wide_data.unsafe_ptr(),
            blas.tree.wide_counts.unsafe_ptr(),
            blas.leaf_spheres.unsafe_ptr(),
            blas.leaf_prims.unsafe_ptr(),
            self.tree.root_idx,
            blas.tree.root_idx,
            d_rays.unsafe_ptr(),
            d_flags.unsafe_ptr(),
            ray_count,
            grid_dim=ceildiv(ray_count, GPU_BOUNDS_BVH_BLOCK_SIZE),
            block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
        )


def trace_gpu_tlas_triangle_primary_kernel[
    tlas_width: Int,
    blas_width: Int,
](
    tlas_wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    tlas_wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_leaf_instances: UnsafePointer[UInt32, MutAnyOrigin],
    inst_inv_transform: UnsafePointer[Float32, MutAnyOrigin],
    inst_blas_indices: UnsafePointer[UInt32, MutAnyOrigin],
    blas_wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    blas_wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    blas_wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    blas_leaf_vertices: UnsafePointer[Float32, MutAnyOrigin],
    blas_leaf_prims: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_root_idx: UInt32,
    blas_root_idx: UInt32,
    rays: UnsafePointer[Float32, MutAnyOrigin],
    hits_f32: UnsafePointer[Float32, MutAnyOrigin],
    hits_u32: UnsafePointer[UInt32, MutAnyOrigin],
    ray_count: Int,
):
    var ray_idx = global_idx.x
    if ray_idx >= ray_count:
        return

    var ray = RayFlat(rays, ray_idx)
    var result = trace_gpu_wide_tlas_triangle_ray[
        tlas_width,
        blas_width,
        TRACE_PRIMARY_FULL,
    ](
        tlas_wide_bounds,
        tlas_wide_data,
        tlas_wide_counts,
        tlas_leaf_instances,
        inst_inv_transform,
        inst_blas_indices,
        blas_wide_bounds,
        blas_wide_data,
        blas_wide_counts,
        blas_leaf_vertices,
        blas_leaf_prims,
        tlas_root_idx,
        blas_root_idx,
        ray,
    )

    var hit = result[0]
    var inst = result[1]

    var hit_base = ray_idx * 3
    hits_f32[hit_base + 0] = hit.t
    hits_f32[hit_base + 1] = hit.u
    hits_f32[hit_base + 2] = hit.v

    var ubase = ray_idx * 2
    hits_u32[ubase + 0] = hit.prim
    hits_u32[ubase + 1] = inst


def trace_gpu_tlas_triangle_shadow_kernel[
    tlas_width: Int,
    blas_width: Int,
](
    tlas_wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    tlas_wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_leaf_instances: UnsafePointer[UInt32, MutAnyOrigin],
    inst_inv_transform: UnsafePointer[Float32, MutAnyOrigin],
    inst_blas_indices: UnsafePointer[UInt32, MutAnyOrigin],
    blas_wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    blas_wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    blas_wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    blas_leaf_vertices: UnsafePointer[Float32, MutAnyOrigin],
    blas_leaf_prims: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_root_idx: UInt32,
    blas_root_idx: UInt32,
    rays: UnsafePointer[Float32, MutAnyOrigin],
    flags: UnsafePointer[UInt32, MutAnyOrigin],
    ray_count: Int,
):
    var ray_idx = global_idx.x
    if ray_idx >= ray_count:
        return

    var ray = RayFlat(rays, ray_idx)
    var result = trace_gpu_wide_tlas_triangle_ray[
        tlas_width,
        blas_width,
        TRACE_SHADOW,
    ](
        tlas_wide_bounds,
        tlas_wide_data,
        tlas_wide_counts,
        tlas_leaf_instances,
        inst_inv_transform,
        inst_blas_indices,
        blas_wide_bounds,
        blas_wide_data,
        blas_wide_counts,
        blas_leaf_vertices,
        blas_leaf_prims,
        tlas_root_idx,
        blas_root_idx,
        ray,
    )

    flags[ray_idx] = result[0].occluded


def trace_gpu_tlas_sphere_primary_kernel[
    tlas_width: Int,
    blas_width: Int,
](
    tlas_wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    tlas_wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_leaf_instances: UnsafePointer[UInt32, MutAnyOrigin],
    inst_inv_transform: UnsafePointer[Float32, MutAnyOrigin],
    inst_blas_indices: UnsafePointer[UInt32, MutAnyOrigin],
    blas_wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    blas_wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    blas_wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    blas_leaf_spheres: UnsafePointer[Float32, MutAnyOrigin],
    blas_leaf_prims: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_root_idx: UInt32,
    blas_root_idx: UInt32,
    rays: UnsafePointer[Float32, MutAnyOrigin],
    hits_f32: UnsafePointer[Float32, MutAnyOrigin],
    hits_u32: UnsafePointer[UInt32, MutAnyOrigin],
    ray_count: Int,
):
    var ray_idx = global_idx.x
    if ray_idx >= ray_count:
        return

    var ray = RayFlat(rays, ray_idx)
    var result = trace_gpu_wide_tlas_sphere_ray[
        tlas_width,
        blas_width,
        TRACE_PRIMARY_FULL,
    ](
        tlas_wide_bounds,
        tlas_wide_data,
        tlas_wide_counts,
        tlas_leaf_instances,
        inst_inv_transform,
        inst_blas_indices,
        blas_wide_bounds,
        blas_wide_data,
        blas_wide_counts,
        blas_leaf_spheres,
        blas_leaf_prims,
        tlas_root_idx,
        blas_root_idx,
        ray,
    )

    var hit = result[0]
    var inst = result[1]

    var hit_base = ray_idx * 3
    hits_f32[hit_base + 0] = hit.t
    hits_f32[hit_base + 1] = hit.u
    hits_f32[hit_base + 2] = hit.v

    var ubase = ray_idx * 2
    hits_u32[ubase + 0] = hit.prim
    hits_u32[ubase + 1] = inst


def trace_gpu_tlas_sphere_shadow_kernel[
    tlas_width: Int,
    blas_width: Int,
](
    tlas_wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    tlas_wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_leaf_instances: UnsafePointer[UInt32, MutAnyOrigin],
    inst_inv_transform: UnsafePointer[Float32, MutAnyOrigin],
    inst_blas_indices: UnsafePointer[UInt32, MutAnyOrigin],
    blas_wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    blas_wide_data: UnsafePointer[UInt32, MutAnyOrigin],
    blas_wide_counts: UnsafePointer[UInt32, MutAnyOrigin],
    blas_leaf_spheres: UnsafePointer[Float32, MutAnyOrigin],
    blas_leaf_prims: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_root_idx: UInt32,
    blas_root_idx: UInt32,
    rays: UnsafePointer[Float32, MutAnyOrigin],
    flags: UnsafePointer[UInt32, MutAnyOrigin],
    ray_count: Int,
):
    var ray_idx = global_idx.x
    if ray_idx >= ray_count:
        return

    var ray = RayFlat(rays, ray_idx)
    var result = trace_gpu_wide_tlas_sphere_ray[
        tlas_width,
        blas_width,
        TRACE_SHADOW,
    ](
        tlas_wide_bounds,
        tlas_wide_data,
        tlas_wide_counts,
        tlas_leaf_instances,
        inst_inv_transform,
        inst_blas_indices,
        blas_wide_bounds,
        blas_wide_data,
        blas_wide_counts,
        blas_leaf_spheres,
        blas_leaf_prims,
        tlas_root_idx,
        blas_root_idx,
        ray,
    )

    flags[ray_idx] = result[0].occluded
