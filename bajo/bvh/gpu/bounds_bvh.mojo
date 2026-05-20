from std.bit import count_leading_zeros
from std.math import min, max, ceildiv, sqrt
from std.time import perf_counter_ns
from std.atomic import Atomic
from std.gpu import DeviceBuffer, DeviceContext, global_idx


from bajo.core.aabb import AABB, AxisAlignedBoundingBox
from bajo.core.vec import Vec3f32, vmin, vmax, Vec3
from bajo.core.mat import Mat44f32, transform_point, transform_vector
from bajo.core.intersect import (
    intersect_ray_aabb,
    intersect_ray_tri,
    RayAabbHit,
)
from bajo.core.morton import morton3
from bajo.bvh.types import Ray, Hit, Sphere
from bajo.bvh.constants import (
    LBVH_LEAF_FLAG,
    LBVH_INDEX_MASK,
    LBVH_SENTINEL,
    GPU_TRAVERSAL_STACK_SIZE,
    TRACE_PRIMARY_FULL,
    TRACE_SHADOW,
    EMPTY_LANE,
)
from bajo.bvh.gpu.validate import (
    validate_sorted_keys,
    validate_topology,
    validate_refit_bounds,
)
from bajo.bvh.gpu.utils import GpuBuildTimings, GpuBVHValidation
from bajo.sort.gpu.radix_sort import device_radix_sort_pairs, RadixSortWorkspace

comptime LBVH_NODE_META_STRIDE = 4
comptime LBVH_NODE_PARENT = 0
comptime LBVH_NODE_LEFT = 1
comptime LBVH_NODE_RIGHT = 2
comptime LBVH_NODE_FENCE = 3

comptime LBVH_NODE_BOUNDS_STRIDE = 12
comptime LBVH_BOUNDS_LEFT = 0
comptime LBVH_BOUNDS_RIGHT = 6

comptime GPU_BOUNDS_BVH_BLOCK_SIZE = 128
comptime GPU_WIDE_BOUNDS_STRIDE = 6
comptime GPU_TRI_LEAF_VERTEX_STRIDE = 9
comptime GPU_SPHERE_STRIDE = 4
comptime _gpu_inf_t = Float32(3.4028234663852886e38)


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
    var stack = InlineArray[UInt32, GPU_TRAVERSAL_STACK_SIZE](
        uninitialized=True
    )
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
            ] = EMPTY_LANE


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
        wide_counts[i] = EMPTY_LANE

        var b = i * GPU_WIDE_BOUNDS_STRIDE
        wide_bounds[b + 0] = 0.0
        wide_bounds[b + 1] = 0.0
        wide_bounds[b + 2] = 0.0
        wide_bounds[b + 3] = 0.0
        wide_bounds[b + 4] = 0.0
        wide_bounds[b + 5] = 0.0

    if i < max_leaf_slots:
        leaf_block_indices[i] = EMPTY_LANE

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
            wide_counts[lane_base] = EMPTY_LANE
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


struct GpuBoundsBvh[width: Int]:
    """Generic GPU Bvh. Build input is only leaf AABBs plus payload ids.

    Wide lane encoding mirrors the CPU BVH:
        count == EMPTY_LANE -> unused lane
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

        n_leaf = max(self.leaf_count, 1)
        n_internal = max(self.internal_count, 1)
        self.blocks_leaves = ceildiv(n_leaf, GPU_BOUNDS_BVH_BLOCK_SIZE)
        self.blocks_internal = ceildiv(n_internal, GPU_BOUNDS_BVH_BLOCK_SIZE)
        self.blocks_init = ceildiv(
            max(n_leaf, n_internal),
            GPU_BOUNDS_BVH_BLOCK_SIZE,
        )

        self.leaf_bounds_host = leaf_bounds.copy()
        self.leaf_payloads_host = leaf_payloads.copy()

        self.leaf_bounds = _copy_f32_to_device(ctx, self.leaf_bounds_host)
        self.leaf_payloads = _copy_u32_to_device(ctx, self.leaf_payloads_host)
        self.keys = ctx.enqueue_create_buffer[DType.uint32](n_leaf)
        self.values = ctx.enqueue_create_buffer[DType.uint32](n_leaf)
        self.node_meta = ctx.enqueue_create_buffer[DType.uint32](
            n_internal * LBVH_NODE_META_STRIDE
        )
        self.leaf_parent = ctx.enqueue_create_buffer[DType.uint32](n_leaf)
        self.node_bounds = ctx.enqueue_create_buffer[DType.float32](
            n_internal * LBVH_NODE_BOUNDS_STRIDE
        )
        self.node_flags = ctx.enqueue_create_buffer[DType.uint32](n_internal)

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
            n_internal
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
        var extent = centroid_bounds.extent()
        var inv = extent.safe_inv()

        ctx.enqueue_function[compute_bounds_morton_codes_kernel](
            self.leaf_bounds,
            self.keys,
            self.values,
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
                self.node_meta,
                self.leaf_parent,
                self.internal_count,
                self.leaf_count,
                grid_dim=self.blocks_init,
                block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
            )
            ctx.enqueue_function[build_lbvh_topology_kernel](
                self.keys,
                self.node_meta,
                self.leaf_parent,
                self.leaf_count,
                grid_dim=self.blocks_internal,
                block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
            )
            ctx.synchronize()
        var t = perf_counter_ns()

        if self.internal_count > 0:
            ctx.enqueue_function[init_lbvh_bounds_kernel](
                self.node_bounds,
                self.node_flags,
                self.internal_count,
                grid_dim=self.blocks_internal,
                block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
            )
            ctx.enqueue_function[refit_lbvh_bounds_from_leaves_kernel](
                self.leaf_bounds,
                self.values,
                self.node_meta,
                self.leaf_parent,
                self.node_bounds,
                self.node_flags,
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
            self.wide_bounds,
            self.wide_data,
            self.wide_counts,
            self.leaf_block_indices,
            self.node_leaf_counts,
            self.leaf_block_counter,
            self.wide_root,
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
                self.leaf_bounds,
                self.leaf_payloads,
                self.wide_bounds,
                self.wide_data,
                self.wide_counts,
                self.leaf_block_indices,
                self.leaf_block_counter,
                self.wide_root,
                grid_dim=1,
                block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
            )
        elif self.leaf_count > 1:
            ctx.enqueue_function[compute_lbvh_subtree_leaf_counts_kernel](
                self.node_meta,
                self.leaf_parent,
                self.node_leaf_counts,
                self.leaf_count,
                grid_dim=self.blocks_leaves,
                block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
            )
            ctx.enqueue_function[find_lbvh_root_kernel](
                self.node_meta,
                self.wide_root,
                self.internal_count,
                grid_dim=self.blocks_internal,
                block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
            )
            ctx.enqueue_function[collapse_lbvh_to_wide_kernel[Self.width]](
                self.leaf_bounds,
                self.leaf_payloads,
                self.values,
                self.node_meta,
                self.node_bounds,
                self.node_leaf_counts,
                self.wide_bounds,
                self.wide_data,
                self.wide_counts,
                self.leaf_block_indices,
                self.leaf_block_counter,
                self.internal_count,
                grid_dim=self.blocks_internal,
                block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
            )

        ctx.synchronize()

        with self.leaf_block_counter.map_to_host() as h:
            self.leaf_block_count = Int(h[0])

        with self.wide_root.map_to_host() as h:
            self.root_idx = UInt32(h[0])


@always_inline
def _wide_lane_base[width: Int](node_idx: UInt32, lane: Int) -> Int:
    return Int(node_idx) * width + lane


@always_inline
def _wide_bounds_base[width: Int](node_idx: UInt32, lane: Int) -> Int:
    return _wide_lane_base[width](node_idx, lane) * GPU_WIDE_BOUNDS_STRIDE


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


@always_inline
def _load_wide_bounds_block[
    dtype: DType,
    width: Int,
](
    wide_bounds: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    node_idx: UInt32,
) -> AxisAlignedBoundingBox[dtype, width]:
    var bmin = Vec3[dtype, width](0)
    var bmax = Vec3[dtype, width](0)

    comptime for lane in range(width):
        var b = _wide_bounds_base[width](node_idx, lane)

        bmin.x[lane] = wide_bounds[b + 0]
        bmin.y[lane] = wide_bounds[b + 1]
        bmin.z[lane] = wide_bounds[b + 2]

        bmax.x[lane] = wide_bounds[b + 3]
        bmax.y[lane] = wide_bounds[b + 4]
        bmax.z[lane] = wide_bounds[b + 5]

    return AxisAlignedBoundingBox(bmin, bmax)


@always_inline
def _intersect_wide_node_bounds[
    width: Int,
](
    wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    node_idx: UInt32,
    ray: Ray,
    t_max: Float32,
) -> RayAabbHit[DType.float32, width]:
    var block = _load_wide_bounds_block[DType.float32, width](
        wide_bounds,
        node_idx,
    )

    var O = Vec3[DType.float32, width](ray.o.x, ray.o.y, ray.o.z)
    var RD = Vec3[DType.float32, width](ray.rd.x, ray.rd.y, ray.rd.z)

    return intersect_ray_aabb[DType.float32, width](
        O,
        RD,
        block._min,
        block._max,
        t_max,
    )


@always_inline
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
        var base = i * LBVH_NODE_META_STRIDE
        node_meta[base + LBVH_NODE_PARENT] = LBVH_SENTINEL  # parent
        node_meta[base + LBVH_NODE_LEFT] = 0  # left child, encoded
        node_meta[base + LBVH_NODE_RIGHT] = 0  # right child, encoded
        # fence/debug: rightmost leaf in range
        node_meta[base + LBVH_NODE_FENCE] = 0

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

    var base = i * LBVH_NODE_META_STRIDE
    node_meta[base + LBVH_NODE_LEFT] = left_encoded
    node_meta[base + LBVH_NODE_RIGHT] = right_encoded
    node_meta[base + LBVH_NODE_FENCE] = UInt32(last)


def init_lbvh_bounds_kernel(
    node_bounds: UnsafePointer[Float32, MutAnyOrigin],
    node_flags: UnsafePointer[UInt32, MutAnyOrigin],
    internal_count: Int,
):
    var i = global_idx.x
    if i >= internal_count:
        return

    var b = i * LBVH_NODE_BOUNDS_STRIDE
    node_bounds[b + 0] = Float32.MAX
    node_bounds[b + 1] = Float32.MAX
    node_bounds[b + 2] = Float32.MAX
    node_bounds[b + 3] = Float32.MIN
    node_bounds[b + 4] = Float32.MIN
    node_bounds[b + 5] = Float32.MIN
    node_bounds[b + 6] = Float32.MAX
    node_bounds[b + 7] = Float32.MAX
    node_bounds[b + 8] = Float32.MAX
    node_bounds[b + 9] = Float32.MIN
    node_bounds[b + 10] = Float32.MIN
    node_bounds[b + 11] = Float32.MIN
    node_flags[i] = UInt32(0)
