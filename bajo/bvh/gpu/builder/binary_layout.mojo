from std.math import max, ceildiv
from std.gpu import DeviceBuffer, DeviceContext, global_idx

from bajo.core.aabb import AABB
from bajo.bvh.constants import (
    LBVH_LEAF_FLAG,
    LBVH_INDEX_MASK,
    BOUNDS_STRIDE,
)
from bajo.bvh.gpu.utils import GpuBuildTimings, GpuBVHValidation, upload_list
from bajo.bvh.gpu.validate import (
    validate_sorted_keys,
    validate_topology,
    validate_refit_bounds,
)

comptime BINARY_BVH_NODE_META_STRIDE = 4
comptime BINARY_BVH_NODE_PARENT = 0
comptime BINARY_BVH_NODE_LEFT = 1
comptime BINARY_BVH_NODE_RIGHT = 2
comptime BINARY_BVH_NODE_FENCE = 3
comptime BINARY_BVH_NODE_BOUNDS_STRIDE = 12

comptime GPU_BOUNDS_BVH_BLOCK_SIZE = 128
comptime BOUNDS_REDUCE_CHUNK = 256
comptime REDUCED_BOUNDS_STRIDE = BOUNDS_STRIDE * 2


def _node_meta_base(node_idx: UInt32) -> Int:
    return Int(node_idx) * BINARY_BVH_NODE_META_STRIDE


def _node_bounds_base(node_idx: UInt32) -> Int:
    return Int(node_idx) * BINARY_BVH_NODE_BOUNDS_STRIDE


def _node_parent_index(node_idx: UInt32) -> Int:
    return _node_meta_base(node_idx) + BINARY_BVH_NODE_PARENT


def _node_left_index(node_idx: UInt32) -> Int:
    return _node_meta_base(node_idx) + BINARY_BVH_NODE_LEFT


def _node_right_index(node_idx: UInt32) -> Int:
    return _node_meta_base(node_idx) + BINARY_BVH_NODE_RIGHT


def _node_left(
    node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    node_idx: UInt32,
) -> UInt32:
    return UInt32(node_meta[_node_left_index(node_idx)])


def _node_right(
    node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    node_idx: UInt32,
) -> UInt32:
    return UInt32(node_meta[_node_right_index(node_idx)])


def _is_encoded_leaf(encoded: UInt32) -> Bool:
    return (encoded & LBVH_LEAF_FLAG) != 0


def _encoded_index(encoded: UInt32) -> UInt32:
    return encoded & LBVH_INDEX_MASK


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


def _load_and_union_node_bounds(
    node_bounds: UnsafePointer[Float32, MutAnyOrigin],
    parent: UInt32,
) -> AABB:
    var b = _node_bounds_base(parent)
    b1 = AABB.load6(node_bounds, b)
    b2 = AABB.load6(node_bounds, b + 6)
    return AABB.merge(b1, b2)


def init_empty_bounds_kernel(
    bounds: UnsafePointer[Float32, MutAnyOrigin], n: Int
):
    var i = global_idx.x
    if i >= n:
        return

    var b = i * BINARY_BVH_NODE_BOUNDS_STRIDE
    var invalid = AABB.invalid()
    invalid.store6(bounds, b)
    invalid.store6(bounds, b + BOUNDS_STRIDE)


def compute_bounds_partials_kernel(
    leaf_bounds: UnsafePointer[Float32, MutAnyOrigin],
    out_partials: UnsafePointer[Float32, MutAnyOrigin],
    leaf_count: Int,
):
    var chunk = global_idx.x
    var first = chunk * BOUNDS_REDUCE_CHUNK
    if first >= leaf_count:
        return

    var last = min(first + BOUNDS_REDUCE_CHUNK, leaf_count)

    var bounds = AABB.invalid()
    var centroid_bounds = AABB.invalid()

    for leaf_idx in range(first, last):
        var b = leaf_idx * BOUNDS_STRIDE
        var aabb = AABB.load6(leaf_bounds, b)

        bounds.grow(aabb)
        centroid_bounds.grow(aabb.centroid())

    var out = chunk * REDUCED_BOUNDS_STRIDE
    bounds.store6(out_partials, out)
    centroid_bounds.store6(out_partials, out + BOUNDS_STRIDE)


def reduce_bounds_partials_kernel(
    in_partials: UnsafePointer[Float32, MutAnyOrigin],
    out_partials: UnsafePointer[Float32, MutAnyOrigin],
    partial_count: Int,
):
    var chunk = global_idx.x
    var first = chunk * BOUNDS_REDUCE_CHUNK
    if first >= partial_count:
        return

    var last = min(first + BOUNDS_REDUCE_CHUNK, partial_count)

    var bounds = AABB.invalid()
    var centroid_bounds = AABB.invalid()

    for i in range(first, last):
        var b = i * REDUCED_BOUNDS_STRIDE

        var partial_bounds = AABB.load6(in_partials, b)
        var partial_centroid_bounds = AABB.load6(in_partials, b + BOUNDS_STRIDE)

        bounds.grow(partial_bounds)
        centroid_bounds.grow(partial_centroid_bounds)

    var out = chunk * REDUCED_BOUNDS_STRIDE
    bounds.store6(out_partials, out)
    centroid_bounds.store6(out_partials, out + BOUNDS_STRIDE)


def copy_lbvh_bounds_result_kernel(
    in_bounds: UnsafePointer[Float32, MutAnyOrigin],
    out_bounds: UnsafePointer[Float32, MutAnyOrigin],
):
    var i = global_idx.x
    if i >= REDUCED_BOUNDS_STRIDE:
        return

    out_bounds[i] = in_bounds[i]


struct GpuBinaryBoundsBvh(Movable):
    var leaf_count: Int
    var internal_count: Int

    var blocks_leaves: Int
    var blocks_internal: Int
    var blocks_init: Int

    var bounds_device: DeviceBuffer[DType.float32]
    """[0..5]  = root bounds, [6..11] = centroid bounds."""

    var leaf_bounds: DeviceBuffer[DType.float32]
    var leaf_payloads: DeviceBuffer[DType.uint32]

    var keys: DeviceBuffer[DType.uint32]
    """Morton keys."""
    var leaf_ids: DeviceBuffer[DType.uint32]

    # Binary node layout:
    #   node_meta   : parent, left encoded child, right encoded child, fence/range end
    #   leaf_parent : parent internal node for each sorted leaf
    #   node_bounds : two child AABBs per internal node, 12 floats total
    #   node_flags  : refit synchronization flags
    var node_meta: DeviceBuffer[DType.uint32]
    var leaf_parent: DeviceBuffer[DType.uint32]
    var node_bounds: DeviceBuffer[DType.float32]
    var node_flags: DeviceBuffer[DType.uint32]

    def __init__(
        out self,
        mut ctx: DeviceContext,
        leaf_bounds: DeviceBuffer[DType.float32],
        leaf_payloads: DeviceBuffer[DType.uint32],
    ) raises:
        self.leaf_count = len(leaf_payloads)
        self.internal_count = max(self.leaf_count - 1, 0)

        var n_leaf = max(self.leaf_count, 1)
        var n_internal = max(self.internal_count, 1)

        self.blocks_leaves = ceildiv(n_leaf, GPU_BOUNDS_BVH_BLOCK_SIZE)
        self.blocks_internal = ceildiv(
            n_internal,
            GPU_BOUNDS_BVH_BLOCK_SIZE,
        )
        self.blocks_init = ceildiv(
            max(n_leaf, n_internal),
            GPU_BOUNDS_BVH_BLOCK_SIZE,
        )

        self.leaf_bounds = leaf_bounds
        self.leaf_payloads = leaf_payloads

        self.bounds_device = ctx.enqueue_create_buffer[DType.float32](
            REDUCED_BOUNDS_STRIDE
        )

        if self.leaf_count == 0:
            ctx.enqueue_function[init_empty_bounds_kernel](
                self.bounds_device,
                1,
                grid_dim=1,
                block_dim=1,
            )
        else:
            var partial_count = ceildiv(
                self.leaf_count,
                BOUNDS_REDUCE_CHUNK,
            )

            var scratch_a = ctx.enqueue_create_buffer[DType.float32](
                max(partial_count, 1) * REDUCED_BOUNDS_STRIDE
            )
            var scratch_b = ctx.enqueue_create_buffer[DType.float32](
                max(
                    ceildiv(partial_count, BOUNDS_REDUCE_CHUNK),
                    1,
                )
                * REDUCED_BOUNDS_STRIDE
            )

            var reduce_grid = ceildiv(
                partial_count,
                GPU_BOUNDS_BVH_BLOCK_SIZE,
            )

            ctx.enqueue_function[compute_bounds_partials_kernel](
                self.leaf_bounds,
                scratch_a,
                self.leaf_count,
                grid_dim=reduce_grid,
                block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
            )

            var in_buf = scratch_a
            var out_buf = scratch_b
            var count = partial_count

            while count > 1:
                var next_count = ceildiv(
                    count,
                    BOUNDS_REDUCE_CHUNK,
                )
                var grid = ceildiv(
                    next_count,
                    GPU_BOUNDS_BVH_BLOCK_SIZE,
                )

                ctx.enqueue_function[reduce_bounds_partials_kernel](
                    in_buf,
                    out_buf,
                    count,
                    grid_dim=grid,
                    block_dim=GPU_BOUNDS_BVH_BLOCK_SIZE,
                )

                swap(in_buf, out_buf)
                count = next_count

            ctx.enqueue_function[copy_lbvh_bounds_result_kernel](
                in_buf,
                self.bounds_device,
                grid_dim=1,
                block_dim=REDUCED_BOUNDS_STRIDE,
            )

            # because scratch_a/scratch_b are temporary buffers
            ctx.synchronize()

        self.keys = ctx.enqueue_create_buffer[DType.uint32](n_leaf)
        self.leaf_ids = ctx.enqueue_create_buffer[DType.uint32](n_leaf)

        self.node_meta = ctx.enqueue_create_buffer[DType.uint32](
            n_internal * BINARY_BVH_NODE_META_STRIDE
        )
        self.leaf_parent = ctx.enqueue_create_buffer[DType.uint32](n_leaf)
        self.node_bounds = ctx.enqueue_create_buffer[DType.float32](
            n_internal * BINARY_BVH_NODE_BOUNDS_STRIDE
        )
        self.node_flags = ctx.enqueue_create_buffer[DType.uint32](n_internal)

    def root_bounds(self) raises -> AABB:
        var out = AABB.invalid()
        with self.bounds_device.map_to_host() as h:
            out = AABB.load6(h.unsafe_ptr(), 0)
        return out

    def centroid_bounds(self) raises -> AABB:
        var out = AABB.invalid()
        with self.bounds_device.map_to_host() as h:
            out = AABB.load6(h.unsafe_ptr(), BOUNDS_STRIDE)
        return out

    def validate(self, bounds: AABB) raises -> GpuBVHValidation:
        var sorted_validation = validate_sorted_keys(
            self.keys,
            self.leaf_ids,
            self.leaf_count,
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
            self.node_meta,
            self.leaf_parent,
            self.leaf_count,
        )
        var refit_validation = validate_refit_bounds(
            self.node_bounds,
            self.node_flags,
            self.node_meta,
            self.leaf_count,
            bounds,
        )
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
