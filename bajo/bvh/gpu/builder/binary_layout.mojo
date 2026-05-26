from std.math import max, ceildiv
from std.gpu import DeviceBuffer, DeviceContext, global_idx

from bajo.core.aabb import AABB
from bajo.bvh.constants import (
    LBVH_LEAF_FLAG,
    LBVH_INDEX_MASK,
    BOUNDS_STRIDE,
)
from bajo.bvh.host_utils import copy_list_to_device
from bajo.bvh.gpu.utils import GpuBuildTimings, GpuBVHValidation
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


def _load_leaf_bounds_host(
    leaf_bounds: List[Float32],
    leaf_idx: UInt32,
) -> AABB:
    var b = Int(leaf_idx) * BOUNDS_STRIDE
    return AABB.load6(leaf_bounds.unsafe_ptr(), b)


def _load_internal_bounds_host(
    node_bounds: List[Float32],
    node_idx: UInt32,
) -> AABB:
    var b = Int(node_idx) * BINARY_BVH_NODE_BOUNDS_STRIDE
    b1 = AABB.load6(node_bounds.unsafe_ptr(), b)
    b2 = AABB.load6(node_bounds.unsafe_ptr(), b + 6)
    return AABB.merge(b1, b2)


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


struct GpuBinaryBoundsBvh:
    var leaf_count: Int
    var internal_count: Int

    var blocks_leaves: Int
    var blocks_internal: Int
    var blocks_init: Int

    var bounds: AABB
    var centroid_bounds: AABB

    var leaf_bounds_host: List[Float32]
    var leaf_payloads_host: List[UInt32]

    var leaf_bounds: DeviceBuffer[DType.float32]
    var leaf_payloads: DeviceBuffer[DType.uint32]

    var keys: DeviceBuffer[DType.uint32]
    """Morton keys."""
    var sorted_leaf_ids: DeviceBuffer[DType.uint32]

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
        leaf_bounds: List[Float32],
        leaf_payloads: List[UInt32],
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

        self.leaf_bounds_host = leaf_bounds.copy()
        self.leaf_payloads_host = leaf_payloads.copy()

        var out = AABB.invalid()
        for i in range(self.leaf_count):
            var b = i * BOUNDS_STRIDE
            var aabb = AABB.load6(self.leaf_bounds_host.unsafe_ptr(), b)
            out.grow(aabb)
        self.bounds = out
        var out2 = AABB.invalid()
        for i in range(self.leaf_count):
            var b = i * BOUNDS_STRIDE
            var aabb = AABB.load6(self.leaf_bounds_host.unsafe_ptr(), b)
            out2.grow(aabb.centroid())
        self.centroid_bounds = out2

        self.leaf_bounds = copy_list_to_device(ctx, self.leaf_bounds_host)
        self.leaf_payloads = copy_list_to_device(ctx, self.leaf_payloads_host)

        self.keys = ctx.enqueue_create_buffer[DType.uint32](n_leaf)
        self.sorted_leaf_ids = ctx.enqueue_create_buffer[DType.uint32](n_leaf)

        self.node_meta = ctx.enqueue_create_buffer[DType.uint32](
            n_internal * BINARY_BVH_NODE_META_STRIDE
        )
        self.leaf_parent = ctx.enqueue_create_buffer[DType.uint32](n_leaf)
        self.node_bounds = ctx.enqueue_create_buffer[DType.float32](
            n_internal * BINARY_BVH_NODE_BOUNDS_STRIDE
        )
        self.node_flags = ctx.enqueue_create_buffer[DType.uint32](n_internal)

    def root_bounds(self) -> AABB:
        return self.bounds

    def validate(self) raises -> GpuBVHValidation:
        var sorted_validation = validate_sorted_keys(
            self.keys,
            self.sorted_leaf_ids,
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
            self.bounds,
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
