from std.math import max, ceildiv
from std.gpu import DeviceBuffer, DeviceContext

from bajo.core.aabb import AABB, AxisAlignedBoundingBox
from bajo.core.vec import Vec3
from bajo.core.intersect import intersect_ray_aabb, RayDistanceHit
from bajo.bvh.types import Ray
from bajo.bvh.constants import EMPTY_LANE, BOUNDS_STRIDE
from bajo.bvh.gpu.validate import (
    validate_sorted_keys,
    validate_topology,
    validate_refit_bounds,
)
from bajo.bvh.gpu.utils import GpuBuildTimings, GpuBVHValidation
from bajo.sort.gpu.radix_sort import RadixSortWorkspace
from bajo.bvh.host_utils import copy_list_to_device
from bajo.bvh.gpu.builder.lbvh import (
    GpuBoundsLbvhBuilder,
    LBVH_NODE_META_STRIDE,
    LBVH_NODE_BOUNDS_STRIDE,
    GPU_BOUNDS_BVH_BLOCK_SIZE,
)


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

        self.leaf_bounds = copy_list_to_device(ctx, self.leaf_bounds_host)
        self.leaf_payloads = copy_list_to_device(ctx, self.leaf_payloads_host)
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
            self.max_wide_nodes * Self.width * BOUNDS_STRIDE
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
        var builder = GpuBoundsLbvhBuilder[Self.width]()
        var timings = builder.build(
            ctx,
            self.leaf_count,
            self.internal_count,
            self.max_wide_nodes,
            self.max_leaf_blocks,
            self.blocks_leaves,
            self.blocks_internal,
            self.blocks_init,
            self.leaf_bounds_host,
            self.leaf_bounds,
            self.leaf_payloads,
            self.keys,
            self.values,
            self.node_meta,
            self.leaf_parent,
            self.node_bounds,
            self.node_flags,
            self.wide_bounds,
            self.wide_data,
            self.wide_counts,
            self.leaf_block_indices,
            self.node_leaf_counts,
            self.leaf_block_counter,
            self.wide_root,
            self.workspace,
        )

        self.root_idx = builder.root_idx
        self.node_count = builder.node_count
        self.leaf_block_count = builder.leaf_block_count
        self.collapse_ns = builder.collapse_ns
        return timings

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
            var b = i * BOUNDS_STRIDE
            aabb = AABB.load6(self.leaf_bounds_host.unsafe_ptr(), b)
            out.grow(aabb)
        return out


def _wide_lane_base[width: Int](node_idx: UInt32, lane: Int) -> Int:
    return Int(node_idx) * width + lane


def _wide_bounds_base[width: Int](node_idx: UInt32, lane: Int) -> Int:
    return _wide_lane_base[width](node_idx, lane) * BOUNDS_STRIDE


def _load_wide_bounds_block[
    dtype: DType,
    width: Int,
](
    wide_bounds: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    node_idx: UInt32,
) -> AxisAlignedBoundingBox[dtype, width]:
    var aabb = AxisAlignedBoundingBox[dtype, width].invalid()

    comptime for lane in range(width):
        var b = _wide_bounds_base[width](node_idx, lane)

        aabb._min.x[lane] = wide_bounds[b + 0]
        aabb._min.y[lane] = wide_bounds[b + 1]
        aabb._min.z[lane] = wide_bounds[b + 2]

        aabb._max.x[lane] = wide_bounds[b + 3]
        aabb._max.y[lane] = wide_bounds[b + 4]
        aabb._max.z[lane] = wide_bounds[b + 5]

    return aabb


def _intersect_wide_node_bounds[
    width: Int,
](
    wide_bounds: UnsafePointer[Float32, MutAnyOrigin],
    node_idx: UInt32,
    ray: Ray,
    t_max: Float32,
) -> RayDistanceHit[DType.float32, width]:
    var block = _load_wide_bounds_block[DType.float32, width](
        wide_bounds,
        node_idx,
    )

    var O = Vec3[DType.float32, width](ray.o.x, ray.o.y, ray.o.z)
    var RD = 1.0 / Vec3[DType.float32, width](ray.d.x, ray.d.y, ray.d.z)

    return intersect_ray_aabb(
        O,
        RD,
        block._min,
        block._max,
        t_max,
    )
