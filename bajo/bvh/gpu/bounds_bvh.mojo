from std.math import max, ceildiv
from std.gpu import DeviceBuffer, DeviceContext
from std.time import perf_counter_ns

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
    build_binary_bvh_with_lbvh,
    BINARY_BVH_NODE_META_STRIDE,
    BINARY_BVH_NODE_BOUNDS_STRIDE,
    GPU_BOUNDS_BVH_BLOCK_SIZE,
)
from bajo.bvh.gpu.builder.wide_collapse import collapse
from bajo.bvh.gpu.builder.binary_layout import GpuBinaryBoundsBvh


struct GpuBoundsBvh[width: Int]:
    """Generic GPU Bvh. Build input is only leaf AABBs plus payload ids.

    Wide lane encoding mirrors the CPU BVH:
        count == EMPTY_LANE -> unused lane
        count == 0          -> child node, data = wide child index
        count > 0           -> leaf block, data = leaf block index
    """

    var leaf_count: Int
    var internal_count: Int
    var root_idx: UInt32
    var node_count: Int
    var leaf_block_count: Int
    var max_wide_nodes: Int
    var max_leaf_blocks: Int

    var leaf_bounds: DeviceBuffer[DType.float32]
    var leaf_payloads: DeviceBuffer[DType.uint32]

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
        leaf_bounds: DeviceBuffer[DType.float32],
        leaf_payloads: DeviceBuffer[DType.uint32],
    ) raises:
        self.leaf_count = len(leaf_payloads)
        self.internal_count = max(self.leaf_count - 1, 0)
        self.root_idx = UInt32(0)
        self.node_count = 0
        self.leaf_block_count = 0
        self.max_wide_nodes = max(self.internal_count, 1)
        self.max_leaf_blocks = max(self.internal_count * Self.width, 1)

        n_internal = max(self.internal_count, 1)

        self.leaf_bounds = leaf_bounds
        self.leaf_payloads = leaf_payloads

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

    def build(mut self, mut ctx: DeviceContext) raises -> GpuBuildTimings:
        if self.leaf_count == 0:
            return GpuBuildTimings.empty()

        var binary = GpuBinaryBoundsBvh(
            ctx, self.leaf_bounds, self.leaf_payloads
        )

        # leaf AABBs -> sorted binary LBVH
        timings = build_binary_bvh_with_lbvh(ctx, binary, self.workspace)

        # binary BVH -> wide BVH
        var collapse_start = perf_counter_ns()

        collapse(ctx, binary, self)

        var end_ns = perf_counter_ns()

        timings.collapse_ns = Int(end_ns - collapse_start)

        return timings

    def build_test(
        mut self, mut ctx: DeviceContext
    ) raises -> GpuBinaryBoundsBvh:
        var binary = GpuBinaryBoundsBvh(
            ctx, self.leaf_bounds_host, self.leaf_payloads_host
        )

        # leaf AABBs -> sorted binary LBVH
        timings = build_binary_bvh_with_lbvh(ctx, binary, self.workspace)

        # binary BVH -> wide BVH
        collapse(ctx, binary, self)

        return binary^

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
