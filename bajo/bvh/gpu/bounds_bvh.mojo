from std.math import max, ceildiv
from std.gpu import DeviceBuffer, DeviceContext
from std.time import perf_counter_ns

from bajo.core import AABB, AxisAlignedBoundingBox, Vec3
from bajo.core.intersect import intersect_ray_aabb, RayDistanceHit
from bajo.bvh.types import Ray
from bajo.bvh.constants import EMPTY_LANE
from bajo.bvh.gpu.validate import (
    validate_sorted_keys,
    validate_topology,
    validate_refit_bounds,
)
from bajo.bvh.gpu.utils import GpuBuildTimings, GpuBVHValidation
from bajo.sort.gpu.radix_sort import RadixSortWorkspace
from bajo.bvh.gpu.builder.lbvh import build_binary_bvh_with_lbvh
from bajo.bvh.gpu.builder.wide_collapse import collapse
from bajo.bvh.gpu.builder.binary_layout import GpuBinaryBoundsBvh


struct GpuBoundsBvh[width: SIMDSize](Movable):
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

    var bounds_device: DeviceBuffer[DType.float32]
    """[0..5]  = root bounds, [6..11] = centroid bounds."""

    var leaf_bounds: DeviceBuffer[DType.float32]
    var leaf_payloads: DeviceBuffer[DType.uint32]

    var wide_bounds: DeviceBuffer[DType.float32]
    var wide_data: DeviceBuffer[DType.uint32]
    var wide_counts: DeviceBuffer[DType.uint32]
    var leaf_block_indices: DeviceBuffer[DType.uint32]
    var leaf_block_counter: DeviceBuffer[DType.uint32]
    var wide_root: DeviceBuffer[DType.uint32]

    var wide_node_counter: DeviceBuffer[DType.uint32]
    var frontier_count: DeviceBuffer[DType.uint32]
    var work_encoded_in: DeviceBuffer[DType.uint32]
    var work_encoded_out: DeviceBuffer[DType.uint32]
    var work_wide_in: DeviceBuffer[DType.uint32]
    var work_wide_out: DeviceBuffer[DType.uint32]

    var workspace: RadixSortWorkspace[DType.uint32, DType.uint32]

    def __init__(
        out self,
        mut ctx: DeviceContext,
        leaf_bounds: DeviceBuffer[DType.float32],
        leaf_payloads: DeviceBuffer[DType.uint32],
    ) raises:
        self.leaf_count = len(leaf_payloads)
        self.internal_count = max(self.leaf_count - 1, 0)
        self.root_idx = 0
        self.node_count = 0
        self.leaf_block_count = 0
        self.max_wide_nodes = max(self.internal_count, 1)
        self.max_leaf_blocks = max(self.leaf_count, 1)

        self.bounds_device = ctx.enqueue_create_buffer[DType.float32](12)

        self.leaf_bounds = leaf_bounds
        self.leaf_payloads = leaf_payloads

        # Wide node index is the binary internal node index. Leaf blocks are
        # allocated on the GPU during collapse with an atomic counter.
        self.wide_bounds = ctx.enqueue_create_buffer[DType.float32](
            self.max_wide_nodes * Self.width * AABB.STRIDE
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
        self.leaf_block_counter = ctx.enqueue_create_buffer[DType.uint32](1)
        self.wide_root = ctx.enqueue_create_buffer[DType.uint32](1)

        self.wide_node_counter = ctx.enqueue_create_buffer[DType.uint32](1)
        self.frontier_count = ctx.enqueue_create_buffer[DType.uint32](1)

        self.work_encoded_in = ctx.enqueue_create_buffer[DType.uint32](
            self.max_wide_nodes
        )
        self.work_encoded_out = ctx.enqueue_create_buffer[DType.uint32](
            self.max_wide_nodes
        )
        self.work_wide_in = ctx.enqueue_create_buffer[DType.uint32](
            self.max_wide_nodes
        )
        self.work_wide_out = ctx.enqueue_create_buffer[DType.uint32](
            self.max_wide_nodes
        )

        self.workspace = RadixSortWorkspace[DType.uint32, DType.uint32](
            ctx, max(self.leaf_count, 1)
        )

    def build(mut self, mut ctx: DeviceContext) raises -> GpuBuildTimings:
        if self.leaf_count == 0:
            return GpuBuildTimings.empty()

        var binary = GpuBinaryBoundsBvh(
            ctx, self.leaf_bounds, self.leaf_payloads
        )
        self.bounds_device = binary.bounds_device.copy()

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
            ctx, self.leaf_bounds, self.leaf_payloads
        )
        self.bounds_device = binary.bounds_device.copy()

        # leaf AABBs -> sorted binary LBVH
        timings = build_binary_bvh_with_lbvh(ctx, binary, self.workspace)

        # binary BVH -> wide BVH
        collapse(ctx, binary, self)

        return binary^

    def root_bounds(self) raises -> AABB:
        with self.bounds_device.map_to_host() as h:
            return AABB.load6(h.unsafe_ptr(), 0)

    def centroid_bounds(self) raises -> AABB:
        with self.bounds_device.map_to_host() as h:
            return AABB.load6(h.unsafe_ptr(), AABB.STRIDE)


def _wide_lane_base[width: SIMDSize](node_idx: UInt32, lane: Int) -> Int:
    return Int(node_idx) * width + lane


def _wide_bounds_base[width: SIMDSize](node_idx: UInt32, lane: Int) -> Int:
    return _wide_lane_base[width](node_idx, lane) * AABB.STRIDE


def _load_wide_bounds_block[
    origin: ImmutOrigin,
    //,
    dtype: DType,
    width: SIMDSize,
](
    wide_bounds: UnsafePointer[Scalar[dtype], origin],
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
    origin: ImmutOrigin,
    //,
    width: SIMDSize,
](
    wide_bounds: UnsafePointer[Float32, origin],
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
