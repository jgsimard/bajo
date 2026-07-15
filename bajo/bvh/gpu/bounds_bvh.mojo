from std.math import max, ceildiv
from std.gpu import DeviceBuffer, DeviceContext
from std.time import perf_counter_ns

from bajo.core import AABB, AxisAlignedBoundingBox, Vec3, Frame, Rayf32
from bajo.core.intersect import intersect_ray_aabb_rcp, RayDistanceHit
from bajo.bvh.constants import EMPTY_LANE, WideNode
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

    var wide_nodes: DeviceBuffer[DType.float32]
    var leaf_block_indices: DeviceBuffer[DType.uint32]

    def __init__(
        out self,
        mut ctx: DeviceContext,
        leaf_count: Int,
    ) raises:
        self.leaf_count = leaf_count
        self.internal_count = max(self.leaf_count - 1, 0)
        self.root_idx = 0
        self.node_count = 0
        self.leaf_block_count = 0
        self.max_wide_nodes = max(self.internal_count, 1)
        self.max_leaf_blocks = max(self.leaf_count, 1)

        self.bounds_device = ctx.enqueue_create_buffer[DType.float32](12)

        self.wide_nodes = ctx.enqueue_create_buffer[DType.float32](
            self.max_wide_nodes * Self.width * WideNode.CHILD_STRIDE
        )
        self.leaf_block_indices = ctx.enqueue_create_buffer[DType.uint32](
            self.max_leaf_blocks * Self.width
        )

    def build(
        mut self,
        mut ctx: DeviceContext,
        leaf_bounds: DeviceBuffer[DType.float32],
        leaf_payloads: DeviceBuffer[DType.uint32],
    ) raises -> GpuBuildTimings:
        debug_assert["safe"](self.leaf_count > 0, "passed empty input.")
        debug_assert["safe"](len(leaf_payloads) == self.leaf_count)

        var binary = GpuBinaryBoundsBvh(ctx, leaf_bounds, leaf_payloads)
        self.bounds_device = binary.bounds_device.copy()

        var workspace = RadixSortWorkspace[DType.uint32, DType.uint32](
            ctx, max(self.leaf_count, 1)
        )

        # leaf AABBs -> sorted binary LBVH
        timings = build_binary_bvh_with_lbvh(ctx, binary, workspace)

        # binary BVH -> wide BVH
        var collapse_start = perf_counter_ns()
        collapse(ctx, binary, self)

        var end_ns = perf_counter_ns()

        timings.collapse_ns = Int(end_ns - collapse_start)

        return timings

    def build_test(
        mut self,
        mut ctx: DeviceContext,
        leaf_bounds: DeviceBuffer[DType.float32],
        leaf_payloads: DeviceBuffer[DType.uint32],
    ) raises -> GpuBinaryBoundsBvh:
        debug_assert["safe"](self.leaf_count > 0, "passed empty input.")
        debug_assert["safe"](len(leaf_payloads) == self.leaf_count)
        var binary = GpuBinaryBoundsBvh(ctx, leaf_bounds, leaf_payloads)
        self.bounds_device = binary.bounds_device.copy()
        var workspace = RadixSortWorkspace[DType.uint32, DType.uint32](
            ctx, max(self.leaf_count, 1)
        )

        # leaf AABBs -> sorted binary LBVH
        _ = build_binary_bvh_with_lbvh(ctx, binary, workspace)

        # binary BVH -> wide BVH
        collapse(ctx, binary, self)

        return binary^

    def root_bounds(self) raises -> AABB[Frame.WORLD]:
        with self.bounds_device.map_to_host() as h:
            return AABB[Frame.WORLD].load6(h.unsafe_ptr(), 0)

    def centroid_bounds(self) raises -> AABB[Frame.WORLD]:
        with self.bounds_device.map_to_host() as h:
            return AABB[Frame.WORLD].load6(
                h.unsafe_ptr(), AABB[Frame.WORLD].STRIDE
            )


def _wide_lane_base[width: SIMDSize](node_idx: UInt32, lane: Int) -> Int:
    return Int(node_idx) * width + lane


def _wide_node_base[width: SIMDSize](node_idx: UInt32, lane: Int) -> Int:
    return _wide_lane_base[width](node_idx, lane) * WideNode.CHILD_STRIDE


def _wide_node_store_child[
    origin: MutOrigin,
    //,
    width: SIMDSize,
](
    wide_nodes: UnsafePointer[Float32, origin],
    node_idx: UInt32,
    lane: Int,
    bounds: AABB,
    meta: UInt32,
):
    var b = _wide_node_base[width](node_idx, lane)

    wide_nodes[b + WideNode.MIN_X] = bounds._min.x
    wide_nodes[b + WideNode.MIN_Y] = bounds._min.y
    wide_nodes[b + WideNode.MIN_Z] = bounds._min.z
    wide_nodes[b + WideNode.MAX_X] = bounds._max.x
    wide_nodes[b + WideNode.MAX_Y] = bounds._max.y
    wide_nodes[b + WideNode.MAX_Z] = bounds._max.z

    var wide_nodes_u32 = wide_nodes.bitcast[UInt32]()
    wide_nodes_u32[b + WideNode.META] = meta
    wide_nodes[b + WideNode.PAD] = 0.0


def _wide_node_load_meta[
    origin: ImmutOrigin,
    //,
    width: SIMDSize,
](
    wide_nodes: UnsafePointer[Float32, origin],
    node_idx: UInt32,
    lane: Int,
) -> UInt32:
    var b = _wide_node_base[width](node_idx, lane)
    return wide_nodes.bitcast[UInt32]()[b + WideNode.META]


def _load_wide_node_bounds_block[
    origin: ImmutOrigin,
    //,
    dtype: DType,
    frame: Frame,
    width: SIMDSize,
](
    wide_nodes: UnsafePointer[Scalar[dtype], origin],
    node_idx: UInt32,
) -> AxisAlignedBoundingBox[dtype, frame, width]:
    var aabb = AxisAlignedBoundingBox[dtype, frame, width].invalid()

    comptime for lane in range(width):
        var b = _wide_node_base[width](node_idx, lane)

        aabb._min.x[lane] = wide_nodes[b + WideNode.MIN_X]
        aabb._min.y[lane] = wide_nodes[b + WideNode.MIN_Y]
        aabb._min.z[lane] = wide_nodes[b + WideNode.MIN_Z]

        aabb._max.x[lane] = wide_nodes[b + WideNode.MAX_X]
        aabb._max.y[lane] = wide_nodes[b + WideNode.MAX_Y]
        aabb._max.z[lane] = wide_nodes[b + WideNode.MAX_Z]

    return aabb


def _intersect_wide_node_bounds[
    origin: ImmutOrigin,
    //,
    frame: Frame,
    width: SIMDSize,
](
    wide_nodes: UnsafePointer[Float32, origin],
    node_idx: UInt32,
    ray: Rayf32[frame],
    t_max: Float32,
) -> RayDistanceHit[DType.float32, width]:
    var block = _load_wide_node_bounds_block[DType.float32, frame, width](
        wide_nodes,
        node_idx,
    )

    var O = ray.origin[width]()
    var rcp_d = ray.rcp_direction[width]()

    return intersect_ray_aabb_rcp(
        O,
        rcp_d,
        block._min,
        block._max,
        t_max,
    )
