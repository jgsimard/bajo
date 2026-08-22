from std.math import max
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh.constants import WideNode
from bajo.core import (
    AABB,
    AxisAlignedBoundingBox,
    Frame,
    Point3,
    Rayf32,
    SegmentOffsets,
    Vec3,
)
from bajo.bvh.gpu.utils import upload_list
from bajo.core.intersect import (
    RayDistanceHit,
    intersect_ray_aabb_octant_fma,
    intersect_ray_aabb_rcp,
)


struct GpuWideBoundsBvh[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
    max_leaf_size: Int = Int(leaf_width),
]:
    """Final GPU BVH data consumed by traversal.

    This type owns no topology builder or temporary construction workspace.
    Node width, leaf storage width, and logical leaf size remain independent.
    """

    var leaf_count: Int
    var root_idx: UInt32
    var node_count: Int
    var leaf_block_count: Int

    var bounds_device: DeviceBuffer[DType.float32]
    """[0..5] = root bounds; [6..11] = centroid bounds."""

    var wide_nodes: DeviceBuffer[DType.float32]
    var leaf_block_indices: DeviceBuffer[DType.uint32]

    def __init__(
        out self,
        leaf_count: Int,
        node_count: Int,
        leaf_block_count: Int,
        var bounds_device: DeviceBuffer[DType.float32],
        var wide_nodes: DeviceBuffer[DType.float32],
        var leaf_block_indices: DeviceBuffer[DType.uint32],
    ):
        """Adopt the only segment of a completed segmented build."""
        self.leaf_count = leaf_count
        self.root_idx = UInt32(0)
        self.node_count = node_count
        self.leaf_block_count = leaf_block_count
        self.bounds_device = bounds_device^
        self.wide_nodes = wide_nodes^
        self.leaf_block_indices = leaf_block_indices^

    def root_bounds(self) raises -> AABB[Frame.WORLD]:
        with self.bounds_device.map_to_host() as host:
            return AABB[Frame.WORLD].load6(
                Span(unsafe_ptr=host.unsafe_ptr(), length=len(host)), 0
            )

    def centroid_bounds(self) raises -> AABB[Frame.WORLD]:
        with self.bounds_device.map_to_host() as host:
            return AABB[Frame.WORLD].load6(
                Span(unsafe_ptr=host.unsafe_ptr(), length=len(host)),
                AABB[Frame.WORLD].STRIDE,
            )


struct GpuWideBoundsBvhBatch[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
    max_leaf_size: Int = Int(leaf_width),
]:
    """Packed destination for one segmented BVH2-to-wide conversion.

    Node and leaf-block ranges are conservative capacities assigned by prefix
    sum before collapse. Child metadata remains local to its segment, while the
    backing buffers are one packed allocation suitable for ``GpuBlasSet``.
    """

    var segments: SegmentOffsets
    var node_segments: SegmentOffsets
    var leaf_block_segments: SegmentOffsets
    var node_segment_offsets: DeviceBuffer[DType.uint32]
    var leaf_block_segment_offsets: DeviceBuffer[DType.uint32]
    var bounds_device: DeviceBuffer[DType.float32]
    var wide_nodes: DeviceBuffer[DType.float32]
    var leaf_block_indices: DeviceBuffer[DType.uint32]
    var node_counts: DeviceBuffer[DType.uint32]
    var leaf_block_counts: DeviceBuffer[DType.uint32]

    def __init__(
        out self,
        mut ctx: DeviceContext,
        segments: SegmentOffsets,
    ) raises:
        debug_assert["safe", _use_compiler_assume=True](
            segments.segment_count() > 0,
            "wide batch requires at least one segment",
        )
        var node_capacities = List[Int](capacity=segments.segment_count())
        var leaf_capacities = List[Int](capacity=segments.segment_count())
        for segment_idx in range(segments.segment_count()):
            var leaf_count = Int(segments.count(segment_idx))
            node_capacities.append(
                max(leaf_count - 1, 1) if leaf_count > 0 else 0
            )
            leaf_capacities.append(leaf_count)

        self.segments = segments.copy()
        self.node_segments = SegmentOffsets.from_counts(node_capacities^)
        self.leaf_block_segments = SegmentOffsets.from_counts(leaf_capacities^)
        self.node_segment_offsets = upload_list(ctx, self.node_segments.offsets)
        self.leaf_block_segment_offsets = upload_list(
            ctx, self.leaf_block_segments.offsets
        )
        self.bounds_device = ctx.enqueue_create_buffer[DType.float32](
            self.segments.segment_count() * 2 * AABB.STRIDE
        )
        self.wide_nodes = ctx.enqueue_create_buffer[DType.float32](
            self.node_segments.item_count()
            * Self.node_width
            * WideNode.CHILD_STRIDE
        )
        self.leaf_block_indices = ctx.enqueue_create_buffer[DType.uint32](
            self.leaf_block_segments.item_count() * Self.leaf_width
        )
        self.node_counts = ctx.enqueue_create_buffer[DType.uint32](
            self.segments.segment_count()
        )
        self.leaf_block_counts = ctx.enqueue_create_buffer[DType.uint32](
            self.segments.segment_count()
        )

    def node_f32_base(self, segment_idx: Int) -> UInt32:
        return self.node_segments.begin(segment_idx) * UInt32(
            Self.node_width * WideNode.CHILD_STRIDE
        )

    def leaf_lane_base(self, segment_idx: Int) -> UInt32:
        return self.leaf_block_segments.begin(segment_idx) * UInt32(
            Self.leaf_width
        )

    def into_single_segment(
        deinit self,
    ) raises -> GpuWideBoundsBvh[
        Self.node_width, Self.leaf_width, Self.max_leaf_size
    ]:
        """Consume a completed one-segment batch without copying its output."""
        debug_assert["safe", _use_compiler_assume=True](
            self.segments.segment_count() == 1,
            "standalone BVH result requires exactly one segment",
        )
        var node_count: Int
        var leaf_block_count: Int
        with self.node_counts.map_to_host() as nodes, self.leaf_block_counts.map_to_host() as leaves:
            node_count = Int(nodes[0])
            leaf_block_count = Int(leaves[0])
        return GpuWideBoundsBvh[
            Self.node_width, Self.leaf_width, Self.max_leaf_size
        ](
            self.segments.item_count(),
            node_count,
            leaf_block_count,
            self.bounds_device^,
            self.wide_nodes^,
            self.leaf_block_indices^,
        )

    def single_segment_view(
        self,
    ) raises -> GpuWideBoundsBvh[
        Self.node_width, Self.leaf_width, Self.max_leaf_size
    ]:
        """Create a shared-buffer standalone view for diagnostic consumers."""
        debug_assert["safe", _use_compiler_assume=True](
            self.segments.segment_count() == 1,
            "standalone BVH view requires exactly one segment",
        )
        var node_count: Int
        var leaf_block_count: Int
        with self.node_counts.map_to_host() as nodes, self.leaf_block_counts.map_to_host() as leaves:
            node_count = Int(nodes[0])
            leaf_block_count = Int(leaves[0])
        return GpuWideBoundsBvh[
            Self.node_width, Self.leaf_width, Self.max_leaf_size
        ](
            self.segments.item_count(),
            node_count,
            leaf_block_count,
            self.bounds_device.copy(),
            self.wide_nodes.copy(),
            self.leaf_block_indices.copy(),
        )


def _wide_lane_base[width: SIMDLength](node_idx: UInt32, lane: Int) -> Int:
    return Int(node_idx) * width + lane


def _wide_node_base[width: SIMDLength](node_idx: UInt32, lane: Int) -> Int:
    return _wide_lane_base[width](node_idx, lane) * WideNode.CHILD_STRIDE


def _wide_node_store_child[
    width: SIMDLength,
](
    wide_nodes: MutPointer[Float32, _],
    node_idx: UInt32,
    lane: Int,
    bounds: AABB,
    meta: UInt32,
):
    var base = _wide_node_base[width](node_idx, lane)

    wide_nodes[unsafe_offset=base + WideNode.MIN_X] = bounds._min.x
    wide_nodes[unsafe_offset=base + WideNode.MIN_Y] = bounds._min.y
    wide_nodes[unsafe_offset=base + WideNode.MIN_Z] = bounds._min.z
    wide_nodes[unsafe_offset=base + WideNode.MAX_X] = bounds._max.x
    wide_nodes[unsafe_offset=base + WideNode.MAX_Y] = bounds._max.y
    wide_nodes[unsafe_offset=base + WideNode.MAX_Z] = bounds._max.z
    wide_nodes.unsafe_bitcast[UInt32]()[
        unsafe_offset=base + WideNode.META
    ] = meta


def _wide_node_load_meta[
    width: SIMDLength,
](wide_nodes: ImmPointer[Float32, _], node_idx: UInt32, lane: Int) -> UInt32:
    var base = _wide_node_base[width](node_idx, lane)
    return wide_nodes.unsafe_bitcast[UInt32]()[
        unsafe_offset=base + WideNode.META
    ]


struct WideNodeIntersection[width: SIMDLength](TrivialRegisterPassable):
    var bounds_hit: RayDistanceHit[DType.float32, Self.width]
    var meta: SIMD[DType.uint32, Self.width]

    def __init__(
        out self,
        bounds_hit: RayDistanceHit[DType.float32, Self.width],
        meta: SIMD[DType.uint32, Self.width],
    ):
        self.bounds_hit = bounds_hit
        self.meta = meta


def _intersect_wide_node[
    frame: Frame,
    width: SIMDLength,
](
    wide_nodes: ImmPointer[Float32, _],
    node_idx: UInt32,
    ray: Rayf32[frame],
    t_max: Float32,
) -> WideNodeIntersection[width]:
    var block = AxisAlignedBoundingBox[DType.float32, frame, width].invalid()
    var meta = SIMD[DType.uint32, width](0)

    comptime for lane in range(width):
        var base = _wide_node_base[width](node_idx, lane)

        block._min.x[lane] = wide_nodes[unsafe_offset=base + WideNode.MIN_X]
        block._min.y[lane] = wide_nodes[unsafe_offset=base + WideNode.MIN_Y]
        block._min.z[lane] = wide_nodes[unsafe_offset=base + WideNode.MIN_Z]
        block._max.x[lane] = wide_nodes[unsafe_offset=base + WideNode.MAX_X]
        block._max.y[lane] = wide_nodes[unsafe_offset=base + WideNode.MAX_Y]
        block._max.z[lane] = wide_nodes[unsafe_offset=base + WideNode.MAX_Z]
        meta[lane] = wide_nodes.unsafe_bitcast[UInt32]()[
            unsafe_offset=base + WideNode.META
        ]
    var bounds_hit = intersect_ray_aabb_rcp(
        ray.origin[width](),
        ray.rcp_direction[width](),
        block._min,
        block._max,
        t_max,
    )
    return WideNodeIntersection[width](bounds_hit, meta)


@always_inline
def _intersect_wide_node_precomputed[
    frame: Frame,
    width: SIMDLength,
](
    wide_nodes: ImmPointer[Float32, _],
    node_idx: UInt32,
    bounds_origin: Point3[DType.float32, frame, width],
    rcp_direction: Vec3[DType.float32, frame, width],
    t_max: Float32,
) -> WideNodeIntersection[width]:
    """Intersect a wide node with reciprocal direction cached per query."""
    var block = AxisAlignedBoundingBox[DType.float32, frame, width].invalid()
    var meta = SIMD[DType.uint32, width](0)

    comptime for lane in range(width):
        var base = _wide_node_base[width](node_idx, lane)
        block._min.x[lane] = wide_nodes[unsafe_offset=base + WideNode.MIN_X]
        block._min.y[lane] = wide_nodes[unsafe_offset=base + WideNode.MIN_Y]
        block._min.z[lane] = wide_nodes[unsafe_offset=base + WideNode.MIN_Z]
        block._max.x[lane] = wide_nodes[unsafe_offset=base + WideNode.MAX_X]
        block._max.y[lane] = wide_nodes[unsafe_offset=base + WideNode.MAX_Y]
        block._max.z[lane] = wide_nodes[unsafe_offset=base + WideNode.MAX_Z]
        meta[lane] = wide_nodes.unsafe_bitcast[UInt32]()[
            unsafe_offset=base + WideNode.META
        ]

    var bounds_hit = intersect_ray_aabb_rcp(
        bounds_origin,
        rcp_direction,
        block,
        SIMD[DType.float32, width](t_max),
    )
    return WideNodeIntersection[width](bounds_hit, meta)


@always_inline
def _intersect_wide_node_precomputed_octant[
    frame: Frame,
    width: SIMDLength,
    positive_x: Bool,
    positive_y: Bool,
    positive_z: Bool,
](
    wide_nodes: ImmPointer[Float32, _],
    node_idx: UInt32,
    origin_rcp_direction: Vec3[DType.float32, frame, width],
    rcp_direction: Vec3[DType.float32, frame, width],
    t_max: Float32,
) -> WideNodeIntersection[width]:
    """Intersect a wide node using ray data prepared once per query."""
    var block = AxisAlignedBoundingBox[DType.float32, frame, width].invalid()
    var meta = SIMD[DType.uint32, width](0)

    comptime for lane in range(width):
        var base = _wide_node_base[width](node_idx, lane)

        block._min.x[lane] = wide_nodes[unsafe_offset=base + WideNode.MIN_X]
        block._min.y[lane] = wide_nodes[unsafe_offset=base + WideNode.MIN_Y]
        block._min.z[lane] = wide_nodes[unsafe_offset=base + WideNode.MIN_Z]
        block._max.x[lane] = wide_nodes[unsafe_offset=base + WideNode.MAX_X]
        block._max.y[lane] = wide_nodes[unsafe_offset=base + WideNode.MAX_Y]
        block._max.z[lane] = wide_nodes[unsafe_offset=base + WideNode.MAX_Z]
        meta[lane] = wide_nodes.unsafe_bitcast[UInt32]()[
            unsafe_offset=base + WideNode.META
        ]

    var bounds_hit = intersect_ray_aabb_octant_fma[
        DType.float32,
        frame,
        width,
        positive_x,
        positive_y,
        positive_z,
    ](
        origin_rcp_direction,
        rcp_direction,
        block,
        SIMD[DType.float32, width](t_max),
    )
    return WideNodeIntersection[width](bounds_hit, meta)
