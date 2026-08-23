from std.math import ceildiv, max
from std.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh.constants import WideNode
from bajo.bvh.wide_meta import _wide_node_base, _wide_node_index
from bajo.core import (
    AABB,
    AxisAlignedBoundingBox,
    Frame,
    Point3,
    Rayf32,
    SegmentOffsets,
    Vec3,
)
from bajo.bvh.gpu.builder.binary_layout import _segment_for_item
from bajo.bvh.gpu.utils import _device_span, upload_list
from bajo.core.intersect import (
    RayDistanceHit,
    intersect_ray_aabb_rcp,
)


comptime GPU_BVH_COMPACT_BLOCK_SIZE = 256


def compact_segmented_buffer_kernel[
    dtype: DType,
    item_stride: Int,
](
    source: Pointer[Scalar[dtype], ImmutAnyOrigin],
    source_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    target_offsets: ImmSpan[UInt32, ImmutAnyOrigin],
    target: Pointer[Scalar[dtype], MutAnyOrigin],
    scalar_count: Int32,
):
    """Copy exact final fields in parallel while preserving segment order."""
    var scalar_idx = global_idx.x
    if scalar_idx >= Int(scalar_count):
        return
    var target_item = scalar_idx // item_stride
    var field = scalar_idx % item_stride
    var segment_idx = _segment_for_item(target_offsets, target_item)
    var local_item = target_item - Int(target_offsets.unsafe_get(segment_idx))
    var source_item = Int(source_offsets.unsafe_get(segment_idx)) + local_item
    target[unsafe_offset=scalar_idx] = source[
        unsafe_offset=source_item * item_stride + field
    ]


@fieldwise_init
struct GpuCompactWideLayout:
    """Exact host/device segment offsets for completed wide output."""

    var node_segments: SegmentOffsets
    var leaf_block_segments: SegmentOffsets
    var node_segment_offsets: DeviceBuffer[.uint32]
    var leaf_block_segment_offsets: DeviceBuffer[.uint32]

    def __init__(
        out self,
        mut ctx: DeviceContext,
        node_counts: DeviceBuffer[.uint32],
        leaf_block_counts: DeviceBuffer[.uint32],
        segment_count: Int,
    ) raises:
        var host_node_counts = List[Int](capacity=segment_count)
        var host_leaf_counts = List[Int](capacity=segment_count)
        with node_counts.map_to_host() as nodes, leaf_block_counts.map_to_host() as leaves:
            for segment_idx in range(segment_count):
                host_node_counts.append(Int(nodes[segment_idx]))
                host_leaf_counts.append(Int(leaves[segment_idx]))
        self.node_segments = SegmentOffsets.from_counts(host_node_counts^)
        self.leaf_block_segments = SegmentOffsets.from_counts(host_leaf_counts^)
        self.node_segment_offsets = upload_list(ctx, self.node_segments.offsets)
        self.leaf_block_segment_offsets = upload_list(
            ctx, self.leaf_block_segments.offsets
        )


def enqueue_compact_segmented_buffer[
    dtype: DType,
    item_stride: Int,
](
    mut ctx: DeviceContext,
    source: DeviceBuffer[dtype],
    source_offsets: DeviceBuffer[.uint32],
    target_offsets: DeviceBuffer[.uint32],
    target_item_count: Int,
    segment_count: Int,
) raises -> DeviceBuffer[dtype]:
    """Enqueue an exact segment-preserving copy of a capacity buffer."""
    comptime assert item_stride > 0
    debug_assert["safe", _use_compiler_assume=True](
        target_item_count > 0,
        "nonempty segmented BVH compaction requires final output",
    )
    debug_assert["safe", _use_compiler_assume=True](
        len(source_offsets) >= segment_count + 1
        and len(target_offsets) >= segment_count + 1,
        "segmented BVH compaction offsets are too short",
    )
    var target = ctx.enqueue_create_buffer[dtype](
        target_item_count * item_stride
    )
    var scalar_count = target_item_count * item_stride
    ctx.enqueue_function[compact_segmented_buffer_kernel[dtype, item_stride]](
        source,
        _device_span[mut=False](source_offsets),
        _device_span[mut=False](target_offsets),
        target,
        Int32(scalar_count),
        grid_dim=ceildiv(scalar_count, GPU_BVH_COMPACT_BLOCK_SIZE),
        block_dim=GPU_BVH_COMPACT_BLOCK_SIZE,
    )
    return target^


struct GpuWideBoundsBvh[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
    max_leaf_size: Int = Int(leaf_width),
]:
    """Final field-major GPU BVH data consumed by traversal.

    This type owns no topology builder or temporary construction workspace.
    Node width, leaf storage width, and logical leaf size remain independent.
    Each node stores ``min_x[W] .. max_z[W], meta[W]``, matching the CPU wide
    node field order while retaining GPU-specific leaf storage.
    """

    var leaf_count: Int
    var root_idx: UInt32
    var node_count: Int
    var leaf_block_count: Int

    var bounds_device: DeviceBuffer[.float32]
    """[0..5] = root bounds; [6..11] = centroid bounds."""

    var wide_nodes: DeviceBuffer[.float32]
    var leaf_block_indices: DeviceBuffer[.uint32]

    def __init__(
        out self,
        leaf_count: Int,
        node_count: Int,
        leaf_block_count: Int,
        var bounds_device: DeviceBuffer[.float32],
        var wide_nodes: DeviceBuffer[.float32],
        var leaf_block_indices: DeviceBuffer[.uint32],
    ):
        """Adopt the only segment of a completed segmented build."""
        self.leaf_count = leaf_count
        self.root_idx = UInt32(0)
        self.node_count = node_count
        self.leaf_block_count = leaf_block_count
        self.bounds_device = bounds_device^
        self.wide_nodes = wide_nodes^
        self.leaf_block_indices = leaf_block_indices^

    def root_bounds(self) raises -> AABB[.WORLD]:
        with self.bounds_device.map_to_host() as host:
            return AABB[.WORLD].load6(
                Span(unsafe_ptr=host.unsafe_ptr(), length=len(host)), 0
            )

    def centroid_bounds(self) raises -> AABB[.WORLD]:
        with self.bounds_device.map_to_host() as host:
            return AABB[.WORLD].load6(
                Span(unsafe_ptr=host.unsafe_ptr(), length=len(host)),
                AABB[.WORLD].STRIDE,
            )


struct GpuWideBoundsBvhBatch[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
    max_leaf_size: Int = Int(leaf_width),
]:
    """Packed destination for one segmented BVH2-to-wide conversion.

    Node and leaf-block ranges are conservative scratch capacities assigned by
    prefix sum before collapse. Final owners compact the used per-segment
    prefixes and rewrite their descriptor bases at the completion boundary.
    """

    var segments: SegmentOffsets
    var node_segments: SegmentOffsets
    var leaf_block_segments: SegmentOffsets
    var node_segment_offsets: DeviceBuffer[.uint32]
    var leaf_block_segment_offsets: DeviceBuffer[.uint32]
    var bounds_device: DeviceBuffer[.float32]
    var wide_nodes: DeviceBuffer[.float32]
    var leaf_block_indices: DeviceBuffer[.uint32]
    var node_counts: DeviceBuffer[.uint32]
    var leaf_block_counts: DeviceBuffer[.uint32]

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
        debug_assert["safe", _use_compiler_assume=True](
            UInt64(self.node_segments.item_count())
            * UInt64(Self.node_width * WideNode.CHILD_STRIDE)
            <= UInt64(0xFFFFFFFF),
            "segmented wide-node descriptor offsets exceed UInt32",
        )
        debug_assert["safe", _use_compiler_assume=True](
            UInt64(self.leaf_block_segments.item_count())
            * UInt64(Self.leaf_width)
            <= UInt64(0xFFFFFFFF),
            "segmented leaf descriptor offsets exceed UInt32",
        )
        self.node_segment_offsets = upload_list(ctx, self.node_segments.offsets)
        self.leaf_block_segment_offsets = upload_list(
            ctx, self.leaf_block_segments.offsets
        )
        self.bounds_device = ctx.enqueue_create_buffer[.float32](
            self.segments.segment_count() * 2 * AABB.STRIDE
        )
        self.wide_nodes = ctx.enqueue_create_buffer[.float32](
            self.node_segments.item_count()
            * Self.node_width
            * WideNode.CHILD_STRIDE
        )
        self.leaf_block_indices = ctx.enqueue_create_buffer[.uint32](
            self.leaf_block_segments.item_count() * Self.leaf_width
        )
        self.node_counts = ctx.enqueue_create_buffer[.uint32](
            self.segments.segment_count()
        )
        self.leaf_block_counts = ctx.enqueue_create_buffer[.uint32](
            self.segments.segment_count()
        )

    def into_single_segment(
        deinit self, mut ctx: DeviceContext
    ) raises -> GpuWideBoundsBvh[
        Self.node_width, Self.leaf_width, Self.max_leaf_size
    ]:
        """Consume a completed one-segment batch as exact final storage."""
        debug_assert["safe", _use_compiler_assume=True](
            self.segments.segment_count() == 1,
            "standalone BVH result requires exactly one segment",
        )
        var layout = GpuCompactWideLayout(
            ctx, self.node_counts, self.leaf_block_counts, 1
        )
        var node_count = layout.node_segments.item_count()
        var leaf_block_count = layout.leaf_block_segments.item_count()
        var compact_nodes = enqueue_compact_segmented_buffer[
            DType.float32, Self.node_width * WideNode.CHILD_STRIDE
        ](
            ctx,
            self.wide_nodes,
            self.node_segment_offsets,
            layout.node_segment_offsets,
            node_count,
            1,
        )
        var compact_leaf_indices = enqueue_compact_segmented_buffer[
            DType.uint32, Self.leaf_width
        ](
            ctx,
            self.leaf_block_indices,
            self.leaf_block_segment_offsets,
            layout.leaf_block_segment_offsets,
            leaf_block_count,
            1,
        )
        ctx.synchronize()
        return GpuWideBoundsBvh[
            Self.node_width, Self.leaf_width, Self.max_leaf_size
        ](
            self.segments.item_count(),
            node_count,
            leaf_block_count,
            self.bounds_device^,
            compact_nodes^,
            compact_leaf_indices^,
        )

    def into_exact_bvh2_leaf1(
        deinit self,
    ) -> GpuWideBoundsBvh[Self.node_width, Self.leaf_width, Self.max_leaf_size]:
        """Consume an already exact one-segment BVH2/leaf1 allocation."""
        comptime assert (
            Self.node_width == 2
            and Self.leaf_width == 1
            and Self.max_leaf_size == 1
        )
        debug_assert["safe", _use_compiler_assume=True](
            self.segments.segment_count() == 1,
            "standalone BVH result requires exactly one segment",
        )
        var leaf_count = self.segments.item_count()
        # A full binary hierarchy with one primitive per leaf has exactly
        # n-1 internal nodes; the single-leaf representation owns one root.
        var node_count = max(leaf_count - 1, 1)
        return GpuWideBoundsBvh[
            Self.node_width, Self.leaf_width, Self.max_leaf_size
        ](
            leaf_count,
            node_count,
            leaf_count,
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


def _wide_node_store_child[
    width: SIMDLength,
](
    wide_nodes: MutPointer[Float32, _],
    node_idx: UInt32,
    lane: Int,
    bounds: AABB,
    meta: UInt32,
):
    wide_nodes[
        unsafe_offset=_wide_node_index[width](node_idx, WideNode.MIN_X, lane)
    ] = bounds._min.x
    wide_nodes[
        unsafe_offset=_wide_node_index[width](node_idx, WideNode.MIN_Y, lane)
    ] = bounds._min.y
    wide_nodes[
        unsafe_offset=_wide_node_index[width](node_idx, WideNode.MIN_Z, lane)
    ] = bounds._min.z
    wide_nodes[
        unsafe_offset=_wide_node_index[width](node_idx, WideNode.MAX_X, lane)
    ] = bounds._max.x
    wide_nodes[
        unsafe_offset=_wide_node_index[width](node_idx, WideNode.MAX_Y, lane)
    ] = bounds._max.y
    wide_nodes[
        unsafe_offset=_wide_node_index[width](node_idx, WideNode.MAX_Z, lane)
    ] = bounds._max.z
    wide_nodes.unsafe_bitcast[UInt32]()[
        unsafe_offset=_wide_node_index[width](node_idx, WideNode.META, lane)
    ] = meta


def _wide_node_load_meta[
    width: SIMDLength,
](wide_nodes: ImmPointer[Float32, _], node_idx: UInt32, lane: Int) -> UInt32:
    return wide_nodes.unsafe_bitcast[UInt32]()[
        unsafe_offset=_wide_node_index[width](node_idx, WideNode.META, lane)
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
    var base = _wide_node_base[width](node_idx)
    block._min.x = wide_nodes.unsafe_load[width=width](
        base + WideNode.MIN_X * width
    )
    block._min.y = wide_nodes.unsafe_load[width=width](
        base + WideNode.MIN_Y * width
    )
    block._min.z = wide_nodes.unsafe_load[width=width](
        base + WideNode.MIN_Z * width
    )
    block._max.x = wide_nodes.unsafe_load[width=width](
        base + WideNode.MAX_X * width
    )
    block._max.y = wide_nodes.unsafe_load[width=width](
        base + WideNode.MAX_Y * width
    )
    block._max.z = wide_nodes.unsafe_load[width=width](
        base + WideNode.MAX_Z * width
    )
    var meta = wide_nodes.unsafe_bitcast[UInt32]().unsafe_load[width=width](
        base + WideNode.META * width
    )
    var bounds_hit = intersect_ray_aabb_rcp(
        ray.origin[width](),
        ray.reciprocal_direction[width](),
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
    reciprocal_direction: Vec3[.float32, frame, width],
    t_max: Float32,
) -> WideNodeIntersection[width]:
    """Intersect a wide node with reciprocal direction cached per query."""
    var block = AxisAlignedBoundingBox[DType.float32, frame, width].invalid()
    var base = _wide_node_base[width](node_idx)
    block._min.x = wide_nodes.unsafe_load[width=width](
        base + WideNode.MIN_X * width
    )
    block._min.y = wide_nodes.unsafe_load[width=width](
        base + WideNode.MIN_Y * width
    )
    block._min.z = wide_nodes.unsafe_load[width=width](
        base + WideNode.MIN_Z * width
    )
    block._max.x = wide_nodes.unsafe_load[width=width](
        base + WideNode.MAX_X * width
    )
    block._max.y = wide_nodes.unsafe_load[width=width](
        base + WideNode.MAX_Y * width
    )
    block._max.z = wide_nodes.unsafe_load[width=width](
        base + WideNode.MAX_Z * width
    )
    var meta = wide_nodes.unsafe_bitcast[UInt32]().unsafe_load[width=width](
        base + WideNode.META * width
    )

    var bounds_hit = intersect_ray_aabb_rcp(
        bounds_origin,
        reciprocal_direction,
        block,
        SIMD[.float32, width](t_max),
    )
    return WideNodeIntersection[width](bounds_hit, meta)
