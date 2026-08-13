from std.math import max
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh.constants import WideNode
from bajo.core import AABB, AxisAlignedBoundingBox, Frame, Rayf32
from bajo.core.intersect import RayDistanceHit, intersect_ray_aabb_rcp


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
    var internal_count: Int
    var root_idx: UInt32
    var node_count: Int
    var leaf_block_count: Int
    var max_wide_nodes: Int
    var max_leaf_blocks: Int

    var bounds_device: DeviceBuffer[DType.float32]
    """[0..5] = root bounds; [6..11] = centroid bounds."""

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
            self.max_wide_nodes * Self.node_width * WideNode.CHILD_STRIDE
        )
        self.leaf_block_indices = ctx.enqueue_create_buffer[DType.uint32](
            self.max_leaf_blocks * Self.leaf_width
        )

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


def _wide_lane_base[width: SIMDLength](node_idx: UInt32, lane: Int) -> Int:
    return Int(node_idx) * width + lane


def _wide_node_base[width: SIMDLength](node_idx: UInt32, lane: Int) -> Int:
    return _wide_lane_base[width](node_idx, lane) * WideNode.CHILD_STRIDE


def _wide_node_store_child[
    width: SIMDLength,
](
    wide_nodes: Pointer[mut=True, Float32, _],
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

    var wide_nodes_u32 = wide_nodes.unsafe_bitcast[UInt32]()
    wide_nodes_u32[unsafe_offset=base + WideNode.META] = meta
    wide_nodes[unsafe_offset=base + WideNode.PAD] = 0.0


def _wide_node_load_meta[
    width: SIMDLength,
](
    wide_nodes: Pointer[mut=False, Float32, _],
    node_idx: UInt32,
    lane: Int,
) -> UInt32:
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
    wide_nodes: Pointer[mut=False, Float32, _],
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
