from bajo.core import AABB, AxisAlignedBoundingBox, Vec3f32, Point3f32
from bajo.bvh.constants import EMPTY_LANE
from bajo.bvh.cpu.builder import BoundsBvhBuilder, BoundsItem


@fieldwise_init
struct WideBvhNode[width: SIMDSize](Copyable):
    """Lane node used by BoundsBvh.

    Lane encoding:
        counts[i] == EMPTY_LANE -> unused lane
        counts[i] == 0          -> child node, data[i] = child node index
        counts[i] > 0           -> leaf range, data[i] = first item
    """

    var aabb: AxisAlignedBoundingBox[DType.float32, Self.width]
    var data: SIMD[DType.uint32, Self.width]
    var counts: SIMD[DType.uint32, Self.width]

    def __init__(out self):
        self.aabb = AxisAlignedBoundingBox[DType.float32, Self.width].invalid()
        self.data = SIMD[DType.uint32, Self.width](0)
        self.counts = SIMD[DType.uint32, Self.width](EMPTY_LANE)


struct BoundsBvh[width: SIMDSize](Copyable):
    """Generic wide/lane BVH layout with range leaves."""

    var nodes: List[WideBvhNode[Self.width]]
    var item_indices: List[UInt32]
    var item_payloads: List[UInt32]

    def __init__(out self, bvh: BoundsBvhBuilder):
        self.nodes = List[WideBvhNode[Self.width]]()
        self.item_indices = bvh.item_indices.copy()
        self.item_payloads = List[UInt32](capacity=len(bvh.items))

        for item in bvh.items:
            self.item_payloads.append(item.payload)

        if bvh.nodes_used > 0:
            _ = self._collapse(bvh, 0)

    def _collapse(mut self, bvh: BoundsBvhBuilder, bin_idx: UInt32) -> UInt32:
        var wide_idx = len(self.nodes)
        self.nodes.append(WideBvhNode[Self.width]())

        var pool = InlineArray[UInt32, Self.width](fill=bin_idx)
        var p_size = 1

        # Pull up the largest internal nodes until we fill the wide node or run
        # out of internal nodes.
        while p_size < Self.width:
            var best_a: Float32 = -1.0
            var best_i: Int = -1

            for i in range(p_size):
                ref candidate = bvh.nodes[Int(pool[i])]

                if not candidate.is_leaf():
                    var a = candidate.surface_area()

                    if a > best_a:
                        best_a = a
                        best_i = i

            if best_i == -1:
                break

            ref n = bvh.nodes[Int(pool[best_i])]
            pool[best_i] = n.left_child()
            pool[p_size] = n.right_child()
            p_size += 1

        var node = WideBvhNode[Self.width]()

        comptime for i in range(Self.width):
            if i < p_size:
                ref n = bvh.nodes[Int(pool[i])]

                node.aabb._min.x[i] = n.aabb._min.x
                node.aabb._min.y[i] = n.aabb._min.y
                node.aabb._min.z[i] = n.aabb._min.z

                node.aabb._max.x[i] = n.aabb._max.x
                node.aabb._max.y[i] = n.aabb._max.y
                node.aabb._max.z[i] = n.aabb._max.z

                if n.is_leaf():
                    node.data[i] = n.first_item()
                    node.counts[i] = n.item_count
                else:
                    node.data[i] = self._collapse(bvh, pool[i])
                    node.counts[i] = 0
            else:
                node.counts[i] = EMPTY_LANE

        self.nodes[wide_idx] = node^
        return UInt32(wide_idx)

    def root_bounds(self) -> AABB:
        var out = AABB.invalid()
        if len(self.nodes) > 0:
            ref root = self.nodes[0]
            comptime for lane in range(Self.width):
                if root.counts[lane] != EMPTY_LANE:
                    out.grow(
                        Point3f32(
                            root.aabb._min.x[lane],
                            root.aabb._min.y[lane],
                            root.aabb._min.z[lane],
                        ),
                        Point3f32(
                            root.aabb._max.x[lane],
                            root.aabb._max.y[lane],
                            root.aabb._max.z[lane],
                        ),
                    )
        return out
