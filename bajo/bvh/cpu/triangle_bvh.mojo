from bajo.core.utils import min_argmin
from bajo.core import (
    Vec3,
    Normal3f32,
    AABB,
    Point3,
    Point3f32,
    Frame,
    cross,
    normalize,
    Rayf32,
)
from bajo.bvh.constants import EMPTY_LANE, TRACE, f32_max
from bajo.bvh.cpu.bounds_bvh import (
    BoundsBvh,
    BoundsItem,
    BoundsBvhBuilder,
)
from bajo.bvh.types import Hit, TriangleLeafBlock, TypedBvh
from bajo.core.intersect import intersect_ray_tri
from bajo.bvh.cpu.trace import trace_bounds_bvh


struct TriangleBvh[frame: Frame, width: SIMDLength](Copyable, TypedBvh):
    comptime bvh_frame: Frame = Self.frame

    """Triangle-specific wrapper around BoundsBvh[width].

    After construction, leaf primitive data is packed into TriangleLeafBlock.
    In this typed BLAS, a leaf lane means:

        node.counts[lane] > 0
        node.data[lane] = TriangleLeafBlock index
    """

    var tree: BoundsBvh[Self.frame, Self.width]
    var leaf_blocks: List[TriangleLeafBlock[Self.frame, Self.width]]
    var tri_count: Int

    def __init__[
        split_method: String = "median"
    ](out self, var vertices: List[Point3f32[Self.frame]]):
        self.tri_count = len(vertices) / 3
        self.leaf_blocks = List[TriangleLeafBlock[Self.frame, Self.width]]()

        var items = List[BoundsItem[Self.frame]](capacity=self.tri_count)

        for i in range(self.tri_count):
            ref v0 = vertices[i * 3 + 0]
            ref v1 = vertices[i * 3 + 1]
            ref v2 = vertices[i * 3 + 2]

            var bounds = AABB[Self.frame].invalid()
            bounds.grow(v0, v1, v2)

            items.append(BoundsItem(bounds, UInt32(i)))

        var builder = BoundsBvhBuilder[Self.frame, Self.width](items)
        builder.build[split_method]()

        self.tree = BoundsBvh[Self.frame, Self.width](builder)

        self._pack_leaves(vertices^)

    def bounds(self) -> AABB[Self.frame]:
        return self.tree.root_bounds()

    def _pack_leaves(mut self, var vertices: List[Point3f32[Self.frame]]):
        self.leaf_blocks = List[TriangleLeafBlock[Self.frame, Self.width]](
            capacity=(self.tri_count + Int(Self.width) - 1) // Int(Self.width)
        )
        for ref node in self.tree.nodes:
            comptime for lane in range(Self.width):
                if node.counts[lane] != EMPTY_LANE and node.counts[lane] > 0:
                    var first_item = node.data[lane]
                    var item_count = node.counts[lane]

                    var block = TriangleLeafBlock[Self.frame, Self.width]()

                    for k in range(Int(item_count)):
                        var item_ref = Int(
                            self.tree.item_indices[Int(first_item) + k]
                        )
                        var prim_idx = self.tree.item_payloads[item_ref]
                        var base = Int(prim_idx) * 3

                        ref p0 = vertices[base + 0]
                        ref p1 = vertices[base + 1]
                        ref p2 = vertices[base + 2]

                        block.v0.x[k] = p0.x
                        block.v0.y[k] = p0.y
                        block.v0.z[k] = p0.z

                        block.v1.x[k] = p1.x
                        block.v1.y[k] = p1.y
                        block.v1.z[k] = p1.z

                        block.v2.x[k] = p2.x
                        block.v2.y[k] = p2.y
                        block.v2.z[k] = p2.z

                        block.prim_indices[k] = prim_idx

                    var block_idx = UInt32(len(self.leaf_blocks))
                    self.leaf_blocks.append(block^)
                    node.data[lane] = block_idx

    def trace[
        mode: TRACE
    ](self, ray: Rayf32[Self.bvh_frame]) -> Hit[Self.bvh_frame]:
        def leaf_fn(
            ray: Rayf32[Self.bvh_frame],
            O: Point3[DType.float32, Self.bvh_frame, Self.width],
            D: Vec3[DType.float32, Self.bvh_frame, Self.width],
            leaf_block_idx: UInt32,
            mut hit: Hit[Self.bvh_frame],
        ) capturing -> Bool:
            ref block = self.leaf_blocks[Int(leaf_block_idx)]
            var tri_hit = intersect_ray_tri(
                O,
                D,
                block.v0,
                block.v1,
                block.v2,
                hit.t,
                ray.t_min,
            )
            var valid_lane = block.prim_indices.ne(EMPTY_LANE)
            var hit_mask = tri_hit.mask & valid_lane

            if not hit_mask.reduce_or():
                return False

            comptime if mode == TRACE.CLOSEST_HIT:
                _t = hit_mask.select(tri_hit.t, f32_max)
                min_t, lane = min_argmin(_t)

                hit.t = min_t
                hit.u = tri_hit.u[lane]
                hit.v = tri_hit.v[lane]
                hit.prim = block.prim_indices[lane]
                hit.inst = EMPTY_LANE
                var normals = normalize(
                    cross(block.v1 - block.v0, block.v2 - block.v0)
                )
                hit.normal = Normal3f32[Self.bvh_frame](
                    normals.x[lane], normals.y[lane], normals.z[lane]
                )

            return True

        return trace_bounds_bvh[
            Self.frame,
            Self.width,
            mode,
            leaf_fn,
        ](
            self.tree,
            ray,
        )
