from std.utils.numerics import max_finite

from bajo.core.vec import Vec3, Vec3f32, vmin, vmax, longest_axis, dot
from bajo.core.bvh.constants import EMPTY_LANE
from bajo.core.bvh.cpu.bounds_bvh import (
    BoundsBvh,
    BoundsItem,
    BoundsBvhBuilder,
)
from bajo.core.aabb import AABB, AxisAlignedBoundingBox
from bajo.core.bvh.types import Ray, Hit, TriangleLeafBlock
from bajo.core.intersect import intersect_ray_tri
from bajo.core.bvh.cpu.traverse import traverse_wide_ray_bvh


struct TriangleBvh[width: Int](Copyable):
    """Triangle-specific wrapper around BoundsBvh[width].

    After construction, leaf primitive data is packed into TriangleLeafBlock.
    In this typed BLAS, a leaf lane means:

        node.counts[lane] > 0
        node.data[lane] = TriangleLeafBlock index
    """

    var tree: BoundsBvh[Self.width]
    var leaf_blocks: List[TriangleLeafBlock[Self.width]]
    var tri_count: UInt32

    def __init__[
        split_method: String = "median"
    ](
        out self,
        vertices: UnsafePointer[Vec3f32, MutAnyOrigin],
        tri_count: UInt32,
    ):
        self.tri_count = tri_count
        self.leaf_blocks = List[TriangleLeafBlock[Self.width]]()

        var items = List[BoundsItem](capacity=Int(tri_count))

        for i in range(Int(tri_count)):
            ref v0 = vertices[i * 3 + 0]
            ref v1 = vertices[i * 3 + 1]
            ref v2 = vertices[i * 3 + 2]

            var bounds = AABB.invalid()
            bounds.grow(v0)
            bounds.grow(v1)
            bounds.grow(v2)

            items.append(BoundsItem(bounds, UInt32(i)))

        var builder = BoundsBvhBuilder[Self.width](items)
        builder.build[split_method]()

        self.tree = BoundsBvh[Self.width](builder)

        self._pack_leaves(vertices)

    def bounds(self) -> AABB:
        return self.tree.root_bounds()

    def _pack_leaves(
        mut self,
        vertices: UnsafePointer[Vec3f32, MutAnyOrigin],
    ):
        self.leaf_blocks = List[TriangleLeafBlock[Self.width]]()

        for n_idx in range(len(self.tree.nodes)):
            ref node = self.tree.nodes[n_idx]

            comptime for lane in range(Self.width):
                if node.counts[lane] != EMPTY_LANE and node.counts[lane] > 0:
                    var first_item = node.data[lane]
                    var item_count = node.counts[lane]

                    var block = TriangleLeafBlock[Self.width]()

                    comptime for k in range(Self.width):
                        if k < Int(item_count):
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
                            block.valid_lane[k] = True
                        else:
                            block.prim_indices[k] = EMPTY_LANE
                            block.valid_lane[k] = False

                    var block_idx = UInt32(len(self.leaf_blocks))
                    self.leaf_blocks.append(block^)
                    node.data[lane] = block_idx

    @always_inline
    def _intersect_leaf[
        is_occlusion: Bool
    ](
        self,
        ray: Ray,
        leaf_block_idx: UInt32,
        item_count: UInt32,
        mut hit: Hit,
    ) -> Bool:
        ref block = self.leaf_blocks[Int(leaf_block_idx)]

        var O = Vec3[DType.float32, Self.width](ray.o.x, ray.o.y, ray.o.z)
        var D = Vec3[DType.float32, Self.width](ray.d.x, ray.d.y, ray.d.z)

        var h = intersect_ray_tri(
            O,
            D,
            block.v0,
            block.v1,
            block.v2,
            hit.t,
        )

        var t_valid = h.t.ge(ray.t_min)
        var hit_mask = h.mask & t_valid & block.valid_lane

        if hit_mask.reduce_or():
            comptime if is_occlusion:
                return True
            else:
                comptime f32_max = max_finite[DType.float32]()
                var min_t = hit_mask.select(h.t, f32_max).reduce_min()

                if min_t < hit.t:
                    comptime for lane in range(Self.width):
                        if hit_mask[lane] and h.t[lane] == min_t:
                            hit.t = min_t
                            hit.u = h.u[lane]
                            hit.v = h.v[lane]
                            hit.prim = block.prim_indices[lane]
                            hit.inst = EMPTY_LANE
                            hit.occluded = UInt32(0)

                return True

        return False

    @always_inline
    def _traverse_generic[is_occlusion: Bool](self, ray: Ray) -> Hit:
        @always_inline
        def leaf_fn(
            ray: Ray,
            leaf_block_idx: UInt32,
            item_count: UInt32,
            mut hit: Hit,
        ) capturing -> Bool:
            return self._intersect_leaf[is_occlusion](
                ray,
                leaf_block_idx,
                item_count,
                hit,
            )

        return traverse_wide_ray_bvh[
            Self.width,
            is_occlusion,
            leaf_fn,
        ](
            self.tree,
            ray,
        )

    def traverse(self, ray: Ray) -> Hit:
        return self._traverse_generic[False](ray)

    def is_occluded(self, ray: Ray) -> Bool:
        var hit = self._traverse_generic[True](ray)
        return hit.is_occluded()
