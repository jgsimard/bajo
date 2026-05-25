from bajo.core.utils import min_argmin
from bajo.core.vec import Vec3, Vec3f32, vmin, vmax, longest_axis, dot
from bajo.bvh.constants import EMPTY_LANE, TRACE, f32_max
from bajo.bvh.cpu.bounds_bvh import (
    BoundsBvh,
    BoundsItem,
    BoundsBvhBuilder,
)
from bajo.core.aabb import AABB, AxisAlignedBoundingBox
from bajo.bvh.types import Ray, Hit, Sphere, SphereLeafBlock
from bajo.core.intersect import intersect_ray_sphere
from bajo.bvh.cpu.trace import trace_bounds_bvh


struct SphereBvh[width: Int](Copyable):
    """Sphere-specific wrapper around BoundsBvh[width].
    The generic tree is built from BoundsItem ranges.
    After construction, sphere leaf data is packed into SphereLeafBlock.
    In this typed BLAS, a leaf lane means:
        node.counts[lane] > 0
        node.data[lane] = SphereLeafBlock index.
    """

    var tree: BoundsBvh[Self.width]
    var spheres: List[Sphere]
    var leaf_blocks: List[SphereLeafBlock[Self.width]]
    var sphere_count: UInt32

    def __init__[
        split_method: String = "median"
    ](
        out self,
        spheres: UnsafePointer[Sphere, MutAnyOrigin],
        sphere_count: UInt32,
    ):
        self.spheres = List[Sphere](capacity=Int(sphere_count))
        self.leaf_blocks = List[SphereLeafBlock[Self.width]]()
        self.sphere_count = sphere_count

        var items = List[BoundsItem](capacity=Int(sphere_count))

        for i in range(Int(sphere_count)):
            var s = spheres[i]
            self.spheres.append(s)
            items.append(BoundsItem(s.bounds(), UInt32(i)))

        var builder = BoundsBvhBuilder[Self.width](items)
        builder.build[split_method]()

        self.tree = BoundsBvh[Self.width](builder)

        self._pack_leaves()

    def bounds(self) -> AABB:
        return self.tree.root_bounds()

    def _pack_leaves(mut self):
        self.leaf_blocks = List[SphereLeafBlock[Self.width]]()

        for ref node in self.tree.nodes:
            comptime for lane in range(Self.width):
                if node.counts[lane] != EMPTY_LANE and node.counts[lane] > 0:
                    var first_item = node.data[lane]
                    var item_count = node.counts[lane]

                    var block = SphereLeafBlock[Self.width]()

                    comptime for k in range(Self.width):
                        if k < Int(item_count):
                            var item_ref = Int(
                                self.tree.item_indices[Int(first_item) + k]
                            )
                            var sphere_idx = self.tree.item_payloads[item_ref]
                            ref s = self.spheres[Int(sphere_idx)]

                            block.center.x[k] = s.center.x
                            block.center.y[k] = s.center.y
                            block.center.z[k] = s.center.z
                            block.radius[k] = s.radius

                            block.prim_indices[k] = sphere_idx
                            block.valid_lane[k] = True
                        else:
                            block.prim_indices[k] = EMPTY_LANE
                            block.valid_lane[k] = False

                    var block_idx = UInt32(len(self.leaf_blocks))
                    self.leaf_blocks.append(block^)
                    node.data[lane] = block_idx

    def trace[mode: TRACE](self, ray: Ray) -> Hit:
        def leaf_fn(
            ray: Ray,
            leaf_block_idx: UInt32,
            item_count: UInt32,
            mut hit: Hit,
        ) capturing -> Bool:
            ref block = self.leaf_blocks[Int(leaf_block_idx)]

            var O = Vec3[DType.float32, Self.width](ray.o.x, ray.o.y, ray.o.z)
            var D = Vec3[DType.float32, Self.width](ray.d.x, ray.d.y, ray.d.z)

            var h = intersect_ray_sphere(
                O,
                D,
                block.center,
                block.radius,
                hit.t,
            )

            var t_valid = h.t.ge(ray.t_min)
            var hit_mask = h.mask & t_valid & block.valid_lane

            if not hit_mask.reduce_or():
                return False

            comptime if mode == TRACE.ANY_HIT:
                return True
            else:
                _t = hit_mask.select(h.t, f32_max)
                min_t, arg_min_t = min_argmin(_t)

                if min_t < hit.t:
                    hit.t = min_t
                    hit.u = 0.0
                    hit.v = 0.0
                    hit.inst = EMPTY_LANE
                    hit.prim = block.prim_indices[arg_min_t]

                return True

        return trace_bounds_bvh[
            Self.width,
            mode,
            leaf_fn,
        ](
            self.tree,
            ray,
        )
