from bajo.core import AABB, Vec3, Point3, Frame
from bajo.core.intersect import intersect_ray_sphere
from bajo.core.utils import min_argmin
from bajo.bvh.constants import EMPTY_LANE, TRACE, f32_max
from bajo.bvh.cpu.bounds_bvh import BoundsBvh, BoundsItem, BoundsBvhBuilder
from bajo.bvh.cpu.trace import trace_bounds_bvh
from bajo.bvh.types import Ray, Hit, Sphere, SphereLeafBlock, TypedBvh


struct SphereBvh[frame: Frame, width: SIMDSize](Copyable, TypedBvh):
    comptime bvh_frame: Frame = Self.frame

    """Sphere-specific wrapper around BoundsBvh[width].
    The generic tree is built from BoundsItem ranges.
    After construction, sphere leaf data is packed into SphereLeafBlock.
    In this typed BLAS, a leaf lane means:
        node.counts[lane] > 0
        node.data[lane] = SphereLeafBlock index.
    """

    var tree: BoundsBvh[Self.frame, Self.width]
    var spheres: List[Sphere[Self.frame]]
    var leaf_blocks: List[SphereLeafBlock[Self.frame, Self.width]]
    var sphere_count: Int

    def __init__[
        split_method: String = "median"
    ](out self, var spheres: List[Sphere[Self.frame]]):
        self.spheres = spheres^
        self.sphere_count = len(self.spheres)
        self.leaf_blocks = []

        var items = [
            BoundsItem(s.bounds(), UInt32(i))
            for i, s in enumerate(self.spheres)
        ]

        var builder = BoundsBvhBuilder[Self.frame, Self.width](items)
        builder.build[split_method]()

        self.tree = BoundsBvh[Self.frame, Self.width](builder)

        self._pack_leaves()

    def bounds(self) -> AABB[Self.frame]:
        return self.tree.root_bounds()

    def _pack_leaves(mut self):
        self.leaf_blocks = List[SphereLeafBlock[Self.frame, Self.width]]()

        for ref node in self.tree.nodes:
            comptime for lane in range(Self.width):
                if node.counts[lane] != EMPTY_LANE and node.counts[lane] > 0:
                    var first_item = node.data[lane]
                    var item_count = node.counts[lane]

                    var block = SphereLeafBlock[Self.frame, Self.width]()

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

                    var block_idx = UInt32(len(self.leaf_blocks))
                    self.leaf_blocks.append(block^)
                    node.data[lane] = block_idx

    def trace[
        mode: TRACE
    ](self, ray: Ray[Self.bvh_frame]) -> Hit[Self.bvh_frame]:
        def leaf_fn(
            ray: Ray[Self.bvh_frame],
            O: Point3[DType.float32, Self.bvh_frame, Self.width],
            D: Vec3[DType.float32, Self.bvh_frame, Self.width],
            leaf_block_idx: UInt32,
            mut hit: Hit[Self.bvh_frame],
        ) capturing -> Bool:
            ref block = self.leaf_blocks[Int(leaf_block_idx)]

            var h = intersect_ray_sphere(
                O,
                D,
                block.center,
                block.radius,
                hit.t,
                ray.t_min,
            )
            var valid_lane = block.prim_indices.ne(EMPTY_LANE)
            var hit_mask = h.mask & valid_lane

            if not hit_mask.reduce_or():
                return False

            comptime if mode == TRACE.CLOSEST_HIT:
                _t = hit_mask.select(h.t, f32_max)
                min_t, arg_min_t = min_argmin(_t)

                hit.t = min_t
                hit.u = 0.0
                hit.v = 0.0
                hit.inst = EMPTY_LANE
                hit.prim = block.prim_indices[arg_min_t]

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
