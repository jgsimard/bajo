from bajo.core import AABB, Vec3, Point3, Frame
from bajo.bvh.types import Ray, Hit, Instance, TypedBvh
from bajo.bvh.constants import TRACE, EMPTY_LANE
from bajo.bvh.cpu.bounds_bvh import BoundsBvh, BoundsBvhBuilder, BoundsItem
from bajo.bvh.cpu.trace import trace_bounds_bvh


def _tree[
    width: SIMDSize, split_method: String
](instances: List[Instance]) -> BoundsBvh[Frame.WORLD, width]:
    var builder = BoundsBvhBuilder[Frame.WORLD, width](
        [BoundsItem(inst.bounds, UInt32(i)) for i, inst in enumerate(instances)]
    )
    builder.build[split_method]()
    return BoundsBvh[Frame.WORLD, width](builder)


struct Tlas[width: SIMDSize](Copyable):
    """Wide TLAS over Instance records."""

    var tree: BoundsBvh[Frame.WORLD, Self.width]
    var instances: List[Instance]
    var leaf_blocks_inst_indices: List[SIMD[DType.uint32, Self.width]]
    var inst_count: Int

    def __init__[
        split_method: String = "lbvh"
    ](out self, instances: List[Instance]):
        self.instances = instances.copy()
        self.inst_count = len(self.instances)
        self.tree = _tree[self.width, split_method](instances)
        self.leaf_blocks_inst_indices = []
        self._pack_leaves()

    def add_instance(mut self, instance: Instance):
        self.instances.append(instance.copy())
        self.inst_count += 1

    def build[split_method: String = "lbvh"](mut self):
        self.tree = _tree[self.width, split_method](self.instances)
        self._pack_leaves()

    def bounds(self) -> AABB[Frame.WORLD]:
        return self.tree.root_bounds()

    def _pack_leaves(mut self):
        self.leaf_blocks_inst_indices = []

        for ref node in self.tree.nodes:
            comptime for lane in range(Self.width):
                if node.counts[lane] != EMPTY_LANE and node.counts[lane] > 0:
                    var first_item = node.data[lane]
                    var item_count = node.counts[lane]

                    var inst_indices = SIMD[DType.uint32, Self.width](
                        EMPTY_LANE
                    )

                    for k in range(Int(item_count)):
                        var item_ref = Int(
                            self.tree.item_indices[Int(first_item) + k]
                        )
                        inst_indices[k] = self.tree.item_payloads[item_ref]

                    var block_idx = UInt32(len(self.leaf_blocks_inst_indices))
                    self.leaf_blocks_inst_indices.append(inst_indices)

                    node.data[lane] = block_idx

    def trace[
        origin: ImmutOrigin,
        //,
        typed_bvh: TypedBvh,
        mode: TRACE,
    ](
        self,
        ray: Ray[Frame.WORLD],
        blases: UnsafePointer[typed_bvh, origin],
    ) -> Hit[Frame.WORLD]:
        comptime assert (
            typed_bvh.bvh_frame == Frame.LOCAL
        ), "TLAS expects BLASes in Frame.LOCAL"

        def leaf_fn(
            ray: Ray[Frame.WORLD],
            O: Point3[DType.float32, Frame.WORLD, Self.width],
            D: Vec3[DType.float32, Frame.WORLD, Self.width],
            leaf_block_idx: UInt32,
            mut hit: Hit[Frame.WORLD],
        ) capturing -> Bool:
            ref inst_indices = self.leaf_blocks_inst_indices[
                Int(leaf_block_idx)
            ]

            var any_hit = False

            comptime for lane in range(Self.width):
                var inst_idx = inst_indices[lane]

                if inst_idx != EMPTY_LANE:
                    ref inst = self.instances[Int(inst_idx)]
                    var local_origin = inst.inv_transform.point(ray.o)
                    var local_dir = inst.inv_transform.vector(ray.d)

                    # var local_ray = Ray[Frame.LOCAL](
                    #     local_origin, local_dir, ray.t_min, hit.t
                    # )
                    var local_ray = Ray[typed_bvh.bvh_frame](
                        local_origin.unsafe_convert_frame[
                            typed_bvh.bvh_frame
                        ](),
                        local_dir.unsafe_convert_frame[typed_bvh.bvh_frame](),
                        ray.t_min,
                        hit.t,
                    )

                    var local_hit = blases[Int(inst.blas_idx)].trace[mode](
                        local_ray
                    )

                    comptime if mode == TRACE.ANY_HIT:
                        if local_hit.is_occluded():
                            return True
                    else:
                        if local_hit.is_hit() and local_hit.t < hit.t:
                            hit.t = local_hit.t
                            hit.u = local_hit.u
                            hit.v = local_hit.v
                            hit.prim = local_hit.prim
                            hit.inst = inst_idx
                            hit.normal = inst.transform.vector(
                                local_hit.normal.unsafe_convert_frame[
                                    Frame.LOCAL
                                ]()
                            )
                            any_hit = True

            return any_hit

        return trace_bounds_bvh[
            Frame.WORLD,
            Self.width,
            mode,
            leaf_fn,
        ](
            self.tree,
            ray,
        )
