from bajo.core import AABB, Vec3, Point3, Frame, GeoKind, Rayf32
from bajo.bvh.types import Hit, Instance, TypedBvh
from bajo.bvh.constants import TRACE, EMPTY_LANE
from bajo.bvh.cpu.bounds_bvh import (
    BoundsBvh,
    BoundsBvhBuilder,
    BoundsItem,
)
from bajo.bvh.cpu.trace import trace_bounds_bvh


def _tree[
    width: SIMDLength, split_method: String
](
    instances: List[Instance],
    mut leaf_blocks: List[SIMD[DType.uint32, width]],
) -> BoundsBvh[Frame.WORLD, width]:
    var items = [
        BoundsItem(inst.bounds, UInt32(i)) for i, inst in enumerate(instances)
    ]
    var builder = BoundsBvhBuilder[Frame.WORLD, width](items^)
    builder.build[split_method]()
    leaf_blocks = List[SIMD[DType.uint32, width]](
        capacity=(Int(builder.nodes_used) + 1) // 2
    )

    @always_inline
    def pack_leaf(first_item: UInt32, item_count: UInt32) capturing -> UInt32:
        var first = Int(first_item)
        var count = Int(item_count)

        debug_assert["safe", _use_compiler_assume=True](
            count <= Int(width),
            "TLAS leaf exceeds SIMD width",
        )
        debug_assert["safe", _use_compiler_assume=True](
            first <= len(builder.item_indices)
            and count <= len(builder.item_indices) - first,
            "TLAS leaf range is outside item indices",
        )

        var inst_indices = SIMD[DType.uint32, width](EMPTY_LANE)

        for k in range(count):
            var item_ref = Int(builder.item_indices.unsafe_get(first + k))
            # Typed items are created in instance order.
            inst_indices[k] = UInt32(item_ref)

        var block_idx = UInt32(len(leaf_blocks))
        leaf_blocks.append(inst_indices)
        return block_idx

    var tree = BoundsBvh[Frame.WORLD, width].__init__[pack_leaf](builder)
    return tree^


struct Tlas[width: SIMDLength](Copyable):
    """Wide TLAS over Instance records.

    Binary-to-wide collapse packs each SIMD instance-index block in the same
    pass, so tagged leaf references point directly at typed blocks.
    """

    var tree: BoundsBvh[Frame.WORLD, Self.width]
    var instances: List[Instance]
    var leaf_blocks_inst_indices: List[SIMD[DType.uint32, Self.width]]
    var inst_count: Int

    def __init__[
        split_method: String = "lbvh"
    ](out self, instances: List[Instance]):
        self.instances = instances.copy()
        self.inst_count = len(self.instances)
        self.leaf_blocks_inst_indices = []
        self.tree = _tree[self.width, split_method](
            instances, self.leaf_blocks_inst_indices
        )

    def add_instance(mut self, instance: Instance):
        self.instances.append(instance.copy())
        self.inst_count += 1

    def build[split_method: String = "lbvh"](mut self):
        self.leaf_blocks_inst_indices = []
        self.tree = _tree[self.width, split_method](
            self.instances, self.leaf_blocks_inst_indices
        )

    def bounds(self) -> AABB[Frame.WORLD]:
        return self.tree.root_bounds()

    def trace[
        typed_bvh: TypedBvh,
        mode: TRACE,
    ](
        self,
        ray: Rayf32[Frame.WORLD],
        blases: Span[mut=False, typed_bvh, _],
    ) -> Hit[Frame.WORLD]:
        comptime assert (
            typed_bvh.bvh_frame == Frame.LOCAL
        ), "TLAS expects BLASes in Frame.LOCAL"

        def leaf_fn(
            ray: Rayf32[Frame.WORLD],
            O: Point3[DType.float32, Frame.WORLD, Self.width],
            D: Vec3[DType.float32, Frame.WORLD, Self.width],
            _ray_a: SIMD[DType.float32, Self.width],
            _ray_inv_a: SIMD[DType.float32, Self.width],
            leaf_block_idx: UInt32,
            mut hit: Hit[Frame.WORLD],
        ) capturing -> Bool:
            ref inst_indices = self.leaf_blocks_inst_indices.unsafe_get(
                Int(leaf_block_idx)
            )

            var any_hit = False

            comptime for lane in range(Self.width):
                var inst_idx = inst_indices[lane]

                if inst_idx != EMPTY_LANE:
                    ref inst = self.instances.unsafe_get(Int(inst_idx))
                    var local_ray_base = inst.inv_transform.ray(ray, hit.t)

                    # TODO: use this version when parametric raises are a thing in mojo
                    # var local_ray = inst.inv_transform.ray(ray, hit.t)
                    var local_ray = Rayf32[typed_bvh.bvh_frame](
                        local_ray_base.o.unsafe_convert[
                            new_frame=typed_bvh.bvh_frame
                        ](),
                        local_ray_base.d.unsafe_convert[
                            new_frame=typed_bvh.bvh_frame
                        ](),
                        local_ray_base.t_min,
                        local_ray_base.t_max,
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
                            hit.normal = inst.transform.normal(
                                local_hit.normal.unsafe_convert[
                                    new_frame=Frame.LOCAL
                                ](),
                                inst.inv_transform,
                            )
                            any_hit = True

            return any_hit

        return trace_bounds_bvh[
            Frame.WORLD,
            Self.width,
            Self.width,
            mode,
            leaf_fn,
        ](
            self.tree,
            ray,
        )
