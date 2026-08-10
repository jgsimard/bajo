from bajo.core import AABB, Vec3, Point3, Frame, GeoKind, Rayf32
from bajo.bvh.types import Hit, Instance, TypedBvh
from bajo.bvh.constants import TRACE, EMPTY_LANE
from bajo.bvh.cpu.bounds_bvh import (
    BoundsBvh,
    BoundsBvhBuilder,
    BoundsItem,
    decode_ref_index,
    encode_leaf_ref,
    is_leaf_ref,
)
from bajo.bvh.cpu.trace import trace_bounds_bvh


def _tree[
    width: SIMDLength, split_method: String
](instances: List[Instance]) -> BoundsBvh[Frame.WORLD, width]:
    var items = [
        BoundsItem(inst.bounds, UInt32(i)) for i, inst in enumerate(instances)
    ]
    var builder = BoundsBvhBuilder[Frame.WORLD, width](items^)
    builder.build[split_method]()
    return BoundsBvh[Frame.WORLD, width](builder)


struct Tlas[width: SIMDLength](Copyable):
    """Wide TLAS over Instance records.

    During BoundsBvh construction, a tagged leaf reference points into
    tree.leaf_ranges.

    After construction, each leaf range is packed into a SIMD block of
    instance indices and the tagged leaf payload is rewritten to the packed
    block index.
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
        self.leaf_blocks_inst_indices = List[SIMD[DType.uint32, Self.width]](
            capacity=(self.inst_count + Int(Self.width) - 1) // Int(Self.width)
        )

        debug_assert["safe", _use_compiler_assume=True](
            len(self.tree.item_indices) == len(self.tree.item_payloads),
            "TLAS item arrays have inconsistent lengths",
        )
        debug_assert["safe", _use_compiler_assume=True](
            len(self.instances) == self.inst_count,
            "TLAS instance count changed while packing leaves",
        )

        for ref node in self.tree.nodes:
            comptime for lane in range(Self.width):
                var child_ref = node.data[lane]

                if child_ref != EMPTY_LANE and is_leaf_ref(child_ref):
                    var leaf_range_idx = decode_ref_index(child_ref)
                    ref leaf_range = self.tree.leaf_ranges.unsafe_get(
                        Int(leaf_range_idx)
                    )

                    var first = Int(leaf_range.first_item)
                    var count = Int(leaf_range.item_count)

                    debug_assert["safe", _use_compiler_assume=True](
                        count <= Int(Self.width),
                        "TLAS leaf exceeds SIMD width",
                    )
                    debug_assert["safe", _use_compiler_assume=True](
                        first <= len(self.tree.item_indices)
                        and count <= len(self.tree.item_indices) - first,
                        "TLAS leaf range is outside item indices",
                    )

                    var inst_indices = SIMD[DType.uint32, Self.width](
                        EMPTY_LANE
                    )

                    for k in range(count):
                        var item_ref = Int(
                            self.tree.item_indices.unsafe_get(first + k)
                        )
                        inst_indices[k] = self.tree.item_payloads.unsafe_get(
                            item_ref
                        )

                    var block_idx = UInt32(len(self.leaf_blocks_inst_indices))
                    self.leaf_blocks_inst_indices.append(inst_indices)

                    # Preserve the leaf tag, but replace the construction-time
                    # leaf-range payload with the packed instance-block index.
                    node.data[lane] = encode_leaf_ref(block_idx)

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
