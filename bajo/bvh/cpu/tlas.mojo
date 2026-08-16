from bajo.core import (
    AABB,
    Affine3f32,
    AxisAlignedBoundingBox,
    Vec3,
    Point3,
    Frame,
    GeoKind,
    Rayf32,
)
from bajo.core.intersect import intersect_ray_aabb_rcp
from bajo.bvh.types import Hit, Instance, TypedBvh
from bajo.bvh.constants import TRACE, EMPTY_LANE
from bajo.bvh.tlas_common import (
    finalize_tlas_hit_normal,
    promote_tlas_local_hit,
)
from bajo.bvh.cpu.bounds_bvh import (
    BoundsBvh,
    BoundsBvhBuilder,
    BoundsItem,
    _checked_typed_leaf_range,
)
from bajo.bvh.cpu.trace import trace_bounds_bvh, trace_bounds_bvh_leaf_rcp


@fieldwise_init
struct TlasLeafBlock[width: SIMDLength](Copyable):
    """SIMD instance bounds and indices consumed together during traversal."""

    var bounds: AxisAlignedBoundingBox[DType.float32, Frame.WORLD, Self.width]
    var inst_indices: SIMD[DType.uint32, Self.width]

    def __init__(out self):
        self.bounds = AxisAlignedBoundingBox[
            DType.float32, Frame.WORLD, Self.width
        ].invalid()
        self.inst_indices = SIMD[DType.uint32, Self.width](EMPTY_LANE)


@fieldwise_init
struct TlasHotInstance(Copyable):
    """Instance data touched for every surviving TLAS leaf candidate."""

    var inv_transform: Affine3f32[Frame.WORLD, Frame.LOCAL]
    var blas_idx: UInt32


@fieldwise_init
struct TlasColdInstance(Copyable):
    """Instance bounds used only while building packed TLAS leaves."""

    var bounds: AABB[Frame.WORLD]


def _split_instances(
    instances: List[Instance],
    mut hot_instances: List[TlasHotInstance],
    mut cold_instances: List[TlasColdInstance],
):
    hot_instances = List[TlasHotInstance](capacity=len(instances))
    cold_instances = List[TlasColdInstance](capacity=len(instances))
    for inst in instances:
        hot_instances.append(
            TlasHotInstance(inst.inv_transform.copy(), inst.blas_idx)
        )
        cold_instances.append(TlasColdInstance(inst.bounds))


def _tree[
    bounds_width: SIMDLength,
    leaf_width: SIMDLength,
    split_method: String,
](
    instances: List[TlasColdInstance],
    mut leaf_blocks: List[TlasLeafBlock[leaf_width]],
) -> BoundsBvh[Frame.WORLD, bounds_width]:
    var items = [
        BoundsItem(inst.bounds, UInt32(i)) for i, inst in enumerate(instances)
    ]
    var builder = BoundsBvhBuilder[Frame.WORLD, leaf_width](items^)
    builder.build[split_method]()
    leaf_blocks = List[TlasLeafBlock[leaf_width]](
        capacity=(Int(builder.nodes_used) + 1) // 2
    )

    @always_inline
    def pack_leaf(
        first_item: UInt32, item_count: UInt32
    ) {imm, mut leaf_blocks} -> UInt32:
        var first, count = _checked_typed_leaf_range[leaf_width](
            first_item, item_count, len(builder.item_indices)
        )

        var block = TlasLeafBlock[leaf_width]()

        for k in range(count):
            var item_ref = Int(builder.item_indices.unsafe_get(first + k))
            # Typed items are created in instance order.
            block.inst_indices[k] = UInt32(item_ref)
            ref bounds = instances.unsafe_get(item_ref).bounds
            block.bounds._min.x[k] = bounds._min.x[0]
            block.bounds._min.y[k] = bounds._min.y[0]
            block.bounds._min.z[k] = bounds._min.z[0]
            block.bounds._max.x[k] = bounds._max.x[0]
            block.bounds._max.y[k] = bounds._max.y[0]
            block.bounds._max.z[k] = bounds._max.z[0]

        var block_idx = UInt32(len(leaf_blocks))
        leaf_blocks.append(block^)
        return block_idx

    var tree = BoundsBvh[Frame.WORLD, bounds_width](builder, pack_leaf)
    return tree^


struct Tlas[
    bounds_width: SIMDLength,
    leaf_width: SIMDLength = bounds_width,
](Copyable):
    """Wide TLAS over Instance records.

    `bounds_width` controls the wide hierarchy and `leaf_width` controls the
    maximum instances in each packed leaf. Binary-to-wide collapse packs each
    SIMD instance-bounds/index block in the same pass, so tagged leaf
    references point directly at typed blocks. Traversal keeps inverse
    transforms and BLAS indices hot while build-only bounds remain cold.
    """

    comptime width = Self.bounds_width

    var tree: BoundsBvh[Frame.WORLD, Self.bounds_width]
    # Hot records are compact and touched after an instance AABB survives.
    var hot_instances: List[TlasHotInstance]
    # Cold records are bounds-only and used while building packed leaves.
    var cold_instances: List[TlasColdInstance]
    var leaf_blocks: List[TlasLeafBlock[Self.leaf_width]]
    var inst_count: Int

    def __init__[
        split_method: String = "lbvh"
    ](out self, instances: List[Instance]):
        self.hot_instances = []
        self.cold_instances = []
        _split_instances(instances, self.hot_instances, self.cold_instances)
        self.inst_count = len(self.cold_instances)
        self.leaf_blocks = []
        self.tree = _tree[Self.bounds_width, Self.leaf_width, split_method](
            self.cold_instances, self.leaf_blocks
        )

    def add_instance(mut self, instance: Instance):
        self.hot_instances.append(
            TlasHotInstance(instance.inv_transform.copy(), instance.blas_idx)
        )
        self.cold_instances.append(TlasColdInstance(instance.bounds))
        self.inst_count += 1

    def build[split_method: String = "lbvh"](mut self):
        self.leaf_blocks = []
        self.tree = _tree[Self.bounds_width, Self.leaf_width, split_method](
            self.cold_instances, self.leaf_blocks
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
            O: Point3[DType.float32, Frame.WORLD, Self.leaf_width],
            D: Vec3[DType.float32, Frame.WORLD, Self.leaf_width],
            _ray_a: SIMD[DType.float32, Self.leaf_width],
            _ray_inv_a: SIMD[DType.float32, Self.leaf_width],
            leaf_block_idx: UInt32,
            mut hit: Hit[Frame.WORLD],
        ) {imm} -> Bool:
            ref block = self.leaf_blocks.unsafe_get(Int(leaf_block_idx))
            var candidate_mask = block.inst_indices.ne(EMPTY_LANE)
            var candidate_t = SIMD[DType.float32, Self.leaf_width](0.0)
            comptime if Self.leaf_width > 1:
                var bounds_hit = intersect_ray_aabb_rcp(
                    O, D, block.bounds, hit.t
                )
                candidate_mask &= bounds_hit.mask
                candidate_t = bounds_hit.t

            var any_hit = False

            comptime for lane in range(Self.leaf_width):
                var inst_idx = block.inst_indices[lane]

                var visit_candidate = candidate_mask[lane]
                comptime if Self.leaf_width > 1:
                    visit_candidate &= candidate_t[lane] <= hit.t

                if visit_candidate:
                    ref hot_inst = self.hot_instances.unsafe_get(Int(inst_idx))
                    var local_ray_base = hot_inst.inv_transform.ray(ray, hit.t)

                    # Rebind Frame.LOCAL to the associated BLAS frame. Mojo
                    # does not yet fold those equivalent types at this call.
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

                    var local_hit = blases[Int(hot_inst.blas_idx)].trace[mode](
                        local_ray
                    )

                    comptime if mode == TRACE.ANY_HIT:
                        if local_hit.is_occluded():
                            return True
                    else:
                        if promote_tlas_local_hit(local_hit, inst_idx, hit):
                            any_hit = True

            return any_hit

        @always_inline
        def trace_tree() {imm} -> Hit[Frame.WORLD]:
            comptime if Self.leaf_width == 1:
                return trace_bounds_bvh[
                    frame=Frame.WORLD,
                    bounds_width=Self.bounds_width,
                    leaf_width=Self.leaf_width,
                    mode=mode,
                ](self.tree, ray, leaf_fn)
            else:
                return trace_bounds_bvh_leaf_rcp[
                    frame=Frame.WORLD,
                    bounds_width=Self.bounds_width,
                    leaf_width=Self.leaf_width,
                    mode=mode,
                ](self.tree, ray, leaf_fn)

        var hit = trace_tree()

        comptime if mode == TRACE.CLOSEST_HIT:
            if hit.is_hit():
                ref hot_inst = self.hot_instances.unsafe_get(Int(hit.inst))
                finalize_tlas_hit_normal(hit, hot_inst.inv_transform)

        return hit
