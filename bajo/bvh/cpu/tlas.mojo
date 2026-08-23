from bajo.core import (
    AABB,
    Affine3f32,
    AxisAlignedBoundingBox,
    Vec3,
    Point3,
    Rayf32,
)
from bajo.core.intersect import intersect_ray_aabb_rcp
from bajo.bvh.cpu.blas_storage import CpuBlasSet
from bajo.bvh.cpu.build_method import CpuBvhBuildMethod
from bajo.bvh.types import Hit, Instance
from bajo.bvh.cpu.blas_set import trace_blas_set
from bajo.bvh.constants import TraceMode, EMPTY_LANE, PrimitiveKind
from bajo.bvh.tlas_common import (
    finalize_tlas_hit_normal,
    promote_tlas_local_hit,
)
from bajo.bvh.cpu.bounds_bvh import (
    BoundsBvh,
    BinaryBoundsBvh,
    BoundsItem,
    _checked_typed_leaf_range,
)
from bajo.bvh.cpu.trace import trace_bounds_bvh, trace_bounds_bvh_leaf_rcp


comptime CpuBlasTraceFn[
    kind: PrimitiveKind,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
] = def(
    CpuBlasSet[kind, node_width, leaf_width],
    UInt32,
    Rayf32[.LOCAL],
) thin -> Hit[
    .LOCAL
]


@fieldwise_init
struct TlasLeafBlock[width: SIMDLength](Copyable):
    """SIMD instance bounds and indices consumed together during traversal."""

    var bounds: AxisAlignedBoundingBox[DType.float32, .WORLD, Self.width]
    var inst_indices: SIMD[DType.uint32, Self.width]

    def __init__(out self):
        self.bounds = AxisAlignedBoundingBox[
            DType.float32, .WORLD, Self.width
        ].invalid()
        self.inst_indices = SIMD[DType.uint32, Self.width](EMPTY_LANE)


@fieldwise_init
struct TlasHotInstance(Copyable):
    """Instance data touched for every surviving TLAS leaf candidate."""

    var inv_transform: Affine3f32[.WORLD, .LOCAL]
    var blas_idx: UInt32


@fieldwise_init
struct TlasColdInstance(Copyable):
    """Instance bounds used only while building packed TLAS leaves."""

    var bounds: AABB[.WORLD]


def _split_instances(
    instances: ImmSpan[Instance, _],
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
    method: CpuBvhBuildMethod,
](
    instances: ImmSpan[TlasColdInstance, _],
    mut leaf_blocks: List[TlasLeafBlock[leaf_width]],
) -> BoundsBvh[.WORLD, bounds_width]:
    var items = [
        BoundsItem(inst.bounds, UInt32(i)) for i, inst in enumerate(instances)
    ]
    var builder = BinaryBoundsBvh[.WORLD, leaf_width, method](items^)
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

    var tree = BoundsBvh[.WORLD, bounds_width](builder, pack_leaf)
    return tree^


@always_inline
def _trace_tlas_tree[
    bounds_width: SIMDLength,
    leaf_width: SIMDLength,
    mode: TraceMode,
    LeafFn: def(
        Rayf32[.WORLD],
        Point3[DType.float32, .WORLD, leaf_width],
        Vec3[.float32, .WORLD, leaf_width],
        SIMD[.float32, leaf_width],
        SIMD[.float32, leaf_width],
        UInt32,
        mut Hit[.WORLD],
    ) -> Bool,
](
    tree: BoundsBvh[.WORLD, bounds_width],
    ray: Rayf32[.WORLD],
    ref leaf_fn: LeafFn,
) -> Hit[.WORLD]:
    """Select the leaf-direction ABI while retaining one traversal call."""
    comptime if leaf_width == 1:
        return trace_bounds_bvh[
            frame=.WORLD,
            bounds_width=bounds_width,
            leaf_width=leaf_width,
            mode=mode,
        ](tree, ray, leaf_fn)
    return trace_bounds_bvh_leaf_rcp[
        frame=.WORLD,
        bounds_width=bounds_width,
        leaf_width=leaf_width,
        mode=mode,
    ](tree, ray, leaf_fn)


struct CpuTlas[
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

    var tree: BoundsBvh[.WORLD, Self.bounds_width]
    # Hot records are compact and touched after an instance AABB survives.
    var hot_instances: List[TlasHotInstance]
    var leaf_blocks: List[TlasLeafBlock[Self.leaf_width]]

    def __init__[
        method: CpuBvhBuildMethod = .LBVH
    ](out self, instances: ImmSpan[Instance, _]):
        self.hot_instances = []
        var cold_instances = List[TlasColdInstance]()
        _split_instances(instances, self.hot_instances, cold_instances)
        self.leaf_blocks = []
        self.tree = _tree[Self.bounds_width, Self.leaf_width, method](
            cold_instances, self.leaf_blocks
        )

    def bounds(self) -> AABB[.WORLD]:
        return self.tree.root_bounds()

    def _trace_packed_blases[
        kind: PrimitiveKind,
        blas_node_width: SIMDLength,
        blas_leaf_width: SIMDLength,
        mode: TraceMode,
        trace_fn: CpuBlasTraceFn[kind, blas_node_width, blas_leaf_width],
    ](
        self,
        ray: Rayf32[.WORLD],
        blases: CpuBlasSet[kind, blas_node_width, blas_leaf_width],
    ) -> Hit[.WORLD]:
        """Shared TLAS traversal for descriptor-backed CPU BLAS sets."""

        def leaf_fn(
            ray: Rayf32[.WORLD],
            O: Point3[DType.float32, .WORLD, Self.leaf_width],
            D: Vec3[.float32, .WORLD, Self.leaf_width],
            _ray_a: SIMD[.float32, Self.leaf_width],
            _ray_inv_a: SIMD[.float32, Self.leaf_width],
            leaf_block_idx: UInt32,
            mut hit: Hit[.WORLD],
        ) {imm} -> Bool:
            ref block = self.leaf_blocks.unsafe_get(Int(leaf_block_idx))
            var candidate_mask = block.inst_indices.ne(EMPTY_LANE)
            var candidate_t = SIMD[.float32, Self.leaf_width](0.0)
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
                    var local_ray = hot_inst.inv_transform.ray(ray, hit.t)
                    var local_hit: Hit[.LOCAL]
                    local_hit = trace_fn(blases, hot_inst.blas_idx, local_ray)

                    comptime if mode == .ANY_HIT:
                        if local_hit.is_occluded():
                            return True
                    else:
                        if promote_tlas_local_hit(local_hit, inst_idx, hit):
                            any_hit = True
            return any_hit

        var hit = _trace_tlas_tree[
            Self.bounds_width, Self.leaf_width, mode
        ](self.tree, ray, leaf_fn)

        comptime if mode == .CLOSEST_HIT:
            if hit.is_hit():
                ref hot_inst = self.hot_instances.unsafe_get(Int(hit.inst))
                finalize_tlas_hit_normal(hit, hot_inst.inv_transform)
        return hit

    def trace_blases[
        blas_node_width: SIMDLength,
        blas_leaf_width: SIMDLength = blas_node_width,
        mode: TraceMode = .CLOSEST_HIT,
    ](
        self,
        ray: Rayf32[.WORLD],
        blases: CpuBlasSet[
            .TRIANGLE, blas_node_width, blas_leaf_width
        ],
    ) -> Hit[.WORLD]:
        return self._trace_packed_blases[
            .TRIANGLE,
            blas_node_width,
            blas_leaf_width,
            mode,
            trace_blas_set[blas_node_width, blas_leaf_width, mode, .LOCAL],
        ](ray, blases)

    def trace_blases[
        blas_node_width: SIMDLength,
        blas_leaf_width: SIMDLength = blas_node_width,
        mode: TraceMode = .CLOSEST_HIT,
    ](
        self,
        ray: Rayf32[.WORLD],
        blases: CpuBlasSet[.SPHERE, blas_node_width, blas_leaf_width],
    ) -> Hit[.WORLD]:
        return self._trace_packed_blases[
            .SPHERE,
            blas_node_width,
            blas_leaf_width,
            mode,
            trace_blas_set[blas_node_width, blas_leaf_width, mode, .LOCAL],
        ](ray, blases)
