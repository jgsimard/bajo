from std.bit import count_trailing_zeros
from std.math import fma
from std.memory import pack_bits

from bajo.core import (
    AABB,
    Affine3f32,
    AxisAlignedBoundingBox,
    Normal3f32,
    Vec3,
    Point3,
    Rayf32,
    Ray,
    normalize,
)
from bajo.core.intersect import intersect_ray_aabb_rcp
from bajo.bvh.cpu.blas_storage import CpuBlasSet
from bajo.bvh.cpu.build_method import CpuBvhBuildMethod
from bajo.bvh.types import BlasDesc, Hit, Instance
from bajo.bvh.cpu.blas_set import (
    _debug_check_blas_index,
    _trace_blas_desc_precomputed_rcp,
    _trace_blas_set_precomputed_rcp,
    trace_blas_set,
)
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
from bajo.bvh.cpu.trace import (
    trace_bounds_bvh,
    trace_bounds_bvh_leaf_rcp,
)
from bajo.bvh.cpu.packet import trace_packet_stack_bounds_bvh
from bajo.bvh.tagged_ref import decode_ref_index, is_leaf_ref


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


__extension SIMD:
    @always_inline
    def to_array(self) -> Array[Scalar[Self.dtype], Self.length]:
        return Array[Scalar[Self.dtype], Self.length](
            fill_with=lambda (i: Int) -> Scalar[Self.dtype]: self[i]
        )


@fieldwise_init
struct TlasLeafBlock[width: SIMDLength](Copyable):
    """SIMD instance bounds and indices consumed together during traversal."""

    var bounds: AxisAlignedBoundingBox[.float32, .WORLD, Self.width]
    var inst_indices: SIMD[.uint32, Self.width]

    def __init__(out self):
        self.bounds = AxisAlignedBoundingBox[
            .float32, .WORLD, Self.width
        ].invalid()
        self.inst_indices = SIMD[.uint32, Self.width](EMPTY_LANE)


@fieldwise_init
struct TlasHotInstance(Copyable):
    """Instance data touched for every surviving TLAS leaf candidate."""

    var inv_transform: Affine3f32[.WORLD, .LOCAL]
    var blas_idx: UInt32
    var translation_only: Bool


@fieldwise_init
struct TlasColdInstance(Copyable):
    """Instance bounds used only while building packed TLAS leaves."""

    var bounds: AABB[.WORLD]


def _is_translation_only(
    inverse: Affine3f32[.WORLD, .LOCAL],
) -> Bool:
    return (
        inverse.m00 == 1.0
        and inverse.m01 == 0.0
        and inverse.m02 == 0.0
        and inverse.m10 == 0.0
        and inverse.m11 == 1.0
        and inverse.m12 == 0.0
        and inverse.m20 == 0.0
        and inverse.m21 == 0.0
        and inverse.m22 == 1.0
    )


@always_inline
def _transform_world_ray_packet[
    length: SIMDLength,
](
    inverse: Affine3f32[.WORLD, .LOCAL],
    rays: Ray[.float32, .WORLD, length],
    t_max: SIMD[.float32, length],
) -> Ray[.float32, .LOCAL, length]:
    var m00 = SIMD[.float32, length](inverse.m00)
    var m01 = SIMD[.float32, length](inverse.m01)
    var m02 = SIMD[.float32, length](inverse.m02)
    var m10 = SIMD[.float32, length](inverse.m10)
    var m11 = SIMD[.float32, length](inverse.m11)
    var m12 = SIMD[.float32, length](inverse.m12)
    var m20 = SIMD[.float32, length](inverse.m20)
    var m21 = SIMD[.float32, length](inverse.m21)
    var m22 = SIMD[.float32, length](inverse.m22)
    return Ray[.float32, .LOCAL, length](
        Point3[.float32, .LOCAL, length](
            fma(
                m00,
                rays.o.x,
                fma(
                    m01,
                    rays.o.y,
                    fma(m02, rays.o.z, SIMD[.float32, length](inverse.tx)),
                ),
            ),
            fma(
                m10,
                rays.o.x,
                fma(
                    m11,
                    rays.o.y,
                    fma(m12, rays.o.z, SIMD[.float32, length](inverse.ty)),
                ),
            ),
            fma(
                m20,
                rays.o.x,
                fma(
                    m21,
                    rays.o.y,
                    fma(m22, rays.o.z, SIMD[.float32, length](inverse.tz)),
                ),
            ),
        ),
        Vec3[.float32, .LOCAL, length](
            fma(m00, rays.d.x, fma(m01, rays.d.y, m02 * rays.d.z)),
            fma(m10, rays.d.x, fma(m11, rays.d.y, m12 * rays.d.z)),
            fma(m20, rays.d.x, fma(m21, rays.d.y, m22 * rays.d.z)),
        ),
        rays.t_min,
        t_max,
    )


@always_inline
def _translate_world_ray_packet[
    length: SIMDLength,
](
    inverse: Affine3f32[.WORLD, .LOCAL],
    rays: Ray[.float32, .WORLD, length],
    t_max: SIMD[.float32, length],
) -> Ray[.float32, .LOCAL, length]:
    return Ray[.float32, .LOCAL, length](
        Point3[.float32, .LOCAL, length](
            rays.o.x + SIMD[.float32, length](inverse.tx),
            rays.o.y + SIMD[.float32, length](inverse.ty),
            rays.o.z + SIMD[.float32, length](inverse.tz),
        ),
        Vec3[.float32, .LOCAL, length](rays.d.x, rays.d.y, rays.d.z),
        rays.t_min,
        t_max,
    )


def _split_instances(
    instances: ImmSpan[Instance, _],
    mut hot_instances: List[TlasHotInstance],
    mut cold_instances: List[TlasColdInstance],
):
    hot_instances = List[TlasHotInstance](capacity=len(instances))
    cold_instances = List[TlasColdInstance](capacity=len(instances))
    for inst in instances:
        hot_instances.append(
            TlasHotInstance(
                inst.inv_transform.copy(),
                inst.blas_idx,
                _is_translation_only(inst.inv_transform),
            )
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
        Point3[.float32, .WORLD, leaf_width],
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
    var all_translation_only: Bool

    def __init__[
        method: CpuBvhBuildMethod = .LBVH
    ](out self, instances: ImmSpan[Instance, _]):
        self.hot_instances = []
        var cold_instances = List[TlasColdInstance]()
        _split_instances(instances, self.hot_instances, cold_instances)
        self.all_translation_only = True
        for hot_inst in self.hot_instances:
            self.all_translation_only &= hot_inst.translation_only
        self.leaf_blocks = []
        self.tree = _tree[Self.bounds_width, Self.leaf_width, method](
            cold_instances, self.leaf_blocks
        )

    def bounds(self) -> AABB[.WORLD]:
        return self.tree.root_bounds()

    def _refit_node(mut self, node_idx: UInt32) -> AABB[.WORLD]:
        """Recompute one wide subtree bottom-up without changing topology."""
        var subtree_bounds = AABB[.WORLD].invalid()
        comptime for lane in range(Self.bounds_width):
            var child_ref = self.tree.nodes[Int(node_idx)].data[lane]
            if child_ref == EMPTY_LANE:
                continue

            var child_bounds = AABB[.WORLD].invalid()
            if is_leaf_ref(child_ref):
                ref block = self.leaf_blocks.unsafe_get(
                    Int(decode_ref_index(child_ref))
                )
                comptime for inst_lane in range(Self.leaf_width):
                    if block.inst_indices[inst_lane] != EMPTY_LANE:
                        child_bounds.grow(
                            Point3[.float32, .WORLD](
                                block.bounds._min.x[inst_lane],
                                block.bounds._min.y[inst_lane],
                                block.bounds._min.z[inst_lane],
                            ),
                            Point3[.float32, .WORLD](
                                block.bounds._max.x[inst_lane],
                                block.bounds._max.y[inst_lane],
                                block.bounds._max.z[inst_lane],
                            ),
                        )
            else:
                child_bounds = self._refit_node(decode_ref_index(child_ref))

            self.tree.nodes[Int(node_idx)].aabb._min.x[
                lane
            ] = child_bounds._min.x
            self.tree.nodes[Int(node_idx)].aabb._min.y[
                lane
            ] = child_bounds._min.y
            self.tree.nodes[Int(node_idx)].aabb._min.z[
                lane
            ] = child_bounds._min.z
            self.tree.nodes[Int(node_idx)].aabb._max.x[
                lane
            ] = child_bounds._max.x
            self.tree.nodes[Int(node_idx)].aabb._max.y[
                lane
            ] = child_bounds._max.y
            self.tree.nodes[Int(node_idx)].aabb._max.z[
                lane
            ] = child_bounds._max.z
            subtree_bounds.grow(child_bounds)
        return subtree_bounds

    def refit(mut self, instances: ImmSpan[Instance, _]):
        """Update instance data and bounds while preserving TLAS topology.

        The instance count and ordering must match construction. Refit is
        intended for animation with modest motion; rebuild after large motion
        to recover spatial quality.
        """
        debug_assert["safe", _use_compiler_assume=True](
            len(instances) == len(self.hot_instances),
            "TLAS refit requires unchanged instance count and ordering",
        )

        self.all_translation_only = True
        for inst_idx in range(len(instances)):
            ref inst = instances[inst_idx]
            self.hot_instances[
                inst_idx
            ].inv_transform = inst.inv_transform.copy()
            self.hot_instances[inst_idx].blas_idx = inst.blas_idx
            self.hot_instances[
                inst_idx
            ].translation_only = _is_translation_only(inst.inv_transform)
            self.all_translation_only &= self.hot_instances[
                inst_idx
            ].translation_only

        for block_idx in range(len(self.leaf_blocks)):
            comptime for lane in range(Self.leaf_width):
                var inst_idx = self.leaf_blocks[block_idx].inst_indices[lane]
                if inst_idx != EMPTY_LANE:
                    ref bounds = instances[Int(inst_idx)].bounds
                    self.leaf_blocks[block_idx].bounds._min.x[
                        lane
                    ] = bounds._min.x
                    self.leaf_blocks[block_idx].bounds._min.y[
                        lane
                    ] = bounds._min.y
                    self.leaf_blocks[block_idx].bounds._min.z[
                        lane
                    ] = bounds._min.z
                    self.leaf_blocks[block_idx].bounds._max.x[
                        lane
                    ] = bounds._max.x
                    self.leaf_blocks[block_idx].bounds._max.y[
                        lane
                    ] = bounds._max.y
                    self.leaf_blocks[block_idx].bounds._max.z[
                        lane
                    ] = bounds._max.z

        if len(self.tree.nodes) > 0:
            _ = self._refit_node(0)

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
            O: Point3[.float32, .WORLD, Self.leaf_width],
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

        var hit = _trace_tlas_tree[Self.bounds_width, Self.leaf_width, mode](
            self.tree, ray, leaf_fn
        )

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
        blases: CpuBlasSet[.TRIANGLE, blas_node_width, blas_leaf_width],
    ) -> Hit[.WORLD]:
        return self._trace_packed_blases[
            .TRIANGLE,
            blas_node_width,
            blas_leaf_width,
            mode,
            trace_blas_set[blas_node_width, blas_leaf_width, mode, .LOCAL],
        ](ray, blases)

    def _trace_blases_packet_impl[
        blas_node_width: SIMDLength,
        blas_leaf_width: SIMDLength,
        length: SIMDLength,
        simd_normal_transforms: Bool = True,
        translations_only: Bool = False,
    ](
        self,
        rays: Ray[.float32, .WORLD, length],
        blases: CpuBlasSet[.TRIANGLE, blas_node_width, blas_leaf_width],
        valid: SIMD[.bool, length] = SIMD[.bool, length](fill=True),
    ) -> Hit[.WORLD, length]:
        """Traverse the TLAS and each surviving BLAS as masked packets."""
        comptime assert length > 1
        var hit = Hit[.WORLD, length].miss(rays.t_max)
        var reciprocal_direction = rays.reciprocal_direction()
        comptime candidate_capacity = 16
        var candidate_tasks = Array[UInt64, candidate_capacity](
            uninitialized=True
        )
        var candidate_count = 0
        var candidate_overflow = List[UInt64]()

        def leaf_fn(
            active: SIMD[.bool, length],
            leaf_block_idx: UInt32,
            mut packet_hit: Hit[.WORLD, length],
        ) {
            imm,
            mut candidate_tasks,
            mut candidate_count,
            mut candidate_overflow,
        }:
            ref block = self.leaf_blocks.unsafe_get(Int(leaf_block_idx))
            comptime for inst_lane in range(Self.leaf_width):
                var inst_idx = block.inst_indices[inst_lane]
                if inst_idx == EMPTY_LANE:
                    continue
                var candidate = active
                comptime if Self.leaf_width > 1:
                    var bmin = Point3[.float32, .WORLD, length](
                        block.bounds._min.x[inst_lane],
                        block.bounds._min.y[inst_lane],
                        block.bounds._min.z[inst_lane],
                    )
                    var bmax = Point3[.float32, .WORLD, length](
                        block.bounds._max.x[inst_lane],
                        block.bounds._max.y[inst_lane],
                        block.bounds._max.z[inst_lane],
                    )
                    candidate &= intersect_ray_aabb_rcp(
                        rays.o,
                        reciprocal_direction,
                        bmin,
                        bmax,
                        packet_hit.t,
                    ).mask
                var bits = UInt32(pack_bits(candidate))
                if bits != 0:
                    var task = UInt64(inst_idx) << 32 | UInt64(bits)
                    if candidate_count < candidate_capacity:
                        candidate_tasks.unsafe_get(candidate_count) = task
                        candidate_count += 1
                    else:
                        candidate_overflow.append(task)

        trace_packet_stack_bounds_bvh[
            frame=.WORLD,
            bounds_width=Self.bounds_width,
            length=length,
            # BLAS candidates are consumed after TLAS traversal, so hit.t
            # cannot tighten here and near-order sorting cannot prune work.
            ordered_tasks=False,
        ](
            self.tree.nodes,
            rays,
            reciprocal_direction,
            valid,
            hit,
            leaf_fn,
            lambda (
                _active: SIMD[.bool, length],
                _child_ref: UInt32,
                mut _packet_hit: Hit[.WORLD, length],
            ): None,
            lambda (_child_ref: UInt32): None,
        )

        if candidate_count == 0:
            return hit

        var closest_t = hit.t.to_array()
        var scalar_rays = Array[Float32, 10 * length](uninitialized=True)
        comptime for lane in range(length):
            scalar_rays.unsafe_get(0 * Int(length) + lane) = rays.o.x[lane]
            scalar_rays.unsafe_get(1 * Int(length) + lane) = rays.o.y[lane]
            scalar_rays.unsafe_get(2 * Int(length) + lane) = rays.o.z[lane]
            scalar_rays.unsafe_get(3 * Int(length) + lane) = rays.d.x[lane]
            scalar_rays.unsafe_get(4 * Int(length) + lane) = rays.d.y[lane]
            scalar_rays.unsafe_get(5 * Int(length) + lane) = rays.d.z[lane]
            scalar_rays.unsafe_get(6 * Int(length) + lane) = rays.t_min[lane]
            scalar_rays.unsafe_get(
                7 * Int(length) + lane
            ) = reciprocal_direction.x[lane]
            scalar_rays.unsafe_get(
                8 * Int(length) + lane
            ) = reciprocal_direction.y[lane]
            scalar_rays.unsafe_get(
                9 * Int(length) + lane
            ) = reciprocal_direction.z[lane]
        var scalar_rays_ptr = scalar_rays.unsafe_ptr()
        var result_u = hit.u.to_array()
        var result_v = hit.v.to_array()
        var result_prim = hit.prim.to_array()
        var result_inst = hit.inst.to_array()
        var result_nx = hit.normal.x.to_array()
        var result_ny = hit.normal.y.to_array()
        var result_nz = hit.normal.z.to_array()

        @always_inline
        def consume_candidates() {
            self,
            blases,
            scalar_rays_ptr,
            candidate_tasks,
            candidate_count,
            mut closest_t,
            mut result_u,
            mut result_v,
            mut result_prim,
            mut result_inst,
            mut result_nx,
            mut result_ny,
            mut result_nz,
        }:
            for task_idx in range(candidate_count):
                var task = candidate_tasks.unsafe_get(task_idx)
                var inst_idx = UInt32(task >> 32)
                ref hot_inst = self.hot_instances.unsafe_get(Int(inst_idx))
                var bits = UInt32(task)
                _debug_check_blas_index(hot_inst.blas_idx, blases.blas_count)
                var desc = BlasDesc.load(
                    blases.descs.unsafe_ptr(), hot_inst.blas_idx
                )
                if desc.prim_count == 0:
                    continue
                while bits != 0:
                    var lane = Int(count_trailing_zeros(bits))
                    bits &= bits - 1
                    var ray = Rayf32[.WORLD](
                        Point3[.float32, .WORLD](
                            scalar_rays_ptr[
                                unsafe_offset=0 * Int(length) + lane
                            ],
                            scalar_rays_ptr[
                                unsafe_offset=1 * Int(length) + lane
                            ],
                            scalar_rays_ptr[
                                unsafe_offset=2 * Int(length) + lane
                            ],
                        ),
                        Vec3[.float32, .WORLD](
                            scalar_rays_ptr[
                                unsafe_offset=3 * Int(length) + lane
                            ],
                            scalar_rays_ptr[
                                unsafe_offset=4 * Int(length) + lane
                            ],
                            scalar_rays_ptr[
                                unsafe_offset=5 * Int(length) + lane
                            ],
                        ),
                        scalar_rays_ptr[unsafe_offset=6 * Int(length) + lane],
                        closest_t.unsafe_get(lane),
                    )
                    var local_ray: Rayf32[.LOCAL]
                    var local_hit: Hit[.LOCAL]
                    comptime if translations_only:
                        local_ray = Rayf32[.LOCAL](
                            Point3[.float32, .LOCAL](
                                ray.o.x + hot_inst.inv_transform.tx,
                                ray.o.y + hot_inst.inv_transform.ty,
                                ray.o.z + hot_inst.inv_transform.tz,
                            ),
                            Vec3[.float32, .LOCAL](ray.d.x, ray.d.y, ray.d.z),
                            ray.t_min,
                            closest_t.unsafe_get(lane),
                        )
                        var local_rcp = Vec3[.float32, .LOCAL, blas_node_width](
                            scalar_rays_ptr[
                                unsafe_offset=7 * Int(length) + lane
                            ],
                            scalar_rays_ptr[
                                unsafe_offset=8 * Int(length) + lane
                            ],
                            scalar_rays_ptr[
                                unsafe_offset=9 * Int(length) + lane
                            ],
                        )
                        local_hit = _trace_blas_desc_precomputed_rcp[
                            blas_node_width,
                            blas_leaf_width,
                            .CLOSEST_HIT,
                            .LOCAL,
                        ](
                            blases,
                            desc,
                            local_ray,
                            local_rcp,
                        )
                    else:
                        if hot_inst.translation_only:
                            local_ray = Rayf32[.LOCAL](
                                Point3[.float32, .LOCAL](
                                    ray.o.x + hot_inst.inv_transform.tx,
                                    ray.o.y + hot_inst.inv_transform.ty,
                                    ray.o.z + hot_inst.inv_transform.tz,
                                ),
                                Vec3[.float32, .LOCAL](
                                    ray.d.x, ray.d.y, ray.d.z
                                ),
                                ray.t_min,
                                closest_t.unsafe_get(lane),
                            )
                            var local_rcp = Vec3[
                                .float32, .LOCAL, blas_node_width
                            ](
                                scalar_rays_ptr[
                                    unsafe_offset=7 * Int(length) + lane
                                ],
                                scalar_rays_ptr[
                                    unsafe_offset=8 * Int(length) + lane
                                ],
                                scalar_rays_ptr[
                                    unsafe_offset=9 * Int(length) + lane
                                ],
                            )
                            local_hit = _trace_blas_desc_precomputed_rcp[
                                blas_node_width,
                                blas_leaf_width,
                                .CLOSEST_HIT,
                                .LOCAL,
                            ](
                                blases,
                                desc,
                                local_ray,
                                local_rcp,
                            )
                        else:
                            local_ray = hot_inst.inv_transform.ray(
                                ray, closest_t.unsafe_get(lane)
                            )
                            local_hit = trace_blas_set[
                                blas_node_width,
                                blas_leaf_width,
                                .CLOSEST_HIT,
                                .LOCAL,
                            ](blases, hot_inst.blas_idx, local_ray)
                    if (
                        local_hit.is_hit()
                        and local_hit.t < closest_t.unsafe_get(lane)
                    ):
                        closest_t.unsafe_get(lane) = local_hit.t
                        result_u.unsafe_get(lane) = local_hit.u
                        result_v.unsafe_get(lane) = local_hit.v
                        result_prim.unsafe_get(lane) = local_hit.prim
                        result_inst.unsafe_get(lane) = inst_idx
                        result_nx.unsafe_get(lane) = local_hit.normal.x
                        result_ny.unsafe_get(lane) = local_hit.normal.y
                        result_nz.unsafe_get(lane) = local_hit.normal.z

        consume_candidates()
        var overflow_offset = 0
        while overflow_offset < len(candidate_overflow):
            candidate_count = min(
                candidate_capacity, len(candidate_overflow) - overflow_offset
            )
            for task_idx in range(candidate_count):
                candidate_tasks.unsafe_get(task_idx) = candidate_overflow[
                    overflow_offset + task_idx
                ]
            consume_candidates()
            overflow_offset += candidate_count

        comptime for lane in range(length):
            hit.t[lane] = closest_t.unsafe_get(lane)
            hit.u[lane] = result_u.unsafe_get(lane)
            hit.v[lane] = result_v.unsafe_get(lane)
            hit.prim[lane] = result_prim.unsafe_get(lane)
            hit.inst[lane] = result_inst.unsafe_get(lane)
            hit.normal.x[lane] = result_nx.unsafe_get(lane)
            hit.normal.y[lane] = result_ny.unsafe_get(lane)
            hit.normal.z[lane] = result_nz.unsafe_get(lane)

        comptime if simd_normal_transforms:
            comptime if not translations_only:
                var hit_mask = hit.is_hit()
                var local_x = SIMD[.float32, length](0.0)
                var local_y = SIMD[.float32, length](0.0)
                var local_z = SIMD[.float32, length](1.0)
                var m00 = SIMD[.float32, length](1.0)
                var m01 = SIMD[.float32, length](0.0)
                var m02 = SIMD[.float32, length](0.0)
                var m10 = SIMD[.float32, length](0.0)
                var m11 = SIMD[.float32, length](1.0)
                var m12 = SIMD[.float32, length](0.0)
                var m20 = SIMD[.float32, length](0.0)
                var m21 = SIMD[.float32, length](0.0)
                var m22 = SIMD[.float32, length](1.0)
                comptime for lane in range(length):
                    if hit_mask[lane]:
                        local_x[lane] = hit.normal.x[lane]
                        local_y[lane] = hit.normal.y[lane]
                        local_z[lane] = hit.normal.z[lane]
                        ref inverse = self.hot_instances.unsafe_get(
                            Int(hit.inst[lane])
                        ).inv_transform
                        m00[lane] = inverse.m00
                        m01[lane] = inverse.m01
                        m02[lane] = inverse.m02
                        m10[lane] = inverse.m10
                        m11[lane] = inverse.m11
                        m12[lane] = inverse.m12
                        m20[lane] = inverse.m20
                        m21[lane] = inverse.m21
                        m22[lane] = inverse.m22
                var world_normal = normalize(
                    Vec3[.float32, .WORLD, length](
                        fma(m00, local_x, fma(m10, local_y, m20 * local_z)),
                        fma(m01, local_x, fma(m11, local_y, m21 * local_z)),
                        fma(m02, local_x, fma(m12, local_y, m22 * local_z)),
                    )
                )
                hit.normal.x = hit_mask.select(world_normal.x, hit.normal.x)
                hit.normal.y = hit_mask.select(world_normal.y, hit.normal.y)
                hit.normal.z = hit_mask.select(world_normal.z, hit.normal.z)
        else:
            comptime for lane in range(length):
                if hit.is_hit()[lane]:
                    var inst_idx = hit.inst[lane]
                    var scalar_hit = Hit[.WORLD](
                        hit.u[lane],
                        hit.v[lane],
                        hit.prim[lane],
                        inst_idx,
                        Normal3f32[.WORLD](
                            hit.normal.x[lane],
                            hit.normal.y[lane],
                            hit.normal.z[lane],
                        ),
                        hit.t[lane],
                    )
                    ref hot_inst = self.hot_instances.unsafe_get(Int(inst_idx))
                    finalize_tlas_hit_normal(scalar_hit, hot_inst.inv_transform)
                    hit.normal.x[lane] = scalar_hit.normal.x
                    hit.normal.y[lane] = scalar_hit.normal.y
                    hit.normal.z[lane] = scalar_hit.normal.z
        return hit

    def trace_blases_packet[
        blas_node_width: SIMDLength,
        blas_leaf_width: SIMDLength,
        length: SIMDLength,
        simd_normal_transforms: Bool = True,
    ](
        self,
        rays: Ray[.float32, .WORLD, length],
        blases: CpuBlasSet[.TRIANGLE, blas_node_width, blas_leaf_width],
        valid: SIMD[.bool, length] = SIMD[.bool, length](fill=True),
    ) -> Hit[.WORLD, length]:
        """Dispatch a specialized translation-only instancing fast path."""
        if self.all_translation_only:
            return self._trace_blases_packet_impl[
                blas_node_width,
                blas_leaf_width,
                length,
                simd_normal_transforms,
                True,
            ](rays, blases, valid)
        return self._trace_blases_packet_impl[
            blas_node_width,
            blas_leaf_width,
            length,
            simd_normal_transforms,
            False,
        ](rays, blases, valid)

    def trace_blases_packet_any_hit[
        blas_node_width: SIMDLength,
        blas_leaf_width: SIMDLength,
        length: SIMDLength,
    ](
        self,
        rays: Ray[.float32, .WORLD, length],
        blases: CpuBlasSet[.TRIANGLE, blas_node_width, blas_leaf_width],
        valid: SIMD[.bool, length] = SIMD[.bool, length](fill=True),
    ) -> SIMD[.bool, length]:
        """Traverse packet visibility rays through the TLAS once.

        BLAS continuation remains scalar per surviving lane. Ray transforms
        are computed SIMD-wide once per candidate instance.
        """
        comptime assert length > 1
        var hit = Hit[.WORLD, length].miss(rays.t_max)
        var reciprocal_direction = rays.reciprocal_direction()
        comptime any_candidate_capacity = 16
        var any_candidate_tasks = Array[UInt64, any_candidate_capacity](
            uninitialized=True
        )
        var any_candidate_count = 0
        var any_candidate_overflow = List[UInt64]()

        def leaf_fn(
            active: SIMD[.bool, length],
            leaf_block_idx: UInt32,
            mut packet_hit: Hit[.WORLD, length],
        ) {
            imm,
            mut any_candidate_tasks,
            mut any_candidate_count,
            mut any_candidate_overflow,
        }:
            ref block = self.leaf_blocks.unsafe_get(Int(leaf_block_idx))
            comptime for inst_lane in range(Self.leaf_width):
                var inst_idx = block.inst_indices[inst_lane]
                if inst_idx == EMPTY_LANE:
                    continue
                var candidate = active & packet_hit.t.ne(0.0)
                comptime if Self.leaf_width > 1:
                    var bmin = Point3[.float32, .WORLD, length](
                        block.bounds._min.x[inst_lane],
                        block.bounds._min.y[inst_lane],
                        block.bounds._min.z[inst_lane],
                    )
                    var bmax = Point3[.float32, .WORLD, length](
                        block.bounds._max.x[inst_lane],
                        block.bounds._max.y[inst_lane],
                        block.bounds._max.z[inst_lane],
                    )
                    candidate &= intersect_ray_aabb_rcp(
                        rays.o,
                        reciprocal_direction,
                        bmin,
                        bmax,
                        packet_hit.t,
                    ).mask
                var bits = UInt32(pack_bits(candidate))
                if bits != 0:
                    var task = UInt64(inst_idx) << 32 | UInt64(bits)
                    if any_candidate_count < any_candidate_capacity:
                        any_candidate_tasks.unsafe_get(
                            any_candidate_count
                        ) = task
                        any_candidate_count += 1
                    else:
                        any_candidate_overflow.append(task)

        trace_packet_stack_bounds_bvh[
            frame=.WORLD,
            bounds_width=Self.bounds_width,
            length=length,
            any_hit=True,
            # Visibility candidates are likewise deferred until traversal
            # returns, leaving priority ordering unable to terminate lanes.
            ordered_tasks=False,
        ](
            self.tree.nodes,
            rays,
            reciprocal_direction,
            valid,
            hit,
            leaf_fn,
            lambda (
                _active: SIMD[.bool, length],
                _child_ref: UInt32,
                mut _packet_hit: Hit[.WORLD, length],
            ): None,
            lambda (_child_ref: UInt32): None,
        )

        if any_candidate_count == 0:
            return valid & hit.t.eq(0.0)

        var any_scalar_rays = Array[Float32, 10 * length](uninitialized=True)
        comptime for lane in range(length):
            any_scalar_rays.unsafe_get(0 * Int(length) + lane) = rays.o.x[lane]
            any_scalar_rays.unsafe_get(1 * Int(length) + lane) = rays.o.y[lane]
            any_scalar_rays.unsafe_get(2 * Int(length) + lane) = rays.o.z[lane]
            any_scalar_rays.unsafe_get(3 * Int(length) + lane) = rays.d.x[lane]
            any_scalar_rays.unsafe_get(4 * Int(length) + lane) = rays.d.y[lane]
            any_scalar_rays.unsafe_get(5 * Int(length) + lane) = rays.d.z[lane]
            any_scalar_rays.unsafe_get(6 * Int(length) + lane) = rays.t_min[
                lane
            ]
            any_scalar_rays.unsafe_get(
                7 * Int(length) + lane
            ) = reciprocal_direction.x[lane]
            any_scalar_rays.unsafe_get(
                8 * Int(length) + lane
            ) = reciprocal_direction.y[lane]
            any_scalar_rays.unsafe_get(
                9 * Int(length) + lane
            ) = reciprocal_direction.z[lane]
        var any_scalar_rays_ptr = any_scalar_rays.unsafe_ptr()

        @no_inline
        def consume_any_candidates() {
            self,
            blases,
            rays,
            any_scalar_rays_ptr,
            any_candidate_tasks,
            any_candidate_count,
            mut hit,
        }:
            for task_idx in range(any_candidate_count):
                var task = any_candidate_tasks.unsafe_get(task_idx)
                var inst_idx = UInt32(task >> 32)
                var bits = UInt32(task) & UInt32(pack_bits(hit.t.ne(0.0)))
                if bits == 0:
                    continue
                ref hot_inst = self.hot_instances.unsafe_get(Int(inst_idx))
                _debug_check_blas_index(hot_inst.blas_idx, blases.blas_count)
                var desc = BlasDesc.load(
                    blases.descs.unsafe_ptr(), hot_inst.blas_idx
                )
                if desc.prim_count == 0:
                    continue
                if hot_inst.translation_only:
                    while bits != 0:
                        var lane = Int(count_trailing_zeros(bits))
                        bits &= bits - 1
                        var local_ray = Rayf32[.LOCAL](
                            Point3[.float32, .LOCAL](
                                any_scalar_rays_ptr[
                                    unsafe_offset=0 * Int(length) + lane
                                ]
                                + hot_inst.inv_transform.tx,
                                any_scalar_rays_ptr[
                                    unsafe_offset=1 * Int(length) + lane
                                ]
                                + hot_inst.inv_transform.ty,
                                any_scalar_rays_ptr[
                                    unsafe_offset=2 * Int(length) + lane
                                ]
                                + hot_inst.inv_transform.tz,
                            ),
                            Vec3[.float32, .LOCAL](
                                any_scalar_rays_ptr[
                                    unsafe_offset=3 * Int(length) + lane
                                ],
                                any_scalar_rays_ptr[
                                    unsafe_offset=4 * Int(length) + lane
                                ],
                                any_scalar_rays_ptr[
                                    unsafe_offset=5 * Int(length) + lane
                                ],
                            ),
                            any_scalar_rays_ptr[
                                unsafe_offset=6 * Int(length) + lane
                            ],
                            hit.t[lane],
                        )
                        var local_rcp = Vec3[.float32, .LOCAL, blas_node_width](
                            any_scalar_rays_ptr[
                                unsafe_offset=7 * Int(length) + lane
                            ],
                            any_scalar_rays_ptr[
                                unsafe_offset=8 * Int(length) + lane
                            ],
                            any_scalar_rays_ptr[
                                unsafe_offset=9 * Int(length) + lane
                            ],
                        )
                        var local_hit = _trace_blas_set_precomputed_rcp[
                            blas_node_width,
                            blas_leaf_width,
                            .ANY_HIT,
                            .LOCAL,
                        ](
                            blases,
                            hot_inst.blas_idx,
                            local_ray,
                            local_rcp,
                        )
                        if local_hit.is_occluded():
                            hit.t[lane] = 0.0
                    continue

                var local_rays = _transform_world_ray_packet[length](
                    hot_inst.inv_transform, rays, hit.t
                )
                while bits != 0:
                    var lane = Int(count_trailing_zeros(bits))
                    bits &= bits - 1
                    var local_ray = Rayf32[.LOCAL](
                        Point3[.float32, .LOCAL](
                            local_rays.o.x[lane],
                            local_rays.o.y[lane],
                            local_rays.o.z[lane],
                        ),
                        Vec3[.float32, .LOCAL](
                            local_rays.d.x[lane],
                            local_rays.d.y[lane],
                            local_rays.d.z[lane],
                        ),
                        rays.t_min[lane],
                        hit.t[lane],
                    )
                    var local_hit = trace_blas_set[
                        blas_node_width,
                        blas_leaf_width,
                        .ANY_HIT,
                        .LOCAL,
                    ](blases, hot_inst.blas_idx, local_ray)
                    if local_hit.is_occluded():
                        hit.t[lane] = 0.0

        consume_any_candidates()
        var any_overflow_offset = 0
        while any_overflow_offset < len(any_candidate_overflow):
            any_candidate_count = min(
                any_candidate_capacity,
                len(any_candidate_overflow) - any_overflow_offset,
            )
            for task_idx in range(any_candidate_count):
                any_candidate_tasks.unsafe_get(
                    task_idx
                ) = any_candidate_overflow[any_overflow_offset + task_idx]
            consume_any_candidates()
            any_overflow_offset += any_candidate_count
        return valid & hit.t.eq(0.0)

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
