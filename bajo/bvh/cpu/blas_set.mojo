from max.algorithm import parallelize
from std.bit import count_trailing_zeros
from std.memory import pack_bits, unsafe_memcpy
from std.sys.intrinsics import prefetch

from bajo.bvh.constants import (
    EMPTY_LANE,
    SPHERE_LEAF_PACKED_STRIDE,
    TraceMode,
    CPU_TRI_LEAF_PACKED_STRIDE,
    WideNode,
)
from bajo.bvh.cpu.sphere_bvh import (
    _SphereBuild,
    _occlude_sphere_packet_primitive,
    _trace_sphere_leaf_block,
    _trace_sphere_packet_primitive,
)
from bajo.bvh.cpu.blas_storage import CpuBlasSet
from bajo.bvh.cpu.build_method import CpuBvhBuildMethod
from bajo.bvh.cpu.traversal_mode import CpuTraversalMode
from bajo.bvh.cpu.triangle_bvh import (
    PARALLEL_TRIANGLE_BUILD_MIN_ITEMS,
    TrianglePacketConfig,
    _TriangleBuild,
    _PacketKernelTuning,
    _occlude_triangle_packet_primitive,
    _trace_triangle_leaf_block,
    _trace_triangle_packet_policy,
    _trace_triangle_packet_primitive,
)
from bajo.bvh.cpu.bounds_bvh import WideBvhNode
from bajo.bvh.cpu.trace import (
    _extract_f32_lane,
    _extract_u32_lane,
    trace_bounds_bvh_from_ref,
    trace_packed_bounds_bvh,
    trace_packed_bounds_bvh_rcp,
    trace_packed_sphere_bounds_bvh,
)
from bajo.bvh.cpu.packet import trace_packet_stack_bounds_bvh
from bajo.bvh.cpu.parallel import _worker_count
from bajo.bvh.tagged_ref import decode_ref_index, is_leaf_ref
from bajo.bvh.types import (
    BlasDesc,
    BlasDescLayout,
    Sphere,
    SphereLeafBlock,
    TriangleLeafBlock,
    Hit,
)
from bajo.bvh.wide_meta import _pack_wide_meta, _wide_node_base
from bajo.core import (
    Frame,
    Normal3f32,
    Point3,
    Point3f32,
    Ray,
    Rayf32,
    Vec3,
    Vec3f32,
    dot,
    normalize,
)


from bajo.bvh.cpu.blas_build import (
    build_cpu_sphere_blas_set,
    build_cpu_triangle_blas_set,
)


trait AdaptiveStreamHitSink:
    """Compile-time hit consumer for adaptive stream traversal."""

    def write[
        length: SIMDLength,
        frame: Frame,
    ](mut self, base: Int, hit: Hit[frame, length]):
        ...


@always_inline
def _debug_check_blas_index(blas_idx: UInt32, blas_count: Int):
    debug_assert["safe", _use_compiler_assume=True](
        UInt64(blas_idx) < UInt64(blas_count), "CPU BLAS index is out of range"
    )


@always_inline
def _load_packed_triangle_leaf[
    frame: Frame,
    leaf_width: SIMDLength,
    load_primitive_indices: Bool = True,
](leaves: ImmPointer[Float32, _], leaf_block_idx: UInt32) -> TriangleLeafBlock[
    frame, leaf_width
]:
    var block_base = (
        Int(leaf_block_idx) * CPU_TRI_LEAF_PACKED_STRIDE * leaf_width
    )
    var block_ptr = leaves.unsafe_offset(block_base)
    var block = TriangleLeafBlock[frame, leaf_width]()
    block.v0.x = block_ptr.unsafe_load[width=leaf_width](0 * leaf_width)
    block.v0.y = block_ptr.unsafe_load[width=leaf_width](1 * leaf_width)
    block.v0.z = block_ptr.unsafe_load[width=leaf_width](2 * leaf_width)
    comptime if load_primitive_indices:
        block.prim_indices = block_ptr.unsafe_bitcast[UInt32]().unsafe_load[
            width=leaf_width
        ](3 * leaf_width)
    else:
        comptime assert leaf_width == 16
    block.e1.x = block_ptr.unsafe_load[width=leaf_width](4 * leaf_width)
    block.e1.y = block_ptr.unsafe_load[width=leaf_width](5 * leaf_width)
    block.e1.z = block_ptr.unsafe_load[width=leaf_width](6 * leaf_width)
    block.e2.x = block_ptr.unsafe_load[width=leaf_width](7 * leaf_width)
    block.e2.y = block_ptr.unsafe_load[width=leaf_width](8 * leaf_width)
    block.e2.z = block_ptr.unsafe_load[width=leaf_width](9 * leaf_width)
    return block^


@always_inline
def _trace_packed_triangle_from_ref[
    frame: Frame,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
](
    nodes: ImmSpan[WideBvhNode[frame, node_width], _],
    leaves: ImmPointer[Float32, _],
    ray: Rayf32[frame],
    initial_ref: UInt32,
    initial_hit: Hit[frame],
) -> Hit[frame]:
    """Continue the proven scalar CPU traversal over one packed subtree."""

    @always_inline
    def leaf_fn(
        ray: Rayf32[frame],
        O: Point3[.float32, frame, leaf_width],
        D: Vec3[.float32, frame, leaf_width],
        _ray_a: SIMD[.float32, leaf_width],
        _ray_inv_a: SIMD[.float32, leaf_width],
        leaf_block_idx: UInt32,
        mut hit: Hit[frame],
    ) {imm} -> Bool:
        var block_base = (
            Int(leaf_block_idx) * CPU_TRI_LEAF_PACKED_STRIDE * leaf_width
        )
        var block_ptr = leaves.unsafe_offset(block_base)
        var block = _load_packed_triangle_leaf[frame, leaf_width](
            leaves, leaf_block_idx
        )
        return _trace_triangle_leaf_block[
            frame,
            leaf_width,
            .CLOSEST_HIT,
            packed_layout=True,
        ](ray, O, D, block, block_ptr, hit)

    var hit = trace_bounds_bvh_from_ref[
        frame=frame,
        bounds_width=node_width,
        leaf_width=leaf_width,
        single_child_fast_path=True,
        terminal_mask_fast_path=True,
        packed_meta=True,
    ](nodes, ray, initial_ref, initial_hit, leaf_fn)
    if hit.is_hit() and (
        hit.prim[0] != initial_hit.prim[0] or hit.t[0] != initial_hit.t[0]
    ):
        var geometric_normal = Vec3f32[frame](
            hit.normal.x, hit.normal.y, hit.normal.z
        )
        var unit_normal = normalize(geometric_normal)
        hit.normal = Normal3f32[frame](
            unit_normal.x, unit_normal.y, unit_normal.z
        )
    return hit


def _trace_packed_triangle_any_from_ref[
    frame: Frame,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
](
    nodes: ImmSpan[WideBvhNode[frame, node_width], _],
    leaves: ImmPointer[Float32, _],
    ray: Rayf32[frame],
    initial_ref: UInt32,
) -> Bool:
    """Continue scalar any-hit traversal over one packed internal subtree."""

    @always_inline
    def leaf_fn(
        ray: Rayf32[frame],
        O: Point3[.float32, frame, leaf_width],
        D: Vec3[.float32, frame, leaf_width],
        _ray_a: SIMD[.float32, leaf_width],
        _ray_inv_a: SIMD[.float32, leaf_width],
        leaf_block_idx: UInt32,
        mut hit: Hit[frame],
    ) {imm} -> Bool:
        var block_base = (
            Int(leaf_block_idx) * CPU_TRI_LEAF_PACKED_STRIDE * leaf_width
        )
        var block_ptr = leaves.unsafe_offset(block_base)
        var block = _load_packed_triangle_leaf[
            frame,
            leaf_width,
            leaf_width != 16,
        ](leaves, leaf_block_idx)
        return _trace_triangle_leaf_block[
            frame,
            leaf_width,
            .ANY_HIT,
            packed_layout=True,
        ](ray, O, D, block, block_ptr, hit)

    return trace_bounds_bvh_from_ref[
        frame=frame,
        bounds_width=node_width,
        leaf_width=leaf_width,
        packed_meta=True,
        mode=.ANY_HIT,
        reverse_any_order=True,
    ](
        nodes,
        ray,
        initial_ref,
        Hit[frame].miss(ray.t_max),
        leaf_fn,
    ).is_occluded()


def trace_blas_set[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
    mode: TraceMode = .CLOSEST_HIT,
    frame: Frame = .LOCAL,
](
    blases: CpuBlasSet[.TRIANGLE, node_width, leaf_width],
    blas_idx: UInt32,
    ray: Rayf32[frame],
) -> Hit[frame]:
    _debug_check_blas_index(blas_idx, blases.blas_count)
    var desc = BlasDesc.load(blases.descs.unsafe_ptr(), blas_idx)
    if desc.prim_count == 0:
        return Hit[frame].miss(ray.t_max)
    var nodes_ptr = (
        blases.nodes.unsafe_ptr()
        .unsafe_offset(Int(desc.node_f32_base))
        .unsafe_bitcast[WideBvhNode[frame, node_width]]()
    )
    var nodes = Span(unsafe_ptr=nodes_ptr, length=Int(desc.node_count))
    var leaves = blases.leaves.unsafe_ptr().unsafe_offset(
        Int(desc.leaf_f32_base)
    )
    comptime if mode == .CLOSEST_HIT:
        return _trace_packed_triangle_from_ref[frame, node_width, leaf_width](
            nodes,
            leaves,
            ray,
            UInt32(0),
            Hit[frame].miss(ray.t_max),
        )

    @always_inline
    def leaf_fn(
        ray: Rayf32[frame],
        O: Point3[.float32, frame, leaf_width],
        D: Vec3[.float32, frame, leaf_width],
        _ray_a: SIMD[.float32, leaf_width],
        _ray_inv_a: SIMD[.float32, leaf_width],
        leaf_block_idx: UInt32,
        mut hit: Hit[frame],
    ) {imm} -> Bool:
        var block_base = (
            Int(leaf_block_idx) * CPU_TRI_LEAF_PACKED_STRIDE * leaf_width
        )
        var block_ptr = leaves.unsafe_offset(block_base)
        var block = _load_packed_triangle_leaf[
            frame,
            leaf_width,
            mode != .ANY_HIT or leaf_width != 16,
        ](leaves, leaf_block_idx)
        return _trace_triangle_leaf_block[
            frame,
            leaf_width,
            mode,
            packed_layout=True,
        ](ray, O, D, block, block_ptr, hit)

    return trace_packed_bounds_bvh[
        frame,
        node_width,
        leaf_width,
        mode,
    ](nodes, ray, leaf_fn)


@always_inline
def _trace_blas_desc_precomputed_rcp[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    mode: TraceMode,
    frame: Frame,
](
    blases: CpuBlasSet[.TRIANGLE, node_width, leaf_width],
    desc: BlasDesc,
    ray: Rayf32[frame],
    reciprocal_direction: Vec3[.float32, frame, node_width],
) -> Hit[frame]:
    """Trace a resolved nonempty BLAS without recomputing reciprocals."""

    var nodes_ptr = (
        blases.nodes.unsafe_ptr()
        .unsafe_offset(Int(desc.node_f32_base))
        .unsafe_bitcast[WideBvhNode[frame, node_width]]()
    )
    var nodes = Span(unsafe_ptr=nodes_ptr, length=Int(desc.node_count))
    var leaves = blases.leaves.unsafe_ptr().unsafe_offset(
        Int(desc.leaf_f32_base)
    )

    @always_inline
    def leaf_fn(
        ray: Rayf32[frame],
        O: Point3[.float32, frame, leaf_width],
        D: Vec3[.float32, frame, leaf_width],
        _ray_a: SIMD[.float32, leaf_width],
        _ray_inv_a: SIMD[.float32, leaf_width],
        leaf_block_idx: UInt32,
        mut hit: Hit[frame],
    ) {imm} -> Bool:
        var block_base = (
            Int(leaf_block_idx) * CPU_TRI_LEAF_PACKED_STRIDE * leaf_width
        )
        var block_ptr = leaves.unsafe_offset(block_base)
        var block = _load_packed_triangle_leaf[
            frame,
            leaf_width,
            mode != .ANY_HIT or leaf_width != 16,
        ](leaves, leaf_block_idx)
        return _trace_triangle_leaf_block[
            frame,
            leaf_width,
            mode,
            packed_layout=True,
        ](ray, O, D, block, block_ptr, hit)

    var hit = trace_packed_bounds_bvh_rcp[
        frame=frame,
        bounds_width=node_width,
        leaf_width=leaf_width,
        mode=mode,
        single_child_fast_path=mode == .CLOSEST_HIT,
        terminal_mask_fast_path=mode == .CLOSEST_HIT,
    ](
        nodes,
        ray,
        reciprocal_direction,
        leaf_fn,
    )
    comptime if mode == .CLOSEST_HIT:
        if hit.is_hit():
            var geometric_normal = Vec3f32[frame](
                hit.normal.x, hit.normal.y, hit.normal.z
            )
            var unit_normal = normalize(geometric_normal)
            hit.normal = Normal3f32[frame](
                unit_normal.x, unit_normal.y, unit_normal.z
            )
    return hit


@always_inline
def _trace_blas_set_precomputed_rcp[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    mode: TraceMode,
    frame: Frame,
](
    blases: CpuBlasSet[.TRIANGLE, node_width, leaf_width],
    blas_idx: UInt32,
    ray: Rayf32[frame],
    reciprocal_direction: Vec3[.float32, frame, node_width],
) -> Hit[frame]:
    """Resolve one BLAS and trace it without recomputing reciprocals.

    The caller validates ``blas_idx`` and the descriptor once before a batch
    of ray continuations.
    """

    var desc = BlasDesc.load(blases.descs.unsafe_ptr(), blas_idx)
    return _trace_blas_desc_precomputed_rcp[
        node_width, leaf_width, mode, frame
    ](blases, desc, ray, reciprocal_direction)


def _trace_blas_set_packet_policy[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    length: SIMDLength,
    common_octant_fma: Bool = False,
    frame: Frame = .LOCAL,
    config: TrianglePacketConfig = .PRODUCTION,
    mode: TraceMode = .CLOSEST_HIT,
](
    blases: CpuBlasSet[.TRIANGLE, node_width, leaf_width],
    blas_idx: UInt32,
    rays: Ray[.float32, frame, length],
    valid: SIMD[.bool, length] = SIMD[.bool, length](fill=True),
) -> Hit[frame, length]:
    """Trace packed storage through the production CPU packet algorithm."""
    comptime assert length > 1
    _debug_check_blas_index(blas_idx, blases.blas_count)
    var desc = BlasDesc.load(blases.descs.unsafe_ptr(), blas_idx)
    if desc.prim_count == 0:
        return Hit[frame, length].miss(rays.t_max)
    var nodes_ptr = (
        blases.nodes.unsafe_ptr()
        .unsafe_offset(Int(desc.node_f32_base))
        .unsafe_bitcast[WideBvhNode[frame, node_width]]()
    )
    var nodes = Span(unsafe_ptr=nodes_ptr, length=Int(desc.node_count))
    var leaves = blases.leaves.unsafe_ptr().unsafe_offset(
        Int(desc.leaf_f32_base)
    )

    @always_inline
    def leaf_fn(
        active: SIMD[.bool, length],
        leaf_block_idx: UInt32,
        mut packet_hit: Hit[frame, length],
    ) {imm}:
        var block_ptr = leaves.unsafe_offset(
            Int(leaf_block_idx) * CPU_TRI_LEAF_PACKED_STRIDE * leaf_width
        )
        var block_u32 = block_ptr.unsafe_bitcast[UInt32]()
        comptime if mode == .ANY_HIT and leaf_width == 16:
            var prim_lane = 0
            while prim_lane < Int(leaf_width):
                var live = active & packet_hit.t.ne(0.0)
                if not live.reduce_or():
                    break
                var prim_idx = block_u32[
                    unsafe_offset=3 * leaf_width + prim_lane
                ]
                if prim_idx == EMPTY_LANE:
                    break
                var v0 = Point3f32[frame](
                    block_ptr[unsafe_offset=0 * leaf_width + prim_lane],
                    block_ptr[unsafe_offset=1 * leaf_width + prim_lane],
                    block_ptr[unsafe_offset=2 * leaf_width + prim_lane],
                )
                var e1 = Vec3f32[frame](
                    block_ptr[unsafe_offset=4 * leaf_width + prim_lane],
                    block_ptr[unsafe_offset=5 * leaf_width + prim_lane],
                    block_ptr[unsafe_offset=6 * leaf_width + prim_lane],
                )
                var e2 = Vec3f32[frame](
                    block_ptr[unsafe_offset=7 * leaf_width + prim_lane],
                    block_ptr[unsafe_offset=8 * leaf_width + prim_lane],
                    block_ptr[unsafe_offset=9 * leaf_width + prim_lane],
                )
                _occlude_triangle_packet_primitive[frame, length](
                    rays, live, v0, e1, e2, packet_hit
                )
                prim_lane += 1
        else:
            comptime for prim_lane in range(leaf_width):
                var prim_idx = block_u32[
                    unsafe_offset=3 * leaf_width + prim_lane
                ]
                if prim_idx != EMPTY_LANE:
                    var v0 = Point3f32[frame](
                        block_ptr[unsafe_offset=0 * leaf_width + prim_lane],
                        block_ptr[unsafe_offset=1 * leaf_width + prim_lane],
                        block_ptr[unsafe_offset=2 * leaf_width + prim_lane],
                    )
                    var e1 = Vec3f32[frame](
                        block_ptr[unsafe_offset=4 * leaf_width + prim_lane],
                        block_ptr[unsafe_offset=5 * leaf_width + prim_lane],
                        block_ptr[unsafe_offset=6 * leaf_width + prim_lane],
                    )
                    var e2 = Vec3f32[frame](
                        block_ptr[unsafe_offset=7 * leaf_width + prim_lane],
                        block_ptr[unsafe_offset=8 * leaf_width + prim_lane],
                        block_ptr[unsafe_offset=9 * leaf_width + prim_lane],
                    )
                    comptime if mode == .ANY_HIT:
                        var live = active & packet_hit.t.ne(0.0)
                        if live.reduce_or():
                            _occlude_triangle_packet_primitive[frame, length](
                                rays, live, v0, e1, e2, packet_hit
                            )
                    else:
                        _trace_triangle_packet_primitive[frame, length](
                            rays, active, prim_idx, v0, e1, e2, packet_hit
                        )

    @always_inline
    def trace_lane(
        lane: Int,
        child_ref: UInt32,
        mut packet_hit: Hit[frame, length],
    ) {imm}:
        var ray = Rayf32[frame](
            Point3f32[frame](
                _extract_f32_lane(rays.o.x, lane),
                _extract_f32_lane(rays.o.y, lane),
                _extract_f32_lane(rays.o.z, lane),
            ),
            Vec3f32[frame](
                _extract_f32_lane(rays.d.x, lane),
                _extract_f32_lane(rays.d.y, lane),
                _extract_f32_lane(rays.d.z, lane),
            ),
            _extract_f32_lane(rays.t_min, lane),
            _extract_f32_lane(rays.t_max, lane),
        )
        comptime if mode == .ANY_HIT:
            if _trace_packed_triangle_any_from_ref[
                frame, node_width, leaf_width
            ](nodes, leaves, ray, child_ref):
                packet_hit.t[lane] = 0.0
            return
        var initial_hit = Hit[frame](
            _extract_f32_lane(packet_hit.u, lane),
            _extract_f32_lane(packet_hit.v, lane),
            _extract_u32_lane(packet_hit.prim, lane),
            _extract_u32_lane(packet_hit.inst, lane),
            Normal3f32[frame](
                _extract_f32_lane(packet_hit.normal.x, lane),
                _extract_f32_lane(packet_hit.normal.y, lane),
                _extract_f32_lane(packet_hit.normal.z, lane),
            ),
            _extract_f32_lane(packet_hit.t, lane),
        )
        var scalar_hit = _trace_packed_triangle_from_ref[
            frame, node_width, leaf_width
        ](nodes, leaves, ray, child_ref, initial_hit)
        packet_hit.u[lane] = scalar_hit.u[0]
        packet_hit.v[lane] = scalar_hit.v[0]
        packet_hit.prim[lane] = scalar_hit.prim[0]
        packet_hit.inst[lane] = scalar_hit.inst[0]
        packet_hit.normal.x[lane] = scalar_hit.normal.x[0]
        packet_hit.normal.y[lane] = scalar_hit.normal.y[0]
        packet_hit.normal.z[lane] = scalar_hit.normal.z[0]
        packet_hit.t[lane] = scalar_hit.t[0]

    def hybrid_fn(
        active: SIMD[.bool, length],
        child_ref: UInt32,
        mut packet_hit: Hit[frame, length],
    ) {imm}:
        comptime if _PacketKernelTuning[length].unroll_root_hybrid:
            if child_ref == 0:
                comptime for lane in range(length):
                    if active[lane]:
                        trace_lane(lane, child_ref, packet_hit)
                return
        var bits = UInt32(pack_bits(active))
        while bits != 0:
            var lane = Int(count_trailing_zeros(bits))
            bits &= bits - 1
            trace_lane(lane, child_ref, packet_hit)

    @always_inline
    def prefetch_fn(child_ref: UInt32) {imm}:
        if is_leaf_ref(child_ref):
            var leaf_ptr = leaves.unsafe_offset(
                Int(decode_ref_index(child_ref))
                * CPU_TRI_LEAF_PACKED_STRIDE
                * leaf_width
            )
            prefetch(leaf_ptr.unsafe_bitcast[UInt8]())
        else:
            var node_ptr = nodes.unsafe_ptr().unsafe_offset(Int(child_ref))
            prefetch(node_ptr.unsafe_bitcast[UInt8]())

    return _trace_triangle_packet_policy[
        frame,
        node_width,
        leaf_width,
        length,
        common_octant_fma,
        True,
        config.use_production_tuning,
        config.hybrid_threshold,
        config.root_scalar_max_tasks,
        config.hybrid_internals,
        config.hybrid_leaves,
        config.coherent_optimizations,
        config.hybrid_min_stack_tasks,
        mode,
    ](nodes, rays, valid, leaf_fn, hybrid_fn, prefetch_fn)


def trace_blas_set_packet[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    length: SIMDLength,
    common_octant_fma: Bool = False,
    frame: Frame = .LOCAL,
](
    blases: CpuBlasSet[.TRIANGLE, node_width, leaf_width],
    blas_idx: UInt32,
    rays: Ray[.float32, frame, length],
    valid: SIMD[.bool, length] = SIMD[.bool, length](fill=True),
) -> Hit[frame, length]:
    """Trace packed storage through the production CPU packet algorithm."""
    return _trace_blas_set_packet_policy[
        node_width,
        leaf_width,
        length,
        common_octant_fma,
        frame,
    ](blases, blas_idx, rays, valid)


def trace_blas_set_packet_any_hit[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    length: SIMDLength,
    common_octant_fma: Bool = False,
    frame: Frame = .LOCAL,
](
    blases: CpuBlasSet[.TRIANGLE, node_width, leaf_width],
    blas_idx: UInt32,
    rays: Ray[.float32, frame, length],
    valid: SIMD[.bool, length] = SIMD[.bool, length](fill=True),
) -> SIMD[.bool, length]:
    """Trace bounded triangle visibility rays without materializing hits."""
    var hit: Hit[frame, length]
    comptime if common_octant_fma:
        # The closest-hit coherent frustum policy can accumulate excessive
        # pending work after any-hit lanes terminate. Use the dedicated
        # visibility policy: octant-specialized slabs plus scalar continuation
        # for sparse internal subtrees.
        hit = _trace_blas_set_packet_policy[
            node_width,
            leaf_width,
            length,
            common_octant_fma,
            frame,
            TrianglePacketConfig.ANY_HIT_COHERENT,
            .ANY_HIT,
        ](blases, blas_idx, rays, valid)
    else:
        hit = _trace_blas_set_packet_policy[
            node_width,
            leaf_width,
            length,
            common_octant_fma,
            frame,
            TrianglePacketConfig.PRODUCTION,
            .ANY_HIT,
        ](blases, blas_idx, rays, valid)
    return valid & hit.t.eq(0.0)


def trace_blas_set[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
    mode: TraceMode = .CLOSEST_HIT,
    frame: Frame = .LOCAL,
](
    blases: CpuBlasSet[.SPHERE, node_width, leaf_width],
    blas_idx: UInt32,
    ray: Rayf32[frame],
) -> Hit[frame]:
    _debug_check_blas_index(blas_idx, blases.blas_count)
    var desc = BlasDesc.load(blases.descs.unsafe_ptr(), blas_idx)
    if desc.prim_count == 0:
        return Hit[frame].miss(ray.t_max)
    var nodes_ptr = (
        blases.nodes.unsafe_ptr()
        .unsafe_offset(Int(desc.node_f32_base))
        .unsafe_bitcast[WideBvhNode[frame, node_width]]()
    )
    var nodes = Span(unsafe_ptr=nodes_ptr, length=Int(desc.node_count))
    var leaves = blases.leaves.unsafe_ptr().unsafe_offset(
        Int(desc.leaf_f32_base)
    )

    @always_inline
    def leaf_fn(
        ray: Rayf32[frame],
        O: Point3[.float32, frame, leaf_width],
        D: Vec3[.float32, frame, leaf_width],
        ray_a: SIMD[.float32, leaf_width],
        ray_inv_a: SIMD[.float32, leaf_width],
        leaf_block_idx: UInt32,
        mut hit: Hit[frame],
    ) {imm} -> Bool:
        var block_base = (
            Int(leaf_block_idx) * SPHERE_LEAF_PACKED_STRIDE * leaf_width
        )
        var block_ptr = leaves.unsafe_offset(block_base)
        var block = SphereLeafBlock[frame, leaf_width]()
        block.center.x = block_ptr.unsafe_load[width=leaf_width](0 * leaf_width)
        block.center.y = block_ptr.unsafe_load[width=leaf_width](1 * leaf_width)
        block.center.z = block_ptr.unsafe_load[width=leaf_width](2 * leaf_width)
        block.radius = block_ptr.unsafe_load[width=leaf_width](3 * leaf_width)
        block.prim_indices = block_ptr.unsafe_bitcast[UInt32]().unsafe_load[
            width=leaf_width
        ](4 * leaf_width)
        return _trace_sphere_leaf_block[frame, leaf_width, mode](
            ray, O, D, ray_a, ray_inv_a, block, hit
        )

    return trace_packed_sphere_bounds_bvh[
        frame,
        node_width,
        leaf_width,
        mode,
    ](nodes, ray, leaf_fn)


def _trace_sphere_blas_set_packet_policy[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    length: SIMDLength,
    mode: TraceMode,
    frame: Frame = .LOCAL,
](
    blases: CpuBlasSet[.SPHERE, node_width, leaf_width],
    blas_idx: UInt32,
    rays: Ray[.float32, frame, length],
    valid: SIMD[.bool, length] = SIMD[.bool, length](fill=True),
) -> Hit[frame, length]:
    """Trace packed spheres with a compile-time closest/any-hit policy."""
    comptime assert length > 1
    _debug_check_blas_index(blas_idx, blases.blas_count)
    var desc = BlasDesc.load(blases.descs.unsafe_ptr(), blas_idx)
    if desc.prim_count == 0:
        return Hit[frame, length].miss(rays.t_max)
    var nodes_ptr = (
        blases.nodes.unsafe_ptr()
        .unsafe_offset(Int(desc.node_f32_base))
        .unsafe_bitcast[WideBvhNode[frame, node_width]]()
    )
    var nodes = Span(unsafe_ptr=nodes_ptr, length=Int(desc.node_count))
    var leaves = blases.leaves.unsafe_ptr().unsafe_offset(
        Int(desc.leaf_f32_base)
    )
    var hit = Hit[frame, length].miss(rays.t_max)
    var ray_a = dot(rays.d, rays.d)
    var ray_inv_a = Float32(1.0) / ray_a
    var reciprocal_direction = rays.reciprocal_direction()

    def leaf_fn(
        active: SIMD[.bool, length],
        leaf_block_idx: UInt32,
        mut packet_hit: Hit[frame, length],
    ) {imm}:
        var block_ptr = leaves.unsafe_offset(
            Int(leaf_block_idx) * SPHERE_LEAF_PACKED_STRIDE * leaf_width
        )
        var block_u32 = block_ptr.unsafe_bitcast[UInt32]()
        comptime for prim_lane in range(leaf_width):
            var prim_idx = block_u32[unsafe_offset=4 * leaf_width + prim_lane]
            if prim_idx != EMPTY_LANE:
                var center = Point3f32[frame](
                    block_ptr[unsafe_offset=0 * leaf_width + prim_lane],
                    block_ptr[unsafe_offset=1 * leaf_width + prim_lane],
                    block_ptr[unsafe_offset=2 * leaf_width + prim_lane],
                )
                var radius = block_ptr[unsafe_offset=3 * leaf_width + prim_lane]
                comptime if mode == .ANY_HIT:
                    var live = active & packet_hit.t.ne(0.0)
                    if live.reduce_or():
                        _occlude_sphere_packet_primitive[frame, length](
                            rays,
                            live,
                            ray_a,
                            ray_inv_a,
                            center,
                            radius,
                            packet_hit,
                        )
                else:
                    _trace_sphere_packet_primitive[frame, length](
                        rays,
                        active,
                        ray_a,
                        ray_inv_a,
                        prim_idx,
                        center,
                        radius,
                        packet_hit,
                    )

    trace_packet_stack_bounds_bvh[
        frame=frame,
        bounds_width=node_width,
        length=length,
        packed_meta=True,
        any_hit=mode == .ANY_HIT,
    ](
        nodes,
        rays,
        reciprocal_direction,
        valid,
        hit,
        leaf_fn,
        lambda (
            _active: SIMD[.bool, length],
            _child_ref: UInt32,
            mut _packet_hit: Hit[frame, length],
        ): None,
        lambda (_child_ref: UInt32): None,
    )
    return hit


def trace_blas_set_packet[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    length: SIMDLength,
    frame: Frame = .LOCAL,
](
    blases: CpuBlasSet[.SPHERE, node_width, leaf_width],
    blas_idx: UInt32,
    rays: Ray[.float32, frame, length],
    valid: SIMD[.bool, length] = SIMD[.bool, length](fill=True),
) -> Hit[frame, length]:
    return _trace_sphere_blas_set_packet_policy[
        node_width, leaf_width, length, .CLOSEST_HIT, frame
    ](blases, blas_idx, rays, valid)


def trace_blas_set_packet_any_hit[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    length: SIMDLength,
    frame: Frame = .LOCAL,
](
    blases: CpuBlasSet[.SPHERE, node_width, leaf_width],
    blas_idx: UInt32,
    rays: Ray[.float32, frame, length],
    valid: SIMD[.bool, length] = SIMD[.bool, length](fill=True),
) -> SIMD[.bool, length]:
    var hit = _trace_sphere_blas_set_packet_policy[
        node_width, leaf_width, length, .ANY_HIT, frame
    ](blases, blas_idx, rays, valid)
    return valid & hit.t.eq(0.0)


@always_inline
def _packet_range_has_common_octant[
    frame: Frame,
    length: SIMDLength,
    range_length: SIMDLength,
](
    rays: Ray[.float32, frame, length],
    valid: SIMD[.bool, length],
    base: Int,
) -> Bool:
    """Return true when a complete active range shares direction signs."""
    if base + range_length > length or not valid[base]:
        return False
    var positive_x = rays.d.x[base] >= 0.0
    var positive_y = rays.d.y[base] >= 0.0
    var positive_z = rays.d.z[base] >= 0.0
    comptime for offset in range(1, range_length):
        var lane = base + offset
        if (
            not valid[lane]
            or (rays.d.x[lane] >= 0.0) != positive_x
            or (rays.d.y[lane] >= 0.0) != positive_y
            or (rays.d.z[lane] >= 0.0) != positive_z
        ):
            return False
    return True


@always_inline
def _extract_ray_range[
    frame: Frame,
    length: SIMDLength,
    range_length: SIMDLength,
](rays: Ray[.float32, frame, length], base: Int) -> Ray[
    .float32, frame, range_length
]:
    var ox = SIMD[.float32, range_length](0.0)
    var oy = SIMD[.float32, range_length](0.0)
    var oz = SIMD[.float32, range_length](0.0)
    var dx = SIMD[.float32, range_length](0.0)
    var dy = SIMD[.float32, range_length](0.0)
    var dz = SIMD[.float32, range_length](0.0)
    var t_min = SIMD[.float32, range_length](0.0)
    var t_max = SIMD[.float32, range_length](0.0)
    comptime for offset in range(range_length):
        var lane = base + offset
        ox[offset] = rays.o.x[lane]
        oy[offset] = rays.o.y[lane]
        oz[offset] = rays.o.z[lane]
        dx[offset] = rays.d.x[lane]
        dy[offset] = rays.d.y[lane]
        dz[offset] = rays.d.z[lane]
        t_min[offset] = rays.t_min[lane]
        t_max[offset] = rays.t_max[lane]
    return Ray[.float32, frame, range_length](
        Point3[.float32, frame, range_length](ox, oy, oz),
        Vec3[.float32, frame, range_length](dx, dy, dz),
        t_min,
        t_max,
    )


@always_inline
def _store_hit_range[
    frame: Frame,
    length: SIMDLength,
    range_length: SIMDLength,
](
    mut destination: Hit[frame, length],
    source: Hit[frame, range_length],
    base: Int,
):
    comptime for offset in range(range_length):
        var lane = base + offset
        destination.u[lane] = source.u[offset]
        destination.v[lane] = source.v[offset]
        destination.prim[lane] = source.prim[offset]
        destination.inst[lane] = source.inst[offset]
        destination.normal.x[lane] = source.normal.x[offset]
        destination.normal.y[lane] = source.normal.y[offset]
        destination.normal.z[lane] = source.normal.z[offset]
        destination.t[lane] = source.t[offset]


@always_inline
def _trace_first_coherent_packet_range[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    length: SIMDLength,
    index: Int,
    *packet_sizes: SIMDLength,
    frame: Frame = .LOCAL,
](
    blases: CpuBlasSet[.TRIANGLE, node_width, leaf_width],
    blas_idx: UInt32,
    rays: Ray[.float32, frame, length],
    valid: SIMD[.bool, length],
    base: Int,
    mut result: Hit[frame, length],
) -> Int:
    """Trace the first applicable configured subpacket, fully specialized."""
    comptime if index == len(packet_sizes):
        return 0
    else:
        comptime packet_size = packet_sizes[index]
        comptime if packet_size >= length:
            return _trace_first_coherent_packet_range[
                node_width,
                leaf_width,
                length,
                index + 1,
                *packet_sizes,
                frame=frame,
            ](blases, blas_idx, rays, valid, base, result)
        else:
            if _packet_range_has_common_octant[frame, length, packet_size](
                rays, valid, base
            ):
                var packet = _extract_ray_range[frame, length, packet_size](
                    rays, base
                )
                var packet_hit = trace_blas_set_packet[
                    node_width,
                    leaf_width,
                    packet_size,
                    True,
                    frame,
                ](
                    blases,
                    blas_idx,
                    packet,
                    SIMD[.bool, packet_size](fill=True),
                )
                _store_hit_range[frame, length, packet_size](
                    result, packet_hit, base
                )
                return packet_size
            return _trace_first_coherent_packet_range[
                node_width,
                leaf_width,
                length,
                index + 1,
                *packet_sizes,
                frame=frame,
            ](blases, blas_idx, rays, valid, base, result)


def trace_blas_set_packet_adaptive[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    length: SIMDLength,
    *packet_sizes: SIMDLength,
    frame: Frame = .LOCAL,
](
    blases: CpuBlasSet[.TRIANGLE, node_width, leaf_width],
    blas_idx: UInt32,
    rays: Ray[.float32, frame, length],
    valid: SIMD[.bool, length] = SIMD[.bool, length](fill=True),
) -> Hit[frame, length]:
    """Adapt within a SIMD packet using a compile-time size sequence.

    `packet_sizes` is strictly descending; complete coherent ranges use the
    first applicable packet width and remaining active lanes trace scalar.
    """
    comptime assert length > 1
    comptime assert len(packet_sizes) > 0
    comptime for index in range(len(packet_sizes)):
        comptime assert packet_sizes[index] > 1
        comptime if index > 0:
            comptime assert packet_sizes[index - 1] > packet_sizes[index]

    # Preserve the input packet when a configured width matches it exactly.
    comptime for packet_size in packet_sizes:
        comptime if packet_size == length:
            if _packet_range_has_common_octant[frame, length, packet_size](
                rays, valid, 0
            ):
                return trace_blas_set_packet[
                    node_width, leaf_width, length, True, frame
                ](blases, blas_idx, rays, valid)

    var result = Hit[frame, length].miss(rays.t_max)
    var base = 0
    while base < length:
        var consumed = _trace_first_coherent_packet_range[
            node_width,
            leaf_width,
            length,
            0,
            *packet_sizes,
            frame=frame,
        ](blases, blas_idx, rays, valid, base, result)
        if consumed != 0:
            base += consumed
            continue
        if valid[base]:
            var ray = Rayf32[frame](
                Point3f32[frame](
                    rays.o.x[base], rays.o.y[base], rays.o.z[base]
                ),
                Vec3f32[frame](rays.d.x[base], rays.d.y[base], rays.d.z[base]),
                rays.t_min[base],
                rays.t_max[base],
            )
            var hit = trace_blas_set[
                node_width, leaf_width, .CLOSEST_HIT, frame
            ](blases, blas_idx, ray)
            _store_hit_range[frame, length, 1](result, hit, base)
        base += 1
    return result


def trace_blas_set_packet_selected[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    length: SIMDLength,
    mode: CpuTraversalMode = .AUTO_COHERENT,
    frame: Frame = .LOCAL,
](
    blases: CpuBlasSet[.TRIANGLE, node_width, leaf_width],
    blas_idx: UInt32,
    rays: Ray[.float32, frame, length],
    valid: SIMD[.bool, length] = SIMD[.bool, length](fill=True),
) -> Hit[frame, length]:
    """Trace triangles with fixed or automatically coherent packet dispatch."""
    comptime assert length > 1
    comptime assert mode == .FIXED_PACKET or mode == .AUTO_COHERENT

    comptime if mode == .FIXED_PACKET:
        return trace_blas_set_packet[
            node_width, leaf_width, length, False, frame
        ](blases, blas_idx, rays, valid)
    else:
        if _packet_range_has_common_octant[frame, length, length](
            rays, valid, 0
        ):
            return trace_blas_set_packet[
                node_width, leaf_width, length, True, frame
            ](blases, blas_idx, rays, valid)
        return trace_blas_set_packet[
            node_width, leaf_width, length, False, frame
        ](blases, blas_idx, rays, valid)


@always_inline
def _stream_ray_octant[frame: Frame](ray: Rayf32[frame]) -> Int:
    return (
        Int(ray.d.x >= 0.0)
        | (Int(ray.d.y >= 0.0) << 1)
        | (Int(ray.d.z >= 0.0) << 2)
    )


@always_inline
def _stream_range_has_common_octant[
    frame: Frame,
    range_length: SIMDLength,
](rays: List[Rayf32[frame]], base: Int, octant: Int) -> Bool:
    comptime for lane in range(1, range_length):
        if _stream_ray_octant(rays.unsafe_get(base + lane)) != octant:
            return False
    return True


def trace_blas_set_adaptive_stream[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    *packet_sizes: SIMDLength,
    sink_type: AdaptiveStreamHitSink,
    frame: Frame = .LOCAL,
](
    blases: CpuBlasSet[.TRIANGLE, node_width, leaf_width],
    blas_idx: UInt32,
    rays: List[Rayf32[frame]],
    mut sink: sink_type,
):
    """Trace a continuous AoS ray stream with adaptive coherent packets.

    `packet_sizes` is a strictly descending compile-time sequence, for example
    `16, 8, 4`; scalar traversal is the implicit final fallback. The sink must
    provide
    `write[range_length, frame](base, hit)`. Keeping consumption generic lets
    renderers fuse hit processing without allocating or rereading a hit array.
    """
    comptime assert len(packet_sizes) > 0
    comptime for index in range(len(packet_sizes)):
        comptime assert packet_sizes[index] > 1
        comptime if index > 0:
            comptime assert packet_sizes[index - 1] > packet_sizes[index]
    var ray_count = len(rays)

    @always_inline
    def trace_range[
        range_length: SIMDLength,
    ](base: Int) {imm, mut sink}:
        var ox = SIMD[.float32, range_length](0.0)
        var oy = SIMD[.float32, range_length](0.0)
        var oz = SIMD[.float32, range_length](0.0)
        var dx = SIMD[.float32, range_length](0.0)
        var dy = SIMD[.float32, range_length](0.0)
        var dz = SIMD[.float32, range_length](1.0)
        var t_min = SIMD[.float32, range_length](0.0)
        var t_max = SIMD[.float32, range_length](0.0)
        comptime for lane in range(range_length):
            ref ray = rays.unsafe_get(base + lane)
            ox[lane] = ray.o.x
            oy[lane] = ray.o.y
            oz[lane] = ray.o.z
            dx[lane] = ray.d.x
            dy[lane] = ray.d.y
            dz[lane] = ray.d.z
            t_min[lane] = ray.t_min
            t_max[lane] = ray.t_max

        var packet = Ray[.float32, frame, range_length](
            Point3[.float32, frame, range_length](ox, oy, oz),
            Vec3[.float32, frame, range_length](dx, dy, dz),
            t_min,
            t_max,
        )
        var packet_hit = trace_blas_set_packet[
            node_width,
            leaf_width,
            range_length,
            True,
            frame,
        ](blases, blas_idx, packet)
        sink.write[range_length, frame](base, packet_hit)

    @always_inline
    def trace_one(base: Int) {imm, mut sink}:
        var hit = trace_blas_set[
            node_width,
            leaf_width,
            .CLOSEST_HIT,
            frame,
        ](blases, blas_idx, rays.unsafe_get(base))
        sink.write[1, frame](base, hit)

    var base = 0
    comptime largest_packet = packet_sizes[0]
    while base + largest_packet <= ray_count:
        var octant = _stream_ray_octant(rays.unsafe_get(base))
        var consumed = 0
        comptime for packet_size in packet_sizes:
            if consumed == 0 and _stream_range_has_common_octant[
                frame, packet_size
            ](rays, base, octant):
                trace_range[packet_size](base)
                consumed = packet_size
        if consumed == 0:
            trace_one(base)
            base += 1
        else:
            base += consumed
    while base < ray_count:
        var octant = _stream_ray_octant(rays.unsafe_get(base))
        var consumed = 0
        comptime for packet_size in packet_sizes:
            if (
                consumed == 0
                and base + packet_size <= ray_count
                and _stream_range_has_common_octant[frame, packet_size](
                    rays, base, octant
                )
            ):
                trace_range[packet_size](base)
                consumed = packet_size
        if consumed == 0:
            trace_one(base)
            base += 1
        else:
            base += consumed
