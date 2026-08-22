from max.algorithm import parallelize
from std.bit import count_trailing_zeros
from std.memory import pack_bits
from std.sys.intrinsics import prefetch

from bajo.bvh.constants import (
    EMPTY_LANE,
    SPHERE_LEAF_PACKED_STRIDE,
    TRACE,
    TRI_LEAF_PACKED_STRIDE,
    WideNode,
)
from bajo.bvh.cpu.sphere_bvh import (
    _SphereBuild,
    _trace_sphere_leaf_block,
    _trace_sphere_packet_primitive,
)
from bajo.bvh.cpu.triangle_bvh import (
    PARALLEL_TRIANGLE_BUILD_MIN_ITEMS,
    _TriangleBuild,
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
    trace_packed_sphere_bounds_bvh,
)
from bajo.bvh.cpu.packet import trace_packet_stack_bounds_bvh
from bajo.bvh.tagged_ref import decode_ref_index, is_leaf_ref
from bajo.bvh.types import (
    BlasDesc,
    BlasDescLayout,
    CpuBlasSet,
    Sphere,
    SphereLeafBlock,
    TriangleLeafBlock,
    Hit,
)
from bajo.bvh.wide_meta import _pack_wide_meta, _wide_node_index
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


comptime CPU_BLAS_OUTER_PARALLEL_MIN_PRIMITIVES = 4096


@always_inline
def _load_packed_triangle_leaf[
    frame: Frame, leaf_width: SIMDLength
](leaves: ImmPointer[Float32, _], leaf_block_idx: UInt32) -> TriangleLeafBlock[
    frame, leaf_width
]:
    var block_base = Int(leaf_block_idx) * TRI_LEAF_PACKED_STRIDE * leaf_width
    var block_ptr = leaves.unsafe_offset(block_base)
    var block = TriangleLeafBlock[frame, leaf_width]()
    block.v0.x = block_ptr.unsafe_load[width=leaf_width](0 * leaf_width)
    block.v0.y = block_ptr.unsafe_load[width=leaf_width](1 * leaf_width)
    block.v0.z = block_ptr.unsafe_load[width=leaf_width](2 * leaf_width)
    block.prim_indices = block_ptr.unsafe_bitcast[UInt32]().unsafe_load[
        width=leaf_width
    ](3 * leaf_width)
    block.e1.x = block_ptr.unsafe_load[width=leaf_width](4 * leaf_width)
    block.e1.y = block_ptr.unsafe_load[width=leaf_width](5 * leaf_width)
    block.e1.z = block_ptr.unsafe_load[width=leaf_width](6 * leaf_width)
    block.e2.x = block_ptr.unsafe_load[width=leaf_width](8 * leaf_width)
    block.e2.y = block_ptr.unsafe_load[width=leaf_width](9 * leaf_width)
    block.e2.z = block_ptr.unsafe_load[width=leaf_width](10 * leaf_width)
    return block^


def _triangle_leaf_count[
    frame: Frame,
    leaf_width: SIMDLength,
](block: TriangleLeafBlock[frame, leaf_width]) -> UInt32:
    var count = UInt32(0)
    comptime for lane in range(leaf_width):
        if block.prim_indices[lane] != EMPTY_LANE:
            count += 1
    return count


def _pack_triangle_blas[
    frame: Frame,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    split_method: String,
](
    vertices: ImmSpan[Point3f32[frame], _],
    descs: MutPointer[UInt32, _],
    nodes: MutPointer[Float32, _],
    leaves: MutPointer[Float32, _],
    blas_idx: Int,
    node_f32_base: Int,
    leaf_f32_base: Int,
):
    var bvh = _TriangleBuild[frame, node_width, leaf_width].__init__[
        split_method
    ](vertices)
    var nodes_u32 = nodes.unsafe_bitcast[UInt32]()
    for node_idx in range(len(bvh.tree.nodes)):
        ref node = bvh.tree.nodes[node_idx]
        comptime for lane in range(node_width):
            var local_node_idx = UInt32(node_idx)
            nodes[
                unsafe_offset=node_f32_base
                + _wide_node_index[node_width](
                    local_node_idx, WideNode.MIN_X, lane
                )
            ] = node.aabb._min.x[lane]
            nodes[
                unsafe_offset=node_f32_base
                + _wide_node_index[node_width](
                    local_node_idx, WideNode.MIN_Y, lane
                )
            ] = node.aabb._min.y[lane]
            nodes[
                unsafe_offset=node_f32_base
                + _wide_node_index[node_width](
                    local_node_idx, WideNode.MIN_Z, lane
                )
            ] = node.aabb._min.z[lane]
            nodes[
                unsafe_offset=node_f32_base
                + _wide_node_index[node_width](
                    local_node_idx, WideNode.MAX_X, lane
                )
            ] = node.aabb._max.x[lane]
            nodes[
                unsafe_offset=node_f32_base
                + _wide_node_index[node_width](
                    local_node_idx, WideNode.MAX_Y, lane
                )
            ] = node.aabb._max.y[lane]
            nodes[
                unsafe_offset=node_f32_base
                + _wide_node_index[node_width](
                    local_node_idx, WideNode.MAX_Z, lane
                )
            ] = node.aabb._max.z[lane]

            var data = node.data[lane]
            var meta = EMPTY_LANE
            if data != EMPTY_LANE:
                if is_leaf_ref(data):
                    var block_idx = decode_ref_index(data)
                    meta = _pack_wide_meta(
                        block_idx,
                        _triangle_leaf_count[frame, leaf_width](
                            bvh.leaf_blocks[Int(block_idx)]
                        ),
                    )
                else:
                    meta = _pack_wide_meta(data, UInt32(0))
            nodes_u32[
                unsafe_offset=node_f32_base
                + _wide_node_index[node_width](
                    local_node_idx, WideNode.META, lane
                )
            ] = meta

    var leaves_u32 = leaves.unsafe_bitcast[UInt32]()
    for block_idx in range(len(bvh.leaf_blocks)):
        ref block = bvh.leaf_blocks[block_idx]
        var out = (
            leaf_f32_base + block_idx * leaf_width * TRI_LEAF_PACKED_STRIDE
        )
        comptime for lane in range(leaf_width):
            leaves[unsafe_offset=out + 0 * leaf_width + lane] = block.v0.x[lane]
            leaves[unsafe_offset=out + 1 * leaf_width + lane] = block.v0.y[lane]
            leaves[unsafe_offset=out + 2 * leaf_width + lane] = block.v0.z[lane]
            leaves_u32[
                unsafe_offset=out + 3 * leaf_width + lane
            ] = block.prim_indices[lane]
            leaves[unsafe_offset=out + 4 * leaf_width + lane] = block.e1.x[lane]
            leaves[unsafe_offset=out + 5 * leaf_width + lane] = block.e1.y[lane]
            leaves[unsafe_offset=out + 6 * leaf_width + lane] = block.e1.z[lane]
            leaves[unsafe_offset=out + 7 * leaf_width + lane] = 0.0
            leaves[unsafe_offset=out + 8 * leaf_width + lane] = block.e2.x[lane]
            leaves[unsafe_offset=out + 9 * leaf_width + lane] = block.e2.y[lane]
            leaves[unsafe_offset=out + 10 * leaf_width + lane] = block.e2.z[
                lane
            ]
            leaves[unsafe_offset=out + 11 * leaf_width + lane] = 0.0

    var desc_base = BlasDescLayout.base(blas_idx)
    descs[unsafe_offset=desc_base + BlasDescLayout.NODE_F32_BASE] = UInt32(
        node_f32_base
    )
    descs[unsafe_offset=desc_base + BlasDescLayout.LEAF_F32_BASE] = UInt32(
        leaf_f32_base
    )
    descs[unsafe_offset=desc_base + BlasDescLayout.ROOT_IDX] = UInt32(0)
    descs[unsafe_offset=desc_base + BlasDescLayout.NODE_COUNT] = UInt32(
        len(bvh.tree.nodes)
    )
    descs[unsafe_offset=desc_base + BlasDescLayout.LEAF_BLOCK_COUNT] = UInt32(
        len(bvh.leaf_blocks)
    )
    descs[unsafe_offset=desc_base + BlasDescLayout.PRIM_COUNT] = UInt32(
        bvh.tri_count
    )


def build_triangle_blases[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
    split_method: String = "sah",
    frame: Frame = Frame.LOCAL,
](
    vertex_sets: ImmSpan[List[Point3f32[frame]], _],
) -> CpuBlasSet[
    node_width, leaf_width
]:
    debug_assert["safe", _use_compiler_assume=True](
        len(vertex_sets) > 0, "CPU BLAS batch must be nonempty"
    )
    var node_bases = List[Int](capacity=len(vertex_sets))
    var leaf_bases = List[Int](capacity=len(vertex_sets))
    var node_f32_count = 0
    var leaf_f32_count = 0
    var allow_across_blas_parallelism = len(vertex_sets) > 1
    var total_triangle_count = 0
    for vertices in vertex_sets:
        debug_assert["safe", _use_compiler_assume=True](
            len(vertices) > 0 and len(vertices) % 3 == 0,
            "each CPU triangle BLAS must contain complete, nonempty triangles",
        )
        var tri_count = len(vertices) / 3
        total_triangle_count += tri_count
        node_bases.append(node_f32_count)
        leaf_bases.append(leaf_f32_count)
        node_f32_count += (
            max(tri_count - 1, 1) * node_width * WideNode.CHILD_STRIDE
        )
        leaf_f32_count += tri_count * leaf_width * TRI_LEAF_PACKED_STRIDE
        if tri_count >= PARALLEL_TRIANGLE_BUILD_MIN_ITEMS:
            allow_across_blas_parallelism = False

    allow_across_blas_parallelism &= (
        total_triangle_count >= CPU_BLAS_OUTER_PARALLEL_MIN_PRIMITIVES
    )

    var descs = List[UInt32](
        length=len(vertex_sets) * BlasDescLayout.STRIDE, fill=0
    )
    var nodes = List[Float32](length=node_f32_count, fill=0.0)
    var leaves = List[Float32](length=leaf_f32_count, fill=0.0)
    var descs_ptr = descs.unsafe_ptr()
    var nodes_ptr = nodes.unsafe_ptr()
    var leaves_ptr = leaves.unsafe_ptr()

    def build_one(blas_idx: Int) {imm}:
        _pack_triangle_blas[frame, node_width, leaf_width, split_method](
            vertex_sets[blas_idx],
            descs_ptr,
            nodes_ptr,
            leaves_ptr,
            blas_idx,
            node_bases[blas_idx],
            leaf_bases[blas_idx],
        )

    if allow_across_blas_parallelism:
        parallelize(build_one, len(vertex_sets))
    else:
        for blas_idx in range(len(vertex_sets)):
            build_one(blas_idx)

    return CpuBlasSet[node_width, leaf_width](
        descs^, nodes^, leaves^, len(vertex_sets)
    )


def _sphere_leaf_count[
    frame: Frame,
    width: SIMDLength,
](block: SphereLeafBlock[frame, width]) -> UInt32:
    var count = UInt32(0)
    comptime for lane in range(width):
        if block.prim_indices[lane] != EMPTY_LANE:
            count += 1
    return count


def _pack_sphere_blas[
    frame: Frame,
    width: SIMDLength,
    split_method: String,
](
    spheres: ImmSpan[Sphere[frame], _],
    descs: MutPointer[UInt32, _],
    nodes: MutPointer[Float32, _],
    leaves: MutPointer[Float32, _],
    blas_idx: Int,
    node_f32_base: Int,
    leaf_f32_base: Int,
):
    var owned_spheres = [sphere.copy() for sphere in spheres]
    var bvh = _SphereBuild[frame, width].__init__[split_method](owned_spheres^)
    var nodes_u32 = nodes.unsafe_bitcast[UInt32]()
    for node_idx in range(len(bvh.tree.nodes)):
        ref node = bvh.tree.nodes[node_idx]
        comptime for lane in range(width):
            var local_node_idx = UInt32(node_idx)
            nodes[
                unsafe_offset=node_f32_base
                + _wide_node_index[width](local_node_idx, WideNode.MIN_X, lane)
            ] = node.aabb._min.x[lane]
            nodes[
                unsafe_offset=node_f32_base
                + _wide_node_index[width](local_node_idx, WideNode.MIN_Y, lane)
            ] = node.aabb._min.y[lane]
            nodes[
                unsafe_offset=node_f32_base
                + _wide_node_index[width](local_node_idx, WideNode.MIN_Z, lane)
            ] = node.aabb._min.z[lane]
            nodes[
                unsafe_offset=node_f32_base
                + _wide_node_index[width](local_node_idx, WideNode.MAX_X, lane)
            ] = node.aabb._max.x[lane]
            nodes[
                unsafe_offset=node_f32_base
                + _wide_node_index[width](local_node_idx, WideNode.MAX_Y, lane)
            ] = node.aabb._max.y[lane]
            nodes[
                unsafe_offset=node_f32_base
                + _wide_node_index[width](local_node_idx, WideNode.MAX_Z, lane)
            ] = node.aabb._max.z[lane]
            var data = node.data[lane]
            var meta = EMPTY_LANE
            if data != EMPTY_LANE:
                if is_leaf_ref(data):
                    var block_idx = decode_ref_index(data)
                    meta = _pack_wide_meta(
                        block_idx,
                        _sphere_leaf_count[frame, width](
                            bvh.leaf_blocks[Int(block_idx)]
                        ),
                    )
                else:
                    meta = _pack_wide_meta(data, UInt32(0))
            nodes_u32[
                unsafe_offset=node_f32_base
                + _wide_node_index[width](local_node_idx, WideNode.META, lane)
            ] = meta

    var leaves_u32 = leaves.unsafe_bitcast[UInt32]()
    for block_idx in range(len(bvh.leaf_blocks)):
        ref block = bvh.leaf_blocks[block_idx]
        var out = leaf_f32_base + block_idx * width * SPHERE_LEAF_PACKED_STRIDE
        comptime for lane in range(width):
            leaves[unsafe_offset=out + 0 * width + lane] = block.center.x[lane]
            leaves[unsafe_offset=out + 1 * width + lane] = block.center.y[lane]
            leaves[unsafe_offset=out + 2 * width + lane] = block.center.z[lane]
            leaves[unsafe_offset=out + 3 * width + lane] = block.radius[lane]
            leaves_u32[
                unsafe_offset=out + 4 * width + lane
            ] = block.prim_indices[lane]

    var desc_base = BlasDescLayout.base(blas_idx)
    descs[unsafe_offset=desc_base + BlasDescLayout.NODE_F32_BASE] = UInt32(
        node_f32_base
    )
    descs[unsafe_offset=desc_base + BlasDescLayout.LEAF_F32_BASE] = UInt32(
        leaf_f32_base
    )
    descs[unsafe_offset=desc_base + BlasDescLayout.ROOT_IDX] = UInt32(0)
    descs[unsafe_offset=desc_base + BlasDescLayout.NODE_COUNT] = UInt32(
        len(bvh.tree.nodes)
    )
    descs[unsafe_offset=desc_base + BlasDescLayout.LEAF_BLOCK_COUNT] = UInt32(
        len(bvh.leaf_blocks)
    )
    descs[unsafe_offset=desc_base + BlasDescLayout.PRIM_COUNT] = UInt32(
        bvh.sphere_count
    )


def build_sphere_blases[
    width: SIMDLength,
    split_method: String = "sah",
    frame: Frame = Frame.LOCAL,
](sphere_sets: ImmSpan[List[Sphere[frame]], _],) -> CpuBlasSet[width]:
    debug_assert["safe", _use_compiler_assume=True](
        len(sphere_sets) > 0, "CPU BLAS batch must be nonempty"
    )
    var node_bases = List[Int](capacity=len(sphere_sets))
    var leaf_bases = List[Int](capacity=len(sphere_sets))
    var node_f32_count = 0
    var leaf_f32_count = 0
    var total_sphere_count = 0
    for spheres in sphere_sets:
        debug_assert["safe", _use_compiler_assume=True](
            len(spheres) > 0, "each CPU sphere BLAS must be nonempty"
        )
        var sphere_count = len(spheres)
        total_sphere_count += sphere_count
        node_bases.append(node_f32_count)
        leaf_bases.append(leaf_f32_count)
        node_f32_count += (
            max(sphere_count - 1, 1) * width * WideNode.CHILD_STRIDE
        )
        leaf_f32_count += sphere_count * width * SPHERE_LEAF_PACKED_STRIDE

    var descs = List[UInt32](
        length=len(sphere_sets) * BlasDescLayout.STRIDE, fill=0
    )
    var nodes = List[Float32](length=node_f32_count, fill=0.0)
    var leaves = List[Float32](length=leaf_f32_count, fill=0.0)
    var descs_ptr = descs.unsafe_ptr()
    var nodes_ptr = nodes.unsafe_ptr()
    var leaves_ptr = leaves.unsafe_ptr()

    def build_one(blas_idx: Int) {imm}:
        _pack_sphere_blas[frame, width, split_method](
            sphere_sets[blas_idx],
            descs_ptr,
            nodes_ptr,
            leaves_ptr,
            blas_idx,
            node_bases[blas_idx],
            leaf_bases[blas_idx],
        )

    if (
        len(sphere_sets) > 1
        and total_sphere_count >= CPU_BLAS_OUTER_PARALLEL_MIN_PRIMITIVES
    ):
        parallelize(build_one, len(sphere_sets))
    else:
        for blas_idx in range(len(sphere_sets)):
            build_one(blas_idx)
    return CpuBlasSet[width](descs^, nodes^, leaves^, len(sphere_sets))


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
        O: Point3[DType.float32, frame, leaf_width],
        D: Vec3[DType.float32, frame, leaf_width],
        _ray_a: SIMD[DType.float32, leaf_width],
        _ray_inv_a: SIMD[DType.float32, leaf_width],
        leaf_block_idx: UInt32,
        mut hit: Hit[frame],
    ) {imm} -> Bool:
        var block_base = (
            Int(leaf_block_idx) * TRI_LEAF_PACKED_STRIDE * leaf_width
        )
        var block_ptr = leaves.unsafe_offset(block_base)
        var block = _load_packed_triangle_leaf[frame, leaf_width](
            leaves, leaf_block_idx
        )
        return _trace_triangle_leaf_block[
            frame,
            leaf_width,
            TRACE.CLOSEST_HIT,
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


def trace_triangle_blas_set[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
    mode: TRACE = TRACE.CLOSEST_HIT,
    frame: Frame = Frame.LOCAL,
](
    blases: CpuBlasSet[node_width, leaf_width],
    blas_idx: UInt32,
    ray: Rayf32[frame],
) -> Hit[frame]:
    var desc = BlasDesc.load(blases.descs.unsafe_ptr(), blas_idx)
    var nodes_ptr = (
        blases.nodes.unsafe_ptr()
        .unsafe_offset(Int(desc.node_f32_base))
        .unsafe_bitcast[WideBvhNode[frame, node_width]]()
    )
    var nodes = Span(unsafe_ptr=nodes_ptr, length=Int(desc.node_count))
    var leaves = blases.leaves.unsafe_ptr().unsafe_offset(
        Int(desc.leaf_f32_base)
    )
    comptime if mode == TRACE.CLOSEST_HIT:
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
        O: Point3[DType.float32, frame, leaf_width],
        D: Vec3[DType.float32, frame, leaf_width],
        _ray_a: SIMD[DType.float32, leaf_width],
        _ray_inv_a: SIMD[DType.float32, leaf_width],
        leaf_block_idx: UInt32,
        mut hit: Hit[frame],
    ) {imm} -> Bool:
        var block_base = (
            Int(leaf_block_idx) * TRI_LEAF_PACKED_STRIDE * leaf_width
        )
        var block_ptr = leaves.unsafe_offset(block_base)
        var block = _load_packed_triangle_leaf[frame, leaf_width](
            leaves, leaf_block_idx
        )
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


def trace_triangle_blas_set_packet[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    length: SIMDLength,
    common_octant_fma: Bool = False,
    frame: Frame = Frame.LOCAL,
](
    blases: CpuBlasSet[node_width, leaf_width],
    blas_idx: UInt32,
    rays: Ray[DType.float32, frame, length],
    valid: SIMD[DType.bool, length] = SIMD[DType.bool, length](fill=True),
) -> Hit[frame, length]:
    """Trace packed storage through the production CPU packet algorithm."""
    comptime assert length > 1
    var desc = BlasDesc.load(blases.descs.unsafe_ptr(), blas_idx)
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
        active: SIMD[DType.bool, length],
        leaf_block_idx: UInt32,
        mut packet_hit: Hit[frame, length],
    ) {imm}:
        var block_ptr = leaves.unsafe_offset(
            Int(leaf_block_idx) * TRI_LEAF_PACKED_STRIDE * leaf_width
        )
        var block_u32 = block_ptr.unsafe_bitcast[UInt32]()
        comptime for prim_lane in range(leaf_width):
            var prim_idx = block_u32[unsafe_offset=3 * leaf_width + prim_lane]
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
                    block_ptr[unsafe_offset=8 * leaf_width + prim_lane],
                    block_ptr[unsafe_offset=9 * leaf_width + prim_lane],
                    block_ptr[unsafe_offset=10 * leaf_width + prim_lane],
                )
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
        active: SIMD[DType.bool, length],
        child_ref: UInt32,
        mut packet_hit: Hit[frame, length],
    ) {imm}:
        comptime if length == 4:
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
                * TRI_LEAF_PACKED_STRIDE
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
    ](nodes, rays, valid, leaf_fn, hybrid_fn, prefetch_fn)


def trace_sphere_blas_set[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
    mode: TRACE = TRACE.CLOSEST_HIT,
    frame: Frame = Frame.LOCAL,
](
    blases: CpuBlasSet[node_width, leaf_width],
    blas_idx: UInt32,
    ray: Rayf32[frame],
) -> Hit[frame]:
    var desc = BlasDesc.load(blases.descs.unsafe_ptr(), blas_idx)
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
        O: Point3[DType.float32, frame, leaf_width],
        D: Vec3[DType.float32, frame, leaf_width],
        ray_a: SIMD[DType.float32, leaf_width],
        ray_inv_a: SIMD[DType.float32, leaf_width],
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


def trace_sphere_blas_set_packet[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    length: SIMDLength,
    frame: Frame = Frame.LOCAL,
](
    blases: CpuBlasSet[node_width, leaf_width],
    blas_idx: UInt32,
    rays: Ray[DType.float32, frame, length],
    valid: SIMD[DType.bool, length] = SIMD[DType.bool, length](fill=True),
) -> Hit[frame, length]:
    """Trace packed spheres through the production CPU packet algorithm."""
    comptime assert length > 1
    var desc = BlasDesc.load(blases.descs.unsafe_ptr(), blas_idx)
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

    def leaf_fn(
        active: SIMD[DType.bool, length],
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
                _trace_sphere_packet_primitive[frame, length](
                    rays,
                    active,
                    ray_a,
                    ray_inv_a,
                    prim_idx,
                    Point3f32[frame](
                        block_ptr[unsafe_offset=0 * leaf_width + prim_lane],
                        block_ptr[unsafe_offset=1 * leaf_width + prim_lane],
                        block_ptr[unsafe_offset=2 * leaf_width + prim_lane],
                    ),
                    block_ptr[unsafe_offset=3 * leaf_width + prim_lane],
                    packet_hit,
                )

    trace_packet_stack_bounds_bvh[
        frame=frame,
        bounds_width=node_width,
        length=length,
        packed_meta=True,
    ](
        nodes,
        rays,
        valid,
        hit,
        leaf_fn,
        lambda (
            _active: SIMD[DType.bool, length],
            _child_ref: UInt32,
            mut _packet_hit: Hit[frame, length],
        ): None,
        lambda (_child_ref: UInt32): None,
    )
    return hit
