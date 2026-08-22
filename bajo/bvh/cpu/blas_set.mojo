from max.algorithm import parallelize
from max.gpu.host import DeviceContext

from bajo.bvh.constants import (
    EMPTY_LANE,
    SPHERE_LEAF_PACKED_STRIDE,
    TRACE,
    TRI_LEAF_PACKED_STRIDE,
    WideNode,
)
from bajo.bvh.cpu.sphere_bvh import SphereBvh, _trace_sphere_leaf_block
from bajo.bvh.cpu.triangle_bvh import (
    PARALLEL_TRIANGLE_BUILD_MIN_ITEMS,
    TriangleBvh,
    _trace_triangle_leaf_block,
)
from bajo.bvh.cpu.bounds_bvh import WideBvhNode
from bajo.bvh.cpu.trace import (
    CpuBvhTraversalStats,
    trace_packed_bounds_bvh,
    trace_packed_sphere_bounds_bvh,
)
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
from bajo.core import Frame, Point3, Point3f32, Rayf32, Vec3


comptime CPU_BLAS_OUTER_PARALLEL_MIN_PRIMITIVES = 4096


def _triangle_leaf_count[
    leaf_width: SIMDLength,
](block: TriangleLeafBlock[Frame.LOCAL, leaf_width]) -> UInt32:
    var count = UInt32(0)
    comptime for lane in range(leaf_width):
        if block.prim_indices[lane] != EMPTY_LANE:
            count += 1
    return count


def _pack_triangle_blas[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    split_method: String,
](
    vertices: ImmSpan[Point3f32[Frame.LOCAL], _],
    descs: MutPointer[UInt32, _],
    nodes: MutPointer[Float32, _],
    leaves: MutPointer[Float32, _],
    blas_idx: Int,
    node_f32_base: Int,
    leaf_f32_base: Int,
):
    var bvh = TriangleBvh[Frame.LOCAL, node_width, leaf_width].__init__[
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
                        _triangle_leaf_count(bvh.leaf_blocks[Int(block_idx)]),
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
](
    mut ctx: DeviceContext,
    vertex_sets: ImmSpan[List[Point3f32[Frame.LOCAL]], _],
) raises -> CpuBlasSet[node_width, leaf_width]:
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

    var descs = ctx.enqueue_create_host_buffer[DType.uint32](
        len(vertex_sets) * BlasDescLayout.STRIDE
    )
    var nodes = ctx.enqueue_create_host_buffer[DType.float32](node_f32_count)
    var leaves = ctx.enqueue_create_host_buffer[DType.float32](leaf_f32_count)
    var descs_ptr = descs.unsafe_ptr()
    var nodes_ptr = nodes.unsafe_ptr()
    var leaves_ptr = leaves.unsafe_ptr()

    def build_one(blas_idx: Int) {imm}:
        _pack_triangle_blas[node_width, leaf_width, split_method](
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
    width: SIMDLength,
](block: SphereLeafBlock[Frame.LOCAL, width]) -> UInt32:
    var count = UInt32(0)
    comptime for lane in range(width):
        if block.prim_indices[lane] != EMPTY_LANE:
            count += 1
    return count


def _pack_sphere_blas[
    width: SIMDLength,
    split_method: String,
](
    spheres: ImmSpan[Sphere[Frame.LOCAL], _],
    descs: MutPointer[UInt32, _],
    nodes: MutPointer[Float32, _],
    leaves: MutPointer[Float32, _],
    blas_idx: Int,
    node_f32_base: Int,
    leaf_f32_base: Int,
):
    var owned_spheres = [sphere.copy() for sphere in spheres]
    var bvh = SphereBvh[Frame.LOCAL, width].__init__[split_method](
        owned_spheres^
    )
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
                        _sphere_leaf_count(bvh.leaf_blocks[Int(block_idx)]),
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
](
    mut ctx: DeviceContext,
    sphere_sets: ImmSpan[List[Sphere[Frame.LOCAL]], _],
) raises -> CpuBlasSet[width]:
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

    var descs = ctx.enqueue_create_host_buffer[DType.uint32](
        len(sphere_sets) * BlasDescLayout.STRIDE
    )
    var nodes = ctx.enqueue_create_host_buffer[DType.float32](node_f32_count)
    var leaves = ctx.enqueue_create_host_buffer[DType.float32](leaf_f32_count)
    var descs_ptr = descs.unsafe_ptr()
    var nodes_ptr = nodes.unsafe_ptr()
    var leaves_ptr = leaves.unsafe_ptr()

    def build_one(blas_idx: Int) {imm}:
        _pack_sphere_blas[width, split_method](
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


def trace_triangle_blas_set[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
    mode: TRACE = TRACE.CLOSEST_HIT,
](
    blases: CpuBlasSet[node_width, leaf_width],
    blas_idx: UInt32,
    ray: Rayf32[Frame.LOCAL],
) -> Hit[Frame.LOCAL]:
    var desc = BlasDesc.load(blases.descs.unsafe_ptr(), blas_idx)
    var nodes_ptr = (
        blases.nodes.unsafe_ptr()
        .unsafe_offset(Int(desc.node_f32_base))
        .unsafe_bitcast[WideBvhNode[Frame.LOCAL, node_width]]()
    )
    var nodes = Span(unsafe_ptr=nodes_ptr, length=Int(desc.node_count))
    var leaves = blases.leaves.unsafe_ptr().unsafe_offset(
        Int(desc.leaf_f32_base)
    )
    var unused_stats = CpuBvhTraversalStats()

    @always_inline
    def leaf_fn(
        ray: Rayf32[Frame.LOCAL],
        O: Point3[DType.float32, Frame.LOCAL, leaf_width],
        D: Vec3[DType.float32, Frame.LOCAL, leaf_width],
        _ray_a: SIMD[DType.float32, leaf_width],
        _ray_inv_a: SIMD[DType.float32, leaf_width],
        leaf_block_idx: UInt32,
        mut hit: Hit[Frame.LOCAL],
    ) {imm, mut unused_stats} -> Bool:
        var block_base = (
            Int(leaf_block_idx) * TRI_LEAF_PACKED_STRIDE * leaf_width
        )
        var block_ptr = leaves.unsafe_offset(block_base)
        var block = TriangleLeafBlock[Frame.LOCAL, leaf_width]()
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
        return _trace_triangle_leaf_block[
            Frame.LOCAL,
            leaf_width,
            mode,
            False,
            packed_layout=True,
        ](ray, O, D, block, block_ptr, unused_stats, hit)

    return trace_packed_bounds_bvh[
        Frame.LOCAL,
        node_width,
        leaf_width,
        mode,
    ](nodes, ray, leaf_fn)


def trace_sphere_blas_set[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
    mode: TRACE = TRACE.CLOSEST_HIT,
](
    blases: CpuBlasSet[node_width, leaf_width],
    blas_idx: UInt32,
    ray: Rayf32[Frame.LOCAL],
) -> Hit[Frame.LOCAL]:
    var desc = BlasDesc.load(blases.descs.unsafe_ptr(), blas_idx)
    var nodes_ptr = (
        blases.nodes.unsafe_ptr()
        .unsafe_offset(Int(desc.node_f32_base))
        .unsafe_bitcast[WideBvhNode[Frame.LOCAL, node_width]]()
    )
    var nodes = Span(unsafe_ptr=nodes_ptr, length=Int(desc.node_count))
    var leaves = blases.leaves.unsafe_ptr().unsafe_offset(
        Int(desc.leaf_f32_base)
    )

    @always_inline
    def leaf_fn(
        ray: Rayf32[Frame.LOCAL],
        O: Point3[DType.float32, Frame.LOCAL, leaf_width],
        D: Vec3[DType.float32, Frame.LOCAL, leaf_width],
        ray_a: SIMD[DType.float32, leaf_width],
        ray_inv_a: SIMD[DType.float32, leaf_width],
        leaf_block_idx: UInt32,
        mut hit: Hit[Frame.LOCAL],
    ) {imm} -> Bool:
        var block_base = (
            Int(leaf_block_idx) * SPHERE_LEAF_PACKED_STRIDE * leaf_width
        )
        var block_ptr = leaves.unsafe_offset(block_base)
        var block = SphereLeafBlock[Frame.LOCAL, leaf_width]()
        block.center.x = block_ptr.unsafe_load[width=leaf_width](0 * leaf_width)
        block.center.y = block_ptr.unsafe_load[width=leaf_width](1 * leaf_width)
        block.center.z = block_ptr.unsafe_load[width=leaf_width](2 * leaf_width)
        block.radius = block_ptr.unsafe_load[width=leaf_width](3 * leaf_width)
        block.prim_indices = block_ptr.unsafe_bitcast[UInt32]().unsafe_load[
            width=leaf_width
        ](4 * leaf_width)
        return _trace_sphere_leaf_block[Frame.LOCAL, leaf_width, mode](
            ray, O, D, ray_a, ray_inv_a, block, hit
        )

    return trace_packed_sphere_bounds_bvh[
        Frame.LOCAL,
        node_width,
        leaf_width,
        mode,
    ](nodes, ray, leaf_fn)
