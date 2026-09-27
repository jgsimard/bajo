"""CPU BLAS construction and packed-storage assembly."""

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

comptime CPU_BLAS_OUTER_PARALLEL_MIN_PRIMITIVES = 4096
comptime EXACT_MULTI_BLAS_MIN_PRIMITIVES = 4096
comptime _U32_MAX_AS_INT = 0xFFFFFFFF


def _store_empty_blas_desc(
    descs: MutPointer[UInt32, _],
    blas_idx: Int,
    node_f32_base: Int,
    leaf_f32_base: Int,
):
    BlasDesc.empty(UInt32(node_f32_base), UInt32(leaf_f32_base)).store(
        descs, blas_idx
    )


def _compact_blas_storage[
    node_f32_stride: Int,
    leaf_f32_stride: Int,
](
    mut descs: List[UInt32],
    nodes: ImmSpan[Float32, _],
    leaves: ImmSpan[Float32, _],
    blas_count: Int,
    mut compact_nodes: List[Float32],
    mut compact_leaves: List[Float32],
):
    """Copy completed BLAS ranges out of conservative build workspace."""
    var exact_node_count = 0
    var exact_leaf_count = 0
    for blas_idx in range(blas_count):
        var desc = BlasDesc.load(descs.unsafe_ptr(), UInt32(blas_idx))
        exact_node_count += Int(desc.node_count) * node_f32_stride
        exact_leaf_count += Int(desc.leaf_block_count) * leaf_f32_stride

    compact_nodes.resize(unsafe_uninit_length=exact_node_count)
    compact_leaves.resize(unsafe_uninit_length=exact_leaf_count)
    var node_out = 0
    var leaf_out = 0
    for blas_idx in range(blas_count):
        var desc = BlasDesc.load(descs.unsafe_ptr(), UInt32(blas_idx))
        var old_node_base = Int(desc.node_f32_base)
        var old_leaf_base = Int(desc.leaf_f32_base)
        var node_count = Int(desc.node_count) * node_f32_stride
        var leaf_count = Int(desc.leaf_block_count) * leaf_f32_stride
        if node_count > 0:
            unsafe_memcpy(
                dest=compact_nodes.unsafe_ptr().unsafe_offset(node_out),
                src=nodes.unsafe_ptr().unsafe_offset(old_node_base),
                count=node_count,
            )
        if leaf_count > 0:
            unsafe_memcpy(
                dest=compact_leaves.unsafe_ptr().unsafe_offset(leaf_out),
                src=leaves.unsafe_ptr().unsafe_offset(old_leaf_base),
                count=leaf_count,
            )
        desc.node_f32_base = UInt32(node_out)
        desc.leaf_f32_base = UInt32(leaf_out)
        desc.store(descs.unsafe_ptr(), blas_idx)
        node_out += node_count
        leaf_out += leaf_count


@always_inline
def _triangle_leaf_count[
    frame: Frame,
    leaf_width: SIMDLength,
](block: TriangleLeafBlock[frame, leaf_width]) -> UInt32:
    var count = UInt32(0)
    comptime for lane in range(leaf_width):
        if block.prim_indices[lane] != EMPTY_LANE:
            count += 1
    return count


def _pack_built_triangle_blas[
    frame: Frame,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    hploc_microleaf_size: Int,
    use_dp_collapse: Bool,
    copy_leaves: Bool = True,
](
    bvh: _TriangleBuild[
        frame,
        node_width,
        leaf_width,
        hploc_microleaf_size,
        use_dp_collapse,
    ],
    descs: MutPointer[UInt32, _],
    nodes: MutPointer[Float32, _],
    leaves: MutPointer[Float32, _],
    blas_idx: Int,
    node_f32_base: Int,
    leaf_f32_base: Int,
):
    def pack_node(node_idx: Int) {imm}:
        ref node = bvh.tree.nodes[node_idx]
        var local_node_idx = UInt32(node_idx)
        var node_base = node_f32_base + _wide_node_base[node_width](
            local_node_idx
        )
        nodes.unsafe_store[width=node_width](
            node_base + WideNode.MIN_X * node_width, node.aabb._min.x
        )
        nodes.unsafe_store[width=node_width](
            node_base + WideNode.MIN_Y * node_width, node.aabb._min.y
        )
        nodes.unsafe_store[width=node_width](
            node_base + WideNode.MIN_Z * node_width, node.aabb._min.z
        )
        nodes.unsafe_store[width=node_width](
            node_base + WideNode.MAX_X * node_width, node.aabb._max.x
        )
        nodes.unsafe_store[width=node_width](
            node_base + WideNode.MAX_Y * node_width, node.aabb._max.y
        )
        nodes.unsafe_store[width=node_width](
            node_base + WideNode.MAX_Z * node_width, node.aabb._max.z
        )
        var packed_meta = SIMD[.uint32, node_width](EMPTY_LANE)
        comptime for lane in range(node_width):
            var data = node.data[lane]
            if data != EMPTY_LANE:
                if is_leaf_ref(data):
                    var block_idx = decode_ref_index(data)
                    packed_meta[lane] = _pack_wide_meta(
                        block_idx,
                        bvh.leaf_primitive_count(Int(block_idx)),
                    )
                else:
                    packed_meta[lane] = _pack_wide_meta(data, UInt32(0))
        nodes.unsafe_bitcast[UInt32]().unsafe_store[width=node_width](
            node_base + WideNode.META * node_width, packed_meta
        )

    def pack_leaf(block_idx: Int) {imm}:
        if len(bvh.packed_leaf_blocks) > 0:
            var block_stride = leaf_width * CPU_TRI_LEAF_PACKED_STRIDE
            unsafe_memcpy(
                dest=leaves.unsafe_offset(
                    leaf_f32_base + block_idx * block_stride
                ),
                src=bvh.packed_leaf_blocks.unsafe_ptr().unsafe_offset(
                    block_idx * block_stride
                ),
                count=block_stride,
            )
            return
        ref block = bvh.leaf_blocks[block_idx]
        var out = (
            leaf_f32_base + block_idx * leaf_width * CPU_TRI_LEAF_PACKED_STRIDE
        )
        leaves.unsafe_store[width=leaf_width](out + 0 * leaf_width, block.v0.x)
        leaves.unsafe_store[width=leaf_width](out + 1 * leaf_width, block.v0.y)
        leaves.unsafe_store[width=leaf_width](out + 2 * leaf_width, block.v0.z)
        leaves.unsafe_bitcast[UInt32]().unsafe_store[width=leaf_width](
            out + 3 * leaf_width, block.prim_indices
        )
        leaves.unsafe_store[width=leaf_width](out + 4 * leaf_width, block.e1.x)
        leaves.unsafe_store[width=leaf_width](out + 5 * leaf_width, block.e1.y)
        leaves.unsafe_store[width=leaf_width](out + 6 * leaf_width, block.e1.z)
        leaves.unsafe_store[width=leaf_width](out + 7 * leaf_width, block.e2.x)
        leaves.unsafe_store[width=leaf_width](out + 8 * leaf_width, block.e2.y)
        leaves.unsafe_store[width=leaf_width](out + 9 * leaf_width, block.e2.z)

    var node_count = len(bvh.tree.nodes)
    var leaf_count = bvh.leaf_block_count()
    comptime if not copy_leaves:
        leaf_count = 0
    var pack_count = node_count + leaf_count
    if bvh.tri_count >= PARALLEL_TRIANGLE_BUILD_MIN_ITEMS:

        def pack_item(item_idx: Int) {imm}:
            if item_idx < node_count:
                pack_node(item_idx)
            else:
                pack_leaf(item_idx - node_count)

        parallelize(pack_item, pack_count, _worker_count(pack_count))
    else:
        for node_idx in range(node_count):
            pack_node(node_idx)
        for block_idx in range(leaf_count):
            pack_leaf(block_idx)

    BlasDesc(
        UInt32(node_f32_base),
        UInt32(leaf_f32_base),
        UInt32(0),
        UInt32(len(bvh.tree.nodes)),
        UInt32(bvh.leaf_block_count()),
        UInt32(bvh.tri_count),
    ).store(descs, blas_idx)


def _pack_triangle_blas[
    frame: Frame,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    method: CpuBvhBuildMethod,
    hploc_microleaf_size: Int,
    use_dp_collapse: Bool,
](
    vertices: ImmSpan[Point3f32[frame], _],
    descs: MutPointer[UInt32, _],
    nodes: MutPointer[Float32, _],
    leaves: MutPointer[Float32, _],
    blas_idx: Int,
    node_f32_base: Int,
    leaf_f32_base: Int,
):
    var bvh = _TriangleBuild[
        frame,
        node_width,
        leaf_width,
        hploc_microleaf_size,
        use_dp_collapse,
    ].__init__[method](vertices)
    _pack_built_triangle_blas[
        frame,
        node_width,
        leaf_width,
        hploc_microleaf_size,
        use_dp_collapse,
    ](
        bvh,
        descs,
        nodes,
        leaves,
        blas_idx,
        node_f32_base,
        leaf_f32_base,
    )


def _build_single_triangle_blas[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    method: CpuBvhBuildMethod,
    frame: Frame,
    hploc_microleaf_size: Int,
    use_dp_collapse: Bool,
](
    vertices: ImmSpan[Point3f32[frame], _],
) -> CpuBlasSet[
    .TRIANGLE, node_width, leaf_width
]:
    """Build one BLAS into exact private packed storage."""
    var descs = List[UInt32](length=BlasDescLayout.STRIDE, fill=0)
    if len(vertices) == 0:
        _store_empty_blas_desc(descs.unsafe_ptr(), 0, 0, 0)
        return CpuBlasSet[.TRIANGLE, node_width, leaf_width](
            descs^, List[Float32](), List[Float32](), 1
        )

    var bvh = _TriangleBuild[
        frame,
        node_width,
        leaf_width,
        hploc_microleaf_size,
        use_dp_collapse,
    ].__init__[method](vertices)
    var exact_node_count = (
        len(bvh.tree.nodes) * node_width * WideNode.CHILD_STRIDE
    )
    var exact_leaf_count = (
        bvh.leaf_block_count() * leaf_width * CPU_TRI_LEAF_PACKED_STRIDE
    )
    var nodes = List[Float32](capacity=exact_node_count)
    var leaves: List[Float32]
    nodes.resize(unsafe_uninit_length=exact_node_count)
    if len(bvh.packed_leaf_blocks) > 0:
        var unused_leaf = List[Float32](length=1, fill=0.0)
        _pack_built_triangle_blas[
            frame,
            node_width,
            leaf_width,
            hploc_microleaf_size,
            use_dp_collapse,
            copy_leaves=False,
        ](
            bvh,
            descs.unsafe_ptr(),
            nodes.unsafe_ptr(),
            unused_leaf.unsafe_ptr(),
            0,
            0,
            0,
        )
        leaves = bvh.take_packed_leaf_blocks()
    else:
        leaves = List[Float32](capacity=exact_leaf_count)
        leaves.resize(unsafe_uninit_length=exact_leaf_count)
        _pack_built_triangle_blas[
            frame,
            node_width,
            leaf_width,
            hploc_microleaf_size,
            use_dp_collapse,
        ](
            bvh,
            descs.unsafe_ptr(),
            nodes.unsafe_ptr(),
            leaves.unsafe_ptr(),
            0,
            0,
            0,
        )
    return CpuBlasSet[.TRIANGLE, node_width, leaf_width](
        descs^, nodes^, leaves^, 1
    )


def _build_exact_triangle_blas_batch[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    method: CpuBvhBuildMethod,
    frame: Frame,
    hploc_microleaf_size: Int,
    use_dp_collapse: Bool,
](
    vertex_sets: ImmSpan[List[Point3f32[frame]], _],
) -> CpuBlasSet[
    .TRIANGLE, node_width, leaf_width
]:
    """Build private exact BLASes, prefix their sizes, then concatenate once."""
    var built = List[CpuBlasSet[.TRIANGLE, node_width, leaf_width]](
        capacity=len(vertex_sets)
    )
    for _ in range(len(vertex_sets)):
        built.append(
            CpuBlasSet[.TRIANGLE, node_width, leaf_width](
                List[UInt32](), List[Float32](), List[Float32](), 1
            )
        )
    var total_triangle_count = 0
    var allow_across_blas_parallelism = len(vertex_sets) > 1
    for vertices in vertex_sets:
        var tri_count = len(vertices) / 3
        total_triangle_count += tri_count
        if tri_count >= PARALLEL_TRIANGLE_BUILD_MIN_ITEMS:
            allow_across_blas_parallelism = False
    allow_across_blas_parallelism &= (
        total_triangle_count >= CPU_BLAS_OUTER_PARALLEL_MIN_PRIMITIVES
    )

    def build_one(blas_idx: Int) {imm, mut built}:
        built[blas_idx] = _build_single_triangle_blas[
            node_width,
            leaf_width,
            method,
            frame,
            hploc_microleaf_size,
            use_dp_collapse,
        ](vertex_sets[blas_idx])

    if allow_across_blas_parallelism:
        parallelize(build_one, len(vertex_sets))
    else:
        for blas_idx in range(len(vertex_sets)):
            build_one(blas_idx)

    var node_count = 0
    var leaf_count = 0
    for blas_idx in range(len(built)):
        node_count += len(built[blas_idx].nodes)
        leaf_count += len(built[blas_idx].leaves)
    debug_assert["safe", _use_compiler_assume=True](
        node_count <= _U32_MAX_AS_INT and leaf_count <= _U32_MAX_AS_INT,
        "CPU triangle BLAS packed offsets exceed UInt32",
    )

    var descs = List[UInt32](
        length=len(vertex_sets) * BlasDescLayout.STRIDE, fill=0
    )
    var nodes = List[Float32](capacity=node_count)
    var leaves = List[Float32](capacity=leaf_count)
    nodes.resize(unsafe_uninit_length=node_count)
    leaves.resize(unsafe_uninit_length=leaf_count)
    var node_base = 0
    var leaf_base = 0
    for blas_idx in range(len(built)):
        ref local = built[blas_idx]
        var desc = BlasDesc.load(local.descs.unsafe_ptr(), UInt32(0))
        if len(local.nodes) > 0:
            unsafe_memcpy(
                dest=nodes.unsafe_ptr().unsafe_offset(node_base),
                src=local.nodes.unsafe_ptr(),
                count=len(local.nodes),
            )
        if len(local.leaves) > 0:
            unsafe_memcpy(
                dest=leaves.unsafe_ptr().unsafe_offset(leaf_base),
                src=local.leaves.unsafe_ptr(),
                count=len(local.leaves),
            )
        desc.node_f32_base = UInt32(node_base)
        desc.leaf_f32_base = UInt32(leaf_base)
        desc.store(descs.unsafe_ptr(), blas_idx)
        node_base += len(local.nodes)
        leaf_base += len(local.leaves)

    return CpuBlasSet[.TRIANGLE, node_width, leaf_width](
        descs^, nodes^, leaves^, len(vertex_sets)
    )


def build_cpu_triangle_blas_set[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
    method: CpuBvhBuildMethod = .SAH,
    frame: Frame = .LOCAL,
    hploc_microleaf_size: Int = 0,
    use_dp_collapse: Bool = method != .LBVH,
](
    vertex_sets: ImmSpan[List[Point3f32[frame]], _],
) -> CpuBlasSet[
    .TRIANGLE, node_width, leaf_width
]:
    debug_assert["safe", _use_compiler_assume=True](
        len(vertex_sets) > 0, "CPU BLAS batch must be nonempty"
    )

    # A single BLAS needs no preassigned inter-BLAS offsets. Build its topology
    # first, then allocate exact uninitialized packed buffers and write once.
    if len(vertex_sets) == 1:
        return _build_single_triangle_blas[
            node_width,
            leaf_width,
            method,
            frame,
            hploc_microleaf_size,
            use_dp_collapse,
        ](vertex_sets[0])

    var exact_candidate_count = 0
    for vertices in vertex_sets:
        exact_candidate_count += len(vertices) / 3
    if exact_candidate_count >= EXACT_MULTI_BLAS_MIN_PRIMITIVES:
        return _build_exact_triangle_blas_batch[
            node_width,
            leaf_width,
            method,
            frame,
            hploc_microleaf_size,
            use_dp_collapse,
        ](vertex_sets)

    var node_bases = List[Int](capacity=len(vertex_sets))
    var leaf_bases = List[Int](capacity=len(vertex_sets))
    var node_f32_count = 0
    var leaf_f32_count = 0
    var allow_across_blas_parallelism = len(vertex_sets) > 1
    var total_triangle_count = 0
    for vertices in vertex_sets:
        debug_assert["safe", _use_compiler_assume=True](
            len(vertices) % 3 == 0,
            "each CPU triangle BLAS must contain complete triangles",
        )
        var tri_count = len(vertices) / 3
        total_triangle_count += tri_count
        node_bases.append(node_f32_count)
        leaf_bases.append(leaf_f32_count)
        if tri_count > 0:
            node_f32_count += (
                max(tri_count - 1, 1) * node_width * WideNode.CHILD_STRIDE
            )
        leaf_f32_count += tri_count * leaf_width * CPU_TRI_LEAF_PACKED_STRIDE
        if tri_count >= PARALLEL_TRIANGLE_BUILD_MIN_ITEMS:
            allow_across_blas_parallelism = False

    allow_across_blas_parallelism &= (
        total_triangle_count >= CPU_BLAS_OUTER_PARALLEL_MIN_PRIMITIVES
    )
    debug_assert["safe", _use_compiler_assume=True](
        node_f32_count <= _U32_MAX_AS_INT and leaf_f32_count <= _U32_MAX_AS_INT,
        "CPU triangle BLAS packed offsets exceed UInt32",
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
        if len(vertex_sets[blas_idx]) == 0:
            _store_empty_blas_desc(
                descs_ptr,
                blas_idx,
                node_bases[blas_idx],
                leaf_bases[blas_idx],
            )
            return
        _pack_triangle_blas[
            frame,
            node_width,
            leaf_width,
            method,
            hploc_microleaf_size,
            use_dp_collapse,
        ](
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

    var compact_nodes = List[Float32]()
    var compact_leaves = List[Float32]()
    _compact_blas_storage[
        node_width * WideNode.CHILD_STRIDE,
        leaf_width * CPU_TRI_LEAF_PACKED_STRIDE,
    ](
        descs,
        nodes,
        leaves,
        len(vertex_sets),
        compact_nodes,
        compact_leaves,
    )
    return CpuBlasSet[.TRIANGLE, node_width, leaf_width](
        descs^, compact_nodes^, compact_leaves^, len(vertex_sets)
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


def _pack_built_sphere_blas[
    frame: Frame,
    width: SIMDLength,
](
    bvh: _SphereBuild[frame, width],
    descs: MutPointer[UInt32, _],
    nodes: MutPointer[Float32, _],
    leaves: MutPointer[Float32, _],
    blas_idx: Int,
    node_f32_base: Int,
    leaf_f32_base: Int,
):
    var nodes_u32 = nodes.unsafe_bitcast[UInt32]()
    for node_idx in range(len(bvh.tree.nodes)):
        ref node = bvh.tree.nodes[node_idx]
        var local_node_idx = UInt32(node_idx)
        var node_base = node_f32_base + _wide_node_base[width](local_node_idx)
        nodes.unsafe_store[width=width](
            node_base + WideNode.MIN_X * width, node.aabb._min.x
        )
        nodes.unsafe_store[width=width](
            node_base + WideNode.MIN_Y * width, node.aabb._min.y
        )
        nodes.unsafe_store[width=width](
            node_base + WideNode.MIN_Z * width, node.aabb._min.z
        )
        nodes.unsafe_store[width=width](
            node_base + WideNode.MAX_X * width, node.aabb._max.x
        )
        nodes.unsafe_store[width=width](
            node_base + WideNode.MAX_Y * width, node.aabb._max.y
        )
        nodes.unsafe_store[width=width](
            node_base + WideNode.MAX_Z * width, node.aabb._max.z
        )
        var packed_meta = SIMD[.uint32, width](EMPTY_LANE)
        comptime for lane in range(width):
            var data = node.data[lane]
            if data != EMPTY_LANE:
                if is_leaf_ref(data):
                    var block_idx = decode_ref_index(data)
                    packed_meta[lane] = _pack_wide_meta(
                        block_idx,
                        _sphere_leaf_count[frame, width](
                            bvh.leaf_blocks[Int(block_idx)]
                        ),
                    )
                else:
                    packed_meta[lane] = _pack_wide_meta(data, UInt32(0))
        nodes_u32.unsafe_store[width=width](
            node_base + WideNode.META * width, packed_meta
        )

    var leaves_u32 = leaves.unsafe_bitcast[UInt32]()
    for block_idx in range(len(bvh.leaf_blocks)):
        ref block = bvh.leaf_blocks[block_idx]
        var out = leaf_f32_base + block_idx * width * SPHERE_LEAF_PACKED_STRIDE
        leaves.unsafe_store[width=width](out + 0 * width, block.center.x)
        leaves.unsafe_store[width=width](out + 1 * width, block.center.y)
        leaves.unsafe_store[width=width](out + 2 * width, block.center.z)
        leaves.unsafe_store[width=width](out + 3 * width, block.radius)
        leaves_u32.unsafe_store[width=width](
            out + 4 * width, block.prim_indices
        )

    BlasDesc(
        UInt32(node_f32_base),
        UInt32(leaf_f32_base),
        UInt32(0),
        UInt32(len(bvh.tree.nodes)),
        UInt32(len(bvh.leaf_blocks)),
        UInt32(bvh.sphere_count),
    ).store(descs, blas_idx)


def _pack_sphere_blas[
    frame: Frame,
    width: SIMDLength,
    method: CpuBvhBuildMethod,
](
    spheres: ImmSpan[Sphere[frame], _],
    descs: MutPointer[UInt32, _],
    nodes: MutPointer[Float32, _],
    leaves: MutPointer[Float32, _],
    blas_idx: Int,
    node_f32_base: Int,
    leaf_f32_base: Int,
):
    var bvh = _SphereBuild[frame, width].__init__[method](spheres)
    _pack_built_sphere_blas[frame, width](
        bvh,
        descs,
        nodes,
        leaves,
        blas_idx,
        node_f32_base,
        leaf_f32_base,
    )


def _build_single_sphere_blas[
    width: SIMDLength,
    method: CpuBvhBuildMethod,
    frame: Frame,
](spheres: ImmSpan[Sphere[frame], _]) -> CpuBlasSet[.SPHERE, width]:
    """Build one sphere BLAS into exact private packed storage."""
    var descs = List[UInt32](length=BlasDescLayout.STRIDE, fill=0)
    if len(spheres) == 0:
        _store_empty_blas_desc(descs.unsafe_ptr(), 0, 0, 0)
        return CpuBlasSet[.SPHERE, width](
            descs^, List[Float32](), List[Float32](), 1
        )

    var bvh = _SphereBuild[frame, width].__init__[method](spheres)
    var exact_node_count = len(bvh.tree.nodes) * width * WideNode.CHILD_STRIDE
    var exact_leaf_count = (
        len(bvh.leaf_blocks) * width * SPHERE_LEAF_PACKED_STRIDE
    )
    var nodes = List[Float32](capacity=exact_node_count)
    var leaves = List[Float32](capacity=exact_leaf_count)
    nodes.resize(unsafe_uninit_length=exact_node_count)
    leaves.resize(unsafe_uninit_length=exact_leaf_count)
    _pack_built_sphere_blas[frame, width](
        bvh,
        descs.unsafe_ptr(),
        nodes.unsafe_ptr(),
        leaves.unsafe_ptr(),
        0,
        0,
        0,
    )
    return CpuBlasSet[.SPHERE, width](descs^, nodes^, leaves^, 1)


def _build_exact_sphere_blas_batch[
    width: SIMDLength,
    method: CpuBvhBuildMethod,
    frame: Frame,
](sphere_sets: ImmSpan[List[Sphere[frame]], _]) -> CpuBlasSet[.SPHERE, width]:
    var built = List[CpuBlasSet[.SPHERE, width]](capacity=len(sphere_sets))
    for _ in range(len(sphere_sets)):
        built.append(
            CpuBlasSet[.SPHERE, width](
                List[UInt32](), List[Float32](), List[Float32](), 1
            )
        )

    def build_one(blas_idx: Int) {imm, mut built}:
        built[blas_idx] = _build_single_sphere_blas[width, method, frame](
            sphere_sets[blas_idx]
        )

    parallelize(build_one, len(sphere_sets))

    var node_count = 0
    var leaf_count = 0
    for blas_idx in range(len(built)):
        node_count += len(built[blas_idx].nodes)
        leaf_count += len(built[blas_idx].leaves)
    debug_assert["safe", _use_compiler_assume=True](
        node_count <= _U32_MAX_AS_INT and leaf_count <= _U32_MAX_AS_INT,
        "CPU sphere BLAS packed offsets exceed UInt32",
    )

    var descs = List[UInt32](
        length=len(sphere_sets) * BlasDescLayout.STRIDE, fill=0
    )
    var nodes = List[Float32](capacity=node_count)
    var leaves = List[Float32](capacity=leaf_count)
    nodes.resize(unsafe_uninit_length=node_count)
    leaves.resize(unsafe_uninit_length=leaf_count)
    var node_base = 0
    var leaf_base = 0
    for blas_idx in range(len(built)):
        ref local = built[blas_idx]
        var desc = BlasDesc.load(local.descs.unsafe_ptr(), UInt32(0))
        if len(local.nodes) > 0:
            unsafe_memcpy(
                dest=nodes.unsafe_ptr().unsafe_offset(node_base),
                src=local.nodes.unsafe_ptr(),
                count=len(local.nodes),
            )
        if len(local.leaves) > 0:
            unsafe_memcpy(
                dest=leaves.unsafe_ptr().unsafe_offset(leaf_base),
                src=local.leaves.unsafe_ptr(),
                count=len(local.leaves),
            )
        desc.node_f32_base = UInt32(node_base)
        desc.leaf_f32_base = UInt32(leaf_base)
        desc.store(descs.unsafe_ptr(), blas_idx)
        node_base += len(local.nodes)
        leaf_base += len(local.leaves)

    return CpuBlasSet[.SPHERE, width](descs^, nodes^, leaves^, len(sphere_sets))


def build_cpu_sphere_blas_set[
    width: SIMDLength,
    method: CpuBvhBuildMethod = .SAH,
    frame: Frame = .LOCAL,
](sphere_sets: ImmSpan[List[Sphere[frame]], _],) -> CpuBlasSet[.SPHERE, width]:
    debug_assert["safe", _use_compiler_assume=True](
        len(sphere_sets) > 0, "CPU BLAS batch must be nonempty"
    )

    if len(sphere_sets) == 1:
        var descs = List[UInt32](length=BlasDescLayout.STRIDE, fill=0)
        if len(sphere_sets[0]) == 0:
            _store_empty_blas_desc(descs.unsafe_ptr(), 0, 0, 0)
            return CpuBlasSet[.SPHERE, width](
                descs^, List[Float32](), List[Float32](), 1
            )

        var bvh = _SphereBuild[frame, width].__init__[method](sphere_sets[0])
        var exact_node_count = (
            len(bvh.tree.nodes) * width * WideNode.CHILD_STRIDE
        )
        var exact_leaf_count = (
            len(bvh.leaf_blocks) * width * SPHERE_LEAF_PACKED_STRIDE
        )
        var nodes = List[Float32](capacity=exact_node_count)
        var leaves = List[Float32](capacity=exact_leaf_count)
        nodes.resize(unsafe_uninit_length=exact_node_count)
        leaves.resize(unsafe_uninit_length=exact_leaf_count)
        _pack_built_sphere_blas[frame, width](
            bvh,
            descs.unsafe_ptr(),
            nodes.unsafe_ptr(),
            leaves.unsafe_ptr(),
            0,
            0,
            0,
        )
        return CpuBlasSet[.SPHERE, width](descs^, nodes^, leaves^, 1)

    var exact_candidate_count = 0
    for spheres in sphere_sets:
        exact_candidate_count += len(spheres)
    if exact_candidate_count >= EXACT_MULTI_BLAS_MIN_PRIMITIVES:
        return _build_exact_sphere_blas_batch[width, method, frame](sphere_sets)

    var node_bases = List[Int](capacity=len(sphere_sets))
    var leaf_bases = List[Int](capacity=len(sphere_sets))
    var node_f32_count = 0
    var leaf_f32_count = 0
    var total_sphere_count = 0
    for spheres in sphere_sets:
        var sphere_count = len(spheres)
        total_sphere_count += sphere_count
        node_bases.append(node_f32_count)
        leaf_bases.append(leaf_f32_count)
        if sphere_count > 0:
            node_f32_count += (
                max(sphere_count - 1, 1) * width * WideNode.CHILD_STRIDE
            )
        leaf_f32_count += sphere_count * width * SPHERE_LEAF_PACKED_STRIDE

    debug_assert["safe", _use_compiler_assume=True](
        node_f32_count <= _U32_MAX_AS_INT and leaf_f32_count <= _U32_MAX_AS_INT,
        "CPU sphere BLAS packed offsets exceed UInt32",
    )

    var descs = List[UInt32](
        length=len(sphere_sets) * BlasDescLayout.STRIDE, fill=0
    )
    var nodes = List[Float32](length=node_f32_count, fill=0.0)
    var leaves = List[Float32](length=leaf_f32_count, fill=0.0)
    var descs_ptr = descs.unsafe_ptr()
    var nodes_ptr = nodes.unsafe_ptr()
    var leaves_ptr = leaves.unsafe_ptr()

    def build_one(blas_idx: Int) {imm}:
        if len(sphere_sets[blas_idx]) == 0:
            _store_empty_blas_desc(
                descs_ptr,
                blas_idx,
                node_bases[blas_idx],
                leaf_bases[blas_idx],
            )
            return
        _pack_sphere_blas[frame, width, method](
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

    var compact_nodes = List[Float32]()
    var compact_leaves = List[Float32]()
    _compact_blas_storage[
        width * WideNode.CHILD_STRIDE,
        width * SPHERE_LEAF_PACKED_STRIDE,
    ](
        descs,
        nodes,
        leaves,
        len(sphere_sets),
        compact_nodes,
        compact_leaves,
    )
    return CpuBlasSet[.SPHERE, width](
        descs^, compact_nodes^, compact_leaves^, len(sphere_sets)
    )
