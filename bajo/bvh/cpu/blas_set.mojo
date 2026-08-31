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
comptime _U32_MAX_AS_INT = Int(UInt32(0xFFFFFFFF))


trait AdaptiveStreamHitSink:
    """Compile-time hit consumer for adaptive stream traversal."""

    def write[
        length: SIMDLength,
        frame: Frame,
    ](mut self, base: Int, hit: Hit[frame, length]):
        ...


@always_inline
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
