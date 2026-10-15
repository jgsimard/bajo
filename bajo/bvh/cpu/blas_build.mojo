"""CPU BLAS construction and packed-storage assembly."""

from max.algorithm import parallelize
from std.memory import unsafe_memcpy

from bajo.bvh.constants import (
    EMPTY_LANE,
    SPHERE_LEAF_PACKED_STRIDE,
    CPU_TRI_LEAF_PACKED_STRIDE,
    WideNode,
)
from bajo.bvh.cpu.bounds_bvh import WideBvhNode
from bajo.bvh.cpu.sphere_bvh import _SphereBuild
from bajo.bvh.cpu.blas_storage import CpuBlasSet
from bajo.bvh.cpu.blas_assembly import (
    _assemble_exact_blases,
    _BlasBuildLayout,
    _store_empty_blas_desc,
    _finish_blas_workspace,
)
from bajo.bvh.cpu.build_method import CpuBvhBuildMethod
from bajo.bvh.cpu.triangle_bvh import (
    PARALLEL_TRIANGLE_BUILD_MIN_ITEMS,
    _TriangleBuild,
)
from bajo.bvh.cpu.parallel import _worker_count
from bajo.bvh.tagged_ref import decode_ref_index, is_leaf_ref
from bajo.bvh.types import (
    BlasDesc,
    BlasDescLayout,
    Sphere,
    SphereLeafBlock,
)
from bajo.bvh.wide_meta import _pack_wide_meta, _wide_node_base
from bajo.core import (
    Frame,
    Point3f32,
)

comptime CPU_BLAS_OUTER_PARALLEL_MIN_PRIMITIVES = 4096
comptime EXACT_MULTI_BLAS_MIN_PRIMITIVES = 4096


@always_inline
def _store_wide_blas_node[
    frame: Frame, width: SIMDLength
](
    node: WideBvhNode[frame, width],
    leaf_counts: SIMD[.uint32, width],
    nodes: MutPointer[Float32, _],
    node_idx: UInt32,
    node_f32_base: Int,
):
    """Serialize one wide node using primitive-specific leaf counts."""
    var base = node_f32_base + _wide_node_base[width](node_idx)
    nodes.unsafe_store[width=width](
        base + WideNode.MIN_X * width, node.aabb._min.x
    )
    nodes.unsafe_store[width=width](
        base + WideNode.MIN_Y * width, node.aabb._min.y
    )
    nodes.unsafe_store[width=width](
        base + WideNode.MIN_Z * width, node.aabb._min.z
    )
    nodes.unsafe_store[width=width](
        base + WideNode.MAX_X * width, node.aabb._max.x
    )
    nodes.unsafe_store[width=width](
        base + WideNode.MAX_Y * width, node.aabb._max.y
    )
    nodes.unsafe_store[width=width](
        base + WideNode.MAX_Z * width, node.aabb._max.z
    )
    var packed_meta = SIMD[.uint32, width](EMPTY_LANE)
    comptime for lane in range(width):
        var data = node.data[lane]
        if data != EMPTY_LANE:
            if is_leaf_ref(data):
                packed_meta[lane] = _pack_wide_meta(
                    decode_ref_index(data), leaf_counts[lane]
                )
            else:
                packed_meta[lane] = _pack_wide_meta(data, UInt32(0))
    nodes.unsafe_bitcast[UInt32]().unsafe_store[width=width](
        base + WideNode.META * width, packed_meta
    )


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
        var leaf_counts = SIMD[.uint32, node_width](0)
        comptime for lane in range(node_width):
            var data = node.data[lane]
            if data != EMPTY_LANE and is_leaf_ref(data):
                leaf_counts[lane] = bvh.leaf_primitive_count(
                    Int(decode_ref_index(data))
                )
        _store_wide_blas_node(
            node, leaf_counts, nodes, UInt32(node_idx), node_f32_base
        )

    def pack_leaf(block_idx: Int) {imm}:
        var block_stride = leaf_width * CPU_TRI_LEAF_PACKED_STRIDE
        unsafe_memcpy(
            dest=leaves.unsafe_offset(leaf_f32_base + block_idx * block_stride),
            src=bvh.packed_leaf_blocks.unsafe_ptr().unsafe_offset(
                block_idx * block_stride
            ),
            count=block_stride,
        )

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
    var nodes = List[Float32](capacity=exact_node_count)
    nodes.resize(unsafe_uninit_length=exact_node_count)
    var unused_leaf = List[Float32]()
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
    var leaves = bvh.take_packed_leaf_blocks()
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
        # Large BLASes already use internal parallelism. Build those first,
        # then distribute the remaining disjoint small builds across workers.
        var small_blases = List[Int](capacity=len(vertex_sets))
        for blas_idx in range(len(vertex_sets)):
            if (
                len(vertex_sets[blas_idx]) // 3
                >= PARALLEL_TRIANGLE_BUILD_MIN_ITEMS
            ):
                build_one(blas_idx)
            else:
                small_blases.append(blas_idx)

        def build_small(task_idx: Int) {imm, mut build_one}:
            build_one(small_blases[task_idx])

        if len(small_blases) > 1:
            parallelize(build_small, len(small_blases))
        elif len(small_blases) == 1:
            build_one(small_blases[0])

    return _assemble_exact_blases[.TRIANGLE, node_width, leaf_width](built)


def build_cpu_triangle_blas_set[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
    method: CpuBvhBuildMethod = .SAH,
    frame: Frame = .LOCAL,
    hploc_microleaf_size: Int = 0,
    use_dp_collapse: Bool = method != .LBVH,
](
    vertices: ImmSpan[Point3f32[frame], _],
) -> CpuBlasSet[
    .TRIANGLE, node_width, leaf_width
]:
    """Build one borrowed triangle mesh without an owning batch wrapper."""
    return _build_single_triangle_blas[
        node_width,
        leaf_width,
        method,
        frame,
        hploc_microleaf_size,
        use_dp_collapse,
    ](vertices)


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

    var layout = _BlasBuildLayout[
        node_width * WideNode.CHILD_STRIDE,
        leaf_width * CPU_TRI_LEAF_PACKED_STRIDE,
    ].__init__[Point3f32[frame], 3](vertex_sets)
    var descs = List[UInt32](
        length=len(vertex_sets) * BlasDescLayout.STRIDE, fill=0
    )
    var nodes = List[Float32](capacity=layout.node_f32_count)
    var leaves = List[Float32](capacity=layout.leaf_f32_count)
    nodes.resize(unsafe_uninit_length=layout.node_f32_count)
    leaves.resize(unsafe_uninit_length=layout.leaf_f32_count)
    var descs_ptr = descs.unsafe_ptr()
    var nodes_ptr = nodes.unsafe_ptr()
    var leaves_ptr = leaves.unsafe_ptr()

    def build_one(blas_idx: Int) {imm}:
        if len(vertex_sets[blas_idx]) == 0:
            _store_empty_blas_desc(
                descs_ptr,
                blas_idx,
                layout.node_bases[blas_idx],
                layout.leaf_bases[blas_idx],
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
            layout.node_bases[blas_idx],
            layout.leaf_bases[blas_idx],
        )

    # Larger batches have already dispatched to the exact parallel builder.
    for blas_idx in range(len(vertex_sets)):
        build_one(blas_idx)

    return _finish_blas_workspace[
        .TRIANGLE,
        node_width,
        leaf_width,
        node_width * WideNode.CHILD_STRIDE,
        leaf_width * CPU_TRI_LEAF_PACKED_STRIDE,
    ](descs^, nodes, leaves)


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
    for node_idx in range(len(bvh.tree.nodes)):
        ref node = bvh.tree.nodes[node_idx]
        var leaf_counts = SIMD[.uint32, width](0)
        comptime for lane in range(width):
            var data = node.data[lane]
            if data != EMPTY_LANE and is_leaf_ref(data):
                leaf_counts[lane] = _sphere_leaf_count[frame, width](
                    bvh.leaf_blocks[Int(decode_ref_index(data))]
                )
        _store_wide_blas_node(
            node, leaf_counts, nodes, UInt32(node_idx), node_f32_base
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

    return _assemble_exact_blases[.SPHERE, width, width](built)


def build_cpu_sphere_blas_set[
    width: SIMDLength,
    method: CpuBvhBuildMethod = .SAH,
    frame: Frame = .LOCAL,
](sphere_sets: ImmSpan[List[Sphere[frame]], _],) -> CpuBlasSet[.SPHERE, width]:
    debug_assert["safe", _use_compiler_assume=True](
        len(sphere_sets) > 0, "CPU BLAS batch must be nonempty"
    )

    if len(sphere_sets) == 1:
        return _build_single_sphere_blas[width, method, frame](sphere_sets[0])

    var exact_candidate_count = 0
    for spheres in sphere_sets:
        exact_candidate_count += len(spheres)
    if exact_candidate_count >= EXACT_MULTI_BLAS_MIN_PRIMITIVES:
        return _build_exact_sphere_blas_batch[width, method, frame](sphere_sets)

    var layout = _BlasBuildLayout[
        width * WideNode.CHILD_STRIDE, width * SPHERE_LEAF_PACKED_STRIDE
    ].__init__[Sphere[frame], 1](sphere_sets)
    var descs = List[UInt32](
        length=len(sphere_sets) * BlasDescLayout.STRIDE, fill=0
    )
    var nodes = List[Float32](capacity=layout.node_f32_count)
    var leaves = List[Float32](capacity=layout.leaf_f32_count)
    nodes.resize(unsafe_uninit_length=layout.node_f32_count)
    leaves.resize(unsafe_uninit_length=layout.leaf_f32_count)
    var descs_ptr = descs.unsafe_ptr()
    var nodes_ptr = nodes.unsafe_ptr()
    var leaves_ptr = leaves.unsafe_ptr()

    def build_one(blas_idx: Int) {imm}:
        if len(sphere_sets[blas_idx]) == 0:
            _store_empty_blas_desc(
                descs_ptr,
                blas_idx,
                layout.node_bases[blas_idx],
                layout.leaf_bases[blas_idx],
            )
            return
        _pack_sphere_blas[frame, width, method](
            sphere_sets[blas_idx],
            descs_ptr,
            nodes_ptr,
            leaves_ptr,
            blas_idx,
            layout.node_bases[blas_idx],
            layout.leaf_bases[blas_idx],
        )

    # Larger batches have already dispatched to the exact parallel builder.
    for blas_idx in range(len(sphere_sets)):
        build_one(blas_idx)

    return _finish_blas_workspace[
        .SPHERE,
        width,
        width,
        width * WideNode.CHILD_STRIDE,
        width * SPHERE_LEAF_PACKED_STRIDE,
    ](descs^, nodes, leaves)
