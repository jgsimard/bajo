"""Primitive-independent sizing and assembly of packed CPU BLAS storage."""

from std.memory import unsafe_memcpy
from bajo.bvh.constants import PrimitiveKind
from bajo.bvh.cpu.blas_storage import CpuBlasSet
from bajo.bvh.types import BlasDesc, BlasDescLayout

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


def _assemble_exact_blases[
    kind: PrimitiveKind,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
](
    built: ImmSpan[CpuBlasSet[kind, node_width, leaf_width], _],
) -> CpuBlasSet[
    kind, node_width, leaf_width
]:
    """Concatenate private BLAS buffers and relocate their descriptor bases."""
    var node_count = 0
    var leaf_count = 0
    for blas_idx in range(len(built)):
        node_count += len(built[blas_idx].nodes)
        leaf_count += len(built[blas_idx].leaves)
    debug_assert["safe", _use_compiler_assume=True](
        node_count <= _U32_MAX_AS_INT and leaf_count <= _U32_MAX_AS_INT,
        "CPU BLAS packed offsets exceed UInt32",
    )

    var descs = List[UInt32](length=len(built) * BlasDescLayout.STRIDE, fill=0)
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

    return CpuBlasSet[kind, node_width, leaf_width](
        descs^, nodes^, leaves^, len(built)
    )


struct _BlasBuildLayout[node_f32_stride: Int, leaf_f32_stride: Int]:
    """Conservative per-BLAS offsets and packed workspace sizes."""

    var node_bases: List[Int]
    var leaf_bases: List[Int]
    var node_f32_count: Int
    var leaf_f32_count: Int

    def __init__[
        T: Copyable, items_per_primitive: Int
    ](out self, sets: ImmSpan[List[T], _]):
        comptime assert items_per_primitive > 0
        self.node_bases = List[Int](capacity=len(sets))
        self.leaf_bases = List[Int](capacity=len(sets))
        var node_count = 0
        var leaf_count = 0
        for items in sets:
            debug_assert["safe", _use_compiler_assume=True](
                len(items) % items_per_primitive == 0,
                "each CPU BLAS must contain complete primitives",
            )
            var count = len(items) / items_per_primitive
            self.node_bases.append(node_count)
            self.leaf_bases.append(leaf_count)
            if count > 0:
                node_count += max(count - 1, 1) * Self.node_f32_stride
            leaf_count += count * Self.leaf_f32_stride
        debug_assert["safe", _use_compiler_assume=True](
            node_count <= _U32_MAX_AS_INT and leaf_count <= _U32_MAX_AS_INT,
            "CPU BLAS packed offsets exceed UInt32",
        )
        self.node_f32_count = node_count
        self.leaf_f32_count = leaf_count


def _finish_blas_workspace[
    kind: PrimitiveKind,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    node_f32_stride: Int,
    leaf_f32_stride: Int,
](
    var descs: List[UInt32],
    nodes: ImmSpan[Float32, _],
    leaves: ImmSpan[Float32, _],
) -> CpuBlasSet[kind, node_width, leaf_width]:
    """Compact completed conservative storage once and transfer ownership."""
    var blas_count = len(descs) / BlasDescLayout.STRIDE
    var compact_nodes = List[Float32]()
    var compact_leaves = List[Float32]()
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
    return CpuBlasSet[kind, node_width, leaf_width](
        descs^, compact_nodes^, compact_leaves^, blas_count
    )
