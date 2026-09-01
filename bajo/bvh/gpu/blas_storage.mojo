"""GPU ownership and upload adapter for packed BLAS storage."""

from std.memory import unsafe_memcpy
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh.cpu.blas_storage import CpuBlasSet
from bajo.bvh.constants import (
    CPU_TRI_LEAF_PACKED_STRIDE,
    TRI_LEAF_PACKED_STRIDE,
    PrimitiveKind,
)
from bajo.bvh.types import BlasDesc, BlasDescLayout


@fieldwise_init
struct GpuBvhLayout(Equatable, ImplicitlyCopyable):
    """Enum-like compile-time selector for GPU BLAS byte layout."""

    comptime WIDE = Self(False)
    comptime CWBVH8 = Self(True)

    var compressed: Bool


@fieldwise_init
struct GpuBlasSet[
    kind: PrimitiveKind,
    layout: GpuBvhLayout,
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
]:
    """Own primitive- and layout-typed BLAS buffers on one GPU device."""

    var descs: DeviceBuffer[.uint32]
    var nodes: DeviceBuffer[.float32]
    var leaves: DeviceBuffer[.float32]
    var blas_count: Int

    @staticmethod
    def empty(mut ctx: DeviceContext, blas_count: Int) raises -> Self:
        """Allocate an empty device BLAS set with `blas_count` descriptors."""
        var descs = ctx.enqueue_create_buffer[.uint32](
            blas_count * BlasDescLayout.STRIDE
        )
        ctx.enqueue_memset(descs, 0)
        var nodes = ctx.enqueue_create_buffer[.float32](1)
        var leaves = ctx.enqueue_create_buffer[.float32](1)
        return Self(descs^, nodes^, leaves^, blas_count)

    @staticmethod
    def from_cpu(
        mut ctx: DeviceContext,
        host: CpuBlasSet[Self.kind, Self.node_width, Self.leaf_width],
    ) raises -> Self:
        """Adapt a borrowed CPU BLAS set into device-owned wide buffers."""
        comptime assert Self.layout == GpuBvhLayout.WIDE

        # CPU triangle leaves omit the two padding planes used by the GPU's
        # float4-aligned edge layout.  Repack them and rewrite per-BLAS leaf
        # offsets; nodes and sphere leaves already share their GPU layouts.
        var upload_descs = List[UInt32](length=len(host.descs), fill=0)
        if len(host.descs) > 0:
            unsafe_memcpy(
                dest=upload_descs.unsafe_ptr(),
                src=host.descs.unsafe_ptr(),
                count=len(host.descs),
            )
        var upload_leaves: List[Float32]
        comptime if Self.kind == .TRIANGLE:
            comptime assert CPU_TRI_LEAF_PACKED_STRIDE == 10
            comptime assert TRI_LEAF_PACKED_STRIDE == 12
            var total_leaf_blocks = 0
            for blas_idx in range(host.blas_count):
                var desc = BlasDesc.load(
                    host.descs.unsafe_ptr(), UInt32(blas_idx)
                )
                total_leaf_blocks += Int(desc.leaf_block_count)
            upload_leaves = List[Float32](
                length=(
                    total_leaf_blocks * Self.leaf_width * TRI_LEAF_PACKED_STRIDE
                ),
                fill=0.0,
            )
            var gpu_leaf_base = 0
            for blas_idx in range(host.blas_count):
                var desc = BlasDesc.load(
                    host.descs.unsafe_ptr(), UInt32(blas_idx)
                )
                var cpu_leaf_base = Int(desc.leaf_f32_base)
                desc.leaf_f32_base = UInt32(gpu_leaf_base)
                desc.store(upload_descs.unsafe_ptr(), blas_idx)
                for block_idx in range(Int(desc.leaf_block_count)):
                    var cpu_block_base = (
                        cpu_leaf_base
                        + block_idx
                        * Self.leaf_width
                        * CPU_TRI_LEAF_PACKED_STRIDE
                    )
                    var gpu_block_base = (
                        gpu_leaf_base
                        + block_idx * Self.leaf_width * TRI_LEAF_PACKED_STRIDE
                    )
                    # v0, primitive id, and e1 occupy planes 0 through 6 in
                    # both layouts.
                    comptime for plane in range(7):
                        unsafe_memcpy(
                            dest=upload_leaves.unsafe_ptr().unsafe_offset(
                                gpu_block_base + plane * Self.leaf_width
                            ),
                            src=host.leaves.unsafe_ptr().unsafe_offset(
                                cpu_block_base + plane * Self.leaf_width
                            ),
                            count=Self.leaf_width,
                        )
                    # GPU planes 7 and 11 remain zero padding around e2.
                    comptime for plane in range(3):
                        unsafe_memcpy(
                            dest=upload_leaves.unsafe_ptr().unsafe_offset(
                                gpu_block_base + (plane + 8) * Self.leaf_width
                            ),
                            src=host.leaves.unsafe_ptr().unsafe_offset(
                                cpu_block_base + (plane + 7) * Self.leaf_width
                            ),
                            count=Self.leaf_width,
                        )
                gpu_leaf_base += (
                    Int(desc.leaf_block_count)
                    * Self.leaf_width
                    * TRI_LEAF_PACKED_STRIDE
                )
        else:
            upload_leaves = List[Float32](length=len(host.leaves), fill=0.0)
            if len(host.leaves) > 0:
                unsafe_memcpy(
                    dest=upload_leaves.unsafe_ptr(),
                    src=host.leaves.unsafe_ptr(),
                    count=len(host.leaves),
                )

        var descs = ctx.enqueue_create_buffer[.uint32](len(upload_descs))
        var node_count = len(host.nodes) if len(host.nodes) > 0 else 1
        var leaf_count = len(upload_leaves) if len(upload_leaves) > 0 else 1
        var nodes = ctx.enqueue_create_buffer[.float32](node_count)
        var leaves = ctx.enqueue_create_buffer[.float32](leaf_count)
        var h_descs = ctx.enqueue_create_host_buffer[.uint32](len(upload_descs))
        h_descs.enqueue_copy_from(upload_descs)
        h_descs.enqueue_copy_to(descs)
        if len(host.nodes) > 0:
            var h_nodes = ctx.enqueue_create_host_buffer[.float32](
                len(host.nodes)
            )
            h_nodes.enqueue_copy_from(host.nodes)
            h_nodes.enqueue_copy_to(nodes)
        else:
            ctx.enqueue_memset(nodes, 0)
        if len(upload_leaves) > 0:
            var h_leaves = ctx.enqueue_create_host_buffer[.float32](
                len(upload_leaves)
            )
            h_leaves.enqueue_copy_from(upload_leaves)
            h_leaves.enqueue_copy_to(leaves)
        else:
            ctx.enqueue_memset(leaves, 0)
        return Self(descs^, nodes^, leaves^, host.blas_count)
