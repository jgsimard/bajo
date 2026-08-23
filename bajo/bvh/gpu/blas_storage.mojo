"""GPU ownership and upload adapter for packed BLAS storage."""

from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh.cpu.blas_storage import CpuBlasSet
from bajo.bvh.constants import Primitive
from bajo.bvh.types import BlasDescLayout


@fieldwise_init
struct GpuBvhLayout(Equatable, ImplicitlyCopyable):
    """Enum-like compile-time selector for GPU BLAS byte layout."""

    comptime WIDE = Self(False)
    comptime CWBVH8 = Self(True)

    var compressed: Bool


@fieldwise_init
struct GpuBlasSet[
    kind: Primitive,
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
        """Copy a borrowed CPU BLAS set into device-owned buffers."""
        comptime assert Self.layout == GpuBvhLayout.WIDE
        var descs = ctx.enqueue_create_buffer[.uint32](len(host.descs))
        var node_count = len(host.nodes) if len(host.nodes) > 0 else 1
        var leaf_count = len(host.leaves) if len(host.leaves) > 0 else 1
        var nodes = ctx.enqueue_create_buffer[.float32](node_count)
        var leaves = ctx.enqueue_create_buffer[.float32](leaf_count)
        var h_descs = ctx.enqueue_create_host_buffer[.uint32](
            len(host.descs)
        )
        h_descs.enqueue_copy_from(host.descs)
        h_descs.enqueue_copy_to(descs)
        if len(host.nodes) > 0:
            var h_nodes = ctx.enqueue_create_host_buffer[.float32](
                len(host.nodes)
            )
            h_nodes.enqueue_copy_from(host.nodes)
            h_nodes.enqueue_copy_to(nodes)
        else:
            ctx.enqueue_memset(nodes, 0)
        if len(host.leaves) > 0:
            var h_leaves = ctx.enqueue_create_host_buffer[.float32](
                len(host.leaves)
            )
            h_leaves.enqueue_copy_from(host.leaves)
            h_leaves.enqueue_copy_to(leaves)
        else:
            ctx.enqueue_memset(leaves, 0)
        return Self(descs^, nodes^, leaves^, host.blas_count)
