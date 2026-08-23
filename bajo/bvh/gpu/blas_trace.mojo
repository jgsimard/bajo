"""Compile-time-selected traversal for typed GPU BLAS storage."""

from bajo.bvh.constants import TraceMode
from bajo.bvh.gpu.blas_storage import GpuBvhLayout
from bajo.bvh.gpu.trace import GpuLeafFn, trace_bounds_bvh
from bajo.bvh.gpu.triangle_bvh import trace_cwbvh8_triangles
from bajo.bvh.types import Hit
from bajo.core import Frame, Rayf32


@always_inline
def trace_gpu_blas[
    frame: Frame,
    node_width: SIMDLength,
    mode: TraceMode,
    leaf_fn: GpuLeafFn[frame],
    layout: GpuBvhLayout = .WIDE,
    distance_aware: Bool = False,
    compact_bvh2: Bool = False,
](
    nodes: ImmPointer[Float32, _],
    leaves: ImmPointer[Float32, _],
    root_idx: UInt32,
    ray: Rayf32[frame],
) -> Hit[frame]:
    """Select the storage-specific traversal while keeping one call ABI."""
    comptime if layout.compressed:
        return trace_cwbvh8_triangles[frame, mode](
            nodes, leaves, root_idx, ray
        )
    return trace_bounds_bvh[
        frame,
        node_width,
        mode,
        leaf_fn,
        distance_aware,
        compact_bvh2,
    ](nodes, leaves, root_idx, ray)
