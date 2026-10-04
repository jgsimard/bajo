"""Compile-time-selected traversal for typed GPU BLAS storage."""

from bajo.bvh.constants import TraceMode, WideNode
from bajo.bvh.gpu.blas_storage import GpuBvhLayout
from bajo.bvh.gpu.trace import GpuLeafFn, trace_bounds_bvh
from bajo.bvh.gpu.triangle_trace import trace_cwbvh8_triangles
from bajo.bvh.types import BlasDesc, Hit
from bajo.bvh.wide_meta import _wide_node_base
from bajo.core import Frame, Point3f32, Rayf32
from bajo.core.intersect import intersect_ray_aabb_rcp


@no_inline
def _trace_compressed_blas_separate[
    frame: Frame,
    mode: TraceMode,
](
    nodes: ImmPointer[Float32, _],
    leaves: ImmPointer[Float32, _],
    root_idx: UInt32,
    ray: Rayf32[frame],
) -> Hit[frame]:
    """Keep compressed traversal separate from a larger GPU shading kernel."""
    return trace_cwbvh8_triangles[frame, mode](nodes, leaves, root_idx, ray)


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
    separate_traversal: Bool = False,
) -> Hit[frame]:
    """Select storage and an optional GPU call boundary for compressed trees."""
    comptime if layout.compressed:
        if separate_traversal:
            return _trace_compressed_blas_separate[frame, mode](
                nodes, leaves, root_idx, ray
            )
        return trace_cwbvh8_triangles[frame, mode](nodes, leaves, root_idx, ray)
    return trace_bounds_bvh[
        frame,
        node_width,
        mode,
        leaf_fn,
        distance_aware,
        compact_bvh2,
    ](nodes, leaves, root_idx, ray)


@always_inline
def trace_gpu_blas_descriptor[
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
    descriptor: BlasDesc,
    ray: Rayf32[frame],
) -> Hit[frame]:
    """Intersect a sole wide leaf without the general traversal loop.

    GPU-built and uploaded wide BLASes pack their sole leaf at local block
    zero, in the root's first child lane. Keep the existing bounds rejection
    before intersecting primitives, including its reciprocal-direction
    policy. Compressed layouts use their existing traversal.
    """
    comptime if layout == .WIDE:
        if descriptor.leaf_block_count == UInt32(1):
            var hit = Hit[frame].miss(ray.t_max)
            var base = _wide_node_base[node_width](descriptor.root_idx)
            var bmin = Point3f32[frame](
                nodes[unsafe_offset=base + WideNode.MIN_X * node_width],
                nodes[unsafe_offset=base + WideNode.MIN_Y * node_width],
                nodes[unsafe_offset=base + WideNode.MIN_Z * node_width],
            )
            var bmax = Point3f32[frame](
                nodes[unsafe_offset=base + WideNode.MAX_X * node_width],
                nodes[unsafe_offset=base + WideNode.MAX_Y * node_width],
                nodes[unsafe_offset=base + WideNode.MAX_Z * node_width],
            )
            if not intersect_ray_aabb_rcp(
                ray.o, ray.reciprocal_direction(), bmin, bmax, ray.t_max
            ).mask:
                return hit
            if leaf_fn(leaves, UInt32(0), descriptor.prim_count, ray, hit):
                comptime if mode == .ANY_HIT:
                    return Hit[frame].shadow_hit()
            return hit
    return trace_gpu_blas[
        frame,
        node_width,
        mode,
        leaf_fn,
        layout,
        distance_aware,
        compact_bvh2,
    ](nodes, leaves, descriptor.root_idx, ray)
