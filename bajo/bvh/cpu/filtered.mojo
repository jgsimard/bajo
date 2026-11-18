"""Triangle BVH traversal with acceptance before closest/any-hit reduction."""

from bajo.bvh.constants import EMPTY_LANE, TraceMode
from bajo.bvh.filter import TriangleHitFilter
from bajo.bvh.types import BlasDesc, Hit
from bajo.bvh.cpu.blas_storage import CpuBlasSet
from bajo.bvh.cpu.blas_set import _load_packed_triangle_leaf
from bajo.bvh.cpu.packet import trace_packet_stack_bounds_bvh
from bajo.bvh.cpu.trace import trace_packed_bounds_bvh
from bajo.core import (
    Normal3,
    Frame,
    Point3,
    Point3f32,
    Rayf32,
    Ray,
    Vec3,
    Vec3f32,
    cross,
    normalize,
)
from bajo.core.intersect import intersect_ray_tri_edges


def trace_filtered_triangle_blas[
    frame: Frame,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    mode: TraceMode,
    Filter: TriangleHitFilter,
](
    blases: CpuBlasSet[.TRIANGLE, node_width, leaf_width],
    blas_idx: UInt32,
    ray: Rayf32[frame],
    ref filter: Filter,
) -> Hit[frame]:
    var desc = BlasDesc.load(blases.descs.unsafe_ptr(), blas_idx)
    if desc.prim_count == 0:
        return Hit[frame].miss(ray.t_max)
    var view = blases._view[frame](desc)

    def leaf_fn(
        local_ray: Rayf32[frame],
        O: Point3[.float32, frame, leaf_width],
        D: Vec3[.float32, frame, leaf_width],
        a: SIMD[.float32, leaf_width],
        inv_a: SIMD[.float32, leaf_width],
        block_idx: UInt32,
        mut hit: Hit[frame],
    ) {imm} -> Bool:
        var block = _load_packed_triangle_leaf[frame, leaf_width](
            view.leaves, block_idx
        )
        var found = False
        for lane in range(Int(leaf_width)):
            var primitive = block.prim_indices[lane]
            if primitive == EMPTY_LANE:
                continue
            var v0 = Point3f32[frame](
                block.v0.x[lane], block.v0.y[lane], block.v0.z[lane]
            )
            var e1 = Vec3f32[frame](
                block.e1.x[lane], block.e1.y[lane], block.e1.z[lane]
            )
            var e2 = Vec3f32[frame](
                block.e2.x[lane], block.e2.y[lane], block.e2.z[lane]
            )
            var candidate = intersect_ray_tri_edges(
                local_ray.o, local_ray.d, v0, e1, e2, hit.t, local_ray.t_min
            )
            if candidate.mask and filter.accept(
                primitive, candidate.u, candidate.v
            ):
                comptime if mode == .ANY_HIT:
                    return True
                hit.t = candidate.t
                hit.u = candidate.u
                hit.v = candidate.v
                hit.prim = primitive
                hit.normal = normalize(cross(e1, e2)).unsafe_convert[
                    new_kind=.NORMAL
                ]()
                found = True
        return found

    return trace_packed_bounds_bvh[frame, node_width, leaf_width, mode](
        view.nodes, ray, leaf_fn
    )


def trace_filtered_triangle_blas_packet[
    frame: Frame,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    mode: TraceMode,
    length: SIMDLength,
    Filter: TriangleHitFilter,
](
    blases: CpuBlasSet[.TRIANGLE, node_width, leaf_width],
    blas_idx: UInt32,
    rays: Ray[.float32, frame, length],
    ref filter: Filter,
    valid: SIMD[.bool, length] = SIMD[.bool, length](fill=True),
) -> Hit[frame, length]:
    """Apply alpha to SIMD candidates before reducing either hit policy."""
    var desc = BlasDesc.load(blases.descs.unsafe_ptr(), blas_idx)
    var hit = Hit[frame, length].miss(rays.t_max)
    if desc.prim_count == 0 or not valid.reduce_or():
        return hit
    var view = blases._view[frame](desc)

    def leaf_fn(
        active: SIMD[.bool, length],
        block_idx: UInt32,
        mut result: Hit[frame, length],
    ) {imm}:
        var block = _load_packed_triangle_leaf[frame, leaf_width](
            view.leaves, block_idx
        )
        for lane in range(Int(leaf_width)):
            var primitive = block.prim_indices[lane]
            if primitive == EMPTY_LANE:
                continue
            var live = active
            comptime if mode == .ANY_HIT:
                live &= result.t.ne(0)
            if not live.reduce_or():
                break
            var v0 = Point3[.float32, frame, length](
                block.v0.x[lane], block.v0.y[lane], block.v0.z[lane]
            )
            var e1 = Vec3[.float32, frame, length](
                block.e1.x[lane], block.e1.y[lane], block.e1.z[lane]
            )
            var e2 = Vec3[.float32, frame, length](
                block.e2.x[lane], block.e2.y[lane], block.e2.z[lane]
            )
            var candidate = intersect_ray_tri_edges(
                rays.o, rays.d, v0, e1, e2, result.t, rays.t_min
            )
            var accepted = live & candidate.mask
            if not accepted.reduce_or():
                continue
            accepted &= filter.accept_packet(
                primitive, candidate.u, candidate.v, accepted
            )
            comptime if mode == .ANY_HIT:
                result.t = accepted.select(Float32(0), result.t)
            else:
                result.t = accepted.select(candidate.t, result.t)
                result.u = accepted.select(candidate.u, result.u)
                result.v = accepted.select(candidate.v, result.v)
                result.prim = accepted.select(primitive, result.prim)
                result.normal = Normal3.select(
                    accepted,
                    normalize(cross(e1, e2)).unsafe_convert[new_kind=.NORMAL](),
                    result.normal,
                )

    trace_packet_stack_bounds_bvh[
        frame=frame,
        bounds_width=node_width,
        length=length,
        packed_meta=True,
        any_hit=mode == .ANY_HIT,
    ](
        view.nodes,
        rays,
        rays.reciprocal_direction(),
        valid,
        hit,
        leaf_fn,
        lambda (
            _active: SIMD[.bool, length],
            _child: UInt32,
            mut _hit: Hit[frame, length],
        ): None,
        lambda (_child: UInt32): None,
    )
    return hit
