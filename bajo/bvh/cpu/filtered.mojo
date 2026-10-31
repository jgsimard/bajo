"""Triangle BVH traversal with acceptance before closest/any-hit reduction."""

from bajo.bvh.constants import EMPTY_LANE, TraceMode
from bajo.bvh.filter import TriangleHitFilter
from bajo.bvh.types import BlasDesc, Hit
from bajo.bvh.cpu.blas_storage import CpuBlasSet
from bajo.bvh.cpu.blas_set import _load_packed_triangle_leaf
from bajo.bvh.cpu.trace import trace_packed_bounds_bvh
from bajo.core import (
    Frame,
    Point3,
    Point3f32,
    Rayf32,
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

    @always_inline
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
