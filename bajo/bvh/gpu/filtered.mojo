"""Filtered triangle traversal over existing wide and compressed GPU storage."""

from bajo.bvh.constants import EMPTY_LANE, TRI_LEAF_PACKED_STRIDE, TraceMode
from bajo.bvh.filter import TriangleHitFilter
from bajo.bvh.types import Hit
from bajo.bvh.gpu.blas_storage import GpuBvhLayout
from bajo.bvh.gpu.trace import trace_bounds_bvh_state
from bajo.bvh.gpu.triangle_trace import trace_cwbvh8_triangles_filtered
from bajo.core import (
    Frame,
    GeoKind,
    Point3f32,
    Rayf32,
    Vec3f32,
    cross,
    normalize,
)
from bajo.core.intersect import intersect_ray_tri_edges_scaled


@fieldwise_init
struct _FilteredTriangleLeaf[Filter: TriangleHitFilter](Copyable):
    var leaves: ImmPointer[Float32, ImmUntrackedOrigin]
    var filter: Self.Filter


@always_inline
def _filtered_triangle_leaf[
    frame: Frame, width: SIMDLength, mode: TraceMode, Filter: TriangleHitFilter
](
    state: _FilteredTriangleLeaf[Filter],
    block: UInt32,
    count: UInt32,
    ray: Rayf32[frame],
    mut hit: Hit[frame],
) -> Bool:
    var found = False
    var base = Int(block) * TRI_LEAF_PACKED_STRIDE * width
    var words = state.leaves.unsafe_bitcast[UInt32]()
    for lane in range(Int(width)):
        var primitive = words[unsafe_offset=base + 3 * width + lane]
        if primitive == EMPTY_LANE:
            continue
        var v0 = Point3f32[frame](
            state.leaves[unsafe_offset=base + lane],
            state.leaves[unsafe_offset=base + width + lane],
            state.leaves[unsafe_offset=base + 2 * width + lane],
        )
        var e1 = Vec3f32[frame](
            state.leaves[unsafe_offset=base + 4 * width + lane],
            state.leaves[unsafe_offset=base + 5 * width + lane],
            state.leaves[unsafe_offset=base + 6 * width + lane],
        )
        var e2 = Vec3f32[frame](
            state.leaves[unsafe_offset=base + 8 * width + lane],
            state.leaves[unsafe_offset=base + 9 * width + lane],
            state.leaves[unsafe_offset=base + 10 * width + lane],
        )
        var candidate = intersect_ray_tri_edges_scaled(
            ray.o, ray.d, v0, e1, e2, hit.t, ray.t_min
        )
        if not candidate.mask:
            continue
        var inv_det = 1.0 / candidate.abs_det
        var u = candidate.u_scaled * inv_det
        var v = candidate.v_scaled * inv_det
        if not state.filter.accept(primitive, u, v):
            continue
        comptime if mode == .ANY_HIT:
            hit = Hit[frame].shadow_hit()
            return True
        hit.t = candidate.t_scaled * inv_det
        hit.u = u
        hit.v = v
        hit.prim = primitive
        hit.inst = EMPTY_LANE
        hit.normal = normalize(cross(e1, e2)).unsafe_convert[
            new_kind=GeoKind.NORMAL
        ]()
        found = True
    return found


@always_inline
def trace_gpu_blas_filtered[
    frame: Frame,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    mode: TraceMode,
    layout: GpuBvhLayout,
    Filter: TriangleHitFilter,
](
    nodes: ImmPointer[Float32, _],
    leaves: ImmPointer[Float32, _],
    root: UInt32,
    ray: Rayf32[frame],
    ref filter: Filter,
) -> Hit[frame]:
    comptime if layout == .CWBVH8:
        return trace_cwbvh8_triangles_filtered[frame, mode](
            nodes, leaves, root, ray, filter
        )
    else:
        var state = _FilteredTriangleLeaf(
            leaves.unsafe_origin_cast[ImmUntrackedOrigin](), filter.copy()
        )
        return trace_bounds_bvh_state[
            frame,
            node_width,
            mode,
            _FilteredTriangleLeaf[Filter],
            _filtered_triangle_leaf[frame, leaf_width, mode, Filter],
            node_width == 2,
        ](nodes, state, root, ray)
