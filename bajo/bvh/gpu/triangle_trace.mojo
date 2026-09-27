"""GPU triangle intersection kernels and traversal implementations."""

from std.bit import log2_floor, pop_count
from std.memory import bitcast
from max.gpu import global_idx

from bajo.bvh.constants import (
    EMPTY_LANE,
    GPU_STACK_SIZE,
    TRI_LEAF_PACKED_STRIDE,
    TRI_LEAF_VERTEX_STRIDE,
    TraceMode,
)
from bajo.bvh.gpu.camera_launch import _camera_ray, _store_camera_hit
from bajo.bvh.gpu.compressed_bounds_bvh import (
    CWBVH_TRIANGLE_WORDS,
    Cwbvh8NodeTasks,
    _intersect_cwbvh8_node_tasks,
    _intersect_cwbvh8_node_tasks_legacy,
)
from bajo.bvh.gpu.ray_launch import _load_packed_ray, _store_packed_hit
from bajo.bvh.gpu.trace import (
    GpuTraversalStats,
    trace_bounds_bvh,
    trace_bounds_bvh_with_stats,
)
from bajo.bvh.types import Hit
from bajo.core.intersect import (
    intersect_ray_tri_edges,
    intersect_ray_tri_edges_scaled,
)
from bajo.core import (
    Frame,
    GeoKind,
    Point3f32,
    Rayf32,
    Vec3f32,
    cross,
    normalize,
)


@always_inline
def _trace_triangle_bvh_camera_ray[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
](
    wide_nodes: Pointer[Float32, ImmutAnyOrigin],
    leaf_vertices: Pointer[Float32, ImmutAnyOrigin],
    root_idx: UInt32,
    ray: Rayf32[.WORLD],
) -> Hit[.WORLD]:
    # extra distance stack benchmarks positively for triangle BVH4;
    # BVH8 retains the lower-memory stack specialization.
    return trace_bounds_bvh[
        .WORLD,
        node_width,
        .CLOSEST_HIT,
        _intersect_triangle_leaf[
            .WORLD,
            leaf_width,
            .CLOSEST_HIT,
            leaf_width > node_width or leaf_width == 8,
        ],
        node_width == 4,
        node_width == 2,
        node_width == 4 and leaf_width == 2,
    ](wide_nodes, leaf_vertices, root_idx, ray)


def trace_triangle_bvh_rays_kernel[
    frame: Frame,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    mode: TraceMode = .CLOSEST_HIT,
](
    wide_nodes: Pointer[Float32, ImmutAnyOrigin],
    leaf_vertices: Pointer[Float32, ImmutAnyOrigin],
    root_idx: UInt32,
    rays: Pointer[Float32, ImmutAnyOrigin],
    hits: Pointer[Float32, MutAnyOrigin],
    ray_count: Int32,
):
    var ray_count_int = Int(ray_count)
    var ray_idx = global_idx.x
    if ray_idx >= ray_count_int:
        return

    var ray = _load_packed_ray[frame](rays, ray_count_int, ray_idx)
    var hit = trace_bounds_bvh[
        frame,
        node_width,
        mode,
        _intersect_triangle_leaf[
            frame,
            leaf_width,
            mode,
            leaf_width > node_width or leaf_width == 8,
        ],
        node_width == 4,
        mode == .CLOSEST_HIT and node_width == 2,
        mode == .CLOSEST_HIT and node_width == 4 and leaf_width == 2,
    ](wide_nodes, leaf_vertices, root_idx, ray)
    _store_packed_hit[frame](hit, hits, ray_count_int, ray_idx)


def trace_triangle_bvh_camera_instrumented_kernel[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
](
    wide_nodes: Pointer[Float32, ImmutAnyOrigin],
    leaf_vertices: Pointer[Float32, ImmutAnyOrigin],
    root_idx: UInt32,
    camera_params: Pointer[Float32, ImmutAnyOrigin],
    hits: Pointer[Float32, MutAnyOrigin],
    stats: Pointer[UInt32, MutAnyOrigin],
    ray_count: Int32,
    width_px: Int32,
    height_px: Int32,
    inv_height: Float32,
):
    var ray_count_int = Int(ray_count)
    var width_px_int = Int(width_px)
    var height_px_int = Int(height_px)
    var ray_idx = global_idx.x
    if ray_idx >= ray_count_int:
        return

    var ray = _camera_ray(
        camera_params,
        ray_count_int,
        ray_idx,
        width_px_int,
        height_px_int,
        inv_height,
    )

    var result = trace_bounds_bvh_with_stats[
        .WORLD,
        node_width,
        .CLOSEST_HIT,
        _intersect_triangle_leaf[
            .WORLD,
            leaf_width,
            .CLOSEST_HIT,
            leaf_width > node_width or leaf_width == 8,
        ],
        node_width == 4,
    ](
        wide_nodes,
        leaf_vertices,
        root_idx,
        ray,
    )
    _store_camera_hit(result.hit, hits, ray_count_int, ray_idx)
    result.stats.store(stats, ray_idx)


# AoSoA :[block][field][lane]
def _intersect_triangle_leaf[
    frame: Frame,
    width: SIMDLength,
    mode: TraceMode,
    division_free: Bool = False,
](
    leaf_vertices: ImmPointer[Float32, _],
    leaf_block_idx: UInt32,
    item_count: UInt32,
    ray: Rayf32[frame],
    mut hit: Hit[frame],
) -> Bool:
    var any_hit = False
    var block_base = Int(leaf_block_idx) * TRI_LEAF_PACKED_STRIDE * width
    var leaf_vertices_u32 = leaf_vertices.unsafe_bitcast[UInt32]()

    comptime for lane in range(width):
        var prim = leaf_vertices_u32[
            unsafe_offset=block_base + 3 * width + lane
        ]
        if prim == EMPTY_LANE:
            continue
        var v0 = Point3f32[frame](
            leaf_vertices[unsafe_offset=block_base + 0 * width + lane],
            leaf_vertices[unsafe_offset=block_base + 1 * width + lane],
            leaf_vertices[unsafe_offset=block_base + 2 * width + lane],
        )
        var e1 = Vec3f32[frame](
            leaf_vertices[unsafe_offset=block_base + 4 * width + lane],
            leaf_vertices[unsafe_offset=block_base + 5 * width + lane],
            leaf_vertices[unsafe_offset=block_base + 6 * width + lane],
        )
        var e2 = Vec3f32[frame](
            leaf_vertices[unsafe_offset=block_base + 8 * width + lane],
            leaf_vertices[unsafe_offset=block_base + 9 * width + lane],
            leaf_vertices[unsafe_offset=block_base + 10 * width + lane],
        )

        # Any-hit only needs the candidate mask, so it can always stay in
        # determinant-scaled space and return without paying for a reciprocal.
        # `division_free` remains the layout-tuned closest-hit policy.
        comptime if mode == .ANY_HIT or division_free:
            # Reject misses and farther candidates in determinant-scaled space.
            # Only a surviving closest-hit candidate pays for a reciprocal.
            var scaled_hit = intersect_ray_tri_edges_scaled(
                ray.o,
                ray.d,
                v0,
                e1,
                e2,
                hit.t,
                ray.t_min,
            )

            if scaled_hit.mask:
                comptime if mode == .ANY_HIT:
                    return True
                else:
                    var inv_det = 1.0 / scaled_hit.abs_det
                    hit.t = scaled_hit.t_scaled * inv_det
                    hit.u = scaled_hit.u_scaled * inv_det
                    hit.v = scaled_hit.v_scaled * inv_det
                    hit.prim = prim
                    hit.inst = EMPTY_LANE
                    hit.normal = normalize(cross(e1, e2)).unsafe_convert[
                        new_kind=GeoKind.NORMAL
                    ]()
                    any_hit = True
        else:
            var tri_hit = intersect_ray_tri_edges(
                ray.o,
                ray.d,
                v0,
                e1,
                e2,
                hit.t,
                ray.t_min,
            )

            if tri_hit.mask:
                comptime if mode == .ANY_HIT:
                    return True
                else:
                    hit.t = tri_hit.t
                    hit.u = tri_hit.u
                    hit.v = tri_hit.v
                    hit.prim = prim
                    hit.inst = EMPTY_LANE
                    hit.normal = normalize(cross(e1, e2)).unsafe_convert[
                        new_kind=GeoKind.NORMAL
                    ]()
                    any_hit = True
    return any_hit


@always_inline
def _intersect_cwbvh_triangle[
    frame: Frame,
    mode: TraceMode,
](
    triangles: ImmPointer[Float32, _],
    triangle_idx: UInt32,
    ray: Rayf32[frame],
    mut hit: Hit[frame],
) -> Bool:
    """Intersect one aligned e1/e2/v0 CWBVH triangle record."""
    var base = Int(triangle_idx) * CWBVH_TRIANGLE_WORDS
    var e1_record = triangles.unsafe_load[width=4, alignment=16](base)
    var e2_record = triangles.unsafe_load[width=4, alignment=16](base + 4)
    var v0_record = triangles.unsafe_load[width=4, alignment=16](base + 8)
    var e1 = Vec3f32[frame](
        e1_record[0],
        e1_record[1],
        e1_record[2],
    )
    var e2 = Vec3f32[frame](
        e2_record[0],
        e2_record[1],
        e2_record[2],
    )
    var v0 = Point3f32[frame](
        v0_record[0],
        v0_record[1],
        v0_record[2],
    )
    var scaled_hit = intersect_ray_tri_edges_scaled(
        ray.o, ray.d, v0, e1, e2, hit.t, ray.t_min
    )
    if not scaled_hit.mask:
        return False
    comptime if mode == .ANY_HIT:
        return True
    else:
        var prim = bitcast[.uint32](v0_record)[3]
        var inv_det = 1.0 / scaled_hit.abs_det
        hit.t = scaled_hit.t_scaled * inv_det
        hit.u = scaled_hit.u_scaled * inv_det
        hit.v = scaled_hit.v_scaled * inv_det
        hit.prim = prim
        hit.inst = EMPTY_LANE
        hit.normal = normalize(cross(e1, e2)).unsafe_convert[
            new_kind=GeoKind.NORMAL
        ]()
        return True


@always_inline
def _intersect_cwbvh_indexed_triangle[
    frame: Frame,
    mode: TraceMode,
](
    primitive_ids: ImmPointer[UInt32, _],
    vertices: ImmPointer[Float32, _],
    triangle_idx: UInt32,
    ray: Rayf32[frame],
    mut hit: Hit[frame],
) -> Bool:
    var prim = primitive_ids[unsafe_offset=Int(triangle_idx)]
    var base = Int(prim) * TRI_LEAF_VERTEX_STRIDE
    var v0 = Point3f32[frame](
        vertices[unsafe_offset=base + 0],
        vertices[unsafe_offset=base + 1],
        vertices[unsafe_offset=base + 2],
    )
    var e1 = Vec3f32[frame](
        vertices[unsafe_offset=base + 3] - v0.x,
        vertices[unsafe_offset=base + 4] - v0.y,
        vertices[unsafe_offset=base + 5] - v0.z,
    )
    var e2 = Vec3f32[frame](
        vertices[unsafe_offset=base + 6] - v0.x,
        vertices[unsafe_offset=base + 7] - v0.y,
        vertices[unsafe_offset=base + 8] - v0.z,
    )
    var scaled_hit = intersect_ray_tri_edges_scaled(
        ray.o, ray.d, v0, e1, e2, hit.t, ray.t_min
    )
    if not scaled_hit.mask:
        return False
    comptime if mode == .ANY_HIT:
        return True
    else:
        var inv_det = 1.0 / scaled_hit.abs_det
        hit.t = scaled_hit.t_scaled * inv_det
        hit.u = scaled_hit.u_scaled * inv_det
        hit.v = scaled_hit.v_scaled * inv_det
        hit.prim = prim
        hit.inst = EMPTY_LANE
        hit.normal = normalize(cross(e1, e2)).unsafe_convert[
            new_kind=GeoKind.NORMAL
        ]()
        return True


@always_inline
def _trace_cwbvh8_triangles_impl[
    frame: Frame,
    mode: TraceMode,
    packed_decode: Bool = True,
    max_leaf_size: Int = 3,
    stack_capacity: Int = GPU_STACK_SIZE,
    indexed_triangles: Bool = False,
](
    nodes: ImmPointer[Float32, _],
    triangles: ImmPointer[Float32, _],
    primitive_ids: ImmPointer[UInt32, _],
    vertices: ImmPointer[Float32, _],
    root_idx: UInt32,
    ray: Rayf32[frame],
) -> Hit[frame]:
    """Traverse CWBVH8 with compressed node/triangle task masks."""
    var hit = Hit[frame].miss(ray.t_max)
    # One packed task encourages a single 64-bit local-memory transaction for
    # each deferred node group instead of separate base and mask accesses.
    comptime assert stack_capacity > 0 and stack_capacity <= GPU_STACK_SIZE
    var stack = Array[UInt64, stack_capacity](uninitialized=True)
    var stack_ptr = 0

    var sign_code = UInt32(0)
    if ray.d.x < 0.0:
        sign_code |= UInt32(4)
    if ray.d.y < 0.0:
        sign_code |= UInt32(2)
    if ray.d.z < 0.0:
        sign_code |= UInt32(1)
    var octant_inverse = UInt32(7) - sign_code
    var ray_rcp = ray.reciprocal_direction[1]()

    # The synthetic root task uses the same compact group representation as
    # internal children. Its relative index is always zero.
    var node_group_base = root_idx
    var node_group_mask = UInt32(1) << UInt32(31)
    var triangle_group_base: UInt32
    var triangle_group_mask: UInt32

    while True:
        if node_group_mask > UInt32(0x00FFFFFF):
            var group_imask = node_group_mask
            var child_bit = Int(log2_floor(node_group_mask))
            node_group_mask &= ~(UInt32(1) << UInt32(child_bit))
            if node_group_mask > UInt32(0x00FFFFFF):
                debug_assert["safe", _use_compiler_assume=True](
                    stack_ptr < stack_capacity,
                    "GPU CWBVH8 traversal stack overflow",
                )
                stack[stack_ptr] = UInt64(node_group_base) | (
                    UInt64(node_group_mask) << UInt64(32)
                )
                stack_ptr += 1

            var slot = UInt32(child_bit - 24) ^ octant_inverse
            var slots_before = (UInt32(1) << slot) - UInt32(1)
            var relative = UInt32(pop_count(group_imask & slots_before))
            var node_idx = node_group_base + relative
            var node_t_max = hit.t
            comptime if mode == .ANY_HIT:
                node_t_max = ray.t_max
            var tasks: Cwbvh8NodeTasks
            comptime if packed_decode:
                tasks = _intersect_cwbvh8_node_tasks[frame, max_leaf_size](
                    nodes,
                    node_idx,
                    ray,
                    ray_rcp.x,
                    ray_rcp.y,
                    ray_rcp.z,
                    node_t_max,
                    octant_inverse,
                )
            else:
                tasks = _intersect_cwbvh8_node_tasks_legacy[frame](
                    nodes,
                    node_idx,
                    ray,
                    ray_rcp.x,
                    ray_rcp.y,
                    ray_rcp.z,
                    node_t_max,
                    octant_inverse,
                )
            node_group_base = tasks.child_base
            node_group_mask = tasks.node_group_mask
            triangle_group_base = tasks.triangle_base
            triangle_group_mask = tasks.triangle_group_mask
        else:
            triangle_group_base = node_group_base
            triangle_group_mask = node_group_mask
            node_group_mask = UInt32(0)

        while triangle_group_mask != 0:
            var triangle_bit = Int(log2_floor(triangle_group_mask))
            triangle_group_mask &= ~(UInt32(1) << UInt32(triangle_bit))
            var triangle_hit: Bool
            comptime if indexed_triangles:
                triangle_hit = _intersect_cwbvh_indexed_triangle[frame, mode](
                    primitive_ids,
                    vertices,
                    triangle_group_base + UInt32(triangle_bit),
                    ray,
                    hit,
                )
            else:
                triangle_hit = _intersect_cwbvh_triangle[frame, mode](
                    triangles,
                    triangle_group_base + UInt32(triangle_bit),
                    ray,
                    hit,
                )
            comptime if mode == .ANY_HIT:
                if triangle_hit:
                    return Hit[frame].shadow_hit()

        if node_group_mask <= UInt32(0x00FFFFFF):
            if stack_ptr == 0:
                break
            stack_ptr -= 1
            var task = stack[stack_ptr]
            node_group_base = UInt32(task)
            node_group_mask = UInt32(task >> UInt64(32))

    return hit


@always_inline
def trace_cwbvh8_triangles[
    frame: Frame,
    mode: TraceMode,
    packed_decode: Bool = True,
    max_leaf_size: Int = 3,
    stack_capacity: Int = GPU_STACK_SIZE,
](
    nodes: ImmPointer[Float32, _],
    triangles: ImmPointer[Float32, _],
    root_idx: UInt32,
    ray: Rayf32[frame],
) -> Hit[frame]:
    return _trace_cwbvh8_triangles_impl[
        frame, mode, packed_decode, max_leaf_size, stack_capacity, False
    ](
        nodes,
        triangles,
        triangles.unsafe_bitcast[UInt32](),
        triangles,
        root_idx,
        ray,
    )


@always_inline
def trace_cwbvh8_indexed_triangles[
    frame: Frame,
    mode: TraceMode,
    max_leaf_size: Int = 3,
    stack_capacity: Int = GPU_STACK_SIZE,
](
    nodes: ImmPointer[Float32, _],
    primitive_ids: ImmPointer[UInt32, _],
    vertices: ImmPointer[Float32, _],
    root_idx: UInt32,
    ray: Rayf32[frame],
) -> Hit[frame]:
    return _trace_cwbvh8_triangles_impl[
        frame, mode, True, max_leaf_size, stack_capacity, True
    ](
        nodes,
        vertices,
        primitive_ids,
        vertices,
        root_idx,
        ray,
    )
