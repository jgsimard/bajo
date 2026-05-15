from std.gpu import DeviceBuffer, thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from std.math import abs, sqrt, clamp, ceildiv
from std.utils.numerics import max_finite

from bajo.core.bvh.gpu.lbvh import GpuLBVH, GPU_LBVH_BLOCK_SIZE
from bajo.core.bvh.gpu.tlas import (
    GpuTlasLayout,
    GPU_TLAS_LEAF_FLAG,
    GPU_TLAS_NODE_META_STRIDE,
    GPU_TLAS_NODE_BOUNDS_STRIDE,
    GPU_TLAS_INSTANCE_META_STRIDE,
    GPU_TLAS_TRANSFORM_STRIDE,
)
from bajo.core.bvh.gpu.kernels import (
    _trace_lbvh_ray,
    _make_camera_ray,
)
from bajo.core.bvh.types import RayFlat, Hit
from bajo.core.intersect import (
    intersect_ray_aabb,
    intersect_ray_tri,
    RayAabbHit,
)
from bajo.core.vec import Vec3f32, cross, normalize
from bajo.core.mat import transform_point, transform_vector


comptime GPU_TLAS_TRAVERSAL_STACK_SIZE = 64
comptime GPU_TLAS_MISS = UInt32(0xFFFFFFFF)
comptime GPU_TLAS_INF_T = max_finite[DType.float32]()


@always_inline
def _safe_rcp(x: Float32) -> Float32:
    if abs(x) <= 1.0e-20:
        if x < 0.0:
            return -GPU_TLAS_INF_T
        return GPU_TLAS_INF_T
    return Float32(1.0) / x


@always_inline
def _write_tlas_primary_result(
    hits_f32: UnsafePointer[Float32, MutAnyOrigin],
    hits_u32: UnsafePointer[UInt32, MutAnyOrigin],
    ray_idx: Int,
    hit: Hit,
    inst: UInt32,
):
    var fbase = ray_idx * 3
    hits_f32[fbase + 0] = hit.t
    hits_f32[fbase + 1] = hit.u
    hits_f32[fbase + 2] = hit.v

    var ubase = ray_idx * 2
    hits_u32[ubase + 0] = hit.prim
    hits_u32[ubase + 1] = inst


@always_inline
def _tlas_node_base(node_idx: UInt32) -> Int:
    return Int(node_idx) * GPU_TLAS_NODE_META_STRIDE


@always_inline
def _tlas_bounds_base(node_idx: UInt32) -> Int:
    return Int(node_idx) * GPU_TLAS_NODE_BOUNDS_STRIDE


@always_inline
def _tlas_node_left_first(
    node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    node_idx: UInt32,
) -> UInt32:
    return node_meta[_tlas_node_base(node_idx) + 0]


@always_inline
def _tlas_node_count(
    node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    node_idx: UInt32,
) -> UInt32:
    return node_meta[_tlas_node_base(node_idx) + 1]


@always_inline
def _tlas_node_flag(
    node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    node_idx: UInt32,
) -> UInt32:
    return node_meta[_tlas_node_base(node_idx) + 3]


@always_inline
def _intersect_tlas_node_bounds(
    node_bounds: UnsafePointer[Float32, MutAnyOrigin],
    node_idx: UInt32,
    ray: RayFlat,
    t_max: Float32,
) -> RayAabbHit[DType.float32, 1]:
    var b = _tlas_bounds_base(node_idx)
    return intersect_ray_aabb(
        ray.o,
        ray.rd,
        Vec3f32.load(node_bounds, b),
        Vec3f32.load(node_bounds, b + 3),
        t_max,
    )


@always_inline
def _inst_inv_base(inst_idx: UInt32) -> Int:
    return Int(inst_idx) * GPU_TLAS_TRANSFORM_STRIDE


@always_inline
def _inst_meta_base(inst_idx: UInt32) -> Int:
    return Int(inst_idx) * GPU_TLAS_INSTANCE_META_STRIDE


@always_inline
def _make_local_ray(
    inv_transform: UnsafePointer[Float32, MutAnyOrigin],
    inst_idx: UInt32,
    ray: RayFlat,
    t_max: Float32,
) -> RayFlat:
    var base = _inst_inv_base(inst_idx)
    var o = transform_point(inv_transform, base, ray.o)
    var d = transform_vector(inv_transform, base, ray.d)

    return RayFlat(
        o,
        d,
        Vec3f32(_safe_rcp(d[0]), _safe_rcp(d[1]), _safe_rcp(d[2])),
        t_max,
    )


@always_inline
def _trace_tlas_lbvh_ray(
    blas_vertices: UnsafePointer[Float32, MutAnyOrigin],
    blas_sorted_prim_ids: UnsafePointer[UInt32, MutAnyOrigin],
    blas_node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    blas_node_bounds: UnsafePointer[Float32, MutAnyOrigin],
    tlas_node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_node_bounds: UnsafePointer[Float32, MutAnyOrigin],
    tlas_inst_indices: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_inst_meta: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_inst_inv_transform: UnsafePointer[Float32, MutAnyOrigin],
    ray: RayFlat,
    blas_root_idx: UInt32,
    tlas_node_count: Int,
) -> Tuple[Hit, UInt32]:
    if tlas_node_count <= 0:
        return (
            Hit(GPU_TLAS_INF_T, 0.0, 0.0, GPU_TLAS_MISS, UInt32(0)),
            GPU_TLAS_MISS,
        )

    var best_hit = Hit(ray.t_max, 0.0, 0.0, GPU_TLAS_MISS, UInt32(0))
    var best_inst = GPU_TLAS_MISS

    var stack = InlineArray[UInt32, GPU_TLAS_TRAVERSAL_STACK_SIZE](fill=0)
    var stack_ptr = 0
    var node_idx = UInt32(0)

    while True:
        var flag = _tlas_node_flag(tlas_node_meta, node_idx)
        var left_first = _tlas_node_left_first(tlas_node_meta, node_idx)
        var count = _tlas_node_count(tlas_node_meta, node_idx)

        if flag == GPU_TLAS_LEAF_FLAG:
            for i in range(Int(count)):
                var inst_idx = UInt32(tlas_inst_indices[Int(left_first) + i])
                var blas_idx = UInt32(
                    tlas_inst_meta[_inst_meta_base(inst_idx) + 0]
                )

                debug_assert["safe"](
                    blas_idx == 0, "TLAS only support one blas at the moment"
                )
                if blas_idx == 0:
                    var local_ray = _make_local_ray(
                        tlas_inst_inv_transform,
                        inst_idx,
                        ray,
                        best_hit.t,
                    )
                    var local_hit = _trace_lbvh_ray(
                        blas_vertices,
                        blas_sorted_prim_ids,
                        blas_node_meta,
                        blas_node_bounds,
                        local_ray,
                        blas_root_idx,
                    )

                    if local_hit.t < best_hit.t:
                        best_hit = local_hit
                        best_inst = inst_idx

            if stack_ptr == 0:
                break
            stack_ptr -= 1
            node_idx = stack[stack_ptr]
            continue

        var left = left_first
        var right = left_first + 1

        var left_hit = _intersect_tlas_node_bounds(
            tlas_node_bounds,
            left,
            ray,
            best_hit.t,
        )
        var right_hit = _intersect_tlas_node_bounds(
            tlas_node_bounds,
            right,
            ray,
            best_hit.t,
        )

        var hit_left = left_hit.mask
        var hit_right = right_hit.mask
        var dist_left = left_hit.tmin
        var dist_right = right_hit.tmin

        if not hit_left and not hit_right:
            if stack_ptr == 0:
                break
            stack_ptr -= 1
            node_idx = stack[stack_ptr]
        elif hit_left and not hit_right:
            node_idx = left
        elif not hit_left and hit_right:
            node_idx = right
        else:
            var near = left
            var far = right
            if dist_left > dist_right:
                near = right
                far = left

            if stack_ptr < GPU_TLAS_TRAVERSAL_STACK_SIZE:
                stack[stack_ptr] = far
                stack_ptr += 1
            node_idx = near

    return (best_hit, best_inst)


def trace_tlas_lbvh_gpu_primary_kernel(
    blas_vertices: UnsafePointer[Float32, MutAnyOrigin],
    blas_sorted_prim_ids: UnsafePointer[UInt32, MutAnyOrigin],
    blas_node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    blas_node_bounds: UnsafePointer[Float32, MutAnyOrigin],
    tlas_node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_node_bounds: UnsafePointer[Float32, MutAnyOrigin],
    tlas_inst_indices: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_inst_meta: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_inst_inv_transform: UnsafePointer[Float32, MutAnyOrigin],
    rays: UnsafePointer[Float32, MutAnyOrigin],
    hits_f32: UnsafePointer[Float32, MutAnyOrigin],
    hits_u32: UnsafePointer[UInt32, MutAnyOrigin],
    ray_count: Int,
    blas_root_idx: UInt32,
    tlas_node_count: Int,
):
    var ray_idx = block_idx.x * block_dim.x + thread_idx.x
    if ray_idx >= ray_count:
        return

    var ray = RayFlat(rays, ray_idx)
    var result = _trace_tlas_lbvh_ray(
        blas_vertices,
        blas_sorted_prim_ids,
        blas_node_meta,
        blas_node_bounds,
        tlas_node_meta,
        tlas_node_bounds,
        tlas_inst_indices,
        tlas_inst_meta,
        tlas_inst_inv_transform,
        ray,
        blas_root_idx,
        tlas_node_count,
    )
    _write_tlas_primary_result(
        hits_f32,
        hits_u32,
        ray_idx,
        result[0],
        result[1],
    )


def launch_tlas_lbvh_uploaded_primary(
    ctx: DeviceContext,
    blas: GpuLBVH,
    tlas: GpuTlasLayout,
    d_rays: DeviceBuffer[DType.float32],
    d_hits_f32: DeviceBuffer[DType.float32],
    d_hits_u32: DeviceBuffer[DType.uint32],
    ray_count: Int,
) raises:
    ctx.enqueue_function[trace_tlas_lbvh_gpu_primary_kernel](
        blas.vertices.unsafe_ptr(),
        blas.values.unsafe_ptr(),
        blas.node_meta.unsafe_ptr(),
        blas.node_bounds.unsafe_ptr(),
        tlas.node_meta.unsafe_ptr(),
        tlas.node_bounds.unsafe_ptr(),
        tlas.inst_indices.unsafe_ptr(),
        tlas.inst_meta.unsafe_ptr(),
        tlas.inst_inv_transform.unsafe_ptr(),
        d_rays.unsafe_ptr(),
        d_hits_f32.unsafe_ptr(),
        d_hits_u32.unsafe_ptr(),
        ray_count,
        blas.root_idx,
        tlas.node_count,
        grid_dim=ceildiv(ray_count, GPU_LBVH_BLOCK_SIZE),
        block_dim=GPU_LBVH_BLOCK_SIZE,
    )


def trace_tlas_lbvh_gpu_camera_kernel(
    blas_vertices: UnsafePointer[Float32, MutAnyOrigin],
    blas_sorted_prim_ids: UnsafePointer[UInt32, MutAnyOrigin],
    blas_node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    blas_node_bounds: UnsafePointer[Float32, MutAnyOrigin],
    tlas_node_meta: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_node_bounds: UnsafePointer[Float32, MutAnyOrigin],
    tlas_inst_indices: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_inst_meta: UnsafePointer[UInt32, MutAnyOrigin],
    tlas_inst_inv_transform: UnsafePointer[Float32, MutAnyOrigin],
    camera_params: UnsafePointer[Float32, MutAnyOrigin],
    hits_f32: UnsafePointer[Float32, MutAnyOrigin],
    hits_u32: UnsafePointer[UInt32, MutAnyOrigin],
    ray_count: Int,
    width: Int,
    height: Int,
    views: Int,
    blas_root_idx: UInt32,
    tlas_node_count: Int,
):
    var ray_idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if ray_idx >= ray_count:
        return

    var pixels_per_view = width * height
    var view_idx = ray_idx / pixels_per_view
    if view_idx >= views:
        _write_tlas_primary_result(
            hits_f32,
            hits_u32,
            ray_idx,
            Hit(GPU_TLAS_INF_T, 0.0, 0.0, GPU_TLAS_MISS, UInt32(0)),
            GPU_TLAS_MISS,
        )
        return

    var ray = _make_camera_ray(camera_params, ray_idx, width, height)
    var result = _trace_tlas_lbvh_ray(
        blas_vertices,
        blas_sorted_prim_ids,
        blas_node_meta,
        blas_node_bounds,
        tlas_node_meta,
        tlas_node_bounds,
        tlas_inst_indices,
        tlas_inst_meta,
        tlas_inst_inv_transform,
        ray,
        blas_root_idx,
        tlas_node_count,
    )
    _write_tlas_primary_result(
        hits_f32,
        hits_u32,
        ray_idx,
        result[0],
        result[1],
    )


def launch_tlas_lbvh_camera_primary(
    ctx: DeviceContext,
    blas: GpuLBVH,
    tlas: GpuTlasLayout,
    d_camera_params: DeviceBuffer[DType.float32],
    d_hits_f32: DeviceBuffer[DType.float32],
    d_hits_u32: DeviceBuffer[DType.uint32],
    ray_count: Int,
    width: Int,
    height: Int,
    views: Int,
) raises:
    ctx.enqueue_function[trace_tlas_lbvh_gpu_camera_kernel](
        blas.vertices.unsafe_ptr(),
        blas.values.unsafe_ptr(),
        blas.node_meta.unsafe_ptr(),
        blas.node_bounds.unsafe_ptr(),
        tlas.node_meta.unsafe_ptr(),
        tlas.node_bounds.unsafe_ptr(),
        tlas.inst_indices.unsafe_ptr(),
        tlas.inst_meta.unsafe_ptr(),
        tlas.inst_inv_transform.unsafe_ptr(),
        d_camera_params.unsafe_ptr(),
        d_hits_f32.unsafe_ptr(),
        d_hits_u32.unsafe_ptr(),
        ray_count,
        width,
        height,
        views,
        blas.root_idx,
        tlas.node_count,
        grid_dim=ceildiv(ray_count, GPU_LBVH_BLOCK_SIZE),
        block_dim=GPU_LBVH_BLOCK_SIZE,
    )


@always_inline
def _load_vertex(
    vertices: UnsafePointer[Float32, MutAnyOrigin],
    prim_idx: UInt32,
    corner: Int,
) -> Vec3f32:
    var base = Int(prim_idx) * 9 + corner * 3
    return Vec3f32(vertices[base + 0], vertices[base + 1], vertices[base + 2])


@always_inline
def _normal_byte(x: Float32) -> UInt32:
    y = clamp(x * 0.5 + 0.5, 0.0, 1.0)
    return UInt32(y * 255.0 + 0.5)


@always_inline
def _pack_rgb(r: UInt32, g: UInt32, b: UInt32) -> UInt32:
    return (r << 16) | (g << 8) | b


@always_inline
def _shade_background(ray_idx: Int, width: Int, height: Int) -> UInt32:
    var local_idx = ray_idx % (width * height)
    var py_i = local_idx / width
    var y = Float32(py_i) / Float32(height)
    var c = UInt32(24.0 + (1.0 - y) * 32.0)
    return _pack_rgb(c, c + 8, c + 20)


def shade_tlas_normals_kernel(
    vertices: UnsafePointer[Float32, MutAnyOrigin],
    tlas_inst_transform: UnsafePointer[Float32, MutAnyOrigin],
    hits_u32: UnsafePointer[UInt32, MutAnyOrigin],
    out_rgb: UnsafePointer[UInt32, MutAnyOrigin],
    pixel_count: Int,
    width: Int,
    height: Int,
):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= pixel_count:
        return

    var prim = UInt32(hits_u32[i * 2 + 0])
    var inst = UInt32(hits_u32[i * 2 + 1])

    if prim == GPU_TLAS_MISS or inst == GPU_TLAS_MISS:
        out_rgb[i] = _shade_background(i, width, height)
        return

    var v0 = _load_vertex(vertices, prim, 0)
    var v1 = _load_vertex(vertices, prim, 1)
    var v2 = _load_vertex(vertices, prim, 2)

    var e1 = v1 - v0
    var e2 = v2 - v0

    var n = cross(e1, e2)
    var ln = normalize(n)

    # TODO: right now only rigid/uniform-scale instance transforms :(
    var base = _inst_inv_base(inst)
    var wn = transform_vector(
        tlas_inst_transform,
        base,
        ln,
    )
    var nn = normalize(wn)

    out_rgb[i] = _pack_rgb(
        _normal_byte(nn[0]),
        _normal_byte(nn[1]),
        _normal_byte(nn[2]),
    )


def launch_shade_tlas_normals(
    ctx: DeviceContext,
    blas: GpuLBVH,
    tlas: GpuTlasLayout,
    d_hits_u32: DeviceBuffer[DType.uint32],
    d_rgb: DeviceBuffer[DType.uint32],
    pixel_count: Int,
    width: Int,
    height: Int,
) raises:
    ctx.enqueue_function[shade_tlas_normals_kernel](
        blas.vertices.unsafe_ptr(),
        tlas.inst_transform.unsafe_ptr(),
        d_hits_u32.unsafe_ptr(),
        d_rgb.unsafe_ptr(),
        pixel_count,
        width,
        height,
        grid_dim=ceildiv(pixel_count, GPU_LBVH_BLOCK_SIZE),
        block_dim=GPU_LBVH_BLOCK_SIZE,
    )
