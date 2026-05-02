from std.gpu import DeviceBuffer, thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from std.math import abs, sqrt
from std.utils.numerics import max_finite

from bajo.core.bvh.gpu.constants import LBVH_LEAF_FLAG, LBVH_INDEX_MASK
from bajo.core.bvh.gpu.lbvh import GpuLBVH, GPU_LBVH_BLOCK_SIZE
from bajo.core.bvh.gpu.tlas import (
    GpuTlasLayout,
    GPU_TLAS_LEAF_FLAG,
    GPU_TLAS_NODE_META_STRIDE,
    GPU_TLAS_NODE_BOUNDS_STRIDE,
    GPU_TLAS_INSTANCE_META_STRIDE,
    GPU_TLAS_TRANSFORM_STRIDE,
)
from bajo.core.bvh.gpu.utils import _blocks_for
from bajo.core.bvh.types import RayFlat, Hit
from bajo.core.intersect import intersect_ray_aabb, intersect_ray_tri


comptime GPU_TLAS_TRAVERSAL_STACK_SIZE = 64
comptime GPU_TLAS_MISS = UInt32(0xFFFFFFFF)
comptime GPU_TLAS_BLAS_NODE_META_STRIDE = 4
comptime GPU_TLAS_BLAS_NODE_LEFT = 1
comptime GPU_TLAS_BLAS_NODE_RIGHT = 2
comptime GPU_TLAS_BLAS_NODE_BOUNDS_STRIDE = 12
comptime GPU_TLAS_BLAS_BOUNDS_LEFT = 0
comptime GPU_TLAS_BLAS_BOUNDS_RIGHT = 6
comptime GPU_TLAS_INF_T = max_finite[DType.float32]()


@always_inline
def _safe_rcp(x: Float32) -> Float32:
    if abs(x) <= 1.0e-20:
        if x < 0.0:
            return -GPU_TLAS_INF_T
        return GPU_TLAS_INF_T
    return Float32(1.0) / x


@always_inline
def _load_buffer_ray(
    rays: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    ray_idx: Int,
) -> RayFlat:
    var ray_base = ray_idx * 10
    return RayFlat(
        rays[ray_base + 0],
        rays[ray_base + 1],
        rays[ray_base + 2],
        rays[ray_base + 3],
        rays[ray_base + 4],
        rays[ray_base + 5],
        rays[ray_base + 6],
        rays[ray_base + 7],
        rays[ray_base + 8],
        rays[ray_base + 9],
    )


@always_inline
def _write_tlas_primary_result(
    hits_f32: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    hits_u32: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
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
def _blas_node_base(node_idx: UInt32) -> Int:
    return Int(node_idx) * GPU_TLAS_BLAS_NODE_META_STRIDE


@always_inline
def _blas_bounds_base(node_idx: UInt32) -> Int:
    return Int(node_idx) * GPU_TLAS_BLAS_NODE_BOUNDS_STRIDE


@always_inline
def _blas_node_left(
    node_meta: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    node_idx: UInt32,
) -> UInt32:
    return UInt32(
        node_meta[_blas_node_base(node_idx) + GPU_TLAS_BLAS_NODE_LEFT]
    )


@always_inline
def _blas_node_right(
    node_meta: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    node_idx: UInt32,
) -> UInt32:
    return UInt32(
        node_meta[_blas_node_base(node_idx) + GPU_TLAS_BLAS_NODE_RIGHT]
    )


@always_inline
def _intersect_blas_child_bounds[
    child_bounds_offset: Int
](
    node_bounds: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    node_idx: UInt32,
    ray: RayFlat,
    t_max: Float32,
) -> Tuple[Bool, Float32]:
    var b = _blas_bounds_base(node_idx) + child_bounds_offset
    return intersect_ray_aabb(
        ray.ox,
        ray.oy,
        ray.oz,
        ray.rdx,
        ray.rdy,
        ray.rdz,
        node_bounds[b + 0],
        node_bounds[b + 1],
        node_bounds[b + 2],
        node_bounds[b + 3],
        node_bounds[b + 4],
        node_bounds[b + 5],
        t_max,
    )


@always_inline
def _trace_blas_lbvh_ray(
    vertices: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    sorted_prim_ids: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    node_meta: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    node_bounds: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    ray: RayFlat,
    root_idx: UInt32,
) -> Hit:
    var best_t = ray.t_max
    var best_u = Float32(0.0)
    var best_v = Float32(0.0)
    var best_prim = GPU_TLAS_MISS

    var stack = InlineArray[UInt32, GPU_TLAS_TRAVERSAL_STACK_SIZE](fill=0)
    var stack_ptr = 0
    var current = root_idx

    while True:
        if (current & LBVH_LEAF_FLAG) != 0:
            var leaf_idx = current & LBVH_INDEX_MASK
            var prim_idx = UInt32(sorted_prim_ids[Int(leaf_idx)])

            var tri_hit = intersect_ray_tri(
                vertices,
                prim_idx,
                ray.ox,
                ray.oy,
                ray.oz,
                ray.dx,
                ray.dy,
                ray.dz,
                best_t,
            )

            if tri_hit.mask[0]:
                best_t = tri_hit.t
                best_u = tri_hit.u
                best_v = tri_hit.v
                best_prim = prim_idx

            if stack_ptr == 0:
                break

            stack_ptr -= 1
            current = stack[stack_ptr]
            continue

        var node_idx = current & LBVH_INDEX_MASK
        var left = _blas_node_left(node_meta, node_idx)
        var right = _blas_node_right(node_meta, node_idx)

        var left_hit = _intersect_blas_child_bounds[GPU_TLAS_BLAS_BOUNDS_LEFT](
            node_bounds,
            node_idx,
            ray,
            best_t,
        )
        var right_hit = _intersect_blas_child_bounds[
            GPU_TLAS_BLAS_BOUNDS_RIGHT
        ](
            node_bounds,
            node_idx,
            ray,
            best_t,
        )

        var hit_left = left_hit[0]
        var hit_right = right_hit[0]
        var dist_left = left_hit[1]
        var dist_right = right_hit[1]

        if not hit_left and not hit_right:
            if stack_ptr == 0:
                break
            stack_ptr -= 1
            current = stack[stack_ptr]
        elif hit_left and not hit_right:
            current = left
        elif not hit_left and hit_right:
            current = right
        else:
            var near = left
            var far = right
            if dist_left > dist_right:
                near = right
                far = left

            if stack_ptr < GPU_TLAS_TRAVERSAL_STACK_SIZE:
                stack[stack_ptr] = far
                stack_ptr += 1

            current = near

    return Hit(best_t, best_u, best_v, best_prim, UInt32(0))


@always_inline
def _tlas_node_base(node_idx: UInt32) -> Int:
    return Int(node_idx) * GPU_TLAS_NODE_META_STRIDE


@always_inline
def _tlas_bounds_base(node_idx: UInt32) -> Int:
    return Int(node_idx) * GPU_TLAS_NODE_BOUNDS_STRIDE


@always_inline
def _tlas_node_left_first(
    node_meta: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    node_idx: UInt32,
) -> UInt32:
    return UInt32(node_meta[_tlas_node_base(node_idx) + 0])


@always_inline
def _tlas_node_count(
    node_meta: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    node_idx: UInt32,
) -> UInt32:
    return UInt32(node_meta[_tlas_node_base(node_idx) + 1])


@always_inline
def _tlas_node_flag(
    node_meta: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    node_idx: UInt32,
) -> UInt32:
    return UInt32(node_meta[_tlas_node_base(node_idx) + 3])


@always_inline
def _intersect_tlas_node_bounds(
    node_bounds: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    node_idx: UInt32,
    ray: RayFlat,
    t_max: Float32,
) -> Tuple[Bool, Float32]:
    var b = _tlas_bounds_base(node_idx)
    return intersect_ray_aabb(
        ray.ox,
        ray.oy,
        ray.oz,
        ray.rdx,
        ray.rdy,
        ray.rdz,
        node_bounds[b + 0],
        node_bounds[b + 1],
        node_bounds[b + 2],
        node_bounds[b + 3],
        node_bounds[b + 4],
        node_bounds[b + 5],
        t_max,
    )


@always_inline
def _inst_inv_base(inst_idx: UInt32) -> Int:
    return Int(inst_idx) * GPU_TLAS_TRANSFORM_STRIDE


@always_inline
def _inst_meta_base(inst_idx: UInt32) -> Int:
    return Int(inst_idx) * GPU_TLAS_INSTANCE_META_STRIDE


@always_inline
def _transform_point_flat(
    m: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    base: Int,
    x: Float32,
    y: Float32,
    z: Float32,
) -> Tuple[Float32, Float32, Float32]:
    return (
        m[base + 0] * x + m[base + 1] * y + m[base + 2] * z + m[base + 3],
        m[base + 4] * x + m[base + 5] * y + m[base + 6] * z + m[base + 7],
        m[base + 8] * x + m[base + 9] * y + m[base + 10] * z + m[base + 11],
    )


@always_inline
def _transform_vector_flat(
    m: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    base: Int,
    x: Float32,
    y: Float32,
    z: Float32,
) -> Tuple[Float32, Float32, Float32]:
    return (
        m[base + 0] * x + m[base + 1] * y + m[base + 2] * z,
        m[base + 4] * x + m[base + 5] * y + m[base + 6] * z,
        m[base + 8] * x + m[base + 9] * y + m[base + 10] * z,
    )


@always_inline
def _make_local_ray(
    inv_transform: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    inst_idx: UInt32,
    ray: RayFlat,
    t_max: Float32,
) -> RayFlat:
    var base = _inst_inv_base(inst_idx)
    var o = _transform_point_flat(inv_transform, base, ray.ox, ray.oy, ray.oz)
    var d = _transform_vector_flat(inv_transform, base, ray.dx, ray.dy, ray.dz)

    return RayFlat(
        o[0],
        o[1],
        o[2],
        d[0],
        d[1],
        d[2],
        _safe_rcp(d[0]),
        _safe_rcp(d[1]),
        _safe_rcp(d[2]),
        t_max,
    )


@always_inline
def _trace_tlas_lbvh_ray(
    blas_vertices: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    blas_sorted_prim_ids: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    blas_node_meta: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    blas_node_bounds: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    tlas_node_meta: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    tlas_node_bounds: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    tlas_inst_indices: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    tlas_inst_meta: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    tlas_inst_inv_transform: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
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

                # Phase C supports one uploaded BLAS. Keep the BLAS index in the
                # layout, but skip non-zero instances until the multi-BLAS path.
                if blas_idx == 0:
                    var local_ray = _make_local_ray(
                        tlas_inst_inv_transform,
                        inst_idx,
                        ray,
                        best_hit.t,
                    )
                    var local_hit = _trace_blas_lbvh_ray(
                        blas_vertices,
                        blas_sorted_prim_ids,
                        blas_node_meta,
                        blas_node_bounds,
                        local_ray,
                        blas_root_idx,
                    )

                    if local_hit.t < best_hit.t:
                        best_hit = local_hit
                        best_inst = UInt32(
                            tlas_inst_meta[_inst_meta_base(inst_idx) + 1]
                        )

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

        var hit_left = left_hit[0]
        var hit_right = right_hit[0]
        var dist_left = left_hit[1]
        var dist_right = right_hit[1]

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
    blas_vertices: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    blas_sorted_prim_ids: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    blas_node_meta: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    blas_node_bounds: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    tlas_node_meta: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    tlas_node_bounds: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    tlas_inst_indices: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    tlas_inst_meta: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    tlas_inst_inv_transform: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    rays: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    hits_f32: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    hits_u32: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    ray_count: Int,
    blas_root_idx: UInt32,
    tlas_node_count: Int,
):
    var ray_idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if ray_idx >= ray_count:
        return

    var ray = _load_buffer_ray(rays, ray_idx)
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
    ctx.enqueue_function[
        trace_tlas_lbvh_gpu_primary_kernel,
        trace_tlas_lbvh_gpu_primary_kernel,
    ](
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
        grid_dim=_blocks_for[GPU_LBVH_BLOCK_SIZE](ray_count),
        block_dim=GPU_LBVH_BLOCK_SIZE,
    )


# -----------------------------------------------------------------------------
# Phase D: GPU-generated TLAS camera rays and GPU normal shading.
#
# These helpers keep the Phase C uploaded-ray path intact and add the minimum
# camera/render path needed by examples/instanced_lbvh_normals.mojo.
# -----------------------------------------------------------------------------

comptime GPU_TLAS_CAMERA_PARAM_STRIDE = 12
comptime GPU_TLAS_CAMERA_ORIGIN = 0
comptime GPU_TLAS_CAMERA_FORWARD = 3
comptime GPU_TLAS_CAMERA_RIGHT = 6
comptime GPU_TLAS_CAMERA_UP = 9


@always_inline
def _normalize3(
    x: Float32,
    y: Float32,
    z: Float32,
) -> Tuple[Float32, Float32, Float32]:
    var len2 = x * x + y * y + z * z
    if len2 <= 1.0e-20:
        return (Float32(0.0), Float32(0.0), Float32(0.0))
    var inv_len = Float32(1.0) / sqrt(len2)
    return (x * inv_len, y * inv_len, z * inv_len)


@always_inline
def _make_tlas_camera_ray(
    camera_params: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    ray_idx: Int,
    width: Int,
    height: Int,
) -> RayFlat:
    var pixels_per_view = width * height
    var view_idx = ray_idx // pixels_per_view
    var local_idx = ray_idx - view_idx * pixels_per_view
    var px_i = local_idx % width
    var py_i = local_idx // width

    var cam_base = view_idx * GPU_TLAS_CAMERA_PARAM_STRIDE

    var ox = camera_params[cam_base + GPU_TLAS_CAMERA_ORIGIN + 0]
    var oy = camera_params[cam_base + GPU_TLAS_CAMERA_ORIGIN + 1]
    var oz = camera_params[cam_base + GPU_TLAS_CAMERA_ORIGIN + 2]

    var fx = camera_params[cam_base + GPU_TLAS_CAMERA_FORWARD + 0]
    var fy = camera_params[cam_base + GPU_TLAS_CAMERA_FORWARD + 1]
    var fz = camera_params[cam_base + GPU_TLAS_CAMERA_FORWARD + 2]

    var rx = camera_params[cam_base + GPU_TLAS_CAMERA_RIGHT + 0]
    var ry = camera_params[cam_base + GPU_TLAS_CAMERA_RIGHT + 1]
    var rz = camera_params[cam_base + GPU_TLAS_CAMERA_RIGHT + 2]

    var ux = camera_params[cam_base + GPU_TLAS_CAMERA_UP + 0]
    var uy = camera_params[cam_base + GPU_TLAS_CAMERA_UP + 1]
    var uz = camera_params[cam_base + GPU_TLAS_CAMERA_UP + 2]

    var aspect = Float32(width) / Float32(height)
    var fov_scale = Float32(0.25)

    var sx = ((Float32(px_i) + 0.5) / Float32(width)) * 2.0 - 1.0
    var sy = 1.0 - ((Float32(py_i) + 0.5) / Float32(height)) * 2.0

    var dir_x = fx + rx * (sx * aspect * fov_scale) + ux * (sy * fov_scale)
    var dir_y = fy + ry * (sx * aspect * fov_scale) + uy * (sy * fov_scale)
    var dir_z = fz + rz * (sx * aspect * fov_scale) + uz * (sy * fov_scale)

    var nd = _normalize3(dir_x, dir_y, dir_z)
    var dx = nd[0]
    var dy = nd[1]
    var dz = nd[2]

    return RayFlat(
        ox,
        oy,
        oz,
        dx,
        dy,
        dz,
        _safe_rcp(dx),
        _safe_rcp(dy),
        _safe_rcp(dz),
        GPU_TLAS_INF_T,
    )


def trace_tlas_lbvh_gpu_camera_kernel(
    blas_vertices: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    blas_sorted_prim_ids: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    blas_node_meta: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    blas_node_bounds: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    tlas_node_meta: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    tlas_node_bounds: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    tlas_inst_indices: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    tlas_inst_meta: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    tlas_inst_inv_transform: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    camera_params: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    hits_f32: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    hits_u32: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
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
    var view_idx = ray_idx // pixels_per_view
    if view_idx >= views:
        _write_tlas_primary_result(
            hits_f32,
            hits_u32,
            ray_idx,
            Hit(GPU_TLAS_INF_T, 0.0, 0.0, GPU_TLAS_MISS, UInt32(0)),
            GPU_TLAS_MISS,
        )
        return

    var ray = _make_tlas_camera_ray(camera_params, ray_idx, width, height)
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
    ctx.enqueue_function[
        trace_tlas_lbvh_gpu_camera_kernel,
        trace_tlas_lbvh_gpu_camera_kernel,
    ](
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
        grid_dim=_blocks_for[GPU_LBVH_BLOCK_SIZE](ray_count),
        block_dim=GPU_LBVH_BLOCK_SIZE,
    )


@always_inline
def _load_vertex(
    vertices: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    prim_idx: UInt32,
    corner: Int,
) -> Tuple[Float32, Float32, Float32]:
    var base = Int(prim_idx) * 9 + corner * 3
    return (vertices[base + 0], vertices[base + 1], vertices[base + 2])


@always_inline
def _cross3(
    ax: Float32,
    ay: Float32,
    az: Float32,
    bx: Float32,
    by: Float32,
    bz: Float32,
) -> Tuple[Float32, Float32, Float32]:
    return (
        ay * bz - az * by,
        az * bx - ax * bz,
        ax * by - ay * bx,
    )


@always_inline
def _normal_byte(x: Float32) -> UInt32:
    var y = x * 0.5 + 0.5
    if y < 0.0:
        y = 0.0
    if y > 1.0:
        y = 1.0
    return UInt32(y * 255.0 + 0.5)


@always_inline
def _pack_rgb(r: UInt32, g: UInt32, b: UInt32) -> UInt32:
    return (r << 16) | (g << 8) | b


@always_inline
def _shade_background(ray_idx: Int, width: Int, height: Int) -> UInt32:
    var local_idx = ray_idx % (width * height)
    var py_i = local_idx // width
    var y = Float32(py_i) / Float32(height)
    var c = UInt32(24.0 + (1.0 - y) * 32.0)
    return _pack_rgb(c, c + 8, c + 20)


def shade_tlas_normals_kernel(
    vertices: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    tlas_inst_transform: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    hits_u32: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    out_rgb: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
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

    var e1x = v1[0] - v0[0]
    var e1y = v1[1] - v0[1]
    var e1z = v1[2] - v0[2]
    var e2x = v2[0] - v0[0]
    var e2y = v2[1] - v0[1]
    var e2z = v2[2] - v0[2]

    var n = _cross3(e1x, e1y, e1z, e2x, e2y, e2z)
    var ln = _normalize3(n[0], n[1], n[2])

    # Phase D uses rigid/uniform-scale instance transforms. For general affine
    # transforms this should become the inverse-transpose normal transform.
    var base = _inst_inv_base(inst)
    var wn = _transform_vector_flat(
        tlas_inst_transform,
        base,
        ln[0],
        ln[1],
        ln[2],
    )
    var nn = _normalize3(wn[0], wn[1], wn[2])

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
    ctx.enqueue_function[
        shade_tlas_normals_kernel,
        shade_tlas_normals_kernel,
    ](
        blas.vertices.unsafe_ptr(),
        tlas.inst_transform.unsafe_ptr(),
        d_hits_u32.unsafe_ptr(),
        d_rgb.unsafe_ptr(),
        pixel_count,
        width,
        height,
        grid_dim=_blocks_for[GPU_LBVH_BLOCK_SIZE](pixel_count),
        block_dim=GPU_LBVH_BLOCK_SIZE,
    )
