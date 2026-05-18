from bajo.core.bvh.types import Ray, Hit
from bajo.core.bvh.constants import (
    _gpu_inf_t,
    _gpu_miss_prim,
    TRACE_PRIMARY_FULL,
    TRACE_PRIMARY_T,
    TRACE_SHADOW,
)
from bajo.core.vec import Vec3f32, normalize


comptime CAMERA_PARAM_STRIDE = 13
comptime CAMERA_ORIGIN = 0
comptime CAMERA_FORWARD = 3
comptime CAMERA_RIGHT = 6
comptime CAMERA_UP = 9
comptime CAMERA_FOV = 12


@always_inline
def _write_primary_full_result(
    hits_f32: UnsafePointer[Float32, MutAnyOrigin],
    hits_u32: UnsafePointer[UInt32, MutAnyOrigin],
    ray_idx: Int,
    hit: Hit,
):
    var hit_base = ray_idx * 3
    hits_f32[hit_base + 0] = hit.t
    hits_f32[hit_base + 1] = hit.u
    hits_f32[hit_base + 2] = hit.v
    hits_u32[ray_idx] = hit.prim


@always_inline
def _make_camera_ray(
    camera_params: UnsafePointer[Float32, MutAnyOrigin],
    ray_idx: Int,
    width: Int,
    height: Int,
) -> Ray:
    var pixels_per_view = width * height
    var view_idx = ray_idx / pixels_per_view
    var local_idx = ray_idx - view_idx * pixels_per_view
    var px_i = local_idx % width
    var py_i = local_idx / width

    var base = view_idx * CAMERA_PARAM_STRIDE

    var o = Vec3f32.load(camera_params, base + CAMERA_ORIGIN)
    var f = Vec3f32.load(camera_params, base + CAMERA_FORWARD)
    var r = Vec3f32.load(camera_params, base + CAMERA_RIGHT)
    var u = Vec3f32.load(camera_params, base + CAMERA_UP)
    var fov_scale = camera_params[base + CAMERA_FOV]

    var aspect = Float32(width) / Float32(height)

    var sx = ((Float32(px_i) + 0.5) / Float32(width)) * 2.0 - 1.0
    var sy = 1.0 - ((Float32(py_i) + 0.5) / Float32(height)) * 2.0

    var dir = f + r * (sx * aspect * fov_scale) + u * (sy * fov_scale)
    var nd = normalize(dir)

    return Ray(
        o,
        nd,
        Float32(0.0),
        _gpu_inf_t,
        UInt32(0xFFFFFFFF),
    )


@always_inline
def _write_camera_miss_result[
    mode: String
](
    out_f32: UnsafePointer[Float32, MutAnyOrigin],
    out_u32: UnsafePointer[UInt32, MutAnyOrigin],
    ray_idx: Int,
):
    comptime if mode == TRACE_PRIMARY_FULL:
        var hit_base = ray_idx * 3
        out_f32[hit_base + 0] = _gpu_inf_t
        out_f32[hit_base + 1] = 0.0
        out_f32[hit_base + 2] = 0.0
        out_u32[ray_idx] = _gpu_miss_prim
    elif mode == TRACE_PRIMARY_T:
        out_f32[ray_idx] = _gpu_inf_t
    else:
        out_u32[ray_idx] = UInt32(0)


@always_inline
def _write_camera_result[
    mode: String
](
    out_f32: UnsafePointer[Float32, MutAnyOrigin],
    out_u32: UnsafePointer[UInt32, MutAnyOrigin],
    ray_idx: Int,
    hit: Hit,
):
    comptime if mode == TRACE_PRIMARY_FULL:
        _write_primary_full_result(out_f32, out_u32, ray_idx, hit)
    elif mode == TRACE_PRIMARY_T:
        out_f32[ray_idx] = hit.t
    else:
        out_u32[ray_idx] = hit.occluded
