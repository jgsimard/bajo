from bajo.bvh.types import Ray, Hit
from bajo.bvh.constants import (
    f32_max,
    EMPTY_LANE,
    TRACE_CLOSEST_HIT,
    TRACE_ANY_HIT,
)
from bajo.core.vec import Vec3f32, normalize


comptime CAMERA_PARAM_STRIDE = 13
comptime CAMERA_ORIGIN = 0
comptime CAMERA_FORWARD = 3
comptime CAMERA_RIGHT = 6
comptime CAMERA_UP = 9
comptime CAMERA_FOV = 12


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
        0.0,
        nd,
        f32_max,
    )
