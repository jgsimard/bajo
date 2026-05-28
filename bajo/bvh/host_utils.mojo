from std.gpu import DeviceBuffer
from std.gpu.host import DeviceContext

from bajo.core.aabb import AABB
from bajo.core.vec import Vec3f32, vmin, vmax, cross, length, normalize
from bajo.bvh.types import Ray
from bajo.bvh.constants import f32_max


def compute_bounds(verts: List[Vec3f32]) -> AABB:
    var bounds = AABB.invalid()
    for vert in verts:
        bounds.grow(vert)
    return bounds


def hit_t_for_checksum(t: Float32) -> Float64:
    if t < f32_max:
        return Float64(t)
    return 0.0
