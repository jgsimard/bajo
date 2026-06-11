from std.gpu import DeviceBuffer
from std.gpu.host import DeviceContext

from bajo.core import AABB, Vec3f32, vmin, vmax, cross, length, normalize
from bajo.bvh.types import Ray, Sphere
from bajo.bvh.constants import f32_max


def compute_bounds(verts: List[Vec3f32]) -> AABB:
    var bounds = AABB.invalid()
    for vert in verts:
        bounds.grow(vert)
    return bounds


def sphere_bounds(spheres: List[Sphere]) -> AABB:
    var bounds = AABB.invalid()
    for s in spheres:
        bounds.grow(s.bounds())
    return bounds
