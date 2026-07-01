from bajo.core import AABB, Vec3f32
from bajo.bvh.types import Sphere


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
