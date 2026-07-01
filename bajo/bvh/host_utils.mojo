from bajo.core import AABB, Vec3f32
from bajo.bvh.types import Sphere


def triangle_bounds(v0: Vec3f32, v1: Vec3f32, v2: Vec3f32) -> AABB:
    var bounds = AABB.invalid()
    bounds.grow(v0, v1, v2)
    return bounds


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
