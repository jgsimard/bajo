from bajo.core import AABB, Vec3f32, Point3f32, Frame
from bajo.bvh.types import Sphere


def triangle_bounds[
    frame: Frame
](v0: Point3f32[frame], v1: Point3f32[frame], v2: Point3f32[frame]) -> AABB[
    frame
]:
    var bounds = AABB[frame].invalid()
    bounds.grow(v0, v1, v2)
    return bounds


def compute_bounds[frame: Frame](verts: List[Point3f32[frame]]) -> AABB[frame]:
    var bounds = AABB[frame].invalid()
    for vert in verts:
        bounds.grow(vert)
    return bounds


def sphere_bounds[frame: Frame](spheres: List[Sphere[frame]]) -> AABB[frame]:
    var bounds = AABB[frame].invalid()
    for s in spheres:
        bounds.grow(s.bounds())
    return bounds
