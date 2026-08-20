from bajo.core import AABB, Vec3f32, Point3f32, Frame
from bajo.bvh.types import Sphere


def compute_bounds[
    frame: Frame
](verts: ImmSpan[Point3f32[frame], _]) -> AABB[frame]:
    var bounds = AABB[frame].invalid()
    for vert in verts:
        bounds.grow(vert)
    return bounds


def sphere_bounds[
    frame: Frame
](spheres: ImmSpan[Sphere[frame], _]) -> AABB[frame]:
    var bounds = AABB[frame].invalid()
    for s in spheres:
        bounds.grow(s.bounds())
    return bounds
