"""Analytic indexed meshes with outward orientation and shared vertices."""
from bajo.core import Point3f32
from bajo.bvh.spatial.mesh import SpatialMesh


def tetrahedron() raises -> SpatialMesh[.WORLD]:
    var points: List[Point3f32[.WORLD]] = [
        Point3f32[.WORLD](0.0),
        Point3f32[.WORLD](1.0, 0.0, 0.0),
        Point3f32[.WORLD](0.0, 1.0, 0.0),
        Point3f32[.WORLD](0.0, 0.0, 1.0),
    ]
    var indices: List[UInt32] = [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]
    var groups: List[UInt32] = [7, 7, 8, 8]
    return SpatialMesh(points, indices, groups)
