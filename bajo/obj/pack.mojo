from bajo.core import Point3f32
from bajo.obj import read_obj, triangulated_indices


def pack_obj_triangles(path: String) raises -> List[Point3f32]:
    var mesh = read_obj(path)
    var idx = triangulated_indices(mesh)

    var out = List[Point3f32](capacity=len(idx))
    for id in idx:
        var p = Int(id.p)
        var base = p * 3
        out.append(
            Point3f32(
                mesh.positions[base + 0],
                mesh.positions[base + 1],
                mesh.positions[base + 2],
            )
        )

    return out^
