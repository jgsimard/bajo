from bajo.core import Vec3f32


def pack_obj_triangles(path: String) raises -> List[Vec3f32]:
    var mesh = read_obj(path)
    var idx = triangulated_indices(mesh)

    var out = List[Vec3f32](capacity=len(idx))
    for id in idx:
        var p = Int(id.p)
        var base = p * 3
        out.append(
            Vec3f32(
                mesh.positions[base + 0],
                mesh.positions[base + 1],
                mesh.positions[base + 2],
            )
        )

    return out^
