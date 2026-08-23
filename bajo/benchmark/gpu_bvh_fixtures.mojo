"""Shared inputs for GPU BVH diagnostics."""

from bajo.core import Point3f32, AABB


def flatten_triangle_bounds(
    vertices: List[Point3f32[.WORLD]],
) -> Tuple[List[Float32], List[UInt32]]:
    var triangle_count = len(vertices) / 3
    var bounds = List[Float32](capacity=triangle_count * 6)
    var payloads = List[UInt32](capacity=triangle_count)
    for i in range(triangle_count):
        var box = AABB(
            vertices[i * 3], vertices[i * 3 + 1], vertices[i * 3 + 2]
        )
        bounds.append(box._min.x)
        bounds.append(box._min.y)
        bounds.append(box._min.z)
        bounds.append(box._max.x)
        bounds.append(box._max.y)
        bounds.append(box._max.z)
        payloads.append(UInt32(i))
    return (bounds^, payloads^)
