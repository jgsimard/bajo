from bajo.core.vec import Vec3f32
from bajo.bvh.types import Ray, Sphere, Hit
from bajo.core.intersect import intersect_ray_tri, intersect_ray_sphere
from bajo.bvh.constants import EMPTY_LANE, TRACE, f32_max


def _append_tri(mut verts: List[Vec3f32], cx: Float32, z: Float32):
    verts.append(Vec3f32(cx - 1.0, -1.0, z))
    verts.append(Vec3f32(cx + 1.0, -1.0, z))
    verts.append(Vec3f32(cx, 1.0, z))


def _append_tri(
    mut verts: List[Vec3f32],
    cx: Float32,
    cy: Float32,
    z: Float32,
):
    verts.append(Vec3f32(cx - 1.0, cy - 1.0, z))
    verts.append(Vec3f32(cx + 1.0, cy - 1.0, z))
    verts.append(Vec3f32(cx, cy + 1.0, z))


def _make_strip(count: Int, z: Float32 = 2.0) -> List[Vec3f32]:
    var verts = List[Vec3f32](capacity=count * 3)

    for i in range(count):
        var cx = Float32(i * 4 - count * 2)
        _append_tri(verts, cx, z)

    return verts^


def _make_two_depth_triangles() -> List[Vec3f32]:
    var verts = List[Vec3f32](capacity=6)
    _append_tri(verts, 0.0, 0.0, 2.0)
    _append_tri(verts, 0.0, 0.0, 4.0)
    return verts^


def _brute_triangle_trace(
    verts: List[Vec3f32],
    O: Vec3f32,
    D: Vec3f32,
) -> Hit:
    var hit = Hit.miss()

    for i in range(len(verts) / 3):
        ref v0 = verts[i * 3 + 0]
        ref v1 = verts[i * 3 + 1]
        ref v2 = verts[i * 3 + 2]

        var tri_hit = intersect_ray_tri(
            O,
            D,
            v0,
            v1,
            v2,
            f32_max,
        )

        if tri_hit.mask and tri_hit.t < hit.t:
            hit.t = tri_hit.t
            hit.u = tri_hit.u
            hit.v = tri_hit.v
            hit.prim = UInt32(i)
            hit.inst = EMPTY_LANE

    return hit


def _brute_sphere_trace(
    spheres: List[Sphere],
    O: Vec3f32,
    D: Vec3f32,
) -> Hit:
    var hit = Hit.miss()

    for i, s in enumerate(spheres):
        var sphere_hit = intersect_ray_sphere(O, D, s.center, s.radius, f32_max)
        if sphere_hit.t > 0.0 and sphere_hit.t < hit.t:
            hit.t = sphere_hit.t
            hit.u = 0.0
            hit.v = 0.0
            hit.prim = UInt32(i)
            hit.inst = EMPTY_LANE

    return hit
