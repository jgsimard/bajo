from std.math import max
from std.gpu import DeviceBuffer

from bajo.core import AABB, Vec3f32, Point3f32
from bajo.bvh.camera import Camera
from bajo.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.bvh.types import Ray, Sphere, Hit
from bajo.core.intersect import intersect_ray_tri, intersect_ray_sphere
from bajo.bvh.constants import EMPTY_LANE, TRACE, f32_max
from bajo.bvh.host_utils import sphere_bounds


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


def _make_small_scene() -> List[Vec3f32]:
    var verts = List[Vec3f32](capacity=8 * 3)
    _append_tri(verts, -1.0, -1.0, 2.0)
    _append_tri(verts, 1.0, -1.0, 2.0)
    _append_tri(verts, -1.0, 1.0, 2.0)
    _append_tri(verts, 1.0, 1.0, 2.0)
    _append_tri(verts, -1.0, -1.0, 4.0)
    _append_tri(verts, 1.0, -1.0, 4.0)
    _append_tri(verts, -1.0, 1.0, 4.0)
    _append_tri(verts, 1.0, 1.0, 4.0)
    return verts^


def _make_single_triangle_scene() -> List[Vec3f32]:
    var verts = List[Vec3f32](capacity=3)
    _append_tri(verts, 0.0, 0.0, 2.0)
    return verts^


def _make_duplicate_centroid_scene() -> List[Vec3f32]:
    var verts = List[Vec3f32](capacity=12 * 3)
    for _ in range(12):
        verts.append(Vec3f32(-0.5, -0.5, 2.0))
        verts.append(Vec3f32(0.5, -0.5, 2.0))
        verts.append(Vec3f32(0.0, 0.5, 2.0))
    return verts^


def _make_small_sphere_scene() -> List[Sphere]:
    return [
        Sphere(Point3f32(0.0, 0.0, 2.0), 1.0),
        Sphere(Point3f32(4.0, 0.0, 4.0), 1.0),
        Sphere(Point3f32(-4.0, 0.0, 6.0), 1.0),
        Sphere(Point3f32(0.0, 4.0, 8.0), 1.0),
        Sphere(Point3f32(4.0, 4.0, 10.0), 1.0),
        Sphere(Point3f32(-4.0, 4.0, 12.0), 1.0),
        Sphere(Point3f32(4.0, -4.0, 14.0), 1.0),
        Sphere(Point3f32(-4.0, -4.0, 16.0), 1.0),
    ]


def _make_small_sphere_scene_with_bounds() -> Tuple[List[Sphere], AABB]:
    var spheres = _make_small_sphere_scene()
    var bounds = sphere_bounds(spheres)
    return (spheres^, bounds)


def _make_single_sphere_scene() -> List[Sphere]:
    return [Sphere(Point3f32(0.0, 0.0, 2.0), 1.0)]


def _make_duplicate_sphere_centroid_scene() -> List[Sphere]:
    var spheres = List[Sphere](capacity=12)
    for i in range(12):
        spheres.append(
            Sphere(Point3f32(0.0, 0.0, 2.0), 1.0 + Float32(i % 3) * 0.01)
        )
    return spheres^


def _camera_for_bounds(bounds: AABB, distance: Float32 = 2.5) -> Camera:
    var center = bounds.centroid()
    var extent = bounds.extent()
    var scene_w = max(max(extent.x, extent.y), extent.z)
    if scene_w < 1.0:
        scene_w = 1.0

    var eye = center + Vec3f32(0.0, 0.0, -scene_w * distance)
    return Camera(
        eye,
        center,
        Vec3f32(0.0, 1.0, 0.0),
        Float32(0.75),
    )


def _make_camera_ray(origin: Vec3f32, direction: Vec3f32) -> Camera:
    return Camera(
        origin,
        origin + direction,
        Vec3f32(0.0, 1.0, 0.0),
        Float32(0.75),
    )


def _make_camera_rays_and_params(
    bounds: AABB,
    width: Int,
    height: Int,
    views: Int,
) -> Tuple[List[Ray], List[Float32]]:
    var center = bounds.centroid()
    var extent = bounds.extent()

    var scene_w = max(max(extent.x, extent.y), extent.z)
    if scene_w < 1.0:
        scene_w = 1.0

    var rays = List[Ray](capacity=width * height * views)
    var params = List[Float32](capacity=views * Camera.STRIDE)

    for view in range(views):
        var view_offset = Float32(view) - Float32(views - 1) * 0.5
        var eye = center + Vec3f32(
            view_offset * scene_w * 0.30,
            extent.y * 0.20,
            -scene_w * 2.50,
        )
        var camera = Camera(
            eye,
            center,
            Vec3f32(0.0, 1.0, 0.0),
            Float32(0.75),
        )
        params.extend(camera.flatten())

        for py in range(height):
            for px in range(width):
                rays.append(camera.make_ray(px, py, width, height))

    return (rays^, params^)


def _brute_triangle_trace(
    verts: List[Vec3f32],
    O: Point3f32,
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
    O: Point3f32,
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


def _trace_cpu_triangle_bvh[
    width: SIMDSize
](mut bvh: TriangleBvh[width], rays: List[Ray]) -> Float64:
    var checksum = Float64(0.0)
    for ray in rays:
        var hit = bvh.trace[TRACE.CLOSEST_HIT](ray)
        if hit.t < f32_max:
            checksum += Float64(hit.t)
    return checksum


def _trace_cpu_triangle_camera[
    width: SIMDSize
](
    mut bvh: TriangleBvh[width], camera: Camera, cwidth: Int, cheight: Int
) -> Tuple[Float64, UInt32]:
    var checksum = Float64(0.0)
    var hits = UInt32(0)

    for py in range(cheight):
        for px in range(cwidth):
            var ray = camera.make_ray(px, py, cwidth, cheight)
            var hit = bvh.trace[TRACE.CLOSEST_HIT](ray)
            if hit.t < f32_max:
                checksum += Float64(hit.t)
                hits += 1

    return (checksum, hits)


def _trace_cpu_sphere_camera(
    spheres: List[Sphere],
    camera: Camera,
    cwidth: Int,
    cheight: Int,
) -> Tuple[Float64, UInt32]:
    var checksum = Float64(0.0)
    var hits = UInt32(0)

    for py in range(cheight):
        for px in range(cwidth):
            var ray = camera.make_ray(px, py, cwidth, cheight)
            var hit = _brute_sphere_trace(spheres, ray.o, ray.d)
            if hit.t < f32_max:
                checksum += Float64(hit.t)
                hits += 1

    return (checksum, hits)


def _trace_cpu_spheres_bruteforce(
    spheres: List[Sphere],
    rays: List[Ray],
) -> Float64:
    var checksum = Float64(0.0)

    for ray in rays:
        var brute = _brute_sphere_trace(spheres, ray.o, ray.d)
        if brute.t < f32_max:
            checksum += Float64(brute.t)
    return checksum


def _download_hit_checksum(
    hits_f32: DeviceBuffer[DType.float32],
    ray_count: Int,
) raises -> Tuple[Float64, UInt32]:
    var checksum = Float64(0.0)
    var hit_count = UInt32(0)

    with hits_f32.map_to_host() as h:
        for i in range(ray_count):
            var base = i * Hit.STRIDE
            var t = h[base + Hit.T]
            if t < f32_max:
                checksum += Float64(t)
                hit_count += 1

    return (checksum, hit_count)


def _download_tlas_checksum(
    d_hits: DeviceBuffer[DType.float32],
    ray_count: Int,
) raises -> Tuple[Float64, UInt32, UInt64]:
    var checksum = Float64(0.0)
    var hits = UInt32(0)
    var inst_checksum = UInt64(0)

    with d_hits.map_to_host() as hf:
        var gpu_hits_ptr = hf.unsafe_ptr()
        for i in range(ray_count):
            var gpu_hit = Hit.load(gpu_hits_ptr, i)
            var t = gpu_hit.t
            if t < f32_max:
                checksum += Float64(t)
                hits += 1

            var inst = gpu_hit.inst
            if inst != EMPTY_LANE:
                inst_checksum += UInt64(inst)

    return (checksum, hits, inst_checksum)
