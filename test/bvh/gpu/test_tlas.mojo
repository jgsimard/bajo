from std.benchmark import keep
from std.math import abs, max
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true, assert_almost_equal
from std.gpu import DeviceBuffer, DeviceContext

from bajo.core import AABB, Vec3f32, Affine3f32
from bajo.bvh.camera import Camera
from bajo.bvh.constants import Primitive, TRACE, f32_max
from bajo.bvh.types import Ray, Sphere, Instance
from bajo.bvh.host_utils import compute_bounds
from bajo.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.bvh.gpu.tlas import GpuTriangleTlas, GpuSphereTlas
from bajo.bvh.gpu.sphere_bvh import build_sphere_blas_set
from bajo.bvh.gpu.triangle_bvh import build_triangle_blas_set
from bajo.bvh.gpu.utils import upload_camera

from test.bvh.fixtures import _append_tri, _brute_sphere_trace


comptime WIDTH = 64
comptime HEIGHT = 48
comptime EPS = 0.05
comptime MISS = UInt32(0xFFFFFFFF)


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


def _make_spheres() -> Tuple[List[Sphere], AABB]:
    var spheres = List[Sphere](capacity=4)
    var bounds = AABB.invalid()

    spheres.append(Sphere(Vec3f32(0.0, 0.0, 2.0), 1.0))
    spheres.append(Sphere(Vec3f32(4.0, 0.0, 4.0), 1.0))
    spheres.append(Sphere(Vec3f32(-4.0, 0.0, 6.0), 1.0))
    spheres.append(Sphere(Vec3f32(0.0, 4.0, 8.0), 1.0))

    for s in spheres:
        var r = s.radius
        bounds.grow(Vec3f32(s.center.x - r, s.center.y - r, s.center.z - r))
        bounds.grow(Vec3f32(s.center.x + r, s.center.y + r, s.center.z + r))

    return (spheres^, bounds)


def _camera_for_bounds(bounds: AABB) -> Camera:
    var center = bounds.centroid()
    var extent = bounds.extent()
    var scene_w = max(max(extent.x, extent.y), extent.z)
    if scene_w < 1.0:
        scene_w = 1.0

    var eye = center + Vec3f32(0.0, 0.0, -scene_w * 2.0)
    return Camera(
        eye,
        center,
        Vec3f32(0.0, 1.0, 0.0),
        Float32(0.75),
    )


def _center_ray_camera(origin: Vec3f32, direction: Vec3f32) -> Camera:
    return Camera(
        origin,
        origin + direction,
        Vec3f32(0.0, 1.0, 0.0),
        Float32(0.75),
    )


def _trace_cpu_triangle_camera[
    width: Int
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


def _download_tlas_checksum(
    hits_f32: DeviceBuffer[DType.float32],
    hits_u32: DeviceBuffer[DType.uint32],
    ray_count: Int,
) raises -> Tuple[Float64, UInt32, UInt64]:
    var checksum = Float64(0.0)
    var hits = UInt32(0)
    var inst_checksum = UInt64(0)

    with hits_f32.map_to_host() as hf:
        for i in range(ray_count):
            var t = hf[i * 3]
            if t < f32_max:
                checksum += Float64(t)
                hits += 1

    with hits_u32.map_to_host() as hu:
        for i in range(ray_count):
            var inst = UInt32(hu[i * 2 + 1])
            if inst != MISS:
                inst_checksum += UInt64(inst)

    return (checksum, hits, inst_checksum)


def test_gpu_tlas_triangle_camera_single_identity_matches_cpu_blas() raises:
    var verts = _make_small_scene()
    var bounds = compute_bounds(verts)
    var camera = _camera_for_bounds(bounds)
    var cpu_bvh = TriangleBvh[4].__init__["lbvh"](verts.copy())
    var cpu_res = _trace_cpu_triangle_camera[4](cpu_bvh, camera, WIDTH, HEIGHT)

    var instances = [
        Instance(
            Affine3f32.identity(),
            Affine3f32.identity(),
            UInt32(0),
            bounds,
            Primitive.TRIANGLE,
        )
    ]

    with DeviceContext() as ctx:
        var blases = build_triangle_blas_set[4](ctx, [verts^])
        var tlas = GpuTriangleTlas[4, 4](ctx, instances)
        var d_camera = upload_camera(ctx, camera)

        var ray_count = WIDTH * HEIGHT
        var d_hits_f32 = ctx.enqueue_create_buffer[DType.float32](ray_count * 3)
        var d_hits_u32 = ctx.enqueue_create_buffer[DType.uint32](ray_count * 2)

        tlas.launch_camera(
            ctx,
            blases,
            d_camera,
            d_hits_f32,
            d_hits_u32,
            ray_count,
            WIDTH,
            HEIGHT,
        )
        ctx.synchronize()

        var tlas_res = _download_tlas_checksum(
            d_hits_f32, d_hits_u32, ray_count
        )

        assert_true(abs(cpu_res[0] - tlas_res[0]) <= EPS)
        assert_true(cpu_res[1] == tlas_res[1])
        assert_true(tlas_res[2] == UInt64(0))


def test_gpu_tlas_triangle_camera_translated_single_instance_hit() raises:
    var verts = _make_single_triangle_scene()
    var bounds = compute_bounds(verts)
    var t = Vec3f32(10.0, 0.0, 0.0)
    var camera = _center_ray_camera(t, Vec3f32(0.0, 0.0, 1.0))

    var instances = [
        Instance(
            Affine3f32.from_translation(t),
            Affine3f32.from_translation(-t),
            UInt32(0),
            bounds,
            Primitive.TRIANGLE,
        )
    ]

    with DeviceContext() as ctx:
        var blases = build_triangle_blas_set[4](ctx, [verts^])
        var tlas = GpuTriangleTlas[4, 4](ctx, instances)
        var d_camera = upload_camera(ctx, camera)

        var d_hits_f32 = ctx.enqueue_create_buffer[DType.float32](3)
        var d_hits_u32 = ctx.enqueue_create_buffer[DType.uint32](2)

        tlas.launch_camera(
            ctx,
            blases,
            d_camera,
            d_hits_f32,
            d_hits_u32,
            1,
            1,
            1,
        )
        ctx.synchronize()

        with d_hits_f32.map_to_host() as hf:
            assert_almost_equal(hf[0], 2.0)

        with d_hits_u32.map_to_host() as hu:
            assert_true(UInt32(hu[0]) == UInt32(0))
            assert_true(UInt32(hu[1]) == UInt32(0))


def test_gpu_tlas_sphere_camera_single_identity_matches_cpu_bruteforce() raises:
    var scene = _make_spheres()
    var spheres = scene[0].copy()
    var bounds = scene[1].copy()
    var camera = _camera_for_bounds(bounds)
    var cpu_res = _trace_cpu_sphere_camera(spheres, camera, WIDTH, HEIGHT)

    var instances = [
        Instance(
            Affine3f32.identity(),
            Affine3f32.identity(),
            UInt32(0),
            bounds,
            Primitive.SPHERE,
        )
    ]

    with DeviceContext() as ctx:
        var blases = build_sphere_blas_set[4](ctx, [spheres^])
        var tlas = GpuSphereTlas[4, 4](ctx, instances)
        var d_camera = upload_camera(ctx, camera)

        var ray_count = WIDTH * HEIGHT
        var d_hits_f32 = ctx.enqueue_create_buffer[DType.float32](ray_count * 3)
        var d_hits_u32 = ctx.enqueue_create_buffer[DType.uint32](ray_count * 2)

        tlas.launch_camera(
            ctx,
            blases,
            d_camera,
            d_hits_f32,
            d_hits_u32,
            ray_count,
            WIDTH,
            HEIGHT,
        )
        ctx.synchronize()

        var tlas_res = _download_tlas_checksum(
            d_hits_f32, d_hits_u32, ray_count
        )

        assert_true(abs(cpu_res[0] - tlas_res[0]) <= EPS)
        assert_true(cpu_res[1] == tlas_res[1])
        assert_true(tlas_res[2] == UInt64(0))


def test_gpu_tlas_sphere_camera_translated_single_instance_hit() raises:
    var spheres = [Sphere(Vec3f32(0.0, 0.0, 2.0), 1.0)]
    var bounds = AABB(Vec3f32(-1.0, -1.0, 1.0), Vec3f32(1.0, 1.0, 3.0))
    var t = Vec3f32(10.0, 0.0, 0.0)
    var camera = _center_ray_camera(t, Vec3f32(0.0, 0.0, 1.0))

    var instances = [
        Instance(
            Affine3f32.from_translation(t),
            Affine3f32.from_translation(-t),
            UInt32(0),
            bounds,
            Primitive.SPHERE,
        )
    ]

    with DeviceContext() as ctx:
        var blases = build_sphere_blas_set[4](ctx, [spheres^])
        var tlas = GpuSphereTlas[4, 4](ctx, instances)
        var d_camera = upload_camera(ctx, camera)

        var d_hits_f32 = ctx.enqueue_create_buffer[DType.float32](3)
        var d_hits_u32 = ctx.enqueue_create_buffer[DType.uint32](2)

        tlas.launch_camera(
            ctx,
            blases,
            d_camera,
            d_hits_f32,
            d_hits_u32,
            1,
            1,
            1,
        )
        ctx.synchronize()

        with d_hits_f32.map_to_host() as hf:
            assert_almost_equal(hf[0], 1.0)

        with d_hits_u32.map_to_host() as hu:
            assert_true(UInt32(hu[0]) == UInt32(0))
            assert_true(UInt32(hu[1]) == UInt32(0))


def main() raises:
    comptime if not has_accelerator():
        raise "No Accelerator found"
    TestSuite.discover_tests[__functions_in_module()]().run()
