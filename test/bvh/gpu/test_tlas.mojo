from std.benchmark import keep
from std.math import abs
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true, assert_almost_equal
from std.gpu import DeviceContext

from bajo.core import AABB, Vec3f32, Affine3f32, Point3f32
from bajo.bvh.constants import Primitive, TRACE, f32_max
from bajo.bvh.types import Sphere, Instance, Hit
from bajo.bvh.host_utils import compute_bounds
from bajo.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.bvh.gpu.tlas import GpuTriangleTlas, GpuSphereTlas
from bajo.bvh.gpu.sphere_bvh import build_sphere_blas_set
from bajo.bvh.gpu.triangle_bvh import build_triangle_blas_set
from bajo.bvh.gpu.utils import upload_camera

from test.bvh.fixtures import (
    _camera_for_bounds,
    _download_tlas_checksum,
    _make_camera_ray,
    _make_single_triangle_scene,
    _make_small_scene,
    _make_small_sphere_scene_with_bounds,
    _trace_cpu_sphere_camera,
    _trace_cpu_triangle_camera,
)


comptime WIDTH = 64
comptime HEIGHT = 48
comptime EPS = 0.05


def test_gpu_tlas_triangle_camera_single_identity_matches_cpu_blas() raises:
    var verts = _make_small_scene()
    var bounds = compute_bounds(verts)
    var camera = _camera_for_bounds(bounds, 2.0)
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
        var d_hits = ctx.enqueue_create_buffer[DType.float32](
            ray_count * Hit.STRIDE
        )
        tlas.launch_camera(
            ctx,
            blases,
            d_camera,
            d_hits,
            ray_count,
            WIDTH,
            HEIGHT,
        )
        ctx.synchronize()

        var tlas_res = _download_tlas_checksum(d_hits, ray_count)

        assert_true(abs(cpu_res[0] - tlas_res[0]) <= EPS)
        assert_true(cpu_res[1] == tlas_res[1])
        assert_true(tlas_res[2] == UInt64(0))


def test_gpu_tlas_triangle_camera_translated_single_instance_hit() raises:
    var verts = _make_single_triangle_scene()
    var bounds = compute_bounds(verts)
    var t = Point3f32(10.0, 0.0, 0.0)
    var camera = _make_camera_ray(t, Vec3f32(0.0, 0.0, 1.0))

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

        var d_hits = ctx.enqueue_create_buffer[DType.float32](Hit.STRIDE)

        tlas.launch_camera(
            ctx,
            blases,
            d_camera,
            d_hits,
            1,
            1,
            1,
        )
        ctx.synchronize()

        with d_hits.map_to_host() as hf:
            var gpu_hit = Hit.load(hf.unsafe_ptr(), 0)

            assert_almost_equal(gpu_hit.t, 2.0)
            assert_true(gpu_hit.prim == UInt32(0))
            assert_true(gpu_hit.inst == UInt32(0))


def test_gpu_tlas_sphere_camera_single_identity_matches_cpu_bruteforce() raises:
    var scene = _make_small_sphere_scene_with_bounds()
    var spheres = scene[0].copy()
    var bounds = scene[1].copy()
    var camera = _camera_for_bounds(bounds, 2.0)
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
        var d_hits = ctx.enqueue_create_buffer[DType.float32](
            ray_count * Hit.STRIDE
        )

        tlas.launch_camera(
            ctx,
            blases,
            d_camera,
            d_hits,
            ray_count,
            WIDTH,
            HEIGHT,
        )
        ctx.synchronize()

        var tlas_res = _download_tlas_checksum(d_hits, ray_count)

        assert_true(abs(cpu_res[0] - tlas_res[0]) <= EPS)
        assert_true(cpu_res[1] == tlas_res[1])
        assert_true(tlas_res[2] == UInt64(0))


def test_gpu_tlas_sphere_camera_translated_single_instance_hit() raises:
    var spheres = [Sphere(Point3f32(0.0, 0.0, 2.0), 1.0)]
    var bounds = AABB(Point3f32(-1.0, -1.0, 1.0), Point3f32(1.0, 1.0, 3.0))
    var t = Point3f32(10.0, 0.0, 0.0)
    var camera = _make_camera_ray(t, Vec3f32(0.0, 0.0, 1.0))

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

        var d_hits = ctx.enqueue_create_buffer[DType.float32](Hit.STRIDE)

        tlas.launch_camera(
            ctx,
            blases,
            d_camera,
            d_hits,
            1,
            1,
            1,
        )
        ctx.synchronize()

        with d_hits.map_to_host() as hf:
            var gpu_hit = Hit.load(hf.unsafe_ptr(), 0)

            assert_almost_equal(gpu_hit.t, 1.0)
            assert_true(gpu_hit.prim == UInt32(0))
            assert_true(gpu_hit.inst == UInt32(0))


def main() raises:
    comptime if not has_accelerator():
        raise "No Accelerator found"
    TestSuite.discover_tests[__functions_in_module()]().run()
