from std.benchmark import keep
from std.math import abs
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true, assert_almost_equal
from max.gpu.host import DeviceContext

from bajo.core import (
    AABB,
    Vec3f32,
    Affine3f32,
    Point3f32,
    Rayf32,
)
from bajo.bvh.constants import f32_max
from bajo.bvh.gpu import GpuBlasSet, GpuBvhLayout
from bajo.bvh.types import Hit, Instance, Sphere
from bajo.bvh.host_utils import compute_bounds, sphere_bounds
from bajo.bvh.cpu.tlas import CpuTlas
from bajo.bvh.cpu.blas_set import (
    build_cpu_sphere_blas_set,
    build_cpu_triangle_blas_set,
    trace_blas_set,
)
from bajo.bvh.gpu.tlas import build_gpu_tlas
from bajo.bvh.gpu.sphere_bvh import build_gpu_sphere_blas_set
from bajo.bvh.gpu.triangle_bvh import build_gpu_triangle_blas_set
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
    var local_verts = _make_small_scene[.LOCAL]()
    var world_verts = _make_small_scene[.WORLD]()
    var local_bounds = compute_bounds(local_verts)
    var world_bounds = compute_bounds(world_verts)
    var camera = _camera_for_bounds(world_bounds, 2.0)
    var cpu_blases = build_cpu_triangle_blas_set[4, 4, .LBVH, .WORLD](
        [world_verts^]
    )
    var cpu_res = _trace_cpu_triangle_camera[4](
        cpu_blases, camera, WIDTH, HEIGHT
    )

    var instances: List = [
        Instance(
            Affine3f32[.LOCAL, .WORLD].identity(),
            UInt32(0),
            local_bounds,
            .TRIANGLE,
        )
    ]

    with DeviceContext() as ctx:
        var blases = build_gpu_triangle_blas_set[4](ctx, [local_verts^])
        var tlas = build_gpu_tlas[.TRIANGLE, 4, 4](ctx, instances)
        var d_camera = upload_camera(ctx, camera)

        var ray_count = WIDTH * HEIGHT
        var d_hits = ctx.enqueue_create_buffer[.float32](ray_count * Hit.STRIDE)
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

        var tlas_res = _download_tlas_checksum[.WORLD](d_hits, ray_count)

        assert_true(abs(cpu_res[0] - tlas_res[0]) <= EPS)
        assert_true(cpu_res[1] == tlas_res[1])
        assert_true(tlas_res[2] == UInt64(0))


def test_gpu_tlas_triangle_camera_translated_single_instance_hit() raises:
    var verts = _make_single_triangle_scene[.LOCAL]()
    var bounds = compute_bounds(verts)
    var t = Point3f32[.WORLD](10.0, 0.0, 0.0)
    var camera = _make_camera_ray(t, Vec3f32[.WORLD](0.0, 0.0, 1.0))

    var instances: List = [
        Instance(
            Affine3f32[.LOCAL, .WORLD].from_translation(t),
            UInt32(0),
            bounds,
            .TRIANGLE,
        )
    ]

    with DeviceContext() as ctx:
        var blases = build_gpu_triangle_blas_set[4](ctx, [verts^])
        var tlas = build_gpu_tlas[.TRIANGLE, 4, 4](ctx, instances)
        var d_camera = upload_camera(ctx, camera)

        var d_hits = ctx.enqueue_create_buffer[.float32](Hit.STRIDE)

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
            var gpu_hit = Hit[.WORLD].load(
                Span(unsafe_ptr=hf.unsafe_ptr(), length=len(hf)), 0
            )

            assert_almost_equal(gpu_hit.t, 2.0)
            assert_true(gpu_hit.prim == UInt32(0))
            assert_true(gpu_hit.inst == UInt32(0))
            assert_almost_equal(gpu_hit.normal.x, 0.0)
            assert_almost_equal(gpu_hit.normal.y, 0.0)
            assert_almost_equal(gpu_hit.normal.z, 1.0)


def test_host_owned_blas_set_upload_preserves_gpu_traversal() raises:
    """CPU triangle leaves are adapted to the padded GPU leaf layout."""
    var near_verts = _make_single_triangle_scene[.LOCAL]()
    var far_verts = List[Point3f32[.LOCAL]](capacity=6)
    for vertex in near_verts:
        far_verts.append(Point3f32[.LOCAL](vertex.x, vertex.y, vertex.z + 4.0))
    for vertex in near_verts:
        far_verts.append(
            Point3f32[.LOCAL](vertex.x + 10.0, vertex.y, vertex.z + 4.0)
        )
    var bounds = compute_bounds(far_verts)
    var camera = _make_camera_ray(
        Point3f32[.WORLD](0.0, 0.0, 0.0),
        Vec3f32[.WORLD](0.0, 0.0, 1.0),
    )
    var instances: List = [
        Instance(
            Affine3f32[.LOCAL, .WORLD].identity(),
            UInt32(1),
            bounds,
            .TRIANGLE,
        )
    ]

    with DeviceContext() as ctx:
        var host = build_cpu_triangle_blas_set[4]([near_verts^, far_verts^])
        var cpu_hit = trace_blas_set(
            host,
            UInt32(1),
            Rayf32[.LOCAL](
                Point3f32[.LOCAL](0.0, 0.0, 0.0),
                Vec3f32[.LOCAL](0.0, 0.0, 1.0),
            ),
        )
        assert_almost_equal(cpu_hit.t, 6.0)
        assert_true(cpu_hit.prim == UInt32(0))
        var any_hit = trace_blas_set[4, 4, .ANY_HIT](
            host,
            UInt32(1),
            Rayf32[.LOCAL](
                Point3f32[.LOCAL](0.0, 0.0, 0.0),
                Vec3f32[.LOCAL](0.0, 0.0, 1.0),
            ),
        )
        assert_true(any_hit.is_occluded())
        var cpu_tlas = CpuTlas[4](instances)
        var cpu_tlas_hit = cpu_tlas.trace_blases(
            Rayf32[.WORLD](
                Point3f32[.WORLD](0.0, 0.0, 0.0),
                Vec3f32[.WORLD](0.0, 0.0, 1.0),
            ),
            host,
        )
        assert_almost_equal(cpu_tlas_hit.t, 6.0)
        assert_true(cpu_tlas_hit.inst == UInt32(0))
        var uploaded = GpuBlasSet[.TRIANGLE, GpuBvhLayout.WIDE, 4].from_cpu(
            ctx, host
        )
        var tlas = build_gpu_tlas[.TRIANGLE, 4, 4](ctx, instances)
        var d_camera = upload_camera(ctx, camera)
        var d_hits = ctx.enqueue_create_buffer[.float32](Hit.STRIDE)
        tlas.launch_camera(ctx, uploaded, d_camera, d_hits, 1, 1, 1)
        ctx.synchronize()

        with d_hits.map_to_host() as hf:
            var hit = Hit[.WORLD].load(
                Span(unsafe_ptr=hf.unsafe_ptr(), length=len(hf)), 0
            )
            assert_almost_equal(hit.t, 6.0)
            assert_true(hit.prim == UInt32(0))
            assert_true(hit.inst == UInt32(0))


def test_gpu_tlas_sphere_camera_single_identity_matches_cpu_bruteforce() raises:
    var local_scene = _make_small_sphere_scene_with_bounds[.LOCAL]()
    var world_scene = _make_small_sphere_scene_with_bounds[.WORLD]()
    var spheres = local_scene[0].copy()
    var bounds = local_scene[1].copy()
    var world_spheres = world_scene[0].copy()
    var world_bounds = world_scene[1].copy()
    var camera = _camera_for_bounds(world_bounds, 2.0)
    var cpu_res = _trace_cpu_sphere_camera(world_spheres, camera, WIDTH, HEIGHT)

    var instances: List = [
        Instance(
            Affine3f32[.LOCAL, .WORLD].identity(),
            UInt32(0),
            bounds,
            .SPHERE,
        )
    ]

    with DeviceContext() as ctx:
        var blases = build_gpu_sphere_blas_set[4](ctx, [spheres^])
        var tlas = build_gpu_tlas[.SPHERE, 4, 4](ctx, instances)
        var d_camera = upload_camera(ctx, camera)

        var ray_count = WIDTH * HEIGHT
        var d_hits = ctx.enqueue_create_buffer[.float32](ray_count * Hit.STRIDE)

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

        var tlas_res = _download_tlas_checksum[.WORLD](d_hits, ray_count)

        assert_true(abs(cpu_res[0] - tlas_res[0]) <= EPS)
        assert_true(cpu_res[1] == tlas_res[1])
        assert_true(tlas_res[2] == UInt64(0))


def test_cpu_sphere_blas_set_upload_preserves_gpu_traversal() raises:
    var spheres: List = [Sphere[.LOCAL](Point3f32[.LOCAL](0.0, 0.0, 2.0), 1.0)]
    var bounds = sphere_bounds(spheres)
    var instances: List = [
        Instance(
            Affine3f32[.LOCAL, .WORLD].identity(),
            UInt32(0),
            bounds,
            .SPHERE,
        )
    ]
    var camera = _make_camera_ray(
        Point3f32[.WORLD](0.0, 0.0, 0.0),
        Vec3f32[.WORLD](0.0, 0.0, 1.0),
    )
    with DeviceContext() as ctx:
        var host = build_cpu_sphere_blas_set[4]([spheres^])
        var cpu_hit = trace_blas_set(
            host,
            UInt32(0),
            Rayf32[.LOCAL](
                Point3f32[.LOCAL](0.0, 0.0, 0.0),
                Vec3f32[.LOCAL](0.0, 0.0, 1.0),
            ),
        )
        assert_almost_equal(cpu_hit.t, 1.0)
        assert_true(cpu_hit.prim == UInt32(0))
        var cpu_tlas = CpuTlas[4](instances)
        var cpu_tlas_hit = cpu_tlas.trace_blases(
            Rayf32[.WORLD](
                Point3f32[.WORLD](0.0, 0.0, 0.0),
                Vec3f32[.WORLD](0.0, 0.0, 1.0),
            ),
            host,
        )
        assert_almost_equal(cpu_tlas_hit.t, 1.0)
        assert_true(cpu_tlas_hit.inst == UInt32(0))
        var uploaded = GpuBlasSet[.SPHERE, GpuBvhLayout.WIDE, 4].from_cpu(
            ctx, host
        )
        var tlas = build_gpu_tlas[.SPHERE, 4, 4](ctx, instances)
        var d_hits = ctx.enqueue_create_buffer[.float32](Hit.STRIDE)
        tlas.launch_camera(
            ctx,
            uploaded,
            upload_camera(ctx, camera),
            d_hits,
            1,
            1,
            1,
        )
        ctx.synchronize()
        with d_hits.map_to_host() as hf:
            var hit = Hit[.WORLD].load(
                Span(unsafe_ptr=hf.unsafe_ptr(), length=len(hf)), 0
            )
            assert_almost_equal(hit.t, 1.0)
            assert_true(hit.prim == UInt32(0))
            assert_true(hit.inst == UInt32(0))


def test_gpu_tlas_sphere_camera_translated_single_instance_hit() raises:
    var spheres: List = [Sphere[.LOCAL](Point3f32[.LOCAL](0.0, 0.0, 2.0), 1.0)]
    var bounds = AABB[.LOCAL](
        Point3f32[.LOCAL](-1.0, -1.0, 1.0),
        Point3f32[.LOCAL](1.0, 1.0, 3.0),
    )
    var t = Point3f32[.WORLD](10.0, 0.0, 0.0)
    var camera = _make_camera_ray(t, Vec3f32[.WORLD](0.0, 0.0, 1.0))

    var instances: List = [
        Instance(
            Affine3f32[.LOCAL, .WORLD].from_translation(t),
            UInt32(0),
            bounds,
            .SPHERE,
        )
    ]

    with DeviceContext() as ctx:
        var blases = build_gpu_sphere_blas_set[4](ctx, [spheres^])
        var tlas = build_gpu_tlas[.SPHERE, 4, 4](ctx, instances)
        var d_camera = upload_camera(ctx, camera)

        var d_hits = ctx.enqueue_create_buffer[.float32](Hit.STRIDE)

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
            var gpu_hit = Hit[.WORLD].load(
                Span(unsafe_ptr=hf.unsafe_ptr(), length=len(hf)), 0
            )

            assert_almost_equal(gpu_hit.t, 1.0)
            assert_true(gpu_hit.prim == UInt32(0))
            assert_true(gpu_hit.inst == UInt32(0))
            assert_almost_equal(gpu_hit.normal.x, 0.0)
            assert_almost_equal(gpu_hit.normal.y, 0.0)
            assert_almost_equal(gpu_hit.normal.z, -1.0)


def test_gpu_tlas_sphere_nonuniform_scale_normal() raises:
    var spheres: List = [Sphere[.LOCAL](Point3f32[.LOCAL](0.0, 0.0, 2.0), 1.0)]
    var bounds = AABB[.LOCAL](
        Point3f32[.LOCAL](-1.0, -1.0, 1.0),
        Point3f32[.LOCAL](1.0, 1.0, 3.0),
    )
    var transform = Affine3f32[.LOCAL, .WORLD].from_scale(
        Vec3f32[.LOCAL](2.0, 1.0, 1.0)
    )
    var instances: List = [
        Instance(
            transform,
            UInt32(0),
            bounds,
            .SPHERE,
        )
    ]
    var camera = _make_camera_ray(
        Point3f32[.WORLD](1.0, 0.0, 0.0),
        Vec3f32[.WORLD](0.0, 0.0, 1.0),
    )

    with DeviceContext() as ctx:
        var blases = build_gpu_sphere_blas_set[4](ctx, [spheres^])
        var tlas = build_gpu_tlas[.SPHERE, 4, 4](ctx, instances)
        var d_camera = upload_camera(ctx, camera)
        var d_hits = ctx.enqueue_create_buffer[.float32](Hit.STRIDE)

        tlas.launch_camera(ctx, blases, d_camera, d_hits, 1, 1, 1)
        ctx.synchronize()

        with d_hits.map_to_host() as hf:
            var gpu_hit = Hit[.WORLD].load(
                Span(unsafe_ptr=hf.unsafe_ptr(), length=len(hf)), 0
            )
            assert_almost_equal(gpu_hit.t, 1.1339746, atol=1.0e-5)
            assert_almost_equal(gpu_hit.normal.x, 0.2773501, atol=1.0e-5)
            assert_almost_equal(gpu_hit.normal.y, 0.0, atol=1.0e-5)
            assert_almost_equal(gpu_hit.normal.z, -0.9607689, atol=1.0e-5)


def test_all_empty_cpu_blas_set_upload_uses_dummy_device_storage() raises:
    var empty = List[Point3f32[.LOCAL]]()
    var host = build_cpu_triangle_blas_set[4]([empty^])
    assert_true(len(host.nodes) == 0)
    assert_true(len(host.leaves) == 0)
    with DeviceContext() as ctx:
        var uploaded = GpuBlasSet[.TRIANGLE, GpuBvhLayout.WIDE, 4].from_cpu(
            ctx, host
        )
        assert_true(len(uploaded.nodes) == 1)
        assert_true(len(uploaded.leaves) == 1)
        assert_true(uploaded.blas_count == 1)


def main() raises:
    comptime if not has_accelerator():
        raise "No Accelerator found"
    TestSuite.discover_tests[__functions_in_module()]().run()
