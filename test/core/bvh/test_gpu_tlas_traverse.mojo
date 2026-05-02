from std.gpu import DeviceBuffer
from std.gpu.host import DeviceContext
from std.math import abs
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal, assert_true

from bajo.core.bvh import (
    compute_bounds,
    copy_list_to_device,
    flatten_rays,
)
from bajo.core.bvh.cpu.binary_bvh import BinaryBvh
from bajo.core.bvh.cpu.tlas import BvhInstance, Tlas
from bajo.core.bvh.gpu.kernels import compute_centroid_bounds
from bajo.core.bvh.gpu.lbvh import GpuLBVH
from bajo.core.bvh.gpu.tlas import GpuTlasLayout
from bajo.core.bvh.gpu.tlas_traverse import launch_tlas_lbvh_uploaded_primary
from bajo.core.bvh.types import Ray
from bajo.core.mat import Mat44f32
from bajo.core.vec import Vec3f32


comptime GPU_TLAS_TEST_MISS = UInt32(0xFFFFFFFF)


def _translation(tx: Float32, ty: Float32, tz: Float32) -> Mat44f32:
    return Mat44f32(
        1.0,
        0.0,
        0.0,
        tx,
        0.0,
        1.0,
        0.0,
        ty,
        0.0,
        0.0,
        1.0,
        tz,
        0.0,
        0.0,
        0.0,
        1.0,
    )


def _append_tri(
    mut verts: List[Vec3f32],
    cx: Float32,
    cy: Float32,
    z: Float32,
):
    verts.append(Vec3f32(cx - 1.0, cy - 1.0, z))
    verts.append(Vec3f32(cx + 1.0, cy - 1.0, z))
    verts.append(Vec3f32(cx, cy + 1.0, z))


def _make_two_depth_triangles() -> List[Vec3f32]:
    # Two local triangles keep the GPU LBVH path on the normal internal-node
    # traversal path while all tests still expect primitive 0 as the closest hit.
    var verts = List[Vec3f32](capacity=6)
    _append_tri(verts, 0.0, 0.0, 2.0)
    _append_tri(verts, 0.0, 0.0, 4.0)
    return verts^


@always_inline
def _safe_inv_extent_axis(a: Float32, b: Float32) -> Float32:
    var e = b - a
    if abs(e) <= 1.0e-20:
        return 0.0
    return Float32(1.0) / e


def _inv_extent(cmin: Vec3f32, cmax: Vec3f32) -> Vec3f32:
    return Vec3f32(
        _safe_inv_extent_axis(cmin.x(), cmax.x()),
        _safe_inv_extent_axis(cmin.y(), cmax.y()),
        _safe_inv_extent_axis(cmin.z(), cmax.z()),
    )


def _build_tlas(mut blas: BinaryBvh, instances: List[BvhInstance]) -> Tlas:
    var tlas = Tlas(instances)
    tlas.build()
    return tlas^


def _make_identity_tlas(mut blas: BinaryBvh) -> Tlas:
    var instances = List[BvhInstance](capacity=1)
    instances.append(
        BvhInstance.from_blas(Mat44f32.identity(), Mat44f32.identity(), 0, blas)
    )
    return _build_tlas(blas, instances)


def _make_two_translated_tlas(mut blas: BinaryBvh) -> Tlas:
    var instances = List[BvhInstance](capacity=2)
    instances.append(
        BvhInstance.from_blas(
            _translation(-10.0, 0.0, 0.0),
            _translation(10.0, 0.0, 0.0),
            0,
            blas,
        )
    )
    instances.append(
        BvhInstance.from_blas(
            _translation(10.0, 0.0, 0.0),
            _translation(-10.0, 0.0, 0.0),
            0,
            blas,
        )
    )
    return _build_tlas(blas, instances)


def _make_nearest_tlas(mut blas: BinaryBvh) -> Tlas:
    var instances = List[BvhInstance](capacity=2)
    instances.append(
        BvhInstance.from_blas(
            _translation(0.0, 0.0, 3.0),
            _translation(0.0, 0.0, -3.0),
            0,
            blas,
        )
    )
    instances.append(
        BvhInstance.from_blas(Mat44f32.identity(), Mat44f32.identity(), 0, blas)
    )
    return _build_tlas(blas, instances)


def _make_grid_tlas(mut blas: BinaryBvh, count: Int = 8) -> Tlas:
    var instances = List[BvhInstance](capacity=count)
    for i in range(count):
        var x = Float32((i % 4) - 2) * 6.0
        var y = Float32(i // 4) * 4.0
        instances.append(
            BvhInstance.from_blas(
                _translation(x, y, 0.0),
                _translation(-x, -y, 0.0),
                0,
                blas,
            )
        )
    return _build_tlas(blas, instances)


def _build_gpu_blas(
    ctx: DeviceContext,
    mut gpu_blas: GpuLBVH,
    verts: List[Vec3f32],
) raises:
    var bounds = compute_bounds(verts)
    var bmin = bounds[0].copy()
    var bmax = bounds[1].copy()
    var cbounds = compute_centroid_bounds(verts)
    var cmin = cbounds[0].copy()
    var cmax = cbounds[1].copy()
    var inv = _inv_extent(cmin, cmax)

    _ = gpu_blas.build(ctx, cmin, inv)
    var validation = gpu_blas.validate(bmin, bmax)
    assert_true(validation.sorted_ok, "GPU BLAS keys sorted")
    assert_true(validation.values_ok, "GPU BLAS values valid")
    assert_true(validation.topology_ok, "GPU BLAS topology valid")
    assert_true(validation.bounds_ok, "GPU BLAS bounds valid")


def _trace_cpu_tlas(
    tlas: Tlas,
    ray: Ray,
    mut blas: BinaryBvh,
) -> Ray:
    var blases = List[BinaryBvh](capacity=1)
    blases.append(blas.copy())
    var out = ray.copy()
    tlas.traverse(out, blases.unsafe_ptr())
    return out^


def _run_gpu_tlas_trace(
    mut ctx: DeviceContext,
    gpu_blas: GpuLBVH,
    gpu_tlas: GpuTlasLayout,
    rays: List[Ray],
) raises -> Tuple[DeviceBuffer[DType.float32], DeviceBuffer[DType.uint32]]:
    var flat_rays = flatten_rays(rays)
    var d_rays = copy_list_to_device(ctx, flat_rays)
    var d_hits_f32 = ctx.enqueue_create_buffer[DType.float32](len(rays) * 3)
    var d_hits_u32 = ctx.enqueue_create_buffer[DType.uint32](len(rays) * 2)

    launch_tlas_lbvh_uploaded_primary(
        ctx,
        gpu_blas,
        gpu_tlas,
        d_rays,
        d_hits_f32,
        d_hits_u32,
        len(rays),
    )
    ctx.synchronize()

    return (d_hits_f32^, d_hits_u32^)


def _assert_gpu_matches_cpu(
    d_hits_f32: DeviceBuffer[DType.float32],
    d_hits_u32: DeviceBuffer[DType.uint32],
    ray_idx: Int,
    cpu_ray: Ray,
) raises:
    with d_hits_f32.map_to_host() as hf:
        with d_hits_u32.map_to_host() as hu:
            var fbase = ray_idx * 3
            var ubase = ray_idx * 2

            if cpu_ray.hit.t > 1.0e20:
                assert_true(hu[ubase + 0] == GPU_TLAS_TEST_MISS)
                assert_true(hu[ubase + 1] == GPU_TLAS_TEST_MISS)
            else:
                assert_almost_equal(hf[fbase + 0], cpu_ray.hit.t, atol=1.0e-4)
                assert_almost_equal(hf[fbase + 1], cpu_ray.hit.u, atol=1.0e-4)
                assert_almost_equal(hf[fbase + 2], cpu_ray.hit.v, atol=1.0e-4)
                assert_true(hu[ubase + 0] == cpu_ray.hit.prim)
                assert_true(hu[ubase + 1] == cpu_ray.hit.inst)


def test_gpu_tlas_identity_instance_matches_cpu_tlas() raises:
    comptime if has_accelerator():
        var verts = _make_two_depth_triangles()
        var cpu_blas = BinaryBvh(verts.unsafe_ptr(), UInt32(len(verts) // 3))
        cpu_blas.build["median", False]()
        var cpu_tlas = _make_identity_tlas(cpu_blas)

        var ray = Ray(Vec3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
        var cpu_ray = _trace_cpu_tlas(cpu_tlas, ray, cpu_blas)
        assert_true(cpu_ray.hit.inst == 0)
        assert_true(cpu_ray.hit.prim == 0)

        var rays = List[Ray](capacity=1)
        rays.append(ray.copy())

        with DeviceContext() as ctx:
            var gpu_blas = GpuLBVH(ctx, verts)
            _build_gpu_blas(ctx, gpu_blas, verts)
            var gpu_tlas = GpuTlasLayout(ctx, cpu_tlas)
            var hits = _run_gpu_tlas_trace(ctx, gpu_blas, gpu_tlas, rays)
            _assert_gpu_matches_cpu(hits[0], hits[1], 0, cpu_ray)
    else:
        assert_true(False, "No Accelerator found")


def test_gpu_tlas_two_translated_instances_report_instance_id() raises:
    comptime if has_accelerator():
        var verts = _make_two_depth_triangles()
        var cpu_blas = BinaryBvh(verts.unsafe_ptr(), UInt32(len(verts) // 3))
        cpu_blas.build["median", False]()
        var cpu_tlas = _make_two_translated_tlas(cpu_blas)

        var rays = List[Ray](capacity=2)
        rays.append(Ray(Vec3f32(-10.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0)))
        rays.append(Ray(Vec3f32(10.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0)))

        var cpu_left = _trace_cpu_tlas(cpu_tlas, rays[0], cpu_blas)
        var cpu_right = _trace_cpu_tlas(cpu_tlas, rays[1], cpu_blas)
        assert_true(cpu_left.hit.inst == 0)
        assert_true(cpu_right.hit.inst == 1)

        with DeviceContext() as ctx:
            var gpu_blas = GpuLBVH(ctx, verts)
            _build_gpu_blas(ctx, gpu_blas, verts)
            var gpu_tlas = GpuTlasLayout(ctx, cpu_tlas)
            var hits = _run_gpu_tlas_trace(ctx, gpu_blas, gpu_tlas, rays)
            _assert_gpu_matches_cpu(hits[0], hits[1], 0, cpu_left)
            _assert_gpu_matches_cpu(hits[0], hits[1], 1, cpu_right)
    else:
        assert_true(False, "No Accelerator found")


def test_gpu_tlas_nearest_instance_wins() raises:
    comptime if has_accelerator():
        var verts = _make_two_depth_triangles()
        var cpu_blas = BinaryBvh(verts.unsafe_ptr(), UInt32(len(verts) // 3))
        cpu_blas.build["median", False]()
        var cpu_tlas = _make_nearest_tlas(cpu_blas)

        var ray = Ray(Vec3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))
        var cpu_ray = _trace_cpu_tlas(cpu_tlas, ray, cpu_blas)
        assert_true(cpu_ray.hit.inst == 1)
        assert_true(cpu_ray.hit.prim == 0)
        assert_almost_equal(cpu_ray.hit.t, 2.0, atol=1.0e-4)

        var rays = List[Ray](capacity=1)
        rays.append(ray.copy())

        with DeviceContext() as ctx:
            var gpu_blas = GpuLBVH(ctx, verts)
            _build_gpu_blas(ctx, gpu_blas, verts)
            var gpu_tlas = GpuTlasLayout(ctx, cpu_tlas)
            var hits = _run_gpu_tlas_trace(ctx, gpu_blas, gpu_tlas, rays)
            _assert_gpu_matches_cpu(hits[0], hits[1], 0, cpu_ray)
    else:
        assert_true(False, "No Accelerator found")


def test_gpu_tlas_matches_cpu_tlas_grid() raises:
    comptime if has_accelerator():
        var verts = _make_two_depth_triangles()
        var cpu_blas = BinaryBvh(verts.unsafe_ptr(), UInt32(len(verts) // 3))
        cpu_blas.build["median", False]()
        var cpu_tlas = _make_grid_tlas(cpu_blas, 8)

        var rays = List[Ray](capacity=8)
        var cpu_rays = List[Ray](capacity=8)
        for i in range(8):
            var x = Float32((i % 4) - 2) * 6.0
            var y = Float32(i // 4) * 4.0
            var ray = Ray(Vec3f32(x, y, 0.0), Vec3f32(0.0, 0.0, 1.0))
            rays.append(ray.copy())
            cpu_rays.append(_trace_cpu_tlas(cpu_tlas, ray, cpu_blas))
            assert_true(cpu_rays[i].hit.inst == UInt32(i))
            assert_true(cpu_rays[i].hit.prim == 0)

        with DeviceContext() as ctx:
            var gpu_blas = GpuLBVH(ctx, verts)
            _build_gpu_blas(ctx, gpu_blas, verts)
            var gpu_tlas = GpuTlasLayout(ctx, cpu_tlas)
            var hits = _run_gpu_tlas_trace(ctx, gpu_blas, gpu_tlas, rays)

            for i in range(8):
                _assert_gpu_matches_cpu(hits[0], hits[1], i, cpu_rays[i])
    else:
        assert_true(False, "No Accelerator found")


# Keep this last so `run_tests.sh` can discover and run the file directly.
def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
