from std.gpu import DeviceBuffer
from std.gpu.host import DeviceContext
from std.math import abs
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal, assert_true

from bajo.core.bvh import (
    compute_bounds,
    copy_list_to_device,
)
from bajo.core.bvh.cpu.binary_bvh import BinaryBvh
from bajo.core.bvh.cpu.tlas import BvhInstance, Tlas
from bajo.core.bvh.gpu.kernels import (
    append_camera_params,
    compute_centroid_bounds,
)
from bajo.core.bvh.gpu.lbvh import GpuLBVH
from bajo.core.bvh.gpu.tlas import GpuTlasLayout
from bajo.core.bvh.gpu.tlas_traverse import (
    launch_tlas_lbvh_camera_primary,
    launch_shade_tlas_normals,
)
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


def _make_two_view_camera_params() -> List[Float32]:
    var params = List[Float32](capacity=24)
    append_camera_params(
        params,
        Vec3f32(-10.0, 0.0, 0.0),
        Vec3f32(-10.0, 0.0, 2.0),
        Vec3f32(0.0, 1.0, 0.0),
    )
    append_camera_params(
        params,
        Vec3f32(10.0, 0.0, 0.0),
        Vec3f32(10.0, 0.0, 2.0),
        Vec3f32(0.0, 1.0, 0.0),
    )
    return params^


def _build_two_instance_tlas(mut cpu_blas: BinaryBvh) -> Tlas:
    var instances = List[BvhInstance](capacity=2)
    instances.append(
        BvhInstance.from_blas(
            _translation(-10.0, 0.0, 0.0),
            _translation(10.0, 0.0, 0.0),
            0,
            cpu_blas,
        )
    )
    instances.append(
        BvhInstance.from_blas(
            _translation(10.0, 0.0, 0.0),
            _translation(-10.0, 0.0, 0.0),
            0,
            cpu_blas,
        )
    )
    var tlas = Tlas(instances)
    tlas.build()
    return tlas^


def _run_camera_trace(
    mut ctx: DeviceContext,
    gpu_blas: GpuLBVH,
    gpu_tlas: GpuTlasLayout,
    camera_params: List[Float32],
    width: Int,
    height: Int,
    views: Int,
) raises -> Tuple[DeviceBuffer[DType.float32], DeviceBuffer[DType.uint32]]:
    var ray_count = width * height * views
    var d_camera = copy_list_to_device(ctx, camera_params)
    var d_hits_f32 = ctx.enqueue_create_buffer[DType.float32](ray_count * 3)
    var d_hits_u32 = ctx.enqueue_create_buffer[DType.uint32](ray_count * 2)

    launch_tlas_lbvh_camera_primary(
        ctx,
        gpu_blas,
        gpu_tlas,
        d_camera,
        d_hits_f32,
        d_hits_u32,
        ray_count,
        width,
        height,
        views,
    )
    ctx.synchronize()

    return (d_hits_f32^, d_hits_u32^)


def test_gpu_tlas_camera_two_views_hit_expected_instances() raises:
    comptime if has_accelerator():
        var verts = _make_two_depth_triangles()
        var cpu_blas = BinaryBvh(verts.unsafe_ptr(), UInt32(len(verts) // 3))
        cpu_blas.build["median", False]()
        var cpu_tlas = _build_two_instance_tlas(cpu_blas)
        var camera_params = _make_two_view_camera_params()

        with DeviceContext() as ctx:
            var gpu_blas = GpuLBVH(ctx, verts)
            _build_gpu_blas(ctx, gpu_blas, verts)
            var gpu_tlas = GpuTlasLayout(ctx, cpu_tlas)
            var hits = _run_camera_trace(
                ctx, gpu_blas, gpu_tlas, camera_params, 1, 1, 2
            )

            with hits[0].map_to_host() as hf:
                with hits[1].map_to_host() as hu:
                    assert_almost_equal(hf[0], 2.0, atol=1.0e-4)
                    assert_true(hu[0] == 0)
                    assert_true(hu[1] == 0)

                    assert_almost_equal(hf[3], 2.0, atol=1.0e-4)
                    assert_true(hu[2] == 0)
                    assert_true(hu[3] == 1)
    else:
        assert_true(False, "No Accelerator found")


def test_gpu_tlas_camera_normal_shading_writes_rgb() raises:
    comptime if has_accelerator():
        var verts = _make_two_depth_triangles()
        var cpu_blas = BinaryBvh(verts.unsafe_ptr(), UInt32(len(verts) // 3))
        cpu_blas.build["median", False]()
        var instances = List[BvhInstance](capacity=1)
        instances.append(
            BvhInstance.from_blas(
                Mat44f32.identity(), Mat44f32.identity(), 0, cpu_blas
            )
        )
        var cpu_tlas = Tlas(instances)
        cpu_tlas.build()

        var camera_params = List[Float32](capacity=12)
        append_camera_params(
            camera_params,
            Vec3f32(0.0, 0.0, 0.0),
            Vec3f32(0.0, 0.0, 2.0),
            Vec3f32(0.0, 1.0, 0.0),
        )

        with DeviceContext() as ctx:
            var gpu_blas = GpuLBVH(ctx, verts)
            _build_gpu_blas(ctx, gpu_blas, verts)
            var gpu_tlas = GpuTlasLayout(ctx, cpu_tlas)
            var hits = _run_camera_trace(
                ctx, gpu_blas, gpu_tlas, camera_params, 1, 1, 1
            )
            var d_rgb = ctx.enqueue_create_buffer[DType.uint32](1)
            launch_shade_tlas_normals(
                ctx, gpu_blas, gpu_tlas, hits[1], d_rgb, 1, 1, 1
            )
            ctx.synchronize()

            with d_rgb.map_to_host() as rgb:
                var packed = UInt32(rgb[0])
                var r = (packed >> 16) & UInt32(255)
                var g = (packed >> 8) & UInt32(255)
                var b = packed & UInt32(255)
                assert_true(b > r)
                assert_true(b > g)
                assert_true(r > 100)
                assert_true(g > 100)
    else:
        assert_true(False, "No Accelerator found")


# Keep this last so `run_tests.sh` can discover and run the file directly.
def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
