from std.gpu import DeviceBuffer
from std.gpu.host import DeviceContext
from std.math import abs
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal, assert_true

from bajo.core.bvh.host_utils import (
    compute_bounds,
    copy_list_to_device,
    append_camera_params,
    compute_centroid_bounds,
)
from bajo.core.bvh.cpu.binary_bvh import BinaryBvh
from bajo.core.bvh.cpu.tlas import BvhInstance, Tlas
from bajo.core.bvh.gpu.lbvh import GpuLBVH
from bajo.core.bvh.gpu.tlas import GpuTlasLayout
from bajo.core.bvh.gpu.tlas_traverse import (
    launch_tlas_lbvh_camera_primary,
    launch_shade_tlas_normals,
)
from bajo.core.mat import Mat44f32, _translation
from bajo.core.vec import Vec3f32
from fixtures import (
    _append_tri,
    _make_two_depth_triangles,
    _build_gpu_blas,
)


comptime GPU_TLAS_TEST_MISS = UInt32(0xFFFFFFFF)


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
    var instances = [
        BvhInstance.from_blas(
            _translation(Float32(-10.0), 0.0, 0.0),
            _translation(Float32(10.0), 0.0, 0.0),
            0,
            cpu_blas,
        ),
        BvhInstance.from_blas(
            _translation(Float32(10.0), 0.0, 0.0),
            _translation(Float32(-10.0), 0.0, 0.0),
            0,
            cpu_blas,
        ),
    ]
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
        var cpu_blas = BinaryBvh(verts.unsafe_ptr(), UInt32(len(verts) / 3))
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
        var cpu_blas = BinaryBvh(verts.unsafe_ptr(), UInt32(len(verts) / 3))
        cpu_blas.build["median", False]()
        var instances = [
            BvhInstance.from_blas(
                Mat44f32.identity(), Mat44f32.identity(), 0, cpu_blas
            )
        ]
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
