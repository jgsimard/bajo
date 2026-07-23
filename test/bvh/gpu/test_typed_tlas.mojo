from std.sys import has_accelerator
from std.math import abs
from std.testing import TestSuite, assert_true, assert_almost_equal
from std.gpu import DeviceBuffer, DeviceContext

from bajo.core import AABB, Vec3f32, Affine3f32, Point3f32, Frame
from bajo.bvh.camera import Camera
from bajo.bvh.constants import Primitive, TRACE, f32_max
from bajo.bvh.host_utils import compute_bounds, sphere_bounds
from bajo.bvh.types import Instance, Sphere, Hit
from bajo.bvh.gpu.utils import upload_camera
from bajo.bvh.gpu.triangle_bvh import build_triangle_blas_set
from bajo.bvh.gpu.sphere_bvh import build_sphere_blas_set
from bajo.bvh.gpu.tlas import GpuTriangleTlas, GpuSphereTlas
from bajo.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.bvh.cpu.tlas import Tlas
from test.bvh.fixtures import (
    _camera_for_bounds,
    _download_tlas_checksum,
    _make_camera_ray,
)


comptime TLAS_WIDTH = 4
comptime BLAS_WIDTH = 4


def _make_triangle_at_z(z: Float32) -> List[Point3f32[Frame.LOCAL]]:
    return [
        Point3f32[Frame.LOCAL](-1.0, -1.0, z),
        Point3f32[Frame.LOCAL](1.0, -1.0, z),
        Point3f32[Frame.LOCAL](0.0, 1.0, z),
    ]


def _instances_bounds(instances: List[Instance]) -> AABB[Frame.WORLD]:
    var bounds = AABB[Frame.WORLD].invalid()
    for inst in instances:
        bounds.grow(inst.bounds)
    return bounds


def _cpu_triangle_tlas_checksum[
    tlas_width: SIMDLength,
    blas_width: SIMDLength,
](
    instances: List[Instance],
    mut cpu_blases: List[TriangleBvh[Frame.LOCAL, blas_width]],
    camera: Camera,
    width: Int,
    height: Int,
) -> Tuple[Float64, UInt32, UInt64]:
    var tlas = Tlas[tlas_width](instances)
    var checksum = Float64(0.0)
    var hits = UInt32(0)
    var inst_checksum = UInt64(0)

    for py in range(height):
        for px in range(width):
            var ray = camera.make_ray(px, py, width, height)
            var hit = tlas.trace[
                TriangleBvh[Frame.LOCAL, blas_width], TRACE.CLOSEST_HIT
            ](
                ray,
                cpu_blases.unsafe_ptr(),
            )
            if hit.t < f32_max:
                checksum += Float64(hit.t)
                hits += 1
                inst_checksum += UInt64(hit.inst)

    return (checksum, hits, inst_checksum)


def _triangle_instance(
    blas_idx: UInt32,
    translation: Point3f32[Frame.WORLD],
    local_bounds: AABB[Frame.LOCAL],
) -> Instance:
    return Instance(
        Affine3f32[Frame.LOCAL, Frame.WORLD].from_translation(translation),
        blas_idx,
        local_bounds,
        Primitive.TRIANGLE,
    )


def _sphere_instance(
    blas_idx: UInt32,
    translation: Point3f32[Frame.WORLD],
    local_bounds: AABB[Frame.LOCAL],
) -> Instance:
    return Instance(
        Affine3f32[Frame.LOCAL, Frame.WORLD].from_translation(translation),
        blas_idx,
        local_bounds,
        Primitive.SPHERE,
    )


def _download_single_hit(
    hits: DeviceBuffer[DType.float32],
) raises -> Tuple[Float32, UInt32, UInt32]:
    with hits.map_to_host() as hf:
        var hit = Hit[Frame.WORLD].load(hf.unsafe_ptr(), 0)
        return (hit.t, hit.prim, hit.inst)


def _assert_hit(
    hit: Tuple[Float32, UInt32, UInt32],
    expected_t: Float32,
    expected_prim: UInt32,
    expected_inst: UInt32,
) raises:
    assert_almost_equal(hit[0], expected_t)
    assert_true(hit[1] == expected_prim)
    assert_true(hit[2] == expected_inst)


def test_gpu_triangle_tlas_uses_instance_blas_index() raises:
    var near_verts = _make_triangle_at_z(2.0)
    var far_verts = _make_triangle_at_z(6.0)
    var near_bounds = compute_bounds(near_verts)
    var far_bounds = compute_bounds(far_verts)

    with DeviceContext() as ctx:
        var blases = build_triangle_blas_set[BLAS_WIDTH](
            ctx, [near_verts^, far_verts^]
        )

        var left = Point3f32[Frame.WORLD](-10.0, 0.0, 0.0)
        var right = Point3f32[Frame.WORLD](10.0, 0.0, 0.0)
        var instances = [
            _triangle_instance(0, left, near_bounds),
            _triangle_instance(1, right, far_bounds),
        ]

        var tlas = GpuTriangleTlas[TLAS_WIDTH, BLAS_WIDTH](ctx, instances)
        var camera = _make_camera_ray(
            right, Vec3f32[Frame.WORLD](0.0, 0.0, 1.0)
        )
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

        # If traversal ignores Instance.blas_idx and always uses BLAS 0, this
        # returns t=2.0 or trips the old blas_idx == 0 assertion. The correct
        # typed TLAS must select BLAS 1 for instance 1 and return t=6.0.
        _assert_hit(_download_single_hit(d_hits), 6.0, 0, 1)


def test_gpu_triangle_tlas_closest_hit_across_different_blas() raises:
    var far_verts = _make_triangle_at_z(8.0)
    var near_verts = _make_triangle_at_z(3.0)
    var far_bounds = compute_bounds(far_verts)
    var near_bounds = compute_bounds(near_verts)

    with DeviceContext() as ctx:
        var blases = build_triangle_blas_set[BLAS_WIDTH](
            ctx, [far_verts^, near_verts^]
        )

        var zero = Point3f32[Frame.WORLD](0.0, 0.0, 0.0)
        var instances = [
            _triangle_instance(0, zero, far_bounds),
            _triangle_instance(1, zero, near_bounds),
        ]

        var tlas = GpuTriangleTlas[TLAS_WIDTH, BLAS_WIDTH](ctx, instances)
        var camera = _make_camera_ray(zero, Vec3f32[Frame.WORLD](0.0, 0.0, 1.0))
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

        # Both TLAS instances are hit by the same ray. Closest-hit traversal must
        # query each instance's own BLAS and keep the nearer BLAS-1 result.
        _assert_hit(_download_single_hit(d_hits), 3.0, 0, 1)


# -----------------------------------------------------------------------------
# Sphere typed TLAS
# -----------------------------------------------------------------------------
def test_gpu_sphere_tlas_uses_instance_blas_index() raises:
    var near_spheres = [
        Sphere[Frame.LOCAL](Point3f32[Frame.LOCAL](0.0, 0.0, 2.0), 1.0)
    ]
    var far_spheres = [
        Sphere[Frame.LOCAL](Point3f32[Frame.LOCAL](0.0, 0.0, 6.0), 1.0)
    ]
    var near_bounds = sphere_bounds(near_spheres)
    var far_bounds = sphere_bounds(far_spheres)

    with DeviceContext() as ctx:
        var blases = build_sphere_blas_set[BLAS_WIDTH](
            ctx, [near_spheres^, far_spheres^]
        )

        var left = Point3f32[Frame.WORLD](-10.0, 0.0, 0.0)
        var right = Point3f32[Frame.WORLD](10.0, 0.0, 0.0)
        var instances = [
            _sphere_instance(0, left, near_bounds),
            _sphere_instance(1, right, far_bounds),
        ]

        var tlas = GpuSphereTlas[TLAS_WIDTH, BLAS_WIDTH](ctx, instances)
        var camera = _make_camera_ray(
            right, Vec3f32[Frame.WORLD](0.0, 0.0, 1.0)
        )
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

        # Far sphere center is at local z=6 with radius 1, so the expected hit
        # distance is 5.0. Using BLAS 0 would incorrectly return 1.0.
        _assert_hit(_download_single_hit(d_hits), 5.0, 0, 1)


def test_gpu_triangle_tlas_stress_8_blas_512_instances_matches_cpu() raises:
    comptime STRESS_BLAS_COUNT = 8
    comptime STRESS_X = 32
    comptime STRESS_Y = 16
    comptime STRESS_WIDTH = 64
    comptime STRESS_HEIGHT = 32

    var vertex_sets = List[List[Point3f32[Frame.LOCAL]]](
        capacity=STRESS_BLAS_COUNT
    )
    var local_bounds = List[AABB[Frame.LOCAL]](capacity=STRESS_BLAS_COUNT)
    var cpu_blases = List[TriangleBvh[Frame.LOCAL, BLAS_WIDTH]](
        capacity=STRESS_BLAS_COUNT
    )

    for b in range(STRESS_BLAS_COUNT):
        var z = Float32(2.0 + Float32(b) * 0.35)
        var verts = [
            Point3f32[Frame.LOCAL](-0.8, -0.7, z),
            Point3f32[Frame.LOCAL](0.8 + Float32(b % 3) * 0.05, -0.7, z),
            Point3f32[Frame.LOCAL](0.0, 0.9, z),
        ]
        var bounds = compute_bounds(verts)
        vertex_sets.append(verts.copy())
        local_bounds.append(bounds)
        cpu_blases.append(
            TriangleBvh[Frame.LOCAL, BLAS_WIDTH].__init__["lbvh"](verts.copy())
        )

    var instances = List[Instance](capacity=STRESS_X * STRESS_Y)
    for y in range(STRESS_Y):
        for x in range(STRESS_X):
            var idx = y * STRESS_X + x
            var blas_idx = UInt32(idx % STRESS_BLAS_COUNT)
            var tx = (Float32(x) - Float32(STRESS_X - 1) * 0.5) * 2.5
            var ty = (Float32(y) - Float32(STRESS_Y - 1) * 0.5) * 2.5
            var tz = Float32((idx * 7) % 11) * 0.03
            instances.append(
                _triangle_instance(
                    blas_idx,
                    Point3f32[Frame.WORLD](tx, ty, tz),
                    local_bounds[Int(blas_idx)],
                )
            )

    var scene_bounds = _instances_bounds(instances)
    var camera = _camera_for_bounds(scene_bounds)
    var cpu = _cpu_triangle_tlas_checksum[TLAS_WIDTH, BLAS_WIDTH](
        instances,
        cpu_blases,
        camera,
        STRESS_WIDTH,
        STRESS_HEIGHT,
    )

    with DeviceContext() as ctx:
        var blases = build_triangle_blas_set[BLAS_WIDTH](ctx, vertex_sets)

        var tlas = GpuTriangleTlas[TLAS_WIDTH, BLAS_WIDTH](ctx, instances)
        var d_camera = upload_camera(ctx, camera)
        var ray_count = STRESS_WIDTH * STRESS_HEIGHT
        var d_hits = ctx.enqueue_create_buffer[DType.float32](
            ray_count * Hit.STRIDE
        )

        tlas.launch_camera(
            ctx,
            blases,
            d_camera,
            d_hits,
            ray_count,
            STRESS_WIDTH,
            STRESS_HEIGHT,
        )
        ctx.synchronize()

        var gpu = _download_tlas_checksum[Frame.WORLD](d_hits, ray_count)

        assert_true(abs(cpu[0] - gpu[0]) <= Float64(0.01))
        assert_true(cpu[1] == gpu[1])
        assert_true(cpu[2] == gpu[2])


def main() raises:
    comptime if not has_accelerator():
        raise "No Accelerator found"
    TestSuite.discover_tests[__functions_in_module()]().run()
