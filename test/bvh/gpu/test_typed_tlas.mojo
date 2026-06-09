from std.sys import has_accelerator
from std.math import abs
from std.testing import TestSuite, assert_true, assert_almost_equal
from std.gpu import DeviceBuffer, DeviceContext

from bajo.core import AABB, Vec3f32, Affine3f32
from bajo.bvh.camera import Camera
from bajo.bvh.constants import Primitive, EMPTY_LANE, TRACE
from bajo.bvh.host_utils import compute_bounds, hit_t_for_checksum
from bajo.bvh.types import Instance, Sphere, Ray
from bajo.bvh.gpu.utils import upload_camera
from bajo.bvh.gpu.triangle_bvh import build_triangle_blas_set
from bajo.bvh.gpu.sphere_bvh import build_sphere_blas_set
from bajo.bvh.gpu.tlas import GpuTriangleTlas, GpuSphereTlas
from bajo.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.bvh.cpu.tlas import Tlas


comptime TLAS_WIDTH = 4
comptime BLAS_WIDTH = 4
comptime MISS = UInt32(0xFFFFFFFF)


def _make_triangle_at_z(z: Float32) -> List[Vec3f32]:
    return [
        Vec3f32(-1.0, -1.0, z),
        Vec3f32(1.0, -1.0, z),
        Vec3f32(0.0, 1.0, z),
    ]


def _make_camera_ray(origin: Vec3f32, direction: Vec3f32) -> Camera:
    return Camera(
        origin,
        origin + direction,
        Vec3f32(0.0, 1.0, 0.0),
        Float32(0.75),
    )


def _camera_for_bounds(bounds: AABB) -> Camera:
    var center = bounds.centroid()
    var extent = bounds.extent()
    var scene_w = extent.x
    if extent.y > scene_w:
        scene_w = extent.y
    if extent.z > scene_w:
        scene_w = extent.z
    if scene_w < 1.0:
        scene_w = 1.0

    var eye = center + Vec3f32(0.0, 0.0, -scene_w * 2.5)
    return Camera(
        eye,
        center,
        Vec3f32(0.0, 1.0, 0.0),
        Float32(0.75),
    )


def _instances_bounds(instances: List[Instance]) -> AABB:
    var bounds = AABB.invalid()
    for inst in instances:
        bounds.grow(inst.bounds)
    return bounds


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
            checksum += hit_t_for_checksum(t)
            if t < 1.0e20:
                hits += 1

    with hits_u32.map_to_host() as hu:
        for i in range(ray_count):
            var inst = UInt32(hu[i * 2 + 1])
            if inst != MISS:
                inst_checksum += UInt64(inst)

    return (checksum, hits, inst_checksum)


def _cpu_triangle_tlas_checksum[
    tlas_width: Int,
    blas_width: Int,
](
    instances: List[Instance],
    mut cpu_blases: List[TriangleBvh[blas_width]],
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
            var hit = tlas.trace[TriangleBvh[blas_width], TRACE.CLOSEST_HIT](
                ray,
                cpu_blases.unsafe_ptr(),
            )
            checksum += hit_t_for_checksum(hit.t)
            if hit.t < 1.0e20:
                hits += 1
                inst_checksum += UInt64(hit.inst)

    return (checksum, hits, inst_checksum)


def _sphere_bounds(spheres: List[Sphere]) -> AABB:
    var bounds = AABB.invalid()
    for s in spheres:
        var r = Vec3f32(s.radius)
        bounds.grow(s.center - r)
        bounds.grow(s.center + r)
    return bounds


def _triangle_instance(
    blas_idx: UInt32,
    translation: Vec3f32,
    local_bounds: AABB,
) -> Instance:
    return Instance(
        Affine3f32.from_translation(translation),
        Affine3f32.from_translation(-translation),
        blas_idx,
        local_bounds,
        Primitive.TRIANGLE,
    )


def _sphere_instance(
    blas_idx: UInt32,
    translation: Vec3f32,
    local_bounds: AABB,
) -> Instance:
    return Instance(
        Affine3f32.from_translation(translation),
        Affine3f32.from_translation(-translation),
        blas_idx,
        local_bounds,
        Primitive.SPHERE,
    )


def _download_single_hit(
    hits_f32: DeviceBuffer[DType.float32],
    hits_u32: DeviceBuffer[DType.uint32],
) raises -> Tuple[Float32, UInt32, UInt32]:
    var t: Float32
    var prim: UInt32
    var inst: UInt32

    with hits_f32.map_to_host() as hf:
        t = hf[0]

    with hits_u32.map_to_host() as hu:
        prim = UInt32(hu[0])
        inst = UInt32(hu[1])

    return (t, prim, inst)


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

        var left = Vec3f32(-10.0, 0.0, 0.0)
        var right = Vec3f32(10.0, 0.0, 0.0)
        var instances = [
            _triangle_instance(0, left, near_bounds),
            _triangle_instance(1, right, far_bounds),
        ]

        var tlas = GpuTriangleTlas[TLAS_WIDTH, BLAS_WIDTH](ctx, instances)
        var camera = _make_camera_ray(right, Vec3f32(0.0, 0.0, 1.0))
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

        # If traversal ignores Instance.blas_idx and always uses BLAS 0, this
        # returns t=2.0 or trips the old blas_idx == 0 assertion. The correct
        # typed TLAS must select BLAS 1 for instance 1 and return t=6.0.
        _assert_hit(_download_single_hit(d_hits_f32, d_hits_u32), 6.0, 0, 1)


def test_gpu_triangle_tlas_closest_hit_across_different_blas() raises:
    var far_verts = _make_triangle_at_z(8.0)
    var near_verts = _make_triangle_at_z(3.0)
    var far_bounds = compute_bounds(far_verts)
    var near_bounds = compute_bounds(near_verts)

    with DeviceContext() as ctx:
        var blases = build_triangle_blas_set[BLAS_WIDTH](
            ctx, [far_verts^, near_verts^]
        )

        var zero = Vec3f32(0.0, 0.0, 0.0)
        var instances = [
            _triangle_instance(0, zero, far_bounds),
            _triangle_instance(1, zero, near_bounds),
        ]

        var tlas = GpuTriangleTlas[TLAS_WIDTH, BLAS_WIDTH](ctx, instances)
        var camera = _make_camera_ray(zero, Vec3f32(0.0, 0.0, 1.0))
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

        # Both TLAS instances are hit by the same ray. Closest-hit traversal must
        # query each instance's own BLAS and keep the nearer BLAS-1 result.
        _assert_hit(_download_single_hit(d_hits_f32, d_hits_u32), 3.0, 0, 1)


# -----------------------------------------------------------------------------
# Sphere typed TLAS
# -----------------------------------------------------------------------------
def test_gpu_sphere_tlas_uses_instance_blas_index() raises:
    var near_spheres = [Sphere(Vec3f32(0.0, 0.0, 2.0), 1.0)]
    var far_spheres = [Sphere(Vec3f32(0.0, 0.0, 6.0), 1.0)]
    var near_bounds = _sphere_bounds(near_spheres)
    var far_bounds = _sphere_bounds(far_spheres)

    with DeviceContext() as ctx:
        var blases = build_sphere_blas_set[BLAS_WIDTH](
            ctx, [near_spheres^, far_spheres^]
        )

        var left = Vec3f32(-10.0, 0.0, 0.0)
        var right = Vec3f32(10.0, 0.0, 0.0)
        var instances = [
            _sphere_instance(0, left, near_bounds),
            _sphere_instance(1, right, far_bounds),
        ]

        var tlas = GpuSphereTlas[TLAS_WIDTH, BLAS_WIDTH](ctx, instances)
        var camera = _make_camera_ray(right, Vec3f32(0.0, 0.0, 1.0))
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

        # Far sphere center is at local z=6 with radius 1, so the expected hit
        # distance is 5.0. Using BLAS 0 would incorrectly return 1.0.
        _assert_hit(_download_single_hit(d_hits_f32, d_hits_u32), 5.0, 0, 1)


def test_gpu_triangle_tlas_stress_8_blas_512_instances_matches_cpu() raises:
    comptime STRESS_BLAS_COUNT = 8
    comptime STRESS_X = 32
    comptime STRESS_Y = 16
    comptime STRESS_WIDTH = 64
    comptime STRESS_HEIGHT = 32

    var vertex_sets = List[List[Vec3f32]](capacity=STRESS_BLAS_COUNT)
    var local_bounds = List[AABB](capacity=STRESS_BLAS_COUNT)
    var cpu_blases = List[TriangleBvh[BLAS_WIDTH]](capacity=STRESS_BLAS_COUNT)

    for b in range(STRESS_BLAS_COUNT):
        var z = Float32(2.0 + Float32(b) * 0.35)
        var verts = [
            Vec3f32(-0.8, -0.7, z),
            Vec3f32(0.8 + Float32(b % 3) * 0.05, -0.7, z),
            Vec3f32(0.0, 0.9, z),
        ]
        var bounds = compute_bounds(verts)
        vertex_sets.append(verts.copy())
        local_bounds.append(bounds)
        cpu_blases.append(
            TriangleBvh[BLAS_WIDTH].__init__["lbvh"](
                verts.unsafe_ptr(),
                UInt32(1),
            )
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
                    Vec3f32(tx, ty, tz),
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
        var d_hits_f32 = ctx.enqueue_create_buffer[DType.float32](ray_count * 3)
        var d_hits_u32 = ctx.enqueue_create_buffer[DType.uint32](ray_count * 2)

        tlas.launch_camera(
            ctx,
            blases,
            d_camera,
            d_hits_f32,
            d_hits_u32,
            ray_count,
            STRESS_WIDTH,
            STRESS_HEIGHT,
        )
        ctx.synchronize()

        var gpu = _download_tlas_checksum(d_hits_f32, d_hits_u32, ray_count)

        assert_true(abs(cpu[0] - gpu[0]) <= Float64(0.01))
        assert_true(cpu[1] == gpu[1])
        assert_true(cpu[2] == gpu[2])


def main() raises:
    comptime if not has_accelerator():
        raise "No Accelerator found"
    TestSuite.discover_tests[__functions_in_module()]().run()
