"""GPU BLAS/TLAS tests using the canonical descriptor-backed ownership."""

from std.sys import has_accelerator
from std.math import abs
from std.testing import (
    TestSuite,
    assert_true,
    assert_almost_equal,
)
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.core import AABB, Vec3f32, Affine3f32, Point3f32, Frame
from bajo.bvh.camera import Camera
from bajo.bvh.constants import f32_max
from bajo.bvh.host_utils import compute_bounds, sphere_bounds
from bajo.bvh.cpu import CpuBlasSet
from bajo.bvh.types import Instance, Sphere, Hit
from bajo.bvh.gpu.utils import upload_camera
from bajo.bvh.gpu import GpuBvhLayout
from bajo.bvh.gpu.triangle_bvh import build_gpu_triangle_blas_set
from bajo.bvh.gpu.sphere_bvh import build_gpu_sphere_blas_set
from bajo.bvh.gpu.tlas import build_gpu_tlas
from bajo.bvh.cpu.blas_set import build_cpu_triangle_blas_set
from bajo.bvh.cpu.tlas import CpuTlas
from test.bvh.fixtures import (
    _camera_for_bounds,
    _download_tlas_checksum,
    _make_camera_ray,
)


comptime TLAS_WIDTH = 4
comptime BLAS_WIDTH = 4


def _make_triangle_at_z(z: Float32) -> List[Point3f32[.LOCAL]]:
    return [
        Point3f32[.LOCAL](-1.0, -1.0, z),
        Point3f32[.LOCAL](1.0, -1.0, z),
        Point3f32[.LOCAL](0.0, 1.0, z),
    ]


def _instances_bounds(instances: List[Instance]) -> AABB[.WORLD]:
    var bounds = AABB[.WORLD].invalid()
    for inst in instances:
        bounds.grow(inst.bounds)
    return bounds


def _cpu_triangle_tlas_checksum[
    tlas_width: SIMDLength,
    blas_width: SIMDLength,
](
    instances: List[Instance],
    cpu_blases: CpuBlasSet[.TRIANGLE, blas_width],
    camera: Camera,
    width: Int,
    height: Int,
) -> Tuple[Float64, UInt32, UInt64]:
    var tlas = CpuTlas[tlas_width](instances)
    var checksum = Float64(0.0)
    var hits = UInt32(0)
    var inst_checksum = UInt64(0)

    for py in range(height):
        for px in range(width):
            var ray = camera.make_ray(px, py, width, height)
            var hit = tlas.trace_blases[
                blas_width, blas_width, .CLOSEST_HIT
            ](ray, cpu_blases)
            if hit.t < f32_max:
                checksum += Float64(hit.t)
                hits += 1
                inst_checksum += UInt64(hit.inst)

    return (checksum, hits, inst_checksum)


def _triangle_instance(
    blas_idx: UInt32,
    translation: Point3f32[.WORLD],
    local_bounds: AABB[.LOCAL],
) -> Instance:
    return Instance(
        Affine3f32[.LOCAL, .WORLD].from_translation(translation),
        blas_idx,
        local_bounds,
        .TRIANGLE,
    )


def _sphere_instance(
    blas_idx: UInt32,
    translation: Point3f32[.WORLD],
    local_bounds: AABB[.LOCAL],
) -> Instance:
    return Instance(
        Affine3f32[.LOCAL, .WORLD].from_translation(translation),
        blas_idx,
        local_bounds,
        .SPHERE,
    )


def _download_single_hit(
    hits: DeviceBuffer[.float32],
) raises -> Tuple[Float32, UInt32, UInt32]:
    with hits.map_to_host() as hf:
        var hit = Hit[.WORLD].load(
            Span(unsafe_ptr=hf.unsafe_ptr(), length=len(hf)), 0
        )
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
        var blases = build_gpu_triangle_blas_set[
            BLAS_WIDTH, BLAS_WIDTH, .LBVH
        ](ctx, [near_verts^, far_verts^])

        var left = Point3f32[.WORLD](-10.0, 0.0, 0.0)
        var right = Point3f32[.WORLD](10.0, 0.0, 0.0)
        var instances: List = [
            _triangle_instance(0, left, near_bounds),
            _triangle_instance(1, right, far_bounds),
        ]

        var tlas = build_gpu_tlas[.TRIANGLE, TLAS_WIDTH, BLAS_WIDTH](ctx, instances)
        var camera = _make_camera_ray(
            right, Vec3f32[.WORLD](0.0, 0.0, 1.0)
        )
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

        # If traversal ignores Instance.blas_idx and always uses BLAS 0, this
        # returns t=2.0 or trips the old blas_idx == 0 assertion. The correct
        # typed TLAS must select BLAS 1 for instance 1 and return t=6.0.
        _assert_hit(_download_single_hit(d_hits), 6.0, 0, 1)


def test_gpu_triangle_tlas_cwbvh8_camera_matches_expected_hit() raises:
    var near_verts = _make_triangle_at_z(2.0)
    var far_verts = _make_triangle_at_z(6.0)
    var near_bounds = compute_bounds(near_verts)
    var far_bounds = compute_bounds(far_verts)

    with DeviceContext() as ctx:
        var blases = build_gpu_triangle_blas_set[
            8, 4, .HPLOC, GpuBvhLayout.CWBVH8
        ](ctx, [near_verts^, far_verts^])
        var left = Point3f32[.WORLD](-10.0, 0.0, 0.0)
        var right = Point3f32[.WORLD](10.0, 0.0, 0.0)
        var instances: List = [
            _triangle_instance(0, left, near_bounds),
            _triangle_instance(1, right, far_bounds),
        ]
        var tlas = build_gpu_tlas[.TRIANGLE,
            2, 8, 2, 4, .LBVH, GpuBvhLayout.CWBVH8
        ](ctx, instances)
        var camera = _make_camera_ray(
            right,
            Vec3f32[.WORLD](0.0, 0.0, 1.0),
        )
        var d_camera = upload_camera(ctx, camera)
        var d_hits = ctx.enqueue_create_buffer[.float32](Hit.STRIDE)

        tlas.launch_camera(ctx, blases, d_camera, d_hits, 1, 1, 1)
        ctx.synchronize()
        _assert_hit(_download_single_hit(d_hits), 6.0, 0, 1)


def test_gpu_triangle_tlas_closest_hit_across_different_blas() raises:
    var far_verts = _make_triangle_at_z(8.0)
    var near_verts = _make_triangle_at_z(3.0)
    var far_bounds = compute_bounds(far_verts)
    var near_bounds = compute_bounds(near_verts)

    with DeviceContext() as ctx:
        var blases = build_gpu_triangle_blas_set[BLAS_WIDTH](
            ctx, [far_verts^, near_verts^]
        )

        var zero = Point3f32[.WORLD](0.0, 0.0, 0.0)
        var instances: List = [
            _triangle_instance(0, zero, far_bounds),
            _triangle_instance(1, zero, near_bounds),
        ]

        var tlas = build_gpu_tlas[.TRIANGLE, TLAS_WIDTH, BLAS_WIDTH](ctx, instances)
        var camera = _make_camera_ray(zero, Vec3f32[.WORLD](0.0, 0.0, 1.0))
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

        # Both TLAS instances are hit by the same ray. Closest-hit traversal must
        # query each instance's own BLAS and keep the nearer BLAS-1 result.
        _assert_hit(_download_single_hit(d_hits), 3.0, 0, 1)


def test_gpu_triangle_tlas_independent_node_and_leaf_widths() raises:
    var near_verts = _make_triangle_at_z(2.0)
    var far_verts = _make_triangle_at_z(6.0)
    var near_bounds = compute_bounds(near_verts)
    var far_bounds = compute_bounds(far_verts)

    with DeviceContext() as ctx:
        var blases = build_gpu_triangle_blas_set[2, 4](
            ctx, [near_verts^, far_verts^]
        )

        var left = Point3f32[.WORLD](-10.0, 0.0, 0.0)
        var right = Point3f32[.WORLD](10.0, 0.0, 0.0)
        var instances: List = [
            _triangle_instance(0, left, near_bounds),
            _triangle_instance(1, right, far_bounds),
        ]

        # TLAS2 nodes with four-wide instance leaves over BLAS2 nodes with
        # four-wide triangle leaves.
        var tlas = build_gpu_tlas[.TRIANGLE, 2, 2, 4, 4](ctx, instances)
        var camera = _make_camera_ray(
            right, Vec3f32[.WORLD](0.0, 0.0, 1.0)
        )
        var d_camera = upload_camera(ctx, camera)
        var d_hits = ctx.enqueue_create_buffer[.float32](Hit.STRIDE)

        tlas.launch_camera(ctx, blases, d_camera, d_hits, 1, 1, 1)
        ctx.synchronize()

        _assert_hit(_download_single_hit(d_hits), 6.0, 0, 1)


# -----------------------------------------------------------------------------
# Sphere typed TLAS
# -----------------------------------------------------------------------------
def test_gpu_sphere_tlas_uses_instance_blas_index() raises:
    var near_spheres: List = [
        Sphere[.LOCAL](Point3f32[.LOCAL](0.0, 0.0, 2.0), 1.0)
    ]
    var far_spheres: List = [
        Sphere[.LOCAL](Point3f32[.LOCAL](0.0, 0.0, 6.0), 1.0)
    ]
    var near_bounds = sphere_bounds(near_spheres)
    var far_bounds = sphere_bounds(far_spheres)

    with DeviceContext() as ctx:
        var blases = build_gpu_sphere_blas_set[
            BLAS_WIDTH, BLAS_WIDTH, .LBVH
        ](ctx, [near_spheres^, far_spheres^])

        var left = Point3f32[.WORLD](-10.0, 0.0, 0.0)
        var right = Point3f32[.WORLD](10.0, 0.0, 0.0)
        var instances: List = [
            _sphere_instance(0, left, near_bounds),
            _sphere_instance(1, right, far_bounds),
        ]

        var tlas = build_gpu_tlas[.SPHERE, TLAS_WIDTH, BLAS_WIDTH](ctx, instances)
        var camera = _make_camera_ray(
            right, Vec3f32[.WORLD](0.0, 0.0, 1.0)
        )
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

        # Far sphere center is at local z=6 with radius 1, so the expected hit
        # distance is 5.0. Using BLAS 0 would incorrectly return 1.0.
        _assert_hit(_download_single_hit(d_hits), 5.0, 0, 1)


def test_gpu_triangle_tlas_builders_stress_8_blas_512_instances_match_cpu() raises:
    comptime STRESS_BLAS_COUNT = 8
    comptime STRESS_X = 32
    comptime STRESS_Y = 16
    comptime STRESS_WIDTH = 64
    comptime STRESS_HEIGHT = 32

    var vertex_sets = List[List[Point3f32[.LOCAL]]](
        capacity=STRESS_BLAS_COUNT
    )
    var local_bounds = List[AABB[.LOCAL]](capacity=STRESS_BLAS_COUNT)

    for b in range(STRESS_BLAS_COUNT):
        var z = Float32(2.0 + Float32(b) * 0.35)
        var verts: List = [
            Point3f32[.LOCAL](-0.8, -0.7, z),
            Point3f32[.LOCAL](0.8 + Float32(b % 3) * 0.05, -0.7, z),
            Point3f32[.LOCAL](0.0, 0.9, z),
        ]
        var bounds = compute_bounds(verts)
        vertex_sets.append(verts.copy())
        local_bounds.append(bounds)

    var cpu_blases = build_cpu_triangle_blas_set[
        BLAS_WIDTH, BLAS_WIDTH, .LBVH, .LOCAL
    ](vertex_sets)

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
                    Point3f32[.WORLD](tx, ty, tz),
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
        var blases = build_gpu_triangle_blas_set[BLAS_WIDTH](ctx, vertex_sets)

        var tlas = build_gpu_tlas[.TRIANGLE, TLAS_WIDTH, BLAS_WIDTH](ctx, instances)
        var d_camera = upload_camera(ctx, camera)
        var ray_count = STRESS_WIDTH * STRESS_HEIGHT
        var d_hits = ctx.enqueue_create_buffer[.float32](
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

        var gpu = _download_tlas_checksum[.WORLD](d_hits, ray_count)

        assert_true(abs(cpu[0] - gpu[0]) <= Float64(0.01))
        assert_true(cpu[1] == gpu[1])
        assert_true(cpu[2] == gpu[2])

        var hploc_tlas = build_gpu_tlas[.TRIANGLE,
            TLAS_WIDTH,
            BLAS_WIDTH,
            TLAS_WIDTH,
            BLAS_WIDTH,
            .HPLOC,
        ](ctx, instances)
        hploc_tlas.launch_camera(
            ctx,
            blases,
            d_camera,
            d_hits,
            ray_count,
            STRESS_WIDTH,
            STRESS_HEIGHT,
        )
        ctx.synchronize()
        var hploc_gpu = _download_tlas_checksum[.WORLD](d_hits, ray_count)
        assert_true(abs(cpu[0] - hploc_gpu[0]) <= Float64(0.01))
        assert_true(cpu[1] == hploc_gpu[1])
        assert_true(cpu[2] == hploc_gpu[2])


def main() raises:
    comptime if not has_accelerator():
        raise "No Accelerator found"
    TestSuite.discover_tests[__functions_in_module()]().run()
