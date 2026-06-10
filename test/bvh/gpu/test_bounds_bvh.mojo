from std.benchmark import keep
from std.math import abs, max
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true
from std.gpu import DeviceContext, DeviceBuffer

from bajo.core import AABB, Vec3f32
from bajo.core.vec import vmin, vmax
from bajo.bvh.camera import Camera
from bajo.bvh.types import Ray, Sphere
from bajo.bvh.host_utils import hit_t_for_checksum, sphere_bounds
from bajo.bvh.constants import EMPTY_LANE, TRACE
from bajo.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.bvh.gpu.bounds_bvh import GpuBoundsBvh
from bajo.bvh.gpu.sphere_bvh import GpuSphereBvh
from bajo.bvh.gpu.triangle_bvh import GpuTriangleBvh
from bajo.bvh.gpu.utils import upload_vertices, upload_list

from test.bvh.fixtures import _append_tri, _brute_sphere_trace


comptime GPU_BOUNDS_TEST_WIDTH = 64
comptime GPU_BOUNDS_TEST_HEIGHT = 48
comptime GPU_BOUNDS_TEST_VIEWS = 3
comptime GPU_BOUNDS_TEST_EPS = 0.05


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
        Sphere(Vec3f32(0.0, 0.0, 2.0), 1.0),
        Sphere(Vec3f32(4.0, 0.0, 4.0), 1.0),
        Sphere(Vec3f32(-4.0, 0.0, 6.0), 1.0),
        Sphere(Vec3f32(0.0, 4.0, 8.0), 1.0),
        Sphere(Vec3f32(4.0, 4.0, 10.0), 1.0),
        Sphere(Vec3f32(-4.0, 4.0, 12.0), 1.0),
        Sphere(Vec3f32(4.0, -4.0, 14.0), 1.0),
        Sphere(Vec3f32(-4.0, -4.0, 16.0), 1.0),
    ]


def _make_single_sphere_scene() -> List[Sphere]:
    return [Sphere(Vec3f32(0.0, 0.0, 2.0), 1.0)]


def _make_duplicate_sphere_centroid_scene() -> List[Sphere]:
    var spheres = List[Sphere](capacity=12)
    for i in range(12):
        spheres.append(
            Sphere(Vec3f32(0.0, 0.0, 2.0), 1.0 + Float32(i % 3) * 0.01)
        )
    return spheres^


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


def _make_sphere_leaf_bounds(
    mut ctx: DeviceContext,
    spheres: List[Sphere],
) raises -> Tuple[DeviceBuffer[DType.float32], DeviceBuffer[DType.uint32]]:
    var leaf_bounds = List[Float32](capacity=max(len(spheres), 1) * 6)
    var payloads = List[UInt32](capacity=max(len(spheres), 1))

    for i in range(len(spheres)):
        ref s = spheres[i]
        var r = s.radius

        leaf_bounds.append(s.center.x - r)
        leaf_bounds.append(s.center.y - r)
        leaf_bounds.append(s.center.z - r)
        leaf_bounds.append(s.center.x + r)
        leaf_bounds.append(s.center.y + r)
        leaf_bounds.append(s.center.z + r)
        payloads.append(UInt32(i))

    var h_leaf_bounds = ctx.enqueue_create_host_buffer[DType.float32](
        len(leaf_bounds)
    )
    var h_payloads = ctx.enqueue_create_host_buffer[DType.uint32](len(payloads))
    var d_leaf_bounds = ctx.enqueue_create_buffer[DType.float32](
        len(leaf_bounds)
    )
    var d_payloads = ctx.enqueue_create_buffer[DType.uint32](len(payloads))
    h_leaf_bounds.enqueue_copy_from(Span(leaf_bounds))
    h_payloads.enqueue_copy_from(Span(payloads))

    h_leaf_bounds.enqueue_copy_to(d_leaf_bounds)
    h_payloads.enqueue_copy_to(d_payloads)

    return (d_leaf_bounds^, d_payloads^)


def _trace_cpu_spheres_bruteforce(
    spheres: List[Sphere],
    rays: List[Ray],
) -> Float64:
    var checksum = Float64(0.0)

    for ray in rays:
        var brute = _brute_sphere_trace(spheres, ray.o, ray.d)
        checksum += hit_t_for_checksum(brute.t)

    return checksum


def _make_degenerate_axis_scene() -> List[Vec3f32]:
    var verts = List[Vec3f32](capacity=16 * 3)
    for i in range(16):
        var cx = Float32(i * 2 - 15)
        _append_tri(verts, cx, 0.0, 2.0)
    return verts^


def _make_triangle_leaf_bounds(
    mut ctx: DeviceContext,
    verts: List[Vec3f32],
) raises -> Tuple[DeviceBuffer[DType.float32], DeviceBuffer[DType.uint32]]:
    var tri_count = len(verts) / 3
    var leaf_bounds = List[Float32](capacity=max(tri_count, 1) * 6)
    var payloads = List[UInt32](capacity=max(tri_count, 1))

    for i in range(tri_count):
        ref v0 = verts[i * 3 + 0]
        ref v1 = verts[i * 3 + 1]
        ref v2 = verts[i * 3 + 2]

        var _min = vmin(vmin(v0, v1), v2)
        var _max = vmax(vmax(v0, v1), v2)

        leaf_bounds.append(_min.x)
        leaf_bounds.append(_min.y)
        leaf_bounds.append(_min.z)
        leaf_bounds.append(_max.x)
        leaf_bounds.append(_max.y)
        leaf_bounds.append(_max.z)
        payloads.append(UInt32(i))

    var h_leaf_bounds = ctx.enqueue_create_host_buffer[DType.float32](
        len(leaf_bounds)
    )
    var h_payloads = ctx.enqueue_create_host_buffer[DType.uint32](len(payloads))
    var d_leaf_bounds = ctx.enqueue_create_buffer[DType.float32](
        len(leaf_bounds)
    )
    var d_payloads = ctx.enqueue_create_buffer[DType.uint32](len(payloads))
    h_leaf_bounds.enqueue_copy_from(Span(leaf_bounds))
    h_payloads.enqueue_copy_from(Span(payloads))

    h_leaf_bounds.enqueue_copy_to(d_leaf_bounds)
    h_payloads.enqueue_copy_to(d_payloads)

    return (d_leaf_bounds^, d_payloads^)


def _trace_cpu_triangle_bvh[
    width: Int
](mut bvh: TriangleBvh[width], rays: List[Ray]) -> Float64:
    var checksum = Float64(0.0)
    for ray in rays:
        var hit = bvh.trace[TRACE.CLOSEST_HIT](ray)
        checksum += hit_t_for_checksum(hit.t)
    return checksum


def _download_hit_checksum(
    hits_f32: DeviceBuffer[DType.float32],
    ray_count: Int,
) raises -> Tuple[Float64, UInt32]:
    var checksum = Float64(0.0)
    var hit_count = UInt32(0)

    with hits_f32.map_to_host() as h:
        for i in range(ray_count):
            var t = h[i * 3]
            checksum += hit_t_for_checksum(t)
            if t < 1.0e20:
                hit_count += 1

    return (checksum, hit_count)


def _assert_gpu_bounds_width[width: Int](verts: List[Vec3f32]) raises:
    with DeviceContext() as ctx:
        var build = _make_triangle_leaf_bounds(ctx, verts)
        var leaf_bounds = build[0].copy()
        var payloads = build[1].copy()

        var bvh = GpuBoundsBvh[width](ctx, leaf_bounds, payloads)
        var binary_bvh = bvh.build_test(ctx)
        var validation = binary_bvh.validate(binary_bvh.root_bounds())

        assert_true(validation.sorted_ok, "generic bounds keys sorted")
        assert_true(validation.values_ok, "generic bounds values valid")
        assert_true(validation.topology_ok, "generic bounds topology valid")
        assert_true(validation.bounds_ok, "generic bounds refit valid")
        assert_true(
            bvh.leaf_block_count > 0, "wide collapse produced leaf blocks"
        )
        assert_true(bvh.leaf_block_count <= bvh.max_leaf_blocks)


def _assert_wide_lane_invariants[width: Int](verts: List[Vec3f32]) raises:
    with DeviceContext() as ctx:
        var build = _make_triangle_leaf_bounds(ctx, verts)
        var leaf_bounds = build[0].copy()
        var payloads = build[1].copy()

        var bvh = GpuBoundsBvh[width](ctx, leaf_bounds, payloads)
        _ = bvh.build(ctx)

        var seen_live_lane = False
        with bvh.wide_counts.map_to_host() as counts:
            with bvh.wide_data.map_to_host() as data:
                for n in range(bvh.node_count):
                    for lane in range(width):
                        var idx = n * width + lane
                        var count = UInt32(counts[idx])

                        if count == EMPTY_LANE:
                            continue

                        seen_live_lane = True
                        if count == 0:
                            assert_true(data[idx] < UInt32(bvh.node_count))
                        else:
                            assert_true(count <= UInt32(width))
                            assert_true(
                                data[idx] < UInt32(bvh.leaf_block_count)
                            )

        assert_true(seen_live_lane, "wide collapse had no live lanes")


def _assert_gpu_triangle_width_matches_cpu_camera[
    width: Int
](verts: List[Vec3f32]) raises:
    var cpu_bvh = TriangleBvh[width].__init__["lbvh"](verts.copy())
    var camera_data = _make_camera_rays_and_params(
        cpu_bvh.bounds(),
        GPU_BOUNDS_TEST_WIDTH,
        GPU_BOUNDS_TEST_HEIGHT,
        GPU_BOUNDS_TEST_VIEWS,
    )
    var rays = camera_data[0].copy()
    var camera_params = camera_data[1].copy()
    var cpu_checksum = _trace_cpu_triangle_bvh[width](cpu_bvh, rays)

    with DeviceContext() as ctx:
        var d_verts = upload_vertices(ctx, verts)
        var gpu_bvh = GpuTriangleBvh[width](ctx, d_verts)
        var d_camera = upload_list(ctx, camera_params)
        var d_hits_f32 = ctx.enqueue_create_buffer[DType.float32](len(rays) * 3)
        var d_hits_u32 = ctx.enqueue_create_buffer[DType.uint32](len(rays))

        gpu_bvh.launch_camera(
            ctx,
            d_camera,
            d_hits_f32,
            d_hits_u32,
            len(rays),
            GPU_BOUNDS_TEST_WIDTH,
            GPU_BOUNDS_TEST_HEIGHT,
        )
        ctx.synchronize()

        var gpu_result = _download_hit_checksum(d_hits_f32, len(rays))
        var gpu_checksum = gpu_result[0]
        var gpu_hits = gpu_result[1]
        var mismatch_count = UInt32(0)

        with d_hits_f32.map_to_host() as hf:
            with d_hits_u32.map_to_host() as hu:
                for i in range(len(rays)):
                    var cpu_hit = cpu_bvh.trace[TRACE.CLOSEST_HIT](rays[i])
                    var gpu_t = hf[i * 3 + 0]
                    var gpu_prim = UInt32(hu[i])
                    var cpu_t = cpu_hit.t
                    var cpu_prim = cpu_hit.prim
                    var same_miss = cpu_t >= 1.0e20 and gpu_t >= 1.0e20
                    var t_diff = abs(Float64(gpu_t) - Float64(cpu_t))

                    if not same_miss and (
                        t_diff > Float64(1.0e-4) or gpu_prim != cpu_prim
                    ):
                        if mismatch_count < 16:
                            print(
                                t"mismatch ray={i} cpu_t={cpu_t} "
                                t"cpu_prim={cpu_prim} gpu_t={gpu_t} "
                                t"gpu_prim={gpu_prim}"
                            )
                        mismatch_count += 1

        var diff = abs(gpu_checksum - cpu_checksum)
        if diff > GPU_BOUNDS_TEST_EPS or mismatch_count != 0:
            print(
                t"width={width} gpu={gpu_checksum} cpu={cpu_checksum} "
                t"diff={diff} mismatches={mismatch_count} hits={gpu_hits}"
            )
        assert_true(diff <= GPU_BOUNDS_TEST_EPS, "GpuTriangleBvh checksum")
        assert_true(mismatch_count == 0, "GpuTriangleBvh primitive/t mismatch")


def _assert_gpu_sphere_width_matches_bruteforce_camera[
    width: Int
](spheres: List[Sphere]) raises:
    var bounds = sphere_bounds(spheres)
    var camera_data = _make_camera_rays_and_params(
        bounds,
        GPU_BOUNDS_TEST_WIDTH,
        GPU_BOUNDS_TEST_HEIGHT,
        GPU_BOUNDS_TEST_VIEWS,
    )
    var rays = camera_data[0].copy()
    var camera_params = camera_data[1].copy()
    var cpu_checksum = _trace_cpu_spheres_bruteforce(spheres, rays)

    with DeviceContext() as ctx:
        var gpu_bvh = GpuSphereBvh[width](ctx, spheres)
        var d_camera = upload_list(ctx, camera_params)
        var d_hits_f32 = ctx.enqueue_create_buffer[DType.float32](len(rays) * 3)
        var d_hits_u32 = ctx.enqueue_create_buffer[DType.uint32](len(rays))

        gpu_bvh.launch_camera(
            ctx,
            d_camera,
            d_hits_f32,
            d_hits_u32,
            len(rays),
            GPU_BOUNDS_TEST_WIDTH,
            GPU_BOUNDS_TEST_HEIGHT,
        )
        ctx.synchronize()

        var gpu_result = _download_hit_checksum(d_hits_f32, len(rays))
        var gpu_checksum = gpu_result[0]
        var hit_count = gpu_result[1]

        var diff = abs(gpu_checksum - cpu_checksum)
        if diff > GPU_BOUNDS_TEST_EPS:
            print(
                t"width={width} gpu={gpu_checksum} cpu={cpu_checksum} "
                t"diff={diff} hits={hit_count}"
            )
        assert_true(diff <= GPU_BOUNDS_TEST_EPS, "GpuSphereBvh checksum")


def _assert_gpu_sphere_bounds_width[width: Int](spheres: List[Sphere]) raises:
    with DeviceContext() as ctx:
        var build = _make_sphere_leaf_bounds(ctx, spheres)
        var leaf_bounds = build[0].copy()
        var payloads = build[1].copy()

        var bvh = GpuBoundsBvh[width](ctx, leaf_bounds, payloads)
        var binary_bvh = bvh.build_test(ctx)
        var validation = binary_bvh.validate(bvh.root_bounds())

        assert_true(validation.sorted_ok, "sphere bounds keys sorted")
        assert_true(validation.values_ok, "sphere bounds values valid")
        assert_true(validation.topology_ok, "sphere bounds topology valid")
        assert_true(validation.bounds_ok, "sphere bounds refit valid")
        assert_true(bvh.node_count > 0, "sphere wide collapse produced nodes")
        assert_true(
            bvh.leaf_block_count > 0,
            "sphere wide collapse produced leaf blocks",
        )


def test_gpu_bounds_bvh_width_N_build_validate_small_scene() raises:
    scene = _make_small_scene()
    _assert_gpu_bounds_width[2](scene)
    _assert_gpu_bounds_width[4](scene)
    _assert_gpu_bounds_width[8](scene)


def test_gpu_bounds_bvh_width4_single_triangle() raises:
    scene = _make_single_triangle_scene()
    _assert_gpu_bounds_width[4](scene)
    _assert_gpu_triangle_width_matches_cpu_camera[4](scene)


def test_gpu_bounds_bvh_width4_duplicate_morton_codes() raises:
    scene = _make_duplicate_centroid_scene()
    _assert_gpu_bounds_width[4](scene)
    _assert_wide_lane_invariants[4](scene)


def test_gpu_bounds_bvh_width8_degenerate_axis() raises:
    scene = _make_degenerate_axis_scene()
    _assert_gpu_bounds_width[8](scene)
    _assert_wide_lane_invariants[8](scene)


def test_gpu_bounds_bvh_width_N_wide_lane_invariants() raises:
    scene = _make_small_scene()
    _assert_wide_lane_invariants[2](scene)
    _assert_wide_lane_invariants[4](scene)
    _assert_wide_lane_invariants[8](scene)


def test_gpu_triangle_bvh_width_N_camera_primary_matches_cpu() raises:
    scene = _make_small_scene()
    _assert_gpu_triangle_width_matches_cpu_camera[2](scene)
    _assert_gpu_triangle_width_matches_cpu_camera[4](scene)
    _assert_gpu_triangle_width_matches_cpu_camera[8](scene)


def test_gpu_sphere_bvh_width_N_camera_primary_matches_bruteforce() raises:
    scene = _make_small_sphere_scene()
    _assert_gpu_sphere_width_matches_bruteforce_camera[2](scene)
    _assert_gpu_sphere_width_matches_bruteforce_camera[4](scene)
    _assert_gpu_sphere_width_matches_bruteforce_camera[8](scene)


def test_gpu_sphere_bvh_width4_single_sphere() raises:
    scene = _make_single_sphere_scene()
    _assert_gpu_sphere_bounds_width[4](scene)
    _assert_gpu_sphere_width_matches_bruteforce_camera[4](scene)


def test_gpu_sphere_bvh_width4_duplicate_morton_codes() raises:
    scene = _make_duplicate_sphere_centroid_scene()
    _assert_gpu_sphere_bounds_width[4](scene)
    _assert_gpu_sphere_width_matches_bruteforce_camera[4](scene)


def main() raises:
    comptime if not has_accelerator():
        raise "No Accelerator found"
    TestSuite.discover_tests[__functions_in_module()]().run()
