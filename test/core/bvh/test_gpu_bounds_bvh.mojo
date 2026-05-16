from std.benchmark import keep
from std.math import abs, min, max, sqrt
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true, assert_almost_equal
from std.gpu import DeviceContext

from bajo.core.aabb import AABB
from bajo.core.vec import Vec3f32, vmin, vmax
from bajo.core.intersect import intersect_ray_sphere
from bajo.core.bvh.types import Ray, Sphere
from bajo.core.bvh.host_utils import (
    flatten_rays,
    generate_primary_rays,
    hit_t_for_checksum,
)
from bajo.core.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.core.bvh.gpu.bounds_bvh import (
    GpuBoundsBvh,
    GPU_WIDE_EMPTY_LANE,
)
from bajo.core.bvh.gpu.sphere_bvh import GpuSphereBvh
from bajo.core.bvh.gpu.triangle_bvh import GpuTriangleBvh

from fixtures import _append_tri


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


def _make_sphere_leaf_bounds(
    spheres: List[Sphere],
) -> Tuple[List[Float32], List[UInt32]]:
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

    return (leaf_bounds^, payloads^)


def _sphere_scene_bounds(spheres: List[Sphere]) -> AABB:
    var bounds = AABB.invalid()
    for i in range(len(spheres)):
        ref s = spheres[i]
        var r = s.radius
        bounds.grow(s.center - r)
        bounds.grow(s.center + r)
    return bounds


def _brute_sphere_trace(
    spheres: List[Sphere],
    O: Vec3f32,
    D: Vec3f32,
) -> Tuple[Bool, UInt32, Float32]:
    var best_t = Float32(1.0e30)
    var best_prim = UInt32(0xFFFFFFFF)

    for i in range(len(spheres)):
        ref s = spheres[i]
        h = intersect_ray_sphere(O, D, s.center, s.radius, Float32.MAX)
        if h.t > 1.0e-4 and h.t < best_t:
            best_t = h.t
            best_prim = UInt32(i)

    return (best_prim != UInt32(0xFFFFFFFF), best_prim, best_t)


def _trace_cpu_spheres_bruteforce(
    spheres: List[Sphere],
    rays: List[Ray],
) -> Float64:
    var checksum = Float64(0.0)

    for i in range(len(rays)):
        var brute = _brute_sphere_trace(spheres, rays[i].O, rays[i].D)
        if brute[0]:
            checksum += hit_t_for_checksum(brute[2])
        else:
            checksum += hit_t_for_checksum(Float32(1.0e30))

    return checksum


def _make_degenerate_axis_scene() -> List[Vec3f32]:
    var verts = List[Vec3f32](capacity=16 * 3)
    for i in range(16):
        var cx = Float32(i * 2 - 15)
        _append_tri(verts, cx, 0.0, 2.0)
    return verts^


def _make_triangle_leaf_bounds(
    verts: List[Vec3f32],
) -> Tuple[List[Float32], List[UInt32]]:
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

    return (leaf_bounds^, payloads^)


def _trace_cpu_triangle_bvh[
    width: Int
](mut bvh: TriangleBvh[width], rays: List[Ray],) -> Float64:
    var checksum = Float64(0.0)
    for i in range(len(rays)):
        var ray = rays[i].copy()
        bvh.traverse(ray)
        checksum += hit_t_for_checksum(ray.hit.t)
    return checksum


def _assert_gpu_bounds_width[width: Int](verts: List[Vec3f32]) raises:
    var build = _make_triangle_leaf_bounds(verts)
    var leaf_bounds = build[0].copy()
    var payloads = build[1].copy()

    with DeviceContext() as ctx:
        var bvh = GpuBoundsBvh[width](ctx, leaf_bounds, payloads)
        _ = bvh.build(ctx)
        var validation = bvh.validate(bvh.root_bounds())

        assert_true(validation.sorted_ok, "generic bounds keys sorted")
        assert_true(validation.values_ok, "generic bounds values valid")
        assert_true(validation.topology_ok, "generic bounds topology valid")
        assert_true(validation.bounds_ok, "generic bounds refit valid")
        assert_true(
            bvh.leaf_block_count > 0, "wide collapse produced leaf blocks"
        )
        assert_true(bvh.leaf_block_count <= bvh.max_leaf_blocks)
        keep(validation.guard)


def _assert_wide_lane_invariants[width: Int](verts: List[Vec3f32]) raises:
    var build = _make_triangle_leaf_bounds(verts)
    var leaf_bounds = build[0].copy()
    var payloads = build[1].copy()

    with DeviceContext() as ctx:
        var bvh = GpuBoundsBvh[width](ctx, leaf_bounds, payloads)
        _ = bvh.build(ctx)

        var seen_live_lane = False
        with bvh.wide_counts.map_to_host() as counts:
            with bvh.wide_data.map_to_host() as data:
                for n in range(bvh.node_count):
                    for lane in range(width):
                        var idx = n * width + lane
                        var count = UInt32(counts[idx])

                        if count == GPU_WIDE_EMPTY_LANE:
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


def _assert_gpu_triangle_width_matches_cpu[
    width: Int
](verts: List[Vec3f32]) raises:
    var v = verts.copy()
    var cpu_bvh = TriangleBvh[width].__init__["lbvh"](
        v.unsafe_ptr(), UInt32(len(verts) / 3)
    )
    var rays = generate_primary_rays(
        cpu_bvh.bounds(),
        GPU_BOUNDS_TEST_WIDTH,
        GPU_BOUNDS_TEST_HEIGHT,
        GPU_BOUNDS_TEST_VIEWS,
    )
    var rays_flat = flatten_rays(rays)
    var cpu_checksum = _trace_cpu_triangle_bvh[width](cpu_bvh, rays)

    with DeviceContext() as ctx:
        var gpu_bvh = GpuTriangleBvh[width](ctx, verts)
        var d_rays = ctx.enqueue_create_buffer[DType.float32](len(rays_flat))
        var d_hits_f32 = ctx.enqueue_create_buffer[DType.float32](len(rays) * 3)
        var d_hits_u32 = ctx.enqueue_create_buffer[DType.uint32](len(rays))

        with d_rays.map_to_host() as h:
            for i in range(len(rays_flat)):
                h[i] = rays_flat[i]
        ctx.synchronize()

        gpu_bvh.launch_uploaded_primary(
            ctx,
            d_rays,
            d_hits_f32,
            d_hits_u32,
            len(rays),
        )
        ctx.synchronize()

        var gpu_checksum = Float64(0.0)
        var hit_count = 0
        with d_hits_f32.map_to_host() as h:
            for i in range(len(rays)):
                var t = h[i * 3]
                gpu_checksum += hit_t_for_checksum(t)
                if t < 1.0e20:
                    hit_count += 1

        var diff = abs(gpu_checksum - cpu_checksum)
        assert_true(diff <= GPU_BOUNDS_TEST_EPS, "GpuTriangleBvh checksum")
        keep(hit_count)


def _assert_gpu_sphere_width_matches_bruteforce[
    width: Int
](spheres: List[Sphere]) raises:
    var bounds = _sphere_scene_bounds(spheres)
    var rays = generate_primary_rays(
        bounds,
        GPU_BOUNDS_TEST_WIDTH,
        GPU_BOUNDS_TEST_HEIGHT,
        GPU_BOUNDS_TEST_VIEWS,
    )
    var rays_flat = flatten_rays(rays)
    var cpu_checksum = _trace_cpu_spheres_bruteforce(spheres, rays)

    with DeviceContext() as ctx:
        var gpu_bvh = GpuSphereBvh[width](ctx, spheres)
        var d_rays = ctx.enqueue_create_buffer[DType.float32](len(rays_flat))
        var d_hits_f32 = ctx.enqueue_create_buffer[DType.float32](len(rays) * 3)
        var d_hits_u32 = ctx.enqueue_create_buffer[DType.uint32](len(rays))

        with d_rays.map_to_host() as h:
            for i in range(len(rays_flat)):
                h[i] = rays_flat[i]
        ctx.synchronize()

        gpu_bvh.launch_uploaded_primary(
            ctx,
            d_rays,
            d_hits_f32,
            d_hits_u32,
            len(rays),
        )
        ctx.synchronize()

        var gpu_checksum = Float64(0.0)
        var hit_count = 0
        with d_hits_f32.map_to_host() as h:
            for i in range(len(rays)):
                var t = h[i * 3]
                gpu_checksum += hit_t_for_checksum(t)
                if t < 1.0e20:
                    hit_count += 1

        var diff = abs(gpu_checksum - cpu_checksum)
        assert_true(diff <= GPU_BOUNDS_TEST_EPS, "GpuSphereBvh checksum")
        keep(hit_count)


def _assert_gpu_sphere_bounds_width[width: Int](spheres: List[Sphere]) raises:
    var build = _make_sphere_leaf_bounds(spheres)
    var leaf_bounds = build[0].copy()
    var payloads = build[1].copy()

    with DeviceContext() as ctx:
        var bvh = GpuBoundsBvh[width](ctx, leaf_bounds, payloads)
        _ = bvh.build(ctx)
        var validation = bvh.validate(bvh.root_bounds())

        assert_true(validation.sorted_ok, "sphere bounds keys sorted")
        assert_true(validation.values_ok, "sphere bounds values valid")
        assert_true(validation.topology_ok, "sphere bounds topology valid")
        assert_true(validation.bounds_ok, "sphere bounds refit valid")
        assert_true(bvh.node_count > 0, "sphere wide collapse produced nodes")
        assert_true(
            bvh.leaf_block_count > 0,
            "sphere wide collapse produced leaf blocks",
        )
        keep(validation.guard)


def test_gpu_bounds_bvh_width_N_build_validate_small_scene() raises:
    scene = _make_small_scene()
    _assert_gpu_bounds_width[2](scene)
    _assert_gpu_bounds_width[4](scene)
    _assert_gpu_bounds_width[8](scene)


def test_gpu_bounds_bvh_width4_single_triangle() raises:
    scene = _make_single_triangle_scene()
    _assert_gpu_bounds_width[4](scene)
    _assert_gpu_triangle_width_matches_cpu[4](scene)


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


def test_gpu_triangle_bvh_width2_uploaded_primary_matches_cpu() raises:
    scene = _make_small_scene()
    _assert_gpu_triangle_width_matches_cpu[2](scene)
    _assert_gpu_triangle_width_matches_cpu[4](scene)
    _assert_gpu_triangle_width_matches_cpu[8](scene)


def test_gpu_sphere_bvh_width_N_uploaded_primary_matches_bruteforce() raises:
    scene = _make_small_sphere_scene()
    _assert_gpu_sphere_width_matches_bruteforce[2](scene)
    _assert_gpu_sphere_width_matches_bruteforce[4](scene)
    _assert_gpu_sphere_width_matches_bruteforce[8](scene)


def test_gpu_sphere_bvh_width4_single_sphere() raises:
    scene = _make_single_sphere_scene()
    _assert_gpu_sphere_bounds_width[4](scene)
    _assert_gpu_sphere_width_matches_bruteforce[4](scene)


def test_gpu_sphere_bvh_width4_duplicate_morton_codes() raises:
    scene = _make_duplicate_sphere_centroid_scene()
    _assert_gpu_sphere_bounds_width[4](scene)
    _assert_gpu_sphere_width_matches_bruteforce[4](scene)


def main() raises:
    comptime if not has_accelerator():
        raise "No Accelerator found"
    TestSuite.discover_tests[__functions_in_module()]().run()
