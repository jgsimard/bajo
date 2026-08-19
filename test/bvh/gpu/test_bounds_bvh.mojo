from std.benchmark import keep
from std.math import abs, max
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true
from max.gpu.host import DeviceContext, DeviceBuffer

from bajo.core import Vec3f32, Point3f32, Frame
from bajo.bvh.types import Sphere, Hit
from bajo.bvh.host_utils import sphere_bounds, triangle_bounds
from bajo.bvh.constants import EMPTY_LANE, TRACE, f32_max
from bajo.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.bvh.gpu.bounds_bvh import build_bounds_bvh
from bajo.bvh.gpu.builder import GpuBvhBuildMethod
from bajo.bvh.gpu.wide_layout import (
    GpuWideBoundsBvh,
    _wide_node_load_meta,
)
from bajo.bvh.gpu.quality import (
    measure_binary_bvh_quality,
    measure_wide_bvh_quality,
)
from bajo.bvh.gpu.diagnostics import (
    build_bounds_bvh_for_diagnostics,
    validate_binary_bvh,
)
from bajo.bvh.gpu.wide_meta import _wide_meta_count, _wide_meta_data
from bajo.bvh.gpu.sphere_bvh import build_sphere_bvh
from bajo.bvh.gpu.triangle_bvh import (
    GpuTriangleBvhBuildArena,
    build_triangle_bvh,
    enqueue_build_triangle_bvh,
    enqueue_build_triangle_bvh_with_arena,
)
from bajo.bvh.gpu.trace import GpuTraversalStats
from bajo.bvh.gpu.utils import upload_vertices, upload_rays, upload_list

from test.bvh.fixtures import (
    _append_tri,
    _download_hit_checksum,
    _make_camera_rays_and_params,
    _make_duplicate_centroid_scene,
    _make_duplicate_sphere_centroid_scene,
    _make_single_sphere_scene,
    _make_single_triangle_scene,
    _make_small_scene,
    _make_small_sphere_scene,
    _trace_cpu_spheres_bruteforce,
    _trace_cpu_triangle_bvh,
)


comptime GPU_BOUNDS_TEST_WIDTH = 64
comptime GPU_BOUNDS_TEST_HEIGHT = 48
comptime GPU_BOUNDS_TEST_VIEWS = 3
comptime GPU_BOUNDS_TEST_EPS = 0.05


def _make_sphere_leaf_bounds(
    mut ctx: DeviceContext,
    spheres: List[Sphere[Frame.WORLD]],
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
    h_leaf_bounds.enqueue_copy_from(leaf_bounds)
    h_payloads.enqueue_copy_from(payloads)

    h_leaf_bounds.enqueue_copy_to(d_leaf_bounds)
    h_payloads.enqueue_copy_to(d_payloads)

    return (d_leaf_bounds^, d_payloads^)


def _make_degenerate_axis_scene() -> List[Point3f32[Frame.WORLD]]:
    var verts = List[Point3f32[Frame.WORLD]](capacity=16 * 3)
    for i in range(16):
        var cx = Float32(i * 2 - 15)
        _append_tri(verts, cx, 0.0, 2.0)
    return verts^


def _make_triangle_leaf_bounds(
    mut ctx: DeviceContext,
    verts: List[Point3f32[Frame.WORLD]],
) raises -> Tuple[DeviceBuffer[DType.float32], DeviceBuffer[DType.uint32]]:
    var tri_count = len(verts) / 3
    var leaf_bounds = List[Float32](capacity=max(tri_count, 1) * 6)
    var payloads = List[UInt32](capacity=max(tri_count, 1))

    for i in range(tri_count):
        ref v0 = verts[i * 3 + 0]
        ref v1 = verts[i * 3 + 1]
        ref v2 = verts[i * 3 + 2]

        var bounds = triangle_bounds(v0, v1, v2)

        leaf_bounds.append(bounds._min.x)
        leaf_bounds.append(bounds._min.y)
        leaf_bounds.append(bounds._min.z)
        leaf_bounds.append(bounds._max.x)
        leaf_bounds.append(bounds._max.y)
        leaf_bounds.append(bounds._max.z)
        payloads.append(UInt32(i))

    var h_leaf_bounds = ctx.enqueue_create_host_buffer[DType.float32](
        len(leaf_bounds)
    )
    var h_payloads = ctx.enqueue_create_host_buffer[DType.uint32](len(payloads))
    var d_leaf_bounds = ctx.enqueue_create_buffer[DType.float32](
        len(leaf_bounds)
    )
    var d_payloads = ctx.enqueue_create_buffer[DType.uint32](len(payloads))
    h_leaf_bounds.enqueue_copy_from(leaf_bounds)
    h_payloads.enqueue_copy_from(payloads)

    h_leaf_bounds.enqueue_copy_to(d_leaf_bounds)
    h_payloads.enqueue_copy_to(d_payloads)

    return (d_leaf_bounds^, d_payloads^)


def _assert_gpu_bounds_width[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
](verts: List[Point3f32[Frame.WORLD]]) raises:
    with DeviceContext() as ctx:
        var build = _make_triangle_leaf_bounds(ctx, verts)
        var leaf_bounds = build[0].copy()
        var payloads = build[1].copy()

        var bvh = GpuWideBoundsBvh[node_width, leaf_width](ctx, len(payloads))
        var diagnostic = build_bounds_bvh_for_diagnostics(
            ctx, bvh, leaf_bounds, payloads
        )
        var validation = validate_binary_bvh(
            diagnostic.binary,
            diagnostic.workspace,
            diagnostic.binary.root_bounds(),
        )
        var binary_quality = measure_binary_bvh_quality(diagnostic.binary)
        var wide_quality = measure_wide_bvh_quality(bvh)

        assert_true(validation.sorted_ok, "generic bounds keys sorted")
        assert_true(validation.values_ok, "generic bounds values valid")
        assert_true(validation.topology_ok, "generic bounds topology valid")
        assert_true(validation.bounds_ok, "generic bounds refit valid")
        assert_true(
            bvh.leaf_block_count > 0, "wide collapse produced leaf blocks"
        )
        assert_true(bvh.leaf_block_count <= bvh.max_leaf_blocks)
        assert_true(binary_quality.quality > 0.0)
        assert_true(wide_quality.quality > 0.0)
        assert_true(binary_quality.primitives == len(payloads))
        assert_true(wide_quality.primitives == len(payloads))
        assert_true(binary_quality.internal_nodes == max(len(payloads) - 1, 0))
        assert_true(wide_quality.internal_nodes > 0)
        assert_true(wide_quality.internal_nodes <= bvh.node_count)
        assert_true(wide_quality.leaf_references == bvh.leaf_block_count)


def _assert_wide_lane_invariants[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
](verts: List[Point3f32[Frame.WORLD]]) raises:
    with DeviceContext() as ctx:
        var build = _make_triangle_leaf_bounds(ctx, verts)
        var leaf_bounds = build[0].copy()
        var payloads = build[1].copy()

        var bvh = GpuWideBoundsBvh[node_width, leaf_width](ctx, len(payloads))
        _ = build_bounds_bvh(ctx, bvh, leaf_bounds, payloads)

        var seen_live_lane = False
        with bvh.wide_nodes.map_to_host() as wide_nodes:
            for n in range(bvh.node_count):
                for lane in range(node_width):
                    var lane_meta = _wide_node_load_meta[node_width](
                        wide_nodes.unsafe_ptr(),
                        UInt32(n),
                        lane,
                    )
                    var count = _wide_meta_count(lane_meta)
                    var data = _wide_meta_data(lane_meta)

                    if count == EMPTY_LANE:
                        continue

                    seen_live_lane = True
                    if count == 0:
                        assert_true(data < UInt32(bvh.node_count))
                    else:
                        assert_true(count <= UInt32(leaf_width))
                        assert_true(data < UInt32(bvh.leaf_block_count))

        assert_true(seen_live_lane, "wide collapse had no live lanes")


def _assert_gpu_triangle_matches_cpu_camera[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
](verts: List[Point3f32[Frame.WORLD]]) raises:
    var cpu_bvh = TriangleBvh[Frame.WORLD, node_width, leaf_width].__init__[
        "lbvh"
    ](verts)
    var camera_data = _make_camera_rays_and_params(
        cpu_bvh.bounds(),
        GPU_BOUNDS_TEST_WIDTH,
        GPU_BOUNDS_TEST_HEIGHT,
        GPU_BOUNDS_TEST_VIEWS,
    )
    var rays = camera_data[0].copy()
    var camera_params = camera_data[1].copy()
    var cpu_checksum = _trace_cpu_triangle_bvh[
        Frame.WORLD, node_width, leaf_width
    ](cpu_bvh, rays)

    with DeviceContext() as ctx:
        var d_verts = upload_vertices(ctx, verts)
        var gpu_bvh = build_triangle_bvh[Frame.WORLD, node_width, leaf_width](
            ctx, d_verts
        )
        var d_camera = upload_list(ctx, camera_params)
        var d_hits = ctx.enqueue_create_buffer[DType.float32](
            len(rays) * Hit.STRIDE
        )

        gpu_bvh.launch_camera(
            ctx,
            d_camera,
            d_hits,
            len(rays),
            GPU_BOUNDS_TEST_WIDTH,
            GPU_BOUNDS_TEST_HEIGHT,
        )
        ctx.synchronize()

        var d_stats = ctx.enqueue_create_buffer[DType.uint32](
            len(rays) * GpuTraversalStats.STRIDE
        )
        gpu_bvh.launch_camera_instrumented(
            ctx,
            d_camera,
            d_hits,
            d_stats,
            len(rays),
            GPU_BOUNDS_TEST_WIDTH,
            GPU_BOUNDS_TEST_HEIGHT,
        )
        ctx.synchronize()

        var total_node_visits = UInt64(0)
        var total_primitive_tests = UInt64(0)
        with d_stats.map_to_host() as stats:
            for i in range(len(rays)):
                var base = i * GpuTraversalStats.STRIDE
                var node_visits = stats[base + GpuTraversalStats.NODE_VISITS]
                assert_true(
                    node_visits > 0, "instrumented traversal visits root"
                )
                total_node_visits += UInt64(node_visits)
                total_primitive_tests += UInt64(
                    stats[base + GpuTraversalStats.PRIMITIVE_TESTS]
                )
        assert_true(total_node_visits >= UInt64(len(rays)))
        assert_true(total_primitive_tests > 0)

        var gpu_result = _download_hit_checksum(d_hits, len(rays))
        var gpu_checksum = gpu_result[0]
        var gpu_hits = gpu_result[1]
        var mismatch_count = UInt32(0)

        with d_hits.map_to_host() as hf:
            var gpu_hits = Span(unsafe_ptr=hf.unsafe_ptr(), length=len(hf))
            for i in range(len(rays)):
                var gpu_hit = Hit[Frame.WORLD].load(gpu_hits, i)
                var cpu_hit = cpu_bvh.trace[TRACE.CLOSEST_HIT](rays[i])
                var gpu_t = gpu_hit.t
                var gpu_prim = gpu_hit.prim
                var cpu_t = cpu_hit.t
                var cpu_prim = cpu_hit.prim
                var same_miss = cpu_t >= f32_max and gpu_t >= f32_max
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
                t"node_width={Int(node_width)} leaf_width={Int(leaf_width)} "
                t"gpu={gpu_checksum} cpu={cpu_checksum} "
                t"diff={diff} mismatches={mismatch_count} hits={gpu_hits}"
            )
        assert_true(diff <= GPU_BOUNDS_TEST_EPS, "GpuTriangleBvh checksum")
        assert_true(mismatch_count == 0, "GpuTriangleBvh primitive/t mismatch")

        var d_rays = upload_rays(ctx, rays)
        gpu_bvh.launch_rays(ctx, d_rays, d_hits, len(rays))
        ctx.synchronize()
        var ray_result = _download_hit_checksum(d_hits, len(rays))
        assert_true(
            abs(ray_result[0] - cpu_checksum) <= GPU_BOUNDS_TEST_EPS,
            "packed-ray GpuTriangleBvh checksum",
        )

        comptime if node_width == 2 and leaf_width == 2:
            gpu_bvh.launch_rays[mode=TRACE.ANY_HIT](
                ctx, d_rays, d_hits, len(rays)
            )
            ctx.synchronize()
            with d_hits.map_to_host() as hf:
                var packed_hits = Span(
                    unsafe_ptr=hf.unsafe_ptr(), length=len(hf)
                )
                for i in range(len(rays)):
                    var gpu_hit = Hit[Frame.WORLD].load(packed_hits, i)
                    var cpu_hit = cpu_bvh.trace[TRACE.ANY_HIT](rays[i])
                    assert_true(
                        gpu_hit.is_occluded() == cpu_hit.is_occluded(),
                        "packed-ray any-hit mismatch",
                    )


def _assert_async_gpu_triangle_matches_cpu_camera[
    build_method: GpuBvhBuildMethod,
](verts: List[Point3f32[Frame.WORLD]]) raises:
    var cpu_bvh = TriangleBvh[Frame.WORLD, 2, 2].__init__["lbvh"](verts)
    var camera_data = _make_camera_rays_and_params(
        cpu_bvh.bounds(),
        GPU_BOUNDS_TEST_WIDTH,
        GPU_BOUNDS_TEST_HEIGHT,
        GPU_BOUNDS_TEST_VIEWS,
    )
    var rays = camera_data[0].copy()
    var camera_params = camera_data[1].copy()
    var cpu_checksum = _trace_cpu_triangle_bvh[Frame.WORLD, 2, 2](cpu_bvh, rays)

    with DeviceContext() as ctx:
        var d_verts = upload_vertices(ctx, verts)
        var pending = enqueue_build_triangle_bvh[
            Frame.WORLD, 2, 2, build_method
        ](ctx, d_verts)
        var gpu_bvh = pending^.wait(ctx)
        var d_camera = upload_list(ctx, camera_params)
        var d_hits = ctx.enqueue_create_buffer[DType.float32](
            len(rays) * Hit.STRIDE
        )
        gpu_bvh.launch_camera(
            ctx,
            d_camera,
            d_hits,
            len(rays),
            GPU_BOUNDS_TEST_WIDTH,
            GPU_BOUNDS_TEST_HEIGHT,
        )
        ctx.synchronize()

        var gpu_result = _download_hit_checksum(d_hits, len(rays))
        assert_true(gpu_bvh.tree.node_count > 0)
        assert_true(gpu_bvh.tree.leaf_block_count > 0)
        assert_true(
            abs(gpu_result[0] - cpu_checksum) <= GPU_BOUNDS_TEST_EPS,
            "asynchronous GpuTriangleBvh checksum",
        )


def _assert_sphere_matches_bruteforce_camera[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
](spheres: List[Sphere[Frame.WORLD]]) raises:
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
        var gpu_bvh = build_sphere_bvh[Frame.WORLD, node_width, leaf_width](
            ctx, spheres
        )
        var d_camera = upload_list(ctx, camera_params)
        var d_hits = ctx.enqueue_create_buffer[DType.float32](
            len(rays) * Hit.STRIDE
        )

        gpu_bvh.launch_camera(
            ctx,
            d_camera,
            d_hits,
            len(rays),
            GPU_BOUNDS_TEST_WIDTH,
            GPU_BOUNDS_TEST_HEIGHT,
        )
        ctx.synchronize()

        var gpu_result = _download_hit_checksum(d_hits, len(rays))
        var gpu_checksum = gpu_result[0]
        var hit_count = gpu_result[1]

        var diff = abs(gpu_checksum - cpu_checksum)
        if diff > GPU_BOUNDS_TEST_EPS:
            print(
                t"node_width={Int(node_width)} leaf_width={Int(leaf_width)} "
                t"gpu={gpu_checksum} cpu={cpu_checksum} "
                t"diff={diff} hits={hit_count}"
            )
        assert_true(diff <= GPU_BOUNDS_TEST_EPS, "GpuSphereBvh checksum")


def _assert_gpu_sphere_bounds[
    node_width: SIMDLength,
    leaf_width: SIMDLength = node_width,
](spheres: List[Sphere[Frame.WORLD]]) raises:
    with DeviceContext() as ctx:
        var build = _make_sphere_leaf_bounds(ctx, spheres)
        var leaf_bounds = build[0].copy()
        var payloads = build[1].copy()

        var bvh = GpuWideBoundsBvh[node_width, leaf_width](ctx, len(payloads))
        var diagnostic = build_bounds_bvh_for_diagnostics(
            ctx, bvh, leaf_bounds, payloads
        )
        var validation = validate_binary_bvh(
            diagnostic.binary, diagnostic.workspace, bvh.root_bounds()
        )

        assert_true(validation.sorted_ok, "sphere bounds keys sorted")
        assert_true(validation.values_ok, "sphere bounds values valid")
        assert_true(validation.topology_ok, "sphere bounds topology valid")
        assert_true(validation.bounds_ok, "sphere bounds refit valid")
        assert_true(bvh.node_count > 0, "sphere wide collapse produced nodes")
        assert_true(
            bvh.leaf_block_count > 0,
            "sphere wide collapse produced leaf blocks",
        )


def test_gpu_bounds_bvh_build_validate_small_scene() raises:
    var scene = _make_small_scene[Frame.WORLD]()
    comptime for N in [2, 4, 8]:
        _assert_gpu_bounds_width[N](scene)


def test_gpu_bounds_bvh_single_triangle() raises:
    var scene = _make_single_triangle_scene[Frame.WORLD]()
    comptime for N in [2, 4, 8]:
        _assert_gpu_bounds_width[N](scene)
        _assert_gpu_triangle_matches_cpu_camera[N](scene)


def test_gpu_bounds_bvh_duplicate_morton_codes() raises:
    var scene = _make_duplicate_centroid_scene[Frame.WORLD]()
    comptime for N in [2, 4, 8]:
        _assert_gpu_bounds_width[N](scene)
        _assert_wide_lane_invariants[N](scene)


def test_gpu_bounds_bvh_degenerate_axis() raises:
    var scene = _make_degenerate_axis_scene()
    comptime for N in [2, 4, 8]:
        _assert_gpu_bounds_width[N](scene)
        _assert_wide_lane_invariants[N](scene)


def test_gpu_bounds_bvh_wide_lane_invariants() raises:
    var scene = _make_small_scene[Frame.WORLD]()
    comptime for N in [2, 4, 8]:
        _assert_wide_lane_invariants[N](scene)


def test_gpu_triangle_bvh_camera_primary_matches_cpu() raises:
    var scene = _make_small_scene[Frame.WORLD]()
    comptime for N in [2, 4, 8]:
        _assert_gpu_triangle_matches_cpu_camera[N](scene)


def test_gpu_sphere_bvh_camera_primary_matches_bruteforce() raises:
    var scene = _make_small_sphere_scene[Frame.WORLD]()
    comptime for N in [2, 4, 8]:
        _assert_sphere_matches_bruteforce_camera[N](scene)


def test_gpu_sphere_bvh_single_sphere() raises:
    var scene = _make_single_sphere_scene[Frame.WORLD]()
    comptime for N in [2, 4, 8]:
        _assert_gpu_sphere_bounds[N](scene)
        _assert_sphere_matches_bruteforce_camera[N](scene)


def test_gpu_sphere_bvh_duplicate_morton_codes() raises:
    var scene = _make_duplicate_sphere_centroid_scene[Frame.WORLD]()
    comptime for N in [2, 4, 8]:
        _assert_gpu_sphere_bounds[N](scene)
        _assert_sphere_matches_bruteforce_camera[N](scene)


def test_gpu_bvh_independent_node_and_leaf_widths() raises:
    var triangles = _make_small_scene[Frame.WORLD]()
    var spheres = _make_small_sphere_scene[Frame.WORLD]()

    _assert_gpu_bounds_width[2, 4](triangles)
    _assert_wide_lane_invariants[2, 4](triangles)
    _assert_gpu_triangle_matches_cpu_camera[2, 4](triangles)
    _assert_sphere_matches_bruteforce_camera[2, 4](spheres)

    _assert_gpu_bounds_width[4, 2](triangles)
    _assert_wide_lane_invariants[4, 2](triangles)
    _assert_gpu_triangle_matches_cpu_camera[4, 2](triangles)
    _assert_sphere_matches_bruteforce_camera[4, 2](spheres)


def test_gpu_triangle_bvh_async_build_methods() raises:
    var scene = _make_small_scene[Frame.WORLD]()
    _assert_async_gpu_triangle_matches_cpu_camera[GpuBvhBuildMethod.LBVH](scene)
    _assert_async_gpu_triangle_matches_cpu_camera[GpuBvhBuildMethod.HPLOC](
        scene
    )


def test_gpu_triangle_bvh_reuses_build_arena() raises:
    var scene = _make_small_scene[Frame.WORLD]()
    with DeviceContext() as ctx:
        var vertices = upload_vertices(ctx, scene)
        var triangle_count = len(scene) / 3
        var arena = GpuTriangleBvhBuildArena(ctx, triangle_count)

        var first = enqueue_build_triangle_bvh_with_arena[
            Frame.WORLD, 2, 2, GpuBvhBuildMethod.LBVH
        ](ctx, vertices, arena)
        var lbvh = first^.wait(ctx)
        assert_true(lbvh.tree.node_count > 0)

        var second = enqueue_build_triangle_bvh_with_arena[
            Frame.WORLD, 2, 2, GpuBvhBuildMethod.HPLOC
        ](ctx, vertices, arena)
        var hploc = second^.wait(ctx)
        assert_true(hploc.tree.node_count > 0)


def main() raises:
    comptime if not has_accelerator():
        raise "No Accelerator found"
    TestSuite.discover_tests[__functions_in_module()]().run()
