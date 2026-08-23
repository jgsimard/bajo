"""Diagnostic H-PLOC quality, build, and traversal comparison."""

from std.math import abs, min, round
from std.sys import has_accelerator
from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh.gpu.diagnostics import build_bounds_bvh_for_diagnostics
from bajo.bvh.gpu.builder import GpuBvhBuildMethod
from bajo.bvh.gpu.quality import (
    measure_binary_bvh_quality,
    measure_wide_bvh_quality,
)
from bajo.bvh.gpu.trace import GpuTraversalStats
from bajo.bvh.gpu.triangle_bvh import (
    build_triangle_bvh,
    build_triangle_bvh_measured,
)
from bajo.bvh.gpu.utils import (
    GpuBuildTimings,
    _download_full_hit_checksum,
    upload_list,
    upload_vertices,
)
from bajo.bvh.host_utils import compute_bounds
from bajo.bvh.types import Hit
from bajo.core import Frame, Point3f32
from bajo.core.utils import ns_to_mrays_per_s, ns_to_ms
from bajo.parser.obj.pack import pack_obj_triangles
from bajo.benchmark.bvh_fixtures import make_camera_rays_and_params
from bajo.benchmark.gpu_bvh_fixtures import flatten_triangle_bounds


comptime DRAGON_PATH = "./assets/dragon/dragon.obj"
comptime SYNTHETIC_TRIANGLES = 4096
comptime SYNTHETIC_WIDTH = 512
comptime SYNTHETIC_HEIGHT = 256
comptime SYNTHETIC_VIEWS = 1
comptime DRAGON_WIDTH = 1280
comptime DRAGON_HEIGHT = 640
comptime DRAGON_VIEWS = 3
comptime FOV_SCALE = 0.2
comptime BENCH_REPEATS = 8


@fieldwise_init
struct HplocBenchResult(Copyable, Writable):
    var label: String
    var build_ns: Int
    var timings: GpuBuildTimings
    var trace_ns: Int
    var ray_count: Int
    var binary_quality: Float64
    var wide_quality: Float64
    var wide_nodes: Int
    var leaf_blocks: Int
    var nodes_per_ray: Float64
    var leaves_per_ray: Float64
    var triangles_per_ray: Float64
    var hit_count: UInt32
    var checksum: Float64


def _make_synthetic_triangles(
    count: Int,
) -> List[Point3f32[.WORLD]]:
    var vertices = List[Point3f32[.WORLD]](capacity=count * 3)
    for i in range(count):
        var x = Float32((i % 64) * 4 - 126)
        var y = Float32(((i / 64) % 64) * 4 - 126)
        var z = Float32((i * 7) % 17)
        var scale = Float32(2.5) if i % 29 == 0 else Float32(0.7)
        vertices.append(Point3f32[.WORLD](x - scale, y - scale, z))
        vertices.append(Point3f32[.WORLD](x + scale, y - scale, z))
        vertices.append(Point3f32[.WORLD](x, y + scale, z + 0.1))
    return vertices^


def _run_case[
    method: GpuBvhBuildMethod,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
](
    mut ctx: DeviceContext,
    d_vertices: DeviceBuffer[.float32],
    d_leaf_bounds: DeviceBuffer[.float32],
    d_payloads: DeviceBuffer[.uint32],
    d_camera: DeviceBuffer[.float32],
    ray_count: Int,
    image_width: Int,
    image_height: Int,
    label: String,
) raises -> HplocBenchResult:
    comptime max_leaf_size = Int(leaf_width)
    # Warm every specialization before collecting build time.
    _ = build_triangle_bvh[.WORLD, node_width, leaf_width, method](
        ctx, d_vertices
    )
    ctx.synchronize()

    var best_build_ns = Int.MAX
    var best_timings = GpuBuildTimings(0, 0, 0, 0, 0, 0, 0)
    for _ in range(BENCH_REPEATS):
        var start = perf_counter_ns()
        var candidate_timings = GpuBuildTimings(0, 0, 0, 0, 0, 0, 0)
        _ = build_triangle_bvh_measured[
            .WORLD, node_width, leaf_width, method
        ](ctx, d_vertices, candidate_timings)
        ctx.synchronize()
        var elapsed = Int(perf_counter_ns() - start)
        if elapsed < best_build_ns:
            best_build_ns = elapsed
            best_timings = candidate_timings

    # Measure binary and final-wide quality independently of the timed build.
    var quality_diagnostic = build_bounds_bvh_for_diagnostics[
        node_width, leaf_width, max_leaf_size, method, True
    ](ctx, d_leaf_bounds.copy(), d_payloads.copy())
    ref quality_tree = quality_diagnostic.wide
    var binary_quality = measure_binary_bvh_quality(
        quality_diagnostic.build.binary
    )
    var wide_quality = measure_wide_bvh_quality(quality_tree)

    var bvh = build_triangle_bvh[.WORLD, node_width, leaf_width, method](
        ctx, d_vertices
    )
    var hits = ctx.enqueue_create_buffer[.float32](ray_count * Hit.STRIDE)
    bvh.launch_camera(
        ctx,
        d_camera,
        hits,
        ray_count,
        image_width,
        image_height,
    )
    ctx.synchronize()

    var best_trace_ns = Int.MAX
    for _ in range(BENCH_REPEATS):
        var start = perf_counter_ns()
        bvh.launch_camera(
            ctx,
            d_camera,
            hits,
            ray_count,
            image_width,
            image_height,
        )
        ctx.synchronize()
        best_trace_ns = min(best_trace_ns, Int(perf_counter_ns() - start))

    var hit_result = _download_full_hit_checksum(ctx, hits, ray_count)
    var stats = ctx.enqueue_create_buffer[.uint32](
        ray_count * GpuTraversalStats.STRIDE
    )
    bvh.launch_camera_instrumented(
        ctx,
        d_camera,
        hits,
        stats,
        ray_count,
        image_width,
        image_height,
    )
    ctx.synchronize()

    var node_visits = UInt64(0)
    var leaf_visits = UInt64(0)
    var primitive_tests = UInt64(0)
    with stats.map_to_host() as host_stats:
        for ray_idx in range(ray_count):
            var base = ray_idx * GpuTraversalStats.STRIDE
            node_visits += UInt64(
                host_stats[base + GpuTraversalStats.NODE_VISITS]
            )
            leaf_visits += UInt64(
                host_stats[base + GpuTraversalStats.LEAF_BLOCKS]
            )
            primitive_tests += UInt64(
                host_stats[base + GpuTraversalStats.PRIMITIVE_TESTS]
            )

    return HplocBenchResult(
        label,
        best_build_ns,
        best_timings,
        best_trace_ns,
        ray_count,
        binary_quality.quality,
        wide_quality.quality,
        quality_tree.node_count,
        quality_tree.leaf_block_count,
        Float64(node_visits) / Float64(ray_count),
        Float64(leaf_visits) / Float64(ray_count),
        Float64(primitive_tests) / Float64(ray_count),
        hit_result[1],
        hit_result[0],
    )


def _print_result(result: HplocBenchResult):
    print(
        result.label.ascii_ljust(17),
        String(round(result.binary_quality, 3)).ascii_rjust(8),
        String(round(result.wide_quality, 3)).ascii_rjust(8),
        String(round(ns_to_ms(result.build_ns), 3)).ascii_rjust(9),
        String(round(ns_to_ms(result.timings.topology_ns), 3)).ascii_rjust(9),
        String(round(ns_to_ms(result.timings.collapse_ns), 3)).ascii_rjust(9),
        String(round(ns_to_ms(result.trace_ns), 3)).ascii_rjust(9),
        String(
            round(ns_to_mrays_per_s(result.trace_ns, result.ray_count), 1)
        ).ascii_rjust(9),
        String(round(result.nodes_per_ray, 3)).ascii_rjust(9),
        String(round(result.triangles_per_ray, 3)).ascii_rjust(9),
        String(result.wide_nodes).ascii_rjust(9),
        String(result.leaf_blocks).ascii_rjust(9),
    )


def _validate_against(
    reference: HplocBenchResult, candidate: HplocBenchResult
) raises:
    if reference.hit_count != candidate.hit_count:
        raise String(
            t"{candidate.label}: hit count {candidate.hit_count} != "
            t"{reference.hit_count}"
        )
    if abs(reference.checksum - candidate.checksum) > 0.05:
        raise String(
            t"{candidate.label}: checksum {candidate.checksum} != "
            t"{reference.checksum}"
        )


def _run_scene(
    name: String,
    vertices: List[Point3f32[.WORLD]],
    image_width: Int,
    image_height: Int,
    views: Int,
) raises:
    var scene_bounds = compute_bounds(vertices)
    var camera = make_camera_rays_and_params(
        scene_bounds.unsafe_convert_frame[.WORLD](),
        image_width,
        image_height,
        views,
        FOV_SCALE,
    )
    var flat = flatten_triangle_bounds(vertices)

    print(t"\n{name}: {len(vertices) / 3} triangles, {len(camera[0])} rays")
    print(
        "case                  binQ    wideQ  build ms   topo ms "
        "  wide ms  trace ms    MRay/s nodes/ray  tris/ray     nodes    leaves"
    )
    print(
        "----------------- -------- -------- --------- --------- "
        "--------- --------- --------- --------- --------- --------- ---------"
    )

    with DeviceContext() as ctx:
        var d_vertices = upload_vertices(ctx, vertices)
        var d_leaf_bounds = upload_list(ctx, flat[0])
        var d_payloads = upload_list(ctx, flat[1])
        var d_camera = upload_list(ctx, camera[1])
        ctx.synchronize()

        var lbvh2l2 = _run_case[.LBVH, 2, 2](
            ctx,
            d_vertices,
            d_leaf_bounds,
            d_payloads,
            d_camera,
            len(camera[0]),
            image_width,
            image_height,
            "LBVH n2/l2",
        )
        var hploc2l2 = _run_case[.HPLOC, 2, 2](
            ctx,
            d_vertices,
            d_leaf_bounds,
            d_payloads,
            d_camera,
            len(camera[0]),
            image_width,
            image_height,
            "HPLOC n2/l2",
        )
        var lbvh2 = _run_case[.LBVH, 2, 4](
            ctx,
            d_vertices,
            d_leaf_bounds,
            d_payloads,
            d_camera,
            len(camera[0]),
            image_width,
            image_height,
            "LBVH n2/l4",
        )
        var hploc2 = _run_case[.HPLOC, 2, 4](
            ctx,
            d_vertices,
            d_leaf_bounds,
            d_payloads,
            d_camera,
            len(camera[0]),
            image_width,
            image_height,
            "HPLOC n2/l4",
        )
        var lbvh4 = _run_case[.LBVH, 4, 4](
            ctx,
            d_vertices,
            d_leaf_bounds,
            d_payloads,
            d_camera,
            len(camera[0]),
            image_width,
            image_height,
            "LBVH n4/l4",
        )
        var hploc4 = _run_case[.HPLOC, 4, 4](
            ctx,
            d_vertices,
            d_leaf_bounds,
            d_payloads,
            d_camera,
            len(camera[0]),
            image_width,
            image_height,
            "HPLOC n4/l4",
        )

        _print_result(lbvh2l2)
        _print_result(hploc2l2)
        _print_result(lbvh2)
        _print_result(hploc2)
        _print_result(lbvh4)
        _print_result(hploc4)

        _validate_against(lbvh2l2, hploc2l2)
        _validate_against(lbvh2l2, lbvh2)
        _validate_against(lbvh2l2, hploc2)
        _validate_against(lbvh2l2, lbvh4)
        _validate_against(lbvh2l2, hploc4)


def main() raises:
    comptime if not has_accelerator():
        raise "No compatible GPU found"

    print("GPU H-PLOC quality/build/traversal benchmark")
    print(t"best of {BENCH_REPEATS}; triangle performance only")
    _run_scene(
        "synthetic",
        _make_synthetic_triangles(SYNTHETIC_TRIANGLES),
        SYNTHETIC_WIDTH,
        SYNTHETIC_HEIGHT,
        SYNTHETIC_VIEWS,
    )
    _run_scene(
        "dragon",
        pack_obj_triangles[.WORLD](DRAGON_PATH),
        DRAGON_WIDTH,
        DRAGON_HEIGHT,
        DRAGON_VIEWS,
    )
