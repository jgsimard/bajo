from std.math import round
from std.time import perf_counter_ns

from bajo.bvh.constants import TRACE, f32_max
from bajo.bvh.cpu.trace import CpuBvhTraversalStats
from bajo.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.bvh.host_utils import compute_bounds
from bajo.core import Frame, Point3f32, Rayf32
from bajo.core.utils import ns_to_mrays_per_s
from bajo.obj.pack import pack_obj_triangles
from bench.bvh.bench_cpu_bvh_grid import (
    make_grid_triangles,
    make_hit_and_miss_rays,
)
from bench.bvh.fixtures import make_camera_rays_and_params


comptime OBJ_PATH = "./assets/dragon/dragon.obj"
comptime RAY_WIDTH = 512
comptime RAY_HEIGHT = 288
comptime TIMING_REPEATS = 4


@fieldwise_init
struct TimingResult(Copyable):
    var ns: Int
    var checksum: Float64


def _ratio(numerator: Int, denominator: Int) -> Float64:
    if denominator == 0:
        return 0.0
    return Float64(numerator) / Float64(denominator)


def trace_normal(
    bvh: TriangleBvh[Frame.WORLD, 16, 16],
    rays: List[Rayf32[Frame.WORLD]],
) -> Float64:
    var checksum = 0.0
    for ray in rays:
        var hit = bvh.trace[TRACE.CLOSEST_HIT](ray)
        if hit.t < f32_max:
            checksum += Float64(hit.t) + Float64(hit.prim)
    return checksum


def time_normal(
    bvh: TriangleBvh[Frame.WORLD, 16, 16],
    rays: List[Rayf32[Frame.WORLD]],
) -> TimingResult:
    var checksum = trace_normal(bvh, rays)
    var best_ns = Int.MAX
    for _ in range(TIMING_REPEATS):
        var t0 = perf_counter_ns()
        checksum = trace_normal(bvh, rays)
        var elapsed = Int(perf_counter_ns() - t0)
        if elapsed < best_ns:
            best_ns = elapsed
    return TimingResult(best_ns, checksum)


def collect_stats(
    bvh: TriangleBvh[Frame.WORLD, 16, 16],
    rays: List[Rayf32[Frame.WORLD]],
    mut stats: CpuBvhTraversalStats,
) -> Float64:
    var checksum = 0.0
    for ray in rays:
        var hit = bvh.trace_with_stats[TRACE.CLOSEST_HIT](ray, stats)
        if hit.t < f32_max:
            checksum += Float64(hit.t) + Float64(hit.prim)
    return checksum


def permute_rays(
    rays: List[Rayf32[Frame.WORLD]],
) -> List[Rayf32[Frame.WORLD]]:
    """Return a deterministic cache-unfriendly permutation of the same rays."""
    # 104729 is prime and coprime to both benchmark ray counts.
    var permuted = List[Rayf32[Frame.WORLD]](capacity=len(rays))
    for i in range(len(rays)):
        var source_idx = (i * 104729) % len(rays)
        permuted.append(rays[source_idx].copy())
    return permuted^


def print_timing(label: String, result: TimingResult, ray_count: Int) raises:
    print(
        t"  {label}: {round(ns_to_mrays_per_s(result.ns, ray_count), 3)} MRay/s"
    )


def print_stats(label: String, stats: CpuBvhTraversalStats) raises:
    print(t"\n{label} traversal counters ({stats.rays} rays)")
    print(
        t"  internal nodes/ray:"
        t" {round(_ratio(stats.internal_nodes, stats.rays), 3)}"
    )
    print(
        t"  nodes producing hits/ray:"
        t" {round(_ratio(stats.nodes_with_hits, stats.rays), 3)}"
    )
    print(
        t"  AABB packet lanes/ray:"
        t" {round(_ratio(stats.aabb_packet_lanes, stats.rays), 3)}"
    )
    print(
        t"  active child lanes/node:"
        t" {round(_ratio(stats.active_child_lanes, stats.internal_nodes), 3)}"
    )
    print(
        t"  intersected child lanes/node:"
        t" {round(_ratio(stats.aabb_hit_lanes, stats.internal_nodes), 3)}"
    )
    print(
        t"  AABB packet occupancy:"
        t" {round(100.0 * _ratio(stats.active_child_lanes, stats.aabb_packet_lanes), 2)}%"
    )
    print(
        t"  leaf blocks/ray: {round(_ratio(stats.leaf_blocks, stats.rays), 3)}"
    )
    print(
        t"  valid primitives/visited leaf:"
        t" {round(_ratio(stats.valid_primitives, stats.leaf_blocks), 3)}"
    )
    print(
        t"  primitive packet occupancy:"
        t" {round(100.0 * _ratio(stats.valid_primitives, stats.primitive_packet_lanes), 2)}%"
    )
    print(
        t"  triangle candidates/ray:"
        t" {round(_ratio(stats.primitive_hit_candidates, stats.rays), 3)}"
    )
    print(
        t"  closer-hit updates/ray:"
        t" {round(_ratio(stats.closer_hit_updates, stats.rays), 3)}"
    )
    print(
        t"  stack pushes/ray:"
        t" {round(_ratio(stats.stack_pushes, stats.rays), 3)}"
    )
    print(
        t"  insertion shifts/push:"
        t" {round(_ratio(stats.stack_insertion_shifts, stats.stack_pushes), 3)}"
    )
    print(t"  stack pops/ray: {round(_ratio(stats.stack_pops, stats.rays), 3)}")
    print(
        t"  bulk-pruned tasks/ray:"
        t" {round(_ratio(stats.stack_pruned_tasks, stats.rays), 3)}"
    )
    print(t"  maximum stack depth: {stats.max_stack_depth}")


def run_case(
    label: String,
    vertices: List[Point3f32[Frame.WORLD]],
    rays: List[Rayf32[Frame.WORLD]],
) raises:
    print(t"\n{label}: {len(vertices) / 3} triangles, {len(rays)} rays")
    var bvh = TriangleBvh[Frame.WORLD, 16, 16].__init__["sah"](vertices)
    var permuted = permute_rays(rays)

    var coherent_timing = time_normal(bvh, rays)
    var permuted_timing = time_normal(bvh, permuted)
    print_timing("coherent", coherent_timing, len(rays))
    print_timing("permuted", permuted_timing, len(permuted))
    print(
        t"  order slowdown:"
        t" {round(Float64(permuted_timing.ns) / Float64(coherent_timing.ns), 3)}x"
    )
    print(
        t"  timing checksum delta:"
        t" {round(permuted_timing.checksum - coherent_timing.checksum, 6)}"
    )

    # Traversal work is ray-local, so a permutation has identical counters.
    # Collect once in coherent order; the separate timings expose cache effects.
    var stats = CpuBvhTraversalStats()
    var measured_checksum = collect_stats(bvh, rays, stats)
    print(
        t"  measured checksum delta:"
        t" {round(measured_checksum - coherent_timing.checksum, 6)}"
    )
    print_stats(label, stats)


def main() raises:
    print("CPU BVH measurement before micro-optimization")
    print("BVH16 / triangle packets of 16 / SAH / closest-hit")

    var dragon_vertices = pack_obj_triangles[Frame.WORLD](OBJ_PATH)
    var dragon_bounds = compute_bounds(dragon_vertices)
    var camera = make_camera_rays_and_params(
        dragon_bounds,
        RAY_WIDTH,
        RAY_HEIGHT,
        1,
        0.2,
    )
    var dragon_rays = camera[0].copy()
    run_case("Dragon camera", dragon_vertices, dragon_rays)

    var grid_vertices = make_grid_triangles()
    var grid_rays = make_hit_and_miss_rays()
    run_case("Regular grid", grid_vertices, grid_rays)
