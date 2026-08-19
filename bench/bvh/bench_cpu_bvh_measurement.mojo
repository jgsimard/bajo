from std.math import round
from std.time import perf_counter_ns

from bajo.bvh.constants import TRACE, f32_max
from bajo.bvh.cpu.trace import CpuBvhTraversalStats
from bajo.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.bvh.host_utils import compute_bounds
from bajo.core import Frame, Point3f32, Vec3f32, Rayf32
from bajo.core.utils import ns_to_mrays_per_s
from bajo.parser.obj.pack import pack_obj_triangles
from bench.bvh.fixtures import (
    make_camera_rays_and_params,
    make_depth_overlap_rays,
    make_depth_overlap_triangles,
    make_grid_triangles,
    make_hit_and_miss_rays,
    permute_rays,
    select_and_repeat_hit_rays,
)
from bench.timing import ratio


comptime OBJ_PATH = "./assets/dragon/dragon.obj"
comptime RAY_WIDTH = 512
comptime RAY_HEIGHT = 288
comptime TIMING_REPEATS = 4


@fieldwise_init
struct TimingResult(Copyable):
    var ns: Int
    var checksum: Float64
    var hits: Int


def trace_normal[
    mode: TRACE
](
    bvh: TriangleBvh[Frame.WORLD, 16, 16],
    rays: List[Rayf32[Frame.WORLD]],
) -> Tuple[Float64, Int]:
    var checksum = 0.0
    var hits = 0
    for ray in rays:
        var hit = bvh.trace[mode](ray)
        comptime if mode == TRACE.CLOSEST_HIT:
            if hit.t < f32_max:
                checksum += Float64(hit.t) + Float64(hit.prim)
                hits += 1
        else:
            if hit.is_occluded():
                checksum += 1.0
                hits += 1
    return (checksum, hits)


def time_normal[
    mode: TRACE
](
    bvh: TriangleBvh[Frame.WORLD, 16, 16],
    rays: List[Rayf32[Frame.WORLD]],
) -> TimingResult:
    var summary = trace_normal[mode](bvh, rays)
    var best_ns = Int.MAX
    for _ in range(TIMING_REPEATS):
        var t0 = perf_counter_ns()
        summary = trace_normal[mode](bvh, rays)
        var elapsed = Int(perf_counter_ns() - t0)
        if elapsed < best_ns:
            best_ns = elapsed
    return TimingResult(best_ns, summary[0], summary[1])


def collect_stats[
    mode: TRACE
](
    bvh: TriangleBvh[Frame.WORLD, 16, 16],
    rays: List[Rayf32[Frame.WORLD]],
    mut stats: CpuBvhTraversalStats,
) -> Tuple[Float64, Int]:
    var checksum = 0.0
    var hits = 0
    for ray in rays:
        var hit = bvh.trace_with_stats[mode](ray, stats)
        comptime if mode == TRACE.CLOSEST_HIT:
            if hit.t < f32_max:
                checksum += Float64(hit.t) + Float64(hit.prim)
                hits += 1
        else:
            if hit.is_occluded():
                checksum += 1.0
                hits += 1
    return (checksum, hits)


def print_timing(label: String, result: TimingResult, ray_count: Int) raises:
    print(
        t"  {label}: {round(ns_to_mrays_per_s(result.ns, ray_count), 3)} MRay/s"
    )


def print_stats(label: String, stats: CpuBvhTraversalStats) raises:
    print(t"\n{label} traversal counters ({stats.rays} rays)")
    print(
        t"  internal nodes/ray:"
        t" {round(ratio(stats.internal_nodes, stats.rays), 3)}"
    )
    print(
        t"  nodes producing hits/ray:"
        t" {round(ratio(stats.nodes_with_hits, stats.rays), 3)}"
    )
    print(
        t"  AABB packet lanes/ray:"
        t" {round(ratio(stats.aabb_packet_lanes, stats.rays), 3)}"
    )
    print(
        t"  active child lanes/node:"
        t" {round(ratio(stats.active_child_lanes, stats.internal_nodes), 3)}"
    )
    print(
        t"  intersected child lanes/node:"
        t" {round(ratio(stats.aabb_hit_lanes, stats.internal_nodes), 3)}"
    )
    print(
        t"  AABB packet occupancy:"
        t" {round(100.0 * ratio(stats.active_child_lanes, stats.aabb_packet_lanes), 2)}%"
    )
    print(
        t"  leaf blocks/ray: {round(ratio(stats.leaf_blocks, stats.rays), 3)}"
    )
    print(
        t"  valid primitives/visited leaf:"
        t" {round(ratio(stats.valid_primitives, stats.leaf_blocks), 3)}"
    )
    print(
        t"  primitive packet occupancy:"
        t" {round(100.0 * ratio(stats.valid_primitives, stats.primitive_packet_lanes), 2)}%"
    )
    print(
        t"  triangle candidates/ray:"
        t" {round(ratio(stats.primitive_hit_candidates, stats.rays), 3)}"
    )
    print(
        t"  closer-hit updates/ray:"
        t" {round(ratio(stats.closer_hit_updates, stats.rays), 3)}"
    )
    print(
        t"  any-hit early exits/ray:"
        t" {round(ratio(stats.any_hit_early_exits, stats.rays), 3)}"
    )
    print(
        t"  stack pushes/ray: {round(ratio(stats.stack_pushes, stats.rays), 3)}"
    )
    print(
        t"  insertion shifts/push:"
        t" {round(ratio(stats.stack_insertion_shifts, stats.stack_pushes), 3)}"
    )
    print(t"  stack pops/ray: {round(ratio(stats.stack_pops, stats.rays), 3)}")
    print(
        t"  bulk-pruned tasks/ray:"
        t" {round(ratio(stats.stack_pruned_tasks, stats.rays), 3)}"
    )
    print(t"  maximum stack depth: {stats.max_stack_depth}")


def run_case[
    mode: TRACE
](
    label: String,
    bvh: TriangleBvh[Frame.WORLD, 16, 16],
    rays: List[Rayf32[Frame.WORLD]],
) raises:
    print(t"\n{label}: {bvh.tri_count} triangles, {len(rays)} rays")
    var permuted = permute_rays(rays)

    var coherent_timing = time_normal[mode](bvh, rays)
    var permuted_timing = time_normal[mode](bvh, permuted)
    print(
        t"  hit rate:"
        t" {round(100.0 * ratio(coherent_timing.hits, len(rays)), 2)}%"
    )
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
    var measured = collect_stats[mode](bvh, rays, stats)
    print(
        t"  measured checksum delta:"
        t" {round(measured[0] - coherent_timing.checksum, 6)}"
    )
    print(t"  measured hit-count delta: {measured[1] - coherent_timing.hits}")
    print_stats(label, stats)


def main() raises:
    print("CPU BVH measurement before micro-optimization")
    print("BVH16 / triangle packets of 16 / SAH")

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
    var dragon_bvh = TriangleBvh[Frame.WORLD, 16, 16].__init__["sah"](
        dragon_vertices
    )
    run_case[TRACE.CLOSEST_HIT](
        "Dragon camera (natural hit rate)", dragon_bvh, dragon_rays
    )

    var dragon_hit_rays = select_and_repeat_hit_rays(dragon_bvh, dragon_rays)
    run_case[TRACE.CLOSEST_HIT](
        "Dragon visible-surface rays (forced high hit rate)",
        dragon_bvh,
        dragon_hit_rays,
    )
    run_case[TRACE.ANY_HIT](
        "Dragon visible-surface shadow rays (any-hit)",
        dragon_bvh,
        dragon_hit_rays,
    )

    var grid_vertices = make_grid_triangles()
    var grid_rays = make_hit_and_miss_rays()
    var grid_bvh = TriangleBvh[Frame.WORLD, 16, 16].__init__["sah"](
        grid_vertices
    )
    run_case[TRACE.CLOSEST_HIT]("Regular grid", grid_bvh, grid_rays)

    var depth_vertices = make_depth_overlap_triangles()
    var depth_rays = make_depth_overlap_rays()
    var depth_bvh = TriangleBvh[Frame.WORLD, 16, 16].__init__["sah"](
        depth_vertices
    )
    run_case[TRACE.CLOSEST_HIT](
        "Layered overlap (pending-stack stress)", depth_bvh, depth_rays
    )
