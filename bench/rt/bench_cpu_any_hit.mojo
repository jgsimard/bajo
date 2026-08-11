from std.math import round
from std.time import perf_counter_ns

from bajo.bvh.camera import Camera
from bajo.bvh.types import Instance, Sphere
from bajo.core import Frame, Point3f32, Rayf32, Vec3f32
from bajo.core.random import Rng, random_on_hemisphere
from bajo.core.utils import ns_to_mrays_per_s
from bajo.rt import Color, SurfaceId, SurfaceStore, World
from bench.bvh.bench_cpu_bvh_grid import (
    make_grid_triangles,
    make_hit_and_miss_rays,
)
from examples.rtiaw import make_weekend_world


comptime TIMING_REPEATS = 6
comptime AO_WIDTH = 320
comptime AO_HEIGHT = 180


@fieldwise_init
struct TimingResult(Copyable):
    var ns: Int
    var hits: Int


def query_closest(world: World, rays: List[Rayf32[Frame.WORLD]]) -> Int:
    var hits = 0
    for ray in rays:
        if world.trace(ray):
            hits += 1
    return hits


def query_any(world: World, rays: List[Rayf32[Frame.WORLD]]) -> Int:
    var hits = 0
    for ray in rays:
        if world.occluded(ray):
            hits += 1
    return hits


def time_closest(world: World, rays: List[Rayf32[Frame.WORLD]]) -> TimingResult:
    var hits = query_closest(world, rays)
    var best_ns = Int.MAX
    for _ in range(TIMING_REPEATS):
        var start = perf_counter_ns()
        hits = query_closest(world, rays)
        var elapsed = Int(perf_counter_ns() - start)
        if elapsed < best_ns:
            best_ns = elapsed
    return TimingResult(best_ns, hits)


def time_any(world: World, rays: List[Rayf32[Frame.WORLD]]) -> TimingResult:
    var hits = query_any(world, rays)
    var best_ns = Int.MAX
    for _ in range(TIMING_REPEATS):
        var start = perf_counter_ns()
        hits = query_any(world, rays)
        var elapsed = Int(perf_counter_ns() - start)
        if elapsed < best_ns:
            best_ns = elapsed
    return TimingResult(best_ns, hits)


def print_case(
    label: String, world: World, rays: List[Rayf32[Frame.WORLD]]
) raises:
    var closest = time_closest(world, rays)
    var any = time_any(world, rays)
    print(t"\n{label}: {len(rays)} visibility rays")
    print(
        t"  closest/materialized:"
        t" {round(ns_to_mrays_per_s(closest.ns, len(rays)), 3)} MRay/s"
    )
    print(t"  any-hit: {round(ns_to_mrays_per_s(any.ns, len(rays)), 3)} MRay/s")
    print(t"  speedup: {round(Float64(closest.ns) / Float64(any.ns), 3)}x")
    print(t"  hit-count delta: {any.hits - closest.hits}")


def make_weekend_ao_rays(
    world: World,
) -> List[Rayf32[Frame.WORLD]]:
    var camera = Camera.from_vfov(
        Point3f32[Frame.WORLD](13.0, 2.0, 3.0),
        Point3f32[Frame.WORLD](0.0, 0.0, 0.0),
        Vec3f32[Frame.WORLD](0.0, 1.0, 0.0),
        20.0,
        10.0,
        0.0,
    )
    var rays = List[Rayf32[Frame.WORLD]](capacity=AO_WIDTH * AO_HEIGHT)
    for py in range(AO_HEIGHT):
        for px in range(AO_WIDTH):
            var primary = camera.make_ray(px, py, AO_WIDTH, AO_HEIGHT)
            var hit = world.trace(primary)
            if hit:
                ref record = hit.value()
                var rng = Rng(seed=UInt64(2026), id=UInt64(py * AO_WIDTH + px))
                var direction = random_on_hemisphere[Frame.WORLD](
                    rng, record.normal
                )
                rays.append(
                    Rayf32[Frame.WORLD](
                        record.p,
                        direction,
                        0.001,
                        4.0,
                    )
                )
    return rays^


def make_triangle_world() -> World:
    var surfaces = SurfaceStore()
    var matte = surfaces.add_lambertian(Color(0.5))
    var vertices = make_grid_triangles()
    var triangle_surfaces = List[SurfaceId](
        length=len(vertices) / 3, fill=matte
    )
    return World(
        List[Sphere[Frame.WORLD]](),
        List[SurfaceId](),
        vertices^,
        triangle_surfaces^,
        List[List[Point3f32[Frame.LOCAL]]](),
        List[Instance](),
        List[SurfaceId](),
        surfaces^,
    )


def make_bounded_grid_rays() -> List[Rayf32[Frame.WORLD]]:
    var source = make_hit_and_miss_rays()
    var rays = List[Rayf32[Frame.WORLD]](capacity=len(source))
    for ray in source:
        rays.append(Rayf32[Frame.WORLD](ray.o, ray.d, ray.t_min, Float32(3.0)))
    return rays^


def main() raises:
    print("CPU renderer visibility query benchmark")
    print("best of six; finite-distance rays; complete World API")

    var sphere_world = make_weekend_world()
    var ao_rays = make_weekend_ao_rays(sphere_world)
    print_case("Weekend sphere scene AO", sphere_world, ao_rays)

    var triangle_world = make_triangle_world()
    var triangle_rays = make_bounded_grid_rays()
    print_case("Regular triangle scene, 75% hit", triangle_world, triangle_rays)
