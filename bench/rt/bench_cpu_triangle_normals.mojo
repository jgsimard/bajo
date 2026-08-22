from std.math import round
from std.time import perf_counter_ns

from bajo.bvh.constants import TRACE
from bajo.bvh.cpu.blas_set import trace_blas_set
from bajo.core import Frame, GeoKind, Rayf32, cross, dot, normalize
from bajo.core.utils import ns_to_mrays_per_s
from bajo.rt import CpuScene
from bajo.rt.types import (
    HitRecord,
    PRIM,
    PrimitiveId,
    ray_at,
)
from bajo.benchmark.rt_fixtures import (
    make_bounded_grid_rays,
    make_grid_triangle_world,
)


comptime TIMING_REPEATS = 8


@fieldwise_init
struct TimingResult(Copyable):
    var ns: Int
    var checksum: Float64
    var hits: Int


def trace_recomputed_normal(
    world: CpuScene[], ray: Rayf32[Frame.WORLD]
) -> Optional[HitRecord]:
    var bvh_hit = trace_blas_set[16, 16, TRACE.CLOSEST_HIT, Frame.WORLD](
        world.triangle_bvh.value(), UInt32(0), ray
    )
    if not bvh_hit.is_hit():
        return None

    var tri_idx = Int(bvh_hit.prim)
    var base = tri_idx * 3
    ref v0 = world.scene_data().triangle_vertices[base + 0]
    ref v1 = world.scene_data().triangle_vertices[base + 1]
    ref v2 = world.scene_data().triangle_vertices[base + 2]
    var p = ray_at(ray, bvh_hit.t)
    var outward_normal = normalize(cross(v1 - v0, v2 - v0))
    var front_face = dot(ray.d, outward_normal) < 0.0
    var normal = outward_normal if front_face else -outward_normal
    return HitRecord(
        PrimitiveId(PRIM.TRIANGLE, bvh_hit.prim),
        p,
        normal,
        world.scene_data().triangle_surfaces[tri_idx].copy(),
        bvh_hit.t,
        front_face,
    )


def trace_bvh_normal(
    world: CpuScene[], ray: Rayf32[Frame.WORLD]
) -> Optional[HitRecord]:
    var bvh_hit = trace_blas_set[16, 16, TRACE.CLOSEST_HIT, Frame.WORLD](
        world.triangle_bvh.value(), UInt32(0), ray
    )
    if not bvh_hit.is_hit():
        return None

    var tri_idx = Int(bvh_hit.prim)
    var p = ray_at(ray, bvh_hit.t)
    var outward_normal = bvh_hit.normal.unsafe_convert[
        new_kind=GeoKind.VECTOR
    ]()
    var front_face = dot(ray.d, outward_normal) < 0.0
    var normal = outward_normal if front_face else -outward_normal
    return HitRecord(
        PrimitiveId(PRIM.TRIANGLE, bvh_hit.prim),
        p,
        normal,
        world.scene_data().triangle_surfaces[tri_idx].copy(),
        bvh_hit.t,
        front_face,
    )


def trace_rays[
    reuse_bvh_normal: Bool
](world: CpuScene[], rays: List[Rayf32[Frame.WORLD]]) -> Tuple[Float64, Int]:
    var checksum = 0.0
    var hits = 0
    for ray in rays:
        var hit: Optional[HitRecord]
        comptime if reuse_bvh_normal:
            hit = trace_bvh_normal(world, ray)
        else:
            hit = trace_recomputed_normal(world, ray)
        if hit:
            ref record = hit.value()
            checksum += (
                Float64(record.t)
                + Float64(record.normal.x)
                + Float64(record.normal.y)
                + Float64(record.normal.z)
                + Float64(record.primitive.value)
            )
            hits += 1
    return (checksum, hits)


def time_rays[
    reuse_bvh_normal: Bool
](world: CpuScene[], rays: List[Rayf32[Frame.WORLD]]) -> TimingResult:
    var summary = trace_rays[reuse_bvh_normal](world, rays)
    var best_ns = Int.MAX
    for _ in range(TIMING_REPEATS):
        var start = perf_counter_ns()
        summary = trace_rays[reuse_bvh_normal](world, rays)
        var elapsed = Int(perf_counter_ns() - start)
        if elapsed < best_ns:
            best_ns = elapsed
    return TimingResult(best_ns, summary[0], summary[1])


def run_benchmark() raises:
    print("CPU renderer standalone-triangle normal benchmark")
    print("identical BVH, rays, hit-record construction, and checksum")
    var world = make_grid_triangle_world()
    var rays = make_bounded_grid_rays()
    var recomputed = time_rays[False](world, rays)
    var reused = time_rays[True](world, rays)

    print(
        t"  reload vertices + cross + normalize:"
        t" {round(ns_to_mrays_per_s(recomputed.ns, len(rays)), 3)} MRay/s"
    )
    print(
        t"  reuse BVH normal:"
        t" {round(ns_to_mrays_per_s(reused.ns, len(rays)), 3)} MRay/s"
    )
    print(
        t"  speedup: {round(Float64(recomputed.ns) / Float64(reused.ns), 3)}x"
    )
    print(t"  checksum delta: {reused.checksum - recomputed.checksum}")
    print(t"  hit-count delta: {reused.hits - recomputed.hits}")


def main() raises:
    run_benchmark()
