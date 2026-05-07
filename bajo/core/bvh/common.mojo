from std.time import perf_counter_ns

from bajo.core.bvh.cpu.binary_bvh import BinaryBvh
from bajo.core.vec import Vec3f32
from bajo.core.bvh.types import Ray
from bajo.core.bvh.host_utils import (
    trace_bvh_primary,
    trace_bvh_shadow,
)
from bajo.core.utils import ns_to_ms, print_vec3_rounded, ns_to_mrays_per_s
from bajo.core.bvh.gpu.utils import GpuBuildResult


@fieldwise_init
struct CpuReferenceResult(Copyable):
    var build_ns: Int
    var trace_ns: Int
    var checksum: Float64
    var occluded: Int


def _build_cpu_reference(
    mut tri_vertices: List[Vec3f32],
    rays: List[Ray],
) raises -> CpuReferenceResult:
    var ref_build_t0 = perf_counter_ns()
    var ref_bvh = BinaryBvh(
        tri_vertices.unsafe_ptr(), UInt32(len(tri_vertices) // 3)
    )
    ref_bvh.build["sah", True]()
    var ref_build_t1 = perf_counter_ns()

    var ref_trace_t0 = perf_counter_ns()
    var ref_checksum = trace_bvh_primary(ref_bvh, rays)
    var ref_occluded = trace_bvh_shadow(ref_bvh, rays)
    var ref_trace_t1 = perf_counter_ns()

    return CpuReferenceResult(
        Int(ref_build_t1 - ref_build_t0),
        Int(ref_trace_t1 - ref_trace_t0),
        ref_checksum,
        ref_occluded,
    )


def _print_scene_summary(
    tri_vertices: List[Vec3f32],
    bmin: Vec3f32,
    bmax: Vec3f32,
    cmin: Vec3f32,
    cmax: Vec3f32,
    load_ns: Int,
):
    var tri_count = len(tri_vertices) // 3
    print(t"Packed vertices: {len(tri_vertices)}")
    print(t"Triangles: {tri_count}")
    print(t"Internal nodes: {tri_count - 1}")
    print(t"Load+pack ms: {_ms(load_ns)}")
    print_vec3_rounded("Bounds min:", bmin)
    print_vec3_rounded("Bounds max:", bmax)
    print_vec3_rounded("Centroid min:", cmin)
    print_vec3_rounded("Centroid max:", cmax)


def _print_cpu_reference(reference: CpuReferenceResult):
    print("\nCPU reference")
    print("-------------")
    print(t"SAH MT build:       {_ms(reference.build_ns)} ms")
    print(
        t"reference queries:  {_ms(reference.trace_ns)} ms | checksum:"
        t" {round(reference.checksum, 3)} | occluded: {reference.occluded}"
    )


def _print_build_result(build: GpuBuildResult):
    var build_ns = build.timings.total_ns
    var staged_ns = build.timings.sum()
    print("\nGPU LBVH build/refit")
    print("-------------------")
    print(t"static setup once:  { _ms(build.timings.static_setup_ns) } ms")
    print(
        t"valid:              sorted={build.validation.sorted_ok} |"
        t" values={build.validation.values_ok} |"
        t" topology={build.validation.topology_ok} |"
        t" bounds={build.validation.bounds_ok} |"
        t" root={build.validation.root_idx}"
    )
    print(t"morton generation:  {_ms(build.timings.morton_ns)} ms")
    print(t"radix sort pairs:   {_ms(build.timings.sort_ns)} ms")
    print(t"topology build:     {_ms(build.timings.topology_ns)} ms")
    print(t"bounds refit:       {_ms(build.timings.refit_ns)} ms")
    print(t"build total:        {_ms(build_ns)} ms")
    print(t"stage-sum total:    {_ms(staged_ns)} ms")
    print(
        t"validation detail: "
        t" topology_roots={build.validation.topology_root_count} |"
        t" topology_root={build.validation.topology_root_idx} |"
        t" refit_root={build.validation.root_idx} |"
        t" bounds_diff={round(build.validation.bounds_diff, 6)} |"
        t" guard={build.validation.guard}"
    )


@always_inline
def _ms(ns: Int) -> Float64:
    return round(ns_to_ms(ns), 3)


@always_inline
def _mrays(ns: Int, ray_count: Int) -> Float64:
    return round(ns_to_mrays_per_s(ns, ray_count), 3)
