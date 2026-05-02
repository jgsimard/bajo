from std.time import perf_counter_ns

from bajo.core.utils import ns_to_ms, print_vec3_rounded


@fieldwise_init
struct GpuBuildTimings(TrivialRegisterPassable):
    var static_setup_ns: Int
    var morton_ns: Int
    var sort_ns: Int
    var topology_ns: Int
    var refit_ns: Int
    var total_ns: Int

    def __init__(out self):
        self.static_setup_ns = Int.MAX
        self.morton_ns = Int.MAX
        self.sort_ns = Int.MAX
        self.topology_ns = Int.MAX
        self.refit_ns = Int.MAX
        self.total_ns = Int.MAX

    def min(mut self, rhs: Self):
        self.morton_ns = min(self.morton_ns, rhs.morton_ns)
        self.sort_ns = min(self.sort_ns, rhs.sort_ns)
        self.topology_ns = min(self.topology_ns, rhs.topology_ns)
        self.refit_ns = min(self.refit_ns, rhs.refit_ns)
        self.total_ns = min(self.total_ns, rhs.total_ns)

    def sum(self) -> Int:
        return self.morton_ns + self.sort_ns + self.topology_ns + self.refit_ns


@fieldwise_init
struct GpuBVHValidation(TrivialRegisterPassable):
    var sorted_ok: Bool
    var values_ok: Bool
    var topology_ok: Bool
    var topology_root_count: UInt32
    var topology_root_idx: UInt32
    var bounds_ok: Bool
    var bounds_diff: Float64
    var root_idx: UInt32
    var guard: UInt64


@fieldwise_init
struct GpuBuildResult(Copyable):
    var static_setup_ns: Int
    var timings: GpuBuildTimings
    var validation: GpuBVHValidation


@fieldwise_init
struct GpuDirectTraversalResult(Copyable):
    var upload_ns: Int
    var kernel_ns: Int
    var download_ns: Int
    var frame_ns: Int
    var checksum: Float64
    var hit_count: UInt32
    var diff: Float64


@fieldwise_init
struct GpuCameraFullResult(Copyable):
    var kernel_ns: Int
    var download_ns: Int
    var frame_ns: Int
    var checksum: Float64
    var hit_count: UInt32
    var diff: Float64


@fieldwise_init
struct GpuPrimaryReduceResult(Copyable):
    var kernel_ns: Int
    var reduce_ns: Int
    var download_ns: Int
    var frame_ns: Int
    var checksum: Float64
    var hit_count: UInt32
    var diff: Float64


@fieldwise_init
struct GpuShadowReduceResult(Copyable):
    var kernel_ns: Int
    var reduce_ns: Int
    var download_ns: Int
    var frame_ns: Int
    var occluded: UInt32
    var diff: Int


@fieldwise_init
struct GpuReduceAndShadowResult(Copyable):
    var primary: GpuPrimaryReduceResult
    var shadow: GpuShadowReduceResult


@fieldwise_init
struct GpuSuiteResult(Copyable):
    var build: GpuBuildResult
    var direct: GpuDirectTraversalResult
    var camera_full: GpuCameraFullResult
    var reduce_shadow: GpuReduceAndShadowResult


@fieldwise_init
struct CpuReferenceResult(Copyable):
    var build_ns: Int
    var trace_ns: Int
    var checksum: Float64
    var occluded: Int


def _download_full_hit_checksum(
    ctx: DeviceContext,
    d_hits_f32: DeviceBuffer[DType.float32],
    ray_count: Int,
) raises -> Tuple[Float64, UInt32]:
    var checksum = 0.0
    var hit_count = UInt32(0)
    with d_hits_f32.map_to_host() as h:
        for i in range(ray_count):
            var t = h[i * 3]
            if t < 1.0e20:
                checksum += Float64(t)
                hit_count += 1
    ctx.synchronize()
    return (checksum, hit_count)


def _download_reduced_hit_t[
    GPU_REDUCE_THREADS: Int
](
    ctx: DeviceContext,
    d_partial_sums: DeviceBuffer[DType.float64],
    d_partial_counts: DeviceBuffer[DType.uint32],
) raises -> Tuple[Float64, UInt32]:
    var checksum = 0.0
    var hit_count = UInt32(0)
    with d_partial_sums.map_to_host() as sums:
        for i in range(GPU_REDUCE_THREADS):
            checksum += sums[i]
    with d_partial_counts.map_to_host() as counts:
        for i in range(GPU_REDUCE_THREADS):
            hit_count += counts[i]
    ctx.synchronize()
    return (checksum, hit_count)


def _download_reduced_u32_count[
    GPU_REDUCE_THREADS: Int
](
    ctx: DeviceContext,
    d_partial_counts: DeviceBuffer[DType.uint32],
) raises -> UInt32:
    var total = UInt32(0)
    with d_partial_counts.map_to_host() as counts:
        for i in range(GPU_REDUCE_THREADS):
            total += counts[i]
    ctx.synchronize()
    return total


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
