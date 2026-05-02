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
