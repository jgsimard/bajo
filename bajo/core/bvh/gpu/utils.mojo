@fieldwise_init
struct GpuLBVHBuildTimings(TrivialRegisterPassable):
    var morton_ns: Int
    var sort_ns: Int
    var topology_ns: Int
    var refit_ns: Int
    var total_ns: Int

    def __init__(out self):
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
struct GpuLBVHValidation(TrivialRegisterPassable):
    var sorted_ok: Bool
    var values_ok: Bool
    var topology_ok: Bool
    var topology_root_count: UInt32
    var topology_root_idx: UInt32
    var bounds_ok: Bool
    var bounds_diff: Float64
    var refit_root_idx: UInt32
    var guard: UInt64


@fieldwise_init
struct GpuBuildResult(Copyable):
    var static_setup_ns: Int
    var timings: GpuLBVHBuildTimings
    var validation: GpuLBVHValidation


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
