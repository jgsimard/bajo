"""Shared CPU RT timing collection and output checksums."""

from bajo.core import Vec3W
from bajo.rt import Camera, CpuScene, RenderResult, RenderSettings
from .timing import summarize_timings


@fieldwise_init
struct CpuRenderTiming(Copyable):
    var median_total_ns: Int
    var median_init_ns: Int
    var median_render_ns: Int
    var min_render_ns: Int
    var max_render_ns: Int
    var checksum: Float64


struct CpuRenderSamples(Copyable):
    var total_times: List[Int]
    var init_times: List[Int]
    var render_times: List[Int]
    var checksum: Float64

    def __init__(out self, checksum: Float64, repeats: Int):
        self.total_times = List[Int](capacity=repeats)
        self.init_times = List[Int](capacity=repeats)
        self.render_times = List[Int](capacity=repeats)
        self.checksum = checksum

    def record(mut self, result: RenderResult) raises:
        if pixel_checksum(result.pixels) != self.checksum:
            raise Error("CPU image changed between benchmark samples")
        self.total_times.append(result.timings.total_ns)
        self.init_times.append(result.timings.init_ns)
        self.render_times.append(result.timings.render_ns)

    def summarize(mut self) -> CpuRenderTiming:
        var total = summarize_timings(self.total_times)
        var init = summarize_timings(self.init_times)
        var render = summarize_timings(self.render_times)
        return CpuRenderTiming(
            total.median_ns,
            init.median_ns,
            render.median_ns,
            render.min_ns,
            render.max_ns,
            self.checksum,
        )


def benchmark_cpu_render[
    RenderFn: def(
        RenderSettings, Camera, CpuScene[16, 16]
    ) raises thin -> RenderResult,
](
    settings: RenderSettings,
    camera: Camera,
    world: CpuScene[16, 16],
    repeats: Int,
) raises -> CpuRenderTiming:
    if repeats < 1:
        raise Error("benchmark repeats must be positive")
    var warmup = RenderFn(settings, camera, world)
    var samples = CpuRenderSamples(pixel_checksum(warmup.pixels), repeats)
    for _ in range(repeats):
        samples.record(RenderFn(settings, camera, world))
    return samples.summarize()


def pixel_checksum(pixels: List[Vec3W]) -> Float64:
    var checksum = 0.0
    for i, p in enumerate(pixels):
        var weight = Float64((i % 251) + 1)
        checksum += weight * (
            Float64(p.x) + 2.0 * Float64(p.y) + 3.0 * Float64(p.z)
        )
    return checksum
