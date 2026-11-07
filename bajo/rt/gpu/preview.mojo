"""Retained progressive sums and host-ready RGBA for GPU preview consumers."""
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.math import ceildiv, fma
from std.sys.intrinsics import llvm_intrinsic
from bajo.rt.render_types import RenderTimings
from bajo.rt.wavefront_contract import WAVE_COUNTER, WAVE_STATUS
from bajo.rt.gpu.resources import GpuRtRenderTarget


@always_inline
def _preview_sum(prior: Float32, incoming: Float32, batch: Float32) -> Float32:
    return fma(incoming, batch, prior)


def gpu_rt_accumulate_preview_kernel(
    incoming: Pointer[Float32, ImmutAnyOrigin],
    summed: Pointer[Float32, MutAnyOrigin],
    rgba: Pointer[UInt8, MutAnyOrigin],
    counters: Pointer[UInt32, ImmutAnyOrigin],
    count: Int32,
    batch: Float32,
    total: Float32,
    reset: UInt32,
):
    var channel = global_idx.x
    if (
        channel >= Int(count) * 3
        or counters[unsafe_offset=WAVE_COUNTER.STATUS] != WAVE_STATUS.OK
    ):
        return
    var prior = Float32(0) if reset != 0 else summed[unsafe_offset=channel]
    var value = _preview_sum(prior, incoming[unsafe_offset=channel], batch)
    summed[unsafe_offset=channel] = value
    var linear = value / total
    # GPU std.math.sqrt is approximate. The LLVM intrinsic retains the CPU
    # conversion at byte boundaries while keeping gamma work on the device.
    var gamma = llvm_intrinsic["llvm.sqrt.f32", Float32](
        linear.le(0).select(Float32(0), linear)
    )
    rgba[unsafe_offset=4 * (channel // 3) + channel % 3] = (
        Float32(256) * gamma.clamp(0, 0.999)
    ).cast[.uint8]()
    if channel % 3 == 0:
        rgba[unsafe_offset=4 * (channel // 3) + 3] = 255


@fieldwise_init
struct GpuRtPreviewResult:
    var rgba: List[UInt8]
    var timings: RenderTimings
    var samples: Int
    var batches: Int


struct GpuRtPreviewAccumulator:
    var pixel_count: Int
    var sums: DeviceBuffer[.float32]
    var rgba: DeviceBuffer[.uint8]
    var host_rgba: HostBuffer[.uint8]
    var host_status: HostBuffer[.uint32]
    var generation: UInt64
    var samples: Int
    var batches: Int

    def __init__(out self, ctx: DeviceContext, pixel_count: Int) raises:
        self.pixel_count = pixel_count
        self.sums = ctx.enqueue_create_buffer[.float32](pixel_count * 3)
        self.rgba = ctx.enqueue_create_buffer[.uint8](pixel_count * 4)
        self.host_rgba = ctx.enqueue_create_host_buffer[.uint8](pixel_count * 4)
        self.host_status = ctx.enqueue_create_host_buffer[.uint32](
            WAVE_COUNTER.COUNT
        )
        self.generation = 0
        self.samples = 0
        self.batches = 0

    def add(
        mut self,
        ctx: DeviceContext,
        target: GpuRtRenderTarget,
        generation: UInt64,
        samples: Int,
    ) raises -> List[UInt8]:
        if target.pixel_count != self.pixel_count or samples <= 0:
            raise Error(
                "GPU preview requires matching pixel count and positive samples"
            )
        var reset = self.samples == 0 or self.generation != generation
        var total = samples if reset else self.samples + samples
        ctx.enqueue_function[gpu_rt_accumulate_preview_kernel](
            target.pixels,
            self.sums,
            self.rgba,
            target.arena.counters,
            Int32(self.pixel_count),
            Float32(samples),
            Float32(total),
            UInt32(reset),
            grid_dim=ceildiv(self.pixel_count * 3, 128),
            block_dim=128,
        )
        self.host_rgba.enqueue_copy_from(self.rgba)
        self.host_status.enqueue_copy_from(target.arena.counters)
        ctx.synchronize()
        if self.host_status.as_span()[WAVE_COUNTER.STATUS] != WAVE_STATUS.OK:
            raise Error("GPU RT wavefront queue overflow")
        var pixels = List[UInt8](unsafe_uninit_length=self.pixel_count * 4)
        Span(pixels).copy_from(self.host_rgba.as_span())
        self.generation = generation
        self.samples = total
        self.batches = 1 if reset else self.batches + 1
        return pixels^
