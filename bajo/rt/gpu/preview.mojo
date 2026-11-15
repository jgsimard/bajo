"""Retained progressive sums and host-ready RGBA for GPU preview consumers."""
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.math import ceildiv, fma
from bajo.core.color import color_to_byte
from bajo.rt.wavefront_contract import WAVE_COUNTER, WAVE_STATUS
from bajo.rt.gpu.resources import GpuRtRenderTarget
from bajo.postprocess.gpu import (
    GpuAtrousWorkspace,
    GpuDenoiseGuides,
    GpuDenoiseStatistics,
    enqueue_atrous_denoise,
)


def _preview_sum(prior: Float32, incoming: Float32, batch: Float32) -> Float32:
    return fma(incoming, batch, prior)


def gpu_rt_accumulate_preview_kernel(
    incoming: ImmPointer[Float32, ImmutAnyOrigin],
    summed: MutPointer[Float32, MutAnyOrigin],
    counters: ImmPointer[UInt32, ImmutAnyOrigin],
    count: Int32,
    batch: Float32,
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


def _encode_preview_kernel(
    source: ImmPointer[Float32, ImmutAnyOrigin],
    rgba: MutPointer[UInt8, MutAnyOrigin],
    count: Int32,
    divisor: Float32,
):
    var i = global_idx.x
    if i >= Int(count) * 3:
        return
    var linear = source[unsafe_offset=i] / divisor
    rgba[unsafe_offset=4 * (i // 3) + i % 3] = color_to_byte(linear)
    if i % 3 == 0:
        rgba[unsafe_offset=4 * (i // 3) + 3] = 255


from bajo.rt.preview_types import PreviewResult as GpuRtPreviewResult


struct GpuRtPreviewAccumulator:
    var pixel_count: Int
    var sums: DeviceBuffer[.float32]
    var rgba: DeviceBuffer[.uint8]
    var host_rgba: HostBuffer[.uint8]
    var host_status: HostBuffer[.uint32]
    var generation: UInt64
    var samples: Int
    var batches: Int
    var guides: Optional[GpuDenoiseGuides]
    var guide_generation: UInt64
    var workspace: Optional[GpuAtrousWorkspace]
    var statistics: Optional[GpuDenoiseStatistics]
    var statistics_error: String
    var pending_samples: Int
    var pending_batches: Int
    var pending_generation: UInt64

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
        self.guides = {}
        self.guide_generation = 0
        self.workspace = {}
        self.statistics = {}
        self.statistics_error = ""
        self.pending_samples = 0
        self.pending_batches = 0
        self.pending_generation = 0

    def enqueue_add(
        mut self,
        ctx: DeviceContext,
        target: GpuRtRenderTarget,
        generation: UInt64,
        samples: Int,
    ) raises:
        if self.pending_samples:
            raise Error(
                "present the pending GPU preview before accumulating another"
                " batch"
            )
        if target.pixel_count != self.pixel_count or samples <= 0:
            raise Error(
                "GPU preview requires matching pixel count and positive samples"
            )
        var reset = self.samples == 0 or self.generation != generation
        var total = samples if reset else self.samples + samples
        ctx.enqueue_function[gpu_rt_accumulate_preview_kernel](
            target.pixels,
            self.sums,
            target.arena.counters,
            Int32(self.pixel_count),
            Float32(samples),
            UInt32(reset),
            grid_dim=ceildiv(self.pixel_count * 3, 128),
            block_dim=128,
        )
        self.pending_generation = generation
        self.pending_samples = total
        self.pending_batches = 1 if reset else self.batches + 1
        if reset:
            self.statistics = {}
            self.statistics_error = ""
        try:
            # Do not commit auxiliary statistics from failed transport.
            self.host_status.enqueue_copy_from(target.arena.counters)
            ctx.synchronize()
            if (
                self.host_status.as_span()[WAVE_COUNTER.STATUS]
                != WAVE_STATUS.OK
            ):
                raise Error("GPU RT wavefront queue overflow")
            if not target.denoise_samples:
                raise Error("complete camera-sample statistics are unavailable")
            if not self.statistics:
                self.statistics = GpuDenoiseStatistics(
                    ctx, target.image_width, target.image_height
                )
            self.statistics.value().add(
                ctx, target.denoise_samples.value(), samples
            )
        except error:
            self.statistics_error = String(error)

    def present(
        mut self,
        ctx: DeviceContext,
        target: GpuRtRenderTarget,
        denoise: Bool = False,
    ) raises -> List[UInt8]:
        var total = (
            self.pending_samples if self.pending_samples else self.samples
        )
        if total <= 0:
            raise Error("GPU preview has no accumulated samples")
        var blocks = ceildiv(self.pixel_count * 3, 128)
        if denoise:
            if not self.guides:
                raise Error("GPU denoising requires surface guides")
            if (
                self.statistics_error
                or not self.statistics
                or self.statistics.value().samples != total
            ):
                raise Error(
                    "denoising requires complete sample statistics: "
                    + self.statistics_error
                )
            if not self.workspace:
                self.workspace = GpuAtrousWorkspace(
                    ctx, target.image_width, target.image_height
                )
            enqueue_atrous_denoise(
                ctx,
                self.statistics.value(),
                self.guides.value(),
                self.workspace.value(),
            )
            ctx.enqueue_function[_encode_preview_kernel](
                self.workspace.value().ping,
                self.rgba,
                Int32(self.pixel_count),
                Float32(1),
                grid_dim=blocks,
                block_dim=128,
            )
        else:
            ctx.enqueue_function[_encode_preview_kernel](
                self.sums,
                self.rgba,
                Int32(self.pixel_count),
                Float32(total),
                grid_dim=blocks,
                block_dim=128,
            )
        self.host_rgba.enqueue_copy_from(self.rgba)
        self.host_status.enqueue_copy_from(target.arena.counters)
        ctx.synchronize()
        if self.host_status.as_span()[WAVE_COUNTER.STATUS] != WAVE_STATUS.OK:
            self.pending_samples = 0
            raise Error("GPU RT wavefront queue overflow")
        var pixels = List[UInt8](unsafe_uninit_length=self.pixel_count * 4)
        Span(pixels).copy_from(self.host_rgba.as_span())
        if self.pending_samples:
            self.generation = self.pending_generation
            self.samples = self.pending_samples
            self.batches = self.pending_batches
            self.pending_samples = 0
        return pixels^

    def add(
        mut self,
        ctx: DeviceContext,
        target: GpuRtRenderTarget,
        generation: UInt64,
        samples: Int,
    ) raises -> List[UInt8]:
        self.enqueue_add(ctx, target, generation, samples)
        return self.present(ctx, target)
