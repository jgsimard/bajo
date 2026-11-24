"""Resident complete-sample statistics and reference SVGF spatial stages."""

from layout import TileTensor, Coord, Idx, row_major
from max.gpu import global_idx, block_idx, block_dim, grid_dim, thread_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceBuffer, DeviceContext
from std.math import ceildiv, isfinite
from std.memory import bitcast
from bajo.gpu.buffers import upload_list
from bajo.core.color import luminance
from .types import AtrousSettings, DenoiseGuides, GuideLayout
from .samples import DenoiseComponent
from .common import _color, _albedo, _reconstruct_pixel, _filter_pixel
from .statistics import (
    CascadeLayout,
    cascade_size,
    accumulate_sample,
)


@fieldwise_init
struct GpuDenoiseGuides:
    var width: Int
    var height: Int
    var fields: DeviceBuffer[.float32]
    var surfaces: DeviceBuffer[.uint32]

    def __init__(out self, ctx: DeviceContext, width: Int, height: Int) raises:
        if width <= 0 or height <= 0:
            raise Error("denoising dimensions must be positive")
        self.width = width
        self.height = height
        self.fields = ctx.enqueue_create_buffer[.float32](
            width * height * GuideLayout.STRIDE
        )
        self.surfaces = ctx.enqueue_create_buffer[.uint32](width * height)

    @staticmethod
    def upload(mut ctx: DeviceContext, guides: DenoiseGuides) raises -> Self:
        guides.validate(guides.width, guides.height)
        return Self(
            guides.width,
            guides.height,
            upload_list(ctx, guides.fields),
            upload_list(ctx, guides.surfaces),
        )


struct GpuDenoiseSampleBatch:
    var count: Int
    var colors: DeviceBuffer[.float32]
    var weights: DeviceBuffer[.float32]
    var components: DeviceBuffer[.uint32]

    def __init__(out self, ctx: DeviceContext, count: Int) raises:
        self.count = count
        self.colors = ctx.enqueue_create_buffer[.float32](count * 3)
        self.weights = ctx.enqueue_create_buffer[.float32](count)
        self.components = ctx.enqueue_create_buffer[.uint32](count)
        self.components.enqueue_fill(DenoiseComponent.PASSTHROUGH.value())


# enqueue_function requires concrete pointer origins at kernel entry points.
def _maximum_kernel(
    colors: ImmPointer[Float32, ImmUntrackedOrigin],
    partial: MutPointer[Float32, MutUntrackedOrigin],
    weights: ImmPointer[Float32, ImmUntrackedOrigin],
    components: ImmPointer[UInt32, ImmUntrackedOrigin],
    count: Int32,
):
    var records = TileTensor(
        ptr=colors, layout=row_major(Coord(Int(count), Idx[3]))
    )
    var largest = Float32(0)
    var i = global_idx.x
    while i < Int(count):
        var color = _color(records, i)
        var y = luminance(color)
        if (
            not color.is_finite()[0]
            or not isfinite(y)
            or not isfinite(weights[unsafe_offset=i])
            or weights[unsafe_offset=i] < 0
            or components[unsafe_offset=i] > DenoiseComponent.GLOSSY.value()
        ):
            largest = bitcast[.float32](UInt32(0x7F800000))
        largest = max(largest, y)
        i += Int(grid_dim.x * block_dim.x)
    largest = block.max[block_size=128, broadcast=True](largest)
    if thread_idx.x == 0:
        partial[unsafe_offset=Int(block_idx.x)] = largest


def _maximum_finish(
    partial: ImmPointer[Float32, ImmUntrackedOrigin],
    result: MutPointer[Float32, MutUntrackedOrigin],
    count: Int32,
):
    if global_idx.x == 0:
        var largest = Float32(0)
        for i in range(Int(count)):
            largest = max(largest, partial[unsafe_offset=i])
        result[unsafe_offset=0] = largest


def _accumulate_statistics(
    colors: ImmPointer[Float32, ImmUntrackedOrigin],
    weights: ImmPointer[Float32, ImmUntrackedOrigin],
    components: ImmPointer[UInt32, ImmUntrackedOrigin],
    bins: MutPointer[Float64, MutUntrackedOrigin],
    meta: MutPointer[Float64, MutUntrackedOrigin],
    count: Int32,
    spp: Int32,
):
    var pixel = global_idx.x
    if pixel >= Int(count):
        return
    var records = TileTensor(
        ptr=colors, layout=row_major(Coord(Int(count) * Int(spp), Idx[3]))
    )
    for s in range(Int(spp)):
        var i = pixel * Int(spp) + s
        accumulate_sample(
            bins,
            meta,
            pixel,
            Int(count),
            _color(records, i),
            weights[unsafe_offset=i],
            DenoiseComponent(components[unsafe_offset=i]),
        )


struct GpuDenoiseStatistics:
    var width: Int
    var height: Int
    var samples: Int
    var n_bins: Int
    var bins: DeviceBuffer[.float64]
    var meta: DeviceBuffer[.float64]
    var partial: DeviceBuffer[.float32]
    var maximum: DeviceBuffer[.float32]

    def __init__(out self, ctx: DeviceContext, width: Int, height: Int) raises:
        if width <= 0 or height <= 0:
            raise Error("denoising dimensions must be positive")
        self.width = width
        self.height = height
        self.samples = 0
        self.n_bins = 0
        self.bins = ctx.enqueue_create_buffer[.float64](1)
        self.meta = ctx.enqueue_create_buffer[.float64](
            width * height * CascadeLayout.META
        )
        self.meta.enqueue_fill(0)
        self.partial = ctx.enqueue_create_buffer[.float32](1024)
        self.maximum = ctx.enqueue_create_buffer[.float32](1)

    def add(
        mut self, ctx: DeviceContext, batch: GpuDenoiseSampleBatch, spp: Int
    ) raises:
        var count = self.width * self.height
        if (
            spp <= 0
            or batch.count != count * spp
            or len(batch.colors) != batch.count * 3
            or len(batch.weights) != batch.count
            or len(batch.components) != batch.count
        ):
            raise Error("denoising sample dimensions do not match")
        var blocks = min(1024, ceildiv(batch.count, 128))
        ctx.enqueue_function[_maximum_kernel](
            batch.colors,
            self.partial,
            batch.weights,
            batch.components,
            Int32(batch.count),
            grid_dim=blocks,
            block_dim=128,
        )
        ctx.enqueue_function[_maximum_finish](
            self.partial, self.maximum, Int32(blocks), grid_dim=1, block_dim=1
        )
        var required: Int
        # Only one scalar crosses the device boundary, not the sample sidecar.
        with self.maximum.map_to_host() as largest:
            if not isfinite(largest[0]):
                raise Error(
                    "denoising samples must be finite with nonnegative weights"
                    " and valid components"
                )
            required = cascade_size(largest[0])
        if required > self.n_bins:
            var next = ctx.enqueue_create_buffer[.float64](
                required * count * CascadeLayout.LEVEL_STRIDE
            )
            next.enqueue_fill(0)
            if self.n_bins:
                ctx.enqueue_copy(
                    next.unsafe_ptr(), self.bins.unsafe_ptr(), len(self.bins)
                )
            self.bins = next^
            self.n_bins = required
        ctx.enqueue_function[_accumulate_statistics](
            batch.colors,
            batch.weights,
            batch.components,
            self.bins,
            self.meta,
            Int32(count),
            Int32(spp),
            grid_dim=ceildiv(count, 128),
            block_dim=128,
        )
        self.samples += spp


struct GpuAtrousWorkspace:
    var width: Int
    var height: Int
    var ping: DeviceBuffer[.float32]
    var pong: DeviceBuffer[.float32]
    var variance: DeviceBuffer[.float32]
    var variance_pong: DeviceBuffer[.float32]
    var residual: DeviceBuffer[.float32]
    var diffuse: DeviceBuffer[.float32]

    def __init__(out self, ctx: DeviceContext, width: Int, height: Int) raises:
        if width <= 0 or height <= 0:
            raise Error("denoising dimensions must be positive")
        self.width = width
        self.height = height
        var count = width * height
        self.ping = ctx.enqueue_create_buffer[.float32](count * 3)
        self.pong = ctx.enqueue_create_buffer[.float32](count * 3)
        self.variance = ctx.enqueue_create_buffer[.float32](count)
        self.variance_pong = ctx.enqueue_create_buffer[.float32](count)
        self.residual = ctx.enqueue_create_buffer[.float32](count * 3)
        self.diffuse = ctx.enqueue_create_buffer[.float32](count * 3)


def _reconstruct_kernel(
    bins: ImmPointer[Float64, ImmUntrackedOrigin],
    meta: ImmPointer[Float64, ImmUntrackedOrigin],
    fields: ImmPointer[Float32, ImmUntrackedOrigin],
    ping: MutPointer[Float32, MutUntrackedOrigin],
    diffuse: MutPointer[Float32, MutUntrackedOrigin],
    residual: MutPointer[Float32, MutUntrackedOrigin],
    variance: MutPointer[Float32, MutUntrackedOrigin],
    width: Int32,
    height: Int32,
    n_bins: Int32,
    samples: Int32,
    firefly: UInt32,
):
    var i = global_idx.x
    if i >= Int(width) * Int(height):
        return
    var guide_records = TileTensor(
        ptr=fields,
        layout=row_major(
            Coord(Int(width) * Int(height), Idx[GuideLayout.STRIDE])
        ),
    )
    var albedo = _albedo(guide_records, i)
    var d, g, p, v = _reconstruct_pixel(
        bins,
        meta,
        i,
        Int(width),
        Int(height),
        Int(n_bins),
        Int(samples),
        albedo,
        firefly != 0,
    )
    var color_layout = row_major(Coord(Int(width) * Int(height), Idx[3]))
    var ping_records = TileTensor(ptr=ping, layout=color_layout)
    var diffuse_records = TileTensor(ptr=diffuse, layout=color_layout)
    var residual_records = TileTensor(ptr=residual, layout=color_layout)
    comptime for c in range(3):
        ping_records[i, c] = d[c] / albedo[c]
        diffuse_records[i, c] = d[c]
        residual_records[i, c] = g[c] + p[c]
    variance[unsafe_offset=i] = v


def _filter_kernel(
    source: ImmPointer[Float32, ImmUntrackedOrigin],
    fields: ImmPointer[Float32, ImmUntrackedOrigin],
    surfaces: ImmPointer[UInt32, ImmUntrackedOrigin],
    variance: ImmPointer[Float32, ImmUntrackedOrigin],
    output: MutPointer[Float32, MutUntrackedOrigin],
    next_variance: MutPointer[Float32, MutUntrackedOrigin],
    width: Int32,
    height: Int32,
    iteration: Int32,
    samples: Int32,
    phi_color: Float32,
    phi_normal: Float32,
):
    var i = global_idx.x
    if i >= Int(width) * Int(height):
        return
    var color, v = _filter_pixel(
        TileTensor(
            ptr=source,
            layout=row_major(Coord(Int(width) * Int(height), Idx[3])),
        ),
        variance,
        TileTensor(
            ptr=fields,
            layout=row_major(
                Coord(Int(width) * Int(height), Idx[GuideLayout.STRIDE])
            ),
        ),
        surfaces,
        Int(width),
        Int(height),
        i,
        Int(iteration),
        Int(samples),
        phi_color,
        phi_normal,
    )
    var output_records = TileTensor(
        ptr=output, layout=row_major(Coord(Int(width) * Int(height), Idx[3]))
    )
    comptime for c in range(3):
        output_records[i, c] = color[c]
    next_variance[unsafe_offset=i] = v


def _compose_kernel(
    pixels: MutPointer[Float32, MutUntrackedOrigin],
    fields: ImmPointer[Float32, ImmUntrackedOrigin],
    diffuse: ImmPointer[Float32, ImmUntrackedOrigin],
    residual: ImmPointer[Float32, ImmUntrackedOrigin],
    count: Int32,
    passes: Int32,
):
    var i = global_idx.x
    if i >= Int(count):
        return
    var albedo = _albedo(
        TileTensor(
            ptr=fields,
            layout=row_major(Coord(Int(count), Idx[GuideLayout.STRIDE])),
        ),
        i,
    )
    comptime for c in range(3):
        var at = i * 3 + c
        var d = (
            diffuse[unsafe_offset=at] if passes
            == 0 else pixels[unsafe_offset=at] * albedo[c]
        )
        pixels[unsafe_offset=at] = d + residual[unsafe_offset=at]


def enqueue_atrous_denoise(
    ctx: DeviceContext,
    stats: GpuDenoiseStatistics,
    guides: GpuDenoiseGuides,
    mut workspace: GpuAtrousWorkspace,
    settings: AtrousSettings = {},
) raises:
    settings.validate()
    if (
        stats.samples <= 0
        or stats.width != guides.width
        or stats.height != guides.height
        or workspace.width != guides.width
        or workspace.height != guides.height
    ):
        raise Error(
            "denoising requires matching camera-sample statistics, guides and"
            " workspace"
        )
    var width = Int32(guides.width)
    var height = Int32(guides.height)
    var count = guides.width * guides.height
    var blocks = ceildiv(count, 128)
    ctx.enqueue_function[_reconstruct_kernel](
        stats.bins,
        stats.meta,
        guides.fields,
        workspace.ping,
        workspace.diffuse,
        workspace.residual,
        workspace.variance,
        width,
        height,
        Int32(stats.n_bins),
        Int32(stats.samples),
        UInt32(settings.firefly),
        grid_dim=blocks,
        block_dim=128,
    )
    for iteration in range(-1 if stats.samples < 4 else 0, settings.passes):
        if settings.passes == 0:
            break
        ctx.enqueue_function[_filter_kernel](
            workspace.ping,
            guides.fields,
            guides.surfaces,
            workspace.variance,
            workspace.pong,
            workspace.variance_pong,
            width,
            height,
            Int32(iteration),
            Int32(stats.samples),
            settings.phi_color,
            settings.phi_normal,
            grid_dim=blocks,
            block_dim=128,
        )
        swap(workspace.ping, workspace.pong)
        swap(workspace.variance, workspace.variance_pong)
    ctx.enqueue_function[_compose_kernel](
        workspace.ping,
        guides.fields,
        workspace.diffuse,
        workspace.residual,
        Int32(count),
        Int32(settings.passes),
        grid_dim=blocks,
        block_dim=128,
    )
