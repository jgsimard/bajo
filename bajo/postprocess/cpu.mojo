"""Reference-based diffuse SVGF spatial reconstruction and firefly removal."""

from layout import TileTensor, Coord, Idx, row_major
from max.algorithm import parallelize
from std.math import iota, ceildiv
from std.sys import simd_width_of
from bajo.core.color import Color
from .types import AtrousSettings, DenoiseGuides, GuideLayout
from .common import _reconstruct_pixel, _filter_pixel, _albedo, _color
from .statistics import DenoiseStatistics


struct DenoiseFrame:
    var samples: Int
    var diffuse: List[Color]
    var glossy: List[Color]
    var passthrough: List[Color]
    var variance: List[Float32]

    def __init__(out self, count: Int, samples: Int):
        self.samples = samples
        self.diffuse = List[Color](length=count, fill=Color(0))
        self.glossy = List[Color](length=count, fill=Color(0))
        self.passthrough = List[Color](length=count, fill=Color(0))
        self.variance = List[Float32](length=count, fill=0)


def reconstruct_frame[
    packet_width: SIMDLength = simd_width_of[Float64]()
](
    stats: DenoiseStatistics, guides: DenoiseGuides, firefly: Bool = True
) raises -> DenoiseFrame:
    guides.validate(stats.width, stats.height)
    if stats.samples <= 0:
        raise Error("denoising requires complete camera-sample statistics")
    var count = stats.width * stats.height
    var frame = DenoiseFrame(count, stats.samples)
    var bins = stats.bins.unsafe_ptr()
    var meta = stats.meta.unsafe_ptr()
    var fields = TileTensor(
        guides.fields,
        layout=row_major(Coord(count, Idx[GuideLayout.STRIDE])),
    )
    var diffuse = frame.diffuse.unsafe_ptr()
    var glossy = frame.glossy.unsafe_ptr()
    var passthrough = frame.passthrough.unsafe_ptr()
    var variance = frame.variance.unsafe_ptr()

    def reconstruct(block: Int) {imm}:
        var indices = iota[.int, packet_width]() + block * Int(packet_width)
        var valid = indices.lt(count)
        var i = valid.select(indices, block * Int(packet_width))
        var albedo = _albedo(fields, i)
        var d, g, p, v = _reconstruct_pixel(
            bins,
            meta,
            i,
            stats.width,
            stats.height,
            stats.n_bins,
            stats.samples,
            albedo,
            firefly,
        )
        for lane in range(Int(packet_width)):
            if valid[lane]:
                var at = indices[lane]
                diffuse[unsafe_offset=at] = Color(
                    d.x[lane], d.y[lane], d.z[lane]
                )
                glossy[unsafe_offset=at] = Color(
                    g.x[lane], g.y[lane], g.z[lane]
                )
                variance[unsafe_offset=at] = v[lane]
                passthrough[unsafe_offset=at] = Color(
                    p.x[lane], p.y[lane], p.z[lane]
                )

    parallelize(reconstruct, ceildiv(count, Int(packet_width)))
    return frame^


struct AtrousWorkspace(Defaultable):
    var ping: List[Color]
    var pong: List[Color]
    var variance: List[Float32]
    var variance_pong: List[Float32]


def atrous_denoise[
    packet_width: SIMDLength = simd_width_of[Float32]()
](
    frame: DenoiseFrame,
    guides: DenoiseGuides,
    mut workspace: AtrousWorkspace,
    settings: AtrousSettings = {},
) raises:
    """Filter diffuse only; glossy and passthrough never read neighbor colors.
    """
    settings.validate()
    guides.validate(guides.width, guides.height)
    var count = guides.width * guides.height
    if (
        frame.samples <= 0
        or len(frame.diffuse) != count
        or len(frame.glossy) != count
        or len(frame.passthrough) != count
        or len(frame.variance) != count
    ):
        raise Error(
            "denoising requires matching component images and sample variance"
        )
    workspace.ping.resize(unsafe_uninit_length=count)
    workspace.pong.resize(unsafe_uninit_length=count)
    workspace.variance.resize(unsafe_uninit_length=count)
    workspace.variance_pong.resize(unsafe_uninit_length=count)
    var fields = TileTensor(
        guides.fields,
        layout=row_major(Coord(count, Idx[GuideLayout.STRIDE])),
    )
    var surfaces = guides.surfaces.unsafe_ptr()
    var output = workspace.ping.unsafe_ptr()
    var var_out = workspace.variance.unsafe_ptr()

    var diffuse = TileTensor(
        ptr=frame.diffuse.unsafe_ptr().unsafe_bitcast[Float32](),
        layout=row_major(Coord(count, Idx[3])),
    )
    var glossy = TileTensor(
        ptr=frame.glossy.unsafe_ptr().unsafe_bitcast[Float32](),
        layout=row_major(Coord(count, Idx[3])),
    )
    var passthrough = TileTensor(
        ptr=frame.passthrough.unsafe_ptr().unsafe_bitcast[Float32](),
        layout=row_major(Coord(count, Idx[3])),
    )
    var sample_variance = frame.variance.unsafe_ptr()
    var packets = ceildiv(count, Int(packet_width))

    def initialize(block: Int) {imm}:
        var indices = iota[.int, packet_width]() + block * Int(packet_width)
        var valid = indices.lt(count)
        var i = valid.select(indices, block * Int(packet_width))
        var color = _color(diffuse, i) / _albedo(fields, i)
        var v = sample_variance.unsafe_gather(i)
        for lane in range(Int(packet_width)):
            if valid[lane]:
                output[unsafe_offset=indices[lane]] = Color(
                    color.x[lane], color.y[lane], color.z[lane]
                )
                var_out[unsafe_offset=indices[lane]] = v[lane]

    parallelize(initialize, packets)
    for iteration in range(-1 if frame.samples < 4 else 0, settings.passes):
        if settings.passes == 0:
            break
        var source = TileTensor(
            ptr=workspace.ping.unsafe_ptr().unsafe_bitcast[Float32](),
            layout=row_major(Coord(count, Idx[3])),
        )
        var variance = workspace.variance.unsafe_ptr()
        var target = workspace.pong.unsafe_ptr()
        var next_variance = workspace.variance_pong.unsafe_ptr()

        def filter_pixel(block: Int) {imm}:
            var indices = iota[.int, packet_width]() + block * Int(packet_width)
            var valid = indices.lt(count)
            var i = valid.select(indices, block * Int(packet_width))
            var color, v = _filter_pixel(
                source,
                variance,
                fields,
                surfaces,
                guides.width,
                guides.height,
                i,
                iteration,
                frame.samples,
                settings.phi_color,
                settings.phi_normal,
            )
            for lane in range(Int(packet_width)):
                if valid[lane]:
                    target[unsafe_offset=indices[lane]] = Color(
                        color.x[lane], color.y[lane], color.z[lane]
                    )
                    next_variance[unsafe_offset=indices[lane]] = v[lane]

        parallelize(filter_pixel, packets)
        swap(workspace.ping, workspace.pong)
        swap(workspace.variance, workspace.variance_pong)
    var final_pixels = workspace.ping.unsafe_ptr()

    var final_words = TileTensor(
        ptr=final_pixels.unsafe_bitcast[Float32](),
        layout=row_major(Coord(count, Idx[3])),
    )

    # Discarded tail lanes read within their own packet, because composition
    # updates the final buffer in place while other packets run in parallel.
    def compose(block: Int) {imm}:
        var indices = iota[.int, packet_width]() + block * Int(packet_width)
        var valid = indices.lt(count)
        var i = valid.select(indices, block * Int(packet_width))
        var d = _color(diffuse, i)
        if settings.passes != 0:
            d = _color(final_words, i) * _albedo(fields, i)
        var color = d + _color(glossy, i) + _color(passthrough, i)
        for lane in range(Int(packet_width)):
            if valid[lane]:
                final_pixels[unsafe_offset=indices[lane]] = Color(
                    color.x[lane], color.y[lane], color.z[lane]
                )

    parallelize(compose, packets)


def denoise(
    stats: DenoiseStatistics,
    guides: DenoiseGuides,
    mut workspace: AtrousWorkspace,
    settings: AtrousSettings = {},
) raises:
    """Reconstruct firefly-weighted components and filter diffuse illumination.
    """
    settings.validate()
    var frame = reconstruct_frame(stats, guides, settings.firefly)
    atrous_denoise(frame, guides, workspace, settings)
