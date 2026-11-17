"""Reference-based diffuse SVGF spatial reconstruction and firefly removal."""
from max.algorithm import parallelize
from bajo.core.color import Color
from .types import AtrousSettings, DenoiseGuides
from .common import _reconstruct_pixel, _filter_pixel, _albedo
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


def reconstruct_frame(
    stats: DenoiseStatistics, guides: DenoiseGuides, firefly: Bool = True
) raises -> DenoiseFrame:
    guides.validate(stats.width, stats.height)
    if stats.samples <= 0:
        raise Error("denoising requires complete camera-sample statistics")
    var count = stats.width * stats.height
    var frame = DenoiseFrame(count, stats.samples)
    var bins = stats.bins.unsafe_ptr().unsafe_origin_cast[ImmutAnyOrigin]()
    var meta = stats.meta.unsafe_ptr().unsafe_origin_cast[ImmutAnyOrigin]()
    var fields = guides.fields.unsafe_ptr().unsafe_origin_cast[ImmutAnyOrigin]()
    var diffuse = frame.diffuse.unsafe_ptr()
    var glossy = frame.glossy.unsafe_ptr()
    var passthrough = frame.passthrough.unsafe_ptr()
    var variance = frame.variance.unsafe_ptr()

    def reconstruct(i: Int) {imm}:
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
        diffuse[unsafe_offset=i] = d
        glossy[unsafe_offset=i] = g
        variance[unsafe_offset=i] = v
        passthrough[unsafe_offset=i] = p

    parallelize(reconstruct, count)
    return frame^


struct AtrousWorkspace(Defaultable):
    var ping: List[Color]
    var pong: List[Color]
    var variance: List[Float32]
    var variance_pong: List[Float32]


def atrous_denoise(
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
    var fields = guides.fields.unsafe_ptr().unsafe_origin_cast[ImmutAnyOrigin]()
    var surfaces = guides.surfaces.unsafe_ptr().unsafe_origin_cast[
        ImmutAnyOrigin
    ]()
    var output = workspace.ping.unsafe_ptr()
    var var_out = workspace.variance.unsafe_ptr()

    def initialize(i: Int) {imm}:
        var albedo = _albedo(fields, i)
        output[unsafe_offset=i] = frame.diffuse[i] / albedo
        var_out[unsafe_offset=i] = frame.variance[i]

    parallelize(initialize, count)
    for iteration in range(-1 if frame.samples < 4 else 0, settings.passes):
        if settings.passes == 0:
            break
        var source = (
            workspace.ping.unsafe_ptr()
            .unsafe_bitcast[Float32]()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin]()
        )
        var variance = (
            workspace.variance.unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin]()
        )
        var target = workspace.pong.unsafe_ptr()
        var next_variance = workspace.variance_pong.unsafe_ptr()

        def filter_pixel(i: Int) {imm}:
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
            target[unsafe_offset=i] = color
            next_variance[unsafe_offset=i] = v

        parallelize(filter_pixel, count)
        swap(workspace.ping, workspace.pong)
        swap(workspace.variance, workspace.variance_pong)
    var final_pixels = workspace.ping.unsafe_ptr()

    def compose(i: Int) {imm}:
        var albedo = _albedo(fields, i)
        var d = (
            frame.diffuse[i] if settings.passes
            == 0 else final_pixels[unsafe_offset=i] * albedo
        )
        final_pixels[unsafe_offset=i] = (
            d + frame.glossy[i] + frame.passthrough[i]
        )

    parallelize(compose, count)


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
