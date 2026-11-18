"""Zirr et al. 2018, sections 4--5: cascaded sample-count framebuffers.

Equations 15--19 split each sample between adjacent base-eight buffers. Extra
second and adjacent cross moments allow reconstruction of uncertainty after
interactive reweighting. All storage is Float64, including weighted sums.
"""

from layout import TileTensor, MixedLayout, Coord, Idx, row_major
from bajo.core.tensor_gather import gather_record_field
from std.math import floor, log2, exp2, isfinite
from max.algorithm import parallelize
from .samples import DenoiseSampleBatch, DenoiseComponent
from bajo.core.color import Color, luminance, LUMINANCE_WEIGHTS
from bajo.core import Vec3


struct CascadeLayout:
    comptime BASE = Float64(1e-6)
    # Covers the entire finite Float32 positive luminance range above BASE.
    comptime MAX_BINS = 52
    comptime DIFFUSE_STRIDE = 27
    # Glossy needs only the sample-count luminance and RGB sums, not variance.
    comptime GLOSSY_STRIDE = 4
    comptime LEVEL_STRIDE = Self.DIFFUSE_STRIDE + Self.GLOSSY_STRIDE
    comptime RGB = 1
    comptime ORIGIN = 4
    comptime SECOND_WEIGHT = 7
    comptime SECOND_OFFSET = 8
    comptime SECOND = 11
    comptime CROSS_WEIGHT = 17
    comptime CROSS_OFFSET = 18
    comptime CROSS = 21
    # Per pixel: sum(w), passthrough RGB, diffuse's inactive squared weights.
    comptime PASSTHROUGH = 1
    comptime INACTIVE_DIFFUSE = 4
    comptime META = 5


def cascade_scale(bin: Int) -> Float64:
    return CascadeLayout.BASE * exp2(Float64(3 * bin))


def cascade_bin(value: Float32) -> Int:
    return max(
        0,
        Int(
            floor(
                log2(
                    max(Float64(value), CascadeLayout.BASE) / CascadeLayout.BASE
                )
                / 3
            )
        ),
    )


def cascade_size(value: Float32) -> Int:
    return min(CascadeLayout.MAX_BINS, cascade_bin(value) + 2)


def _covariance_index(i: Int, j: Int) -> Int:
    if i == j:
        return i
    if i == 0:
        return 2 + j
    return 5


def accumulate_sample(
    bins: MutPointer[Float64, _],
    meta: MutPointer[Float64, _],
    pixel: Int,
    count: Int,
    color: Color,
    weight: Float32,
    component: DenoiseComponent,
):
    var metadata = TileTensor(
        ptr=meta, layout=row_major(Coord(count, Idx[CascadeLayout.META]))
    )
    var w = Float64(weight)
    metadata[pixel, 0] += w
    var y = luminance(color)
    # Inactive camera samples have a zero contribution to this component.
    # Store their squared weights directly, avoiding subtraction of two large
    # nearly equal weight sums when reconstructing a constant HDR image.
    if component != DenoiseComponent.DIFFUSE or y <= 0:
        metadata[pixel, CascadeLayout.INACTIVE_DIFFUSE] += w * w
    if component == DenoiseComponent.PASSTHROUGH or y <= 0:
        comptime for c in range(3):
            metadata[pixel, CascadeLayout.PASSTHROUGH + c] += w * Float64(
                color[c]
            )
        return
    var bin = cascade_bin(y)
    var a = Float64(1)
    if Float64(y) >= CascadeLayout.BASE:
        # Eq. 16. Below the numerical floor samples belong entirely to B0.
        a = ((cascade_scale(bin) / Float64(y) - 0.125) / 0.875).clamp(0, 1)
    var layer = Int(component.value()) - 1
    var stride = (
        CascadeLayout.DIFFUSE_STRIDE if layer
        == 0 else CascadeLayout.GLOSSY_STRIDE
    )
    # Each level stores all diffuse records, followed by all glossy records.
    # The required adjacent levels have already been reserved by the caller.
    var layers = TileTensor(
        ptr=bins.unsafe_offset(
            0 if layer == 0 else count * CascadeLayout.DIFFUSE_STRIDE
        ),
        layout=MixedLayout(
            Coord(bin + 2, count, stride),
            Coord(count * CascadeLayout.LEVEL_STRIDE, stride, Idx[1]),
        ),
    )
    for part in range(2):
        var alpha = a if part == 0 else 1 - a
        var records = layers[bin + part, :, :]
        var aw = alpha * w
        records[pixel, 0] += alpha * Float64(y)
        comptime for c in range(3):
            records[pixel, CascadeLayout.RGB + c] += aw * Float64(color[c])
        if layer == 1:
            continue
        if records[pixel, CascadeLayout.SECOND_WEIGHT] == 0:
            comptime for c in range(3):
                records[pixel, CascadeLayout.ORIGIN + c] = Float64(color[c])
        var q_weight = aw * aw
        var cross_weight = a * (1 - a) * w * w if part == 0 else Float64(0)
        records[pixel, CascadeLayout.SECOND_WEIGHT] += q_weight
        records[pixel, CascadeLayout.CROSS_WEIGHT] += cross_weight
        comptime for c in range(3):
            var dc = (
                Float64(color[c]) - records[pixel, CascadeLayout.ORIGIN + c]
            )
            records[pixel, CascadeLayout.SECOND_OFFSET + c] += q_weight * dc
            records[pixel, CascadeLayout.CROSS_OFFSET + c] += cross_weight * dc
        for c in range(3):
            for d in range(c, 3):
                var dc = (
                    Float64(color[c]) - records[pixel, CascadeLayout.ORIGIN + c]
                )
                var dd = (
                    Float64(color[d]) - records[pixel, CascadeLayout.ORIGIN + d]
                )
                var k = _covariance_index(c, d)
                records[pixel, CascadeLayout.SECOND + k] += q_weight * dc * dd
                records[pixel, CascadeLayout.CROSS + k] += (
                    cross_weight * dc * dd
                )


def _sample_count[
    length: SIMDLength = 1
](
    bins: TileTensor[.float64, _, _],
    pixel: SIMD[.int, length],
    bin: Int,
) -> SIMD[.float64, length]:
    comptime assert bins.flat_rank == 3
    var n = SIMD[.float64, length](0)
    for j in range(max(0, bin - 1), min(Int(bins.dim[0]()), bin + 2)):
        n += gather_record_field[0](bins[j, :, :], pixel) / cascade_scale(j)
    return n


def reconstruct_component[
    length: SIMDLength = 1
](
    bins: ImmPointer[Float64, _],
    meta: ImmPointer[Float64, _],
    pixel: SIMD[.int, length],
    width: Int,
    height: Int,
    layer: Int,
    n_bins: Int,
    samples: Int,
    coefficients: Vec3[.float32, .WORLD, length],
    reweight: Bool,
) -> Tuple[Vec3[.float32, .WORLD, length], SIMD[.float32, length]]:
    var count = width * height
    var metadata = TileTensor(
        ptr=meta, layout=row_major(Coord(count, Idx[CascadeLayout.META]))
    )
    var stride = (
        CascadeLayout.DIFFUSE_STRIDE if layer
        == 0 else CascadeLayout.GLOSSY_STRIDE
    )
    var layers = TileTensor(
        ptr=bins.unsafe_offset(
            0 if layer == 0 else count * CascadeLayout.DIFFUSE_STRIDE
        ),
        layout=MixedLayout(
            Coord(n_bins, count, stride),
            Coord(count * CascadeLayout.LEVEL_STRIDE, stride, Idx[1]),
        ),
    )
    var normalization = gather_record_field[0](metadata, pixel)
    var valid = ~normalization.le(0)
    if not valid.reduce_or():
        return Vec3[.float32, .WORLD, length](0), SIMD[.float32, length](0)
    normalization = valid.select(normalization, 1)
    var weights = Array[SIMD[.float64, length], CascadeLayout.MAX_BINS](fill=0)
    var result = Array[SIMD[.float64, length], 3](fill=0)
    comptime luma = LUMINANCE_WEIGHTS[.float64]
    var expected = SIMD[.float64, length](0)
    for bin in range(n_bins):
        var records = layers[bin, :, :]
        var w = SIMD[.float64, length](1)
        if reweight and bin > 0:
            var n = _sample_count(layers, pixel, bin)
            var neighbors = SIMD[.float64, length](0)
            var n_pixels = SIMD[.float64, length](0)
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    var x = pixel % width + dx
                    var y = pixel // width + dy
                    var inside = x.ge(0) & x.lt(width) & y.ge(0) & y.lt(height)
                    var q = inside.select(y * width + x, 0)
                    neighbors += inside.select(_sample_count(layers, q, bin), 0)
                    n_pixels += inside.cast[.float64]()
            # Sections 4.2, 4.3 and 5.3.1: kappa=kappa_min=1.
            w = max(
                n - 1, Float64(samples) * expected / cascade_scale(bin)
            ).clamp(0, 1)
            w = neighbors.le(n_pixels).select(SIMD[.float64, length](0), w)
        weights[bin] = w
        comptime for c in range(3):
            result[c] += w * gather_record_field(
                records, pixel, CascadeLayout.RGB + c
            )
        expected = max(
            SIMD[.float64, length](0),
            (luma.x * result[0] + luma.y * result[1] + luma.z * result[2])
            / normalization,
        )
    var color = Vec3[.float32, .WORLD, length](
        (result[0] / normalization).cast[.float32](),
        (result[1] / normalization).cast[.float32](),
        (result[2] / normalization).cast[.float32](),
    )
    color = Vec3.select(valid, color, Vec3[.float32, .WORLD, length](0))
    if (
        layer == 1
        or (
            coefficients.x.eq(0) & coefficients.y.eq(0) & coefficients.z.eq(0)
        ).reduce_and()
    ):
        return color, SIMD[.float32, length](0)
    var mean = SIMD[.float64, length](0)
    comptime for c in range(3):
        mean += coefficients[c].cast[.float64]() * result[c] / normalization
    var squared_residual = SIMD[.float64, length](0)
    var shifted_residual = SIMD[.float64, length](0)
    var lost_weight = gather_record_field[CascadeLayout.INACTIVE_DIFFUSE](
        metadata, pixel
    )
    for bin in range(n_bins):
        var records = layers[bin, :, :]
        var r = weights[bin]
        var next_r = weights[bin + 1] if bin + 1 < n_bins else SIMD[
            .float64, length
        ](1)
        var delta = r - 1
        var next_delta = next_r - 1
        var origin = SIMD[.float64, length](0)
        var q_first = SIMD[.float64, length](0)
        var cross_first = SIMD[.float64, length](0)
        var q_second = SIMD[.float64, length](0)
        var cross_second = SIMD[.float64, length](0)
        for c in range(3):
            origin += coefficients[c].cast[.float64]() * gather_record_field(
                records, pixel, CascadeLayout.ORIGIN + c
            )
            q_first += coefficients[c].cast[.float64]() * gather_record_field(
                records, pixel, CascadeLayout.SECOND_OFFSET + c
            )
            cross_first += coefficients[c].cast[
                .float64
            ]() * gather_record_field(
                records, pixel, CascadeLayout.CROSS_OFFSET + c
            )
            for d in range(c, 3):
                var k = _covariance_index(c, d)
                var factor = (
                    coefficients[c].cast[.float64]()
                    * coefficients[d].cast[.float64]()
                    * Float64(1 if c == d else 2)
                )
                q_second += factor * gather_record_field(
                    records, pixel, CascadeLayout.SECOND + k
                )
                cross_second += factor * gather_record_field(
                    records, pixel, CascadeLayout.CROSS + k
                )
        var displacement = origin - mean
        var qw = gather_record_field(
            records, pixel, CascadeLayout.SECOND_WEIGHT
        )
        var cw = gather_record_field(records, pixel, CascadeLayout.CROSS_WEIGHT)
        q_second += (
            2 * displacement * q_first + displacement * displacement * qw
        )
        cross_second += (
            2 * displacement * cross_first + displacement * displacement * cw
        )
        q_first += displacement * qw
        cross_first += displacement * cw
        squared_residual += r * r * q_second + 2 * r * next_r * cross_second
        shifted_residual += (
            r * delta * q_first
            + (r * next_delta + next_r * delta) * cross_first
        )
        lost_weight += delta * delta * qw + 2 * delta * next_delta * cw
    # Centered form of sum[w^2 * (reweighted_luminance - mean)^2].
    # It includes adjacent cascade covariance and zero/inactive samples, and
    # avoids E[x^2]-E[x]^2 cancellation on constant high-dynamic-range images.
    var residual = (
        squared_residual
        + 2 * mean * shifted_residual
        + mean * mean * lost_weight
    )
    var variance = max(residual, 0) / (normalization * normalization)
    if samples > 1:
        variance *= Float64(samples) / Float64(samples - 1)
    else:
        variance = 0
    return color, valid.select(variance.cast[.float32](), 0)


struct DenoiseStatistics:
    var width: Int
    var height: Int
    var samples: Int
    var n_bins: Int
    var bins: List[Float64]
    var meta: List[Float64]

    def __init__(out self, width: Int, height: Int) raises:
        if width <= 0 or height <= 0:
            raise Error("denoising dimensions must be positive")
        self.width = width
        self.height = height
        self.samples = 0
        self.n_bins = 0
        self.bins = {}
        self.meta = List[Float64](
            length=width * height * CascadeLayout.META, fill=0
        )

    def reserve(mut self, n_bins: Int):
        if n_bins > self.n_bins:
            self.bins.resize(
                n_bins * self.width * self.height * CascadeLayout.LEVEL_STRIDE,
                fill=0,
            )
            self.n_bins = n_bins

    def add(mut self, batch: DenoiseSampleBatch) raises:
        if (
            batch.pixel_count != self.width * self.height
            or batch.samples_per_pixel <= 0
        ):
            raise Error("denoising sample dimensions do not match")
        var count = batch.pixel_count * batch.samples_per_pixel
        if (
            len(batch.colors) != count
            or len(batch.weights) != count
            or len(batch.components) != count
        ):
            raise Error("incomplete denoising sample sidecar")
        var largest = Float32(0)
        for i in range(count):
            var y = luminance(batch.colors[i])
            if (
                not batch.colors[i].is_finite()[0]
                or not isfinite(y)
                or not isfinite(batch.weights[i])
                or batch.weights[i] < 0
                or batch.components[i] > DenoiseComponent.GLOSSY.value()
            ):
                raise Error(
                    "denoising samples must be finite with nonnegative weights"
                    " and valid components"
                )
            largest = max(largest, y)
        self.reserve(cascade_size(largest))
        var bins = self.bins.unsafe_ptr()
        var meta = self.meta.unsafe_ptr()
        var spp = batch.samples_per_pixel
        var pixels = batch.pixel_count

        def accumulate(pixel: Int) {imm}:
            for s in range(spp):
                var i = pixel * spp + s
                accumulate_sample(
                    bins,
                    meta,
                    pixel,
                    pixels,
                    batch.colors[i],
                    batch.weights[i],
                    DenoiseComponent(batch.components[i]),
                )

        parallelize(accumulate, pixels)
        self.samples += spp
