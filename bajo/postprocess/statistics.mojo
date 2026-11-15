"""Zirr et al. 2018, sections 4--5: cascaded sample-count framebuffers.

Equations 15--19 split each sample between adjacent base-eight buffers. Extra
second and adjacent cross moments allow reconstruction of uncertainty after
interactive reweighting. All storage is Float64, including weighted sums.
"""

from std.math import floor, log2, exp2, isfinite
from max.algorithm import parallelize
from .samples import DenoiseSampleBatch
from bajo.core.color import Color, luminance


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


def _record(bin: Int, layer: Int, pixel: Int, count: Int) -> Int:
    var first = bin * count * CascadeLayout.LEVEL_STRIDE
    if layer == 0:
        return first + pixel * CascadeLayout.DIFFUSE_STRIDE
    return (
        first
        + count * CascadeLayout.DIFFUSE_STRIDE
        + pixel * CascadeLayout.GLOSSY_STRIDE
    )


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
    component: UInt32,
):
    var m = pixel * CascadeLayout.META
    var w = Float64(weight)
    meta[unsafe_offset=m] += w
    var y = luminance(color)
    # Inactive camera samples have a zero contribution to this component.
    # Store their squared weights directly, avoiding subtraction of two large
    # nearly equal weight sums when reconstructing a constant HDR image.
    if component != 1 or y <= 0:
        meta[unsafe_offset=m + CascadeLayout.INACTIVE_DIFFUSE] += w * w
    if component == 0 or y <= 0:
        comptime for c in range(3):
            meta[
                unsafe_offset=m + CascadeLayout.PASSTHROUGH + c
            ] += w * Float64(color[c])
        return
    var bin = cascade_bin(y)
    var a = Float64(1)
    if Float64(y) >= CascadeLayout.BASE:
        # Eq. 16. Below the numerical floor samples belong entirely to B0.
        a = ((cascade_scale(bin) / Float64(y) - 0.125) / 0.875).clamp(0, 1)
    var layer = Int(component) - 1
    for part in range(2):
        var alpha = a if part == 0 else 1 - a
        var at = _record(bin + part, layer, pixel, count)
        var aw = alpha * w
        bins[unsafe_offset=at] += alpha * Float64(y)
        comptime for c in range(3):
            bins[unsafe_offset=at + CascadeLayout.RGB + c] += aw * Float64(
                color[c]
            )
        if layer == 1:
            continue
        if bins[unsafe_offset=at + CascadeLayout.SECOND_WEIGHT] == 0:
            comptime for c in range(3):
                bins[unsafe_offset=at + CascadeLayout.ORIGIN + c] = Float64(
                    color[c]
                )
        var q_weight = aw * aw
        var cross_weight = a * (1 - a) * w * w if part == 0 else Float64(0)
        bins[unsafe_offset=at + CascadeLayout.SECOND_WEIGHT] += q_weight
        bins[unsafe_offset=at + CascadeLayout.CROSS_WEIGHT] += cross_weight
        comptime for c in range(3):
            var dc = (
                Float64(color[c])
                - bins[unsafe_offset=at + CascadeLayout.ORIGIN + c]
            )
            bins[unsafe_offset=at + CascadeLayout.SECOND_OFFSET + c] += (
                q_weight * dc
            )
            bins[unsafe_offset=at + CascadeLayout.CROSS_OFFSET + c] += (
                cross_weight * dc
            )
        for c in range(3):
            for d in range(c, 3):
                var dc = (
                    Float64(color[c])
                    - bins[unsafe_offset=at + CascadeLayout.ORIGIN + c]
                )
                var dd = (
                    Float64(color[d])
                    - bins[unsafe_offset=at + CascadeLayout.ORIGIN + d]
                )
                var k = _covariance_index(c, d)
                bins[unsafe_offset=at + CascadeLayout.SECOND + k] += (
                    q_weight * dc * dd
                )
                bins[unsafe_offset=at + CascadeLayout.CROSS + k] += (
                    cross_weight * dc * dd
                )


def _sample_count(
    bins: ImmPointer[Float64, _],
    pixel: Int,
    count: Int,
    layer: Int,
    bin: Int,
    n_bins: Int,
) -> Float64:
    var n = Float64(0)
    for j in range(max(0, bin - 1), min(n_bins, bin + 2)):
        n += bins[
            unsafe_offset=_record(j, layer, pixel, count)
        ] / cascade_scale(j)
    return n


def reconstruct_component(
    bins: ImmPointer[Float64, _],
    meta: ImmPointer[Float64, _],
    pixel: Int,
    width: Int,
    height: Int,
    layer: Int,
    n_bins: Int,
    samples: Int,
    coefficients: Color,
    reweight: Bool,
) -> Tuple[Color, Float32]:
    var count = width * height
    var normalization = meta[unsafe_offset=pixel * CascadeLayout.META]
    if normalization <= 0:
        return Color(0), Float32(0)
    var weights = Array[Float64, CascadeLayout.MAX_BINS](fill=0)
    var result = Array[Float64, 3](fill=0)
    var expected = Float64(0)
    for bin in range(n_bins):
        var at = _record(bin, layer, pixel, count)
        var w = Float64(1)
        if reweight and bin > 0:
            var n = _sample_count(bins, pixel, count, layer, bin, n_bins)
            var neighbors = Float64(0)
            var n_pixels = Float64(0)
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    var x = pixel % width + dx
                    var y = pixel // width + dy
                    if x >= 0 and x < width and y >= 0 and y < height:
                        neighbors += _sample_count(
                            bins, y * width + x, count, layer, bin, n_bins
                        )
                        n_pixels += 1
            # Sections 4.2, 4.3 and 5.3.1: kappa=kappa_min=1.
            w = max(
                n - 1, Float64(samples) * expected / cascade_scale(bin)
            ).clamp(0, 1)
            if neighbors <= n_pixels:
                w = 0
        weights[bin] = w
        comptime for c in range(3):
            result[c] += w * bins[unsafe_offset=at + CascadeLayout.RGB + c]
        expected = max(
            Float64(0),
            (0.2126 * result[0] + 0.7152 * result[1] + 0.0722 * result[2])
            / normalization,
        )
    var color = Color(
        Float32(result[0] / normalization),
        Float32(result[1] / normalization),
        Float32(result[2] / normalization),
    )
    if layer == 1 or (
        coefficients.x == 0 and coefficients.y == 0 and coefficients.z == 0
    ):
        return color, Float32(0)
    var mean = Float64(0)
    comptime for c in range(3):
        mean += Float64(coefficients[c]) * result[c] / normalization
    var squared_residual = Float64(0)
    var shifted_residual = Float64(0)
    var lost_weight = meta[
        unsafe_offset=pixel * CascadeLayout.META
        + CascadeLayout.INACTIVE_DIFFUSE
    ]
    for bin in range(n_bins):
        var at = _record(bin, layer, pixel, count)
        var r = weights[bin]
        var next_r = weights[bin + 1] if bin + 1 < n_bins else Float64(1)
        var delta = r - 1
        var next_delta = next_r - 1
        var origin = Float64(0)
        var q_first = Float64(0)
        var cross_first = Float64(0)
        var q_second = Float64(0)
        var cross_second = Float64(0)
        for c in range(3):
            origin += (
                Float64(coefficients[c])
                * bins[unsafe_offset=at + CascadeLayout.ORIGIN + c]
            )
            q_first += (
                Float64(coefficients[c])
                * bins[unsafe_offset=at + CascadeLayout.SECOND_OFFSET + c]
            )
            cross_first += (
                Float64(coefficients[c])
                * bins[unsafe_offset=at + CascadeLayout.CROSS_OFFSET + c]
            )
            for d in range(c, 3):
                var k = _covariance_index(c, d)
                var factor = (
                    Float64(coefficients[c])
                    * Float64(coefficients[d])
                    * Float64(1 if c == d else 2)
                )
                q_second += (
                    factor * bins[unsafe_offset=at + CascadeLayout.SECOND + k]
                )
                cross_second += (
                    factor * bins[unsafe_offset=at + CascadeLayout.CROSS + k]
                )
        var displacement = origin - mean
        var qw = bins[unsafe_offset=at + CascadeLayout.SECOND_WEIGHT]
        var cw = bins[unsafe_offset=at + CascadeLayout.CROSS_WEIGHT]
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
    return color, Float32(variance)


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
            if (
                not batch.colors[i].is_finite()[0]
                or not isfinite(luminance(batch.colors[i]))
                or not isfinite(batch.weights[i])
                or batch.weights[i] < 0
                or batch.components[i] > 2
            ):
                raise Error(
                    "denoising samples must be finite with nonnegative weights"
                    " and valid components"
                )
            largest = max(largest, luminance(batch.colors[i]))
        self.reserve(cascade_size(largest))
        var bins = self.bins.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var meta = self.meta.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
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
                    batch.components[i],
                )

        parallelize(accumulate, pixels)
        self.samples += spp
