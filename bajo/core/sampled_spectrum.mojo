"""Four sampled wavelengths and spectral arithmetic shared by CPU and GPU.

Sampling conventions follow PBRT v4: four cyclically stratified visible
wavelengths, per-wavelength PDFs, and explicit secondary termination.
"""

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.builtin.globals import global_constant
from std.math import atanh, cosh, floor, sqrt, exp
from bajo.core.cie_data import CIE_XYZ
from bajo.core.vec import Vec3

comptime SPECTRUM_SAMPLES = 4
comptime LAMBDA_MIN = Float32(360)
comptime LAMBDA_MAX = Float32(830)
comptime CIE_Y_INTEGRAL = Float32(106.856895)


struct SampledSpectrum[width: SIMDLength = 1](
    Copyable, DevicePassable, TrivialRegisterPassable, Writable
):
    """Four spectral components; SIMD lanes represent independent paths."""

    var s0: SIMD[.float32, Self.width]
    var s1: SIMD[.float32, Self.width]
    var s2: SIMD[.float32, Self.width]
    var s3: SIMD[.float32, Self.width]

    def __init__(out self, value: SIMD[.float32, Self.width] = 0):
        self.s0 = value
        self.s1 = value
        self.s2 = value
        self.s3 = value

    def __init__(
        out self,
        s0: SIMD[.float32, Self.width],
        s1: SIMD[.float32, Self.width],
        s2: SIMD[.float32, Self.width],
        s3: SIMD[.float32, Self.width],
    ):
        self.s0 = s0
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "SampledSpectrum"

    @always_inline
    def __getitem__(self, index: Int) -> SIMD[.float32, Self.width]:
        debug_assert["safe"](
            index >= 0 and index < 4, "spectral index out of range"
        )
        if index == 0:
            return self.s0
        if index == 1:
            return self.s1
        if index == 2:
            return self.s2
        return self.s3

    @always_inline
    def __setitem__(mut self, index: Int, value: SIMD[.float32, Self.width]):
        debug_assert["safe"](
            index >= 0 and index < 4, "spectral index out of range"
        )
        if index == 0:
            self.s0 = value
        elif index == 1:
            self.s1 = value
        elif index == 2:
            self.s2 = value
        else:
            self.s3 = value

    @always_inline
    def get(self, lane: Int) -> SampledSpectrum[1]:
        return SampledSpectrum[1](
            self.s0[lane], self.s1[lane], self.s2[lane], self.s3[lane]
        )

    @always_inline
    def set(mut self, lane: Int, value: SampledSpectrum[1]):
        self.s0[lane] = value.s0
        self.s1[lane] = value.s1
        self.s2[lane] = value.s2
        self.s3[lane] = value.s3

    @always_inline
    def __add__(self, other: Self) -> Self:
        return Self(
            self.s0 + other.s0,
            self.s1 + other.s1,
            self.s2 + other.s2,
            self.s3 + other.s3,
        )

    @always_inline
    def __sub__(self, other: Self) -> Self:
        return Self(
            self.s0 - other.s0,
            self.s1 - other.s1,
            self.s2 - other.s2,
            self.s3 - other.s3,
        )

    @always_inline
    def __neg__(self) -> Self:
        return Self(-self.s0, -self.s1, -self.s2, -self.s3)

    @always_inline
    def __mul__(self, other: Self) -> Self:
        return Self(
            self.s0 * other.s0,
            self.s1 * other.s1,
            self.s2 * other.s2,
            self.s3 * other.s3,
        )

    @always_inline
    def __mul__(self, scale: SIMD[.float32, Self.width]) -> Self:
        return Self(
            self.s0 * scale, self.s1 * scale, self.s2 * scale, self.s3 * scale
        )

    @always_inline
    def __truediv__(self, scale: SIMD[.float32, Self.width]) -> Self:
        return Self(
            self.s0 / scale, self.s1 / scale, self.s2 / scale, self.s3 / scale
        )

    @always_inline
    def __truediv__(self, other: Self) -> Self:
        return Self(
            self.s0 / other.s0,
            self.s1 / other.s1,
            self.s2 / other.s2,
            self.s3 / other.s3,
        )

    @always_inline
    def __iadd__(mut self, other: Self):
        self = self + other

    @always_inline
    def __isub__(mut self, other: Self):
        self = self - other

    @always_inline
    def __imul__(mut self, other: Self):
        self = self * other

    @always_inline
    def __imul__(mut self, scale: SIMD[.float32, Self.width]):
        self = self * scale

    @always_inline
    def __itruediv__(mut self, scale: SIMD[.float32, Self.width]):
        self = self / scale

    @always_inline
    def safe_divide(self, other: Self) -> Self:
        var result = Self()
        comptime for i in range(4):
            var valid = other[i].ne(0)
            result[i] = valid.select(self[i] / valid.select(other[i], 1), 0)
        return result

    @always_inline
    def average(self) -> SIMD[.float32, Self.width]:
        return (self.s0 + self.s1 + self.s2 + self.s3) * 0.25

    @always_inline
    def max_component(self) -> SIMD[.float32, Self.width]:
        return max(max(self.s0, self.s1), max(self.s2, self.s3))

    @always_inline
    def clamp(self, low: Float32, high: Float32) -> Self:
        return Self(
            self.s0.clamp(low, high),
            self.s1.clamp(low, high),
            self.s2.clamp(low, high),
            self.s3.clamp(low, high),
        )

    @always_inline
    def sqrt(self) -> Self:
        return Self(sqrt(self.s0), sqrt(self.s1), sqrt(self.s2), sqrt(self.s3))

    @always_inline
    def exp(self) -> Self:
        return Self(exp(self.s0), exp(self.s1), exp(self.s2), exp(self.s3))

    @staticmethod
    @always_inline
    def select(mask: SIMD[.bool, Self.width], yes: Self, no: Self) -> Self:
        return Self(
            mask.select(yes.s0, no.s0),
            mask.select(yes.s1, no.s1),
            mask.select(yes.s2, no.s2),
            mask.select(yes.s3, no.s3),
        )


@always_inline
def visible_wavelength_pdf[
    width: SIMDLength
](wavelength: SIMD[.float32, width]) -> SIMD[.float32, width]:
    var c = cosh(Float32(0.0072) * (wavelength - 538))
    return (wavelength.ge(LAMBDA_MIN) & wavelength.le(LAMBDA_MAX)).select(
        Float32(0.0039398042) / (c * c), 0
    )


struct SampledWavelengths[width: SIMDLength = 1](
    Copyable, DevicePassable, TrivialRegisterPassable, Writable
):
    var lambda_nm: SampledSpectrum[Self.width]
    var pdf: SampledSpectrum[Self.width]

    def __init__(
        out self,
        lambda_nm: SampledSpectrum[Self.width],
        pdf: SampledSpectrum[Self.width],
    ):
        self.lambda_nm = lambda_nm
        self.pdf = pdf

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "SampledWavelengths"

    @always_inline
    def get(self, lane: Int) -> SampledWavelengths[1]:
        return SampledWavelengths[1](
            self.lambda_nm.get(lane), self.pdf.get(lane)
        )

    @always_inline
    def set(mut self, lane: Int, value: SampledWavelengths[1]):
        self.lambda_nm.set(lane, value.lambda_nm)
        self.pdf.set(lane, value.pdf)

    @staticmethod
    def visible(u: SIMD[.float32, Self.width]) -> Self:
        var wavelengths = SampledSpectrum[Self.width]()
        var pdf = SampledSpectrum[Self.width]()
        comptime for i in range(4):
            var sample = u + Float32(i) * 0.25
            sample = sample.gt(1).select(sample - 1, sample)
            wavelengths[i] = 538 - Float32(138.888889) * atanh(
                Float32(0.85691062) - Float32(1.82750197) * sample
            )
            pdf[i] = visible_wavelength_pdf(wavelengths[i])
        return Self(wavelengths, pdf)

    @staticmethod
    def uniform(u: SIMD[.float32, Self.width]) -> Self:
        var wavelengths = SampledSpectrum[Self.width]()
        wavelengths.s0 = LAMBDA_MIN * (1 - u) + LAMBDA_MAX * u
        comptime for i in range(1, 4):
            var sample = wavelengths.s0 + Float32(i) * (
                (LAMBDA_MAX - LAMBDA_MIN) * 0.25
            )
            wavelengths[i] = sample.gt(LAMBDA_MAX).select(
                sample - (LAMBDA_MAX - LAMBDA_MIN), sample
            )
        return Self(
            wavelengths,
            SampledSpectrum[Self.width](Float32(1) / (LAMBDA_MAX - LAMBDA_MIN)),
        )

    @always_inline
    def terminate_secondary(mut self, active: SIMD[.bool, Self.width]):
        var terminated = (
            self.pdf.s1.eq(0) & self.pdf.s2.eq(0) & self.pdf.s3.eq(0)
        )
        var change = active & ~terminated
        self.pdf.s0 = change.select(self.pdf.s0 * 0.25, self.pdf.s0)
        self.pdf.s1 = change.select(Float32(0), self.pdf.s1)
        self.pdf.s2 = change.select(Float32(0), self.pdf.s2)
        self.pdf.s3 = change.select(Float32(0), self.pdf.s3)


@always_inline
def cie_xyz[
    width: SIMDLength
](wavelength: SIMD[.float32, width]) -> Vec3[.float32, .WORLD, width]:
    """PBRT dense observer lookup: nearest original integer-nanometer sample."""
    ref table = global_constant[CIE_XYZ]()
    var result = Vec3[.float32, .WORLD, width](0)
    for lane in range(Int(width)):
        if not (wavelength[lane] >= 359.5 and wavelength[lane] < 830.5):
            continue
        var rounded = Int(floor(wavelength[lane] + Float32(0.5)))
        if rounded >= 360 and rounded <= 830:
            result.x[lane] = table[rounded - 360].x
            result.y[lane] = table[rounded - 360].y
            result.z[lane] = table[rounded - 360].z
    return result


@always_inline
def sampled_sensor_xyz[
    width: SIMDLength
](
    spectrum: SampledSpectrum[width], wavelengths: SampledWavelengths[width]
) -> Vec3[.float32, .WORLD, width]:
    """CIE sensor estimator; photometric light scaling belongs at light setup.
    """
    var weighted = spectrum.safe_divide(wavelengths.pdf)
    var result = Vec3[.float32, .WORLD, width](0)
    comptime for i in range(4):
        result += cie_xyz(wavelengths.lambda_nm[i]) * weighted[i]
    return result * Float32(0.25)


@always_inline
def spectral_to_xyz[
    width: SIMDLength
](
    spectrum: SampledSpectrum[width], wavelengths: SampledWavelengths[width]
) -> Vec3[.float32, .WORLD, width]:
    """Normalized XYZ colorimetry, distinct from the film sensor estimator."""
    return sampled_sensor_xyz(spectrum, wavelengths) / CIE_Y_INTEGRAL
