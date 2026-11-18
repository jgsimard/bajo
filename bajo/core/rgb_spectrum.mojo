"""PBRT's sRGB sigmoid-polynomial reconstruction, using its original table."""

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.builtin.globals import global_constant
from std.memory import bitcast
from std.math import floor, sqrt, fma
from std.utils.numerics import isfinite
from std.reflection import source_location
from std.os import getenv
from std.ffi import external_call
from std.os.path import dirname, join, exists
from std.sys.arg import argv
from std.sys.intrinsics import masked_load
from bajo.core.vec import Vec3
from bajo.core.cie_data import CIE_XYZ, D65
from bajo.core.sampled_spectrum import (
    SampledSpectrum,
    SampledWavelengths,
    CIE_Y_INTEGRAL,
)

comptime RGB_SPECTRUM_RESOLUTION = 64
comptime RGB_SPECTRUM_COEFFICIENTS = 3 * 64 * 64 * 64 * 3
comptime RGB_SPECTRUM_D65_OFFSET = 64 + RGB_SPECTRUM_COEFFICIENTS


@inline(.always)
def _trilinear_coefficients[
    width: SIMDLength
](
    c000: SIMD[.float32, width],
    c100: SIMD[.float32, width],
    c010: SIMD[.float32, width],
    c110: SIMD[.float32, width],
    c001: SIMD[.float32, width],
    c101: SIMD[.float32, width],
    c011: SIMD[.float32, width],
    c111: SIMD[.float32, width],
    dx: SIMD[.float32, width],
    dy: SIMD[.float32, width],
    dz: SIMD[.float32, width],
) -> SIMD[.float32, width]:
    var a00 = c000 * (1 - dx) + c100 * dx
    var a10 = c010 * (1 - dx) + c110 * dx
    var a01 = c001 * (1 - dx) + c101 * dx
    var a11 = c011 * (1 - dx) + c111 * dx
    return ((1 - dy) * a00 + dy * a10) * (1 - dz) + (
        (1 - dy) * a01 + dy * a11
    ) * dz


@fieldwise_init
struct SigmoidPolynomial[width: SIMDLength = 1](TrivialRegisterPassable):
    var a: SIMD[.float32, Self.width]
    var b: SIMD[.float32, Self.width]
    var c: SIMD[.float32, Self.width]

    @inline(.always)
    def evaluate(
        self, wavelength: SIMD[.float32, Self.width]
    ) -> SIMD[.float32, Self.width]:
        var x = fma(fma(self.a, wavelength, self.b), wavelength, self.c)
        var finite = isfinite(x)
        var safe = finite.select(x, 0)
        var value = 0.5 + safe / (2 * sqrt(1 + safe * safe))
        return finite.select(value, x.gt(0).select(Float32(1), 0))


struct RgbSpectrumView(Copyable, DevicePassable, TrivialRegisterPassable):
    comptime ALBEDO = UInt32(0)
    comptime UNBOUNDED = UInt32(1)
    comptime ILLUMINANT = UInt32(2)
    var data: ImmPointer[Float32, ImmUntrackedOrigin]

    def __init__(out self, data: ImmPointer[Float32, ImmUntrackedOrigin]):
        self.data = data

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "RgbSpectrumView"

    @inline(.always)
    def _coefficient[
        width: SIMDLength
    ](
        self,
        axis: SIMD[.int32, width],
        x: SIMD[.int32, width],
        y: SIMD[.int32, width],
        z: SIMD[.int32, width],
        channel: Int,
    ) -> SIMD[.float32, width]:
        var indices = 64 + ((((axis * 64) + z) * 64 + y) * 64 + x) * 3
        return self.data.unsafe_gather(indices + Int32(channel))

    @inline(.always)
    def coefficients[
        width: SIMDLength
    ](self, rgb: Vec3[.float32, .WORLD, width]) -> SigmoidPolynomial[width]:
        comptime if width == 2 or width == 4:
            # Short packets use SIMD over each color's adjacent coefficients;
            # wider packets amortize gathers across independent paths instead.
            var result = SigmoidPolynomial[width](0, 0, 0)
            comptime for lane in range(Int(width)):
                var scalar = self.coefficients(
                    Vec3[.float32, .WORLD](
                        rgb.x[lane], rgb.y[lane], rgb.z[lane]
                    )
                )
                result.a[lane] = scalar.a[0]
                result.b[lane] = scalar.b[0]
                result.c[lane] = scalar.c[0]
            return result
        var r = rgb.x
        var g = rgb.y
        var b = rgb.z
        debug_assert["safe"](
            (
                r.ge(0) & r.le(1) & g.ge(0) & g.le(1) & b.ge(0) & b.le(1)
            ).reduce_and(),
            "RGB spectral coefficients require finite values in [0, 1]",
        )
        var gray = r.eq(g) & g.eq(b)
        if gray.reduce_and():
            return SigmoidPolynomial[width](0, 0, (r - 0.5) / sqrt(r * (1 - r)))
        # Preserve PBRT's axis tie-breaking and sanitize gray lanes before
        # divisions and gathers, including black and white in mixed packets.
        var red = r.gt(g) & r.gt(b)
        var green = r.le(g) & g.gt(b)
        var axis = red.select(Int32(0), green.select(Int32(1), Int32(2)))
        var z = gray.select(Float32(1), red.select(r, green.select(g, b)))
        var x = red.select(g, green.select(b, r)) * 63 / z
        var y = red.select(b, green.select(r, g)) * 63 / z
        var xi = min(x.cast[.int32](), 62)
        var yi = min(y.cast[.int32](), 62)
        var lo = SIMD[.int32, width](0)
        var hi = SIMD[.int32, width](63)
        comptime if width == 1:
            while lo + 1 < hi:
                var mid = (lo + hi) // 2
                if self.data.unsafe_gather(mid) < z:
                    lo = mid
                else:
                    hi = mid
        else:
            comptime for _ in range(6):
                var searching = (lo + 1).lt(hi)
                var mid = (lo + hi) // 2
                var right = self.data.unsafe_gather(mid).lt(z)
                lo = (searching & right).select(mid, lo)
                hi = (searching & ~right).select(mid, hi)
        var zi = lo
        var dx = x - xi.cast[.float32]()
        var dy = y - yi.cast[.float32]()
        var z0 = self.data.unsafe_gather(zi)
        var dz = (z - z0) / (self.data.unsafe_gather(zi + 1) - z0)
        comptime if width == 1:
            # At width one, SIMD runs across the adjacent polynomial
            # coefficients instead, masking the fourth word of each record.
            var base = Int(64 + ((((axis * 64) + zi) * 64 + yi) * 64 + xi) * 3)
            var ptr = self.data.unsafe_offset(base)
            comptime row = RGB_SPECTRUM_RESOLUTION * 3
            comptime slab = RGB_SPECTRUM_RESOLUTION * row
            var mask = SIMD[.bool, 4](True, True, True, False)
            var zero = SIMD[.float32, 4](0)
            var c = _trilinear_coefficients(
                masked_load(ptr, mask, zero),
                masked_load(ptr.unsafe_offset(3), mask, zero),
                masked_load(ptr.unsafe_offset(row), mask, zero),
                masked_load(ptr.unsafe_offset(row + 3), mask, zero),
                masked_load(ptr.unsafe_offset(slab), mask, zero),
                masked_load(ptr.unsafe_offset(slab + 3), mask, zero),
                masked_load(ptr.unsafe_offset(slab + row), mask, zero),
                masked_load(ptr.unsafe_offset(slab + row + 3), mask, zero),
                SIMD[.float32, 4](dx[0]),
                SIMD[.float32, 4](dy[0]),
                SIMD[.float32, 4](dz[0]),
            )
            return SigmoidPolynomial[width](c[0], c[1], c[2])
        var c = Array[SIMD[.float32, width], 3](uninitialized=True)
        comptime for channel in range(3):
            c[channel] = _trilinear_coefficients(
                self._coefficient(axis, xi, yi, zi, channel),
                self._coefficient(axis, xi + 1, yi, zi, channel),
                self._coefficient(axis, xi, yi + 1, zi, channel),
                self._coefficient(axis, xi + 1, yi + 1, zi, channel),
                self._coefficient(axis, xi, yi, zi + 1, channel),
                self._coefficient(axis, xi + 1, yi, zi + 1, channel),
                self._coefficient(axis, xi, yi + 1, zi + 1, channel),
                self._coefficient(axis, xi + 1, yi + 1, zi + 1, channel),
                dx,
                dy,
                dz,
            )
        var gray_c = SIMD[.float32, width](0)
        if gray.reduce_or():
            var value = gray.select(r, 0.5)
            gray_c = (value - 0.5) / sqrt(value * (1 - value))
        return SigmoidPolynomial[width](
            gray.select(Float32(0), c[0]),
            gray.select(Float32(0), c[1]),
            gray.select(gray_c, c[2]),
        )

    @inline(.always)
    def illuminant[
        width: SIMDLength
    ](self, wavelength: SIMD[.float32, width]) -> SIMD[.float32, width]:
        var valid = wavelength.ge(359.5) & wavelength.lt(830.5)
        comptime if width == 1:
            if not valid[0]:
                return SIMD[.float32, width](0)
        var nm = valid.select(wavelength, 360)
        var index = floor(nm + 0.5).cast[.int32]() - 360
        return valid.select(
            self.data.unsafe_gather(index + RGB_SPECTRUM_D65_OFFSET), 0
        )

    @inline(.always)
    def sample[
        width: SIMDLength
    ](
        self,
        rgb: Vec3[.float32, .WORLD, width],
        wavelengths: SampledWavelengths[width],
        kind: UInt32 = 0,
    ) -> SampledSpectrum[width]:
        var scale = SIMD[.float32, width](1)
        var normalized = rgb
        if kind != Self.ALBEDO:
            scale = 2 * rgb.max_component()
            normalized = rgb / scale.ne(0).select(scale, 1)
        var polynomial = self.coefficients(normalized)
        var result = SampledSpectrum[width]()
        comptime for i in range(4):
            result[i] = scale * polynomial.evaluate(wavelengths.lambda_nm[i])
            if kind == Self.ILLUMINANT:
                result[i] *= self.illuminant(wavelengths.lambda_nm[i])
        return result


struct RgbSpectrumTable:
    var values: List[Float32]

    def __init__(out self, var values: List[Float32]) raises:
        if len(values) != RGB_SPECTRUM_D65_OFFSET:
            raise Error(
                "sRGB spectrum table must retain all 64-cubed coefficients"
            )
        if values[0] != 0 or values[63] != 1:
            raise Error("invalid sRGB spectrum table nodes")
        for index in range(len(values)):
            if not isfinite(values[index]):
                raise Error("nonfinite sRGB spectrum table entry")
            if index > 0 and index < 64 and values[index] <= values[index - 1]:
                raise Error("sRGB spectrum nodes must increase")
        self.values = values^
        ref observer = global_constant[CIE_XYZ]()
        ref d65 = global_constant[D65]()
        var integral = Float32(0)
        for nm in range(360, 831):
            var i = 2 * ((nm - 300) // 5)
            var t = Float32(nm % 5) / 5
            var v = (
                d65[i + 1] if t == 0 else (1 - t) * d65[i + 1] + t * d65[i + 3]
            )
            integral += v * observer[nm - 360].y
        var normalization = CIE_Y_INTEGRAL / integral
        for nm in range(360, 831):
            var i = 2 * ((nm - 300) // 5)
            var t = Float32(nm % 5) / 5
            # PBRT normalizes the piecewise spectrum before densely sampling it.
            var v = d65[i + 1] * normalization
            if t != 0:
                v = (1 - t) * v + t * (d65[i + 3] * normalization)
            self.values.append(v)

    @staticmethod
    def read_default() raises -> Self:
        """Use an explicit override, package location, or bundled executable data.

        Installed/source packages compiled with an absolute import path retain
        their module location. Portable binaries can bundle srgb_spectra.bin
        beside the executable or set BAJO_SPECTRUM_TABLE to its installed path.
        """
        var configured = getenv("BAJO_SPECTRUM_TABLE")
        if configured:
            return Self.read(configured)
        var package_path = join(
            dirname(String(source_location().file_name())),
            "data/srgb_spectra.bin",
        )
        if package_path.startswith("/") and exists(package_path):
            return Self.read(package_path)
        var executable = String(argv()[0])
        # Linux is the supported package platform. /proc resolves PATH-based
        # invocation and symlinks independently of the current directory.
        var link = String("/proc/self/exe")
        var buffer = List[UInt8](unsafe_uninit_length=4096)
        var count = external_call["readlink", Int](
            link.as_c_string_span(), buffer.unsafe_ptr(), UInt(len(buffer))
        )
        if count > 0 and count < len(buffer):
            executable = String(
                StringSpan(
                    unsafe_from_utf8=ImmSpan(
                        unsafe_ptr=buffer.unsafe_ptr(), length=count
                    )
                )
            )
        var bundled = join(dirname(executable), "srgb_spectra.bin")
        if exists(bundled):
            return Self.read(bundled)
        # Source execution with a relative -I remains convenient in a checkout.
        if exists(package_path):
            return Self.read(package_path)
        raise Error(
            "spectral reconstruction table not found; bundle srgb_spectra.bin"
            " beside the executable or set BAJO_SPECTRUM_TABLE"
        )

    @staticmethod
    def read(path: String) raises -> Self:
        var bytes: List[UInt8]
        with open(path, "r") as file:
            bytes = file.read_bytes()
        if len(bytes) != 16 + 4 * RGB_SPECTRUM_D65_OFFSET:
            raise Error("invalid sRGB spectrum table size")
        var magic = "BJRGB001".as_bytes()
        for index in range(8):
            if bytes[index] != magic[index]:
                raise Error("invalid sRGB spectrum table signature")
        if bytes[8] != 64 or bytes[9] != 0 or bytes[10] != 0 or bytes[11] != 0:
            raise Error("sRGB spectral reconstruction requires resolution 64")
        var count = UInt32(0)
        comptime for byte in range(4):
            count |= UInt32(bytes[12 + byte]) << UInt32(8 * byte)
        if count != UInt32(RGB_SPECTRUM_D65_OFFSET):
            raise Error("invalid sRGB spectrum coefficient count")
        var values = List[Float32](unsafe_uninit_length=RGB_SPECTRUM_D65_OFFSET)
        for index in range(len(values)):
            var word = UInt32(0)
            comptime for byte in range(4):
                word |= UInt32(bytes[16 + 4 * index + byte]) << UInt32(8 * byte)
            values[index] = bitcast[.float32](word)
        return Self(values^)

    def view(self) -> RgbSpectrumView:
        """Borrow spectral coefficients; keep this owner alive through sampling.
        """
        return RgbSpectrumView(
            self.values.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin]()
        )
