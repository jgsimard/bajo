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


@fieldwise_init
struct SigmoidPolynomial[width: SIMDLength = 1](TrivialRegisterPassable):
    var a: SIMD[.float32, Self.width]
    var b: SIMD[.float32, Self.width]
    var c: SIMD[.float32, Self.width]

    @always_inline
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

    @always_inline
    def _coefficient(
        self, axis: Int, x: Int, y: Int, z: Int, channel: Int
    ) -> Float32:
        return self.data[
            unsafe_offset=64
            + ((((axis * 64) + z) * 64 + y) * 64 + x) * 3
            + channel
        ]

    @always_inline
    def coefficients[
        width: SIMDLength
    ](self, rgb: Vec3[.float32, .WORLD, width]) -> SigmoidPolynomial[width]:
        var result = SigmoidPolynomial[width](0, 0, 0)
        for lane in range(Int(width)):
            var r = rgb.x[lane]
            var g = rgb.y[lane]
            var b = rgb.z[lane]
            debug_assert["safe"](
                r >= 0 and r <= 1 and g >= 0 and g <= 1 and b >= 0 and b <= 1,
                "RGB spectral coefficients require finite values in [0, 1]",
            )
            if r == g and g == b:
                result.c[lane] = (r - 0.5) / sqrt(r * (1 - r))
                continue
            var axis = (0 if r > b else 2) if r > g else (1 if g > b else 2)
            var z = rgb[axis][lane]
            var x = rgb[(axis + 1) % 3][lane] * 63 / z
            var y = rgb[(axis + 2) % 3][lane] * 63 / z
            var xi = min(Int(x), 62)
            var yi = min(Int(y), 62)
            var lo = 0
            var hi = 63
            while lo + 1 < hi:
                var mid = (lo + hi) // 2
                if self.data[unsafe_offset=mid] < z:
                    lo = mid
                else:
                    hi = mid
            var zi = lo
            var dx = x - Float32(xi)
            var dy = y - Float32(yi)
            var dz = (z - self.data[unsafe_offset=zi]) / (
                self.data[unsafe_offset=zi + 1] - self.data[unsafe_offset=zi]
            )
            var c = SIMD[.float32, 4](0)
            for channel in range(3):
                var a00 = (
                    self._coefficient(axis, xi, yi, zi, channel) * (1 - dx)
                    + self._coefficient(axis, xi + 1, yi, zi, channel) * dx
                )
                var a10 = (
                    self._coefficient(axis, xi, yi + 1, zi, channel) * (1 - dx)
                    + self._coefficient(axis, xi + 1, yi + 1, zi, channel) * dx
                )
                var a01 = (
                    self._coefficient(axis, xi, yi, zi + 1, channel) * (1 - dx)
                    + self._coefficient(axis, xi + 1, yi, zi + 1, channel) * dx
                )
                var a11 = (
                    self._coefficient(axis, xi, yi + 1, zi + 1, channel)
                    * (1 - dx)
                    + self._coefficient(axis, xi + 1, yi + 1, zi + 1, channel)
                    * dx
                )
                c[channel] = ((1 - dy) * a00 + dy * a10) * (1 - dz) + (
                    (1 - dy) * a01 + dy * a11
                ) * dz
            result.a[lane] = c[0]
            result.b[lane] = c[1]
            result.c[lane] = c[2]
        return result

    @always_inline
    def illuminant[
        width: SIMDLength
    ](self, wavelength: SIMD[.float32, width]) -> SIMD[.float32, width]:
        var result = SIMD[.float32, width](0)
        for lane in range(Int(width)):
            if wavelength[lane] >= 359.5 and wavelength[lane] < 830.5:
                var index = Int(floor(wavelength[lane] + Float32(0.5))) - 360
                result[lane] = self.data[
                    unsafe_offset=RGB_SPECTRUM_D65_OFFSET + index
                ]
        return result

    @always_inline
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
