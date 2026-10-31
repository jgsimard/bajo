"""Film response and sensor-space clamping of complete path samples."""

from std.algorithm import vectorize
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.sys import simd_width_of
from std.sys.info import size_of
from std.utils import IndexList
from std.utils.numerics import isfinite
from std.memory import bitcast
from bajo.core import dot, Vec3
from bajo.core import Vec3f32
from bajo.core.tensor_layout import RecordTile, record_tile
from bajo.core.colorimetry import srgb_to_xyz_matrix
from bajo.core.mat import inverse
from bajo.core.sampled_spectrum import (
    SampledSpectrum,
    SampledWavelengths,
    sampled_sensor_xyz,
)

comptime Color = Vec3f32[.WORLD]


def _channel_mask[length: Int, channel: Int]() -> IndexList[2 * length]:
    var mask = IndexList[2 * length]()
    comptime for lane in range(2 * length):
        comptime offset = 3 * lane + channel
        mask[lane] = (
            offset if offset < 2 * length else offset + length
        ) if lane < length else 0
    return mask


def _rg_mask[length: Int, part: Int]() -> IndexList[length]:
    var mask = IndexList[length]()
    comptime for lane in range(length):
        comptime offset = part * length + lane
        mask[lane] = (
            offset // 3 + (offset % 3) * length if offset % 3 < 2 else 0
        )
    return mask


def _blue_mask[length: Int, part: Int]() -> IndexList[length]:
    var mask = IndexList[length]()
    comptime for lane in range(length):
        comptime offset = part * length + lane
        mask[lane] = length + offset // 3 if offset % 3 == 2 else lane
    return mask


struct FilmResponse(Copyable, DevicePassable, Writable):
    var exposure: Float32
    var red: Color
    var green: Color
    var blue: Color
    var max_component_value: Float32
    # Convert linear input RGB to sensor coordinates for the clamp. These rows
    # are identity for programmatic RGB films and CIE XYZ for PBRT's default.
    var sensor_red: Color
    var sensor_green: Color
    var sensor_blue: Color
    var xyz_red: Color
    var xyz_green: Color
    var xyz_blue: Color
    var cie_sensor: UInt32

    def __init__(
        out self,
        exposure: Float32 = 1.0,
        red: Color = Color(1.0, 0.0, 0.0),
        green: Color = Color(0.0, 1.0, 0.0),
        blue: Color = Color(0.0, 0.0, 1.0),
        max_component_value: Float32 = bitcast[.float32](UInt32(0x7F800000)),
    ):
        self.exposure = exposure
        self.red = red
        self.green = green
        self.blue = blue
        self.max_component_value = max_component_value
        self.sensor_red = Color(1, 0, 0)
        self.sensor_green = Color(0, 1, 0)
        self.sensor_blue = Color(0, 0, 1)
        self.xyz_red = Color(0)
        self.xyz_green = Color(0)
        self.xyz_blue = Color(0)
        self.cie_sensor = 0

    def use_cie1931(mut self) raises:
        var matrix = srgb_to_xyz_matrix()
        self.sensor_red = Color(
            Float32(matrix[0][0]), Float32(matrix[0][1]), Float32(matrix[0][2])
        )
        self.sensor_green = Color(
            Float32(matrix[1][0]), Float32(matrix[1][1]), Float32(matrix[1][2])
        )
        self.sensor_blue = Color(
            Float32(matrix[2][0]), Float32(matrix[2][1]), Float32(matrix[2][2])
        )
        var rgb = inverse(matrix)
        self.xyz_red = Color(
            Float32(rgb[0][0]), Float32(rgb[0][1]), Float32(rgb[0][2])
        )
        self.xyz_green = Color(
            Float32(rgb[1][0]), Float32(rgb[1][1]), Float32(rgb[1][2])
        )
        self.xyz_blue = Color(
            Float32(rgb[2][0]), Float32(rgb[2][1]), Float32(rgb[2][2])
        )
        self.cie_sensor = 1

    @always_inline
    def apply_spectral_sample[
        width: SIMDLength
    ](
        self,
        radiance: SampledSpectrum[width],
        wavelengths: SampledWavelengths[width],
    ) -> Vec3[.float32, .WORLD, width]:
        """Integrate four actual wavelengths, clamp sensor XYZ, then output RGB.

        Only the physical CIE observer is implemented here. RGB-fitted camera
        response matrices are not a substitute for a spectral sensor response.
        The caller supplies radiance with the light's photometric scale already
        applied, matching PBRT's PixelSensor convention.
        """
        debug_assert["safe"](
            self.cie_sensor == 1,
            "spectral film requires a configured CIE sensor",
        )
        var sensor = sampled_sensor_xyz(radiance, wavelengths) * self.exposure
        var maximum = sensor.max_component()
        var limit = SIMD[.float32, width](self.max_component_value)
        sensor = sensor * maximum.gt(limit).select(
            limit / maximum.gt(0).select(maximum, 1), 1
        )
        var rgb = Vec3[.float32, .WORLD, width](
            dot(self.xyz_red.broadcast[width](), sensor),
            dot(self.xyz_green.broadcast[width](), sensor),
            dot(self.xyz_blue.broadcast[width](), sensor),
        )
        return Vec3[.float32, .WORLD, width](
            dot(self.red.broadcast[width](), rgb),
            dot(self.green.broadcast[width](), rgb),
            dot(self.blue.broadcast[width](), rgb),
        )

    @always_inline
    def apply_sample[
        length: SIMDLength = 1
    ](self, color: Vec3[.float32, .WORLD, length]) -> Vec3[
        .float32, .WORLD, length
    ]:
        """Clamp the complete exposed sensor sample, then convert to output.

        The clamp is a common scale on all channels. Applying that scale to
        input RGB commutes with the linear output matrix and avoids a lossy
        RGB -> XYZ -> RGB round trip in the current RGB transport path.
        Spectral transport must supply actual wavelength-integrated sensor
        coordinates rather than reconstructing them from an RGB projection.
        """
        var exposed = color * self.exposure
        var sensor = Vec3[.float32, .WORLD, length](
            dot(self.sensor_red.broadcast[length](), exposed),
            dot(self.sensor_green.broadcast[length](), exposed),
            dot(self.sensor_blue.broadcast[length](), exposed),
        )
        var maximum = sensor.max_component()
        var limit = SIMD[.float32, length](self.max_component_value)
        var clamped = exposed * maximum.gt(limit).select(
            limit / maximum.gt(0.0).select(maximum, 1.0), 1.0
        )
        return Vec3[.float32, .WORLD, length](
            dot(self.red.broadcast[length](), clamped),
            dot(self.green.broadcast[length](), clamped),
            dot(self.blue.broadcast[length](), clamped),
        )

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "FilmResponse"

    @always_inline
    def apply[
        length: SIMDLength = 1
    ](self, color: Vec3[.float32, .WORLD, length]) -> Vec3[
        .float32, .WORLD, length
    ]:
        return (
            Vec3[.float32, .WORLD, length](
                dot(self.red.broadcast[length](), color),
                dot(self.green.broadcast[length](), color),
                dot(self.blue.broadcast[length](), color),
            )
            * self.exposure
        )

    @always_inline
    def _apply_block[
        length: SIMDLength, origin: Origin[mut=True]
    ](
        self,
        pixels: RecordTile[.float32, 3, origin],
        first: Int,
        scale: Float32 = 1.0,
    ):
        """Transform `length` pixels; callers provide a complete block."""
        var color: Vec3[.float32, .WORLD, length]
        comptime if length == 1:
            color = Vec3[.float32, .WORLD, length](
                pixels[first, 0], pixels[first, 1], pixels[first, 2]
            )
        else:
            # Two overlapping contiguous loads cover exactly 3*length floats.
            # Shuffles separate RGB channels without gathers or a layout copy.
            var a = pixels.load[width=2 * length]((first, 0))
            var b = pixels.load[width=2 * length](
                (first + length // 3, length % 3)
            )
            color = Vec3[.float32, .WORLD, length](
                a.shuffle[_channel_mask[Int(length), 0]()](b).slice[length](),
                a.shuffle[_channel_mask[Int(length), 1]()](b).slice[length](),
                a.shuffle[_channel_mask[Int(length), 2]()](b).slice[length](),
            )
        var result = self.apply(color * scale)
        comptime if length == 1:
            pixels[first, 0] = result.x[0]
            pixels[first, 1] = result.y[0]
            pixels[first, 2] = result.z[0]
        else:
            comptime for part in range(3):
                var packed = result.x.shuffle[_rg_mask[Int(length), part]()](
                    result.y
                ).shuffle[_blue_mask[Int(length), part]()](result.z)
                pixels.store[width=length](
                    (first + part * length // 3, part * length % 3), packed
                )

    def apply_to[
        width: SIMDLength = simd_width_of[Float32]()
    ](self, pixels: MutSpan[Color, _], scale: Float32 = 1.0):
        """Apply sample scaling and film response to a CPU pixel buffer."""
        comptime assert size_of[Color]() == 3 * size_of[Float32]()
        var view = record_tile[.float32, 3](
            pixels.unsafe_ptr().unsafe_bitcast[Float32](), len(pixels)
        )

        def apply_block[lanes: Int](first: Int) {imm}:
            self._apply_block[lanes](view, first, scale)

        vectorize[width](len(pixels), apply_block)

    def is_identity(self) -> Bool:
        return (
            self.exposure == 1.0
            and self.red.x == 1.0
            and self.red.y == 0.0
            and self.red.z == 0.0
            and self.green.x == 0.0
            and self.green.y == 1.0
            and self.green.z == 0.0
            and self.blue.x == 0.0
            and self.blue.y == 0.0
            and self.blue.z == 1.0
            and not isfinite(self.max_component_value)
        )

    def validate(self) raises:
        if (
            self.max_component_value != self.max_component_value
            or self.max_component_value < 0.0
        ):
            raise Error(
                "film max component value must be non-negative and not NaN"
            )
        if (
            not isfinite(self.exposure)
            or self.exposure < 0.0
            or not self.red.is_finite()[0]
            or not self.green.is_finite()[0]
            or not self.blue.is_finite()[0]
            or not self.sensor_red.is_finite()[0]
            or not self.sensor_green.is_finite()[0]
            or not self.sensor_blue.is_finite()[0]
            or not self.xyz_red.is_finite()[0]
            or not self.xyz_green.is_finite()[0]
            or not self.xyz_blue.is_finite()[0]
        ):
            raise Error(
                "film response must be finite with non-negative exposure"
            )
