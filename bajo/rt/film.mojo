"""Linear RGB film response, applied once after sample accumulation."""

from std.algorithm import vectorize
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.sys import simd_width_of
from std.sys.info import size_of
from std.utils import IndexList
from std.utils.numerics import isfinite
from bajo.core import dot, Vec3
from bajo.core import Vec3f32
from bajo.core.tensor_layout import RecordTile, record_tile

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

    def __init__(
        out self,
        exposure: Float32 = 1.0,
        red: Color = Color(1.0, 0.0, 0.0),
        green: Color = Color(0.0, 1.0, 0.0),
        blue: Color = Color(0.0, 0.0, 1.0),
    ):
        self.exposure = exposure
        self.red = red
        self.green = green
        self.blue = blue

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
        )

    def validate(self) raises:
        if (
            not isfinite(self.exposure)
            or self.exposure < 0.0
            or not self.red.is_finite()[0]
            or not self.green.is_finite()[0]
            or not self.blue.is_finite()[0]
        ):
            raise Error(
                "film response must be finite with non-negative exposure"
            )
