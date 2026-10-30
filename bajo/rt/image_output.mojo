"""Image output shared by renderer backends and examples."""

from std.algorithm import vectorize
from std.math import sqrt
from std.memory import bitcast
from std.sys import simd_width_of
from bajo.core.color import Color


def write_linear_rgb(path: String, pixels: ImmSpan[Color, _]) raises:
    """Write tightly packed little-endian Float32 RGB, preserving channel bits.
    """
    var bytes = List[UInt8](unsafe_uninit_length=len(pixels) * 12)
    var index = 0
    for pixel in pixels:
        comptime for channel in range(3):
            var word = bitcast[.uint32](pixel[channel])
            comptime for byte in range(4):
                bytes[index + byte] = UInt8(word >> UInt32(byte * 8))
            index += 4
    with open(path, "w") as f:
        f.write_bytes(bytes)


@always_inline
def linear_to_gamma[
    length: SIMDLength = 1
](x: SIMD[.float32, length]) -> SIMD[.float32, length]:
    return sqrt(x.le(0.0).select(Float32(0.0), x))


@always_inline
def color_to_byte[
    length: SIMDLength = 1,
    *,
    gamma_correct: Bool = True,
](x: SIMD[.float32, length]) -> SIMD[.uint8, length]:
    comptime if gamma_correct:
        return (256.0 * linear_to_gamma(x).clamp(0.0, 0.999)).cast[.uint8]()
    else:
        return (255.0 * x.clamp(0.0, 1.0)).cast[.uint8]()


def _encode_ppm[
    width: SIMDLength = simd_width_of[Float32](),
    *,
    gamma_correct: Bool = True,
](pixels: ImmSpan[Color, _]) -> List[UInt8]:
    var count = len(pixels) * 3
    var bytes = List[UInt8](unsafe_uninit_length=count)
    var source = pixels.unsafe_ptr().unsafe_bitcast[Float32]()
    var dest = bytes.unsafe_ptr()

    def encode[lanes: Int](first: Int) {imm}:
        dest.unsafe_store(
            first,
            color_to_byte[gamma_correct=gamma_correct](
                source.unsafe_load[width=lanes](first)
            ),
        )

    vectorize[width](count, encode)
    return bytes^


def write_ppm_from_colors[
    gamma_correct: Bool = True
](path: String, width: Int, height: Int, pixels: ImmSpan[Color, _],) raises:
    with open(path, "w") as f:
        f.write(t"P6\n{width} {height}\n255\n")
        var bytes = _encode_ppm[gamma_correct=gamma_correct](pixels)
        f.write_bytes(bytes)
