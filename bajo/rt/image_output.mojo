"""Image output shared by renderer backends and examples."""

from std.math import sqrt
from bajo.rt.render_types import Color


def linear_to_gamma(x: Float32) -> Float32:
    if x <= 0.0:
        return 0.0
    return sqrt(x)


def color_to_byte(x: Float32) -> UInt8:
    return UInt8((256.0 * linear_to_gamma(x).clamp(0.0, 0.999)))


def write_ppm_from_colors(
    path: String,
    width: Int,
    height: Int,
    pixels: ImmSpan[Color, _],
) raises:
    with open(path, "w") as f:
        f.write(t"P6\n{width} {height}\n255\n")

        var bytes = List[UInt8](length=width * height * 3, fill=0)
        var out_i = 0

        for p in pixels:
            bytes[out_i + 0] = color_to_byte(p.x)
            bytes[out_i + 1] = color_to_byte(p.y)
            bytes[out_i + 2] = color_to_byte(p.z)
            out_i += 3

        f.write_bytes(bytes)
