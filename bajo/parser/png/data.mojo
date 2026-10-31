"""PNG sample decoding for data textures, without color or alpha compositing.

Noninterlaced grayscale, RGB, indexed, grayscale-alpha and RGBA images retain
their actual samples. PNG encoding metadata never changes normal/mask data.
"""

from std.math import abs
from bajo.parser.checksum import crc32
from bajo.parser.deflate import inflate_zlib
from bajo.rt.material_types import ImageTexture


@always_inline
def _u32(bytes: ImmSpan[UInt8, _], offset: Int) -> UInt32:
    return (
        (UInt32(bytes[offset]) << 24)
        | (UInt32(bytes[offset + 1]) << 16)
        | (UInt32(bytes[offset + 2]) << 8)
        | UInt32(bytes[offset + 3])
    )


@always_inline
def _u16(bytes: ImmSpan[UInt8, _], offset: Int) -> Int:
    return Int(bytes[offset]) * 256 + Int(bytes[offset + 1])


def _unfilter_data(
    mut raw: List[UInt8], row_bytes: Int, height: Int, bpp: Int
) raises:
    var stride = row_bytes + 1
    for y in range(height):
        var start = y * stride + 1
        var filter = raw[y * stride]
        if filter > 4:
            raise Error("invalid PNG data filter type")
        for x in range(row_bytes):
            var left = Int(raw[start + x - bpp]) if x >= bpp else 0
            var up = Int(raw[start + x - stride]) if y else 0
            var upper_left = (
                Int(raw[start + x - stride - bpp]) if y and x >= bpp else 0
            )
            var predictor = 0
            match filter:
                case 1:
                    predictor = left
                case 2:
                    predictor = up
                case 3:
                    predictor = (left + up) // 2
                case 4:
                    var a = abs(up - upper_left)
                    var b = abs(left - upper_left)
                    var c = abs(left + up - 2 * upper_left)
                    predictor = (
                        left if a <= b
                        and a <= c else up if b <= c else upper_left
                    )
            raw[start + x] += UInt8(predictor)


def parse_data_png(bytes: ImmSpan[UInt8, _]) raises -> ImageTexture:
    """Return unassociated linear sample values and an optional alpha plane."""
    if (
        len(bytes) < 33
        or _u32(bytes, 0) != 0x89504E47
        or _u32(bytes, 4) != 0x0D0A1A0A
    ):
        raise Error("invalid PNG data image signature")
    if _u32(bytes, 8) != 13 or _u32(bytes, 12) != 0x49484452:
        raise Error("PNG data image requires an IHDR chunk")
    var width = Int(_u32(bytes, 16))
    var height = Int(_u32(bytes, 20))
    var depth = Int(bytes[24])
    var color = bytes[25]
    if width <= 0 or height <= 0 or width > 0x7FFFFFFF or height > 0x7FFFFFFF:
        raise Error("invalid PNG data image dimensions")
    if bytes[26] != 0 or bytes[27] != 0 or bytes[28] != 0:
        raise Error("PNG data textures require standard noninterlaced encoding")
    var channels: Int
    match color:
        case 0 | 3:
            channels = 1
        case 2:
            channels = 3
        case 4:
            channels = 2
        case 6:
            channels = 4
        case _:
            raise Error("invalid PNG data image color type")
    if not (
        depth == 1 or depth == 2 or depth == 4 or depth == 8 or depth == 16
    ):
        raise Error("invalid PNG data image bit depth")
    if (color != 0 and color != 3 and depth < 8) or (
        color == 3 and depth == 16
    ):
        raise Error("invalid PNG data image color type / bit depth")
    var row_bytes = (width * channels * depth + 7) // 8
    var stride = row_bytes + 1
    if height > Int.MAX // stride or width > Int.MAX // height // 3:
        raise Error("PNG data image dimensions overflow storage")
    var bpp = max(1, (channels * depth + 7) // 8)
    var compressed = List[UInt8]()
    var palette = List[UInt8]()
    var transparency = List[UInt8]()
    var seen_palette = False
    var seen_transparency = False
    var seen_data = False
    var data_ended = False
    var ended = False
    var offset = 8
    while offset <= len(bytes) - 12:
        var size = Int(_u32(bytes, offset))
        if size > 0x7FFFFFFF or size > len(bytes) - offset - 12:
            raise Error("truncated PNG data image chunk")
        var tag = _u32(bytes, offset + 4)
        var begin = offset + 8
        var end = begin + size
        if crc32(bytes[offset + 4 : end]) != _u32(bytes, end):
            raise Error("PNG data image chunk checksum mismatch")
        match tag:
            case 0x49484452:  # IHDR
                if offset != 8:
                    raise Error("duplicate PNG data image header")
            case 0x504C5445:  # PLTE
                if (
                    seen_palette
                    or seen_data
                    or size == 0
                    or size % 3
                    or size > 768
                    or color == 0
                    or color == 4
                ):
                    raise Error("invalid PNG data image palette")
                palette.extend(bytes[begin:end])
                seen_palette = True
            case 0x74524E53:  # tRNS
                if seen_transparency or seen_data or color == 4 or color == 6:
                    raise Error("invalid PNG data image transparency")
                if (
                    (color == 0 and size != 2)
                    or (color == 2 and size != 6)
                    or (
                        color == 3
                        and (not seen_palette or size > len(palette) // 3)
                    )
                ):
                    raise Error("invalid PNG data image transparency length")
                transparency.extend(bytes[begin:end])
                seen_transparency = True
            case 0x49444154:  # IDAT
                if data_ended or (color == 3 and not seen_palette):
                    raise Error("invalid PNG data image chunk order")
                compressed.extend(bytes[begin:end])
                seen_data = True
            case 0x49454E44:  # IEND
                if size != 0 or not seen_data:
                    raise Error("invalid PNG data image end chunk")
                ended = True
            case _:
                if tag & 0x20000000 == 0:
                    raise Error("unknown critical PNG data image chunk")
        if seen_data and tag != 0x49444154:
            data_ended = True
        offset = end + 4
        if ended:
            break
    if not ended:
        raise Error("PNG data image is missing its end chunk")
    if color == 3 and len(palette) // 3 > (1 << depth):
        raise Error("PNG data image palette exceeds its bit depth")
    var raw = List[UInt8](unsafe_uninit_length=stride * height)
    if inflate_zlib(compressed, raw) != len(raw):
        raise Error("unexpected PNG data image scanline byte count")
    _unfilter_data(raw, row_bytes, height, bpp)
    var pixels = List[Float32](unsafe_uninit_length=width * height * 3)
    var has_alpha = color == 4 or color == 6 or seen_transparency
    var alpha = List[Float32](
        length=width * height if has_alpha else 0, fill=1.0
    )
    var maximum = (1 << depth) - 1
    for y in range(height):
        var row = y * stride + 1
        for x in range(width):
            var component = SIMD[.int32, 4](0)
            for channel in range(channels):
                var sample = x * channels + channel
                var value: Int
                if depth == 16:
                    value = _u16(raw, row + 2 * sample)
                elif depth == 8:
                    value = Int(raw[row + sample])
                else:
                    var bit = sample * depth
                    value = (
                        Int(raw[row + bit // 8]) >> (8 - depth - bit % 8)
                    ) & maximum
                component[channel] = Int32(value)
            var pixel = y * width + x
            if color == 3:
                var index = Int(component[0])
                if index >= len(palette) // 3:
                    raise Error("PNG data image palette index is out of range")
                for channel in range(3):
                    pixels[3 * pixel + channel] = (
                        Float32(palette[3 * index + channel]) / 255.0
                    )
                if has_alpha and index < len(transparency):
                    alpha[pixel] = Float32(transparency[index]) / 255.0
            else:
                for channel in range(3):
                    pixels[3 * pixel + channel] = Float32(
                        component[channel if color == 2 or color == 6 else 0]
                    ) / Float32(maximum)
                if color == 4 or color == 6:
                    alpha[pixel] = Float32(component[channels - 1]) / Float32(
                        maximum
                    )
                elif seen_transparency:
                    if color == 0:
                        alpha[pixel] = (
                            0.0 if Int(component[0])
                            == _u16(transparency, 0) else 1.0
                        )
                    else:
                        var transparent = True
                        for channel in range(3):
                            transparent &= Int(component[channel]) == _u16(
                                transparency, 2 * channel
                            )
                        alpha[pixel] = 0.0 if transparent else 1.0
    return ImageTexture(width, height, pixels^, alpha^)
