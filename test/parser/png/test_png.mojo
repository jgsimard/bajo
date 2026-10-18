from std.memory import bitcast
from std.ffi import OwnedDLHandle
from std.testing import TestSuite, assert_equal, assert_raises

from bajo.parser.png import (
    _linearize_srgb_u8,
    _srgb_to_linear,
    _parse_png_libpng,
    _try_parse_png_simd,
    parse_png,
)


def _append_u32(mut bytes: List[UInt8], value: UInt32):
    for shift in range(24, -1, -8):
        bytes.append(UInt8((value >> UInt32(shift)) & 255))


def _chunk(mut bytes: List[UInt8], tag: String, payload: List[UInt8]) raises:
    _append_u32(bytes, UInt32(len(payload)))
    var begin = len(bytes)
    for byte in StringSpan(tag).as_bytes():
        bytes.append(byte)
    for byte in payload:
        bytes.append(byte)
    var lib = OwnedDLHandle("libz.so.1")
    var crc = lib.call["crc32", UInt](
        UInt(0),
        bytes.unsafe_ptr().unsafe_offset(begin),
        UInt32(4 + len(payload)),
    )
    _append_u32(bytes, UInt32(crc))


def _encode(
    raw: List[UInt8],
    width: Int,
    height: Int,
    color: UInt8 = 2,
    depth: UInt8 = 8,
    interlace: UInt8 = 0,
    split: Bool = False,
    metadata: String = "",
) raises -> List[UInt8]:
    var bytes: List[UInt8] = [137, 80, 78, 71, 13, 10, 26, 10]
    var header = List[UInt8]()
    _append_u32(header, UInt32(width))
    _append_u32(header, UInt32(height))
    header.extend([depth, color, UInt8(0), UInt8(0), interlace])
    _chunk(bytes, "IHDR", header)
    if color == 3:
        _chunk(bytes, "PLTE", [UInt8(255), 0, 0, 0, 255, 0])
    if metadata == "gAMA":
        var gamma = List[UInt8]()
        _append_u32(gamma, UInt32(100000))
        _chunk(bytes, "gAMA", gamma)
    elif metadata == "tRNS":
        _chunk(bytes, "tRNS", [UInt8(0), 1, 0, 2, 0, 3])
    elif metadata == "sRGB":
        _chunk(bytes, "sRGB", [UInt8(0)])
    var compressed = List[UInt8](length=len(raw) * 2 + 128, fill=0)
    var size = List[UInt](length=1, fill=UInt(len(compressed)))
    var lib = OwnedDLHandle("libz.so.1")
    assert_equal(
        lib.call["compress2", Int32](
            compressed.unsafe_ptr(),
            size.unsafe_ptr(),
            raw.unsafe_ptr(),
            UInt(len(raw)),
            Int32(6),
        ),
        Int32(0),
    )
    compressed.resize(unsafe_uninit_length=Int(size[0]))
    if split:
        _chunk(bytes, "IDAT", [])
        var first = List[UInt8]()
        var second = List[UInt8]()
        for i in range(len(compressed)):
            if i < len(compressed) // 2:
                first.append(compressed[i])
            else:
                second.append(compressed[i])
        _chunk(bytes, "IDAT", first)
        _chunk(bytes, "IDAT", [])
        _chunk(bytes, "IDAT", second)
    else:
        _chunk(bytes, "IDAT", compressed)
    _chunk(bytes, "IEND", [])
    return bytes^


def _sample(x: Int, y: Int) -> Int:
    return (x * 73 + y * 151 + (x * y) % 17) & 255


def _rgb_rows(width: Int, height: Int, filter: Int) -> List[UInt8]:
    var raw = List[UInt8]()
    for y in range(height):
        var kind = y % 5 if filter == -1 else filter
        raw.append(UInt8(kind))
        for x in range(width * 3):
            var a = _sample(x - 3, y) if x >= 3 else 0
            var b = _sample(x, y - 1) if y > 0 else 0
            var c = _sample(x - 3, y - 1) if x >= 3 and y > 0 else 0
            var predictor = 0
            if kind == 1:
                predictor = a
            elif kind == 2:
                predictor = b
            elif kind == 3:
                predictor = (a + b) // 2
            elif kind == 4:
                var p = a + b - c
                var pa = abs(p - a)
                var pb = abs(p - b)
                var pc = abs(p - c)
                predictor = a if pa <= pb and pa <= pc else b if pb <= pc else c
            raw.append(UInt8((_sample(x, y) - predictor) & 255))
    return raw^


def _assert_matches_reference(bytes: List[UInt8]) raises:
    var actual = parse_png(bytes)
    var expected = _parse_png_libpng(ImmSpan(bytes))
    assert_equal(actual.width, expected.width)
    assert_equal(actual.height, expected.height)
    assert_equal(len(actual.pixels), len(expected.pixels))
    for i in range(len(actual.pixels)):
        assert_equal(
            bitcast[.uint32](actual.pixels[i]),
            bitcast[.uint32](expected.pixels[i]),
        )


def test_all_filters_and_simd_tails_match_original_pixels_and_libpng() raises:
    for width in [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        15,
        16,
        17,
        21,
        22,
        31,
        32,
        33,
        63,
        64,
        65,
    ]:
        for filter in range(-1, 5):
            var bytes = _encode(_rgb_rows(width, 13, filter), width, 13)
            var fast = _try_parse_png_simd(bytes)
            assert_equal(Bool(fast), True)
            var image = fast.take()
            for y in range(13):
                for x in range(width * 3):
                    var expected = _srgb_to_linear(
                        Float32(_sample(x, y)) / 255.0
                    )
                    assert_equal(
                        bitcast[.uint32](image.pixels[y * width * 3 + x]),
                        bitcast[.uint32](expected),
                    )
            _assert_matches_reference(bytes)


def test_multiple_and_empty_idat_chunks_and_srgb() raises:
    var bytes = _encode(
        _rgb_rows(33, 11, -1), 33, 11, split=True, metadata="sRGB"
    )
    assert_equal(Bool(_try_parse_png_simd(bytes)), True)
    _assert_matches_reference(bytes)


def test_fallback_preserves_other_formats_and_color_transforms() raises:
    var gray = _encode([UInt8(0), 20, 200], 2, 1, color=0)
    var gray_bit = _encode([UInt8(0), 64], 2, 1, color=0, depth=1)
    var rgba = _encode(
        [UInt8(0), 20, 80, 200, 0, 200, 80, 20, 128, 40, 160, 240, 255],
        3,
        1,
        color=6,
    )
    var palette = _encode([UInt8(0), 0, 1], 2, 1, color=3)
    var palette_bit = _encode([UInt8(0), 64], 2, 1, color=3, depth=1)
    var deep = _encode([UInt8(0), 255, 255, 128, 128, 0, 0], 1, 1, depth=16)
    var interlaced = _encode([UInt8(0), 20, 80, 200], 1, 1, interlace=1)
    var gamma = _encode(_rgb_rows(2, 1, 0), 2, 1, metadata="gAMA")
    var transparent = _encode(
        [UInt8(0), 1, 2, 3, 4, 5, 6], 2, 1, metadata="tRNS"
    )
    for bytes in [
        gray^,
        gray_bit^,
        rgba^,
        palette^,
        palette_bit^,
        deep^,
        interlaced^,
        gamma^,
        transparent^,
    ]:
        assert_equal(Bool(_try_parse_png_simd(bytes)), False)
        _assert_matches_reference(bytes)


def test_malformed_png_is_rejected() raises:
    var good = _encode(_rgb_rows(5, 3, -1), 5, 3)
    for index in [29, 41, len(good) - 1]:
        var broken = good.copy()
        broken[index] ^= UInt8(1)
        with assert_raises():
            _ = parse_png(broken)
    for count in [0, 1, 7, 32, len(good) - 1, len(good) - 12]:
        with assert_raises():
            _ = parse_png(ImmSpan(good)[:count])
    var raw = _rgb_rows(5, 3, -1)
    raw[0] = UInt8(5)
    with assert_raises():
        _ = parse_png(_encode(raw, 5, 3))
    for size in [len(raw) - 1, len(raw) + 1]:
        var wrong_size = raw.copy()
        wrong_size[0] = UInt8(0)
        wrong_size.resize(size, fill=UInt8(0))
        with assert_raises():
            _ = parse_png(_encode(wrong_size, 5, 3))


def test_transparent_pixels_use_explicit_black_background() raises:
    var bytes = _encode(
        [UInt8(0), 255, 128, 64, 0, 50, 100, 200, 255], 2, 1, color=6
    )
    for _ in range(3):
        var image = parse_png(bytes)
        for i in range(3):
            assert_equal(image.pixels[i], Float32(0))
        for i in range(3):
            var channel = [50, 100, 200][i]
            assert_equal(
                image.pixels[3 + i], _srgb_to_linear(Float32(channel) / 255.0)
            )


def test_invalid_chunk_order_lengths_and_compressed_stream() raises:
    var good = _encode(_rgb_rows(2, 2, 0), 2, 2)
    # Reuse only the valid signature and header, then construct invalid chunks
    # with correct CRCs so these tests reach the structural/inflate checks.
    for kind in range(7):
        var bytes = List[UInt8]()
        for i in range(33):
            bytes.append(good[i])
        if kind == 0:
            var header = List[UInt8]()
            for i in range(16, 29):
                header.append(good[i])
            _chunk(bytes, "IHDR", header)
        elif kind == 1:
            _chunk(bytes, "IEND", [])  # Missing IDAT.
        elif kind == 2:
            _chunk(bytes, "IDAT", [UInt8(1), 2, 3])  # Not a zlib stream.
        elif kind == 3:
            _chunk(bytes, "IDAT", [])
            _chunk(bytes, "tEXt", [UInt8(120), 0, 121])
            _chunk(bytes, "IDAT", [])  # Nonconsecutive IDAT.
        elif kind == 4:
            _append_u32(bytes, UInt32(0x80000000))  # Invalid chunk length.
            bytes.extend([UInt8(73), 68, 65, 84, 0, 0, 0, 0])
        elif kind == 5:
            _append_u32(bytes, UInt32(100))  # Truncated chunk payload.
            bytes.extend([UInt8(73), 68, 65, 84, 0, 0, 0, 0])
        else:
            _chunk(bytes, "ABCD", [])  # Unknown critical chunk.
        _chunk(bytes, "IEND", [])
        with assert_raises():
            _ = parse_png(bytes)
    for width in [0, 0x80000000]:
        with assert_raises():
            _ = parse_png(_encode([UInt8(0)], width, 1))


def _assert_exact(
    encoded: List[UInt8], actual: List[Float32], index: Int
) raises:
    var expected = _srgb_to_linear(Float32(encoded[index]) / 255.0)
    assert_equal(
        bitcast[.uint32](actual[index]),
        bitcast[.uint32](expected),
    )


def test_simd_conversion_is_bit_exact() raises:
    # Include every possible channel and a non-SIMD-aligned tail.
    var encoded = List[UInt8](length=259, fill=0)
    for index in range(len(encoded)):
        encoded[index] = UInt8(index % 256)

    var actual = _linearize_srgb_u8(encoded)
    assert_equal(len(actual), len(encoded))
    for index in range(len(encoded)):
        _assert_exact(encoded, actual, index)


def test_threaded_simd_conversion_is_bit_exact() raises:
    comptime channels_per_task = 256
    var encoded = List[UInt8](length=channels_per_task + 17, fill=0)
    for index in range(len(encoded)):
        encoded[index] = UInt8(index % 256)

    var actual = _linearize_srgb_u8[channels_per_task](encoded)
    assert_equal(len(actual), len(encoded))
    for index in range(256):
        _assert_exact(encoded, actual, index)
    for index in range(channels_per_task - 32, channels_per_task + 17):
        _assert_exact(encoded, actual, index)


def test_conversion_handles_empty_and_subvector_inputs() raises:
    for count in range(0, 33):
        var encoded = List[UInt8](length=count, fill=UInt8(127))
        var actual = _linearize_srgb_u8(encoded)
        assert_equal(len(actual), count)
        for index in range(count):
            _assert_exact(encoded, actual, index)


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
