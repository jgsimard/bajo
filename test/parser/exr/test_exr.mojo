from std.memory import bitcast
from std.ffi import OwnedDLHandle
from std.sys import simd_width_of
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_raises,
)

from bajo.parser.exr import (
    parse_exr,
    _parse_exr_bulk,
    _parse_exr_serial,
    _parse_header,
    _undo_zip_prediction,
    _undo_zip_interleave,
    _undo_zip_interleave_into,
    _decode_channel,
    _half_to_float,
)


def _append_u32(mut bytes: List[UInt8], value: UInt32):
    bytes.append(UInt8(value & UInt32(255)))
    bytes.append(UInt8((value >> UInt32(8)) & UInt32(255)))
    bytes.append(UInt8((value >> UInt32(16)) & UInt32(255)))
    bytes.append(UInt8(value >> UInt32(24)))


def _append_u64(mut bytes: List[UInt8], value: UInt64):
    for shift in range(8):
        bytes.append(UInt8(value >> UInt64(8 * shift)))


def _append_f32(mut bytes: List[UInt8], value: Float32):
    _append_u32(bytes, bitcast[.uint32](value))


def _append_string(mut bytes: List[UInt8], value: String):
    var span = StringSpan(value).as_bytes()
    for byte in span:
        bytes.append(byte)
    bytes.append(UInt8(0))


def _append_attribute(
    mut bytes: List[UInt8],
    name: String,
    attribute_type: String,
    payload: List[UInt8],
):
    _append_string(bytes, name)
    _append_string(bytes, attribute_type)
    _append_u32(bytes, UInt32(len(payload)))
    for byte in payload:
        bytes.append(byte)


def _channel_list() -> List[UInt8]:
    var channels = List[UInt8]()
    for name in ["B", "G", "R"]:
        _append_string(channels, name)
        _append_u32(channels, UInt32(2))  # FLOAT
        channels.append(UInt8(0))  # pLinear
        channels.extend([UInt8(0), UInt8(0), UInt8(0)])
        _append_u32(channels, UInt32(1))
        _append_u32(channels, UInt32(1))
    channels.append(UInt8(0))
    return channels^


def _box2i(max_x: UInt32, max_y: UInt32) -> List[UInt8]:
    var payload = List[UInt8]()
    _append_u32(payload, UInt32(0))
    _append_u32(payload, UInt32(0))
    _append_u32(payload, max_x)
    _append_u32(payload, max_y)
    return payload^


def _tiny_uncompressed_exr() -> List[UInt8]:
    var bytes = List[UInt8]()
    _append_u32(bytes, UInt32(20000630))
    _append_u32(bytes, UInt32(2))
    _append_attribute(bytes, "channels", "chlist", _channel_list())
    _append_attribute(bytes, "compression", "compression", [UInt8(0)])
    _append_attribute(bytes, "dataWindow", "box2i", _box2i(1, 0))
    bytes.append(UInt8(0))  # End of header.

    var chunk_offset = UInt64(len(bytes) + 8)
    _append_u64(bytes, chunk_offset)
    _append_u32(bytes, UInt32(0))  # Scanline y.
    _append_u32(bytes, UInt32(24))
    # OpenEXR scanline order is B, G, R for this channel list.
    _append_f32(bytes, 0.3)
    _append_f32(bytes, 0.6)
    _append_f32(bytes, 0.2)
    _append_f32(bytes, 0.5)
    _append_f32(bytes, 0.1)
    _append_f32(bytes, 0.4)
    return bytes^


def test_decodes_float_rgb_scanline() raises:
    var image = parse_exr(_tiny_uncompressed_exr())
    assert_equal(image.width, 2)
    assert_equal(image.height, 1)
    assert_equal(len(image.pixels), 6)
    assert_almost_equal(image.pixels[0], 0.1)
    assert_almost_equal(image.pixels[1], 0.2)
    assert_almost_equal(image.pixels[2], 0.3)
    assert_almost_equal(image.pixels[3], 0.4)
    assert_almost_equal(image.pixels[4], 0.5)
    assert_almost_equal(image.pixels[5], 0.6)


def test_rejects_invalid_exr() raises:
    with assert_raises():
        _ = parse_exr([UInt8(1), UInt8(2), UInt8(3), UInt8(4)])


def _zip_payload(source: List[UInt8]) raises -> List[UInt8]:
    var shuffled = List[UInt8](length=len(source), fill=0)
    var half = (len(source) + 1) // 2
    for i in range(len(source)):
        shuffled[i // 2 if i % 2 == 0 else half + i // 2] = source[i]
    for i in range(len(shuffled) - 1, 0, -1):
        shuffled[i] = UInt8(
            (Int(shuffled[i]) - Int(shuffled[i - 1]) + 128) & 255
        )
    var compressed = List[UInt8](length=len(source) * 2 + 128, fill=0)
    var size = List[UInt](length=1, fill=UInt(len(compressed)))
    var lib = OwnedDLHandle("libz.so.1")
    assert_equal(
        lib.call["compress2", Int32](
            compressed.unsafe_ptr(),
            size.unsafe_ptr(),
            shuffled.unsafe_ptr(),
            UInt(len(shuffled)),
            Int32(6),
        ),
        Int32(0),
    )
    compressed.resize(unsafe_uninit_length=Int(size[0]))
    return compressed^


def _multi_chunk_fixture(
    compression: UInt8, mixed: Bool = False, ignored_name: String = "A"
) raises -> List[UInt8]:
    var bytes = List[UInt8]()
    _append_u32(bytes, UInt32(20000630))
    _append_u32(bytes, UInt32(2))
    var channels = List[UInt8]()
    for name in [ignored_name, "B", "G", "R"]:
        _append_string(channels, name)
        var kind = UInt32(2)
        if mixed and name == "G":
            kind = 0
        elif mixed and name == "R":
            kind = 1
        _append_u32(channels, kind)
        channels.extend([UInt8(0), UInt8(0), UInt8(0), UInt8(0)])
        _append_u32(channels, UInt32(1))
        _append_u32(channels, UInt32(1))
    channels.append(UInt8(0))
    _append_attribute(bytes, "channels", "chlist", channels)
    _append_attribute(bytes, "compression", "compression", [compression])
    var window = List[UInt8]()
    for value in [-2, -3, 28, 33]:
        _append_u32(window, bitcast[.uint32](Int32(value)))
    _append_attribute(bytes, "dataWindow", "box2i", window)
    bytes.append(UInt8(0))
    var lines_per_chunk = 16 if compression == 3 else 1
    var chunks = List[List[UInt8]]()
    for row in range(0, 37, lines_per_chunk):
        var data = List[UInt8]()
        for y in range(row, min(row + lines_per_chunk, 37)):
            for channel in range(4):
                for x in range(31):
                    if channel == 0:
                        _append_u32(
                            data, UInt32(0x7FC00000)
                        )  # Ignored auxiliary-channel NaN.
                    elif mixed and channel == 2:
                        _append_u32(data, UInt32(y * 31 + x))
                    elif mixed and channel == 3:
                        var bits = UInt16(0x8000) if x % 2 == 0 else UInt16(
                            0x3C00
                        )
                        data.append(UInt8(bits))
                        data.append(UInt8(bits >> UInt16(8)))
                    else:
                        _append_f32(
                            data, Float32(channel * 100 + y * 31 + x) / 7.0
                        )
        var payload = data.copy() if compression == 0 else _zip_payload(data)
        var chunk = List[UInt8]()
        _append_u32(chunk, bitcast[.uint32](Int32(row - 3)))
        _append_u32(chunk, UInt32(len(payload)))
        chunk.extend(Span(payload))
        chunks.append(chunk^)
    var offsets = List[UInt64]()
    var offset = len(bytes) + len(chunks) * 8
    for chunk in chunks:
        offsets.append(UInt64(offset))
        offset += len(chunk)
    # Offset table order can differ from scanline or physical payload order.
    for i in range(len(offsets) - 1, -1, -1):
        _append_u64(bytes, offsets[i])
    for chunk in chunks:
        bytes.extend(Span(chunk))
    return bytes^


def test_parallel_chunks_match_serial_for_zip_zips_and_mixed_channels() raises:
    for compression in [UInt8(0), UInt8(2), UInt8(3)]:
        var bytes = _multi_chunk_fixture(compression, mixed=True)
        var expected = _parse_exr_serial(bytes)
        for workers in [1, 2, 8]:
            var actual = _parse_exr_bulk(bytes, workers)
            assert_equal(actual.width, expected.width)
            assert_equal(actual.height, expected.height)
            assert_equal(len(actual.pixels), len(expected.pixels))
            for i in range(len(actual.pixels)):
                assert_equal(
                    bitcast[.uint32](actual.pixels[i]),
                    bitcast[.uint32](expected.pixels[i]),
                )
        assert_equal(bitcast[.uint32](expected.pixels[0]), UInt32(0x80000000))


def test_bulk_rejects_truncated_and_nonfinite_rgb() raises:
    var bytes = _multi_chunk_fixture(UInt8(0))
    var bad = bytes.copy()
    _ = bad.pop()
    with assert_raises():
        _ = _parse_exr_bulk(bad, 8)
    # The final four bytes belong to R in the last scanline.
    var bits = UInt32(0x7F800000)
    for shift in range(4):
        bytes[len(bytes) - 4 + shift] = UInt8(bits >> UInt32(8 * shift))
    with assert_raises():
        _ = _parse_exr_bulk(bytes, 8)


def test_arbitrary_auxiliary_channel_names_are_ignored() raises:
    var expected = parse_exr(_multi_chunk_fixture(UInt8(0), mixed=True))
    for name in ["A.custom", "AO"]:
        var bytes = _multi_chunk_fixture(
            UInt8(0), mixed=True, ignored_name=name
        )
        var serial = parse_exr(bytes)
        var bulk = _parse_exr_bulk(bytes, 2)
        for i in range(len(expected.pixels)):
            assert_equal(
                bitcast[.uint32](serial.pixels[i]),
                bitcast[.uint32](expected.pixels[i]),
            )
            assert_equal(
                bitcast[.uint32](bulk.pixels[i]),
                bitcast[.uint32](expected.pixels[i]),
            )


def test_overlapping_chunks_keep_serial_fallback_behavior() raises:
    var bytes = _multi_chunk_fixture(UInt8(0))
    var header = _parse_header(bytes)
    for i in range(8):
        bytes[header.data_offset + i] = bytes[header.data_offset + 8 + i]
    bytes.resize(256 * 1024, fill=UInt8(0))
    with assert_raises():
        _ = _parse_exr_bulk(bytes, 8)
    var expected = _parse_exr_serial(bytes)
    var actual = parse_exr(bytes)
    for i in range(len(actual.pixels)):
        assert_equal(
            bitcast[.uint32](actual.pixels[i]),
            bitcast[.uint32](expected.pixels[i]),
        )


def test_compressed_chunk_failure_is_reported_after_workers_finish() raises:
    var bytes = _multi_chunk_fixture(UInt8(3))
    var header = _parse_header(bytes)
    var offset = UInt64(0)
    for i in range(8):
        offset |= UInt64(bytes[header.data_offset + i]) << UInt64(8 * i)
    bytes[Int(offset) + 8] = UInt8(0)  # Corrupt the zlib header in one chunk.
    bytes.resize(256 * 1024, fill=UInt8(0))
    with assert_raises():
        _ = _parse_exr_bulk(bytes, 8)
    with assert_raises():
        _ = parse_exr(bytes)


def test_zip_postprocessing_matches_byte_reference_at_vector_boundaries() raises:
    var lengths = List[Int]()
    for count in range(130):
        lengths.append(count)
    lengths.extend([4095, 4096, 4097, 65543])
    for count in lengths:
        for seed in [0, 1, 127, 128, 255]:
            var encoded = List[UInt8](
                length=count,
                fill_with=lambda (i: Int) -> UInt8: UInt8(
                    (197 * i + 29 * seed + 13 * (i >> 2)) & 255
                ),
            )
            var expected = encoded.copy()
            for i in range(1, count):
                expected[i] = UInt8(
                    (Int(expected[i - 1]) + Int(expected[i]) - 128) & 255
                )
            var actual = encoded.copy()
            _undo_zip_prediction(actual)
            var guarded = List[UInt8](length=count + 7, fill=UInt8(0xA5))
            var target = Span(guarded)[3 : 3 + count]
            for i in range(count):
                target[i] = encoded[i]
            _undo_zip_prediction(target)
            for i in range(count):
                assert_equal(actual[i], expected[i])
                assert_equal(target[i], expected[i])
            for i in [0, 1, 2, count + 3, count + 4, count + 5, count + 6]:
                assert_equal(guarded[i], UInt8(0xA5))
            var interleaved = _undo_zip_interleave(encoded)
            _undo_zip_interleave_into(encoded, target)
            var half = (count + 1) // 2
            for i in range(count):
                var value = encoded[i // 2 if i % 2 == 0 else half + i // 2]
                assert_equal(interleaved[i], value)
                assert_equal(target[i], value)
            for i in [0, 1, 2, count + 3, count + 4, count + 5, count + 6]:
                assert_equal(guarded[i], UInt8(0xA5))


def test_simd_half_conversion_matches_every_finite_encoding() raises:
    var encodings = List[UInt16](capacity=65536)
    for bits in range(65536):
        if (bits & 0x7C00) != 0x7C00:
            encodings.append(UInt16(bits))
    encodings.extend([UInt16(0), UInt16(0x8000), UInt16(0x0400)])
    var bytes = List[UInt8](length=2 * len(encodings) + 1, fill=UInt8(0xA5))
    for i in range(len(encodings)):
        bytes[2 * i + 1] = UInt8(encodings[i])
        bytes[2 * i + 2] = UInt8(encodings[i] >> UInt16(8))
    var output = List[Float32](length=3 * len(encodings) + 2, fill=-9.0)
    for lane in range(3):
        _decode_channel[1](
            Span(bytes)[1:],
            Span(output)[1 : len(output) - 1],
            lane,
            len(encodings),
        )
    for i in range(len(encodings)):
        var expected = bitcast[.uint32](_half_to_float(encodings[i]))
        for lane in range(3):
            assert_equal(bitcast[.uint32](output[3 * i + lane + 1]), expected)
    assert_equal(output[0], Float32(-9.0))
    assert_equal(output[len(output) - 1], Float32(-9.0))


def test_simd_uint_and_float_channels_preserve_bits_and_tails() raises:
    comptime vector_width = simd_width_of[Float32]()
    for width in range(2 * vector_width + 4):
        var integers = List[UInt8]()
        var floats = List[UInt8]()
        # A prefix deliberately misaligns the subsequent 32-bit loads.
        integers.append(UInt8(0xA5))
        floats.append(UInt8(0xA5))
        var expected_int = List[Float32]()
        var expected_float = List[UInt32]()
        for x in range(width):
            var value = UInt32.MAX - UInt32(7919 * x)
            if x % 3 == 0:
                value = UInt32(16777217 + x)
            _append_u32(integers, value)
            expected_int.append(Float32(value))
            var bits = (value & UInt32(0x807FFFFF)) | UInt32((x % 255) << 23)
            _append_u32(floats, bits)
            expected_float.append(bits)
        var output = List[Float32](length=3 * width + 2, fill=-9.0)
        var target = Span(output)[1 : len(output) - 1]
        for lane in range(3):
            _decode_channel[0](Span(integers)[1:], target, lane, width)
            for x in range(width):
                assert_equal(
                    bitcast[.uint32](target[3 * x + lane]),
                    bitcast[.uint32](expected_int[x]),
                )
            _decode_channel[2](Span(floats)[1:], target, lane, width)
            for x in range(width):
                assert_equal(
                    bitcast[.uint32](target[3 * x + lane]), expected_float[x]
                )
        assert_equal(output[0], Float32(-9.0))
        assert_equal(output[len(output) - 1], Float32(-9.0))


def test_simd_rejects_nonfinite_values_in_vectors_and_tails() raises:
    comptime vector_width = simd_width_of[Float32]()
    var width = vector_width + 3
    var output = List[Float32](length=3 * width, fill=0)
    for position in [0, vector_width - 1, vector_width, width - 1]:
        for bits in [
            UInt32(0x7F800000),
            UInt32(0xFF800000),
            UInt32(0x7FC00000),
            UInt32(0x7F800001),
        ]:
            var bytes = List[UInt8](length=4 * width, fill=0)
            for byte in range(4):
                bytes[4 * position + byte] = UInt8(bits >> UInt32(8 * byte))
            with assert_raises():
                _decode_channel[2](bytes, Span(output), 1, width)
        for bits in [
            UInt16(0x7C00),
            UInt16(0xFC00),
            UInt16(0x7E00),
            UInt16(0x7C01),
        ]:
            var bytes = List[UInt8](length=2 * width, fill=0)
            bytes[2 * position] = UInt8(bits)
            bytes[2 * position + 1] = UInt8(bits >> UInt16(8))
            with assert_raises():
                _decode_channel[1](bytes, Span(output), 2, width)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
