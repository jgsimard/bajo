from std.ffi import OwnedDLHandle
from std.testing import TestSuite, assert_equal, assert_raises

from bajo.parser.checksum import adler32, crc32, _crc32_portable
from bajo.parser.deflate import inflate_raw, inflate_zlib


def _compressed(source: List[UInt8], level: Int) raises -> List[UInt8]:
    var output = List[UInt8](length=len(source) * 2 + 128, fill=0)
    var size = List[UInt](length=1, fill=UInt(len(output)))
    var lib = OwnedDLHandle("libz.so.1")
    assert_equal(
        lib.call["compress2", Int32](
            output.unsafe_ptr(),
            size.unsafe_ptr(),
            source.unsafe_ptr(),
            UInt(len(source)),
            Int32(level),
        ),
        Int32(0),
    )
    output.resize(unsafe_uninit_length=Int(size[0]))
    return output^


def test_checksum_vectors_and_incremental_updates() raises:
    var bytes = StringSpan("123456789").as_bytes()
    assert_equal(crc32(bytes), UInt32(0xCBF43926))
    assert_equal(adler32(bytes), UInt32(0x091E01DE))
    assert_equal(crc32(bytes[:0]), UInt32(0))
    assert_equal(adler32(bytes[:0]), UInt32(1))
    assert_equal(crc32(bytes[4:], crc32(bytes[:4])), crc32(bytes))
    assert_equal(adler32(bytes[4:], adler32(bytes[:4])), adler32(bytes))


def test_checksum_simd_and_block_boundaries_match_zlib() raises:
    var bytes = List[UInt8](length=12000, fill=0)
    for i in range(len(bytes)):
        bytes[i] = UInt8((i * 73 + i // 9) & 255)
    var lib = OwnedDLHandle("libz.so.1")
    for offset in range(8):
        for count in [
            0,
            1,
            7,
            8,
            15,
            16,
            31,
            32,
            33,
            255,
            5551,
            5552,
            5553,
            11105,
        ]:
            var span = ImmSpan(bytes)[offset : offset + count]
            for seed in [UInt32(0), 1, 0x12345678, 0xFFFFFFFF]:
                assert_equal(
                    adler32(span, seed),
                    UInt32(
                        lib.call["adler32", UInt](
                            UInt(seed),
                            span.unsafe_ptr(),
                            UInt32(count),
                        )
                    ),
                )
                assert_equal(
                    crc32(span, seed),
                    UInt32(
                        lib.call["crc32", UInt](
                            UInt(seed),
                            span.unsafe_ptr(),
                            UInt32(count),
                        )
                    ),
                )


def test_adler_maximal_sums_do_not_overflow_at_nmax() raises:
    var maximal = List[UInt8](length=11105, fill=255)
    var lib = OwnedDLHandle("libz.so.1")
    for count in [
        32,
        64,
        128,
        5503,
        5504,
        5505,
        5535,
        5536,
        5537,
        5551,
        5552,
        5553,
        11008,
        11104,
        11105,
    ]:
        var span = ImmSpan(maximal)[:count]
        assert_equal(
            adler32(span, 0xFFF0FFF0),
            UInt32(
                lib.call["adler32", UInt](
                    UInt(0xFFF0FFF0),
                    span.unsafe_ptr(),
                    UInt32(count),
                )
            ),
        )


def test_adler_pair_batches_and_vector_tails_match_zlib() raises:
    var bytes = List[UInt8](length=11200, fill=0)
    for i in range(len(bytes)):
        bytes[i] = UInt8((i * 73 + i // 13) & 255)
    var lib = OwnedDLHandle("libz.so.1")
    var counts = List[Int]()
    for count in range(258):
        counts.append(count)
    for count in [5503, 5504, 5505, 5535, 5536, 5537, 11007, 11008, 11009]:
        counts.append(count)
    for offset in [0, 1, 7, 15, 31, 32, 63]:
        for count in counts:
            var span = ImmSpan(bytes)[offset : offset + count]
            for seed in [UInt32(0), 1, 0xFFFFFFFF]:
                assert_equal(
                    adler32(span, seed),
                    UInt32(
                        lib.call["adler32", UInt](
                            UInt(seed), span.unsafe_ptr(), UInt32(count)
                        )
                    ),
                )
                var split = count // 3
                assert_equal(
                    adler32(span[split:], adler32(span[:split], seed)),
                    adler32(span, seed),
                )


def test_crc_vector_folding_tails_and_seeded_updates() raises:
    var bytes = List[UInt8](length=1024, fill=0)
    for i in range(len(bytes)):
        bytes[i] = UInt8((i * 73 + i // 13) & 255)
    for offset in [0, 1, 3, 7, 15, 16, 17, 31, 32, 33, 47, 63]:
        for count in range(258):
            var span = ImmSpan(bytes)[offset : offset + count]
            for seed in [UInt32(0), 0x5A3C9571]:
                assert_equal(crc32(span, seed), _crc32_portable(span, seed))


def test_crc_alignment_prologue_matches_zlib() raises:
    var bytes = List[UInt8](length=131200, fill=0)
    for i in range(len(bytes)):
        bytes[i] = UInt8((i * 73 + i // 13) & 255)
    var lib = OwnedDLHandle("libz.so.1")
    for offset in range(64):
        for count in [65535, 65536, 65537, 65551, 131071]:
            var span = ImmSpan(bytes)[offset : offset + count]
            for seed in [UInt32(0), 0x5A3C9571]:
                assert_equal(
                    crc32(span, seed),
                    UInt32(
                        lib.call["crc32", UInt](
                            UInt(seed),
                            span.unsafe_ptr(),
                            UInt32(count),
                        )
                    ),
                )


def test_stored_fixed_dynamic_blocks_and_history_wrap() raises:
    for count in [0, 1, 2, 31, 258, 32768, 65536, 131073]:
        var source = List[UInt8](length=count, fill=0)
        for i in range(count):
            source[i] = UInt8((i * 31 + i // 127) & 255)
        for level in [0, 1, 6, 9]:
            var compressed = _compressed(source, level)
            var output = List[UInt8](length=count + 17, fill=0xA5)
            assert_equal(inflate_zlib(compressed, Span(output)[:count]), count)
            for i in range(count):
                assert_equal(output[i], source[i])
            for i in range(count, len(output)):
                assert_equal(output[i], UInt8(0xA5))
            var raw = inflate_raw(ImmSpan(compressed)[2:], Span(output)[:count])
            assert_equal(raw[0], len(compressed) - 6)
            assert_equal(raw[1], count)


def test_match_overlap_distances() raises:
    for distance in range(1, 34):
        var source = List[UInt8](length=2048, fill=0)
        for i in range(len(source)):
            source[i] = UInt8((i % distance) * 7 & 255)
        var compressed = _compressed(source, 9)
        var output = List[UInt8](length=len(source), fill=0)
        assert_equal(inflate_zlib(compressed, output), len(source))
        for i in range(len(source)):
            assert_equal(output[i], source[i])


def test_truncation_checksums_and_output_capacity_are_checked() raises:
    var source = List[UInt8](length=3072, fill=42)
    var compressed = _compressed(source, 6)
    var output = List[UInt8](length=len(source) + 1, fill=0xA5)
    for size in range(len(compressed)):
        with assert_raises():
            _ = inflate_zlib(ImmSpan(compressed)[:size], output)
    for index in [0, 1, len(compressed) - 1]:
        var corrupt = compressed.copy()
        corrupt[index] ^= UInt8(128)
        with assert_raises():
            _ = inflate_zlib(corrupt, output)
    for size in [0, 1, len(source) - 1]:
        with assert_raises():
            _ = inflate_zlib(compressed, Span(output)[:size])
    assert_equal(output[len(source)], UInt8(0xA5))


def test_raw_rejects_reserved_blocks_and_invalid_stored_lengths() raises:
    var output = List[UInt8](length=16, fill=0xA5)
    with assert_raises():
        _ = inflate_raw([UInt8(7)], output)
    with assert_raises():
        _ = inflate_raw([UInt8(1), 1, 0, 0, 0, 42], output)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
