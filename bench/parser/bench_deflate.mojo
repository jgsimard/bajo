"""Measure zlib decoding and checksums separately from PNG filters and I/O.

Usage: mojo -I . bench/parser/bench_deflate.mojo stream.zlib decoded_bytes
Append `compare` to also measure libdeflate (optional benchmark-only library).
Output allocation is outside the timings for both implementations.
"""

from std.benchmark import Unit, keep, run
from std.ffi import OwnedDLHandle
from std.sys.arg import argv
from std.testing import assert_equal

from bajo.parser.checksum import adler32, crc32
from bajo.parser.deflate import inflate_zlib
from bajo.parser.mmap import MMap


def main() raises:
    var args = argv()
    if len(args) != 3 and len(args) != 4:
        raise Error("usage: bench_deflate stream.zlib decoded_bytes [compare]")
    var size = 0
    for byte in StringSpan(args[2]).as_bytes():
        if byte < 48 or byte > 57 or size > (Int.MAX - Int(byte - 48)) // 10:
            raise Error("invalid output size")
        size = size * 10 + Int(byte - 48)
    var mapped = MMap(String(args[1]))
    var bytes = mapped.as_bytes_span()
    var raw = List[UInt8](length=size, fill=0)
    assert_equal(inflate_zlib(bytes, raw), size)

    def decode() raises {bytes, mut raw}:
        keep(inflate_zlib(bytes, raw))
        keep(raw.unsafe_ptr())

    def crc() {bytes}:
        keep(crc32(bytes))

    def adler() {raw}:
        keep(adler32(raw))

    print("inflate_mojo_us", run(decode, max_iters=10).mean(Unit.us))
    print("crc32_mojo_us", run(crc, max_iters=10).mean(Unit.us))
    print("adler32_mojo_us", run(adler, max_iters=10).mean(Unit.us))

    if len(args) == 4:
        if args[3] != "compare":
            raise Error("unknown reference mode")
        var lib = OwnedDLHandle("libdeflate.so.0")
        var decoder = lib.call["libdeflate_alloc_decompressor", UInt]()
        if decoder == 0:
            raise Error("libdeflate could not allocate decompressor")
        var reference = List[UInt8](length=size, fill=0)

        def decode_reference() raises {imm, mut reference}:
            var status = lib.call["libdeflate_zlib_decompress", Int32](
                decoder,
                bytes.unsafe_ptr(),
                UInt(len(bytes)),
                reference.unsafe_ptr(),
                UInt(size),
                Int(0),
            )
            assert_equal(status, Int32(0))
            keep(reference.unsafe_ptr())

        def crc_reference() {bytes, lib}:
            keep(
                lib.call["libdeflate_crc32", UInt32](
                    UInt32(0), bytes.unsafe_ptr(), UInt(len(bytes))
                )
            )

        def adler_reference() {raw, lib}:
            keep(
                lib.call["libdeflate_adler32", UInt32](
                    UInt32(1), raw.unsafe_ptr(), UInt(len(raw))
                )
            )

        decode_reference()
        for i in range(size):
            assert_equal(raw[i], reference[i])
        assert_equal(
            crc32(bytes),
            lib.call["libdeflate_crc32", UInt32](
                UInt32(0), bytes.unsafe_ptr(), UInt(len(bytes))
            ),
        )
        assert_equal(
            adler32(raw),
            lib.call["libdeflate_adler32", UInt32](
                UInt32(1), raw.unsafe_ptr(), UInt(len(raw))
            ),
        )
        print("inflate_c_us", run(decode_reference, max_iters=10).mean(Unit.us))
        print("crc32_c_us", run(crc_reference, max_iters=10).mean(Unit.us))
        print("adler32_c_us", run(adler_reference, max_iters=10).mean(Unit.us))
        lib.call["libdeflate_free_decompressor"](decoder)
