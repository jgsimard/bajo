"""Native Mojo Adler-32 and IEEE CRC-32 for zlib and PNG.

Adapted from libdeflate v1.25; see LICENSE.libdeflate in this directory.
"""

from std.memory import bitcast
from std.sys.info import CompilationTarget
from std.sys.intrinsics import llvm_intrinsic


@always_inline
def _adler_sum[
    width: Int
](values: SIMD[.uint8, width]) -> SIMD[.uint32, width // 4]:
    comptime if width == 64:
        return bitcast[.uint32, width // 4](
            llvm_intrinsic[
                "llvm.x86.avx512.psad.bw.512",
                SIMD[.uint64, width // 8],
                has_side_effect=False,
            ](values, SIMD[.uint8, width](0))
        )
    else:
        return bitcast[.uint32, width // 4](
            llvm_intrinsic[
                "llvm.x86.avx2.psad.bw",
                SIMD[.uint64, width // 8],
                has_side_effect=False,
            ](values, SIMD[.uint8, width](0))
        )


@always_inline
def _adler_dot[
    width: Int
](
    accumulator: SIMD[.uint32, width // 4],
    values: SIMD[.uint8, width],
    weights: SIMD[.uint8, width],
) -> SIMD[.uint32, width // 4]:
    comptime if width == 64 and CompilationTarget._has_feature["avx512vnni"]():
        return llvm_intrinsic[
            "llvm.x86.avx512.vpdpbusd.512",
            SIMD[.uint32, width // 4],
            has_side_effect=False,
        ](accumulator, values, weights)
    elif width == 64:
        var pairs = llvm_intrinsic[
            "llvm.x86.avx512.pmaddubs.w.512",
            SIMD[.int16, width // 2],
            has_side_effect=False,
        ](values, weights)
        return (
            accumulator
            + llvm_intrinsic[
                "llvm.x86.avx512.pmaddw.d.512",
                SIMD[.int32, width // 4],
                has_side_effect=False,
            ](pairs, SIMD[.int16, width // 2](1)).cast[.uint32]()
        )
    else:
        var pairs = llvm_intrinsic[
            "llvm.x86.avx2.pmadd.ub.sw",
            SIMD[.int16, width // 2],
            has_side_effect=False,
        ](values, weights)
        return (
            accumulator
            + llvm_intrinsic[
                "llvm.x86.avx2.pmadd.wd",
                SIMD[.int32, width // 4],
                has_side_effect=False,
            ](pairs, SIMD[.int16, width // 2](1)).cast[.uint32]()
        )


def _adler32_simd[
    width: Int
](bytes: ImmSpan[UInt8, _], initial: UInt32) -> UInt32:
    var a = (initial & UInt32(65535)) % UInt32(65521)
    var b = (initial >> 16) % UInt32(65521)
    var offset = 0
    var weights = SIMD[.uint8, width](0)
    comptime for lane in range(width):
        weights[lane] = UInt8(width - lane)
    # A whole number of pairs keeps scalar tails out of every NMAX block.
    # Both sums and the final weighted sum stay below 2**32 at 5552 bytes.
    comptime batch = 5552 // (2 * width) * (2 * width)
    while offset < len(bytes):
        var end = min(offset + batch, len(bytes))
        var vector_end = offset + (end - offset) // (2 * width) * (2 * width)
        var sum0 = SIMD[.uint32, width // 4](0)
        var sum1 = sum0
        var prefix0 = sum0
        var prefix1 = sum0
        var weighted0 = sum0
        var weighted1 = sum0
        for i in range(offset, vector_end, 2 * width):
            var v0 = (
                bytes.unsafe_ptr().unsafe_offset(i).unsafe_load[width=width]()
            )
            var v1 = (
                bytes.unsafe_ptr()
                .unsafe_offset(i + width)
                .unsafe_load[width=width]()
            )
            prefix0 += sum0
            prefix1 += sum1
            sum0 += _adler_sum(v0)
            sum1 += _adler_sum(v1)
            weighted0 = _adler_dot(weighted0, v0, weights)
            weighted1 = _adler_dot(weighted1, v1, weights)
        b += (
            UInt32(vector_end - offset) * a
            + (
                (prefix0 + prefix1) * UInt32(2 * width)
                + weighted0
                + weighted1
                + sum0 * UInt32(width)
            ).reduce_add()
        )
        a += (sum0 + sum1).reduce_add()
        for i in range(vector_end, end):
            a += UInt32(bytes[i])
            b += a
        a %= UInt32(65521)
        b %= UInt32(65521)
        offset = end
    return (b << 16) | a


def adler32(bytes: ImmSpan[UInt8, _], initial: UInt32 = 1) -> UInt32:
    comptime if CompilationTarget.has_avx2():
        comptime width = 64 if CompilationTarget._has_feature[
            "avx512bw"
        ]() else 32
        return _adler32_simd[width](bytes, initial)
    else:
        var a = (initial & UInt32(65535)) % UInt32(65521)
        var b = (initial >> 16) % UInt32(65521)
        var offset = 0
        while offset < len(bytes):
            var end = min(offset + 5536, len(bytes))
            var vector_end = offset + (end - offset) // 32 * 32
            var sums = SIMD[.uint32, 32](0)
            var weighted = sums
            var weights = SIMD[.uint32, 32](0)
            comptime for lane in range(32):
                weights[lane] = UInt32(32 - lane)
            for i in range(offset, vector_end, 32):
                var values = (
                    bytes.unsafe_ptr()
                    .unsafe_offset(i)
                    .unsafe_load[width=32]()
                    .cast[.uint32]()
                )
                weighted += sums * UInt32(32) + values * weights
                sums += values
            b += UInt32(vector_end - offset) * a + weighted.reduce_add()
            a += sums.reduce_add()
            for i in range(vector_end, end):
                a += UInt32(bytes[i])
                b += a
            a %= UInt32(65521)
            b %= UInt32(65521)
            offset = end
        return (b << 16) | a


def _crc32_tables() -> List[UInt32]:
    var tables = List[UInt32](length=2048, fill=0)
    for i in range(256):
        var value = UInt32(i)
        for _ in range(8):
            value = (value >> 1) ^ (
                UInt32(0xEDB88320) if value & 1 else UInt32(0)
            )
        tables[i] = value
    for table in range(1, 8):
        for i in range(256):
            var value = tables[(table - 1) * 256 + i]
            tables[table * 256 + i] = (value >> 8) ^ tables[Int(value & 255)]
    return tables^


def _crc32_portable(bytes: ImmSpan[UInt8, _], initial: UInt32 = 0) -> UInt32:
    var tables = _crc32_tables()
    var crc = ~initial
    var end = len(bytes) // 8 * 8
    for i in range(0, end, 8):
        var word = bytes.unsafe_ptr().unsafe_offset(i).unsafe_bitcast[
            UInt64
        ]().unsafe_load[alignment=1]() ^ UInt64(crc)
        var next = UInt32(0)
        comptime for lane in range(8):
            next ^= tables[
                (7 - lane) * 256 + Int((word >> UInt64(8 * lane)) & 255)
            ]
        crc = next
    for i in range(end, len(bytes)):
        crc = (crc >> 8) ^ tables[Int((crc ^ UInt32(bytes[i])) & 255)]
    return ~crc


@always_inline
def _clmul[
    select: Int, lanes: SIMDLength
](a: SIMD[.uint64, lanes], b: SIMD[.uint64, lanes]) -> SIMD[.uint64, lanes]:
    comptime if lanes == 8:
        return llvm_intrinsic[
            "llvm.x86.pclmulqdq.512",
            SIMD[.uint64, lanes],
            has_side_effect=False,
        ](a, b, Int8(select))
    elif lanes == 4:
        return llvm_intrinsic[
            "llvm.x86.pclmulqdq.256",
            SIMD[.uint64, lanes],
            has_side_effect=False,
        ](a, b, Int8(select))
    else:
        return llvm_intrinsic[
            "llvm.x86.pclmulqdq", SIMD[.uint64, lanes], has_side_effect=False
        ](a, b, Int8(select))


@always_inline
def _fold_crc[
    lanes: SIMDLength
](
    a: SIMD[.uint64, lanes],
    b: SIMD[.uint64, lanes],
    multipliers: SIMD[.uint64, lanes],
) -> SIMD[.uint64, lanes]:
    return b ^ _clmul[0](a, multipliers) ^ _clmul[0x11](a, multipliers)


def _crc32_pclmul(bytes: ImmSpan[UInt8, _], initial: UInt32) -> UInt32:
    # Reflected IEEE polynomial folding constants from libdeflate's
    # crc32_multipliers.h. Use four independent chains to hide CLMUL latency.
    comptime wide = CompilationTarget._has_feature[
        "vpclmulqdq"
    ]() and CompilationTarget.has_avx2()
    comptime lanes = 8 if wide and CompilationTarget.has_avx512f() else 4 if wide else 2
    comptime vector_bytes = lanes * 8
    var ptr = bytes.unsafe_ptr()
    var x = SIMD[.uint64, 2](UInt64(~initial), 0)
    var offset = 0
    comptime one = SIMD[.uint64, 2](0xAE689191, 0xCCAA009E)
    if len(bytes) >= 4 * vector_bytes:
        var four = SIMD[.uint64, lanes](0)
        var two = four
        var step = four
        comptime for lane in range(lanes // 2):
            four[2 * lane] = UInt64(
                0xCE3371CB if lanes == 8 else 0x33FFF533 if wide else 0x8F352D95
            )
            four[2 * lane + 1] = UInt64(
                0xE95C1271 if lanes == 8 else 0x910EEEC1 if wide else 0x1D9513D7
            )
            two[2 * lane] = UInt64(
                0x33FFF533 if lanes == 8 else 0x8F352D95 if wide else 0xF1DA05AA
            )
            two[2 * lane + 1] = UInt64(
                0x910EEEC1 if lanes == 8 else 0x1D9513D7 if wide else 0x81256527
            )
            step[2 * lane] = UInt64(
                0x8F352D95 if lanes == 8 else 0xF1DA05AA if wide else 0xAE689191
            )
            step[2 * lane + 1] = UInt64(
                0x1D9513D7 if lanes == 8 else 0x81256527 if wide else 0xCCAA009E
            )
        var seed = SIMD[.uint64, lanes](0)
        seed[0] = UInt64(~initial)
        var a = (
            ptr.unsafe_bitcast[UInt64]().unsafe_load[width=lanes, alignment=1]()
            ^ seed
        )
        var b = (
            ptr.unsafe_offset(vector_bytes)
            .unsafe_bitcast[UInt64]()
            .unsafe_load[width=lanes, alignment=1]()
        )
        var c = (
            ptr.unsafe_offset(2 * vector_bytes)
            .unsafe_bitcast[UInt64]()
            .unsafe_load[width=lanes, alignment=1]()
        )
        var d = (
            ptr.unsafe_offset(3 * vector_bytes)
            .unsafe_bitcast[UInt64]()
            .unsafe_load[width=lanes, alignment=1]()
        )
        offset = 4 * vector_bytes
        while offset <= len(bytes) - 4 * vector_bytes:
            a = _fold_crc(
                a,
                ptr.unsafe_offset(offset)
                .unsafe_bitcast[UInt64]()
                .unsafe_load[width=lanes, alignment=1](),
                four,
            )
            b = _fold_crc(
                b,
                ptr.unsafe_offset(offset + vector_bytes)
                .unsafe_bitcast[UInt64]()
                .unsafe_load[width=lanes, alignment=1](),
                four,
            )
            c = _fold_crc(
                c,
                ptr.unsafe_offset(offset + 2 * vector_bytes)
                .unsafe_bitcast[UInt64]()
                .unsafe_load[width=lanes, alignment=1](),
                four,
            )
            d = _fold_crc(
                d,
                ptr.unsafe_offset(offset + 3 * vector_bytes)
                .unsafe_bitcast[UInt64]()
                .unsafe_load[width=lanes, alignment=1](),
                four,
            )
            offset += 4 * vector_bytes
        a = _fold_crc(a, c, two)
        b = _fold_crc(b, d, two)
        a = _fold_crc(a, b, step)
        x = a.slice[2]()
        comptime if lanes == 8:
            comptime half = SIMD[.uint64, 4](
                0xF1DA05AA, 0x81256527, 0xF1DA05AA, 0x81256527
            )
            var y = _fold_crc(a.slice[4](), a.slice[4, offset=4](), half)
            x = _fold_crc(y.slice[2](), y.slice[2, offset=2](), one)
        elif wide:
            x = _fold_crc(x, a.slice[2, offset=2](), one)
    elif len(bytes) >= 16:
        x ^= ptr.unsafe_bitcast[UInt64]().unsafe_load[width=2, alignment=1]()
        offset = 16
    while offset <= len(bytes) - 16:
        x = _fold_crc(
            x,
            ptr.unsafe_offset(offset)
            .unsafe_bitcast[UInt64]()
            .unsafe_load[width=2, alignment=1](),
            one,
        )
        offset += 16
    var crc = ~initial
    if offset != 0:
        # Reduce the 128-bit message polynomial to its 32-bit CRC with Barrett
        # reduction, then finish the short tail without any out-of-span load.
        x = _clmul[0x10](x, one) ^ x.shuffle[1, 2](SIMD[.uint64, 2](0))
        comptime barrett = SIMD[.uint64, 2](0xB4E5B025F7011641, 0x1DB710641)
        var reduction = _clmul[0](x, barrett)
        reduction = _clmul[0x10](reduction, barrett)
        crc = bitcast[.uint32, 4](x ^ reduction)[2]
    for i in range(offset, len(bytes)):
        crc ^= UInt32(bytes[i])
        comptime for _ in range(8):
            crc = (crc >> 1) ^ (UInt32(0xEDB88320) if crc & 1 else UInt32(0))
    return ~crc


def crc32(bytes: ImmSpan[UInt8, _], initial: UInt32 = 0) -> UInt32:
    comptime if CompilationTarget._has_feature["pclmul"]():
        comptime alignment = 64 if CompilationTarget.has_avx512f() and CompilationTarget._has_feature[
            "vpclmulqdq"
        ]() else 32 if CompilationTarget.has_avx2() and CompilationTarget._has_feature[
            "vpclmulqdq"
        ]() else 16
        var prefix = (-Int(bytes.unsafe_ptr())) & (alignment - 1)
        if len(bytes) > 65536 and prefix != 0:
            return _crc32_pclmul(
                bytes[prefix:], _crc32_pclmul(bytes[:prefix], initial)
            )
        return _crc32_pclmul(bytes, initial)
    else:
        return _crc32_portable(bytes, initial)
