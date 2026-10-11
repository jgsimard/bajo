"""OBJ decimal fast path with the shared parser's exponent and rounding behavior."""

from bajo.parser.number import F32ParseResult, parse_f32_at
from std.bit import count_trailing_zeros
from .primitives import _is_digit
from .constants import MINUS, PLUS, ZERO, DOT, CHAR_e, CHAR_E


@always_inline
def parse_obj_f32_at(
    bytes: ImmSpan[UInt8, _], pos: Int
) raises -> F32ParseResult:
    var end = len(bytes)
    if pos >= end:
        return F32ParseResult(0.0, pos)
    var p = pos
    var sign: Float64 = 1.0
    if bytes.unsafe_get(p) == MINUS:
        sign = -1.0
        p += 1
    elif bytes.unsafe_get(p) == PLUS:
        p += 1

    # Integer arithmetic is exact through 15 digits. Convert once rather than
    # performing a floating-point multiply/add for every digit.
    var whole: UInt64 = 0
    var whole_digits = 0
    while p < end and _is_digit(bytes.unsafe_get(p)):
        if whole_digits == 15:
            return parse_f32_at(bytes, pos)
        whole = whole * 10 + UInt64(bytes.unsafe_get(p) - ZERO)
        whole_digits += 1
        p += 1
    var number = Float64(whole)

    if p < end and bytes.unsafe_get(p) == DOT:
        p += 1
        var fraction: UInt64 = 0
        var digits = 0
        if end - p >= 8:
            # Find the first non-digit in an eight-byte block. Any carry or
            # borrow only affects bytes after that first non-digit.
            var word = UInt64(0)
            comptime for lane in range(8):
                word |= UInt64(bytes.unsafe_get(p + lane)) << UInt64(8 * lane)
            var lower = word - UInt64(0x3030303030303030)
            var upper = word + UInt64(0x4646464646464646)
            var nondigits = (lower | upper) & UInt64(0x8080808080808080)
            digits = Int(count_trailing_zeros(nondigits)) // 8
            if digits > 0:
                if digits < 8:
                    lower &= (UInt64(1) << UInt64(8 * digits)) - 1
                    # Zero-pad the beginning, retaining the decimal value.
                    lower <<= UInt64(8 * (8 - digits))
                var pairs = (lower * 10 + (lower >> 8)) & UInt64(
                    0x00FF00FF00FF00FF
                )
                var groups = (pairs * 100 + (pairs >> 16)) & UInt64(
                    0x0000FFFF0000FFFF
                )
                fraction = (groups * 10000 + (groups >> 32)) & UInt64(
                    0xFFFFFFFF
                )
                p += digits
        while digits < 15 and p < end and _is_digit(bytes.unsafe_get(p)):
            fraction = fraction * 10 + UInt64(bytes.unsafe_get(p) - ZERO)
            digits += 1
            p += 1
        comptime powers = SIMD[.float64, 16](
            1.0,
            10.0,
            100.0,
            1000.0,
            10000.0,
            100000.0,
            1000000.0,
            10000000.0,
            100000000.0,
            1000000000.0,
            10000000000.0,
            100000000000.0,
            1000000000000.0,
            10000000000000.0,
            100000000000000.0,
            1000000000000000.0,
        )
        var value = Float64(fraction)
        var divisor = powers[digits]
        # Long decimals continue with the original operations, preserving
        # float64 accumulation order once an integer mantissa is no longer exact.
        while p < end and _is_digit(bytes.unsafe_get(p)):
            value = value * 10.0 + Float64(Int(bytes.unsafe_get(p) - ZERO))
            divisor *= 10.0
            p += 1
        number += value / divisor

    if p < end and (
        bytes.unsafe_get(p) == CHAR_e or bytes.unsafe_get(p) == CHAR_E
    ):
        return parse_f32_at(bytes, pos)
    return F32ParseResult(Float32(sign * number), p)
