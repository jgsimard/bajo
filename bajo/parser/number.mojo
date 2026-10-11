from std.bit import count_trailing_zeros
from std.memory import bitcast, pack_bits


comptime _MINUS = UInt8(45)
comptime _PLUS = UInt8(43)
comptime _ZERO = UInt8(48)
comptime _NINE = UInt8(57)
comptime _DOT = UInt8(46)
comptime _LOWER_E = UInt8(101)
comptime _UPPER_E = UInt8(69)
comptime _MAX_DECIMAL_EXPONENT = 308


@always_inline
def _is_digit(value: UInt8) -> Bool:
    return value >= _ZERO and value <= _NINE


@fieldwise_init
struct F32ParseResult(TrivialRegisterPassable):
    var value: Float32
    var pos: Int


@always_inline
def parse_f32_at(bytes: ImmSpan[UInt8, _], pos: Int) raises -> F32ParseResult:
    debug_assert["safe", _use_compiler_assume=True](
        pos >= 0 and pos <= len(bytes),
        "float parse position is outside the input bytes",
    )
    var end = len(bytes)
    if pos >= end:
        return F32ParseResult(0.0, pos)

    var p = pos
    var sign: Float64 = 1.0

    if bytes.unsafe_get(p) == _MINUS:
        sign = -1.0
        p += 1
    elif bytes.unsafe_get(p) == _PLUS:
        p += 1

    var num: Float64 = 0.0
    while p < end:
        var b = bytes.unsafe_get(p)
        if _is_digit(b):
            num = num * 10.0 + Float64(Int(b - _ZERO))
            p += 1
        else:
            break

    if p < end and bytes.unsafe_get(p) == _DOT:
        p += 1
        var fraction: Float64 = 0.0
        var divisor: Float64 = 1.0

        while p < end:
            var b = bytes.unsafe_get(p)
            if _is_digit(b):
                fraction = fraction * 10.0 + Float64(Int(b - _ZERO))
                divisor *= 10.0
                p += 1
            else:
                break

        num += fraction / divisor

    if p < end:
        var b = bytes.unsafe_get(p)
        if b == _LOWER_E or b == _UPPER_E:
            p += 1
            var exponent_sign = 1

            if p < end and bytes.unsafe_get(p) == _MINUS:
                exponent_sign = -1
                p += 1
            elif p < end and bytes.unsafe_get(p) == _PLUS:
                p += 1

            var exponent = 0
            var has_exponent_digit = False
            while p < end:
                var exponent_byte = bytes.unsafe_get(p)
                if _is_digit(exponent_byte):
                    has_exponent_digit = True
                    var digit = Int(exponent_byte - _ZERO)
                    if exponent > (_MAX_DECIMAL_EXPONENT - digit) // 10:
                        raise String(
                            t"decimal exponent exceeds {_MAX_DECIMAL_EXPONENT}"
                        )

                    exponent = exponent * 10 + digit
                    p += 1
                else:
                    break

            if not has_exponent_digit:
                raise String("missing decimal exponent digits")

            if exponent > 0:
                var power: Float64 = 1.0
                var base: Float64 = 10.0
                var remaining = exponent

                while remaining > 0:
                    if remaining & 1:
                        power *= base
                    remaining //= 2
                    if remaining > 0:
                        base *= base

                if exponent_sign == 1:
                    num *= power
                else:
                    num /= power

    return F32ParseResult(Float32(sign * num), p)


@always_inline
def _decimal_prefix8(word: UInt64, digits: Int) -> UInt64:
    # The first 1..8 bytes are digits. Drop the suffix and zero-pad the start.
    var lower = word - UInt64(0x3030303030303030)
    if digits < 8:
        lower &= (UInt64(1) << UInt64(8 * digits)) - 1
        lower <<= UInt64(8 * (8 - digits))
    var pairs = (lower * 10 + (lower >> 8)) & UInt64(0x00FF00FF00FF00FF)
    var groups = (pairs * 100 + (pairs >> 16)) & UInt64(0x0000FFFF0000FFFF)
    return (groups * 10000 + (groups >> 32)) & UInt64(0xFFFFFFFF)


@always_inline
def parse_f32_fast_at(
    bytes: ImmSpan[UInt8, _], pos: Int
) raises -> F32ParseResult:
    var end = len(bytes)
    if pos >= end:
        return F32ParseResult(0.0, pos)
    var p = pos
    var sign: Float64 = 1.0
    if bytes.unsafe_get(p) == _MINUS:
        sign = -1.0
        p += 1
    elif bytes.unsafe_get(p) == _PLUS:
        p += 1

    # Integer arithmetic is exact through 15 digits. Convert once rather than
    # performing a floating-point multiply/add for every digit.
    var whole: UInt64 = 0
    var whole_digits = 0
    while p < end and _is_digit(bytes.unsafe_get(p)):
        if whole_digits == 15:
            return parse_f32_at(bytes, pos)
        whole = whole * 10 + UInt64(bytes.unsafe_get(p) - _ZERO)
        whole_digits += 1
        p += 1
    var number = Float64(whole)

    if p < end and bytes.unsafe_get(p) == _DOT:
        p += 1
        var fraction: UInt64 = 0
        var digits = 0
        if end - p >= 8:
            # Find the first non-digit in an eight-byte block. Any carry or
            # borrow only affects bytes after that first non-digit.
            var word = (
                bytes.unsafe_ptr()
                .unsafe_offset(p)
                .unsafe_bitcast[UInt64]()
                .unsafe_load[alignment=1]()
            )
            var lower = word - UInt64(0x3030303030303030)
            var upper = word + UInt64(0x4646464646464646)
            var nondigits = (lower | upper) & UInt64(0x8080808080808080)
            digits = Int(count_trailing_zeros(nondigits)) // 8
            if digits > 0:
                fraction = _decimal_prefix8(word, digits)
                p += digits
        if digits == 8 and end - p >= 8:
            # Long decimals commonly have 15..17 fraction digits. Classify
            # the next eight bytes together, then accumulate up to 15 digits
            # exactly before continuing the original float64 operations.
            var block = (
                bytes.unsafe_ptr()
                .unsafe_offset(p)
                .unsafe_load[width=8, alignment=1]()
            )
            var extra = min(
                7,
                Int(
                    count_trailing_zeros(
                        ~UInt32(pack_bits((block - _ZERO).le(UInt8(9))))
                    )
                ),
            )
            if extra > 0:
                comptime powers = SIMD[.uint64, 8](
                    1, 10, 100, 1000, 10000, 100000, 1000000, 10000000
                )
                fraction = fraction * powers[extra] + _decimal_prefix8(
                    bitcast[.uint64, 1](block), extra
                )
                digits += extra
                p += extra
        while digits < 15 and p < end and _is_digit(bytes.unsafe_get(p)):
            fraction = fraction * 10 + UInt64(bytes.unsafe_get(p) - _ZERO)
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
            value = value * 10.0 + Float64(Int(bytes.unsafe_get(p) - _ZERO))
            divisor *= 10.0
            p += 1
        number += value / divisor

    if p < end and (
        bytes.unsafe_get(p) == _LOWER_E or bytes.unsafe_get(p) == _UPPER_E
    ):
        return parse_f32_at(bytes, pos)
    return F32ParseResult(Float32(sign * number), p)


def parse_int_f32(
    text: ImmStringSpan, context: ImmStringSpan = ""
) raises -> Int:
    """Parse integer tokens with float32-compatible rounding and alternate syntax.

    Plain decimals within +/- 2**24 use exact integer arithmetic. Other
    spellings retain float32 conversion, including its rounding of large
    integers. The optional context only labels malformed-token diagnostics.
    """
    # Common mesh indices fit exactly in float32. Larger values and alternate
    # spellings keep the existing float32 conversion and validation behavior.
    var bytes = text.as_bytes()
    var pos = 0
    var sign = 1
    if len(bytes) and (bytes[0] == UInt8(45) or bytes[0] == UInt8(43)):
        sign = -1 if bytes[0] == UInt8(45) else 1
        pos = 1
    var start = pos
    var integer = 0
    while pos < len(bytes):
        var digit = Int(bytes[pos]) - 48
        if digit < 0 or digit > 9:
            break
        integer = integer * 10 + digit
        if integer > 16777216:
            break
        pos += 1
    if pos == len(bytes) and pos > start:
        return sign * integer
    var parsed = parse_f32_fast_at(text.as_bytes(), 0)
    if parsed.pos != text.byte_length():
        var prefix = String(context) + " " if context.byte_length() else ""
        raise Error("invalid " + prefix + "number: " + String(text))
    var value = parsed.value
    var converted = Int(value)
    if Float32(converted) != value:
        var prefix = String(context) + " " if context.byte_length() else ""
        raise Error("invalid " + prefix + "integer: " + String(text))
    return converted
