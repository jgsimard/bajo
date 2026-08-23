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
                            t"decimal exponent exceeds "
                            t"{_MAX_DECIMAL_EXPONENT}"
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
