from .constants import *
from .primitives import _is_digit

comptime _MAX_DECIMAL_EXPONENT = 308


@fieldwise_init
struct F32ParseResult(TrivialRegisterPassable):
    var value: Float32
    var pos: Int


@always_inline
def parse_f32_at[
    o: Origin
](ptr: UnsafePointer[UInt8, o], pos: Int, end: Int) raises -> F32ParseResult:
    if pos >= end:
        return F32ParseResult(0.0, pos)

    var p = pos
    var sign: Float64 = 1.0

    if ptr.load(p) == MINUS:
        sign = -1.0
        p += 1
    elif ptr.load(p) == PLUS:
        p += 1

    var num: Float64 = 0.0
    while p < end:
        var b = ptr.load(p)
        if _is_digit(b):
            num = num * 10.0 + Float64(Int(b - ZERO))
            p += 1
        else:
            break

    if p < end and ptr.load(p) == DOT:
        p += 1
        var fra: Float64 = 0.0
        var div: Float64 = 1.0

        while p < end:
            var b = ptr.load(p)
            if _is_digit(b):
                fra = fra * 10.0 + Float64(Int(b - ZERO))
                div *= 10.0
                p += 1
            else:
                break

        num += fra / div

    if p < end:
        var b = ptr.load(p)
        if b == CHAR_e or b == CHAR_E:
            p += 1
            var exp_sign = 1

            if p < end and ptr.load(p) == MINUS:
                exp_sign = -1
                p += 1
            elif p < end and ptr.load(p) == PLUS:
                p += 1

            var eval = 0
            var has_exp_digit = False
            while p < end:
                var eb = ptr.load(p)
                if _is_digit(eb):
                    has_exp_digit = True
                    var digit = Int(eb - ZERO)
                    if eval > (_MAX_DECIMAL_EXPONENT - digit) // 10:
                        raise String(
                            t"OBJ decimal exponent exceeds "
                            t"{_MAX_DECIMAL_EXPONENT}"
                        )

                    eval = eval * 10 + digit
                    p += 1
                else:
                    break

            if not has_exp_digit:
                raise String("missing OBJ decimal exponent digits")

            if eval > 0:
                var power: Float64 = 1.0
                var base: Float64 = 10.0
                var remaining = eval

                while remaining > 0:
                    if remaining & 1:
                        power *= base
                    remaining //= 2
                    if remaining > 0:
                        base *= base

                if exp_sign == 1:
                    num *= power
                else:
                    num /= power

    return F32ParseResult(Float32(sign * num), p)
