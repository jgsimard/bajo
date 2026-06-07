from .constants import *
from .primitives import _is_digit


@fieldwise_init
struct F32ParseResult(TrivialRegisterPassable):
    var value: Float32
    var pos: Int


@always_inline
def parse_f32_at[
    o: Origin
](ptr: UnsafePointer[UInt8, o], pos: Int, end: Int,) -> F32ParseResult:
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
            while p < end:
                var eb = ptr.load(p)
                if _is_digit(eb):
                    eval = eval * 10 + Int(eb - ZERO)
                    p += 1
                else:
                    break

            if eval > 0:
                var power: Float64 = 1.0
                for _ in range(eval):
                    power *= 10.0

                if exp_sign == 1:
                    num *= power
                else:
                    num /= power

    return F32ParseResult(Float32(sign * num), p)
