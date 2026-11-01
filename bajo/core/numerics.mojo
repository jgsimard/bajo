"""Float32 ordering, finite extrema, and outward endpoint conversion.

Integer comparisons preserve subnormals even when floating-point arithmetic
flushes them to zero. These helpers require finite inputs and distinguish
negative zero from positive zero.
"""

from std.memory import bitcast


@always_inline
def finite_f32_key[
    width: SIMDLength
](bits: SIMD[.uint32, width]) -> SIMD[.uint32, width]:
    # IEEE finite values have integer total order after sign reversal.
    return (bits >> 31).ne(0).select(~bits, bits ^ UInt32(0x80000000))


@always_inline
def finite_f32_min[
    width: SIMDLength
](a: SIMD[.float32, width], b: SIMD[.float32, width]) -> SIMD[.float32, width]:
    var ua = bitcast[.uint32](a)
    var ub = bitcast[.uint32](b)
    return bitcast[.float32](
        finite_f32_key(ua).lt(finite_f32_key(ub)).select(ua, ub)
    )


@always_inline
def finite_f32_max[
    width: SIMDLength
](a: SIMD[.float32, width], b: SIMD[.float32, width]) -> SIMD[.float32, width]:
    var ua = bitcast[.uint32](a)
    var ub = bitcast[.uint32](b)
    return bitcast[.float32](
        finite_f32_key(ua).gt(finite_f32_key(ub)).select(ua, ub)
    )


def round_float32_outward(value: Float64, upper: Bool) -> Float32:
    """Round a Float64 endpoint toward +infinity (upper) or -infinity."""
    var rounded = Float32(value)
    if (upper and Float64(rounded) < value) or (
        not upper and Float64(rounded) > value
    ):
        if rounded == 0.0:
            return bitcast[.float32](UInt32(1) if upper else UInt32(0x80000001))
        var bits = bitcast[.uint32](rounded)
        if (rounded > 0.0) == upper:
            bits += 1
        else:
            bits -= 1
        rounded = bitcast[.float32](bits)
    return rounded
