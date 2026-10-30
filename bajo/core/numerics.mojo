"""Bitwise ordering and extrema for finite Float32 values.

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
