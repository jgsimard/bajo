"""Float32 error bounds, ordering, extrema, and outward endpoint conversion.

Integer comparisons preserve subnormals even when floating-point arithmetic
flushes them to zero. The ordering helpers require finite inputs and distinguish
negative zero from positive zero.
"""

from std.memory import bitcast


# Relative rounding-error bound for n Float32 operations (0 <= n < 2**24),
# evaluated in Float32 or Float64 as requested.
comptime FLOAT32_UNIT_ROUNDOFF[dtype: DType] = Scalar[dtype](
    5.9604644775390625e-8
)
comptime FLOAT32_GAMMA[operations: Int, dtype: DType = .float64] = (
    Scalar[dtype](operations) * FLOAT32_UNIT_ROUNDOFF[dtype]
) / (1 - Scalar[dtype](operations) * FLOAT32_UNIT_ROUNDOFF[dtype])


@inline(.always)
def finite_f32_key[
    width: SIMDLength
](bits: SIMD[.uint32, width]) -> SIMD[.uint32, width]:
    # IEEE finite values have integer total order after sign reversal.
    return (bits >> 31).ne(0).select(~bits, bits ^ UInt32(0x80000000))


@inline(.always)
def finite_f32_min[
    width: SIMDLength
](a: SIMD[.float32, width], b: SIMD[.float32, width]) -> SIMD[.float32, width]:
    var ua = bitcast[.uint32](a)
    var ub = bitcast[.uint32](b)
    return bitcast[.float32](
        finite_f32_key(ua).lt(finite_f32_key(ub)).select(ua, ub)
    )


@inline(.always)
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
