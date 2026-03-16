from std.math import sqrt, clamp
from std.sys.info import size_of
from std.reflection import get_type_name

from bajo.core.vec import Vec, vclamp


@fieldwise_init
struct QuadraticSolutions[dtype: DType, size: Int](
    Copyable, TrivialRegisterPassable
):
    var roots_0: SIMD[Self.dtype, Self.size]
    var roots_1: SIMD[Self.dtype, Self.size]
    var mask: SIMD[DType.bool, Self.size]


def solve_quadratic[
    dtype: DType, size: Int
](
    a: SIMD[dtype, size], b: SIMD[dtype, size], c: SIMD[dtype, size]
) -> QuadraticSolutions[dtype, size]:
    """Solves the quadratic equation `ax^2 + bx + c = 0` element-wise for SIMD vectors.

    This function uses a numerically stable implementation (Citardauq's formula)
    to prevent catastrophic cancellation when 'b' is much larger than 'ac'.
    see: https://en.wikipedia.org/wiki/Quadratic_formula#Square_root_in_the_denominator

    Args:
        a: The quadratic coefficients.
        b: The linear coefficients.
        c: The constant terms.

    Returns:
        QuadraticSolutions[roots_0, roots_1, mask].
    """
    det = b * b - 4.0 * a * c
    mask = det.ge(0.0)  # Element-wise >= 0.0

    sqrt_det = sqrt(max(det, 0.0))

    q = -0.5 * (b + b.ge(0.0).select(sqrt_det, -sqrt_det))

    t0 = c / q
    t1 = q / a

    return QuadraticSolutions(t0, t1, mask)


def solve_quadratic[
    dtype: DType, size: Int
](b: SIMD[dtype, size], c: SIMD[dtype, size]) -> QuadraticSolutions[
    dtype, size
]:
    """Solves the quadratic equation `x^2 + bx + c = 0` element-wise for SIMD vectors.

    This function uses a numerically stable implementation (Citardauq's formula)
    to prevent catastrophic cancellation when 'b' is much larger than 'ac'.
    see: https://en.wikipedia.org/wiki/Quadratic_formula#Square_root_in_the_denominator

    Args:
        b: The linear coefficients.
        c: The constant terms.

    Returns:
        QuadraticSolutions[roots_0, roots_1, mask].
    """
    det = b * b - 4.0 * c
    mask = det.ge(0.0)  # Element-wise >= 0.0

    sqrt_det = sqrt(max(det, 0.0))

    q = -0.5 * (b + b.ge(0.0).select(sqrt_det, -sqrt_det))

    t0 = c / q
    t1 = q

    return QuadraticSolutions(t0, t1, mask)


def smoothstep[
    dtype: DType, size: Int
](a: SIMD[dtype, size], b: SIMD[dtype, size], x: SIMD[dtype, size]) -> SIMD[
    dtype, size
]:
    y = clamp((x - a) / (b - a), 0, 1)
    return y * y * (3 - 2 * y)


def smoothstep[
    dtype: DType, size: Int
](a: Vec[dtype, size], b: Vec[dtype, size], x: Vec[dtype, size]) -> Vec[
    dtype, size
]:
    y = vclamp((x - a) / (b - a), 0, 1)
    return y * y * (-2 * y + 3)


def print_size_of[type: AnyType]():
    comptime name = get_type_name[type]()
    size_bytes = size_of[type]()
    size_32 = size_bytes / 4
    print(t"{name}: {size_bytes} bytes, {size_32} x 32 bits")


def is_power_of_2(n: Int) -> Bool:
    return n > 0 and (n & (n - 1)) == 0
