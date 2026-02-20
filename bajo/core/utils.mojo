from math import sqrt


@fieldwise_init
struct QuadraticSolutions[type: DType, size: Int](
    Copyable, TrivialRegisterPassable
):
    var roots_0: SIMD[Self.type, Self.size]
    var roots_1: SIMD[Self.type, Self.size]
    var mask: SIMD[DType.bool, Self.size]


fn solve_quadratic[
    type: DType, size: Int
](
    a: SIMD[type, size], b: SIMD[type, size], c: SIMD[type, size]
) -> QuadraticSolutions[type, size]:
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


fn solve_quadratic[
    type: DType, size: Int
](b: SIMD[type, size], c: SIMD[type, size]) -> QuadraticSolutions[type, size]:
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
