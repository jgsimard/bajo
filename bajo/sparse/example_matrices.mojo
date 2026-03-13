from random import random_float64

from msparse.base import COOMatrix, CSCMatrix

comptime DefaultValue = DType.float64


def davis_example_small() -> COOMatrix[DefaultValue]:
    """4 x 4 non-symmetric example. Davis, pp 7-8, Eqn (2.1)."""

    # fmt: off
    i = [2, 1, 3, 0, 1, 3, 3, 1, 0, 2]
    j = [2, 0, 3, 2, 1, 0, 1, 3, 0, 1]
    v = [ 3.0, 3.1, 1.0, 3.2, 2.9, 3.5, 0.4, 0.9, 4.5, 1.7]
    # fmt: on
    return COOMatrix[DefaultValue](v^, i^, j^, 4, 4)


def davis_example_chol() -> CSCMatrix[DefaultValue]:
    """11 x 11 symmetric, positive definite Cholesky example. Davis, Fig 4.2, p 39.
    """

    n = 11
    coo = COOMatrix[DefaultValue](n, n)

    # Off-diagonal elements (lower triangle)
    rows = [5, 6, 2, 7, 9, 10, 5, 9, 7, 10, 8, 9, 10, 9, 10, 10]
    cols = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 9]

    for r, c in zip(rows, cols):
        coo.insert_sym(r, c, 1.0)

    # Diagonal elements: i + 10
    for i in range(n):
        coo.insert(i, i, Float64(i + 10))

    return coo.to_csc()


def davis_example_qr(
    add_diag: Float64 = 0.0, random_vals: Bool = False
) -> CSCMatrix[DefaultValue]:
    """8 x 8, non-symmetric QR example. Davis, Figure 5.1, p 74."""

    # fmt: off
    rows = [0, 1, 2, 3, 4, 5, 6, 3, 6, 1, 6, 0, 2, 5, 7, 4, 7, 0, 1, 3, 7, 5, 6]
    cols = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7]
    # fmt: on

    coo = COOMatrix[DefaultValue](8, 8)

    for k in range(len(rows)):
        var val: Float64
        if random_vals:
            val = random_float64(0.0, 1.0)
        else:
            # Diagonal gets 1..7, others get 1.0
            if k < 7:
                val = Float64(k + 1)
            else:
                val = 1.0
        coo.insert(rows[k], cols[k], val)

    if add_diag != 0.0:
        for i in range(8):
            coo.insert(i, i, add_diag)

    return coo.to_csc()


def davis_example_amd() -> CSCMatrix[DefaultValue,]:
    """10 x 10 symmetric, positive definite AMD example. Davis, Figure 7.1, p 101.
    """

    n = 10
    coo = COOMatrix[DefaultValue](n, n)
    # fmt: off
    rows = [0, 3, 5, 1, 4, 5, 8, 2, 4, 5, 6, 3, 6, 7, 4, 6, 8, 5, 6, 7, 8, 9, 7, 8, 9, 8, 9, 9]
    cols = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6, 6, 6, 6, 7, 7, 7, 8, 8, 9]
    # fmt: on

    for r, c in zip(rows, cols):
        coo.insert_sym(r, c, 1.0)

    for i in range(n):
        coo.insert(i, i, Float64(i + 10))

    return coo.to_csc()


# Simple 3x3 matrices for internal testing
def E_mat() -> CSCMatrix[DefaultValue]:
    coo = COOMatrix[DefaultValue](3, 3)
    coo.insert(0, 0, 1.0)
    coo.insert(1, 0, -2.0)
    coo.insert(1, 1, 1.0)
    coo.insert(2, 2, 1.0)
    return coo.to_csc()


def A_mat() -> CSCMatrix[DefaultValue]:
    v = [2.0, 4, -2, 1, -6, 7, 1, 2]
    i = [0, 1, 2, 0, 1, 2, 0, 2]
    j = [0, 0, 0, 1, 1, 1, 2, 2]
    coo = COOMatrix[DefaultValue](v^, i^, j^, 3, 3)
    return coo.to_csc()
