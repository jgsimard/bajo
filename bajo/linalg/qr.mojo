from math import copysign, sqrt
from os import abort
from random import rand, seed

from layout import Layout, LayoutTensor, UNKNOWN_VALUE
from memory import memcpy
from linalg.matmul import matmul

from testing import assert_almost_equal


fn qr_factorization[
    dtype: DType,
    element_layout: Layout,
](
    sigma: LayoutTensor[mut=True, dtype, element_layout=element_layout, ...],
    A: LayoutTensor[mut=True, dtype, element_layout=element_layout, ...],
):
    m, n = Int(A.runtime_layout.shape[0]), Int(A.runtime_layout.shape[1])
    for k in range(n):
        x_0 = A[k, k]
        x_norm = SIMD[dtype, A.element_layout.size()](0.0)
        for i in range(m - k):
            x_norm += A[k + i, k] * A[k + i, k]
        x_norm = sqrt(x_norm)
        nu = copysign(x_norm, x_0)
        A[k, k] = -nu
        xi = x_0 + nu
        inv_xi = 1.0 / xi
        for i in range(m - k - 1):
            A[k + i + 1, k] *= inv_xi
        sigma[k] = xi / nu
        # apply reflector to A[k + 1:m, k + 1:n] for each column vector v in A[k
        # :m, k + 1:n], we compute:
        #   (I - \sigma [1; w] [1; w]^T) v = v - \sigma [1; w] ([1; w]^T v)
        # = v - \sigma ([1; w]^T v) [1; w]
        # = v - s [1; w]            where  s = \sigma * (v[0] + w^T v[1:])
        # v[0] -= s
        # v[1:] -= s * w
        for j in range(n - k - 1):
            dot = A[k, k + j + 1]  # v[0]
            for i in range(m - k - 1):
                wi = A[k + i + 1, k]  # w[i]
                vi = A[k + i + 1, k + j + 1]  # v[i + 1]
                dot += wi * vi
            s = sigma[k] * dot
            A[k, k + j + 1] -= s  # v[0] -= s
            for i in range(m - k - 1):
                A[k + i + 1, k + j + 1] -= (
                    s * A[k + i + 1, k]
                )  # v[i + 1] -= s * w


fn apply_q[
    dtype: DType,
    element_layout: Layout,
](
    sigma: LayoutTensor[dtype, element_layout=element_layout, ...],
    A: LayoutTensor[dtype, element_layout=element_layout, ...],
    X: LayoutTensor[mut=True, dtype, element_layout=element_layout, ...],
):
    """Applies the implicit Q factor stored in `A` and `sigma` after calling
    `qr_factorization` to the `X` matrix.

    See `qr_factorization` for more details on the construction of the
    Householder reflector.
    """
    m, n = Int(A.runtime_layout.shape[0]), Int(A.runtime_layout.shape[1])
    q_m, q_n = Int(X.runtime_layout.shape[0]), Int(X.runtime_layout.shape[1])
    if q_m != m:
        abort("apply_q: X must have the same number of rows as A")
    for k in range(n - 1, -1, -1):
        for j in range(q_n):
            dot = X[k, j]  # v[0]
            for i in range(m - k - 1):
                wi = A[k + i + 1, k]  # w[i]
                vi = X[k + i + 1, j]  # v[i + 1]
                dot += wi * vi
            s = sigma[k] * dot
            X[k, j] -= s  # v[0] -= s
            for i in range(m - k - 1):
                X[k + i + 1, j] -= s * A[k + i + 1, k]  # v[i + 1] -= s * w


fn form_q[
    dtype: DType,
    element_layout: Layout,
](
    sigma: LayoutTensor[dtype, element_layout=element_layout, ...],
    A: LayoutTensor[dtype, element_layout=element_layout, ...],
    Q: LayoutTensor[mut=True, dtype, element_layout=element_layout, ...],
):
    """Forms the Q factor from the implicit Q factor stored in `A` and `sigma`
    after calling `qr_factorization` and stores the result in `Q`.
    """
    q_m, q_n = Int(Q.runtime_layout.shape[0]), Int(Q.runtime_layout.shape[1])
    min_mn = min(q_m, q_n)

    _ = Q.fill(0.0)

    # Set diagonal to 1.0
    for i in range(min_mn):
        Q[i, i] = 1.0

    apply_q[dtype](sigma, A, Q)


# A is a general matrix, B is a non-unit upper triangular matrix
fn trmm[
    dtype: DType,
    element_layout: Layout,
](
    A: LayoutTensor[dtype, element_layout=element_layout, ...],
    B: LayoutTensor[dtype, element_layout=element_layout, ...],
    C: LayoutTensor[mut=True, dtype, element_layout=element_layout, ...],
):
    m, k1 = Int(A.runtime_layout.shape[0]), Int(A.runtime_layout.shape[1])
    k, n = Int(B.runtime_layout.shape[0]), Int(B.runtime_layout.shape[1])
    min_kn = min(k, n)
    if k1 < min_kn:
        abort("trmm: A and B must have the at least the same number of columns")

    _ = C.fill(0.0)

    for i in range(m):
        for j in range(min_kn):
            for p in range(j + 1):
                C[i, j] += A[i, p] * B[p, j]


fn a_mul_bt[
    dtype: DType,
    element_layout: Layout,
](
    A: LayoutTensor[dtype, element_layout=element_layout, ...],
    B: LayoutTensor[dtype, element_layout=element_layout, ...],
    C: LayoutTensor[mut=True, dtype, element_layout=element_layout, ...],
):
    m, k1 = Int(A.runtime_layout.shape[0]), Int(A.runtime_layout.shape[1])
    n, k = Int(B.runtime_layout.shape[0]), Int(B.runtime_layout.shape[1])
    if k1 != k:
        abort("a_mul_bt: A and B must have the same number of columns")

    _ = C.fill(0.0)

    for i in range(m):
        for j in range(n):
            for p in range(k):
                C[i, j] += A[i, p] * B[j, p]


def all_almost_id[
    dtype: DType,
    element_layout: Layout,
](
    A: LayoutTensor[dtype, element_layout=element_layout, ...],
    atol: Float64,
    rtol: Float64,
):
    m, n = Int(A.runtime_layout.shape[0]), Int(A.runtime_layout.shape[1])
    for i in range(m):
        for j in range(n):
            reference = SIMD[dtype, A.element_layout.size()](
                1.0 if i == j else 0.0
            )
            assert_almost_equal(A[i, j], reference, atol=atol, rtol=rtol)


fn create_vector[
    dtype: DType, layout: Layout
](
    m: Int,
    ptr: UnsafePointer[Scalar[dtype]],
    out result: LayoutTensor[dtype, layout, ptr.origin],
):
    var dynamic_layout = type_of(result.runtime_layout)(
        type_of(result.runtime_layout.shape)(m),
        type_of(result.runtime_layout.stride)(1),
    )
    return {ptr, dynamic_layout}


fn create_tensor[
    dtype: DType, layout: Layout
](
    m: Int,
    n: Int,
    ptr: UnsafePointer[Scalar[dtype]],
    out result: LayoutTensor[dtype, layout, ptr.origin],
):
    var dynamic_layout = type_of(result.runtime_layout)(
        type_of(result.runtime_layout.shape)(m, n),
        type_of(result.runtime_layout.stride)(1, m),
    )
    return {ptr, dynamic_layout}

fn create_tensor2[
    dtype: DType, layout: Layout
](
    m: Int,
    n: Int,
    out result: LayoutTensor[dtype, layout, MutExternalOrigin],
):
    var ptr = alloc[Scalar[dtype]](m * n)
    var dynamic_layout = type_of(result.runtime_layout)(
        type_of(result.runtime_layout.shape)(m, n),
        type_of(result.runtime_layout.stride)(1, m),
    )
    return {ptr, dynamic_layout}


fn assert_almost_equal_ptr[
    dtype: DType
](
    ptr_a: UnsafePointer[Scalar[dtype]],
    ptr_b: UnsafePointer[Scalar[dtype]],
    n: Int,
    atol: Float64,
    rtol: Float64,
) raises:
    for i in range(n):
        assert_almost_equal((ptr_a + i)[], (ptr_b + i)[], atol=atol, rtol=rtol)


def main():
    atol = 1e-5
    rtol = 1e-3
    m, n = 80, 50
    min_mn = min(m, n)
    
    comptime a_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime v_layout = Layout(UNKNOWN_VALUE)
    comptime T = Float32

    var a_ptr = alloc[T](m * n)
    var a_ptr_copy = alloc[T](m * n)
    var v_ptr = alloc[T](min_mn)
    seed(123)
    rand[DType.float32](a_ptr, m * n)
    var a = create_tensor[DType.float32, a_layout](m, n, a_ptr)
    memcpy(dest=a_ptr_copy, src=a_ptr, count=m * n)

    # factorize
    var a_copy = create_tensor[DType.float32, a_layout](m, n, a_ptr_copy)
    var v = create_vector[DType.float32, v_layout](min_mn, v_ptr)
    qr_factorization[DType.float32](v, a)

    # form Q
    var q_ptr = alloc[T](m * m)
    var q = create_tensor[DType.float32, a_layout](m, m, q_ptr)
    form_q[DType.float32](v, a, q)
    print("check backward stability")
    var q_mul_r_ptr = alloc[T](m * n)
    var q_mul_r = create_tensor[DType.float32, a_layout](m, n, q_mul_r_ptr)
    trmm[DType.float32](q, a, q_mul_r)

    assert_almost_equal_ptr(
        q_mul_r.ptr, a_copy.ptr, m * n, atol=atol, rtol=rtol
    )
    print("check orthogonality")
    var q_mul_qt_ptr = alloc[T](m * m)
    var q_mul_qt = create_tensor[DType.float32, a_layout](m, m, q_mul_qt_ptr)
    a_mul_bt[DType.float32](q, q, q_mul_qt)
    all_almost_id(q_mul_qt, atol=atol, rtol=rtol)

    a_ptr.free()
    a_ptr_copy.free()
    v_ptr.free()
    q_ptr.free()
    q_mul_r_ptr.free()
    q_mul_qt_ptr.free()
