from linalg.matmul import matmul
from layout import Layout, LayoutTensor, UNKNOWN_VALUE
from random import rand, seed, randint, random_si64
from time.time import time_function

# from bajo.linalg import create_tensor, create_vector


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


fn trtrs_row[
    dtype: DType, element_layout: Layout
](
    mut x: LayoutTensor[mut=True, dtype, element_layout=element_layout, ...],
    L: LayoutTensor[dtype, element_layout=element_layout, ...],
    b: LayoutTensor[dtype, element_layout=element_layout, ...],
):
    """Row-oriented forward substitution for Lx = b."""
    m = Int(L.runtime_layout.shape[0])
    _n = Int(L.runtime_layout.shape[1])

    for i in range(m):
        var z = SIMD[dtype, element_layout.size()](0.0)
        for j in range(i):
            z += L[i, j] * x[j, 0]
        x[i, 0] = (b[i, 0] - z) / L[i, i]


def main():
    comptime T = DType.float64
    n = 5
    seed(2018)

    comptime mat_layout = Layout.col_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime vec_layout = Layout.col_major(UNKNOWN_VALUE, 1)

    l_ptr = alloc[Scalar[T]](n * n)
    xe_ptr = alloc[Scalar[T]](n)
    b_ptr = alloc[Scalar[T]](n)
    x_ptr = alloc[Scalar[T]](n)

    for i in range(n * n):
        (l_ptr + i).store(0.0)

    # 2. Fill Lower Triangular Matrix L
    # L[j,j] = 1, others are rand(-2, 2)
    var L = create_tensor[T, mat_layout](n, n, l_ptr)
    for j in range(n):
        L[j, j] = 1.0
        for i in range(j + 1, n):
            L[i, j] = Scalar[T](random_si64(-2, 2))

    # 3. Create Solution xe and Right-hand side b
    var xe = create_vector[T, vec_layout](n, xe_ptr)
    for i in range(n):
        xe[i, 0] = Scalar[T](random_si64(0, 9))  # 0 to 9

    var b = create_vector[T, vec_layout](n, b_ptr)
    print("L")
    print(L)
    print(L.shape[0](), L.shape[1]())
    print("xe")
    print(xe)
    print("b")
    print(b.shape[0](), b.shape[1]())
    print(b)

    matmul(b, L, xe, None)

    # 4. Solve and Benchmark
    var x = create_vector[T, vec_layout](n, x_ptr)

    # Row-based Solve
    trtrs_row[T](x, L, b)
    print("x")
    print(x)
    # print("L", L)
    # print("x", x)
    # print("xe", xe)
    # print("b", b)
    # fn solve_with_trtrs_row() capturing:
    #     trtrs_row[T](x, L, b)

    # t_row = time_function[solve_with_trtrs_row]()
    # t_row_sec = Float64(t_row)/Float64(1e9)
    # print("Row-oriented solve time: ", t_row_sec, " seconds")

    # Verify result (first 5 elements)
    var good = True
    for i in range(n):
        if x[i, 0] != xe[i, 0]:
            good = False
    print("Row solve result matches xe: ", good)

    # # Column-based Solve
    # start = now()
    # trtrs_col[T](L, b, x)
    # end = now()
    # print("Column-oriented solve time: ", Float64(end - start)/1e9, " seconds")

    # # Final verification
    # match = True
    # for i in range(n):
    #     if x[i] != xe[i]:
    #         match = False
    # print("Column solve result matches xe: ", match)

    # Clean up
    l_ptr.free()
    xe_ptr.free()
    b_ptr.free()
    x_ptr.free()
