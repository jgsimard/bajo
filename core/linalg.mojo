import benchmark


struct Vector[dim: Int](Movable):
    var data: InlineArray[Float32, dim * dim]


struct Matrix[dim: Int](Copyable, Movable, Stringable):
    var data: InlineArray[Float32, dim * dim]

    fn __init__(out self, data: InlineArray[Float32, dim * dim]):
        self.data = data

    fn __init__(out self, uninitialized: Bool):
        self.data = InlineArray[Float32, dim * dim](uninitialized=True)

    fn __getitem__(self, row: Int, col: Int) -> Float32:
        return self.data[row * dim + col]

    fn __setitem__(var self, row: Int, col: Int, value: Float32):
        self.data[row * dim + col] = value

    fn inverse[use_simd: Bool](self: Matrix[2]) raises -> Matrix[2]:
        @parameter
        if use_simd:
            return inverse_m22_simd(self)
        else:
            return inverse_m22(self)

    fn inverse[use_simd: Bool](self: Matrix[4]) raises -> Matrix[4]:
        @parameter
        if use_simd:
            return inverse_m44_simd(self)
        else:
            return inverse_m44(self)

    fn __str__(self) -> String:
        var str = ""
        for r in range(dim):
            str += String(
                self[r, 0],
                "\t",
                self[r, 1],
                "\t",
                self[r, 2],
                "\t",
                self[r, 3],
                "\n",
            )
        return str

    fn __eq__(self, other: Self) -> Bool:
        for i in range(0, 16):
            if self.data[i] != other.data[i]:
                return False
        return True


fn bench_inv4[use_simd: Bool]() raises:
    # fmt: off
    var matrix_data = InlineArray[Float32, 16](
        2, -1, 0, 0, 
        -5, 2, -1, 0, 
        0, -1, 2, -1, 
        0, 0, -1, 2,
    )
    # fmt: on

    var mat = Matrix[4](matrix_data)
    for _ in range(1e6):
        var mat_inv = mat.inverse[use_simd]()

        @parameter
        for i in range(16):
            benchmark.keep(mat_inv.data[0])


fn main() raises:
    # fmt: off
    var matrix_data = InlineArray[Float32, 16](
         2, -1,  0,  0,
        -5,  2, -1,  0,
         0, -1,  2, -1,
         0,  0, -1,  2,
    )
    # fmt: on

    var mat = Matrix[4](matrix_data)

    print("Original Matrix:")
    print(String(mat))

    var mat_inv = mat.inverse[False]()
    print(String(mat_inv))

    var mat_inv_simd = mat.inverse[True]()
    print(String(mat_inv_simd))
    print(mat_inv == mat_inv_simd)

    var report = benchmark.run[bench_inv4[False]]()
    report.print()

    var report_simd = benchmark.run[bench_inv4[True]]()
    report_simd.print()

    var speedup = report.mean() / report_simd.mean()
    print(speedup)


fn inverse_m22(mat: Matrix[2]) raises -> Matrix[2]:
    ref m = mat.data

    var det = 1.0

    if abs(det) < 1e-6:
        raise Error("Matrix2 is not invertable")

    var inv_det = 1.0 / det
    var inv_data = InlineArray[Float32, 4](uninitialized=True)

    return Matrix[2](inv_data)


fn inverse_m22_simd(mat: Matrix[2]) raises -> Matrix[2]:
    ref m = mat.data

    var det = 1.0

    if abs(det) < 1e-6:
        raise Error("Matrix2 is not invertable")

    var inv_det = 1.0 / det
    var inv_data = InlineArray[Float32, 4](uninitialized=True)

    return Matrix[2](inv_data)


fn inverse_m44(mat: Matrix[4]) raises -> Matrix[4]:
    """
    Computes the inverse of a 4x4 matrix using the adjugate method.
    This is a direct, loop-unrolled implementation for maximum efficiency.
    """
    ref m = mat.data
    # fmt: off
    var cofactor00 = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10]
    var cofactor01 = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10]
    var cofactor02 = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9]
    var cofactor03 = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9]

    var det = m[0] * cofactor00 + m[1] * cofactor01 + m[2] * cofactor02 + m[3] * cofactor03

    if abs(det) < 1e-6: 
        raise Error("Matrix4 is not invertable")

    var inv_det = 1.0 / det
    var inv_data = InlineArray[Float32, 16](uninitialized=True)

    # Calculate the adjugate matrix and multiply by 1/det
    # Adjugate matrix is the transpose of the cofactor matrix
    inv_data[0] = cofactor00 * inv_det
    inv_data[4] = cofactor01 * inv_det
    inv_data[8] = cofactor02 * inv_det
    inv_data[12] = cofactor03 * inv_det

    inv_data[1] = (-m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10]) * inv_det
    inv_data[5] = (m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10]) * inv_det
    inv_data[9] = (-m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9]) * inv_det
    inv_data[13] = (m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9]) * inv_det

    inv_data[2] = (m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6]) * inv_det
    inv_data[6] = (-m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6]) * inv_det
    inv_data[10] = (m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5]) * inv_det
    inv_data[14] = (-m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5]) * inv_det

    inv_data[3] = (-m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6]) * inv_det
    inv_data[7] = (m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6]) * inv_det
    inv_data[11] = (-m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5]) * inv_det
    inv_data[15] = (m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5]) * inv_det
    # fmt: on
    return Matrix[4](inv_data)


fn inverse_m44_simd(mat: Matrix[4]) raises -> Matrix[4]:
    """
    Computes the inverse of a 4x4 matrix using a SIMD algorithm that is a
    direct translation of a classic, highly-optimized C/LLVM implementation.
    based on https://github.com/niswegmann/small-matrix-inverse/blob/master/invert4x4_llvm.h .
    """
    # works on row major layout, would need to transpo
    var ptr = mat.data.unsafe_ptr()
    var row0 = ptr.load[width=4](0)
    var row1 = ptr.load[width=4](4)
    var row2 = ptr.load[width=4](8)
    var row3 = ptr.load[width=4](12)

    # Compute adjoint
    row1 = row1.shuffle[2, 3, 0, 1]()
    row3 = row3.shuffle[2, 3, 0, 1]()

    var tmp1 = row2 * row3
    tmp1 = tmp1.shuffle[1, 0, 3, 2]()

    var col0 = row1 * tmp1
    var col1 = row0 * tmp1

    tmp1 = tmp1.shuffle[2, 3, 0, 1]()

    col0 = row1 * tmp1 - col0
    col1 = row0 * tmp1 - col1
    col1 = col1.shuffle[2, 3, 0, 1]()

    tmp1 = row1 * row2
    tmp1 = tmp1.shuffle[1, 0, 3, 2]()

    col0 = row3 * tmp1 + col0
    var col3 = row0 * tmp1

    tmp1 = tmp1.shuffle[2, 3, 0, 1]()

    col0 = col0 - row3 * tmp1
    col3 = row0 * tmp1 - col3
    col3 = col3.shuffle[2, 3, 0, 1]()

    tmp1 = row1.shuffle[2, 3, 0, 1]() * row3
    tmp1 = tmp1.shuffle[1, 0, 3, 2]()
    row2 = row2.shuffle[2, 3, 0, 1]()

    col0 = row2 * tmp1 + col0
    var col2 = row0 * tmp1

    tmp1 = tmp1.shuffle[2, 3, 0, 1]()

    col0 = col0 - row2 * tmp1
    col2 = row0 * tmp1 - col2
    col2 = col2.shuffle[2, 3, 0, 1]()

    tmp1 = row0 * row1
    tmp1 = tmp1.shuffle[1, 0, 3, 2]()

    col2 = row3 * tmp1 + col2
    col3 = row2 * tmp1 - col3

    tmp1 = tmp1.shuffle[2, 3, 0, 1]()

    col2 = row3 * tmp1 - col2
    col3 = col3 - row2 * tmp1

    tmp1 = row0 * row3
    tmp1 = tmp1.shuffle[1, 0, 3, 2]()

    col1 = col1 - row2 * tmp1
    col2 = row1 * tmp1 + col2

    tmp1 = tmp1.shuffle[2, 3, 0, 1]()

    col1 = row2 * tmp1 + col1
    col2 = col2 - row1 * tmp1

    tmp1 = row0 * row2
    tmp1 = tmp1.shuffle[1, 0, 3, 2]()

    col1 = row3 * tmp1 + col1
    col3 = col3 - row1 * tmp1

    tmp1 = tmp1.shuffle[2, 3, 0, 1]()

    col1 = col1 - row3 * tmp1
    col3 = row1 * tmp1 + col3

    # End Adjugate Calculation.
    # The adjugate matrix is now stored in col0, col1, col2, col3.

    # Compute determinant
    var det = (row0 * col0).reduce_add()

    # Check for non-invertible matrix
    if abs(det) < 1e-6:
        raise Error("Matrix4 is not invertible, det=", det)

    # Compute reciprocal of determinant
    var inv_det = SIMD[DType.float32, 4](1.0 / det)

    # Multiply adjugate by 1/det to get the inverse
    col0 *= inv_det
    col1 *= inv_det
    col2 *= inv_det
    col3 *= inv_det

    # The result is in column vectors. Transpose them back into rows
    var tmp_t0 = col0.shuffle[0, 1, 4, 5](col1)
    var tmp_t1 = col2.shuffle[0, 1, 4, 5](col3)
    var tmp_t2 = col0.shuffle[2, 3, 6, 7](col1)
    var tmp_t3 = col2.shuffle[2, 3, 6, 7](col3)

    var res_R0 = tmp_t0.shuffle[0, 2, 4, 6](tmp_t1)
    var res_R1 = tmp_t0.shuffle[1, 3, 5, 7](tmp_t1)
    var res_R2 = tmp_t2.shuffle[0, 2, 4, 6](tmp_t3)
    var res_R3 = tmp_t2.shuffle[1, 3, 5, 7](tmp_t3)

    # Store result
    var inv_data = InlineArray[Float32, 16](uninitialized=True)
    var ptr_inv = inv_data.unsafe_ptr()
    ptr_inv.store[width=4](0, res_R0)
    ptr_inv.store[width=4](4, res_R1)
    ptr_inv.store[width=4](8, res_R2)
    ptr_inv.store[width=4](12, res_R3)

    return Matrix[4](inv_data)
