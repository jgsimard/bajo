from benchmark import run, keep, Unit


struct Vector[dim: Int](Copyable):
    var data: InlineArray[Float32, Self.dim * Self.dim]


@fieldwise_init
struct Matrix[dim: Int](Copyable, Stringable):
    var data: InlineArray[Float32, Self.dim * Self.dim]

    fn __getitem__(self, row: Int, col: Int) -> Float32:
        return self.data[row * Self.dim + col]

    fn __setitem__(var self, row: Int, col: Int, value: Float32):
        self.data[row * Self.dim + col] = value

    # fn inverse[use_simd: Bool](self: Matrix[2]) raises -> Matrix[2]:
    #     @parameter
    #     if use_simd:
    #         return inverse_m22_simd(self)
    #     else:
    #         return inverse_m22(self)

    fn inverse[use_simd: Bool](self: Matrix[4]) raises -> Matrix[4]:
        @parameter
        if use_simd:
            return inverse_m44_simd(self)
        else:
            return inverse_m44(self)

    fn __str__(self) -> String:
        str = ""
        for r in range(Self.dim):
            for c in range(Self.dim):
                str += String(self[r, c], "\t")
            str += String("\n")
        return str

    fn __eq__(self, other: Self) -> Bool:
        for i in range(0, 16):
            if self.data[i] != other.data[i]:
                return False
        return True


fn bench_inv4[use_simd: Bool]() raises:
    # fmt: off
    matrix_data : InlineArray[Float32, 16]= [
        2, -1, 0, 0, 
        -5, 2, -1, 0, 
        0, -1, 2, -1, 
        0, 0, -1, 2,
    ]
    # fmt: on
    mat = Matrix[4](matrix_data^)

    fn bench_fn() raises capturing:
        for _ in range(1e3):
            mat = mat.inverse[use_simd]()
        keep(mat.data)

    time_ns = round(run[func3=bench_fn](max_iters=1000).mean(Unit.ns), 1)

    print(t"bench_inv4 : simd={use_simd} : {time_ns} us")


fn main() raises:
    # fmt: off
    matrix_data: InlineArray[Float32, 16] = [
         2, -1,  0,  0,
        -5,  2, -1,  0,
         0, -1,  2, -1,
         0,  0, -1,  2,
    ]
    # fmt: on
    mat = Matrix[4](matrix_data^)

    print("Original Matrix:")
    print(String(mat))

    mat_inv = mat.inverse[False]()
    print(String(mat_inv))

    mat_inv_simd = mat.inverse[True]()
    print(String(mat_inv_simd))
    print(mat_inv == mat_inv_simd)

    bench_inv4[False]()
    bench_inv4[True]()


# fn inverse_m22(mat: Matrix[2]) raises -> Matrix[2]:
#     ref m = mat.data

#     det = 1.0

#     if abs(det) < 1e-6:
#         raise Error("Matrix2 is not invertable")

#     inv_det = 1.0 / det
#     inv_data = InlineArray[Float32, 4](uninitialized=True)

#     return Matrix[2](inv_data^)


# fn inverse_m22_simd(mat: Matrix[2]) raises -> Matrix[2]:
#     ref m = mat.data

#     det = 1.0

#     if abs(det) < 1e-6:
#         raise Error("Matrix2 is not invertable")

#     inv_det = 1.0 / det
#     inv_data = InlineArray[Float32, 4](uninitialized=True)

#     return Matrix[2](inv_data^)


fn inverse_m44(mat: Matrix[4]) raises -> Matrix[4]:
    """
    Computes the inverse of a 4x4 matrix using the adjugate method.
    This is a direct, loop-unrolled implementation for maximum efficiency.
    """
    ref m = mat.data
    # fmt: off
    cofactor00 = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10]
    cofactor01 = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10]
    cofactor02 = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9]
    cofactor03 = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9]

    det = m[0] * cofactor00 + m[1] * cofactor01 + m[2] * cofactor02 + m[3] * cofactor03

    if abs(det) < 1e-6: 
        raise Error("Matrix4 is not invertable")

    inv_det = 1.0 / det
    inv_data = InlineArray[Float32, 16](uninitialized=True)

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
    return Matrix[4](inv_data^)


fn inverse_m44_simd(mat: Matrix[4]) raises -> Matrix[4]:
    """
    Computes the inverse of a 4x4 matrix using a SIMD algorithm that is a
    direct translation of a classic, highly-optimized C/LLVM implementation.
    based on https://github.com/niswegmann/small-matrix-inverse/blob/master/invert4x4_llvm.h .
    """
    # works on row major layout, would need to transpo
    ptr = mat.data.unsafe_ptr()
    row0 = ptr.load[width=4](0)
    row1 = ptr.load[width=4](4)
    row2 = ptr.load[width=4](8)
    row3 = ptr.load[width=4](12)

    # Compute adjoint
    row1 = row1.shuffle[2, 3, 0, 1]()
    row3 = row3.shuffle[2, 3, 0, 1]()

    tmp1 = row2 * row3
    tmp1 = tmp1.shuffle[1, 0, 3, 2]()

    col0 = row1 * tmp1
    col1 = row0 * tmp1

    tmp1 = tmp1.shuffle[2, 3, 0, 1]()

    col0 = row1 * tmp1 - col0
    col1 = row0 * tmp1 - col1
    col1 = col1.shuffle[2, 3, 0, 1]()

    tmp1 = row1 * row2
    tmp1 = tmp1.shuffle[1, 0, 3, 2]()

    col0 = row3 * tmp1 + col0
    col3 = row0 * tmp1

    tmp1 = tmp1.shuffle[2, 3, 0, 1]()

    col0 = col0 - row3 * tmp1
    col3 = row0 * tmp1 - col3
    col3 = col3.shuffle[2, 3, 0, 1]()

    tmp1 = row1.shuffle[2, 3, 0, 1]() * row3
    tmp1 = tmp1.shuffle[1, 0, 3, 2]()
    row2 = row2.shuffle[2, 3, 0, 1]()

    col0 = row2 * tmp1 + col0
    col2 = row0 * tmp1

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
    det = (row0 * col0).reduce_add()

    # Check for non-invertible matrix
    if abs(det) < 1e-6:
        raise Error("Matrix4 is not invertible, det=", det)

    # Compute reciprocal of determinant
    inv_det = SIMD[DType.float32, 4](1.0 / det)

    # Multiply adjugate by 1/det to get the inverse
    col0 *= inv_det
    col1 *= inv_det
    col2 *= inv_det
    col3 *= inv_det

    # The result is in column vectors. Transpose them back into rows
    tmp_t0 = col0.shuffle[0, 1, 4, 5](col1)
    tmp_t1 = col2.shuffle[0, 1, 4, 5](col3)
    tmp_t2 = col0.shuffle[2, 3, 6, 7](col1)
    tmp_t3 = col2.shuffle[2, 3, 6, 7](col3)

    res_R0 = tmp_t0.shuffle[0, 2, 4, 6](tmp_t1)
    res_R1 = tmp_t0.shuffle[1, 3, 5, 7](tmp_t1)
    res_R2 = tmp_t2.shuffle[0, 2, 4, 6](tmp_t3)
    res_R3 = tmp_t2.shuffle[1, 3, 5, 7](tmp_t3)

    # Store result
    inv_data = InlineArray[Float32, 16](uninitialized=True)
    ptr_inv = inv_data.unsafe_ptr()
    ptr_inv.store[width=4](0, res_R0)
    ptr_inv.store[width=4](4, res_R1)
    ptr_inv.store[width=4](8, res_R2)
    ptr_inv.store[width=4](12, res_R3)

    return Matrix[4](inv_data^)
