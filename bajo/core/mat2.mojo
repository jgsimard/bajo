from math import abs

from bajo.core.quat import Quaternion
from bajo.core.vec import Vec, Vec3, Vec4, dot, cross

comptime Mat22 = Mat[_, 2, 2]
comptime Mat33 = Mat[_, 3, 3]
comptime Mat44 = Mat[_, 4, 4]

comptime Mat22f32 = Mat22[DType.float32]
comptime Mat33f32 = Mat33[DType.float32]
comptime Mat44f32 = Mat44[DType.float32]


@fieldwise_init
struct Mat[
    dtype: DType,
    rows: Int where rows >= 1,
    cols: Int where cols >= 1,
](Copyable, Equatable, Writable):
    comptime V = Vec[Self.dtype, Self.cols]
    comptime TD = InlineArray[Self.V, Self.rows]  # TD = Type Data
    comptime msize = min(Self.rows, Self.cols)

    var data: Self.TD

    fn __init__(out self, s: Scalar[Self.dtype]):
        self.data = Self.TD(fill=Self.V(s))

    fn __init__(out self, uninitialized: Bool):
        self.data = Self.TD(uninitialized=uninitialized)

    fn __init__(out self, var *elems: Self.V):
        debug_assert["safe"](
            len(elems) == Self.rows, "No. of elems must match array size"
        )
        self.data = Self.TD(storage=elems^)

    fn __init__(out self, *elems: Scalar[Self.dtype]):
        debug_assert(
            len(elems) == Self.rows * Self.cols,
            "Matrix scalar constructor requires exactly rows * cols elements",
        )
        self.data = Self.TD(uninitialized=True)

        comptime for i in range(Self.rows):
            var row_vec = Self.V(uninitialized=True)
            comptime for j in range(Self.cols):
                comptime flat_idx = i * Self.cols + j
                row_vec[j] = elems[flat_idx]

            self.data[i] = row_vec^

    @staticmethod
    fn identity() -> Self:
        res = Self(Scalar[Self.dtype](0))
        comptime for i in range(min(Self.rows, Self.cols)):
            res[i][i] = 1
        return res^

    @always_inline
    fn __getitem__(
        ref self, i: Int
    ) -> ref[self.data] Vec[Self.dtype, Self.cols]:
        return self.data[i]

    # arithmetic
    fn __neg__(self) -> Self:
        res = Self(uninitialized=True)
        comptime for i in range(Self.rows):
            res[i] = -self[i]
        return res^

    fn __add__(self, other: Self) -> Self:
        res = Self(uninitialized=True)
        comptime for i in range(Self.rows):
            res[i] = self[i] + other[i]
        return res^

    fn __sub__(self, other: Self) -> Self:
        res = Self(uninitialized=True)
        comptime for i in range(Self.rows):
            res[i] = self[i] - other[i]
        return res^

    # scalar
    fn __mul__(self, s: Scalar[Self.dtype]) -> Self:
        res = Self(uninitialized=True)
        comptime for i in range(Self.rows):
            res[i] = self[i] * s
        return res^

    fn __imul__(mut self, s: Scalar[Self.dtype]):
        comptime for i in range(Self.rows):
            self.data[i] = self.data[i] * s

    # fn __rmul__(self, s: Scalar[Self.dtype]) -> Self:
    #     return self * s

    fn __truediv__(self, s: Scalar[Self.dtype]) -> Self:
        res = Self(uninitialized=True)
        comptime for i in range(Self.rows):
            res[i] = self[i] / s
        return res^

    # linear algebra
    fn transpose(self) -> Mat[Self.dtype, Self.cols, Self.rows]:
        res = Mat[Self.dtype, Self.cols, Self.rows](uninitialized=True)
        comptime for i in range(Self.rows):
            comptime for j in range(Self.cols):
                res[j][i] = self[i][j]
        return res^

    fn trace(self) -> Scalar[Self.dtype]:
        res: Scalar[Self.dtype] = 0
        comptime for i in range(min(Self.rows, Self.cols)):
            res += self[i][i]
        return res

    fn get_diag(self) -> Vec[Self.dtype, Self.msize]:
        res = Vec[Self.dtype, Self.msize](uninitialized=True)
        comptime for i in range(Self.msize):
            res[i] = self[i][i]
        return res^

    @staticmethod
    fn diag(v: Vec[Self.dtype, Self.rows]) -> Self:
        """Create a square diagonal matrix from a vector."""
        comptime assert Self.rows == Self.cols, "diag() requires square matrix"
        res = Self(Scalar[Self.dtype](0))
        comptime for i in range(Self.rows):
            res[i][i] = v[i]
        return res^

    fn __mul__(self, other: Self) -> Self:
        res = Self(uninitialized=True)
        comptime for i in range(Self.rows):
            res[i] = self[i] * other[i]
        return res^

    fn __div__(self, other: Self) -> Self:
        res = Self(uninitialized=True)
        comptime for i in range(Self.rows):
            res[i] = self[i] / other[i]
        return res^

    # Contracting Products
    fn ddot(self, other: Self) -> Scalar[Self.dtype]:
        """Double dot product (sum of all element-wise products)."""
        res: Scalar[Self.dtype] = 0
        comptime for i in range(Self.rows):
            res += dot(self[i], other[i])
        return res

    # --- In-place Arithmetic (Missing Overloads) ---
    fn __iadd__(mut self, s: Scalar[Self.dtype]):
        comptime for i in range(Self.rows):
            self[i] += s

    fn __isub__(mut self, s: Scalar[Self.dtype]):
        comptime for i in range(Self.rows):
            self[i] -= s

    # fn __itruediv__(mut self, s: Scalar[Self.dtype]):
    #     comptime for i in range(Self.rows): self[i] /= s

    # Bitwise Unary
    fn __invert__(self) -> Self:
        """Bitwise NOT (~)."""
        res = Self(uninitialized=True)
        comptime for i in range(Self.rows):
            res[i] = ~self[i]  # Uses Vec.__invert__
        return res^

    # --- Bitwise AND (&) ---
    fn __and__(self, other: Self) -> Self:
        res = Self(uninitialized=True)
        comptime for i in range(Self.rows):
            res[i] = self[i] & other[i]
        return res^

    fn __and__(self, s: Scalar[Self.dtype]) -> Self:
        res = Self(uninitialized=True)
        comptime for i in range(Self.rows):
            res[i] = self[i] & s
        return res^

    # fn __iand__(mut self, other: Self):
    #     comptime for i in range(Self.rows):
    #         self[i] &= other[i]

    fn __or__(self, other: Self) -> Self:
        res = Self(uninitialized=True)
        comptime for i in range(Self.rows):
            res[i] = self[i] | other[i]
        return res^

    fn __or__(self, s: Scalar[Self.dtype]) -> Self:
        res = Self(uninitialized=True)
        comptime for i in range(Self.rows):
            res[i] = self[i] | s
        return res^

    # fn __ior__(mut self, other: Self):
    #     comptime for i in range(Self.rows):
    #         self[i] |= other[i]

    # --- Bitwise XOR (^) ---
    fn __xor__(self, other: Self) -> Self:
        res = Self(uninitialized=True)
        comptime for i in range(Self.rows):
            res[i] = self[i] ^ other[i]
        return res^

    fn __xor__(self, s: Scalar[Self.dtype]) -> Self:
        res = Self(uninitialized=True)
        comptime for i in range(Self.rows):
            res[i] = self[i] ^ s
        return res^

    # fn __ixor__(mut self, other: Self):
    #     comptime for i in range(Self.rows):
    #         self[i] ^= other[i]

    # --- Bit Shifts (<<, >>) ---
    fn __lshift__(self, other: Self) -> Self:
        res = Self(uninitialized=True)
        comptime for i in range(Self.rows):
            res[i] = self[i] << other[i]
        return res^

    fn __lshift__(self, s: Scalar[Self.dtype]) -> Self:
        res = Self(uninitialized=True)
        comptime for i in range(Self.rows):
            res[i] = self[i] << s
        return res^

    fn __rshift__(self, other: Self) -> Self:
        res = Self(uninitialized=True)
        comptime for i in range(Self.rows):
            res[i] = self[i] >> other[i]
        return res^

    fn __rshift__(self, s: Scalar[Self.dtype]) -> Self:
        res = Self(uninitialized=True)
        comptime for i in range(Self.rows):
            res[i] = self[i] >> s
        return res^


fn outer[
    dtype: DType, r: Int where r >= 1, c: Int where c >= 1
](a: Vec[dtype, r], b: Vec[dtype, c]) -> Mat[dtype, r, c]:
    res = Mat[dtype, r, c](uninitialized=True)
    comptime for i in range(r):
        comptime for j in range(c):
            res[i][j] = a[i] * b[j]
    return res^


fn skew[dtype: DType](a: Vec[dtype, 3]) -> Mat33[dtype]:
    # fmt: off
    return Mat33[dtype](
        0, -a[2], a[1],
        a[2], 0, -a[0],
        -a[1], a[0], 0,
    )
    # fmt: on


# free functions : Linear Algebra


##############
# matmul
##############
fn _matmul[
    dtype: DType,
    a_rows: Int where a_rows >= 1,
    a_cols: Int where a_cols >= 1,
    b_cols: Int where b_cols >= 1,
](a: Mat[dtype, a_rows, a_cols], b: Mat[dtype, a_cols, b_cols]) -> Mat[
    dtype, a_rows, b_cols
]:
    """Matrix-Matrix product."""
    # res = Mat[dtype, a_rows,b_cols](uninitialized=True)
    # comptime for i in range(a_rows):
    #     comptime for j in range(b_cols):
    #         dot_val: Scalar[dtype] = 0
    #         comptime for k in range(a_cols):
    #             dot_val += a[i][k] * b[k][j]
    #         res[i][j] = dot_val
    # return res^
    bT = b.transpose()
    res = Mat[dtype, a_rows, b_cols](uninitialized=True)
    comptime for i in range(a_rows):
        comptime for j in range(b_cols):
            res[i][j] = dot(a[i], bT[j])
    return res^


fn _matvec[
    dtype: DType,
    rows: Int where rows >= 1,
    cols: Int where cols >= 1,
](m: Mat[dtype, rows, cols], v: Vec[dtype, cols]) -> Vec[dtype, rows]:
    """Matrix-Vector product."""
    res = Vec[dtype, rows](uninitialized=True)
    for i in range(rows):
        print(i)
        res[i] = dot(m[i], v)
    return res^


##############
# determinant
##############
fn determinant[dtype: DType](m: Mat22[dtype]) -> Scalar[dtype]:
    return m[0][0] * m[1][1] - m[0][1] * m[1][0]


fn determinant[dtype: DType](m: Mat33[dtype]) -> Scalar[dtype]:
    return dot(m[0], cross(m[1], m[2]))


fn determinant[dtype: DType](m: Mat44[dtype]) -> Scalar[dtype]:
    """Adapted from USD - see licenses/usd-LICENSE.txt Copyright 2016 Pixar."""
    # Pickle 1st two columns of matrix into registers
    x00 = m[0][0]
    x01 = m[0][1]
    x10 = m[1][0]
    x11 = m[1][1]
    x20 = m[2][0]
    x21 = m[2][1]
    x30 = m[3][0]
    x31 = m[3][1]

    # Pickle 2nd two columns of matrix into registers
    x02 = m[0][2]
    x03 = m[0][3]
    x12 = m[1][2]
    x13 = m[1][3]
    x22 = m[2][2]
    x23 = m[2][3]
    x32 = m[3][2]
    x33 = m[3][3]

    # Compute all six 2x2 determinants of 2nd two columns
    y01 = x02 * x13 - x12 * x03
    y02 = x02 * x23 - x22 * x03
    y03 = x02 * x33 - x32 * x03
    y12 = x12 * x23 - x22 * x13
    y13 = x12 * x33 - x32 * x13
    y23 = x22 * x33 - x32 * x23

    # Compute all 3x3 cofactors for 1st two columns
    z30 = x11 * y02 - x21 * y01 - x01 * y12
    z20 = x01 * y13 - x11 * y03 + x31 * y01
    z10 = x21 * y03 - x31 * y02 - x01 * y23
    z00 = x11 * y23 - x21 * y13 + x31 * y12

    # compute 4x4 determinant & its reciprocal
    return x30 * z30 + x20 * z20 + x10 * z10 + x00 * z00


##############
# inverse
##############
fn inverse[dtype: DType](m: Mat22[dtype]) raises -> Mat22[dtype]:
    comptime EPSILON = 1e-8
    det = determinant(m)
    if abs(det) < EPSILON:
        raise "nope"
    rcp = 1.0 / det
    out = Mat22[dtype](uninitialized=True)
    out[0][0] = m[1][1] * rcp
    out[0][1] = -m[0][1] * rcp
    out[1][0] = -m[1][0] * rcp
    out[1][1] = m[0][0] * rcp
    return out^


fn inverse[dtype: DType](m: Mat33[dtype]) raises -> Mat33[dtype]:
    comptime EPSILON = 1e-8
    det = determinant(m)
    if abs(det) < EPSILON:
        raise "nope"
    rcp = 1.0 / det
    out = Mat33[dtype](uninitialized=True)
    out[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * rcp
    out[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * rcp
    out[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * rcp

    out[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * rcp
    out[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * rcp
    out[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * rcp

    out[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * rcp
    out[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * rcp
    out[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * rcp

    return out^


fn inverse[dtype: DType](m: Mat44[dtype]) raises -> Mat44[dtype]:
    """Adapted from USD - see licenses/usd-LICENSE.txt Copyright 2016 Pixar."""

    comptime EPSILON = 1e-8

    # Pickle 1st two columns of matrix into registers
    x00 = m[0][0]
    x01 = m[0][1]
    x10 = m[1][0]
    x11 = m[1][1]
    x20 = m[2][0]
    x21 = m[2][1]
    x30 = m[3][0]
    x31 = m[3][1]

    # Compute all six 2x2 determinants of 1st two columns
    y01 = x00 * x11 - x10 * x01
    y02 = x00 * x21 - x20 * x01
    y03 = x00 * x31 - x30 * x01
    y12 = x10 * x21 - x20 * x11
    y13 = x10 * x31 - x30 * x11
    y23 = x20 * x31 - x30 * x21

    # Pickle 2nd two columns of matrix into registers
    x02 = m[0][2]
    x03 = m[0][3]
    x12 = m[1][2]
    x13 = m[1][3]
    x22 = m[2][2]
    x23 = m[2][3]
    x32 = m[3][2]
    x33 = m[3][3]

    # Compute all 3x3 cofactors for 2nd two columns
    z33 = x02 * y12 - x12 * y02 + x22 * y01
    z23 = x12 * y03 - x32 * y01 - x02 * y13
    z13 = x02 * y23 - x22 * y03 + x32 * y02
    z03 = x22 * y13 - x32 * y12 - x12 * y23
    z32 = x13 * y02 - x23 * y01 - x03 * y12
    z22 = x03 * y13 - x13 * y03 + x33 * y01
    z12 = x23 * y03 - x33 * y02 - x03 * y23
    z02 = x13 * y23 - x23 * y13 + x33 * y12

    # Compute all six 2x2 determinants of 2nd two columns
    y01 = x02 * x13 - x12 * x03
    y02 = x02 * x23 - x22 * x03
    y03 = x02 * x33 - x32 * x03
    y12 = x12 * x23 - x22 * x13
    y13 = x12 * x33 - x32 * x13
    y23 = x22 * x33 - x32 * x23

    # Compute all 3x3 cofactors for 1st two columns
    z30 = x11 * y02 - x21 * y01 - x01 * y12
    z20 = x01 * y13 - x11 * y03 + x31 * y01
    z10 = x21 * y03 - x31 * y02 - x01 * y23
    z00 = x11 * y23 - x21 * y13 + x31 * y12
    z31 = x00 * y12 - x10 * y02 + x20 * y01
    z21 = x10 * y03 - x30 * y01 - x00 * y13
    z11 = x00 * y23 - x20 * y03 + x30 * y02
    z01 = x20 * y13 - x30 * y12 - x10 * y23

    # compute 4x4 determinant & its reciprocal
    det = x30 * z30 + x20 * z20 + x10 * z10 + x00 * z00

    if abs(det) > EPSILON:
        raise "nope"

    rcp = 1.0 / det

    invm = Mat44[dtype](uninitialized=True)

    # Multiply all 3x3 cofactors by reciprocal & transpose
    invm[0][0] = z00 * rcp
    invm[0][1] = z10 * rcp
    invm[1][0] = z01 * rcp
    invm[0][2] = z20 * rcp
    invm[2][0] = z02 * rcp
    invm[0][3] = z30 * rcp
    invm[3][0] = z03 * rcp
    invm[1][1] = z11 * rcp
    invm[1][2] = z21 * rcp
    invm[2][1] = z12 * rcp
    invm[1][3] = z31 * rcp
    invm[3][1] = z13 * rcp
    invm[2][2] = z22 * rcp
    invm[2][3] = z32 * rcp
    invm[3][2] = z23 * rcp
    invm[3][3] = z33 * rcp

    return invm^


##############
# transform
##############
fn transform_point[
    dtype: DType
](m: Mat44[dtype], v: Vec3[dtype]) -> Vec3[dtype]:
    v4 = Vec4[dtype](v.x(), v.y(), v.z(), 1)
    return _matvec(m, v4).xyz()


fn transform_vector[
    dtype: DType
](m: Mat44[dtype], v: Vec3[dtype]) -> Vec3[dtype]:
    v4 = Vec4[dtype](v.x(), v.y(), v.z(), 0)
    return _matvec(m, v4).xyz()


from sys.info import size_of


fn main():
    print("core.mat2")

    comptime T = DType.float32
    comptime size = 4
    v = Vec[T, size](1)
    m = Mat[T, size, size](2)
    m2 = Mat[T, size, size](3)

    mv = _matvec(m, v)
    print(mv)

    mm2 = _matmul(m, m2)
    print(mm2)
