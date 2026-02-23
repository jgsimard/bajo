from bajo.core.quat import Quaternion
from bajo.core.vec import Vec, dot, cross

comptime Mat22f32 = Mat[DType.float32, 2, 2]
comptime Mat33f32 = Mat[DType.float32, 3, 3]
comptime Mat44f32 = Mat[DType.float32, 4, 4]


@fieldwise_init
struct Mat[
    dtype: DType,
    rows: Int where rows >= 1,
    cols: Int where cols >= 1,
](Copyable, Equatable, Writable):
    comptime V = Vec[Self.dtype, Self.cols]
    comptime TD = InlineArray[Self.V, Self.rows]
    comptime msize = min(self.rows, self.cols)

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

    # fn __init__(out self, *elems: Scalar[Self.dtype]):
    #     debug_assert(
    #         len(elems) == Self.rows * Self.cols,
    #         "Matrix scalar constructor requires exactly rows * cols elements"
    #     )
    #     self.data = Self.TD(uninitialized=True)

    #     comptime for i in range(Self.rows):
    #         var col_vec = Self.V(uninitialized=True)
    #         comptime for j in range(Self.cols):
    #             comptime let flat_idx = i * Self.cols + j
    #             col_vec[j] = elems[flat_idx]

    #         self.data[i] = col_vec

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

    fn __rmul__(self, s: Scalar[Self.dtype]) -> Self:
        return self * s

    fn __truediv__(self, s: Scalar[Self.dtype]) -> Self:
        res = Self(uninitialized=True)
        comptime for i in range(Self.rows):
            res[i] = self[i] / s
        return res^

    # matmul
    fn __mul__[
        out_cols: Int where out_cols >= 1
    ](self, other: Mat[Self.dtype, Self.cols, out_cols]) -> Mat[
        Self.dtype, Self.rows, out_cols
    ]:
        res = Mat[Self.dtype, Self.rows, out_cols](Scalar[Self.dtype](0))
        comptime for i in range(Self.rows):
            comptime for j in range(out_cols):
                # Using dot product logic between row of A and column of B
                dot_val: Scalar[Self.dtype] = 0
                comptime for k in range(Self.cols):
                    dot_val += self[i][k] * other[k][j]
                res[i][j] = dot_val
        return res^

    fn __mul__(
        self, v: Vec[Self.dtype, Self.cols]
    ) -> Vec[Self.dtype, Self.rows]:
        """Matrix-Vector product."""
        res = Vec[Self.dtype, Self.rows](uninitialized=True)
        comptime for i in range(Self.rows):
            res[i] = dot(self[i], v)
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
        var res = Vec[Self.dtype, Self.msize](uninitialized=True)
        comptime for i in range(Self.msize):
            res[i] = self[i][i]
        return res^


# free functions : Linear Algebra
fn determinant[dtype: DType](m: Mat[dtype, 2, 2]) -> Scalar[dtype]:
    return m[0][0] * m[1][1] - m[0][1] * m[1][0]


fn determinant[dtype: DType](m: Mat[dtype, 3, 3]) -> Scalar[dtype]:
    return dot(m[0], cross(m[1], m[2]))


fn inverse[dtype: DType](m: Mat[dtype, 2, 2]) -> Mat[dtype, 2, 2]:
    det = determinant(m)
    if fabs(det) < 1e-8:
        return Mat[dtype, 2, 2](0)
    return Mat[dtype, 2, 2](m[1][1], -m[0][1], -m[1][0], m[0][0]) * (1.0 / det)


fn inverse[dtype: DType](m: Mat[dtype, 3, 3]) -> Mat[dtype, 3, 3]:
    det = determinant(m)
    if fabs(det) < 1e-8:
        return Mat[dtype, 3, 3](0)

    # Adjugate matrix
    r0 = cross(m[1], m[2])
    r1 = cross(m[2], m[0])
    r2 = cross(m[0], m[1])

    # Transpose rows (which are currently the columns of the adjugate)
    adj = Mat[dtype, 3, 3](r0, r1, r2).transpose()
    return adj * (1.0 / det)


fn outer[
    dtype: DType, r: Int, c: Int
](a: Vec[dtype, r], b: Vec[dtype, c]) -> Mat[dtype, r, c]:
    res = Mat[dtype, r, c](uninitialized=True)
    comptime for i in range(r):
        comptime for j in range(c):
            res[i][j] = a[i] * b[j]
    return res


fn skew[dtype: DType](a: Vec[dtype, 3]) -> Mat[dtype, 3, 3]:
    z = Scalar[dtype](0)
    return Mat[dtype, 3, 3](
        Vec[dtype, 3](z, -a[2], a[1]),
        Vec[dtype, 3](a[2], z, -a[0]),
        Vec[dtype, 3](-a[1], a[0], z),
    )
