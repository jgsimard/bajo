from std.math import abs, cos, sin
from std.collections.inline_array import InlineArray
from std.testing import assert_almost_equal

from bajo.core.vec import Vec3
from bajo.core.quat import Quaternion
from bajo.core.frame import Frame


struct Mat[
    dtype: DType,
    rows: Int,
    cols: Int,
    frame: Frame,
    width: SIMDSize = 1,
](Copyable, Writable):
    comptime Elem = SIMD[Self.dtype, Self.width]
    comptime Row = InlineArray[Self.Elem, Self.cols]
    comptime Data = InlineArray[Self.Row, Self.rows]

    var data: Self.Data

    def __init__(out self, s: Scalar[Self.dtype]):
        comptime assert Self.rows >= 1
        comptime assert Self.cols >= 1

        self.data = Self.Data(uninitialized=True)

        comptime for i in range(Self.rows):
            var row = Self.Row(uninitialized=True)
            comptime for j in range(Self.cols):
                row[j] = Self.Elem(s)
            self.data[i] = row^

    def __init__(out self, uninitialized: Bool):
        comptime assert Self.rows >= 1
        comptime assert Self.cols >= 1
        self.data = Self.Data(uninitialized=uninitialized)

    def __init__(out self, *elems: Scalar[Self.dtype]):
        comptime assert Self.rows >= 1
        comptime assert Self.cols >= 1

        debug_assert["safe"](
            len(elems) == Self.rows * Self.cols,
            (
                t"Matrix constructor requires exactly rows ({Self.rows})"
                t" * cols ({Self.cols}) = {Self.rows * Self.cols} elements"
            ),
        )

        self.data = Self.Data(uninitialized=True)

        comptime for i in range(Self.rows):
            var row = Self.Row(uninitialized=True)

            comptime for j in range(Self.cols):
                comptime flat_idx = i * Self.cols + j
                row[j] = Self.Elem(elems[flat_idx])

            self.data[i] = row^

    @staticmethod
    def identity() -> Self:
        var res = Self(Scalar[Self.dtype](0))

        comptime msize = Self.rows if Self.rows < Self.cols else Self.cols
        comptime for i in range(msize):
            res[i][i] = Self.Elem(1.0)

        return res^

    @staticmethod
    def from_cols(
        c0: Vec3[Self.dtype, Self.frame, Self.width],
        c1: Vec3[Self.dtype, Self.frame, Self.width],
        c2: Vec3[Self.dtype, Self.frame, Self.width],
    ) -> Self:
        comptime assert Self.rows == 3
        comptime assert Self.cols == 3

        var m = Self(uninitialized=True)

        m[0][0] = c0.x
        m[1][0] = c0.y
        m[2][0] = c0.z

        m[0][1] = c1.x
        m[1][1] = c1.y
        m[2][1] = c1.z

        m[0][2] = c2.x
        m[1][2] = c2.y
        m[2][2] = c2.z

        return m^

    @staticmethod
    def from_rotation_scale(
        r: Quaternion[Self.dtype],
        s: Vec3[Self.dtype, Self.width],
    ) -> Self where Self.dtype.is_floating_point():
        comptime assert Self.rows == 3
        comptime assert Self.cols == 3

        var x = r.x
        var y = r.y
        var z = r.z
        var w = r.w

        var x2 = x * x
        var y2 = y * y
        var z2 = z * z

        var xy = x * y
        var xz = x * z
        var yz = y * z

        var wx = w * x
        var wy = w * y
        var wz = w * z

        var sx = s.x
        var sy = s.y
        var sz = s.z

        var dsx = sx * 2.0
        var dsy = sy * 2.0
        var dsz = sz * 2.0

        var m = Self(uninitialized=True)

        # R * diag(scale), row-major storage.
        m[0][0] = sx - dsx * (y2 + z2)
        m[0][1] = dsy * (xy - wz)
        m[0][2] = dsz * (xz + wy)

        m[1][0] = dsx * (xy + wz)
        m[1][1] = sy - dsy * (x2 + z2)
        m[1][2] = dsz * (yz - wx)

        m[2][0] = dsx * (xz - wy)
        m[2][1] = dsy * (yz + wx)
        m[2][2] = sz - dsz * (x2 + y2)

        return m^

    def __getitem_param__[i: Int](ref self) -> ref[self.data] Self.Row:
        return self.data[i]

    def __getitem__(ref self, i: Int) -> ref[self.data] Self.Row:
        return self.data[i]

    def __neg__(self) -> Self:
        var res = Self(uninitialized=True)

        comptime for i in range(Self.rows):
            var row = Self.Row(uninitialized=True)
            comptime for j in range(Self.cols):
                row[j] = -self[i][j]
            res.data[i] = row^

        return res^

    def __add__(self, other: Self) -> Self:
        var res = Self(uninitialized=True)

        comptime for i in range(Self.rows):
            var row = Self.Row(uninitialized=True)
            comptime for j in range(Self.cols):
                row[j] = self[i][j] + other[i][j]
            res.data[i] = row^

        return res^

    def __sub__(self, other: Self) -> Self:
        var res = Self(uninitialized=True)

        comptime for i in range(Self.rows):
            var row = Self.Row(uninitialized=True)
            comptime for j in range(Self.cols):
                row[j] = self[i][j] - other[i][j]
            res.data[i] = row^

        return res^

    def __mul__(self, s: SIMD[Self.dtype, Self.width]) -> Self:
        var res = Self(uninitialized=True)

        comptime for i in range(Self.rows):
            var row = Self.Row(uninitialized=True)
            comptime for j in range(Self.cols):
                row[j] = self[i][j] * s
            res.data[i] = row^

        return res^

    def __truediv__(self, s: SIMD[Self.dtype, Self.width]) -> Self:
        var res = Self(uninitialized=True)

        comptime for i in range(Self.rows):
            var row = Self.Row(uninitialized=True)
            comptime for j in range(Self.cols):
                row[j] = self[i][j] / s
            res.data[i] = row^

        return res^

    def elem_mul(self, other: Self) -> Self:
        var res = Self(uninitialized=True)

        comptime for i in range(Self.rows):
            var row = Self.Row(uninitialized=True)
            comptime for j in range(Self.cols):
                row[j] = self[i][j] * other[i][j]
            res.data[i] = row^

        return res^

    def col[j: Int](self) -> Vec3[Self.dtype, Self.frame, Self.width]:
        comptime assert Self.rows == 3
        comptime assert Self.cols == 3
        comptime assert j >= 0 and j < 3

        return Vec3[Self.dtype, Self.frame, Self.width](
            self[0][j],
            self[1][j],
            self[2][j],
        )

    def transpose(
        self,
    ) -> Mat[Self.dtype, Self.cols, Self.rows, Self.frame, Self.width]:
        var res = Mat[Self.dtype, Self.cols, Self.rows, Self.frame, Self.width](
            uninitialized=True
        )

        comptime for i in range(Self.rows):
            comptime for j in range(Self.cols):
                res[j][i] = self[i][j]

        return res^

    def trace(self) -> SIMD[Self.dtype, Self.width]:
        comptime msize = Self.rows if Self.rows < Self.cols else Self.cols

        var res = SIMD[Self.dtype, Self.width](0.0)
        comptime for i in range(msize):
            res = res + self[i][i]

        return res

    def ddot(self, other: Self) -> SIMD[Self.dtype, Self.width]:
        var res = SIMD[Self.dtype, Self.width](0.0)

        comptime for i in range(Self.rows):
            comptime for j in range(Self.cols):
                res = res + self[i][j] * other[i][j]

        return res

    def __eq__(self, other: Self) -> SIMD[DType.bool, Self.width]:
        var mask = self[0][0].eq(other[0][0])

        comptime for i in range(Self.rows):
            comptime for j in range(Self.cols):
                mask = mask & self[i][j].eq(other[i][j])

        return mask

    def __str__(self) -> String:
        var res = String("[")
        for i in range(Self.rows):
            if i > 0:
                res += " "
            res += "["

            for j in range(Self.cols):
                res += String(self[i][j])
                if j < Self.cols - 1:
                    res += ", "

            res += "]"
            if i < Self.rows - 1:
                res += "\n"

        res += "]"
        return res

    def write_to(self, mut writer: Some[Writer]):
        writer.write(String(self))


def _matmul[
    dtype: DType,
    a_rows: Int,
    a_cols: Int,
    b_cols: Int,
    frame: Frame,
    width: SIMDSize,
](
    a: Mat[dtype, a_rows, a_cols, frame, width],
    b: Mat[dtype, a_cols, b_cols, frame, width],
) -> Mat[dtype, a_rows, b_cols, frame, width]:
    var res = Mat[dtype, a_rows, b_cols, frame, width](uninitialized=True)

    comptime for i in range(a_rows):
        comptime for j in range(b_cols):
            var acc = SIMD[dtype, width](0.0)

            comptime for k in range(a_cols):
                acc = acc + a[i][k] * b[k][j]

            res[i][j] = acc

    return res^


def _matvec[
    dtype: DType, frame: Frame, width: SIMDSize
](m: Mat[dtype, 3, 3, frame, width], v: Vec3[dtype, frame, width]) -> Vec3[
    dtype, frame, width
]:
    return Vec3[dtype, frame, width](
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
    )


def assert_mat_equal[
    dtype: DType,
    rows: Int,
    cols: Int,
    frame: Frame,
    width: SIMDSize,
](
    a: Mat[dtype, rows, cols, frame, width],
    b: Mat[dtype, rows, cols, frame, width],
    atol: Float64 = 1e-5,
) raises:
    comptime for i in range(rows):
        comptime for j in range(cols):
            comptime for lane in range(width):
                assert_almost_equal(
                    a[i][j][lane],
                    b[i][j][lane],
                    msg=String(t"[{i}][{j}][{lane}]"),
                    atol=atol,
                )


##############
# determinant
##############
def determinant[
    dtype: DType, width: SIMDSize
](m: Mat[dtype, 2, 2, _, width]) -> SIMD[dtype, width]:
    return m[0][0] * m[1][1] - m[0][1] * m[1][0]


def determinant[
    dtype: DType, width: SIMDSize
](m: Mat[dtype, 3, 3, _, width]) -> SIMD[dtype, width]:
    var a00 = m[0][0]
    var a01 = m[0][1]
    var a02 = m[0][2]

    var a10 = m[1][0]
    var a11 = m[1][1]
    var a12 = m[1][2]

    var a20 = m[2][0]
    var a21 = m[2][1]
    var a22 = m[2][2]

    return (
        a00 * (a11 * a22 - a12 * a21)
        - a01 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * a21 - a11 * a20)
    )


def determinant[
    dtype: DType, width: SIMDSize
](m: Mat[dtype, 4, 4, _, width]) -> SIMD[dtype, width]:
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
@fieldwise_init
struct MatInverseResult[
    dtype: DType,
    n: Int,
    frame: Frame,
    width: SIMDSize,
](Movable):
    var mask: SIMD[DType.bool, Self.width]
    var value: Mat[Self.dtype, Self.n, Self.n, Self.frame, Self.width]


def inverse[
    dtype: DType, frame: Frame
](m: Mat[dtype, 2, 2, frame]) raises -> Mat[dtype, 2, 2, frame]:
    comptime EPSILON = 1e-8
    x00 = m[0][0]
    x01 = m[0][1]
    x10 = m[1][0]
    x11 = m[1][1]
    det = x00 * x11 - x10 * x01
    if abs(det) < EPSILON:
        raise "nope"
    rcp = 1.0 / det
    # fmt: off
    return Mat[dtype, 2, 2, frame](
        x00 * rcp, -x01 * rcp,
        -x10 * rcp, x11 * rcp
    )
    # fmt: on


def inverse[
    dtype: DType, frame: Frame
](m: Mat[dtype, 3, 3, frame]) raises -> Mat[dtype, 3, 3, frame]:
    comptime EPSILON = 1e-8

    var x00 = m[0][0]
    var x01 = m[0][1]
    var x02 = m[0][2]
    var x10 = m[1][0]
    var x11 = m[1][1]
    var x12 = m[1][2]
    var x20 = m[2][0]
    var x21 = m[2][1]
    var x22 = m[2][2]

    # cofactors, dont call determinant(m), to not recompute them
    var z00 = x11 * x22 - x12 * x21
    var z01 = x02 * x21 - x01 * x22
    var z02 = x01 * x12 - x02 * x11
    var z10 = x12 * x20 - x10 * x22
    var z11 = x00 * x22 - x02 * x20
    var z12 = x02 * x10 - x00 * x12
    var z20 = x10 * x21 - x11 * x20
    var z21 = x01 * x20 - x00 * x21
    var z22 = x00 * x11 - x01 * x10

    var det = x00 * z00 + x01 * z10 + x02 * z20

    if abs(det) < EPSILON:
        raise "nope"

    var rcp = 1.0 / det
    # fmt: off
    return Mat[dtype, 3, 3, frame](
        z00 * rcp, z01 * rcp, z02 * rcp,
        z10 * rcp, z11 * rcp, z12 * rcp,
        z20 * rcp, z21 * rcp, z22 * rcp
    )
    # fmt: on


def inverse[
    dtype: DType, frame: Frame
](m: Mat[dtype, 4, 4, frame]) raises -> Mat[dtype, 4, 4, frame]:
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

    if abs(det) < EPSILON:
        raise "nope"

    rcp = 1.0 / det

    invm = Mat[dtype, 4, 4, frame](uninitialized=True)

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
