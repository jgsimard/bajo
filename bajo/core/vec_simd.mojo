from std.math import fma, min, max, clamp, sqrt
from std.testing import assert_almost_equal

comptime Vec3f32 = Vec3[DType.float32]


@fieldwise_init
struct Vec2[dtype: DType, width: Int](TrivialRegisterPassable, Writable):
    var x: SIMD[Self.dtype, Self.width]
    var y: SIMD[Self.dtype, Self.width]

    def __init__(
        out self,
        x: Scalar[Self.dtype],
        y: Scalar[Self.dtype],
    ):
        self.x = SIMD[Self.dtype, Self.width](x)
        self.y = SIMD[Self.dtype, Self.width](y)

    @always_inline
    def __add__(self, rhs: Self) -> Self:
        return Self(
            self.x + rhs.x,
            self.y + rhs.y,
        )

    @always_inline
    def __sub__(self, rhs: Self) -> Self:
        return Self(
            self.x - rhs.x,
            self.y - rhs.y,
        )

    @always_inline
    def __mul__(self, rhs: Self) -> Self:
        return Self(
            self.x * rhs.x,
            self.y * rhs.y,
        )

    @always_inline
    def __mul__(self, rhs: SIMD[Self.dtype, Self.width]) -> Self:
        return Self(
            self.x * rhs,
            self.y * rhs,
        )

    @always_inline
    def __truediv__(self, rhs: SIMD[Self.dtype, Self.width]) -> Self:
        return Self(
            self.x / rhs,
            self.y / rhs,
        )

    @always_inline
    def __neg__(self) -> Self:
        return Self(-self.x, -self.y)


@fieldwise_init
struct Vec3[dtype: DType, width: Int = 1](TrivialRegisterPassable, Writable):
    var x: SIMD[Self.dtype, Self.width]
    var y: SIMD[Self.dtype, Self.width]
    var z: SIMD[Self.dtype, Self.width]

    def __init__(
        out self,
        x: Scalar[Self.dtype],
    ):
        self.x = SIMD[Self.dtype, Self.width](x)
        self.y = SIMD[Self.dtype, Self.width](x)
        self.z = SIMD[Self.dtype, Self.width](x)

    @always_inline
    def __add__(self, rhs: Self) -> Self:
        return Self(
            self.x + rhs.x,
            self.y + rhs.y,
            self.z + rhs.z,
        )

    @always_inline
    def __iadd__(mut self, rhs: Self):
        self.x += rhs.x
        self.y += rhs.y
        self.z += rhs.z

    @always_inline
    def __sub__(self, rhs: Self) -> Self:
        return Self(
            self.x - rhs.x,
            self.y - rhs.y,
            self.z - rhs.z,
        )

    @always_inline
    def __mul__(self, rhs: Self) -> Self:
        return Self(
            self.x * rhs.x,
            self.y * rhs.y,
            self.z * rhs.z,
        )

    @always_inline
    def __mul__(self, rhs: SIMD[Self.dtype, Self.width]) -> Self:
        return Self(
            self.x * rhs,
            self.y * rhs,
            self.z * rhs,
        )

    @always_inline
    def __rmul__(self, rhs: SIMD[Self.dtype, Self.width]) -> Self:
        return Self(
            self.x * rhs,
            self.y * rhs,
            self.z * rhs,
        )

    @always_inline
    def __truediv__(self, rhs: SIMD[Self.dtype, Self.width]) -> Self:
        return Self(
            self.x / rhs,
            self.y / rhs,
            self.z / rhs,
        )

    @always_inline
    def __truediv__(self, rhs: Self) -> Self:
        return Self(
            self.x / rhs.x,
            self.y / rhs.y,
            self.z / rhs.z,
        )

    @always_inline
    def __neg__(self) -> Self:
        return Self(-self.x, -self.y, -self.z)

    @always_inline
    def __eq__(self, rhs: Self) -> SIMD[DType.bool, Self.width]:
        return self.x.eq(rhs.x) & self.y.eq(rhs.y) & self.z.eq(rhs.z)

    @always_inline
    def __rtruediv__(self, lhs: SIMD[Self.dtype, Self.width]) -> Self:
        return Self(
            lhs / self.x,
            lhs / self.y,
            lhs / self.z,
        )

    @always_inline
    def __getitem__(self, i: Int) -> SIMD[Self.dtype, Self.width]:
        if i == 0:
            return self.x
        elif i == 1:
            return self.y
        else:
            return self.z

    @always_inline
    def __getitem_param__[
        i: Int
    ](self,) -> SIMD[Self.dtype, Self.width]:
        comptime assert i >= 0 and i < 3

        comptime if i == 0:
            return self.x
        elif i == 1:
            return self.y
        else:
            return self.z

    @always_inline
    def __setitem__(
        mut self,
        i: Int,
        value: SIMD[Self.dtype, Self.width],
    ):
        if i == 0:
            self.x = value
        elif i == 1:
            self.y = value
        else:
            self.z = value

    @always_inline
    def __setitem__[
        i: Int
    ](mut self, value: SIMD[Self.dtype, Self.width],):
        comptime assert i >= 0 and i < 3

        comptime if i == 0:
            self.x = value
        elif i == 1:
            self.y = value
        else:
            self.z = value

    @always_inline
    def set_axis[
        i: Int
    ](mut self, value: SIMD[Self.dtype, Self.width],):
        comptime assert i >= 0 and i < 3

        comptime if i == 0:
            self.x = value
        elif i == 1:
            self.y = value
        else:
            self.z = value

    @always_inline
    def add_axis[
        i: Int
    ](mut self, value: SIMD[Self.dtype, Self.width],):
        comptime assert i >= 0 and i < 3

        comptime if i == 0:
            self.x += value
        elif i == 1:
            self.y += value
        else:
            self.z += value


@always_inline
def dot[
    dtype: DType, width: Int
](a: Vec3[dtype, width], b: Vec3[dtype, width],) -> SIMD[dtype, width]:
    return fma(a.x, b.x, fma(a.y, b.y, a.z * b.z))


@always_inline
def vmin[
    dtype: DType, width: Int
](a: Vec3[dtype, width], b: Vec3[dtype, width],) -> Vec3[dtype, width]:
    return Vec3[dtype, width](
        min(a.x, b.x),
        min(a.y, b.y),
        min(a.z, b.z),
    )


@always_inline
def vmin[
    dtype: DType, width: Int
](a: Vec3[dtype, width], b: Vec3[dtype, width], c: Vec3[dtype, width]) -> Vec3[
    dtype, width
]:
    return Vec3[dtype, width](
        min(min(a.x, b.x), c.x),
        min(min(a.y, b.y), c.y),
        min(min(a.z, b.z), c.z),
    )


@always_inline
def vmax[
    dtype: DType, width: Int
](a: Vec3[dtype, width], b: Vec3[dtype, width],) -> Vec3[dtype, width]:
    return Vec3[dtype, width](
        max(a.x, b.x),
        max(a.y, b.y),
        max(a.z, b.z),
    )


@always_inline
def vmax[
    dtype: DType, width: Int
](a: Vec3[dtype, width], b: Vec3[dtype, width], c: Vec3[dtype, width]) -> Vec3[
    dtype, width
]:
    return Vec3[dtype, width](
        max(max(a.x, b.x), c.x),
        max(max(a.y, b.y), c.y),
        max(max(a.z, b.z), c.z),
    )


@always_inline
def vclamp[
    dtype: DType, width: Int
](
    p: Vec3[dtype, width],
    lower: Vec3[dtype, width],
    upper: Vec3[dtype, width],
) -> Vec3[dtype, width]:
    return vmin(vmax(p, lower), upper)


@always_inline
def length2[
    dtype: DType, width: Int
](v: Vec3[dtype, width]) -> SIMD[dtype, width]:
    return dot(v, v)


@always_inline
def length[
    dtype: DType, width: Int
](v: Vec3[dtype, width]) -> SIMD[dtype, width]:
    return sqrt(dot(v, v))


@always_inline
def cross[
    dtype: DType, width: Int
](a: Vec3[dtype, width], b: Vec3[dtype, width],) -> Vec3[dtype, width]:
    return Vec3[dtype, width](
        fma(a.y, b.z, -(a.z * b.y)),
        fma(a.z, b.x, -(a.x * b.z)),
        fma(a.x, b.y, -(a.y * b.x)),
    )


@always_inline
def normalize[
    dtype: DType, width: Int
](v: Vec3[dtype, width]) -> Vec3[dtype, width]:
    comptime assert dtype in [DType.float32, DType.float64]

    var l = length(v)
    var mask = l.gt(SIMD[dtype, width](1.0e-6))

    var inv_l = mask.select(
        SIMD[dtype, width](1.0) / l,
        SIMD[dtype, width](0.0),
    )

    return v * inv_l


def assert_vec_equal[
    dtype: DType, width: Int
](a: Vec3[dtype, width], b: Vec3[dtype, width], atol: Float64 = 1e-5) raises:
    assert_almost_equal(
        a.x,
        b.x,
        msg=String("x"),
        atol=atol,
    )
    assert_almost_equal(
        a.y,
        b.y,
        msg=String("y"),
        atol=atol,
    )
    assert_almost_equal(
        a.z,
        b.z,
        msg=String("z"),
        atol=atol,
    )


@always_inline
def longest_axis[dtype: DType, width: Int](v: Vec3[dtype, width]) -> Int:
    comptime assert width == 1

    var x = v.x[0]
    var y = v.y[0]
    var z = v.z[0]

    if x > y:
        if x > z:
            return 0
        return 2

    if y > z:
        return 1

    return 2
