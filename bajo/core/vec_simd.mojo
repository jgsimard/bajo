from std.math import fma, min, max, clamp, sqrt
from std.testing import assert_almost_equal


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
    def __truediv__(self, rhs: SIMD[Self.dtype, Self.width]) -> Self:
        return Self(
            self.x / rhs,
            self.y / rhs,
            self.z / rhs,
        )

    @always_inline
    def __neg__(self) -> Self:
        return Self(-self.x, -self.y, -self.z)

    @always_inline
    def __eq__(self, rhs: Self) -> SIMD[DType.bool, Self.width]:
        return self.x.eq(rhs.x) & self.y.eq(rhs.y) & self.z.eq(rhs.z)


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
def vmax[
    dtype: DType, width: Int
](a: Vec3[dtype, width], b: Vec3[dtype, width],) -> Vec3[dtype, width]:
    return Vec3[dtype, width](
        max(a.x, b.x),
        max(a.y, b.y),
        max(a.z, b.z),
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
