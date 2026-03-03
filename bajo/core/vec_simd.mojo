from math import sqrt, min, max
from testing import assert_almost_equal
from sys import CompilationTarget
from std.bit import next_power_of_two

comptime Vec2 = Vec[_, 2]
comptime Vec3 = Vec[_, 3]
comptime Vec4 = Vec[_, 4]

comptime Vec2i8 = Vec2[DType.int]
comptime Vec3i8 = Vec3[DType.int]
comptime Vec4i8 = Vec4[DType.int]

comptime Vec2u8 = Vec2[DType.uint8]
comptime Vec3u8 = Vec3[DType.uint8]
comptime Vec4u8 = Vec4[DType.uint8]

comptime Vec2i16 = Vec2[DType.int16]
comptime Vec3i16 = Vec3[DType.int16]
comptime Vec4i16 = Vec4[DType.int16]

comptime Vec2u16 = Vec2[DType.uint16]
comptime Vec3u16 = Vec3[DType.uint16]
comptime Vec4u16 = Vec4[DType.uint16]

comptime Vec2i32 = Vec2[DType.int32]
comptime Vec3i32 = Vec3[DType.int32]
comptime Vec4i32 = Vec4[DType.int32]

comptime Vec2u32 = Vec2[DType.uint32]
comptime Vec3u32 = Vec3[DType.uint32]
comptime Vec4u32 = Vec4[DType.uint32]

comptime Vec2i64 = Vec2[DType.int64]
comptime Vec3i64 = Vec3[DType.int64]
comptime Vec4i64 = Vec4[DType.int64]

comptime Vec2u64 = Vec2[DType.uint64]
comptime Vec3u64 = Vec3[DType.uint64]
comptime Vec4u64 = Vec4[DType.uint64]

comptime Vec2f16 = Vec2[DType.float16]
comptime Vec3f16 = Vec3[DType.float16]
comptime Vec4f16 = Vec4[DType.float16]

comptime Vec2f32 = Vec2[DType.float32]
comptime Vec3f32 = Vec3[DType.float32]
comptime Vec4f32 = Vec4[DType.float32]

comptime Vec2f64 = Vec2[DType.float64]
comptime Vec3f64 = Vec3[DType.float64]
comptime Vec4f64 = Vec4[DType.float64]


@fieldwise_init
struct Vec[dtype: DType, size: Int](
    Copyable, Equatable, Roundable, TrivialRegisterPassable, Writable
):
    """Wrapper around SIMD."""

    comptime psize = next_power_of_two(Self.size)
    comptime S = Scalar[Self.dtype]
    comptime T = SIMD[Self.dtype, Self.psize]
    var data: Self.T

    # TODO: change this
    # ugly but it works
    fn __init__(out self: Vec3f32, v: Int):
        self.data = rebind[Vec[DType.float32, 3].T](
            SIMD[DType.float32, 4](Float32(v), Float32(v), Float32(v), 0)
        )

    fn __init__(out self: Vec3f32, v: Float64):
        self.data = rebind[Vec[DType.float32, 3].T](
            SIMD[DType.float32, 4](Float32(v), Float32(v), Float32(v), 0)
        )

    fn __init__(
        out self: Vec[Self.dtype, 2],
        x: Self.S,
        y: Self.S,
    ):
        self.data = [x, y]

    fn __init__(
        out self: Vec[Self.dtype, 3],
        x: Self.S,
        y: Self.S,
        z: Self.S,
    ):
        self.data = [x, y, z, Self.S(0)]

    fn __init__(
        out self: Vec[Self.dtype, 4],
        x: Self.S,
        y: Self.S,
        z: Self.S,
        w: Self.S,
    ):
        self.data = [x, y, z, w]

    fn __init__(
        out self: Vec[Self.dtype, 6],
        v: Vec3[Self.dtype],
        w: Vec3[Self.dtype],
    ):
        comptime zero = Self.S(0)
        self.data = [v.x(), v.y(), v.z(), w.x(), w.y(), w.z(), zero, zero]

    fn x(self) -> Self.S:
        return self.data[0]

    fn y(self) -> Self.S:
        return self.data[1]

    fn z(self) -> Self.S:
        comptime assert Self.size >= 3
        return self.data[2]

    fn w(self) -> Self.S:
        comptime assert Self.size >= 4
        return self.data[3]

    # Swizzles
    fn xy(self) -> Vec2[Self.dtype]:
        return Vec2[self.dtype](self.x(), self.y())

    fn yz(self) -> Vec2[Self.dtype]:
        comptime assert Self.size >= 3
        return Vec2[self.dtype](self.y(), self.z())

    fn xz(self) -> Vec2[Self.dtype]:
        comptime assert Self.size >= 3
        return Vec2[self.dtype](self.x(), self.z())

    fn xyz(self) -> Vec3[Self.dtype]:
        comptime assert Self.size >= 3
        return Vec3[Self.dtype](self.x(), self.y(), self.z())

    # accessors
    fn __getitem__(self, i: Int) -> Scalar[Self.dtype]:
        return self.data[i]

    fn __setitem__(mut self, i: Int, v: Scalar[Self.dtype]):
        self.data[i] = v

    # inplace modifier
    fn __iadd__(mut self, other: Self):
        comptime for i in range(Self.size):
            self[i] += other[i]

    fn __isub__(mut self, other: Self):
        comptime for i in range(Self.size):
            self[i] -= other[i]

    fn __imul__(mut self, other: Self):
        comptime for i in range(Self.size):
            self[i] *= other[i]

    # methods (produce new Self)
    fn __neg__(self) -> Self:
        return Self(-self.data)

    fn __add__(self, other: Self) -> Self:
        return Self(self.data + other.data)

    fn __sub__(self, other: Self) -> Self:
        return Self(self.data - other.data)

    fn __mul__(self, other: Self) -> Self:
        return Self(self.data * other.data)

    fn __truediv__(self, other: Self) -> Self:
        return Self(self.data / other.data)

    fn __and__(self, other: Self) -> Self:
        return Self(self.data & other.data)

    fn __or__(self, other: Self) -> Self:
        return Self(self.data | other.data)

    fn __xor__(self, other: Self) -> Self:
        return Self(self.data ^ other.data)

    fn __lshift__(self, other: Self) -> Self:
        return Self(self.data << other.data)

    fn __rshift__(self, other: Self) -> Self:
        return Self(self.data >> other.data)

    fn __round__(self) -> Self:
        return Self(round(self.data))

    fn __round__(self, ndigits: Int) -> Self:
        return Self(round(self.data, ndigits))

    fn __invert__(self) -> Self:
        return Self(~self.data)

    # Scalar inplace
    fn __iadd__(mut self, s: Self.S):
        self.data += s

    fn __isub__(mut self, s: Self.S):
        self.data -= s

    fn __imul__(mut self, s: Self.S):
        self.data *= s

    # Scalar methods
    fn __add__(self, s: Self.S) -> Self:
        return Self(self.data + s)

    fn __sub__(self, s: Self.S) -> Self:
        return Self(self.data - s)

    fn __mul__(self, s: Self.S) -> Self:
        return Self(self.data * s)

    fn __rmul__(self, s: Self.S) -> Self:
        return Self(self.data * s)

    fn __truediv__(self, s: Self.S) -> Self:
        return Self(self.data / s)

    fn __rtruediv__(self, s: Self.S) -> Self:
        return Self(s / self.data)

    fn __and__(self, s: Self.S) -> Self:
        return Self(self.data & s)

    fn __or__(self, s: Self.S) -> Self:
        return Self(self.data | s)

    fn __xor__(self, s: Self.S) -> Self:
        return Self(self.data ^ s)

    fn __lshift__(self, s: Self.S) -> Self:
        return Self(self.data << s)

    fn __rshift__(self, s: Self.S) -> Self:
        return Self(self.data >> s)

    # property
    fn is_near_zero[s: Self.S = 1e-8](self) -> Bool:
        return self.data.le(s).reduce_and()


fn dot[
    dtype: DType, size: Int
](a: Vec[dtype, size], b: Vec[dtype, size]) -> Scalar[dtype]:
    return (a.data * b.data).reduce_add()


comptime length2 = length_sq


fn length_sq[dtype: DType, size: Int](v: Vec[dtype, size]) -> Scalar[dtype]:
    return dot(v, v)


fn length[dtype: DType, size: Int](v: Vec[dtype, size]) -> Scalar[dtype]:
    return sqrt(length_sq(v))


fn normalize[dtype: DType, size: Int](v: Vec[dtype, size]) -> Vec[dtype, size]:
    l = length(v)
    if l > 1e-6:
        return v / l
    return Vec[dtype, size](0)


fn cross[dtype: DType](a: Vec[dtype, 3], b: Vec[dtype, 3]) -> Vec[dtype, 3]:
    return Vec[dtype, 3](
        a.y() * b.z() - a.z() * b.y(),
        a.z() * b.x() - a.x() * b.z(),
        a.x() * b.y() - a.y() * b.x(),
    )


fn cross[dtype: DType](a: Vec2[dtype], b: Vec2[dtype]) -> Scalar[dtype]:
    return a.x() * b.y() - a.y() * b.x()


fn lerp[
    dtype: DType, size: Int
](a: Vec[dtype, size], b: Vec[dtype, size], u: Scalar[dtype]) -> Vec[
    dtype, size
]:
    return a * (1.0 - u) + b * u


fn vmin[
    dtype: DType, size: Int
](a: Vec[dtype, size], b: Vec[dtype, size]) -> Vec[dtype, size]:
    return Vec[dtype, size](min(a.data, b.data))


fn vmax[
    dtype: DType, size: Int
](a: Vec[dtype, size], b: Vec[dtype, size]) -> Vec[dtype, size]:
    return Vec[dtype, size](max(a.data, b.data))


fn longest_axis[dtype: DType](v: Vec3[dtype]) -> Int:
    """Returns the index of the longest component of the vector."""
    if v[0] > v[1] and v[0] > v[2]:
        return 0
    if v[1] > v[2]:
        return 1
    return 2


fn assert_vec_equal[
    dtype: DType, size: Int
](a: Vec[dtype, size], b: Vec[dtype, size], atol: Float64 = 1e-5) raises:
    comptime for i in range(size):
        assert_almost_equal(a[i], b[i], msg=t"i={i}", atol=atol)


fn main():
    print("hello warp.vec")
    a = Vec3f32(1, 2, 3)
    b = Vec3f32(4, 5, 6)
    aa = Vec4f32(1, 2, 3, 2)
    bb = Vec4f32(4, 5, 6, 5)
    print(dot(a, b))
    print(dot(aa, bb))
