from math import sqrt, min, max
from testing import assert_almost_equal
from sys import CompilationTarget

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
struct Vec[dtype: DType, size: Int](Copyable, Writable):
    # TODO: maybe use psize for SIMD for Vec3 on cpu
    # comptime psize = 4 if Self.size == 3 and CompilationTarget.is_x86() else Self.size
    comptime T = InlineArray[Scalar[Self.dtype], Self.size]
    var data: Self.T

    fn __init__(out self, s: Scalar[Self.dtype]):
        comptime assert Self.size > 0
        self.data = Self.T(fill=s)

    fn __init__(out self, uninitialized: Bool):
        comptime assert Self.size > 0
        self.data = Self.T(uninitialized=uninitialized)

    fn __init__(out self, var *elems: Scalar[Self.dtype]):
        debug_assert(
            len(elems) == Self.size, "No. of elems must match array size"
        )
        self.data = Self.T(storage=elems^)

    fn __init__(
        out self: Vec[Self.dtype, 6],
        v: Vec[Self.dtype, 3],
        w: Vec[Self.dtype, 3],
    ):
        self.data = [v.x(), v.y(), v.z(), w.x(), w.y(), w.z()]

    fn x(self) -> Scalar[Self.dtype]:
        return self.data[0]

    fn y(self) -> Scalar[Self.dtype]:
        return self.data[1]

    fn z(self) -> Scalar[Self.dtype]:
        comptime assert Self.size >= 3
        return self.data[2]

    fn w(self) -> Scalar[Self.dtype]:
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

    @always_inline
    fn __getitem__[
        I: Indexer, //, idx: I
    ](ref self) -> ref[self.data] Scalar[Self.dtype]:
        """With compile-time bounds checking."""
        return self.data[materialize[idx]()]

    @always_inline
    fn __getitem__[
        I: Indexer
    ](ref self, idx: I) -> ref[self.data] Scalar[Self.dtype]:
        # return self.data[idx] # bounds checking
        return self.data.unsafe_get(idx)  # no bounds checking

    fn __neg__(self) -> Self:
        out = Self(uninitialized=True)
        comptime for i in range(Self.size):
            out[i] = -self[i]
        return out^

    fn __add__(self, other: Self) -> Self:
        out = Self(uninitialized=True)
        comptime for i in range(Self.size):
            out[i] = self[i] + other[i]
        return out^

    fn __iadd__(mut self, other: Self):
        comptime for i in range(Self.size):
            self[i] += other[i]

    fn __sub__(self, other: Self) -> Self:
        out = Self(uninitialized=True)
        comptime for i in range(Self.size):
            out[i] = self[i] - other[i]
        return out^

    fn __isub__(mut self, other: Self):
        comptime for i in range(Self.size):
            self[i] -= other[i]

    fn __mul__(self, other: Self) -> Self:
        out = Self(uninitialized=True)
        comptime for i in range(Self.size):
            out[i] = self[i] * other[i]
        return out^

    fn __imul__(mut self, other: Self):
        comptime for i in range(Self.size):
            self[i] *= other[i]

    fn __truediv__(self, other: Self) -> Self:
        out = Self(uninitialized=True)
        comptime for i in range(Self.size):
            out[i] = self[i] / other[i]
        return out^

    # --- Scalar Operators ---
    fn __add__(self, s: Scalar[Self.dtype]) -> Self:
        out = Self(uninitialized=True)
        comptime for i in range(Self.size):
            out[i] = self[i] + s
        return out^

    fn __iadd__(mut self, s: Scalar[Self.dtype]):
        comptime for i in range(Self.size):
            self[i] += s

    fn __sub__(self, s: Scalar[Self.dtype]) -> Self:
        out = Self(uninitialized=True)
        comptime for i in range(Self.size):
            out[i] = self[i] - s
        return out^

    fn __isub__(mut self, s: Scalar[Self.dtype]):
        comptime for i in range(Self.size):
            self[i] -= s

    fn __mul__(self, s: Scalar[Self.dtype]) -> Self:
        out = Self(uninitialized=True)
        comptime for i in range(Self.size):
            out[i] = self[i] * s
        return out^

    fn __rmul__(self, s: Scalar[Self.dtype]) -> Self:
        return self * s

    fn __truediv__(self, s: Scalar[Self.dtype]) -> Self:
        out = Self(uninitialized=True)
        comptime for i in range(Self.size):
            out[i] = self[i] / s
        return out^

    fn __rtruediv__(self, s: Scalar[Self.dtype]) -> Self:
        out = Self(uninitialized=True)
        comptime for i in range(Self.size):
            out[i] = s / self[i]
        return out^

    fn __and__(self, other: Self) -> Self:
        out = Self(uninitialized=True)
        comptime for i in range(Self.size):
            out[i] = self[i] & other[i]
        return out^

    fn __or__(self, other: Self) -> Self:
        out = Self(uninitialized=True)
        comptime for i in range(Self.size):
            out[i] = self[i] | other[i]
        return out^

    fn __xor__(self, other: Self) -> Self:
        out = Self(uninitialized=True)
        comptime for i in range(Self.size):
            out[i] = self[i] ^ other[i]
        return out^

    fn __lshift__(self, other: Self) -> Self:
        out = Self(uninitialized=True)
        comptime for i in range(Self.size):
            out[i] = self[i] << other[i]
        return out^

    fn __rshift__(self, other: Self) -> Self:
        out = Self(uninitialized=True)
        comptime for i in range(Self.size):
            out[i] = self[i] >> other[i]
        return out^

    fn near_zero[s: Scalar[Self.dtype] = 1e-8](self) -> Bool:
        nz = True
        comptime for i in range(Self.size):
            nz &= abs(self[i]) < s
        return nz


fn is_power_of_2(n: Int) -> Bool:
    return n > 0 and (n & (n - 1)) == 0


fn dot[
    dtype: DType, size: Int
](a: Vec[dtype, size], b: Vec[dtype, size]) -> Scalar[dtype]:
    comptime if is_power_of_2(size):
        aa = a.data.unsafe_ptr().load[width=size]()
        bb = b.data.unsafe_ptr().load[width=size]()
        return (aa * bb).reduce_add()
    else:
        res: Scalar[dtype] = 0

        comptime for i in range(size):
            res += a[i] * b[i]
        return res


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


fn cross[type: DType](a: Vec2[type], b: Vec2[type]) -> Scalar[type]:
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
    out = Vec[dtype, size](uninitialized=True)
    comptime for i in range(size):
        out.data[i] = min(a[i], b[i])
    return out^


fn vmax[
    dtype: DType, size: Int
](a: Vec[dtype, size], b: Vec[dtype, size]) -> Vec[dtype, size]:
    out = Vec[dtype, size](uninitialized=True)
    comptime for i in range(size):
        out.data[i] = max(a[i], b[i])
    return out^


fn longest_axis[dtype: DType](v: Vec3[dtype]) -> Int:
    """Returns the index of the longest component of the vector."""
    if v[0] > v[1] and v[0] > v[2]:
        return 0
    if v[1] > v[2]:
        return 1
    return 2


fn assert_vec_equal[
    s: Int
](a: Vec[DType.float32, s], b: Vec[DType.float32, s]) raises:
    comptime for i in range(s):
        assert_almost_equal(a[i], b[i], msg="i={}".format(i), atol=1e-5)


fn main():
    print("hello warp.vec")
    a = Vec3f32(1, 2, 3)
    b = Vec3f32(4, 5, 6)
    aa = Vec4f32(1, 2, 3, 2)
    bb = Vec4f32(4, 5, 6, 5)
    print(dot(a, b))
    print(dot(aa, bb))
