from std.math import sqrt, min, max, clamp
from std.testing import assert_almost_equal
from std.sys import CompilationTarget

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
struct Vec[dtype: DType, size: Int](Copyable, Equatable, Roundable, Writable):
    # TODO: maybe use psize for SIMD for Vec3 on cpu
    # comptime psize = 4 if Self.size == 3 and CompilationTarget.is_x86() else Self.size
    comptime S = Scalar[Self.dtype]
    comptime T = InlineArray[Self.S, Self.size]
    var data: Self.T

    fn __init__(out self, s: Self.S):
        comptime assert Self.size > 0
        self.data = Self.T(fill=s)

    fn __init__(out self, uninitialized: Bool):
        comptime assert Self.size > 0
        self.data = Self.T(uninitialized=uninitialized)

    fn __init__(out self, var *elems: Self.S):
        debug_assert(
            len(elems) == Self.size, "No. of elems must match array size"
        )
        self.data = Self.T(storage=elems^)

    # TODO: should we keep this ?
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
        self.data = [x, y, z]

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
        self.data = [v.x(), v.y(), v.z(), w.x(), w.y(), w.z()]

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
    @always_inline
    fn __getitem__[I: Indexer, //, idx: I](ref self) -> ref[self.data] Self.S:
        """With compile-time bounds checking."""
        return self.data[materialize[idx]()]

    @always_inline
    fn __getitem__[I: Indexer](ref self, idx: I) -> ref[self.data] Self.S:
        # return self.data[idx] # bounds checking
        return self.data.unsafe_get(idx)  # no bounds checking

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
        out = Self(uninitialized=True)
        comptime for i in range(Self.size):
            out[i] = -self[i]
        return out^

    fn __add__(self, other: Self) -> Self:
        return _vv[Self.S.__add__](self, other)

    fn __sub__(self, other: Self) -> Self:
        return _vv[Self.S.__sub__](self, other)

    fn __mul__(self, other: Self) -> Self:
        return _vv[Self.S.__mul__](self, other)

    fn __truediv__(self, other: Self) -> Self:
        return _vv[Self.S.__truediv__](self, other)

    fn __and__(self, other: Self) -> Self:
        return _vv[Self.S.__and__](self, other)

    fn __or__(self, other: Self) -> Self:
        return _vv[Self.S.__or__](self, other)

    fn __xor__(self, other: Self) -> Self:
        return _vv[Self.S.__xor__](self, other)

    fn __lshift__(self, other: Self) -> Self:
        return _vv[Self.S.__lshift__](self, other)

    fn __rshift__(self, other: Self) -> Self:
        return _vv[Self.S.__rshift__](self, other)

    fn __round__(self) -> Self:
        res = Self(uninitialized=True)
        comptime for i in range(Self.size):
            res[i] = round(self[i])
        return res^

    fn __round__(self, ndigits: Int) -> Self:
        res = Self(uninitialized=True)
        comptime for i in range(Self.size):
            res[i] = round(self[i], ndigits)
        return res^

    fn __invert__(self) -> Self:
        out = Self(uninitialized=True)
        comptime for i in range(Self.size):
            out[i] = ~self[i]
        return out^

    # Scalar inplace
    fn __iadd__(mut self, s: Self.S):
        comptime for i in range(Self.size):
            self[i] += s

    fn __isub__(mut self, s: Self.S):
        comptime for i in range(Self.size):
            self[i] -= s

    fn __imul__(mut self, s: Self.S):
        comptime for i in range(Self.size):
            self[i] *= s

    # Scalar methods
    fn __add__(self, s: Self.S) -> Self:
        return _vs[Self.S.__add__](self, s)

    fn __sub__(self, s: Self.S) -> Self:
        return _vs[Self.S.__sub__](self, s)

    fn __mul__(self, s: Self.S) -> Self:
        return _vs[Self.S.__mul__](self, s)

    fn __rmul__(self, s: Self.S) -> Self:
        return self * s

    fn __truediv__(self, s: Self.S) -> Self:
        return _vs[Self.S.__truediv__](self, s)

    fn __rtruediv__(self, s: Self.S) -> Self:
        return _vs[Self.S.__rtruediv__](self, s)

    fn __and__(self, s: Self.S) -> Self:
        return _vs[Self.S.__and__](self, s)

    fn __or__(self, s: Self.S) -> Self:
        return _vs[Self.S.__or__](self, s)

    fn __xor__(self, s: Self.S) -> Self:
        return _vs[Self.S.__xor__](self, s)

    fn __lshift__(self, s: Self.S) -> Self:
        return _vs[Self.S.__lshift__](self, s)

    fn __rshift__(self, s: Self.S) -> Self:
        return _vs[Self.S.__rshift__](self, s)

    fn __eq__(self, other: Self) -> Bool:
        eq = True
        comptime for i in range(Self.size):
            eq &= self[i] == other[i]
        return eq

    # property
    fn is_near_zero[s: Self.S = 1e-8](self) -> Bool:
        nz = True
        comptime for i in range(Self.size):
            nz &= abs(self[i]) < s
        return nz


@always_inline
fn _vv[
    dtype: DType,
    size: Int,
    //,
    func: fn(Scalar[dtype], Scalar[dtype]) -> Scalar[dtype],
](lhs: Vec[dtype, size], rhs: Vec[dtype, size]) -> Vec[dtype, size]:
    out = Vec[dtype, size](uninitialized=True)
    comptime for i in range(size):
        out[i] = func(lhs[i], rhs[i])
    return out^


@always_inline
fn _vs[
    dtype: DType,
    size: Int,
    //,
    func: fn(Scalar[dtype], Scalar[dtype]) -> Scalar[dtype],
](v: Vec[dtype, size], s: Scalar[dtype]) -> Vec[dtype, size]:
    out = Vec[dtype, size](uninitialized=True)
    comptime for i in range(size):
        out[i] = func(v[i], s[i])
    return out^


fn is_power_of_2(n: Int) -> Bool:
    return n > 0 and (n & (n - 1)) == 0


fn dot[
    dtype: DType, size: Int
](a: Vec[dtype, size], b: Vec[dtype, size]) -> Scalar[dtype]:
    comptime if is_power_of_2(size):
        aa = a.data.unsafe_ptr().load[width=size]()
        bb = b.data.unsafe_ptr().load[width=size]()
        return (aa * bb).reduce_add()
    # # is this faster ?
    # elif size == 3:
    #     a3 = a.data.unsafe_ptr().load[width=4]()
    #     a3[3] = 0
    #     b3 = b.data.unsafe_ptr().load[width=4]()
    #     b3[3] = 0
    #     return (a3 * b3).reduce_add()
    # elif size == 3:
    #     a3 = a.data.unsafe_ptr().load[width=4]()
    #     b3 = b.data.unsafe_ptr().load[width=4]()
    #     return (a3 * b3).shift_rigt[1].reduce_add()
    # # mabye thie version ??
    #     comptime np2 = next_power_of_two(size)
    #     comptime pad = np2 - size
    #     aa = a.data.unsafe_ptr().load[width=np2]()
    #     ba = b.data.unsafe_ptr().load[width=np2]()
    #     return (a3 * b3).shift_rigt[pad].reduce_add()
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


fn vclamp[
    dtype: DType, size: Int
](
    v: Vec[dtype, size], lower_bound: Scalar[dtype], upper_bound: Scalar[dtype]
) -> Vec[dtype, size]:
    out = Vec[dtype, size](uninitialized=True)
    comptime for i in range(size):
        out.data[i] = clamp(v[i], lower_bound, upper_bound)
    return out^


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
        assert_almost_equal(a[i], b[i], msg=String(t"i={i}"), atol=atol)


fn main():
    print("hello warp.vec")
    a = Vec3f32(1, 2, 3)
    b = Vec3f32(4, 5, 6)
    aa = Vec4f32(1, 2, 3, 2)
    bb = Vec4f32(4, 5, 6, 5)
    print(dot(a, b))
    print(dot(aa, bb))
