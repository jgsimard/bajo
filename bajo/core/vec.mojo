from std.math import sqrt, min, max, clamp
from std.sys import CompilationTarget
from std.testing import assert_almost_equal

from bajo.core.utils import is_power_of_2

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

    def __init__(out self, s: Self.S):
        comptime assert Self.size > 0
        self.data = Self.T(fill=s)

    def __init__(out self, uninitialized: Bool):
        comptime assert Self.size > 0
        self.data = Self.T(uninitialized=uninitialized)

    def __init__(
        out self: Vec[Self.dtype, 2],
        x: Self.S,
        y: Self.S,
    ):
        self.data = [x, y]

    def __init__(
        out self: Vec[Self.dtype, 3],
        x: Self.S,
        y: Self.S,
        z: Self.S,
    ):
        self.data = [x, y, z]

    def __init__(
        out self: Vec[Self.dtype, 4],
        x: Self.S,
        y: Self.S,
        z: Self.S,
        w: Self.S,
    ):
        self.data = [x, y, z, w]

    def __init__(
        out self: Vec[Self.dtype, 6],
        v: Vec3[Self.dtype],
        w: Vec3[Self.dtype],
    ):
        self.data = [v.x(), v.y(), v.z(), w.x(), w.y(), w.z()]

    def x(self) -> Self.S:
        return self.data[0]

    def y(self) -> Self.S:
        return self.data[1]

    def z(self) -> Self.S:
        comptime assert Self.size >= 3
        return self.data[2]

    def w(self) -> Self.S:
        comptime assert Self.size >= 4
        return self.data[3]

    # Swizzles
    def xy(self) -> Vec2[Self.dtype]:
        return Vec2[self.dtype](self.x(), self.y())

    def yz(self) -> Vec2[Self.dtype]:
        comptime assert Self.size >= 3
        return Vec2[self.dtype](self.y(), self.z())

    def xz(self) -> Vec2[Self.dtype]:
        comptime assert Self.size >= 3
        return Vec2[self.dtype](self.x(), self.z())

    def xyz(self) -> Vec3[Self.dtype]:
        comptime assert Self.size >= 3
        return Vec3[Self.dtype](self.x(), self.y(), self.z())

    # accessors
    @always_inline
    def __getitem_param__[idx: Int](ref self) -> ref[self.data] Self.S:
        """With compile-time bounds checking."""
        return self.data[idx]

    @always_inline
    def __getitem__(ref self, idx: Int) -> ref[self.data] Self.S:
        # return self.data[idx] # bounds checking
        return self.data.unsafe_get(idx)  # no bounds checking

    # inplace modifier
    def __iadd__(mut self, other: Self):
        comptime for i in range(Self.size):
            self[i] += other[i]

    def __isub__(mut self, other: Self):
        comptime for i in range(Self.size):
            self[i] -= other[i]

    def __imul__(mut self, other: Self):
        comptime for i in range(Self.size):
            self[i] *= other[i]

    # methods (produce new Self)
    def __neg__(self) -> Self:
        out = Self(uninitialized=True)
        comptime for i in range(Self.size):
            out[i] = -self[i]
        return out^

    def __add__(self, other: Self) -> Self:
        return _vv[Self.S.__add__](self, other)

    def __sub__(self, other: Self) -> Self:
        return _vv[Self.S.__sub__](self, other)

    def __mul__(self, other: Self) -> Self:
        return _vv[Self.S.__mul__](self, other)

    def __truediv__(self, other: Self) -> Self:
        return _vv[Self.S.__truediv__](self, other)

    def __and__(self, other: Self) -> Self:
        return _vv[Self.S.__and__](self, other)

    def __or__(self, other: Self) -> Self:
        return _vv[Self.S.__or__](self, other)

    def __xor__(self, other: Self) -> Self:
        return _vv[Self.S.__xor__](self, other)

    def __lshift__(self, other: Self) -> Self:
        return _vv[Self.S.__lshift__](self, other)

    def __rshift__(self, other: Self) -> Self:
        return _vv[Self.S.__rshift__](self, other)

    def __round__(self) -> Self:
        res = Self(uninitialized=True)
        comptime for i in range(Self.size):
            res[i] = round(self[i])
        return res^

    def __round__(self, ndigits: Int) -> Self:
        res = Self(uninitialized=True)
        comptime for i in range(Self.size):
            res[i] = round(self[i], ndigits)
        return res^

    def __invert__(self) -> Self:
        out = Self(uninitialized=True)
        comptime for i in range(Self.size):
            out[i] = ~self[i]
        return out^

    # Scalar inplace
    def __iadd__(mut self, s: Self.S):
        comptime for i in range(Self.size):
            self[i] += s

    def __isub__(mut self, s: Self.S):
        comptime for i in range(Self.size):
            self[i] -= s

    def __imul__(mut self, s: Self.S):
        comptime for i in range(Self.size):
            self[i] *= s

    # Scalar methods
    def __add__(self, s: Self.S) -> Self:
        return _vs[Self.S.__add__](self, s)

    def __sub__(self, s: Self.S) -> Self:
        return _vs[Self.S.__sub__](self, s)

    def __mul__(self, s: Self.S) -> Self:
        return _vs[Self.S.__mul__](self, s)

    def __rmul__(self, s: Self.S) -> Self:
        return self * s

    def __truediv__(self, s: Self.S) -> Self:
        return _vs[Self.S.__truediv__](self, s)

    def __rtruediv__(self, s: Self.S) -> Self:
        return _vs[Self.S.__rtruediv__](self, s)

    def __and__(self, s: Self.S) -> Self:
        return _vs[Self.S.__and__](self, s)

    def __or__(self, s: Self.S) -> Self:
        return _vs[Self.S.__or__](self, s)

    def __xor__(self, s: Self.S) -> Self:
        return _vs[Self.S.__xor__](self, s)

    def __lshift__(self, s: Self.S) -> Self:
        return _vs[Self.S.__lshift__](self, s)

    def __rshift__(self, s: Self.S) -> Self:
        return _vs[Self.S.__rshift__](self, s)

    def __eq__(self, other: Self) -> Bool:
        eq = True
        comptime for i in range(Self.size):
            eq &= self[i] == other[i]
        return eq

    # property
    def is_near_zero[s: Self.S = 1e-8](self) -> Bool:
        nz = True
        comptime for i in range(Self.size):
            nz &= abs(self[i]) < s
        return nz


@always_inline
def _vv[
    dtype: DType,
    size: Int,
    //,
    func: def(Scalar[dtype], Scalar[dtype]) -> Scalar[dtype],
](lhs: Vec[dtype, size], rhs: Vec[dtype, size]) -> Vec[dtype, size]:
    out = Vec[dtype, size](uninitialized=True)
    comptime for i in range(size):
        out[i] = func(lhs[i], rhs[i])
    return out^


@always_inline
def _vs[
    dtype: DType,
    size: Int,
    //,
    func: def(Scalar[dtype], Scalar[dtype]) -> Scalar[dtype],
](v: Vec[dtype, size], s: Scalar[dtype]) -> Vec[dtype, size]:
    out = Vec[dtype, size](uninitialized=True)
    comptime for i in range(size):
        out[i] = func(v[i], s)
    return out^


def dot[
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


def length_sq[dtype: DType, size: Int](v: Vec[dtype, size]) -> Scalar[dtype]:
    return dot(v, v)


def length[dtype: DType, size: Int](v: Vec[dtype, size]) -> Scalar[dtype]:
    return sqrt(length_sq(v))


def normalize[dtype: DType, size: Int](v: Vec[dtype, size]) -> Vec[dtype, size]:
    l = length(v)
    if l > 1e-6:
        return v / l
    return Vec[dtype, size](0)


def cross[dtype: DType](a: Vec[dtype, 3], b: Vec[dtype, 3]) -> Vec[dtype, 3]:
    return Vec[dtype, 3](
        a.y() * b.z() - a.z() * b.y(),
        a.z() * b.x() - a.x() * b.z(),
        a.x() * b.y() - a.y() * b.x(),
    )


def cross[dtype: DType](a: Vec2[dtype], b: Vec2[dtype]) -> Scalar[dtype]:
    return a.x() * b.y() - a.y() * b.x()


def lerp[
    dtype: DType, size: Int
](a: Vec[dtype, size], b: Vec[dtype, size], u: Scalar[dtype]) -> Vec[
    dtype, size
]:
    return a * (1.0 - u) + b * u


def vmin[
    dtype: DType, size: Int
](a: Vec[dtype, size], *bs: Vec[dtype, size]) -> Vec[dtype, size]:
    out = a.copy()
    for b in bs:
        comptime for i in range(size):
            out.data[i] = min(out.data[i], b.data[i])
    return out^


def vmax[
    dtype: DType, size: Int
](a: Vec[dtype, size], *bs: Vec[dtype, size]) -> Vec[dtype, size]:
    out = a.copy()
    for b in bs:
        comptime for i in range(size):
            out.data[i] = max(out.data[i], b.data[i])
    return out^


def vclamp[
    dtype: DType, size: Int
](
    v: Vec[dtype, size], lower_bound: Scalar[dtype], upper_bound: Scalar[dtype]
) -> Vec[dtype, size]:
    out = Vec[dtype, size](uninitialized=True)
    comptime for i in range(size):
        out.data[i] = clamp(v[i], lower_bound, upper_bound)
    return out^


def longest_axis[dtype: DType](v: Vec3[dtype]) -> Int:
    """Returns the index of the longest component of the vector."""
    if v[0] > v[1] and v[0] > v[2]:
        return 0
    if v[1] > v[2]:
        return 1
    return 2


def assert_vec_equal[
    dtype: DType, size: Int
](a: Vec[dtype, size], b: Vec[dtype, size], atol: Float64 = 1e-5) raises:
    comptime for i in range(size):
        assert_almost_equal(a[i], b[i], msg=String(t"i={i}"), atol=atol)


def main():
    print("hello warp.vec")
    a = Vec3f32(1, 2, 3)
    b = Vec3f32(4, 5, 6)
    aa = Vec4f32(1, 2, 3, 2)
    bb = Vec4f32(4, 5, 6, 5)
    print(dot(a, b))
    print(dot(aa, bb))
