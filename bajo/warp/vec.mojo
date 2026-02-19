from math import sqrt

comptime Vec2i8 = Vec[DType.int8, 2]
comptime Vec3i8 = Vec[DType.int8, 3]
comptime Vec4i8 = Vec[DType.int8, 4]

comptime Vec2u8 = Vec[DType.uint8, 2]
comptime Vec3u8 = Vec[DType.uint8, 3]
comptime Vec4u8 = Vec[DType.uint8, 4]

comptime Vec2i16 = Vec[DType.int16, 2]
comptime Vec3i16 = Vec[DType.int16, 3]
comptime Vec4i16 = Vec[DType.int16, 4]

comptime Vec2u16 = Vec[DType.uint16, 2]
comptime Vec3u16 = Vec[DType.uint16, 3]
comptime Vec4u16 = Vec[DType.uint16, 4]

comptime Vec2i32 = Vec[DType.int32, 2]
comptime Vec3i32 = Vec[DType.int32, 3]
comptime Vec4i32 = Vec[DType.int32, 4]

comptime Vec2u32 = Vec[DType.uint32, 2]
comptime Vec3u32 = Vec[DType.uint32, 3]
comptime Vec4u32 = Vec[DType.uint32, 4]

comptime Vec2i64 = Vec[DType.int64, 2]
comptime Vec3i64 = Vec[DType.int64, 3]
comptime Vec4i64 = Vec[DType.int64, 4]

comptime Vec2u64 = Vec[DType.uint64, 2]
comptime Vec3u64 = Vec[DType.uint64, 3]
comptime Vec4u64 = Vec[DType.uint64, 4]

comptime Vec2f16 = Vec[DType.float16, 2]
comptime Vec3f16 = Vec[DType.float16, 3]
comptime Vec4f16 = Vec[DType.float16, 4]

comptime Vec2 = Vec[_, 2]
comptime Vec3 = Vec[_, 3]
comptime Vec4 = Vec[_, 4]

comptime Vec2f32 = Vec[DType.float32, 2]
comptime Vec3f32 = Vec[DType.float32, 3]
comptime Vec4f32 = Vec[DType.float32, 4]

comptime Vec2f64 = Vec[DType.float64, 2]
comptime Vec3f64 = Vec[DType.float64, 3]
comptime Vec4f64 = Vec[DType.float64, 4]


@fieldwise_init
struct Vec[dtype: DType, size: Int](Copyable, Writable):
    var data: InlineArray[Scalar[Self.dtype], Self.size]

    fn __init__(out self, s: Scalar[Self.dtype]):
        self.data = InlineArray[Scalar[Self.dtype], Self.size](fill=s)

    fn __init__(out self, uninitialized: Bool):
        self.data = InlineArray[Scalar[Self.dtype], Self.size](
            uninitialized=uninitialized
        )

    fn __init__(
        out self: Vec[Self.dtype, 2],
        x: Scalar[Self.dtype],
        y: Scalar[Self.dtype],
    ):
        self.data = [x, y]

    fn __init__(
        out self: Vec[Self.dtype, 3],
        x: Scalar[Self.dtype],
        y: Scalar[Self.dtype],
        z: Scalar[Self.dtype],
    ):
        self.data = [x, y, z]

    fn __init__(
        out self: Vec[Self.dtype, 4],
        v: Vec[Self.dtype, 3],
        w: Vec[Self.dtype, 3],
    ):
        self.data = [v.x(), v.y(), v.z(), w.x(), w.y(), w.z()]

    fn __init__(
        out self: Vec[Self.dtype, 6],
        x: Scalar[Self.dtype],
        y: Scalar[Self.dtype],
        z: Scalar[Self.dtype],
        w: Scalar[Self.dtype],
    ):
        self.data = [x, y, z, w]

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
        var data_out = InlineArray[Scalar[Self.dtype], Self.size](
            uninitialized=True
        )

        @parameter
        for i in range(Self.size):
            data_out[i] = -self.data[i]
        return Self(data_out^)

    fn __add__(self, other: Self) -> Self:
        var data_out = InlineArray[Scalar[Self.dtype], Self.size](
            uninitialized=True
        )

        @parameter
        for i in range(Self.size):
            data_out[i] = self[i] + other[i]
        return Self(data_out^)

    fn __sub__(self, other: Self) -> Self:
        var data_out = InlineArray[Scalar[Self.dtype], Self.size](
            uninitialized=True
        )

        @parameter
        for i in range(Self.size):
            data_out[i] = self[i] - other[i]
        return Self(data_out^)

    fn __mul__(self, other: Self) -> Self:
        var data_out = InlineArray[Scalar[Self.dtype], Self.size](
            uninitialized=True
        )

        @parameter
        for i in range(Self.size):
            data_out[i] = self[i] * other[i]
        return Self(data_out^)

    fn __truediv__(self, other: Self) -> Self:
        var data_out = InlineArray[Scalar[Self.dtype], Self.size](
            uninitialized=True
        )

        @parameter
        for i in range(Self.size):
            data_out[i] = self[i] / other[i]
        return Self(data_out^)

    # --- Scalar Operators ---
    fn __add__(self, s: Scalar[Self.dtype]) -> Self:
        var data_out = InlineArray[Scalar[Self.dtype], Self.size](
            uninitialized=True
        )

        @parameter
        for i in range(Self.size):
            data_out[i] = self[i] + s
        return Self(data_out^)

    fn __sub__(self, s: Scalar[Self.dtype]) -> Self:
        var data_out = InlineArray[Scalar[Self.dtype], Self.size](
            uninitialized=True
        )

        @parameter
        for i in range(Self.size):
            data_out[i] = self[i] - s
        return Self(data_out^)

    fn __mul__(self, s: Scalar[Self.dtype]) -> Self:
        var data_out = InlineArray[Scalar[Self.dtype], Self.size](
            uninitialized=True
        )

        @parameter
        for i in range(Self.size):
            data_out[i] = self[i] * s
        return Self(data_out^)

    fn __rmul__(self, s: Scalar[Self.dtype]) -> Self:
        return self * s

    fn __truediv__(self, s: Scalar[Self.dtype]) -> Self:
        var data_out = InlineArray[Scalar[Self.dtype], Self.size](
            uninitialized=True
        )

        @parameter
        for i in range(Self.size):
            data_out[i] = self[i] / s
        return Self(data_out^)

    fn __rtruediv__(self, s: Scalar[Self.dtype]) -> Self:
        var data_out = InlineArray[Scalar[Self.dtype], Self.size](
            uninitialized=True
        )

        @parameter
        for i in range(Self.size):
            data_out[i] = s / self[i]
        return Self(data_out^)

    fn __and__(self, other: Self) -> Self:
        var data_out = InlineArray[Scalar[Self.dtype], Self.size](
            uninitialized=True
        )

        @parameter
        for i in range(Self.size):
            data_out[i] = self[i] & other[i]
        return Self(data_out^)

    fn __or__(self, other: Self) -> Self:
        var data_out = InlineArray[Scalar[Self.dtype], Self.size](
            uninitialized=True
        )

        @parameter
        for i in range(Self.size):
            data_out[i] = self[i] | other[i]
        return Self(data_out^)

    fn __xor__(self, other: Self) -> Self:
        var data_out = InlineArray[Scalar[Self.dtype], Self.size](
            uninitialized=True
        )

        @parameter
        for i in range(Self.size):
            data_out[i] = self[i] ^ other[i]
        return Self(data_out^)

    fn __lshift__(self, other: Self) -> Self:
        var data_out = InlineArray[Scalar[Self.dtype], Self.size](
            uninitialized=True
        )

        @parameter
        for i in range(Self.size):
            data_out[i] = self[i] << other[i]
        return Self(data_out^)

    fn __rshift__(self, other: Self) -> Self:
        var data_out = InlineArray[Scalar[Self.dtype], Self.size](
            uninitialized=True
        )

        @parameter
        for i in range(Self.size):
            data_out[i] = self[i] >> other[i]
        return Self(data_out^)


fn dot[
    dtype: DType, size: Int
](a: Vec[dtype, size], b: Vec[dtype, size]) -> Scalar[dtype]:
    var res: Scalar[dtype] = 0

    @parameter
    for i in range(size):
        res += a[i] * b[i]
    return res


fn length_sq[dtype: DType, size: Int](v: Vec[dtype, size]) -> Scalar[dtype]:
    return dot(v, v)


fn length[dtype: DType, size: Int](v: Vec[dtype, size]) -> Scalar[dtype]:
    return sqrt(length_sq(v))


fn normalize[dtype: DType, size: Int](v: Vec[dtype, size]) -> Vec[dtype, size]:
    var l = length(v)
    if l > 1e-6:
        return v / l
    return Vec[dtype, size](0)


fn cross[dtype: DType](a: Vec[dtype, 3], b: Vec[dtype, 3]) -> Vec[dtype, 3]:
    return Vec[dtype, 3](
        a.y() * b.z() - a.z() * b.y(),
        a.z() * b.x() - a.x() * b.z(),
        a.x() * b.y() - a.y() * b.x(),
    )


fn min[
    dtype: DType, size: Int
](a: Vec[dtype, size], b: Vec[dtype, size]) -> Vec[dtype, size]:
    var data_out = InlineArray[Scalar[dtype], size](uninitialized=True)

    @parameter
    for i in range(size):
        data_out[i] = a[i] if a[i] < b[i] else b[i]
    return Vec[dtype, size](data_out^)


fn max[
    dtype: DType, size: Int
](a: Vec[dtype, size], b: Vec[dtype, size]) -> Vec[dtype, size]:
    var data_out = InlineArray[Scalar[dtype], size](uninitialized=True)

    @parameter
    for i in range(size):
        data_out[i] = a[i] if a[i] > b[i] else b[i]
    return Vec[dtype, size](data_out^)


fn main():
    print("hello warp.vec")
