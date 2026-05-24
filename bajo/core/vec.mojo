from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.math import fma, sqrt
from std.testing import assert_almost_equal

from bajo.core.utils import fmin, fmax

comptime Vec2f32 = Vec2[DType.float32]
comptime Vec3f32 = Vec3[DType.float32]


@fieldwise_init
struct Vec2[dtype: DType, width: Int = 1](TrivialRegisterPassable, Writable):
    var x: SIMD[Self.dtype, Self.width]
    var y: SIMD[Self.dtype, Self.width]

    def __add__(self, rhs: Self) -> Self:
        return Self(
            self.x + rhs.x,
            self.y + rhs.y,
        )

    def __sub__(self, rhs: Self) -> Self:
        return Self(
            self.x - rhs.x,
            self.y - rhs.y,
        )

    def __mul__(self, rhs: Self) -> Self:
        return Self(
            self.x * rhs.x,
            self.y * rhs.y,
        )

    def __mul__(self, rhs: SIMD[Self.dtype, Self.width]) -> Self:
        return Self(
            self.x * rhs,
            self.y * rhs,
        )

    def __truediv__(self, rhs: SIMD[Self.dtype, Self.width]) -> Self:
        return Self(
            self.x / rhs,
            self.y / rhs,
        )

    def __neg__(self) -> Self:
        return Self(-self.x, -self.y)


@fieldwise_init
struct Vec3[dtype: DType, width: Int = 1](
    DevicePassable, TrivialRegisterPassable, Writable
):
    var x: SIMD[Self.dtype, Self.width]
    var y: SIMD[Self.dtype, Self.width]
    var z: SIMD[Self.dtype, Self.width]

    comptime device_type: AnyType = Self
    """The device-side type for this array."""

    def _to_device_type(self, target: MutOpaquePointer[_]):
        """Convert the host type object to a device_type and store it at the
        target address.

        Args:
            target: The target address to store the device type.
        """
        target.bitcast[Self.device_type]()[] = self.copy()

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        """Convert the host type object to a device_type and store it at the
        target address.

        Args:
            encoder: Target specific device type encoder.
            target: The target address to store the device type.
        """
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        """Gets the name of the host type (the one implementing this trait).

        Returns:
            The host type's name.
        """
        return String(
            "Vec3[",
            reflect[Scalar[Self.dtype]].name(),
            ", ",
            Self.width,
            "]",
        )

    def __init__(
        out self,
        x: Scalar[Self.dtype],
    ):
        self.x = SIMD[Self.dtype, Self.width](x)
        self.y = SIMD[Self.dtype, Self.width](x)
        self.z = SIMD[Self.dtype, Self.width](x)

    def __add__(self, rhs: Self) -> Self:
        return Self(
            self.x + rhs.x,
            self.y + rhs.y,
            self.z + rhs.z,
        )

    def __add__(self, rhs: SIMD[Self.dtype, Self.width]) -> Self:
        return Self(
            self.x + rhs,
            self.y + rhs,
            self.z + rhs,
        )

    def __iadd__(mut self, rhs: Self):
        self.x += rhs.x
        self.y += rhs.y
        self.z += rhs.z

    def __sub__(self, rhs: Self) -> Self:
        return Self(
            self.x - rhs.x,
            self.y - rhs.y,
            self.z - rhs.z,
        )

    def __sub__(self, rhs: SIMD[Self.dtype, Self.width]) -> Self:
        return Self(
            self.x - rhs,
            self.y - rhs,
            self.z - rhs,
        )

    def __mul__(self, rhs: Self) -> Self:
        return Self(
            self.x * rhs.x,
            self.y * rhs.y,
            self.z * rhs.z,
        )

    def __imul__(mut self, rhs: Self):
        self.x *= rhs.x
        self.y *= rhs.y
        self.z *= rhs.z

    def __mul__(self, rhs: SIMD[Self.dtype, Self.width]) -> Self:
        return Self(
            self.x * rhs,
            self.y * rhs,
            self.z * rhs,
        )

    def __rmul__(self, rhs: SIMD[Self.dtype, Self.width]) -> Self:
        return Self(
            self.x * rhs,
            self.y * rhs,
            self.z * rhs,
        )

    def __truediv__(self, rhs: SIMD[Self.dtype, Self.width]) -> Self:
        return Self(
            self.x / rhs,
            self.y / rhs,
            self.z / rhs,
        )

    def __truediv__(self, rhs: Self) -> Self:
        return Self(
            self.x / rhs.x,
            self.y / rhs.y,
            self.z / rhs.z,
        )

    def __neg__(self) -> Self:
        return Self(-self.x, -self.y, -self.z)

    def __eq__(self, rhs: Self) -> SIMD[DType.bool, Self.width]:
        return self.x.eq(rhs.x) & self.y.eq(rhs.y) & self.z.eq(rhs.z)

    def __rtruediv__(self, lhs: SIMD[Self.dtype, Self.width]) -> Self:
        return Self(
            lhs / self.x,
            lhs / self.y,
            lhs / self.z,
        )

    def __getitem__(self, i: Int) -> SIMD[Self.dtype, Self.width]:
        if i == 0:
            return self.x
        elif i == 1:
            return self.y
        else:
            return self.z

    def __getitem_param__[i: Int](self) -> SIMD[Self.dtype, Self.width]:
        comptime assert i >= 0 and i < 3

        comptime if i == 0:
            return self.x
        elif i == 1:
            return self.y
        else:
            return self.z

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

    def __setitem__[i: Int](mut self, value: SIMD[Self.dtype, Self.width]):
        comptime assert i >= 0 and i < 3

        comptime if i == 0:
            self.x = value
        elif i == 1:
            self.y = value
        else:
            self.z = value

    def set_axis[i: Int](mut self, value: SIMD[Self.dtype, Self.width]):
        comptime assert i >= 0 and i < 3

        comptime if i == 0:
            self.x = value
        elif i == 1:
            self.y = value
        else:
            self.z = value

    def add_axis[i: Int](mut self, value: SIMD[Self.dtype, Self.width]):
        comptime assert i >= 0 and i < 3

        comptime if i == 0:
            self.x += value
        elif i == 1:
            self.y += value
        else:
            self.z += value

    def is_near_zero(
        self,
        eps: Scalar[Self.dtype] = 1.0e-8,
    ) -> SIMD[DType.bool, Self.width]:
        var s = SIMD[Self.dtype, Self.width](eps)

        return abs(self.x).lt(s) & abs(self.y).lt(s) & abs(self.z).lt(s)

    def safe_inv(
        self,
        eps: Scalar[Self.dtype] = 1.0e-20,
    ) -> Self where Self.dtype.is_floating_point():
        comptime one = SIMD[Self.dtype, Self.width](1.0)
        comptime zero = SIMD[Self.dtype, Self.width](0.0)
        var e = SIMD[Self.dtype, Self.width](eps)

        var mx = abs(self.x).gt(e)
        var my = abs(self.y).gt(e)
        var mz = abs(self.z).gt(e)

        # avoid even forming 1.0 / 0.0 in masked-off lanes
        # necessary ?
        var dx = mx.select(self.x, one)
        var dy = my.select(self.y, one)
        var dz = mz.select(self.z, one)

        return Self(
            mx.select(one / dx, zero),
            my.select(one / dy, zero),
            mz.select(one / dz, zero),
        )

    @staticmethod
    def load[
        origin: Origin
    ](ptr: UnsafePointer[Scalar[Self.dtype], origin], base: Int) -> Self:
        comptime assert Self.width == 1
        return Self(
            ptr[base + 0],
            ptr[base + 1],
            ptr[base + 2],
        )

    def store[
        origin: Origin[mut=True]
    ](self, ptr: UnsafePointer[Scalar[Self.dtype], origin], base: Int):
        comptime assert Self.width == 1
        ptr[base + 0] = self.x[0]
        ptr[base + 1] = self.y[0]
        ptr[base + 2] = self.z[0]


def dot[
    dtype: DType, width: Int
](a: Vec3[dtype, width], b: Vec3[dtype, width]) -> SIMD[dtype, width]:
    return fma(a.x, b.x, fma(a.y, b.y, a.z * b.z))


def vmin[
    dtype: DType, width: Int
](a: Vec3[dtype, width], b: Vec3[dtype, width]) -> Vec3[dtype, width]:
    return Vec3[dtype, width](
        fmin(a.x, b.x),
        fmin(a.y, b.y),
        fmin(a.z, b.z),
    )


def vmin[
    dtype: DType, width: Int
](a: Vec3[dtype, width], b: Vec3[dtype, width], c: Vec3[dtype, width]) -> Vec3[
    dtype, width
]:
    return Vec3[dtype, width](
        fmin(fmin(a.x, b.x), c.x),
        fmin(fmin(a.y, b.y), c.y),
        fmin(fmin(a.z, b.z), c.z),
    )


def vmax[
    dtype: DType, width: Int
](a: Vec3[dtype, width], b: Vec3[dtype, width]) -> Vec3[dtype, width]:
    return Vec3[dtype, width](
        fmax(a.x, b.x),
        fmax(a.y, b.y),
        fmax(a.z, b.z),
    )


def vmax[
    dtype: DType, width: Int
](a: Vec3[dtype, width], b: Vec3[dtype, width], c: Vec3[dtype, width]) -> Vec3[
    dtype, width
]:
    return Vec3[dtype, width](
        fmax(fmax(a.x, b.x), c.x),
        fmax(fmax(a.y, b.y), c.y),
        fmax(fmax(a.z, b.z), c.z),
    )


def vclamp[
    dtype: DType, width: Int
](
    p: Vec3[dtype, width],
    lower: Vec3[dtype, width],
    upper: Vec3[dtype, width],
) -> Vec3[dtype, width]:
    return vmin(vmax(p, lower), upper)


def length2[
    dtype: DType, width: Int
](v: Vec3[dtype, width]) -> SIMD[dtype, width]:
    return dot(v, v)


def length[
    dtype: DType, width: Int
](v: Vec3[dtype, width]) -> SIMD[dtype, width]:
    return sqrt(dot(v, v))


def cross[
    dtype: DType, width: Int
](a: Vec3[dtype, width], b: Vec3[dtype, width]) -> Vec3[dtype, width]:
    return Vec3[dtype, width](
        fma(a.y, b.z, -(a.z * b.y)),
        fma(a.z, b.x, -(a.x * b.z)),
        fma(a.x, b.y, -(a.y * b.x)),
    )


def normalize[
    dtype: DType, width: Int
](v: Vec3[dtype, width], threshold: Scalar[dtype] = 1.0e-20) -> Vec3[
    dtype, width
]:
    comptime assert dtype in [DType.float32, DType.float64]

    var l = length(v)
    var mask = l.gt(threshold)
    var inv_l = mask.select(1.0 / l, 0.0)
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
