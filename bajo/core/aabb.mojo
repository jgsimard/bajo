from std.utils.numerics import max_finite, min_finite

from bajo.core.transform import Affine3
from bajo.core.quat import Quaternion
from bajo.core.vec import Vec3, vmin, vmax

comptime AABB = AxisAlignedBoundingBox[DType.float32]


@fieldwise_init
struct AxisAlignedBoundingBox[dtype: DType, width: Int = 1](
    TrivialRegisterPassable, Writable
):
    var _min: Vec3[Self.dtype, Self.width]
    var _max: Vec3[Self.dtype, Self.width]

    @staticmethod
    def invalid() -> Self:
        comptime flt_max = max_finite[Self.dtype]()
        comptime flt_min = min_finite[Self.dtype]()
        return Self(
            Vec3[Self.dtype, Self.width](flt_max),
            Vec3[Self.dtype, Self.width](flt_min),
        )

    @staticmethod
    def point(p: Vec3[Self.dtype, Self.width]) -> Self:
        return Self(p, p)

    @staticmethod
    def merge(a: Self, b: Self) -> Self:
        return Self(
            vmin(a._min, b._min),
            vmax(a._max, b._max),
        )

    def surface_area(self) -> SIMD[Self.dtype, self.width]:
        d = self._max - self._min
        return 2.0 * (d.x * d.y + d.x * d.z + d.y * d.z)

    comptime area = Self.surface_area

    def centroid(self) -> Vec3[Self.dtype, Self.width]:
        return (self._min + self._max) * 0.5

    def clear(mut self):
        self = Self.invalid()

    def grow(mut self, *vs: Vec3[Self.dtype, Self.width]):
        for v in vs:
            self._min = vmin(self._min, v)
            self._max = vmax(self._max, v)

    def grow(mut self, *others: Self):
        for other in others:
            self._min = vmin(self._min, other._min)
            self._max = vmax(self._max, other._max)

    def edges(self) -> Vec3[Self.dtype, Self.width]:
        return self._max - self._min

    def extent(self) -> Vec3[Self.dtype, Self.width]:
        return self.edges()

    def overlaps(self, o: Self) -> SIMD[DType.bool, Self.width]:
        return (
            self._min.x.le(o._max.x)
            & o._min.x.le(self._max.x)
            & self._min.y.le(o._max.y)
            & o._min.y.le(self._max.y)
            & self._min.z.le(o._max.z)
            & o._min.z.le(self._max.z)
        )

    def contains(
        self, p: Vec3[Self.dtype, Self.width]
    ) -> SIMD[DType.bool, Self.width]:
        return (
            self._min.x.le(p.x)
            & p.x.le(self._max.x)
            & self._min.y.le(p.y)
            & p.y.le(self._max.y)
            & self._min.z.le(p.z)
            & p.z.le(self._max.z)
        )

    def apply_transform(
        self, transform: Affine3[Self.dtype, Self.width]
    ) -> AxisAlignedBoundingBox[Self.dtype, Self.width]:
        var new_min = transform.translation()
        var new_max = transform.translation()

        # X column
        var c0 = Vec3(transform.m00, transform.m10, transform.m20)
        var c0_a = c0 * self._min.x
        var c0_b = c0 * self._max.x
        new_min += vmin(c0_a, c0_b)
        new_max += vmax(c0_a, c0_b)

        # Y column
        var c1 = Vec3(transform.m01, transform.m11, transform.m21)
        var c1_a = c1 * self._min.y
        var c1_b = c1 * self._max.y
        new_min += vmin(c1_a, c1_b)
        new_max += vmax(c1_a, c1_b)

        # Z column
        var c2 = Vec3(transform.m02, transform.m12, transform.m22)
        var c2_a = c2 * self._min.z
        var c2_b = c2 * self._max.z
        new_min += vmin(c2_a, c2_b)
        new_max += vmax(c2_a, c2_b)

        return AxisAlignedBoundingBox[Self.dtype, Self.width](new_min, new_max)

    @staticmethod
    def load6[
        origin: Origin
    ](ptr: UnsafePointer[Scalar[Self.dtype], origin], base: Int) -> Self:
        comptime assert Self.width == 1
        return Self(
            Vec3[Self.dtype, Self.width].load(ptr, base),
            Vec3[Self.dtype, Self.width].load(ptr, base + 3),
        )

    def store6[
        origin: Origin[mut=True]
    ](self, ptr: UnsafePointer[Scalar[Self.dtype], origin], base: Int):
        comptime assert Self.width == 1
        self._min.store(ptr, base)
        self._max.store(ptr + 3, base)

    def translate(self, translation: Vec3[Self.dtype, Self.width]) -> Self:
        return Self(self._min + translation, self._max + translation)
