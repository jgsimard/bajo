from std.utils.numerics import max_finite, min_finite

from bajo.core.transform import Affine3
from bajo.core.vec import Vec3, vmin, vmax, Point3, GeoKind
from bajo.core.frame import Frame


@fieldwise_init
struct AxisAlignedBoundingBox[
    dtype: DType, frame: Frame, width: SIMDLength = 1
](TrivialRegisterPassable, Writable):
    comptime STRIDE = 6
    var _min: Point3[Self.dtype, Self.frame, Self.width]
    var _max: Point3[Self.dtype, Self.frame, Self.width]

    @staticmethod
    def invalid() -> Self:
        comptime flt_max = max_finite[Self.dtype]()
        comptime flt_min = min_finite[Self.dtype]()
        return Self(
            Point3[Self.dtype, Self.frame, Self.width](flt_max),
            Point3[Self.dtype, Self.frame, Self.width](flt_min),
        )

    @staticmethod
    def point(p: Point3[Self.dtype, Self.frame, Self.width]) -> Self:
        return Self(p, p)

    @staticmethod
    def merge(a: Self, b: Self) -> Self:
        return Self(
            vmin(a._min, b._min),
            vmax(a._max, b._max),
        )

    def surface_area(self) -> SIMD[Self.dtype, Self.width]:
        d = self._max - self._min
        return 2.0 * (d.x * d.y + d.x * d.z + d.y * d.z)

    comptime area = Self.surface_area

    def centroid(self) -> Point3[Self.dtype, Self.frame, Self.width]:
        return self._min.unsafe_add(self._max) * 0.5

    def clear(mut self):
        self = Self.invalid()

    def grow(mut self, *vs: Point3[Self.dtype, Self.frame, Self.width]):
        for v in vs:
            self._min = vmin(self._min, v)
            self._max = vmax(self._max, v)

    def grow(mut self, *others: Self):
        for other in others:
            self._min = vmin(self._min, other._min)
            self._max = vmax(self._max, other._max)

    def edges(self) -> Vec3[Self.dtype, Self.frame, Self.width]:
        return self._max - self._min

    def extent(self) -> Vec3[Self.dtype, Self.frame, Self.width]:
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

    def contains_point(
        self, p: Point3[Self.dtype, Self.frame, Self.width]
    ) -> SIMD[DType.bool, Self.width]:
        return (
            self._min.x.le(p.x)
            & p.x.le(self._max.x)
            & self._min.y.le(p.y)
            & p.y.le(self._max.y)
            & self._min.z.le(p.z)
            & p.z.le(self._max.z)
        )

    def apply_transform[
        To: Frame
    ](
        self, transform: Affine3[Self.dtype, Self.frame, To, Self.width]
    ) -> AxisAlignedBoundingBox[Self.dtype, To, Self.width]:
        var new_min = transform.translation[GeoKind.POINT]()
        var new_max = transform.translation[GeoKind.POINT]()

        # X column
        var c0 = Point3[Self.dtype, To, self.width](
            transform.m00, transform.m10, transform.m20
        )
        var c0_a = c0 * self._min.x
        var c0_b = c0 * self._max.x
        new_min += vmin(c0_a, c0_b)
        new_max += vmax(c0_a, c0_b)

        # Y column
        var c1 = Point3[Self.dtype, To, self.width](
            transform.m01, transform.m11, transform.m21
        )
        var c1_a = c1 * self._min.y
        var c1_b = c1 * self._max.y
        new_min += vmin(c1_a, c1_b)
        new_max += vmax(c1_a, c1_b)

        # Z column
        var c2 = Point3[Self.dtype, To, self.width](
            transform.m02, transform.m12, transform.m22
        )
        var c2_a = c2 * self._min.z
        var c2_b = c2 * self._max.z
        new_min += vmin(c2_a, c2_b)
        new_max += vmax(c2_a, c2_b)

        return AxisAlignedBoundingBox[Self.dtype, To, Self.width](
            new_min, new_max
        )

    @staticmethod
    def load6[
        origin: Origin
    ](ptr: UnsafePointer[Scalar[Self.dtype], origin], base: Int) -> Self:
        comptime assert Self.width == 1
        return Self(
            Point3[Self.dtype, Self.frame, Self.width].load(ptr, base),
            Point3[Self.dtype, Self.frame, Self.width].load(ptr, base + 3),
        )

    def store6[
        origin: Origin[mut=True]
    ](self, ptr: UnsafePointer[Scalar[Self.dtype], origin], base: Int):
        comptime assert Self.width == 1
        self._min.store(ptr, base)
        self._max.store(ptr, base + 3)

    def translate(
        self, translation: Vec3[Self.dtype, Self.frame, Self.width]
    ) -> Self:
        return Self(self._min + translation, self._max + translation)

    def unsafe_convert_frame[
        new_frame: Frame
    ](self) -> AxisAlignedBoundingBox[Self.dtype, new_frame, Self.width]:
        return AxisAlignedBoundingBox[Self.dtype, new_frame, Self.width](
            self._min.unsafe_convert[new_frame=new_frame](),
            self._max.unsafe_convert[new_frame=new_frame](),
        )
