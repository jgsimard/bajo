from std.utils.numerics import max_finite, min_finite
from std.math import abs

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

    def __init__(
        out self,
        p0: Point3[Self.dtype, Self.frame, Self.width],
        p1: Point3[Self.dtype, Self.frame, Self.width],
        p2: Point3[Self.dtype, Self.frame, Self.width],
    ):
        return Self(
            vmin(vmin(p0, p1), p2),
            vmax(vmax(p0, p1), p2),
        )

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

    @always_inline
    def is_finite(
        self,
    ) -> SIMD[.bool, Self.width] where Self.dtype.is_floating_point():
        comptime limit = max_finite[Self.dtype]()
        return (
            abs(self._min.x).le(limit)
            & abs(self._min.y).le(limit)
            & abs(self._min.z).le(limit)
            & abs(self._max.x).le(limit)
            & abs(self._max.y).le(limit)
            & abs(self._max.z).le(limit)
        )

    @always_inline
    def is_valid(
        self,
    ) -> SIMD[.bool, Self.width] where Self.dtype.is_floating_point():
        return (
            self.is_finite()
            & self._min.x.le(self._max.x)
            & self._min.y.le(self._max.y)
            & self._min.z.le(self._max.z)
        )

    @staticmethod
    def merge(a: Self, b: Self) -> Self:
        return Self(
            vmin(a._min, b._min),
            vmax(a._max, b._max),
        )

    def surface_area(self) -> SIMD[Self.dtype, Self.width]:
        var d = self._max - self._min
        return 2.0 * (d.x * d.y + d.x * d.z + d.y * d.z)

    def centroid(self) -> Point3[Self.dtype, Self.frame, Self.width]:
        return self._min.unsafe_add(self._max) * 0.5

    def clear(mut self):
        self = Self.invalid()

    def grow(mut self, v: Point3[Self.dtype, Self.frame, Self.width]):
        self._min = vmin(self._min, v)
        self._max = vmax(self._max, v)

    def grow(mut self, *vs: Point3[Self.dtype, Self.frame, Self.width]):
        for v in vs:
            self._min = vmin(self._min, v)
            self._max = vmax(self._max, v)

    def grow(mut self, *others: Self):
        for other in others:
            self._min = vmin(self._min, other._min)
            self._max = vmax(self._max, other._max)

    def extent(self) -> Vec3[Self.dtype, Self.frame, Self.width]:
        return self._max - self._min

    def overlaps(self, o: Self) -> SIMD[.bool, Self.width]:
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
    ) -> SIMD[.bool, Self.width]:
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
    def load6(data: ImmSpan[Scalar[Self.dtype], _], base: Int) -> Self:
        comptime assert Self.width == 1
        debug_assert["safe", _use_compiler_assume=True](
            base >= 0 and base <= len(data) - Self.STRIDE,
            "AABB load is outside the input span",
        )
        return Self(
            Point3[Self.dtype, Self.frame, Self.width](
                data.unsafe_get(base + 0),
                data.unsafe_get(base + 1),
                data.unsafe_get(base + 2),
            ),
            Point3[Self.dtype, Self.frame, Self.width](
                data.unsafe_get(base + 3),
                data.unsafe_get(base + 4),
                data.unsafe_get(base + 5),
            ),
        )

    def store6(self, data: MutSpan[Scalar[Self.dtype], _], base: Int):
        comptime assert Self.width == 1
        debug_assert["safe", _use_compiler_assume=True](
            base >= 0 and base <= len(data) - Self.STRIDE,
            "AABB store is outside the output span",
        )
        data.unsafe_get(base + 0) = self._min.x[0]
        data.unsafe_get(base + 1) = self._min.y[0]
        data.unsafe_get(base + 2) = self._min.z[0]
        data.unsafe_get(base + 3) = self._max.x[0]
        data.unsafe_get(base + 4) = self._max.y[0]
        data.unsafe_get(base + 5) = self._max.z[0]

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
