from std.utils.numerics import max_finite, min_finite

from bajo.core.mat import Mat33
from bajo.core.quat import Quaternion
from bajo.core.vec import Vec3, vmin, vmax

comptime AABB = AxisAlignedBoundingBox[DType.float32]


@fieldwise_init
struct AxisAlignedBoundingBox[dtype: DType, width: Int = 1](
    TrivialRegisterPassable, Writable
):
    var _min: Vec3[Self.dtype, Self.width]
    var _max: Vec3[Self.dtype, Self.width]

    def __init__(out self, a: Self, b: Self):
        self._min = vmin(a._min, b._min)
        self._max = vmax(a._max, b._max)

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

    @always_inline
    def edges(self) -> Vec3[Self.dtype, Self.width]:
        return self._max - self._min

    @always_inline
    def extent(self) -> Vec3[Self.dtype, Self.width]:
        return self.edges()

    def overlaps(self, o: Self) -> SIMD[DType.bool, Self.width]:
        return (
            self._min.x.lt(o._max.x)
            & o._min.x.lt(self._max.x)
            & self._min.y.lt(o._max.y)
            & o._min.y.lt(self._max.y)
            & self._min.z.lt(o._max.z)
            & o._min.z.lt(self._max.z)
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

    def apply_trs(
        self: AxisAlignedBoundingBox[Self.dtype],
        translation: Vec3[Self.dtype],
        rotation: Quaternion[Self.dtype],
        scale: Vec3[Self.dtype],
    ) -> AxisAlignedBoundingBox[
        Self.dtype
    ] where Self.dtype.is_floating_point():
        rot_mat = Mat33[Self.dtype].from_rotation_scale(rotation, scale)
        txfmed = AxisAlignedBoundingBox[Self.dtype](translation, translation)

        # use comptime to fully unroll
        # in bench_aabb, 3.3x faster then version without comptime
        # 18% faster then arvo version
        comptime for j in range(3):
            comptime for i in range(3):
                e = rot_mat[i][j] * self._min[j]
                f = rot_mat[i][j] * self._max[j]

                if e[0] < f[0]:
                    txfmed._min.add_axis[i](e)
                    txfmed._max.add_axis[i](f)
                else:
                    txfmed._min.add_axis[i](f)
                    txfmed._max.add_axis[i](e)
        return txfmed
