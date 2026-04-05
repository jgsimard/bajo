from std.utils.numerics import max_finite, min_finite

from bajo.core.mat import Mat33
from bajo.core.quat import Quaternion
from bajo.core.vec import Vec, Vec3, vmin, vmax

comptime AABB = AxisAlignedBoundingBox[DType.float32]


@fieldwise_init
struct AxisAlignedBoundingBox[dtype: DType, size: Int = 3](Copyable, Writable):
    var _min: Vec[Self.dtype, Self.size]
    var _max: Vec[Self.dtype, Self.size]

    def __init__(out self, a: Self, b: Self):
        self._min = vmin(a._min, b._min)
        self._max = vmax(a._max, b._max)

    @staticmethod
    def invalid() -> Self:
        comptime flt_max = max_finite[Self.dtype]()
        comptime flt_min = min_finite[Self.dtype]()
        return Self(
            Vec[Self.dtype, Self.size](flt_max),
            Vec[Self.dtype, Self.size](flt_min),
        )

    @staticmethod
    def point(p: Vec[Self.dtype, Self.size]) -> Self:
        return Self(p.copy(), p.copy())

    @staticmethod
    def merge(a: Self, b: Self) -> Self:
        return Self(
            vmin(a._min, b._min),
            vmax(a._max, b._max),
        )

    def surface_area(self) -> Scalar[Self.dtype]:
        d = self._max - self._min
        return d.x() * d.y() + d.x() * d.z() + d.y() * d.z()

    def centroid(self) -> Vec[Self.dtype, Self.size]:
        return (self._min + self._max) * 0.5

    def area(self) -> Scalar[Self.dtype]:
        diff = self._max - self._min
        return diff.x() * diff.y() + diff.y() * diff.z() + diff.z() * diff.x()

    def clear(mut self):
        comptime _min = min_finite[Self.dtype]()
        comptime _max = max_finite[Self.dtype]()
        self._min = Vec[Self.dtype, Self.size](_min)
        self._max = Vec[Self.dtype, Self.size](_max)

    def grow(mut self, *vs: Vec[Self.dtype, Self.size]):
        for v in vs:
            self._min = vmin(self._min, v)
            self._max = vmax(self._max, v)

    def grow(mut self, *others: Self):
        for other in others:
            self._min = vmin(self._min, other._min)
            self._max = vmax(self._max, other._max)

    def edges(self) -> Vec[Self.dtype, Self.size]:
        return self._max - self._min

    def overlaps(self, o: Self) -> Bool:
        return (
            self._min.x() < o._max.x()
            and o._min.x() < self._max.x()
            and self._min.y() < o._max.y()
            and o._min.y() < self._max.y()
            and self._min.z() < o._max.z()
            and o._min.z() < self._max.z()
        )

    def contains(self, p: Vec[Self.dtype, Self.size]) -> Bool:
        return (
            self._min.x() <= p.x()
            and self._min.y() <= p.y()
            and self._min.z() <= p.z()
            and self._max.x() >= p.x()
            and self._max.y() >= p.y()
            and self._max.z() >= p.z()
        )

    def ray_intersects(
        self,
        ray_o: Vec[Self.dtype, Self.size],
        inv_ray_d: Vec[Self.dtype, Self.size],
        ray_t_min: Scalar[Self.dtype],
        ray_t_max: Scalar[Self.dtype],
    ) -> Bool:
        t_lower = inv_ray_d * (self._min - ray_o)
        t_upper = inv_ray_d * (self._max - ray_o)

        t_min_vec = vmin(t_lower, t_upper)
        t_max_vec = vmax(t_lower, t_upper)

        t_box_min = max(t_min_vec.x(), t_min_vec.y(), t_min_vec.z(), ray_t_min)

        t_box_max = min(t_max_vec.x(), t_max_vec.y(), t_max_vec.z(), ray_t_max)

        return t_box_min <= t_box_max

    def apply_trs[
        _dtype: DType where _dtype.is_floating_point()
    ](
        self: AxisAlignedBoundingBox[_dtype],
        translation: Vec3[_dtype],
        rotation: Quaternion[_dtype],
        scale: Vec3[_dtype],
    ) -> AxisAlignedBoundingBox[_dtype]:
        rot_mat = Mat33[_dtype].from_rotation_scale(rotation, scale)
        txfmed = AxisAlignedBoundingBox[_dtype](
            translation.copy(), translation.copy()
        )

        # use comptime to fully unroll
        # in bench_aabb, 3.3x faster then version without comptime
        # 18% faster then arvo version
        comptime for j in range(3):
            comptime for i in range(3):
                e = rot_mat[i][j] * self._min[j]
                f = rot_mat[i][j] * self._max[j]

                if e < f:
                    txfmed._min[i] += e
                    txfmed._max[i] += f
                else:
                    txfmed._min[i] += f
                    txfmed._max[i] += e
        return txfmed^
