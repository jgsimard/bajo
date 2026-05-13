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
        return d.x * d.y + d.x * d.z + d.y * d.z

    comptime area = Self.surface_area

    def centroid(self) -> Vec3[Self.dtype, Self.width]:
        return (self._min + self._max) * 0.5

    def clear(mut self):
        comptime _min = min_finite[Self.dtype]()
        comptime _max = max_finite[Self.dtype]()
        self._min = Vec3[Self.dtype, Self.width](_min)
        self._max = Vec3[Self.dtype, Self.width](_max)

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

    def overlaps(self, o: Self) -> Bool:
        return (
            self._min.x < o._max.x
            and o._min.x < self._max.x
            and self._min.y < o._max.y
            and o._min.y < self._max.y
            and self._min.z < o._max.z
            and o._min.z < self._max.z
        )

    def contains(self, p: Vec3[Self.dtype, Self.width]) -> Bool:
        return (
            self._min.x <= p.x
            and self._min.y <= p.y
            and self._min.z <= p.z
            and self._max.x >= p.x
            and self._max.y >= p.y
            and self._max.z >= p.z
        )

    def ray_intersects(
        self,
        ray_o: Vec3[Self.dtype, Self.width],
        inv_ray_d: Vec3[Self.dtype, Self.width],
        ray_t_min: Scalar[Self.dtype],
        ray_t_max: Scalar[Self.dtype],
    ) -> Bool:
        t_lower = inv_ray_d * (self._min - ray_o)
        t_upper = inv_ray_d * (self._max - ray_o)

        t_min_vec = vmin(t_lower, t_upper)
        t_max_vec = vmax(t_lower, t_upper)

        t_box_min = max(t_min_vec.x, t_min_vec.y, t_min_vec.z, ray_t_min)
        t_box_max = min(t_max_vec.x, t_max_vec.y, t_max_vec.z, ray_t_max)

        return t_box_min <= t_box_max

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
