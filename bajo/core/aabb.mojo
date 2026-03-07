from std.utils.numerics import max_finite, min_finite

from bajo.core.mat import Mat33
from bajo.core.quat import Quaternion
from bajo.core.vec import Vec3, vmin, vmax

comptime AABB = AxisAlignedBoundingBox[DType.float32]


struct AxisAlignedBoundingBox[dtype: DType](Copyable, Writable):
    var _min: Vec3[Self.dtype]
    var _max: Vec3[Self.dtype]

    fn __init__(out self, v0: Vec3[Self.dtype], v1: Vec3[Self.dtype]):
        self._min = vmin(v0, v1)
        self._max = vmax(v0, v1)

    fn __init__(
        out self,
        v0: Vec3[Self.dtype],
        v1: Vec3[Self.dtype],
        v2: Vec3[Self.dtype],
    ):
        self._min = vmin(vmin(v0, v1), v2)
        self._max = vmax(vmax(v0, v1), v2)

    fn __init__(out self, a: Self, b: Self):
        self._min = vmin(a._min, b._min)
        self._max = vmax(a._max, b._max)

    @staticmethod
    fn invalid() -> Self:
        comptime flt_max = max_finite[Self.dtype]()
        comptime flt_min = min_finite[Self.dtype]()
        return Self(
            Vec3[Self.dtype](flt_max),
            Vec3[Self.dtype](flt_min),
        )

    @staticmethod
    fn point(p: Vec3[Self.dtype]) -> Self:
        return Self(p.copy(), p.copy())

    @staticmethod
    fn merge(a: Self, b: Self) -> Self:
        return Self(
            vmin(a._min, b._min),
            vmax(a._max, b._max),
        )

    fn surface_area(self) -> Scalar[Self.dtype]:
        d = self._max - self._min
        return 2.0 * (d.x() * d.y() + d.x() * d.z() + d.y() * d.z())

    fn centroid(self) -> Vec3[Self.dtype]:
        return (self._min + self._max) * 0.5

    fn area(self) -> Scalar[Self.dtype]:
        diff = self._max - self._min
        return diff.x() * diff.y() + diff.y() * diff.z() + diff.z() * diff.x()

    fn clear(mut self):
        comptime _min = min_finite[Self.dtype]()
        comptime _max = max_finite[Self.dtype]()
        self._min = Vec3[Self.dtype](_min)
        self._max = Vec3[Self.dtype](_max)

    fn grow(mut self, v: Vec3[Self.dtype]):
        self._min = vmin(self._min, v)
        self._max = vmax(self._max, v)

    fn grow(mut self, other: Self):
        self._min = vmin(self._min, other._min)
        self._max = vmax(self._max, other._max)

    fn edges(self) -> Vec3[Self.dtype]:
        return self._max - self._min

    fn overlaps(self, o: Self) -> Bool:
        return (
            self._min.x() < o._max.x()
            and o._min.x() < self._max.x()
            and self._min.y() < o._max.y()
            and o._min.y() < self._max.y()
            and self._min.z() < o._max.z()
            and o._min.z() < self._max.z()
        )

    fn contains(self, p: Vec3[Self.dtype]) -> Bool:
        return (
            self._min.x() <= p.x()
            and self._min.y() <= p.y()
            and self._min.z() <= p.z()
            and self._max.x() >= p.x()
            and self._max.y() >= p.y()
            and self._max.z() >= p.z()
        )

    fn ray_intersects(
        self,
        ray_o: Vec3[Self.dtype],
        inv_ray_d: Vec3[Self.dtype],
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

    fn apply_trs[
        _dtype: DType where _dtype.is_floating_point()
    ](
        self: AxisAlignedBoundingBox[_dtype],
        translation: Vec3[_dtype],
        rotation: Quaternion[_dtype],
        scale: Vec3[_dtype],
    ) -> AxisAlignedBoundingBox[_dtype]:
        """
        Transforms the AABB using Jim Arvo algorithm (from "Graphics Gems", Academic Press, 1990) with explicit SIMD.
        """
        mat = Mat33[_dtype].from_rotation_scale(rotation, scale).transpose()

        new_min = translation.copy()
        new_max = translation.copy()

        # X
        c0_a = mat[0] * self._min.x()
        c0_b = mat[0] * self._max.x()
        new_min += vmin(c0_a, c0_b)
        new_max += vmax(c0_a, c0_b)

        # Y
        c1_a = mat[1] * self._min.y()
        c1_b = mat[1] * self._max.y()
        new_min += vmin(c1_a, c1_b)
        new_max += vmax(c1_a, c1_b)

        # Z
        c2_a = mat[2] * self._min.z()
        c2_b = mat[2] * self._max.z()
        new_min += vmin(c2_a, c2_b)
        new_max += vmax(c2_a, c2_b)

        return AxisAlignedBoundingBox(new_min^, new_max^)
