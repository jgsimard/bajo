from std.utils.numerics import max_finite, min_finite

from bajo.core.mat import Mat33
from bajo.core.quat import Quaternion
from bajo.core.vec import Vec3, vmin, vmax

comptime AABB = AxisAlignedBoundingBox[DType.float32]


@fieldwise_init
struct AxisAlignedBoundingBox[dtype: DType where dtype.is_floating_point()](
    Copyable, Writable
):
    var min: Vec3[Self.dtype]
    var max: Vec3[Self.dtype]

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
            vmin(a.min, b.min),
            vmax(a.max, b.max),
        )

    fn surface_area(self) -> Scalar[Self.dtype]:
        d = self.max - self.min
        return 2.0 * (d.x() * d.y() + d.x() * d.z() + d.y() * d.z())

    fn centroid(self) -> Vec3[Self.dtype]:
        return (self.min + self.max) * 0.5

    fn overlaps(self, o: Self) -> Bool:
        return (
            self.min.x() < o.max.x()
            and o.min.x() < self.max.x()
            and self.min.y() < o.max.y()
            and o.min.y() < self.max.y()
            and self.min.z() < o.max.z()
            and o.min.z() < self.max.z()
        )

    fn contains(self, p: Vec3[Self.dtype]) -> Bool:
        return (
            self.min.x() <= p.x()
            and self.min.y() <= p.y()
            and self.min.z() <= p.z()
            and self.max.x() >= p.x()
            and self.max.y() >= p.y()
            and self.max.z() >= p.z()
        )

    fn ray_intersects(
        self,
        ray_o: Vec3[Self.dtype],
        inv_ray_d: Vec3[Self.dtype],
        ray_t_min: Scalar[Self.dtype],
        ray_t_max: Scalar[Self.dtype],
    ) -> Bool:
        t_lower = inv_ray_d * (self.min - ray_o)
        t_upper = inv_ray_d * (self.max - ray_o)

        t_min_vec = vmin(t_lower, t_upper)
        t_max_vec = vmax(t_lower, t_upper)

        t_box_min = max(t_min_vec.x(), t_min_vec.y(), t_min_vec.z(), ray_t_min)

        t_box_max = min(t_max_vec.x(), t_max_vec.y(), t_max_vec.z(), ray_t_max)

        return t_box_min <= t_box_max

    fn apply_trs(
        self,
        translation: Vec3[Self.dtype],
        rotation: Quaternion[Self.dtype],
        scale: Vec3[Self.dtype],
    ) -> Self:
        """
        Transforms the AABB using Jim Arvo algorithm (from "Graphics Gems", Academic Press, 1990) with explicit SIMD.
        """
        mat = Mat33[Self.dtype].from_rotation_scale(rotation, scale).transpose()

        new_min = translation.copy()
        new_max = translation.copy()

        # X
        c0_a = mat[0] * self.min.x()
        c0_b = mat[0] * self.max.x()
        new_min += vmin(c0_a, c0_b)
        new_max += vmax(c0_a, c0_b)

        # Y
        c1_a = mat[1] * self.min.y()
        c1_b = mat[1] * self.max.y()
        new_min += vmin(c1_a, c1_b)
        new_max += vmax(c1_a, c1_b)

        # Z
        c2_a = mat[2] * self.min.z()
        c2_b = mat[2] * self.max.z()
        new_min += vmin(c2_a, c2_b)
        new_max += vmax(c2_a, c2_b)

        return AxisAlignedBoundingBox(new_min^, new_max^)
