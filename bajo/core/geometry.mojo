"""Geometry values, triangle attributes, surface orientation and bounds."""

from std.math import abs, sqrt
from std.utils.numerics import isfinite
from .aabb import AxisAlignedBoundingBox
from .frame import Frame
from .vec import Point3, Vec3, cross, dot, length2
from .tensor_layout import RecordTile, gather_record_field


@always_inline
def triangle_uv_tangent[
    frame: Frame
](
    p0: Point3[.float32, frame],
    p1: Point3[.float32, frame],
    p2: Point3[.float32, frame],
    u0: Float32,
    v0: Float32,
    u1: Float32,
    v1: Float32,
    u2: Float32,
    v2: Float32,
) -> Vec3[.float32, frame]:
    """Position derivative with respect to U; zero marks degenerate UVs."""
    var du1 = u1 - u0
    var dv1 = v1 - v0
    var du2 = u2 - u0
    var dv2 = v2 - v0
    var determinant = du1 * dv2 - dv1 * du2
    if determinant == 0.0:
        return Vec3[.float32, frame](0.0)
    return ((p1 - p0) * dv2 - (p2 - p0) * dv1) / determinant


@fieldwise_init
struct Sphere[frame: Frame = .WORLD](TrivialRegisterPassable):
    comptime STRIDE = 4
    var center: Point3[.float32, Self.frame]
    var radius: Float32

    def physical_radius(self) -> Float32:
        return abs(self.radius)

    def for_acceleration(self) -> Self:
        """Return acceleration geometry without the shading-orientation sign."""
        return Self(self.center, self.physical_radius())

    def bounds(self) -> AxisAlignedBoundingBox[.float32, Self.frame]:
        var r = Vec3[.float32, Self.frame](self.physical_radius())
        return AxisAlignedBoundingBox[.float32, Self.frame](
            self.center - r, self.center + r
        )


def sphere_bounds[
    frame: Frame
](spheres: ImmSpan[Sphere[frame], _]) -> AxisAlignedBoundingBox[
    .float32, frame
]:
    var bounds = AxisAlignedBoundingBox[.float32, frame].invalid()
    for s in spheres:
        bounds.grow(s.bounds())
    return bounds


@always_inline
def interpolate_triangle_attribute[
    mut: Bool,
    origin: Origin[mut=mut],
    //,
    components: Int,
    component: Int,
    index_dtype: DType,
    length: SIMDLength,
](
    attributes: RecordTile[.float32, 3 * components, origin],
    triangles: SIMD[index_dtype, length],
    u: SIMD[.float32, length],
    v: SIMD[.float32, length],
) -> SIMD[.float32, length]:
    """Interpolate one attribute component from triangle-ordered records."""
    return (
        (Float32(1.0) - u - v)
        * gather_record_field[component](attributes, triangles)
        + u * gather_record_field[components + component](attributes, triangles)
        + v
        * gather_record_field[2 * components + component](attributes, triangles)
    )


@fieldwise_init
struct OrientedSurfaceNormal[frame: Frame, length: SIMDLength = 1](
    TrivialRegisterPassable
):
    """An outward normal oriented against the incident ray direction."""

    var normal: Vec3[.float32, Self.frame, Self.length]
    var front_face: SIMD[.bool, Self.length]


@always_inline
def orient_surface_normal[
    frame: Frame, length: SIMDLength
](
    ray_direction: Vec3[.float32, frame, length],
    outward_normal: Vec3[.float32, frame, length],
) -> OrientedSurfaceNormal[frame, length]:
    """Return the consistently oriented normal and front-face classification."""
    var front_face = dot(ray_direction, outward_normal).lt(0.0)
    return OrientedSurfaceNormal[frame, length](
        Vec3.select(front_face, outward_normal, -outward_normal), front_face
    )


@always_inline
def triangle_is_valid[
    frame: Frame
](
    v0: Point3[.float32, frame],
    v1: Point3[.float32, frame],
    v2: Point3[.float32, frame],
) -> Bool:
    if not (v0.is_finite()[0] and v1.is_finite()[0] and v2.is_finite()[0]):
        return False
    var twice_area_squared = length2(cross(v1 - v0, v2 - v0))[0]
    return isfinite(twice_area_squared) and twice_area_squared > 0.0


@always_inline
def triangle_area[
    frame: Frame
](
    v0: Point3[.float32, frame],
    v1: Point3[.float32, frame],
    v2: Point3[.float32, frame],
) -> Float32:
    return 0.5 * sqrt(length2(cross(v1 - v0, v2 - v0)))
