"""SIMD mesh validation and ordered emissive-instance light preparation."""

from std.algorithm import vectorize
from std.math import iota, sqrt
from std.sys import simd_width_of
from std.sys.info import size_of
from std.utils.numerics import isfinite
from bajo.core import (
    AABB,
    Affine3,
    Affine3f32,
    Point3,
    Point3f32,
    cross,
    length2,
)
from bajo.core.utils import fmin, fmax
from bajo.bvh.constants import PrimitiveKind
from bajo.rt.lighting_types import LightRecord, LightStore, _light_importance
from bajo.rt.material_types import PrimitiveId, SurfaceId
from bajo.core.color import Color


@always_inline
def _validate_mesh_bounds(
    vertices: ImmSpan[Point3f32[.LOCAL], _]
) raises -> AABB[.LOCAL]:
    comptime assert size_of[Point3f32[.LOCAL]]() == 3 * size_of[Float32]()
    var result = AABB[.LOCAL].invalid()
    var invalid = False
    var source = vertices.unsafe_ptr().unsafe_bitcast[Float32]()

    def block[lanes: Int](first: Int) {imm, mut result, mut invalid}:
        var ids = (iota[.int, lanes]() + first) * 9
        var points = (
            Point3[.float32, .LOCAL, lanes](
                source.unsafe_gather(ids),
                source.unsafe_gather(ids + 1),
                source.unsafe_gather(ids + 2),
            ),
            Point3[.float32, .LOCAL, lanes](
                source.unsafe_gather(ids + 3),
                source.unsafe_gather(ids + 4),
                source.unsafe_gather(ids + 5),
            ),
            Point3[.float32, .LOCAL, lanes](
                source.unsafe_gather(ids + 6),
                source.unsafe_gather(ids + 7),
                source.unsafe_gather(ids + 8),
            ),
        )
        var area2 = length2(cross(points[1] - points[0], points[2] - points[0]))
        if not (
            points[0].is_finite()
            & points[1].is_finite()
            & points[2].is_finite()
            & isfinite(area2)
            & area2.gt(0)
        ).reduce_and():
            invalid = True
            return
        comptime for axis in range(3):
            var low = min(
                points[0][axis], min(points[1][axis], points[2][axis])
            ).reduce_min()
            var high = max(
                points[0][axis], max(points[1][axis], points[2][axis])
            ).reduce_max()
            if low == 0 or high == 0:
                # Preserve the original zero-tie order, including signed zero.
                for lane in range(lanes):
                    comptime for corner in range(3):
                        var value = points[corner][axis][lane]
                        result._min[axis] = fmin(result._min[axis], value)
                        result._max[axis] = fmax(result._max[axis], value)
            else:
                result._min[axis] = fmin(result._min[axis], low)
                result._max[axis] = fmax(result._max[axis], high)

    vectorize[simd_width_of[Float32]()](len(vertices) // 3, block)
    if invalid:
        raise Error("triangle mesh vertices must be finite and non-degenerate")
    return result


@always_inline
def _transformed_triangles[
    lanes: Int
](
    vertices: ImmSpan[Point3f32[.LOCAL], _],
    first: Int,
    transform: Affine3f32[.LOCAL, .WORLD],
    reverse: Bool,
    radiance: Color,
) -> Tuple[
    Point3[.float32, .WORLD, lanes],
    Point3[.float32, .WORLD, lanes],
    Point3[.float32, .WORLD, lanes],
    SIMD[.float32, lanes],
]:
    comptime assert size_of[Point3f32[.LOCAL]]() == 3 * size_of[Float32]()
    var source = vertices.unsafe_ptr().unsafe_bitcast[Float32]()
    var ids = (iota[.int, lanes]() + first) * 9
    var wide = Affine3[.float32, .LOCAL, .WORLD, lanes](
        transform.m00,
        transform.m01,
        transform.m02,
        transform.tx,
        transform.m10,
        transform.m11,
        transform.m12,
        transform.ty,
        transform.m20,
        transform.m21,
        transform.m22,
        transform.tz,
    )
    var p0 = wide.point(
        Point3[.float32, .LOCAL, lanes](
            source.unsafe_gather(ids),
            source.unsafe_gather(ids + 1),
            source.unsafe_gather(ids + 2),
        )
    )
    var p1 = wide.point(
        Point3[.float32, .LOCAL, lanes](
            source.unsafe_gather(ids + 3),
            source.unsafe_gather(ids + 4),
            source.unsafe_gather(ids + 5),
        )
    )
    var p2 = wide.point(
        Point3[.float32, .LOCAL, lanes](
            source.unsafe_gather(ids + 6),
            source.unsafe_gather(ids + 7),
            source.unsafe_gather(ids + 8),
        )
    )
    if reverse:
        swap(p1, p2)
    var weight = (
        Float32(0.5) * sqrt(length2(cross(p1 - p0, p2 - p0)))
    ) * _light_importance(radiance)
    return (p0, p1, p2, weight)


@always_inline
def _instance_light_record[
    lanes: Int
](
    values: Tuple[
        Point3[.float32, .WORLD, lanes],
        Point3[.float32, .WORLD, lanes],
        Point3[.float32, .WORLD, lanes],
        SIMD[.float32, lanes],
    ],
    lane: Int,
    surface: SurfaceId[1],
    instance_idx: Int,
    radiance: Color,
    two_sided: Bool,
    texcoords: ImmSpan[Float32, _],
    triangle: Int,
    reverse: Bool,
) -> LightRecord:
    var u0 = Float32(0)
    var v0 = Float32(0)
    var u1 = Float32(1)
    var v1 = Float32(0)
    var u2 = Float32(1)
    var v2 = Float32(1)
    if len(texcoords):
        var base = 6 * triangle
        u0 = texcoords[base]
        v0 = texcoords[base + 1]
        u1 = texcoords[base + 2]
        v1 = texcoords[base + 3]
        u2 = texcoords[base + 4]
        v2 = texcoords[base + 5]
    if reverse:
        swap(u1, u2)
        swap(v1, v2)
    return LightRecord.triangle(
        PrimitiveId(PrimitiveKind.TRIANGLE_INSTANCE, UInt32(instance_idx)),
        surface,
        values[3][lane] * Float32(2 if two_sided else 1),
        Point3f32[.WORLD](
            values[0].x[lane], values[0].y[lane], values[0].z[lane]
        ),
        Point3f32[.WORLD](
            values[1].x[lane], values[1].y[lane], values[1].z[lane]
        ),
        Point3f32[.WORLD](
            values[2].x[lane], values[2].y[lane], values[2].z[lane]
        ),
        u0,
        v0,
        u1,
        v1,
        u2,
        v2,
        radiance,
        two_sided,
    )


@always_inline
def _append_instance_lights(
    vertices: ImmSpan[Point3f32[.LOCAL], _],
    transform: Affine3f32[.LOCAL, .WORLD],
    surface: SurfaceId[1],
    instance_idx: Int,
    radiance: Color,
    mut output: LightStore,
    texcoords: ImmSpan[Float32, _] = ImmSpan[Float32, ImmutAnyOrigin](),
    two_sided: Bool = False,
) raises:
    var status = 0
    var reverse = transform.reverses_orientation()[0]

    def block[lanes: Int](first: Int) {imm, mut output, mut status}:
        if status:
            return
        var values = _transformed_triangles[lanes](
            vertices, first, transform, reverse, radiance
        )
        comptime for lane in range(lanes):
            if status == 0:
                var record = _instance_light_record(
                    values,
                    lane,
                    surface,
                    instance_idx,
                    radiance,
                    two_sided,
                    texcoords,
                    first + lane,
                    reverse,
                )
                if not isfinite(record.weight):
                    status = 1
                elif record.weight > 0:
                    output.append(record^)
                    if not isfinite(output.total_weight):
                        status = 2

    vectorize[simd_width_of[Float32]()](len(vertices) // 3, block)
    if status == 1:
        raise Error("triangle instance light weight must be finite")
    if status == 2:
        raise Error("total light weight must be finite")
