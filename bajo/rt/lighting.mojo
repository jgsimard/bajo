"""Host/device-safe light selection, surface sampling, PDFs, and weighting."""

from std.math import sqrt

from bajo.core import (
    Point3,
    Vec3,
    cross,
    dot,
    length2,
    normalize,
)
from bajo.rt.rays import RT_RAY_T_MIN, RT_SHADOW_END_OFFSET
from bajo.rt.material_types import Integrator
from bajo.rt.lighting_types import _light_importance


@fieldwise_init
struct _AliasDraw[length: SIMDLength = 1](TrivialRegisterPassable):
    var column: SIMD[.int64, Self.length]
    var fraction: SIMD[.float32, Self.length]


@always_inline
def _draw_alias_column[
    length: SIMDLength = 1
](random_u: SIMD[.float32, length], count: Int) -> _AliasDraw[length]:
    var scaled = random_u * Float32(count)
    var column = scaled.cast[.int64]().clamp(0, SIMD[.int64, length](count - 1))
    return _AliasDraw[length](column, scaled - column.cast[.float32]())


@always_inline
def _resolve_alias_draw[
    length: SIMDLength = 1
](
    draw: _AliasDraw[length],
    probability: SIMD[.float32, length],
    alias_index: SIMD[.uint32, length],
) -> SIMD[.int64, length]:
    # Strict comparison prevents a zero-weight column from winning at u=0.
    return draw.fraction.lt(probability).select(
        draw.column, alias_index.cast[.int64]()
    )


@fieldwise_init
struct _LightSurfaceSample[length: SIMDLength = 1](TrivialRegisterPassable):
    var valid: SIMD[.bool, Self.length]
    var point: Point3[.float32, .WORLD, Self.length]
    var normal: Vec3[.float32, .WORLD, Self.length]


@always_inline
def _sample_triangle_light_surface[
    length: SIMDLength = 1
](
    p0: Point3[.float32, .WORLD, length],
    p1: Point3[.float32, .WORLD, length],
    p2: Point3[.float32, .WORLD, length],
    random_u: SIMD[.float32, length],
    random_v: SIMD[.float32, length],
) -> _LightSurfaceSample[length]:
    var edge1 = p1 - p0
    var edge2 = p2 - p0
    var area_vector = cross(edge1, edge2)
    var twice_area_squared = length2(area_vector)
    var valid = twice_area_squared.gt(0.0)
    var root_u = sqrt(random_u)
    var barycentric1 = root_u * (1.0 - random_v)
    var barycentric2 = root_u - barycentric1
    return _LightSurfaceSample[length](
        valid,
        p0 + barycentric1 * edge1 + barycentric2 * edge2,
        Vec3.select(
            valid,
            area_vector / sqrt(valid.select(twice_area_squared, 1.0)),
            Vec3[.float32, .WORLD, length](0.0, 1.0, 0.0),
        ),
    )


@always_inline
def _sample_sphere_light_surface[
    length: SIMDLength = 1
](
    center: Point3[.float32, .WORLD, length],
    radius: SIMD[.float32, length],
    normal: Vec3[.float32, .WORLD, length],
) -> _LightSurfaceSample[length]:
    return _LightSurfaceSample[length](
        SIMD[.bool, length](fill=True), center + radius * normal, normal
    )


@fieldwise_init
struct _DirectLightGeometrySample[length: SIMDLength = 1](
    TrivialRegisterPassable
):
    var valid: SIMD[.bool, Self.length]
    var direction: Vec3[.float32, .WORLD, Self.length]
    var surface_cosine: SIMD[.float32, Self.length]
    var light_pdf: SIMD[.float32, Self.length]
    var shadow_t_max: SIMD[.float32, Self.length]


@always_inline
def _empty_direct_light_geometry[
    length: SIMDLength = 1
]() -> _DirectLightGeometrySample[length]:
    return _DirectLightGeometrySample[length](
        SIMD[.bool, length](fill=False),
        Vec3[.float32, .WORLD, length](0.0),
        0.0,
        0.0,
        0.0,
    )


@always_inline
def _solid_angle_light_pdf[
    length: SIMDLength = 1
](
    distance_squared: SIMD[.float32, length],
    light_cosine: SIMD[.float32, length],
    emission: Vec3[.float32, .WORLD, length],
    total_light_weight: Float32,
) -> SIMD[.float32, length]:
    if total_light_weight <= 0.0:
        return 0.0
    var valid = light_cosine.gt(0.0)
    return valid.select(
        distance_squared
        * _light_importance(emission)
        / (valid.select(light_cosine, 1.0) * total_light_weight),
        0.0,
    )


@always_inline
def _finish_direct_light_geometry[
    length: SIMDLength = 1
](
    point: Point3[.float32, .WORLD, length],
    normal: Vec3[.float32, .WORLD, length],
    light: _LightSurfaceSample[length],
    emission: Vec3[.float32, .WORLD, length],
    total_light_weight: Float32,
) -> _DirectLightGeometrySample[length]:
    if total_light_weight <= 0.0:
        return _empty_direct_light_geometry[length]()
    var to_light = light.point - point
    var distance_squared = length2(to_light)
    var valid = light.valid & distance_squared.gt(1.0e-8)
    var distance = sqrt(valid.select(distance_squared, 1.0))
    var direction = to_light / distance
    var surface_cosine = max(dot(normal, direction), 0.0)
    var light_cosine = max(dot(light.normal, -direction), 0.0)
    var shadow_t_max = distance - RT_SHADOW_END_OFFSET
    valid &= (
        surface_cosine.gt(0.0)
        & light_cosine.gt(0.0)
        & shadow_t_max.gt(RT_RAY_T_MIN)
    )
    return _DirectLightGeometrySample[length](
        valid,
        Vec3.select(valid, direction, Vec3[.float32, .WORLD, length](0.0)),
        valid.select(surface_cosine, 0.0),
        valid.select(
            _solid_angle_light_pdf(
                distance_squared, light_cosine, emission, total_light_weight
            ),
            0.0,
        ),
        valid.select(shadow_t_max, 0.0),
    )


@always_inline
def power_heuristic[
    length: SIMDLength = 1
](
    pdf_a: SIMD[.float32, length],
    pdf_b: SIMD[.float32, length],
) -> SIMD[
    .float32, length
]:
    var a2 = pdf_a * pdf_a
    var b2 = pdf_b * pdf_b
    var denominator = a2 + b2
    var nonzero = denominator.gt(0.0)
    return nonzero.select(a2 / nonzero.select(denominator, 1.0), 0.0)


@always_inline
def _emissive_hit_light_pdf[
    length: SIMDLength = 1
](
    ray_direction: Vec3[.float32, .WORLD, length],
    hit_t: SIMD[.float32, length],
    normal: Vec3[.float32, .WORLD, length],
    emission: Vec3[.float32, .WORLD, length],
    total_light_weight: Float32,
) -> SIMD[.float32, length]:
    """Evaluate emissive-hit light PDFs at any width, including one."""
    var light_cosine = max(dot(normal, -normalize(ray_direction)), 0.0)
    return _solid_angle_light_pdf(
        hit_t * hit_t * length2(ray_direction),
        light_cosine,
        emission,
        total_light_weight,
    )


@always_inline
def _emissive_hit_weight_from_pdf[
    integrator: Integrator,
    length: SIMDLength = 1,
](
    bounce: UInt32,
    previous_delta: SIMD[.bool, length],
    previous_bsdf_pdf: SIMD[.float32, length],
    light_pdf: SIMD[.float32, length],
) -> SIMD[.float32, length]:
    """Apply one NEE/MIS policy to scalar and packet emission."""
    if bounce > 0:
        comptime match integrator:
            case .NEE:
                return previous_delta.select(Float32(1.0), Float32(0.0))
            case .MIS:
                return previous_delta.select(
                    Float32(1.0), power_heuristic(previous_bsdf_pdf, light_pdf)
                )
            case .PATH | .AO | .NORMALS:
                pass
    return 1.0


@always_inline
def _direct_light_scale[
    integrator: Integrator, length: SIMDLength
](
    surface_cosine: SIMD[.float32, length],
    light_pdf: SIMD[.float32, length],
    bsdf_pdf: SIMD[.float32, length],
    valid: SIMD[.bool, length],
) -> SIMD[.float32, length]:
    comptime assert Integrator.uses_direct_lighting[integrator]
    var ok = valid & light_pdf.gt(0.0) & bsdf_pdf.gt(0.0)
    var safe_light_pdf = ok.select(light_pdf, Float32(1.0))
    var estimator_weight = SIMD[.float32, length](1.0)
    comptime if integrator == .MIS:
        estimator_weight = power_heuristic(light_pdf, bsdf_pdf)
    return ok.select(
        surface_cosine * estimator_weight / safe_light_pdf, Float32(0.0)
    )
