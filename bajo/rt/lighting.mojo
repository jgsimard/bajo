"""Host/device-safe light selection, surface sampling, PDFs, and weighting."""

from std.math import sqrt

from bajo.core import (
    Point3f32,
    Vec3f32,
    cross,
    dot,
    length2,
)
from bajo.rt.types import Color, Integrator, _light_importance


@fieldwise_init
struct _AliasDraw(TrivialRegisterPassable):
    var column: Int
    var fraction: Float32


@always_inline
def _draw_alias_column(random_u: Float32, count: Int) -> _AliasDraw:
    var scaled = random_u * Float32(count)
    var column = Int(scaled)
    return _AliasDraw(column, scaled - Float32(column))


@always_inline
def _resolve_alias_draw(
    draw: _AliasDraw,
    probability: Float32,
    alias_index: UInt32,
) -> Int:
    return Int(alias_index) if draw.fraction > probability else draw.column


@fieldwise_init
struct _LightSurfaceSample(TrivialRegisterPassable):
    var valid: Bool
    var point: Point3f32[.WORLD]
    var normal: Vec3f32[.WORLD]


@always_inline
def _sample_triangle_light_surface(
    p0: Point3f32[.WORLD],
    p1: Point3f32[.WORLD],
    p2: Point3f32[.WORLD],
    random_u: Float32,
    random_v: Float32,
) -> _LightSurfaceSample:
    var edge1 = p1 - p0
    var edge2 = p2 - p0
    var area_vector = cross(edge1, edge2)
    var twice_area_squared = length2(area_vector)
    if twice_area_squared <= 0.0:
        return _LightSurfaceSample(False, p0, Vec3f32[.WORLD](0.0, 1.0, 0.0))
    var root_u = sqrt(random_u)
    var barycentric1 = root_u * (1.0 - random_v)
    var barycentric2 = root_u - barycentric1
    return _LightSurfaceSample(
        True,
        p0 + barycentric1 * edge1 + barycentric2 * edge2,
        area_vector / sqrt(twice_area_squared),
    )


@always_inline
def _sample_sphere_light_surface(
    center: Point3f32[.WORLD],
    radius: Float32,
    normal: Vec3f32[.WORLD],
) -> _LightSurfaceSample:
    return _LightSurfaceSample(True, center + radius * normal, normal)


@fieldwise_init
struct _DirectLightGeometrySample(TrivialRegisterPassable):
    var valid: Bool
    var direction: Vec3f32[.WORLD]
    var surface_cosine: Float32
    var light_pdf: Float32
    var shadow_t_max: Float32


@always_inline
def _empty_direct_light_geometry() -> _DirectLightGeometrySample:
    return _DirectLightGeometrySample(
        False, Vec3f32[.WORLD](0.0), 0.0, 0.0, 0.0
    )


@always_inline
def _solid_angle_light_pdf(
    distance_squared: Float32,
    light_cosine: Float32,
    emission: Color,
    total_light_weight: Float32,
) -> Float32:
    if light_cosine <= 0.0 or total_light_weight <= 0.0:
        return 0.0
    return (
        distance_squared
        * _light_importance(emission)
        / (light_cosine * total_light_weight)
    )


@always_inline
def _finish_direct_light_geometry(
    point: Point3f32[.WORLD],
    normal: Vec3f32[.WORLD],
    light: _LightSurfaceSample,
    emission: Color,
    total_light_weight: Float32,
) -> _DirectLightGeometrySample:
    if not light.valid or total_light_weight <= 0.0:
        return _empty_direct_light_geometry()
    var to_light = light.point - point
    var distance_squared = length2(to_light)
    if distance_squared <= 1.0e-8:
        return _empty_direct_light_geometry()
    var distance = sqrt(distance_squared)
    var direction = to_light / distance
    var surface_cosine = max(dot(normal, direction), 0.0)
    var light_cosine = max(dot(light.normal, -direction), 0.0)
    if surface_cosine <= 0.0 or light_cosine <= 0.0:
        return _empty_direct_light_geometry()
    var shadow_t_max = distance - 0.002
    if shadow_t_max <= 0.001:
        return _empty_direct_light_geometry()
    return _DirectLightGeometrySample(
        True,
        direction,
        surface_cosine,
        _solid_angle_light_pdf(
            distance_squared, light_cosine, emission, total_light_weight
        ),
        shadow_t_max,
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
    return nonzero.select(
        a2 / nonzero.select(denominator, Float32(1.0)), Float32(0.0)
    )


@always_inline
def _direct_light_scale[
    integrator: Integrator, length: SIMDLength
](
    surface_cosine: SIMD[.float32, length],
    light_pdf: SIMD[.float32, length],
    bsdf_pdf: SIMD[.float32, length],
    valid: SIMD[.bool, length],
) -> SIMD[.float32, length]:
    comptime assert integrator in (Integrator.NEE, Integrator.MIS)
    var ok = valid & light_pdf.gt(0.0) & bsdf_pdf.gt(0.0)
    var safe_light_pdf = ok.select(light_pdf, Float32(1.0))
    var estimator_weight = SIMD[.float32, length](1.0)
    comptime if integrator == .MIS:
        estimator_weight = power_heuristic(light_pdf, bsdf_pdf)
    return ok.select(
        surface_cosine * estimator_weight / safe_light_pdf, Float32(0.0)
    )
