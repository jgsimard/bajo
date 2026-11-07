"""Host/device-safe light selection, surface sampling, PDFs, and weighting."""

from bajo.core.sampling import power_heuristic

from std.math import abs, sqrt
from bajo.core.geometry import sample_triangle_surface, sample_sphere_surface

from bajo.core import Point3, Vec3, dot, length2, normalize
from bajo.rt.rays import RT_RAY_T_MIN, RT_SHADOW_END_OFFSET
from bajo.rt.material_types import Integrator
from bajo.rt.lighting_types import _light_importance
from bajo.rt.opacity import opacity_sphere_uv


@fieldwise_init
struct _LightSurfaceSample[length: SIMDLength = 1](TrivialRegisterPassable):
    var valid: SIMD[.bool, Self.length]
    var point: Point3[.float32, .WORLD, Self.length]
    var normal: Vec3[.float32, .WORLD, Self.length]
    var u: SIMD[.float32, Self.length]
    var v: SIMD[.float32, Self.length]


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
    var sample = sample_triangle_surface(p0, p1, p2, random_u, random_v)
    var root_u = sqrt(random_u)
    var barycentric1 = root_u * (1.0 - random_v)
    var barycentric2 = root_u - barycentric1
    return _LightSurfaceSample[length](
        sample.valid, sample.point, sample.normal, barycentric1, barycentric2
    )


@always_inline
def _sample_sphere_light_surface[
    length: SIMDLength = 1
](
    center: Point3[.float32, .WORLD, length],
    radius: SIMD[.float32, length],
    normal: Vec3[.float32, .WORLD, length],
) -> _LightSurfaceSample[length]:
    var sample = sample_sphere_surface(center, radius, normal)
    var uv = opacity_sphere_uv(sample.normal)
    return _LightSurfaceSample[length](
        sample.valid,
        sample.point,
        sample.normal,
        uv[0],
        uv[1],
    )


@always_inline
def light_sample_uv[
    length: SIMDLength = 1
](
    u0: SIMD[.float32, length],
    v0: SIMD[.float32, length],
    u1: SIMD[.float32, length],
    v1: SIMD[.float32, length],
    u2: SIMD[.float32, length],
    v2: SIMD[.float32, length],
    b1: SIMD[.float32, length],
    b2: SIMD[.float32, length],
) -> Tuple[SIMD[.float32, length], SIMD[.float32, length]]:
    var b0 = 1.0 - b1 - b2
    return (b0 * u0 + b1 * u1 + b2 * u2, b0 * v0 + b1 * v1 + b2 * v2)


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
    allow_transmission: SIMD[.bool, length] = SIMD[.bool, length](fill=False),
    two_sided: SIMD[.bool, length] = SIMD[.bool, length](fill=False),
) -> _DirectLightGeometrySample[length]:
    if total_light_weight <= 0.0:
        return _empty_direct_light_geometry[length]()
    var to_light = light.point - point
    var distance_squared = length2(to_light)
    var valid = light.valid & distance_squared.gt(1.0e-8)
    var distance = sqrt(valid.select(distance_squared, 1.0))
    var direction = to_light / distance
    var cosine = dot(normal, direction)
    var surface_cosine = allow_transmission.select(
        abs(cosine), max(cosine, 0.0)
    )
    var cosine_at_light = dot(light.normal, -direction)
    var light_cosine = two_sided.select(
        abs(cosine_at_light), max(cosine_at_light, 0.0)
    )
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
            )
            * two_sided.select(Float32(2.0), Float32(1.0)),
            0.0,
        ),
        valid.select(shadow_t_max, 0.0),
    )


@always_inline
def _emissive_hit_light_pdf[
    length: SIMDLength = 1
](
    ray_direction: Vec3[.float32, .WORLD, length],
    hit_t: SIMD[.float32, length],
    normal: Vec3[.float32, .WORLD, length],
    emission: Vec3[.float32, .WORLD, length],
    total_light_weight: Float32,
    two_sided: SIMD[.bool, length] = SIMD[.bool, length](fill=False),
) -> SIMD[.float32, length]:
    """Evaluate emissive-hit light PDFs at any width, including one."""
    var cosine = dot(normal, -normalize(ray_direction))
    var light_cosine = two_sided.select(abs(cosine), max(cosine, 0.0))
    return _solid_angle_light_pdf(
        hit_t * hit_t * length2(ray_direction),
        light_cosine,
        emission,
        total_light_weight,
    ) * two_sided.select(Float32(2.0), Float32(1.0))


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
