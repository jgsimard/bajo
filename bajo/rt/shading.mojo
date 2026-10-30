"""Host/device-safe BSDF sampling and evaluation primitives."""

from std.math import cos, pi, pow, sin, sqrt

from bajo.core import (
    dot,
    fresnel_schlick,
    normalize,
    orthonormal_basis,
    reflect,
    refract,
    Vec3,
)
from bajo.core.random import unit_vector_from_samples
from bajo.rt.coated_diffuse import (
    _evaluate_coated_diffuse,
    _sample_coated_diffuse,
)
from bajo.rt.material_types import MaterialKind
from bajo.rt.shading_types import (
    BsdfEvaluation,
    BsdfSample,
    RefractionGeometry,
)


comptime INV_PI = 1.0 / pi
comptime METAL_DELTA_FUZZ_THRESHOLD = 1.0e-4


@always_inline
def _metal_lobe_parameters[
    length: SIMDLength
](fuzz: SIMD[.float32, length]) -> Tuple[
    SIMD[.bool, length], SIMD[.float32, length]
]:
    var smooth = fuzz.le(METAL_DELTA_FUZZ_THRESHOLD)
    var safe_fuzz = smooth.select(Float32(1.0), fuzz)
    var exponent = max(2.0 / (safe_fuzz * safe_fuzz) - 2.0, 0.0)
    return smooth, exponent


@always_inline
def _evaluate_lambertian[
    length: SIMDLength
](
    normal: Vec3[.float32, .WORLD, length],
    albedo: Vec3[.float32, .WORLD, length],
    out_direction: Vec3[.float32, .WORLD, length],
) -> BsdfEvaluation[length]:
    var cosine = max(dot(normal, normalize(out_direction)), 0.0)
    return BsdfEvaluation[length](
        albedo * INV_PI,
        cosine * INV_PI,
        SIMD[.bool, length](fill=False),
    )


@always_inline
def _evaluate_metal[
    length: SIMDLength
](
    ray_direction: Vec3[.float32, .WORLD, length],
    normal: Vec3[.float32, .WORLD, length],
    albedo: Vec3[.float32, .WORLD, length],
    fuzz: SIMD[.float32, length],
    out_direction: Vec3[.float32, .WORLD, length],
) -> BsdfEvaluation[length]:
    var smooth, exponent = _metal_lobe_parameters(fuzz)
    var direction = normalize(out_direction)
    var reflected = normalize(reflect(normalize(ray_direction), normal))
    var surface_valid = dot(normal, direction).gt(0.0)
    var lobe_cosine = max(dot(reflected, direction), 0.0)
    var lobe_valid = lobe_cosine.gt(0.0)
    var lobe = pow(lobe_cosine, exponent)
    var valid = (~smooth) & surface_valid & lobe_valid
    var value_scale = (exponent + 2.0) * lobe / Float32(2.0 * pi)
    var pdf = (exponent + 1.0) * lobe / Float32(2.0 * pi)
    var zero = SIMD[.float32, length](0.0)
    var zero_value = Vec3[.float32, .WORLD, length](0.0)
    return BsdfEvaluation[length](
        Vec3.select(valid, albedo * value_scale, zero_value),
        valid.select(pdf, zero),
        smooth,
    )


@always_inline
def _evaluate_material[
    MATERIAL_KIND: MaterialKind, length: SIMDLength
](
    ray_direction: Vec3[.float32, .WORLD, length],
    normal: Vec3[.float32, .WORLD, length],
    albedo: Vec3[.float32, .WORLD, length],
    parameter: SIMD[.float32, length],
    out_direction: Vec3[.float32, .WORLD, length],
) -> BsdfEvaluation[length]:
    """Evaluate one homogeneous material group at compile time."""
    comptime match MATERIAL_KIND:
        case .LAMBERTIAN:
            return _evaluate_lambertian(normal, albedo, out_direction)
        case .METAL:
            return _evaluate_metal(
                ray_direction, normal, albedo, parameter, out_direction
            )
        case .DIELECTRIC:
            return BsdfEvaluation[length](
                Vec3[.float32, .WORLD, length](0.0),
                SIMD[.float32, length](0.0),
                SIMD[.bool, length](fill=True),
            )
        case .EMISSIVE:
            return BsdfEvaluation[length](
                Vec3[.float32, .WORLD, length](0.0),
                SIMD[.float32, length](0.0),
                SIMD[.bool, length](fill=False),
            )
        case .COATED_DIFFUSE | .MIX:
            comptime assert (
                False
            ), "layered and mixed materials require their BSDF adapters"


@always_inline
def _sample_lambertian[
    length: SIMDLength
](
    normal: Vec3[.float32, .WORLD, length],
    albedo: Vec3[.float32, .WORLD, length],
    random_u: SIMD[.float32, length],
    random_v: SIMD[.float32, length],
) -> BsdfSample[length]:
    var random_direction = unit_vector_from_samples[.WORLD, length](
        random_u, random_v
    )
    var scatter_direction = normal + random_direction
    scatter_direction = Vec3.select(
        scatter_direction.is_near_zero(), normal, scatter_direction
    )
    var direction = normalize(scatter_direction)
    var evaluation = _evaluate_lambertian(normal, albedo, direction)
    return BsdfSample[length](
        direction,
        albedo,
        evaluation.pdf,
        SIMD[.bool, length](fill=False),
        SIMD[.bool, length](fill=True),
    )


@always_inline
def _sample_metal[
    length: SIMDLength
](
    ray_direction: Vec3[.float32, .WORLD, length],
    normal: Vec3[.float32, .WORLD, length],
    albedo: Vec3[.float32, .WORLD, length],
    fuzz: SIMD[.float32, length],
    random_u: SIMD[.float32, length],
    random_v: SIMD[.float32, length],
) -> BsdfSample[length]:
    var reflected = normalize(reflect(normalize(ray_direction), normal))
    var smooth, exponent = _metal_lobe_parameters(fuzz)
    var cos_theta = pow(random_u, 1.0 / (exponent + 1.0))
    var sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0))
    var phi = Float32(2.0 * pi) * random_v
    var tangent, bitangent = orthonormal_basis(reflected)
    var rough_direction = normalize(
        tangent * (cos(phi) * sin_theta)
        + bitangent * (sin(phi) * sin_theta)
        + reflected * cos_theta
    )
    var direction = Vec3.select(smooth, reflected, rough_direction)
    var evaluation = _evaluate_metal(
        ray_direction, normal, albedo, fuzz, direction
    )
    var surface_cosine = max(dot(normal, direction), 0.0)
    var rough_ok = evaluation.pdf.gt(0.0) & surface_cosine.gt(0.0)
    var safe_pdf = rough_ok.select(evaluation.pdf, Float32(1.0))
    var scale = surface_cosine / safe_pdf
    var rough_weight = evaluation.value * scale
    var weight = Vec3.select(smooth, albedo, rough_weight)
    var pdf = smooth.select(Float32(1.0), evaluation.pdf)
    var ok = smooth.select(dot(reflected, normal).gt(0.0), rough_ok)
    return BsdfSample[length](direction, weight, pdf, smooth, ok)


@always_inline
def dielectric_geometry[
    length: SIMDLength
](
    direction: Vec3[.float32, .WORLD, length],
    normal: Vec3[.float32, .WORLD, length],
    relative_eta: SIMD[.float32, length],
) -> RefractionGeometry[length]:
    """Shared refraction geometry for BSDF sampling and its conditional RNG draw.
    """
    var unit_direction = normalize(direction)
    var cos_theta = min(dot(-unit_direction, normal), 1.0)
    var sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0))
    return {unit_direction, cos_theta, relative_eta * sin_theta}


@always_inline
def _sample_dielectric[
    length: SIMDLength
](
    ray_direction: Vec3[.float32, .WORLD, length],
    normal: Vec3[.float32, .WORLD, length],
    refraction_index: SIMD[.float32, length],
    front_face: SIMD[.bool, length],
    reflect_random: SIMD[.float32, length],
) -> BsdfSample[length]:
    var ri = front_face.select(
        Float32(1.0) / refraction_index, refraction_index
    )
    var geometry = dielectric_geometry(ray_direction, normal, ri)
    var unit_direction = geometry.unit_direction
    var cos_theta = geometry.incident_cosine
    var cannot_refract = geometry.transmitted_sine.gt(1.0)
    var reflection_probability = fresnel_schlick(cos_theta, ri)
    var reflect_sample = cannot_refract | reflection_probability.gt(
        reflect_random
    )
    var reflected = reflect(unit_direction, normal)
    var refracted = refract(unit_direction, normal, ri)
    var direction = normalize(Vec3.select(reflect_sample, reflected, refracted))
    var pdf = reflect_sample.select(
        cannot_refract.select(Float32(1.0), reflection_probability),
        Float32(1.0) - reflection_probability,
    )
    return BsdfSample[length](
        direction,
        Vec3[.float32, .WORLD, length](1.0, 1.0, 1.0),
        pdf,
        SIMD[.bool, length](fill=True),
        SIMD[.bool, length](fill=True),
    )


@always_inline
def _sample_material[
    MATERIAL_KIND: MaterialKind, length: SIMDLength
](
    ray_direction: Vec3[.float32, .WORLD, length],
    normal: Vec3[.float32, .WORLD, length],
    albedo: Vec3[.float32, .WORLD, length],
    parameter: SIMD[.float32, length],
    front_face: SIMD[.bool, length],
    random_u: SIMD[.float32, length],
    random_v: SIMD[.float32, length],
) -> BsdfSample[length]:
    """Dispatch one homogeneous SIMD material group at compile time."""
    comptime match MATERIAL_KIND:
        case .LAMBERTIAN:
            return _sample_lambertian(normal, albedo, random_u, random_v)
        case .METAL:
            return _sample_metal(
                ray_direction, normal, albedo, parameter, random_u, random_v
            )
        case _:
            comptime assert MATERIAL_KIND == .DIELECTRIC
            return _sample_dielectric(
                ray_direction, normal, parameter, front_face, random_u
            )
