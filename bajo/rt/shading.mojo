"""Host/device-safe BSDF sampling and evaluation primitives."""

from std.math import abs, cos, fma, pi, pow, sin, sqrt

from bajo.core import dot, cross, length2, normalize, Frame, Vec3
from bajo.rt.types import BsdfEvaluation, BsdfSample, MaterialKind


comptime BSDF_INV_PI = Float32(0.3183098861837907)


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
        albedo * BSDF_INV_PI,
        cosine * BSDF_INV_PI,
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
    var smooth = fuzz.le(1.0e-4)
    var safe_fuzz = smooth.select(Float32(1.0), fuzz)
    var direction = normalize(out_direction)
    var reflected = normalize(reflect(normalize(ray_direction), normal))
    var surface_valid = dot(normal, direction).gt(0.0)
    var lobe_cosine = max(dot(reflected, direction), 0.0)
    var lobe_valid = lobe_cosine.gt(0.0)
    var exponent = max(2.0 / (safe_fuzz * safe_fuzz) - 2.0, 0.0)
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
def _evaluate_coated_diffuse[
    length: SIMDLength
](
    ray_direction: Vec3[.float32, .WORLD, length],
    normal: Vec3[.float32, .WORLD, length],
    albedo: Vec3[.float32, .WORLD, length],
    roughness: SIMD[.float32, length],
    eta: SIMD[.float32, length],
    out_direction: Vec3[.float32, .WORLD, length],
) -> BsdfEvaluation[length]:
    """Approximate a dielectric coat over a Lambertian substrate.

    The glossy and diffuse lobes share an energy-conserving Fresnel mixture.
    """
    var incoming_cosine = max(dot(normal, -normalize(ray_direction)), 0.0)
    var outgoing_cosine = max(dot(normal, normalize(out_direction)), 0.0)
    var incoming_fresnel = reflectance(incoming_cosine, 1.0 / eta)
    var outgoing_fresnel = reflectance(outgoing_cosine, 1.0 / eta)
    var specular = _evaluate_metal(
        ray_direction,
        normal,
        Vec3[.float32, .WORLD, length](1.0),
        roughness,
        out_direction,
    )
    var diffuse = _evaluate_lambertian(normal, albedo, out_direction)
    var diffuse_transmission = (1.0 - incoming_fresnel) * (
        1.0 - outgoing_fresnel
    )
    var value = (
        specular.value * incoming_fresnel + diffuse.value * diffuse_transmission
    )
    var pdf = specular.pdf * incoming_fresnel + diffuse.pdf * (
        1.0 - incoming_fresnel
    )
    return BsdfEvaluation[length](value, pdf, specular.delta)


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
    comptime if MATERIAL_KIND == .LAMBERTIAN:
        return _evaluate_lambertian(normal, albedo, out_direction)
    elif MATERIAL_KIND == .METAL:
        return _evaluate_metal(
            ray_direction, normal, albedo, parameter, out_direction
        )
    elif MATERIAL_KIND == .DIELECTRIC:
        return BsdfEvaluation[length](
            Vec3[.float32, .WORLD, length](0.0),
            SIMD[.float32, length](0.0),
            SIMD[.bool, length](fill=True),
        )
    else:
        comptime assert MATERIAL_KIND == .EMISSIVE
        return BsdfEvaluation[length](
            Vec3[.float32, .WORLD, length](0.0),
            SIMD[.float32, length](0.0),
            SIMD[.bool, length](fill=False),
        )


@always_inline
def _sample_lambertian[
    length: SIMDLength
](
    normal: Vec3[.float32, .WORLD, length],
    albedo: Vec3[.float32, .WORLD, length],
    random_u: SIMD[.float32, length],
    random_v: SIMD[.float32, length],
) -> BsdfSample[length]:
    var theta = Float32(2.0 * pi) * random_u
    var z = 1.0 - 2.0 * random_v
    var radius = sqrt(max(1.0 - z * z, 0.0))
    var random_direction = Vec3[.float32, .WORLD, length](
        radius * cos(theta), radius * sin(theta), z
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
    var smooth = fuzz.le(1.0e-4)
    var safe_fuzz = smooth.select(Float32(1.0), fuzz)
    var exponent = max(2.0 / (safe_fuzz * safe_fuzz) - 2.0, 0.0)
    var cos_theta = pow(random_u, 1.0 / (exponent + 1.0))
    var sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0))
    var phi = Float32(2.0 * pi) * random_v
    var helper_y = Vec3[.float32, .WORLD, length](0.0, 1.0, 0.0)
    var helper_x = Vec3[.float32, .WORLD, length](1.0, 0.0, 0.0)
    var helper = Vec3.select(abs(reflected.y).gt(0.99), helper_x, helper_y)
    var tangent = normalize(cross(helper, reflected))
    var bitangent = cross(reflected, tangent)
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
def _sample_coated_diffuse[
    length: SIMDLength
](
    ray_direction: Vec3[.float32, .WORLD, length],
    normal: Vec3[.float32, .WORLD, length],
    albedo: Vec3[.float32, .WORLD, length],
    roughness: SIMD[.float32, length],
    eta: SIMD[.float32, length],
    random_u: SIMD[.float32, length],
    random_v: SIMD[.float32, length],
) -> BsdfSample[length]:
    var incoming_cosine = max(dot(normal, -normalize(ray_direction)), 0.0)
    var specular_probability = reflectance(incoming_cosine, 1.0 / eta)
    var choose_specular = random_u.lt(specular_probability)
    var safe_specular_probability = max(specular_probability, 1.0e-6)
    var safe_diffuse_probability = max(1.0 - specular_probability, 1.0e-6)
    var specular_u = random_u / safe_specular_probability
    var diffuse_u = (random_u - specular_probability) / safe_diffuse_probability
    var specular_sample = _sample_metal(
        ray_direction,
        normal,
        Vec3[.float32, .WORLD, length](1.0),
        roughness,
        specular_u,
        random_v,
    )
    var diffuse_sample = _sample_lambertian(normal, albedo, diffuse_u, random_v)
    var direction = Vec3.select(
        choose_specular, specular_sample.direction, diffuse_sample.direction
    )
    var evaluation = _evaluate_coated_diffuse(
        ray_direction, normal, albedo, roughness, eta, direction
    )
    var cosine = max(dot(normal, direction), 0.0)
    var safe_pdf = max(evaluation.pdf, 1.0e-20)
    var rough_weight = evaluation.value * (cosine / safe_pdf)
    var smooth_specular = choose_specular & roughness.le(1.0e-4)
    var weight = Vec3.select(
        smooth_specular,
        Vec3[.float32, .WORLD, length](1.0),
        rough_weight,
    )
    var pdf = smooth_specular.select(specular_probability, evaluation.pdf)
    var ok = smooth_specular | pdf.gt(0.0)
    return BsdfSample[length](direction, weight, pdf, smooth_specular, ok)


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
    var unit_direction = normalize(ray_direction)
    var cos_theta = min(dot(-unit_direction, normal), 1.0)
    var sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0))
    var cannot_refract = (ri * sin_theta).gt(1.0)
    var reflection_probability = reflectance(cos_theta, ri)
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
    comptime if MATERIAL_KIND == .LAMBERTIAN:
        return _sample_lambertian(normal, albedo, random_u, random_v)
    elif MATERIAL_KIND == .METAL:
        return _sample_metal(
            ray_direction, normal, albedo, parameter, random_u, random_v
        )
    else:
        comptime assert MATERIAL_KIND == .DIELECTRIC
        return _sample_dielectric(
            ray_direction, normal, parameter, front_face, random_u
        )


def reflect[
    dtype: DType, frame: Frame, length: SIMDLength
](v: Vec3[dtype, frame, length], n: Vec3[dtype, frame, length]) -> Vec3[
    dtype, frame, length
]:
    return v - 2.0 * dot(v, n) * n


def refract[
    dtype: DType, frame: Frame, length: SIMDLength
](
    uv: Vec3[dtype, frame, length],
    n: Vec3[dtype, frame, length],
    etai_over_etat: SIMD[dtype, length],
) -> Vec3[dtype, frame, length]:
    var cos_theta = min(dot(-uv, n), 1.0)
    var r_out_perp = etai_over_etat * (uv + cos_theta * n)
    var r_out_parallel = -sqrt(abs(1.0 - length2(r_out_perp))) * n
    return r_out_perp + r_out_parallel


def reflectance[
    dtype: DType, length: SIMDLength
](cosine: SIMD[dtype, length], ref_idx: SIMD[dtype, length]) -> SIMD[
    dtype, length
]:
    var root = (1.0 - ref_idx) / (1.0 + ref_idx)
    var r2 = root * root
    var x = 1.0 - cosine
    var x2 = x * x
    var x5 = x2 * x2 * x
    return fma(1.0 - r2, x5, r2)
