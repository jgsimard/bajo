"""CPU BSDF sampling, evaluation, and compatibility scatter functions."""

from std.math import abs, cos, fma, pi, pow, sin, sqrt

from bajo.core import (
    Vec3f32,
    dot,
    cross,
    length2,
    normalize,
    Point3f32,
    Frame,
    Vec3,
    Rayf32,
)
from bajo.core.random import (
    Rng,
    random_on_hemisphere,
    random_unit_vector,
)
from bajo.bvh.constants import f32_max
from bajo.rt.types import (
    BsdfEvaluation,
    BsdfSample,
    Color,
    Dielectric,
    Lambertian,
    MAT,
    Metal,
    ShadingPoint,
    SurfaceId,
    SurfaceStore,
)


comptime BSDF_INV_PI = Float32(0.3183098861837907)


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


def _sample_lambertian(
    ref material: Lambertian,
    hit: ShadingPoint,
    mut rng: Rng,
) -> BsdfSample:
    var scatter_direction = hit.normal + random_unit_vector[Frame.WORLD](rng)
    if scatter_direction.is_near_zero():
        scatter_direction = hit.normal

    var out_direction = normalize(scatter_direction)
    var pdf = max(dot(hit.normal, out_direction), 0.0) * BSDF_INV_PI
    return BsdfSample(
        True,
        Rayf32[Frame.WORLD](hit.p, out_direction, 0.001, f32_max),
        material.albedo,
        pdf,
        False,
    )


def _evaluate_metal(
    ref material: Metal,
    ray: Rayf32[Frame.WORLD],
    hit: ShadingPoint,
    out_direction: Vec3f32[Frame.WORLD],
) -> BsdfEvaluation:
    if material.fuzz <= 1.0e-4:
        return BsdfEvaluation(Color(0.0), 0.0, True)
    var direction = normalize(out_direction)
    if dot(hit.normal, direction) <= 0.0:
        return BsdfEvaluation(Color(0.0), 0.0, False)
    var reflected = normalize(
        reflect[DType.float32, Frame.WORLD](normalize(ray.d), hit.normal)
    )
    var lobe_cosine = max(dot(reflected, direction), 0.0)
    if lobe_cosine <= 0.0:
        return BsdfEvaluation(Color(0.0), 0.0, False)
    var exponent = max(2.0 / (material.fuzz * material.fuzz) - 2.0, 0.0)
    var lobe = pow(lobe_cosine, exponent)
    return BsdfEvaluation(
        material.albedo * ((exponent + 2.0) * lobe / (2.0 * pi)),
        (exponent + 1.0) * lobe / (2.0 * pi),
        False,
    )


def _sample_metal(
    ref material: Metal,
    ray: Rayf32[Frame.WORLD],
    hit: ShadingPoint,
    mut rng: Rng,
) -> BsdfSample:
    debug_assert["safe", _use_compiler_assume=True](
        material.fuzz >= 0.0 and material.fuzz <= 1.0
    )

    var reflected = normalize(
        reflect[DType.float32, Frame.WORLD](normalize(ray.d), hit.normal)
    )
    if material.fuzz <= 1.0e-4:
        return BsdfSample(
            dot(reflected, hit.normal) > 0.0,
            Rayf32[Frame.WORLD](hit.p, reflected, 0.001, f32_max),
            material.albedo,
            1.0,
            True,
        )

    var exponent = max(2.0 / (material.fuzz * material.fuzz) - 2.0, 0.0)
    var cos_theta = pow(rng.f32(), 1.0 / (exponent + 1.0))
    var sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0))
    var phi = 2.0 * pi * rng.f32()
    var helper = Vec3f32[Frame.WORLD](0.0, 1.0, 0.0)
    if abs(reflected.y) > 0.99:
        helper = Vec3f32[Frame.WORLD](1.0, 0.0, 0.0)
    var tangent = normalize(cross(helper, reflected))
    var bitangent = cross(reflected, tangent)
    var direction = normalize(
        tangent * (cos(phi) * sin_theta)
        + bitangent * (sin(phi) * sin_theta)
        + reflected * cos_theta
    )
    var evaluation = _evaluate_metal(material, ray, hit, direction)
    var surface_cosine = max(dot(hit.normal, direction), 0.0)
    if evaluation.pdf <= 0.0 or surface_cosine <= 0.0:
        return BsdfSample(
            False,
            Rayf32[Frame.WORLD](hit.p, direction, 0.001, f32_max),
            Color(0.0),
            0.0,
            False,
        )
    return BsdfSample(
        True,
        Rayf32[Frame.WORLD](hit.p, direction, 0.001, f32_max),
        evaluation.value * (surface_cosine / evaluation.pdf),
        evaluation.pdf,
        False,
    )


def _sample_dielectric(
    ref material: Dielectric,
    ray: Rayf32[Frame.WORLD],
    hit: ShadingPoint,
    mut rng: Rng,
) -> BsdfSample:
    debug_assert["safe", _use_compiler_assume=True](
        material.refraction_index > 0.0
    )

    var ri = (
        1.0
        / material.refraction_index if hit.front_face else material.refraction_index
    )
    var unit_direction = normalize(ray.d)
    var cos_theta = min(dot(-unit_direction, hit.normal), 1.0)
    var sin_theta = sqrt(1.0 - cos_theta * cos_theta)

    var cannot_refract = ri * sin_theta > 1.0
    var reflection_probability = reflectance(cos_theta, ri)
    var reflect_sample = cannot_refract
    if not cannot_refract:
        reflect_sample = reflection_probability > rng.f32()

    var direction: Vec3f32[Frame.WORLD]
    var pdf: Float32
    if reflect_sample:
        direction = reflect(unit_direction, hit.normal)
        pdf = 1.0 if cannot_refract else reflection_probability
    else:
        direction = refract(unit_direction, hit.normal, ri)
        pdf = 1.0 - reflection_probability

    return BsdfSample(
        True,
        Rayf32(hit.p, normalize(direction), 0.001, f32_max),
        Color(1.0),
        pdf,
        True,
    )


def evaluate_bsdf(
    surface: SurfaceId,
    surfaces: SurfaceStore,
    ray: Rayf32[Frame.WORLD],
    hit: ShadingPoint,
    out_direction: Vec3f32[Frame.WORLD],
) -> BsdfEvaluation:
    """Evaluate the non-delta BSDF and its solid-angle sampling PDF."""
    if surface.kind() == MAT.LAMBERTIAN:
        ref material = surfaces.lambertians[Int(surface.index())]
        var cosine = max(dot(hit.normal, normalize(out_direction)), 0.0)
        return BsdfEvaluation(
            material.albedo * BSDF_INV_PI,
            cosine * BSDF_INV_PI,
            False,
        )
    if surface.kind() == MAT.METAL:
        ref material = surfaces.metals[Int(surface.index())]
        return _evaluate_metal(material, ray, hit, out_direction)

    if surface.kind() == MAT.DIELECTRIC:
        return BsdfEvaluation(Color(0.0), 0.0, True)

    if surface.kind() == MAT.EMISSIVE:
        return BsdfEvaluation(Color(0.0), 0.0, False)

    debug_assert["safe", _use_compiler_assume=True](
        False, "unknown RT surface kind"
    )
    return BsdfEvaluation(Color(0.0), 0.0, False)


def sample_bsdf(
    surface: SurfaceId,
    surfaces: SurfaceStore,
    ray: Rayf32[Frame.WORLD],
    hit: ShadingPoint,
    mut rng: Rng,
) -> BsdfSample:
    if surface.kind() == MAT.LAMBERTIAN:
        ref material = surfaces.lambertians[Int(surface.index())]
        return _sample_lambertian(material, hit, rng)

    if surface.kind() == MAT.METAL:
        ref material = surfaces.metals[Int(surface.index())]
        return _sample_metal(material, ray, hit, rng)

    if surface.kind() == MAT.DIELECTRIC:
        ref material = surfaces.dielectrics[Int(surface.index())]
        return _sample_dielectric(material, ray, hit, rng)

    if surface.kind() == MAT.EMISSIVE:
        return BsdfSample(
            False,
            Rayf32(hit.p, hit.normal, 0.001, f32_max),
            Color(0.0),
            0.0,
            False,
        )

    debug_assert["safe", _use_compiler_assume=True](
        False, "unknown RT surface kind"
    )
    return BsdfSample(
        False,
        Rayf32(hit.p, hit.normal, 0.001, f32_max),
        Color(0.0),
        0.0,
        False,
    )
