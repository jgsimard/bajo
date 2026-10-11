"""CPU BSDF adapters over the shared host/device shading implementation."""

from std.math import sqrt

from bajo.core import Vec3f32, dot, normalize, Rayf32
from bajo.core.random import Rng
from bajo.rt.shading import (
    _evaluate_coated_diffuse,
    _evaluate_material,
    _sample_coated_diffuse,
    _sample_material,
)
from bajo.rt.render_types import Color
from bajo.rt.material_types import (
    SurfaceId,
    SurfaceStore,
)
from bajo.rt.shading_types import (
    BsdfEvaluation,
    BsdfSample,
    ShadingPoint,
)


def evaluate_bsdf(
    surface: SurfaceId[1],
    surfaces: SurfaceStore,
    ray: Rayf32[.WORLD],
    hit: ShadingPoint[1],
    out_direction: Vec3f32[.WORLD],
) -> BsdfEvaluation[1]:
    """Evaluate the non-delta BSDF and its solid-angle sampling PDF."""
    __match surface.kind():
        case .LAMBERTIAN:
            return _evaluate_material[.LAMBERTIAN, 1](
                ray.d,
                hit.normal,
                surfaces.sample_albedo(surface, hit.uv_u[0], hit.uv_v[0]),
                SIMD[.float32, 1](1.0),
                out_direction,
            )
        case .METAL:
            ref material = surfaces.metals[Int(surface.index())]
            return _evaluate_material[.METAL, 1](
                ray.d,
                hit.normal,
                surfaces.sample_albedo(surface, hit.uv_u[0], hit.uv_v[0]),
                SIMD[.float32, 1](material.fuzz),
                out_direction,
            )

        case .COATED_DIFFUSE:
            ref material = surfaces.coated_diffuses[Int(surface.index())]
            return _evaluate_coated_diffuse(
                ray.d,
                hit.normal,
                surfaces.sample_albedo(surface, hit.uv_u[0], hit.uv_v[0]),
                SIMD[.float32, 1](material.roughness),
                SIMD[.float32, 1](material.eta),
                SIMD[.float32, 1](material.thickness),
                material.layer_albedo,
                SIMD[.float32, 1](material.g),
                SIMD[.float32, 1](Float32(material.max_depth)),
                SIMD[.float32, 1](Float32(material.n_samples)),
                out_direction,
                conductor_eta=material.conductor_eta,
                conductor_k=material.conductor_k,
                conductor_alpha=material.conductor_roughness,
            )

        case .DIELECTRIC:
            return _evaluate_material[.DIELECTRIC, 1](
                ray.d,
                hit.normal,
                Color(0.0),
                SIMD[.float32, 1](1.0),
                out_direction,
            )

        case .EMISSIVE:
            return _evaluate_material[.EMISSIVE, 1](
                ray.d,
                hit.normal,
                Color(0.0),
                SIMD[.float32, 1](1.0),
                out_direction,
            )
        case .MIX:
            var index = Int(surface.index())
            var first = evaluate_bsdf(
                SurfaceId.from_raw(surfaces.mix_surfaces[2 * index]),
                surfaces,
                ray,
                hit,
                out_direction,
            )
            var second = evaluate_bsdf(
                SurfaceId.from_raw(surfaces.mix_surfaces[2 * index + 1]),
                surfaces,
                ray,
                hit,
                out_direction,
            )
            var amount = surfaces.mix_amounts[index]
            return BsdfEvaluation[1](
                first.value * (1.0 - amount) + second.value * amount,
                first.pdf * (1.0 - amount) + second.pdf * amount,
                first.delta & second.delta,
            )


def sample_bsdf(
    surface: SurfaceId[1],
    surfaces: SurfaceStore,
    ray: Rayf32[.WORLD],
    hit: ShadingPoint[1],
    mut rng: Rng,
) -> BsdfSample[1]:
    __match surface.kind():
        case .LAMBERTIAN:
            return _sample_material[.LAMBERTIAN, 1](
                ray.d,
                hit.normal,
                surfaces.sample_albedo(surface, hit.uv_u[0], hit.uv_v[0]),
                SIMD[.float32, 1](1.0),
                hit.front_face,
                SIMD[.float32, 1](rng.f32()),
                SIMD[.float32, 1](rng.f32()),
            )

        case .METAL:
            ref material = surfaces.metals[Int(surface.index())]
            debug_assert["safe", _use_compiler_assume=True](
                material.fuzz >= 0.0 and material.fuzz <= 1.0
            )
            var random_u = Float32(0.0)
            var random_v = Float32(0.0)
            if material.fuzz > 1.0e-4:
                random_u = rng.f32()
                random_v = rng.f32()
            return _sample_material[.METAL, 1](
                ray.d,
                hit.normal,
                surfaces.sample_albedo(surface, hit.uv_u[0], hit.uv_v[0]),
                SIMD[.float32, 1](material.fuzz),
                hit.front_face,
                SIMD[.float32, 1](random_u),
                SIMD[.float32, 1](random_v),
            )

        case .COATED_DIFFUSE:
            ref material = surfaces.coated_diffuses[Int(surface.index())]
            return _sample_coated_diffuse(
                ray.d,
                hit.normal,
                surfaces.sample_albedo(surface, hit.uv_u[0], hit.uv_v[0]),
                SIMD[.float32, 1](material.roughness),
                SIMD[.float32, 1](material.eta),
                SIMD[.float32, 1](material.thickness),
                material.layer_albedo,
                SIMD[.float32, 1](material.g),
                SIMD[.float32, 1](Float32(material.max_depth)),
                SIMD[.float32, 1](Float32(material.n_samples)),
                SIMD[.float32, 1](rng.f32()),
                SIMD[.float32, 1](rng.f32()),
                conductor_eta=material.conductor_eta,
                conductor_k=material.conductor_k,
                conductor_alpha=material.conductor_roughness,
            )

        case .DIELECTRIC:
            ref material = surfaces.dielectrics[Int(surface.index())]
            debug_assert["safe", _use_compiler_assume=True](
                material.refraction_index > 0.0
            )
            var ri = (
                1.0
                / material.refraction_index if hit.front_face else material.refraction_index
            )
            var unit_direction = normalize(ray.d)
            var cos_theta = min(dot(-unit_direction, hit.normal), 1.0)
            var sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0))
            var random_u = Float32(0.0)
            if ri * sin_theta <= 1.0:
                random_u = rng.f32()
            return _sample_material[.DIELECTRIC, 1](
                ray.d,
                hit.normal,
                Color(0.0),
                SIMD[.float32, 1](material.refraction_index),
                hit.front_face,
                SIMD[.float32, 1](random_u),
                SIMD[.float32, 1](0.0),
            )

        case .EMISSIVE:
            return BsdfSample(
                hit.normal,
                Color(0.0),
                0.0,
                False,
                False,
            )
        case .MIX:
            var index = Int(surface.index())
            var amount = surfaces.mix_amounts[index]
            var choose_second = rng.f32() < amount
            var selected = SurfaceId.from_raw(
                surfaces.mix_surfaces[2 * index + (1 if choose_second else 0)]
            )
            var sampled = sample_bsdf(selected, surfaces, ray, hit, rng)
            if sampled.ok[0]:
                if sampled.delta[0]:
                    sampled.pdf *= amount if choose_second else (1.0 - amount)
                else:
                    var evaluated = evaluate_bsdf(
                        surface, surfaces, ray, hit, sampled.direction
                    )
                    sampled.pdf = evaluated.pdf
                    sampled.weight = evaluated.value * (
                        max(dot(hit.normal, sampled.direction), 0.0)
                        / max(evaluated.pdf, Float32(1.0e-20))
                    )
            return sampled^
