"""CPU BSDF adapters over the shared host/device shading implementation."""

from std.math import sqrt

from bajo.core import Vec3f32, dot, normalize, Rayf32
from bajo.core.random import Rng
from bajo.rt.shading import _evaluate_material, _sample_material
from bajo.rt.types import (
    BsdfEvaluation,
    BsdfSample,
    Color,
    ShadingPoint,
    SurfaceId,
    SurfaceStore,
)


def evaluate_bsdf(
    surface: SurfaceId[1],
    surfaces: SurfaceStore,
    ray: Rayf32[.WORLD],
    hit: ShadingPoint[1],
    out_direction: Vec3f32[.WORLD],
) -> BsdfEvaluation[1]:
    """Evaluate the non-delta BSDF and its solid-angle sampling PDF."""
    if surface.kind() == .LAMBERTIAN:
        ref material = surfaces.lambertians[Int(surface.index())]
        return _evaluate_material[.LAMBERTIAN, 1](
            ray.d,
            hit.normal,
            material.albedo,
            SIMD[.float32, 1](1.0),
            out_direction,
        )
    if surface.kind() == .METAL:
        ref material = surfaces.metals[Int(surface.index())]
        return _evaluate_material[.METAL, 1](
            ray.d,
            hit.normal,
            material.albedo,
            SIMD[.float32, 1](material.fuzz),
            out_direction,
        )

    if surface.kind() == .DIELECTRIC:
        return _evaluate_material[.DIELECTRIC, 1](
            ray.d,
            hit.normal,
            Color(0.0),
            SIMD[.float32, 1](1.0),
            out_direction,
        )

    if surface.kind() == .EMISSIVE:
        return _evaluate_material[.EMISSIVE, 1](
            ray.d,
            hit.normal,
            Color(0.0),
            SIMD[.float32, 1](1.0),
            out_direction,
        )

    debug_assert["safe", _use_compiler_assume=True](
        False, "unknown RT surface kind"
    )
    return BsdfEvaluation(Color(0.0), 0.0, False)


def sample_bsdf(
    surface: SurfaceId[1],
    surfaces: SurfaceStore,
    ray: Rayf32[.WORLD],
    hit: ShadingPoint[1],
    mut rng: Rng,
) -> BsdfSample[1]:
    if surface.kind() == .LAMBERTIAN:
        ref material = surfaces.lambertians[Int(surface.index())]
        return _sample_material[.LAMBERTIAN, 1](
            ray.d,
            hit.normal,
            material.albedo,
            SIMD[.float32, 1](1.0),
            hit.front_face,
            SIMD[.float32, 1](rng.f32()),
            SIMD[.float32, 1](rng.f32()),
        )

    if surface.kind() == .METAL:
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
            material.albedo,
            SIMD[.float32, 1](material.fuzz),
            hit.front_face,
            SIMD[.float32, 1](random_u),
            SIMD[.float32, 1](random_v),
        )

    if surface.kind() == .DIELECTRIC:
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

    if surface.kind() == .EMISSIVE:
        return BsdfSample(
            hit.normal,
            Color(0.0),
            0.0,
            False,
            False,
        )

    debug_assert["safe", _use_compiler_assume=True](
        False, "unknown RT surface kind"
    )
    return BsdfSample(
        hit.normal,
        Color(0.0),
        0.0,
        False,
        False,
    )
