"""Four-wavelength BSDF adapters over shared geometric sampling primitives."""

from bajo.core import Vec3
from bajo.core.sampled_spectrum import SampledSpectrum
from bajo.rt.material_types import MaterialKind
from bajo.rt.shading import _sample_material, _evaluate_material
from bajo.rt.spectral_shading_types import (
    SpectralBsdfSample,
    SpectralBsdfEvaluation,
)


@always_inline
def sample_spectral_material[
    kind: MaterialKind, width: SIMDLength
](
    incident: Vec3[.float32, .WORLD, width],
    normal: Vec3[.float32, .WORLD, width],
    albedo: SampledSpectrum[width],
    parameter: SIMD[.float32, width],
    front_face: SIMD[.bool, width],
    u: SIMD[.float32, width],
    v: SIMD[.float32, width],
    roughness: SIMD[.float32, width] = 0,
    w: SIMD[.float32, width] = 0,
) -> SpectralBsdfSample[width]:
    # These geometric lobes do not use color to choose a direction. Their
    # scalar factor multiplies the actual sampled reflectance, never RGB.
    var sampled = _sample_material[kind, width](
        incident,
        normal,
        Vec3[.float32, .WORLD, width](1),
        parameter,
        front_face,
        u,
        v,
        roughness,
        w,
    )
    var weight = SampledSpectrum[width](sampled.weight.x)
    comptime if kind != .DIELECTRIC:
        weight = weight * albedo
    return SpectralBsdfSample[width](
        sampled.direction, weight, sampled.pdf, sampled.delta, sampled.ok
    )


@always_inline
def evaluate_spectral_material[
    kind: MaterialKind, width: SIMDLength
](
    incident: Vec3[.float32, .WORLD, width],
    normal: Vec3[.float32, .WORLD, width],
    albedo: SampledSpectrum[width],
    parameter: SIMD[.float32, width],
    outgoing: Vec3[.float32, .WORLD, width],
    roughness: SIMD[.float32, width] = 0,
    front_face: SIMD[.bool, width] = SIMD[.bool, width](fill=True),
) -> SpectralBsdfEvaluation[width]:
    var evaluated = _evaluate_material[kind, width](
        incident,
        normal,
        Vec3[.float32, .WORLD, width](1),
        parameter,
        outgoing,
        roughness,
        front_face,
    )
    var value = SampledSpectrum[width](evaluated.value.x)
    comptime if kind != .DIELECTRIC:
        value = value * albedo
    return SpectralBsdfEvaluation[width](value, evaluated.pdf, evaluated.delta)
