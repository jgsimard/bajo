"""Register material parameters used by scalar GPU and SIMD CPU shading."""

from bajo.core import Vec3
from bajo.core.sampled_spectrum import SampledSpectrum
from bajo.rt.material_types import CoatedDiffuse


@fieldwise_init
struct SpectralCoatedParameters[length: SIMDLength = 1](
    TrivialRegisterPassable
):
    var roughness: SIMD[.float32, Self.length]
    var eta: SIMD[.float32, Self.length]
    var thickness: SIMD[.float32, Self.length]
    var layer_albedo: SampledSpectrum[Self.length]
    var g: SIMD[.float32, Self.length]
    var max_depth: SIMD[.float32, Self.length]
    var n_samples: SIMD[.float32, Self.length]
    var conductor_eta: SampledSpectrum[Self.length]
    var conductor_k: SampledSpectrum[Self.length]
    var conductor_roughness: SIMD[.float32, Self.length]

    def __init__(out self):
        self.roughness = 1.0
        self.eta = 1.5
        self.thickness = 0.01
        self.layer_albedo = SampledSpectrum[Self.length](0.0)
        self.g = 0.0
        self.max_depth = 10.0
        self.n_samples = 1.0
        self.conductor_eta = SampledSpectrum[Self.length](0.0)
        self.conductor_k = SampledSpectrum[Self.length](0.0)
        self.conductor_roughness = 0.0


@fieldwise_init
struct SpectralBsdfSample[length: SIMDLength = 1](Copyable, Writable):
    var direction: Vec3[.float32, .WORLD, Self.length]
    var weight: SampledSpectrum[Self.length]
    var pdf: SIMD[.float32, Self.length]
    var delta: SIMD[.bool, Self.length]
    var ok: SIMD[.bool, Self.length]


@fieldwise_init
struct SpectralBsdfEvaluation[length: SIMDLength = 1](Copyable, Writable):
    var value: SampledSpectrum[Self.length]
    var pdf: SIMD[.float32, Self.length]
    var delta: SIMD[.bool, Self.length]
