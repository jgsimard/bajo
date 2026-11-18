"""CPU material setup at the camera sample's actual wavelengths."""

from std.math import sqrt
from bajo.core import Vec3
from bajo.core.sampled_spectrum import SampledSpectrum, SampledWavelengths
from bajo.core.rgb_spectrum import RgbSpectrumView
from bajo.core.random import Rng
from bajo.rt.material_types import (
    SurfaceStore,
    SurfaceId,
    MaterialKind,
    NO_TEXTURE,
)
from bajo.rt.spectrum_sources import SpectrumSourcesView, NO_SPECTRUM
from bajo.rt.spectral_shading import (
    sample_spectral_material,
    evaluate_spectral_material,
)
from bajo.rt.spectral_shading_types import (
    SpectralCoatedParameters,
    SpectralBsdfSample,
    SpectralBsdfEvaluation,
)
from bajo.rt.coated_spectral import (
    _sample_coated_diffuse,
    _evaluate_coated_diffuse,
)
from bajo.rt.shading import METAL_DELTA_FUZZ_THRESHOLD, dielectric_geometry
from .bsdf import _MaterialParameters


struct SpectralMaterialParameters[width: SIMDLength]:
    var albedo: SampledSpectrum[Self.width]
    var parameter: SIMD[.float32, Self.width]
    var roughness: SIMD[.float32, Self.width]
    var coated: SpectralCoatedParameters[Self.width]

    def __init__(out self):
        self.albedo = {}
        self.parameter = 1
        self.roughness = 0
        self.coated = {}

    def load[
        kind: MaterialKind
    ](
        mut self,
        surfaces: SurfaceStore,
        spectra: SpectrumSourcesView,
        indices: SIMD[.uint32, Self.width],
        u: SIMD[.float32, Self.width],
        v: SIMD[.float32, Self.width],
        active: SIMD[.bool, Self.width],
        mut wavelengths: SampledWavelengths[Self.width],
        fields: ImmSpan[Float32, _],
    ):
        if not active.reduce_or():
            return
        var base = _MaterialParameters[Self.width]()
        base.load[kind, False, False](surfaces, indices, u, v, active, fields)
        self.parameter = base.parameter
        self.roughness = base.roughness
        comptime if kind == .COATED_DIFFUSE:
            self.coated.roughness = base.coated.roughness
            self.coated.eta = base.coated.eta
            self.coated.thickness = base.coated.thickness
            self.coated.g = base.coated.g
            self.coated.max_depth = base.coated.max_depth
            self.coated.n_samples = base.coated.n_samples
            self.coated.conductor_roughness = base.coated.conductor_roughness
        # Width one avoids packet metadata/masking overhead; keep its direct
        # source dispatch while using the same wavelength reconstruction APIs.
        comptime if Self.width == 1:
            for lane in range(Int(Self.width)):
                if not active[lane]:
                    continue
                var surface = SurfaceId[1](kind, indices[lane])
                var lambda_sample = wavelengths.get(lane)
                comptime if kind == .DIELECTRIC:
                    ref material = surfaces.dielectrics[Int(indices[lane])]
                    var source = surfaces.dielectric_eta_sources.get(
                        surface.value[0]
                    ).or_else(NO_SPECTRUM)
                    if source != NO_SPECTRUM:
                        self.parameter[lane] = spectra.sample(
                            source, lambda_sample, RgbSpectrumView.UNBOUNDED
                        ).s0
                        var descriptor = spectra.sources[
                            unsafe_offset=Int(source)
                        ].copy()
                        var dispersive = (
                            descriptor.kind == descriptor.CURVE
                            or descriptor.kind == descriptor.PRODUCT
                            or (
                                descriptor.kind == descriptor.RGB
                                and (
                                    descriptor.value.x != descriptor.value.y
                                    or descriptor.value.y != descriptor.value.z
                                )
                            )
                        )
                        if dispersive:
                            lambda_sample.terminate_secondary(True)
                            wavelengths.set(lane, lambda_sample)
                    elif material.eta_spectrum != NO_TEXTURE:
                        self.parameter[lane] = spectra.curves.sample(
                            SIMD[.uint32, 1](material.eta_spectrum),
                            lambda_sample,
                        ).s0
                        lambda_sample.terminate_secondary(True)
                        wavelengths.set(lane, lambda_sample)
                else:
                    var metadata = surfaces._albedo_metadata(surface)
                    var albedo: SampledSpectrum[1]
                    if case .Some(source) = surfaces.albedo_spectra.get(
                        surface.value[0]
                    ):
                        albedo = spectra.sample(
                            source, lambda_sample, RgbSpectrumView.ALBEDO
                        )
                    else:
                        albedo = spectra.rgb.sample(
                            metadata[0].clamp(0, 1),
                            lambda_sample,
                            RgbSpectrumView.ALBEDO,
                        )
                    if metadata[1] != NO_TEXTURE:
                        var texel = surfaces.image_textures[
                            Int(metadata[1])
                        ].sample(u[lane], v[lane])
                        albedo = albedo * spectra.rgb.sample(
                            texel.clamp(0, 1),
                            lambda_sample,
                            RgbSpectrumView.ALBEDO,
                        )
                    self.albedo.set(lane, albedo.clamp(0, 1))
                comptime if kind == .COATED_DIFFUSE:
                    ref material = surfaces.coated_diffuses[Int(indices[lane])]
                    var eta = material.eta
                    var layer = spectra.rgb.sample(
                        material.layer_albedo,
                        lambda_sample,
                        RgbSpectrumView.ALBEDO,
                    )
                    if case .Some(source) = surfaces.layer_albedo_spectra.get(
                        surface.value[0]
                    ):
                        layer = spectra.sample(
                            source, lambda_sample, RgbSpectrumView.ALBEDO
                        )
                    self.coated.layer_albedo.set(lane, layer.clamp(0, 1))
                    var conductor_eta = spectra.rgb.sample(
                        material.conductor_eta,
                        lambda_sample,
                        RgbSpectrumView.UNBOUNDED,
                    )
                    var conductor_k = spectra.rgb.sample(
                        material.conductor_k,
                        lambda_sample,
                        RgbSpectrumView.UNBOUNDED,
                    )
                    if material.conductor_eta_spectrum != NO_TEXTURE:
                        conductor_eta = (
                            spectra.curves.sample(
                                SIMD[.uint32, 1](
                                    material.conductor_eta_spectrum
                                ),
                                lambda_sample,
                            )
                            / eta
                        )
                    if material.conductor_k_spectrum != NO_TEXTURE:
                        conductor_k = (
                            spectra.curves.sample(
                                SIMD[.uint32, 1](material.conductor_k_spectrum),
                                lambda_sample,
                            )
                            / eta
                        )
                    if material.conductor_reflectance:
                        conductor_eta = SampledSpectrum[1](1 / eta)
                        comptime for channel in range(4):
                            var r = self.albedo[channel][lane].clamp(0, 0.9999)
                            conductor_k[channel] = 2 * sqrt(r / (1 - r)) / eta
                    self.coated.conductor_eta.set(lane, conductor_eta)
                    self.coated.conductor_k.set(lane, conductor_k)
            return
        var source_ids = SIMD[.uint32, Self.width](NO_SPECTRUM)
        var curve_ids = SIMD[.uint32, Self.width](NO_TEXTURE)
        var layer_ids = SIMD[.uint32, Self.width](NO_SPECTRUM)
        var eta_curves = SIMD[.uint32, Self.width](NO_TEXTURE)
        var k_curves = SIMD[.uint32, Self.width](NO_TEXTURE)
        var dispersive = SIMD[.bool, Self.width](fill=False)
        var reflectance = SIMD[.bool, Self.width](fill=False)
        # Sparse authoring bindings remain scalar metadata; wavelength math,
        # texture reconstruction and expression evaluation run across packets.
        for lane in range(Int(Self.width)):
            if not active[lane]:
                continue
            var surface = SurfaceId[1](kind, indices[lane])
            comptime if kind == .DIELECTRIC:
                source_ids[lane] = surfaces.dielectric_eta_sources.get(
                    surface.value[0]
                ).or_else(NO_SPECTRUM)
                if source_ids[lane] != NO_SPECTRUM:
                    var descriptor = spectra.sources[
                        unsafe_offset=Int(source_ids[lane])
                    ].copy()
                    dispersive[lane] = (
                        descriptor.kind == descriptor.CURVE
                        or descriptor.kind == descriptor.PRODUCT
                        or (
                            descriptor.kind == descriptor.RGB
                            and (
                                descriptor.value.x != descriptor.value.y
                                or descriptor.value.y != descriptor.value.z
                            )
                        )
                    )
                else:
                    curve_ids[lane] = surfaces.dielectrics[
                        Int(indices[lane])
                    ].eta_spectrum
                    dispersive[lane] = curve_ids[lane] != NO_TEXTURE
            else:
                source_ids[lane] = surfaces.albedo_spectra.get(
                    surface.value[0]
                ).or_else(NO_SPECTRUM)
            comptime if kind == .COATED_DIFFUSE:
                ref material = surfaces.coated_diffuses[Int(indices[lane])]
                layer_ids[lane] = surfaces.layer_albedo_spectra.get(
                    surface.value[0]
                ).or_else(NO_SPECTRUM)
                eta_curves[lane] = material.conductor_eta_spectrum
                k_curves[lane] = material.conductor_k_spectrum
                reflectance[lane] = material.conductor_reflectance
        comptime if kind == .DIELECTRIC:
            self.parameter = spectra.sample_packet(
                source_ids,
                wavelengths,
                RgbSpectrumView.UNBOUNDED,
                active,
                SampledSpectrum[Self.width](self.parameter),
            ).s0
            self.parameter = spectra.curves.sample(
                curve_ids, wavelengths, self.parameter
            ).s0
            wavelengths.terminate_secondary(dispersive)
        else:
            var metadata = surfaces._albedo_metadata(
                SurfaceId[Self.width](SurfaceId.pack_raw(kind, 0) | indices),
                active,
            )
            var albedo = spectra.rgb.sample(
                metadata[0].clamp(0, 1), wavelengths, RgbSpectrumView.ALBEDO
            )
            albedo = spectra.sample_packet(
                source_ids, wavelengths, RgbSpectrumView.ALBEDO, active, albedo
            )
            var textured = active & metadata[1].ne(NO_TEXTURE)
            if textured.reduce_or():
                var texel = surfaces.sample_image_textures(metadata[1], u, v)
                var modulation = spectra.rgb.sample(
                    texel.clamp(0, 1), wavelengths, RgbSpectrumView.ALBEDO
                )
                comptime for channel in range(4):
                    albedo[channel] = textured.select(
                        albedo[channel] * modulation[channel], albedo[channel]
                    )
            comptime for channel in range(4):
                self.albedo[channel] = active.select(
                    albedo[channel].clamp(0, 1), self.albedo[channel]
                )
        comptime if kind == .COATED_DIFFUSE:
            var layer = spectra.rgb.sample(
                base.coated.layer_albedo, wavelengths, RgbSpectrumView.ALBEDO
            )
            layer = spectra.sample_packet(
                layer_ids, wavelengths, RgbSpectrumView.ALBEDO, active, layer
            )
            var eta = base.coated.eta
            var n = spectra.rgb.sample(
                base.coated.conductor_eta,
                wavelengths,
                RgbSpectrumView.UNBOUNDED,
            )
            var k = spectra.rgb.sample(
                base.coated.conductor_k, wavelengths, RgbSpectrumView.UNBOUNDED
            )
            var measured_n = (
                spectra.curves.sample(eta_curves, wavelengths) / eta
            )
            var measured_k = spectra.curves.sample(k_curves, wavelengths) / eta
            comptime for channel in range(4):
                n[channel] = eta_curves.ne(NO_TEXTURE).select(
                    measured_n[channel], n[channel]
                )
                k[channel] = k_curves.ne(NO_TEXTURE).select(
                    measured_k[channel], k[channel]
                )
                var r = self.albedo[channel].clamp(0, 0.9999)
                n[channel] = reflectance.select(1 / eta, n[channel])
                k[channel] = reflectance.select(
                    2 * sqrt(r / (1 - r)) / eta, k[channel]
                )
                self.coated.layer_albedo[channel] = active.select(
                    layer[channel].clamp(0, 1),
                    self.coated.layer_albedo[channel],
                )
                self.coated.conductor_eta[channel] = active.select(
                    n[channel], self.coated.conductor_eta[channel]
                )
                self.coated.conductor_k[channel] = active.select(
                    k[channel], self.coated.conductor_k[channel]
                )

    def sample[
        kind: MaterialKind
    ](
        self,
        incident: Vec3[.float32, .WORLD, Self.width],
        normal: Vec3[.float32, .WORLD, Self.width],
        front_face: SIMD[.bool, Self.width],
        mut rng: Rng[Self.width],
        estimator_seed: SIMD[.uint32, Self.width],
        active: SIMD[.bool, Self.width],
    ) -> SpectralBsdfSample[Self.width]:
        var u = SIMD[.float32, Self.width](0)
        var v = SIMD[.float32, Self.width](0)
        var w = SIMD[.float32, Self.width](0)
        var eligible = active
        comptime if kind == .DIELECTRIC:
            var ri = front_face.select(1 / self.parameter, self.parameter)
            var geometry = dielectric_geometry(incident, normal, ri)
            var rough = active & self.roughness.ge(1e-3) & self.parameter.ne(1)
            eligible &= geometry.transmitted_sine.le(1) | rough
            if eligible.reduce_or():
                u = eligible.select(rng.f32(), 0)
            if rough.reduce_or():
                v = rough.select(rng.f32(), 0)
                w = rough.select(rng.f32(), 0)
        else:
            comptime if kind == .METAL:
                eligible &= self.parameter.gt(METAL_DELTA_FUZZ_THRESHOLD)
            if eligible.reduce_or():
                u = eligible.select(rng.f32(), 0)
                v = eligible.select(rng.f32(), 0)
        var sampled: SpectralBsdfSample[Self.width]
        comptime if kind == .COATED_DIFFUSE:
            sampled = _sample_coated_diffuse(
                incident,
                normal,
                self.albedo,
                self.coated,
                u,
                v,
                active,
                estimator_seed,
            )
        else:
            sampled = sample_spectral_material[kind, Self.width](
                incident,
                normal,
                self.albedo,
                self.parameter,
                front_face,
                u,
                v,
                self.roughness,
                w,
            )
        sampled.ok &= active
        return sampled^

    def evaluate[
        kind: MaterialKind
    ](
        self,
        incident: Vec3[.float32, .WORLD, Self.width],
        normal: Vec3[.float32, .WORLD, Self.width],
        outgoing: Vec3[.float32, .WORLD, Self.width],
        estimator_seed: SIMD[.uint32, Self.width],
        active: SIMD[.bool, Self.width],
        front_face: SIMD[.bool, Self.width],
    ) -> SpectralBsdfEvaluation[Self.width]:
        comptime if kind == .COATED_DIFFUSE:
            return _evaluate_coated_diffuse(
                incident,
                normal,
                self.albedo,
                self.coated,
                outgoing,
                active,
                estimator_seed,
            )
        else:
            return evaluate_spectral_material[kind, Self.width](
                incident,
                normal,
                self.albedo,
                self.parameter,
                outgoing,
                self.roughness,
                front_face,
            )
