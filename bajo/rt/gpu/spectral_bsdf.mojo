"""GPU setup of the shared four-wavelength BSDF register values."""

from std.math import sqrt
from bajo.core.color import Color
from bajo.core.sampled_spectrum import SampledSpectrum, SampledWavelengths
from bajo.core.rgb_spectrum import RgbSpectrumView
from bajo.rt.cpu.spectral_bsdf import SpectralMaterialParameters
from bajo.rt.material_types import MaterialKind, SurfaceId, NO_TEXTURE
from bajo.rt.spectrum_sources import NO_SPECTRUM
from bajo.rt.material_storage import (
    CoatedDiffuseLayout,
    MetalLayout,
    DielectricLayout,
)
from bajo.rt.gpu.views import GpuRtSceneView
from bajo.rt.gpu.textures import (
    _sample_gpu_coated_parameters,
    _sample_gpu_image_texture,
)


@always_inline
def load_spectral_material[
    kind: MaterialKind
](
    scene: GpuRtSceneView,
    index: Int,
    u: Float32,
    v: Float32,
    mut wavelengths: SampledWavelengths[1],
) -> SpectralMaterialParameters[1]:
    var result = SpectralMaterialParameters[1]()
    var spectral = scene.spectral.value()
    var spectra = spectral.sources
    var rgb = Color(0)
    var texture = NO_TEXTURE
    var source = NO_SPECTRUM
    var texture_descs = scene.texture_descs.unsafe_origin_cast[ImmutAnyOrigin]()
    var texture_pixels = scene.texture_pixels.unsafe_origin_cast[
        ImmutAnyOrigin
    ]()
    comptime if kind == .LAMBERTIAN:
        rgb = Color(scene.lambertians.unsafe_offset(3 * index))
        texture = scene.lambertian_texture_indices[unsafe_offset=index]
        source = spectral.lambertians[unsafe_offset=index]
    elif kind == .METAL:
        rgb = Color(scene.metals.unsafe_offset(MetalLayout.STRIDE * index))
        result.parameter = scene.metals[
            unsafe_offset=MetalLayout.STRIDE * index + MetalLayout.FUZZ
        ]
        source = spectral.metals[unsafe_offset=index]
    elif kind == .DIELECTRIC:
        result.parameter = scene.dielectrics[
            unsafe_offset=DielectricLayout.STRIDE * index + DielectricLayout.ETA
        ]
        result.roughness = scene.dielectrics[
            unsafe_offset=DielectricLayout.STRIDE * index
            + DielectricLayout.ROUGHNESS
        ]
        var source = spectral.dielectrics[unsafe_offset=index]
        var eta = scene.spectra.dielectric_eta[unsafe_offset=index]
        if source != NO_SPECTRUM:
            result.parameter = spectra.sample(
                source, wavelengths, RgbSpectrumView.UNBOUNDED
            ).s0
            if spectral.dielectric_dispersive[unsafe_offset=index] != 0:
                wavelengths.terminate_secondary(True)
        elif eta != NO_SPECTRUM:
            result.parameter = spectra.curves.sample(
                SIMD[.uint32, 1](eta), wavelengths
            ).s0
            wavelengths.terminate_secondary(True)
        return result^
    else:
        comptime assert kind == .COATED_DIFFUSE
        var base = _sample_gpu_coated_parameters(
            scene.coated_diffuses.unsafe_origin_cast[ImmutAnyOrigin](),
            index,
            u,
            v,
            scene.coated_diffuse_texture_indices.unsafe_origin_cast[
                ImmutAnyOrigin
            ](),
            texture_descs,
            texture_pixels,
        )
        rgb = Color(
            scene.coated_diffuses.unsafe_offset(
                CoatedDiffuseLayout.STRIDE * index
            )
        )
        texture = scene.coated_diffuse_texture_indices[unsafe_offset=3 * index]
        source = spectral.coated[unsafe_offset=index]
        result.coated.roughness = base.roughness
        result.coated.eta = base.eta
        result.coated.thickness = base.thickness
        result.coated.g = base.g
        result.coated.max_depth = base.max_depth
        result.coated.n_samples = base.n_samples
        result.coated.conductor_roughness = base.conductor_roughness
        result.coated.layer_albedo = spectra.rgb.sample(
            base.layer_albedo, wavelengths, RgbSpectrumView.ALBEDO
        )
        var layer = spectral.layers[unsafe_offset=index]
        if layer != NO_SPECTRUM:
            result.coated.layer_albedo = spectra.sample(
                layer, wavelengths, RgbSpectrumView.ALBEDO
            )
        result.coated.conductor_eta = spectra.rgb.sample(
            base.conductor_eta, wavelengths, RgbSpectrumView.UNBOUNDED
        )
        result.coated.conductor_k = spectra.rgb.sample(
            base.conductor_k, wavelengths, RgbSpectrumView.UNBOUNDED
        )
        var eta = scene.spectra.coated_eta_k[unsafe_offset=2 * index]
        var k = scene.spectra.coated_eta_k[unsafe_offset=2 * index + 1]
        if eta != NO_SPECTRUM:
            result.coated.conductor_eta = (
                spectra.curves.sample(SIMD[.uint32, 1](eta), wavelengths)
                / base.eta
            )
        if k != NO_SPECTRUM:
            result.coated.conductor_k = (
                spectra.curves.sample(SIMD[.uint32, 1](k), wavelengths)
                / base.eta
            )
    result.albedo = spectra.rgb.sample(
        rgb.clamp(0, 1), wavelengths, RgbSpectrumView.ALBEDO
    )
    if source != NO_SPECTRUM:
        result.albedo = spectra.sample(
            source, wavelengths, RgbSpectrumView.ALBEDO
        )
    if texture != NO_TEXTURE:
        var texel = _sample_gpu_image_texture(
            texture, u, v, texture_descs, texture_pixels
        )
        result.albedo = result.albedo * spectra.rgb.sample(
            texel.clamp(0, 1), wavelengths, RgbSpectrumView.ALBEDO
        )
    result.albedo = result.albedo.clamp(0, 1)
    comptime if kind == .COATED_DIFFUSE:
        result.coated.layer_albedo = result.coated.layer_albedo.clamp(0, 1)
        if spectral.reflectance[unsafe_offset=index] != 0:
            result.coated.conductor_eta = SampledSpectrum[1](
                1 / result.coated.eta
            )
            comptime for channel in range(4):
                var r = result.albedo[channel].clamp(0, 0.9999)
                result.coated.conductor_k[channel] = (
                    2 * sqrt(r / (1 - r)) / result.coated.eta
                )
    return result^
