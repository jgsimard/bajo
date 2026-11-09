"""Spectral packet transport with radiometric state owned by stable sample IDs."""

from std.math import abs, iota, exp
from bajo.bvh.cpu.traversal_mode import CpuTraversalMode
from bajo.core import Point3, Vec3, Ray, dot, length as vector_length
from bajo.core.color import Color
from bajo.core.sampled_spectrum import (
    SampledSpectrum,
    SampledWavelengths,
    CIE_Y_INTEGRAL,
)
from bajo.core.rgb_spectrum import RgbSpectrumView
from bajo.rt.material_types import (
    MaterialKind,
    Integrator,
    SurfaceId,
    EnvironmentKind,
)
from bajo.rt.render_types import RenderSettings, SamplingConfig
from bajo.rt.spectrum_sources import SpectrumSourcesView, NO_SPECTRUM
from bajo.rt.shading_types import SurfaceHit, ShadingPoint
from bajo.rt.wavefront_queue import PacketPathQueue, PathPacket
from bajo.rt.path_values import PathContinuation
from bajo.rt.continuation import continuation_medium
from bajo.rt.common import (
    path_stage_rng,
    path_stage_seed,
    environment_radiance,
    RUSSIAN_ROULETTE_START_DEPTH,
    RUSSIAN_ROULETTE_MIN_SURVIVAL,
    RUSSIAN_ROULETTE_MAX_SURVIVAL,
)
from bajo.rt.wavefront_contract import (
    wavefront_rng_stage,
    wavefront_rng_light_stage,
    wavefront_rng_material_stage,
    wavefront_rng_roulette_stage,
)
from bajo.rt.rays import spawn_surface_ray
from bajo.rt.lighting import (
    _direct_light_scale,
    _emissive_hit_light_pdf,
    _emissive_hit_weight_from_pdf,
)
from bajo.rt.spectral_light_distribution import SpectralLightDistributionView
from bajo.rt.cpu.scene import CpuScene
from bajo.rt.cpu.spectral_bsdf import SpectralMaterialParameters
from bajo.rt.cpu.lighting import (
    _sample_direct_light_candidate,
    _emissive_hit_weight,
    _environment_miss_weight,
)
from bajo.core.mappings import equal_area_sphere_to_square


struct SpectralSampleState:
    var wavelengths: List[SampledWavelengths[1]]
    var throughput: List[SampledSpectrum[1]]
    var radiance: List[SampledSpectrum[1]]
    var eta_scale: List[Float32]

    def __init__(out self):
        self.wavelengths = List[SampledWavelengths[1]]()
        self.throughput = List[SampledSpectrum[1]]()
        self.radiance = List[SampledSpectrum[1]]()
        self.eta_scale = List[Float32]()

    def initialize(mut self, count: Int, base: Int, sampling: SamplingConfig):
        self.wavelengths.clear()
        self.throughput.clear()
        self.radiance.clear()
        self.eta_scale.clear()
        self.wavelengths.reserve(count)
        self.throughput.reserve(count)
        self.radiance.reserve(count)
        self.eta_scale.reserve(count)
        for index in range(count):
            var rng = path_stage_rng(sampling, UInt32(base + index), UInt32(0))
            _ = rng.f32()
            _ = rng.f32()
            _ = rng.f32()
            _ = rng.f32()
            self.wavelengths.append(SampledWavelengths[1].visible(rng.f32()))
            self.throughput.append(SampledSpectrum[1](1))
            self.radiance.append(SampledSpectrum[1]())
            self.eta_scale.append(1)


@always_inline
def _environment[
    width: SIMDLength, world_width: SIMDLength, instance_width: SIMDLength
](
    world: CpuScene[world_width, instance_width],
    spectra: SpectrumSourcesView,
    direction: Vec3[.float32, .WORLD, width],
    wavelengths: SampledWavelengths[width],
) -> SampledSpectrum[width]:
    ref data = world.scene_data()
    ref environment = data.environment()
    var result = SampledSpectrum[width]()
    if environment.kind == .BLACK:
        return result
    var fallback = environment_radiance(environment, data.surfaces(), direction)
    for lane in range(Int(width)):
        var lambda_sample = wavelengths.get(lane)
        var value: SampledSpectrum[1]
        if environment.spectrum == NO_SPECTRUM:
            var rgb = Color(
                fallback.x[lane], fallback.y[lane], fallback.z[lane]
            )
            if environment.kind == .IMAGE:
                rgb = environment.scale
            value = (
                spectra.rgb.sample(
                    rgb,
                    lambda_sample,
                    RgbSpectrumView.ILLUMINANT,
                )
                / CIE_Y_INTEGRAL
            )
        else:
            value = spectra.emission(environment.spectrum, lambda_sample)
        if environment.kind == .IMAGE:
            var local = environment.world_to_light.vector(
                Vec3[.float32, .WORLD](
                    direction.x[lane], direction.y[lane], direction.z[lane]
                )
            )
            var uv = equal_area_sphere_to_square(local)
            var texel = (
                data.surfaces()
                .image_textures[Int(environment.texture_index)]
                .sample_environment(uv.u, uv.v)
            )
            value = value * spectra.rgb.sample(
                texel, lambda_sample, RgbSpectrumView.UNBOUNDED
            )
        result.set(lane, value)
    return result


@always_inline
def _absorption[
    width: SIMDLength, world_width: SIMDLength, instance_width: SIMDLength
](
    world: CpuScene[world_width, instance_width],
    spectra: SpectrumSourcesView,
    ids: SIMD[.uint32, width],
    distance: SIMD[.float32, width],
    wavelengths: SampledWavelengths[width],
    active: SIMD[.bool, width],
) -> SampledSpectrum[width]:
    var result = SampledSpectrum[width](1)
    ref media = world.scene_data().media()
    for lane in range(Int(width)):
        if not active[lane] or ids[lane] == 0:
            continue
        var index = Int(ids[lane])
        var lambda_sample = wavelengths.get(lane)
        var sigma = spectra.rgb.sample(
            media.absorption[index], lambda_sample, RgbSpectrumView.UNBOUNDED
        )
        if media.spectra[index] != NO_SPECTRUM:
            sigma = spectra.sample(
                media.spectra[index], lambda_sample, RgbSpectrumView.UNBOUNDED
            )
        var attenuation = SampledSpectrum[1](1)
        comptime for channel in range(4):
            if sigma[channel] > 0:
                var depth = sigma[channel] * distance[lane]
                attenuation[channel] = depth.lt(80).select(
                    exp(-min(depth, 80)), 0
                )
        result.set(lane, attenuation)
    return result


def _shade[
    kind: MaterialKind,
    width: SIMDLength,
    integrator: Integrator,
    world_width: SIMDLength,
    instance_width: SIMDLength,
](
    settings: RenderSettings,
    sampling: SamplingConfig,
    world: CpuScene[world_width, instance_width],
    spectra: SpectrumSourcesView,
    lighting: SpectralLightDistributionView,
    packet: PathPacket[width],
    ray: Ray[.float32, .WORLD, width],
    hit: SurfaceHit[width],
    throughput: SampledSpectrum[width],
    mut wavelengths: SampledWavelengths[width],
    active: SIMD[.bool, width],
    lane_count: Int,
    bounce: Int,
    sample_base: Int,
    mut state: SpectralSampleState,
    mut next_paths: PacketPathQueue[width],
):
    var mask = active & (hit.surface.value >> UInt32(28)).eq(kind.value())
    if not mask.reduce_or():
        return
    var stage = wavefront_rng_stage(UInt32(bounce))
    var material = SpectralMaterialParameters[width]()
    material.load[kind](
        world.scene_data().surfaces(),
        spectra,
        hit.surface.value & UInt32(0x0FFFFFFF),
        hit.uv_u,
        hit.uv_v,
        mask,
        wavelengths,
        world.coated_materials(),
    )
    var beta = throughput
    comptime for channel in range(4):
        beta[channel] = wavelengths.pdf[channel].gt(0).select(beta[channel], 0)
    var point = ShadingPoint[width].from_hit(ray, hit)
    var estimator_seed = path_stage_seed(sampling, packet.path_ids, stage)
    comptime if Integrator.uses_direct_lighting[integrator]:
        var rng = path_stage_rng(
            sampling, packet.path_ids, wavefront_rng_light_stage(UInt32(bounce))
        )
        var light = _sample_direct_light_candidate(
            world,
            point,
            rng,
            mask,
            SIMD[.bool, width](fill=kind == .DIELECTRIC),
            Optional(lighting),
        )
        if light.valid.reduce_or():
            var emission = SampledSpectrum[width]()
            var environment = _environment(
                world, spectra, light.direction, wavelengths
            )
            for lane in range(lane_count):
                if not light.valid[lane]:
                    continue
                var lambda_sample = wavelengths.get(lane)
                var value: SampledSpectrum[1]
                if light.is_environment[lane]:
                    value = environment.get(lane)
                elif light.spectrum_ids[lane] != NO_SPECTRUM:
                    value = spectra.emission(
                        light.spectrum_ids[lane], lambda_sample
                    )
                else:
                    value = (
                        spectra.rgb.sample(
                            Color(
                                light.emission.x[lane],
                                light.emission.y[lane],
                                light.emission.z[lane],
                            ),
                            lambda_sample,
                            RgbSpectrumView.ILLUMINANT,
                        )
                        / CIE_Y_INTEGRAL
                    )
                emission.set(lane, value)
            var shadow = spawn_surface_ray(
                point.p, light.direction, light.shadow_t_max
            )
            var visible = light.valid & ~world.occluded(shadow, light.valid)
            var evaluated = material.evaluate[kind](
                ray.d,
                hit.normal,
                light.direction,
                estimator_seed,
                visible,
                hit.front_face,
            )
            var scale = _direct_light_scale[integrator, width](
                light.surface_cosine, light.light_pdf, evaluated.pdf, visible
            )
            var shadow_media = continuation_medium(
                packet.medium_ids,
                hit.medium_boundary,
                hit.front_face,
                light.direction,
                hit.geometric_normal,
            )
            var attenuation = _absorption(
                world,
                spectra,
                shadow_media,
                light.shadow_t_max,
                wavelengths,
                visible,
            )
            var contribution = (
                beta * evaluated.value * emission * attenuation * scale
            )
            for lane in range(lane_count):
                if visible[lane]:
                    state.radiance[
                        Int(packet.path_ids[lane]) - sample_base
                    ] += contribution.get(lane)
    var rng = path_stage_rng(sampling, packet.path_ids, UInt32(bounce + 1))
    var sampled = material.sample[kind](
        ray.d,
        hit.normal,
        hit.front_face,
        rng,
        path_stage_seed(sampling, packet.path_ids, UInt32(bounce + 1)),
        mask,
    )
    var continued = beta * sampled.weight
    var survived = mask & sampled.ok
    var media = continuation_medium(
        packet.medium_ids,
        hit.medium_boundary,
        hit.front_face,
        sampled.direction,
        hit.geometric_normal,
    )
    var eta_scales = SIMD[.float32, width](1)
    for lane in range(lane_count):
        if mask[lane]:
            var id = Int(packet.path_ids[lane]) - sample_base
            state.wavelengths[id] = wavelengths.get(lane)
            comptime if kind == .DIELECTRIC:
                if dot(sampled.direction, hit.geometric_normal)[lane] < 0:
                    var eta = material.parameter[lane]
                    state.eta_scale[id] *= (
                        eta * eta if hit.front_face[lane] else 1 / (eta * eta)
                    )
            eta_scales[lane] = state.eta_scale[id]
    if UInt32(bounce + 1) >= RUSSIAN_ROULETTE_START_DEPTH:
        var importance = wavelengths.pdf.s0.gt(0).select(continued.s0, 0)
        comptime for channel in range(1, 4):
            importance = max(
                importance,
                wavelengths.pdf[channel].gt(0).select(continued[channel], 0),
            )
        importance *= eta_scales
        var probability = importance.clamp(
            RUSSIAN_ROULETTE_MIN_SURVIVAL, RUSSIAN_ROULETTE_MAX_SURVIVAL
        )
        var roulette = path_stage_rng(
            sampling,
            packet.path_ids,
            wavefront_rng_roulette_stage(UInt32(bounce)),
        )
        survived &= importance.gt(0) & roulette.f32().lt(probability)
        continued = continued / survived.select(probability, 1)
    for lane in range(lane_count):
        if mask[lane]:
            state.throughput[
                Int(packet.path_ids[lane]) - sample_base
            ] = continued.get(lane)
    # Geometry queues reference sample-owned radiometry. Their historical RGB
    # fields are not used for spectral throughput or accumulation.
    next_paths.append_continuation(
        PathContinuation[width](
            packet.path_ids,
            ray.at(hit.t),
            sampled.direction,
            Vec3[.float32, .WORLD, width](1),
            sampled.pdf,
            sampled.delta,
            media,
        ),
        survived,
        lane_count,
    )


def trace_spectral_packets[
    width: SIMDLength,
    integrator: Integrator,
    world_width: SIMDLength,
    instance_width: SIMDLength,
    traversal_mode: CpuTraversalMode,
    *adaptive_packet_sizes: SIMDLength,
](
    settings: RenderSettings,
    sampling: SamplingConfig,
    world: CpuScene[world_width, instance_width],
    spectra: SpectrumSourcesView,
    lighting: SpectralLightDistributionView,
    mut state: SpectralSampleState,
    mut paths: PacketPathQueue[width],
    mut next_paths: PacketPathQueue[width],
    sample_base: Int,
    sample_count: Int,
):
    state.initialize(sample_count, sample_base, sampling)
    for bounce in range(settings.max_depth + 1):
        if len(paths) == 0:
            break
        next_paths.clear()
        for packet_index in range(len(paths.packets)):
            ref packet = paths.packets[packet_index]
            var count = min(Int(width), len(paths) - packet_index * Int(width))
            var active = iota[.int, width]().lt(count)
            var wavelengths = SampledWavelengths[width].visible(
                SIMD[.float32, width](0.5)
            )
            var beta = SampledSpectrum[width]()
            for lane in range(count):
                var id = Int(packet.path_ids[lane]) - sample_base
                wavelengths.set(lane, state.wavelengths[id])
                beta.set(lane, state.throughput[id])
            var ray = Ray[.float32, .WORLD, width](
                Point3[.float32, .WORLD, width](
                    packet.ox, packet.oy, packet.oz
                ),
                Vec3[.float32, .WORLD, width](packet.dx, packet.dy, packet.dz),
                packet.t_min,
                packet.t_max,
            )
            var hit = world.trace_surface_configured[
                width, traversal_mode, *adaptive_packet_sizes
            ](ray, active)
            var mix_rng = path_stage_rng(
                sampling,
                packet.path_ids,
                wavefront_rng_material_stage(UInt32(bounce)),
            )
            var mix_random = mix_rng.f32()
            for lane in range(count):
                if hit.hit[lane] and hit.surface.get(lane).kind() == .MIX:
                    hit.surface.value[lane] = (
                        world.scene_data()
                        .surfaces()
                        .resolve_mix(
                            hit.surface.get(lane),
                            mix_random[lane],
                            hit.uv_u[lane],
                            hit.uv_v[lane],
                        )
                        .value
                    )
            if world.has_shading_normals():
                hit.normal = (
                    world.scene_data()
                    .surfaces()
                    .shading_normal(
                        hit.surface,
                        hit.normal,
                        hit.uv_u,
                        hit.uv_v,
                        hit.hit & active,
                        hit.dpdu,
                        hit.front_face,
                    )
                )
            var distance = hit.hit.select(hit.t, packet.t_max) * vector_length(
                ray.d
            )
            beta = beta * _absorption(
                world, spectra, packet.medium_ids, distance, wavelengths, active
            )
            var environment = _environment(world, spectra, ray.d, wavelengths)
            var rgb_environment = environment_radiance(
                world.scene_data().environment(),
                world.scene_data().surfaces(),
                ray.d,
            )
            if world.scene_data().environment().kind == .IMAGE:
                ref environment_desc = world.scene_data().environment()
                ref transform = environment_desc.world_to_light
                var local = Vec3[.float32, .LOCAL, width](
                    transform.m00 * ray.d.x
                    + transform.m01 * ray.d.y
                    + transform.m02 * ray.d.z,
                    transform.m10 * ray.d.x
                    + transform.m11 * ray.d.y
                    + transform.m12 * ray.d.z,
                    transform.m20 * ray.d.x
                    + transform.m21 * ray.d.y
                    + transform.m22 * ray.d.z,
                )
                var uv = equal_area_sphere_to_square(local)
                rgb_environment = (
                    world.scene_data()
                    .surfaces()
                    .image_textures[Int(environment_desc.texture_index)]
                    .sample_environment(uv.u, uv.v)
                )
            var miss_pdf = lighting.environment_pdf(
                world.scene_data().environment().kind, rgb_environment
            )
            var miss_weight = _emissive_hit_weight_from_pdf[integrator](
                UInt32(bounce), packet.deltas, packet.bsdf_pdfs, miss_pdf
            )
            var importance = Vec3[.float32, .WORLD, width](0)
            for lane in range(count):
                if hit.hit[lane]:
                    var rgb = Color(
                        hit.emission.x[lane],
                        hit.emission.y[lane],
                        hit.emission.z[lane],
                    )
                    var value = lighting.emission(
                        hit.emission_spectrum[lane], rgb
                    )
                    importance.x[lane] = value
                    importance.y[lane] = value
                    importance.z[lane] = value
            var hit_pdf = _emissive_hit_light_pdf(
                ray.d,
                hit.t,
                hit.geometric_normal,
                importance,
                lighting.total_weight,
                hit.emission_two_sided,
            )
            var hit_weight = _emissive_hit_weight_from_pdf[integrator](
                UInt32(bounce), packet.deltas, packet.bsdf_pdfs, hit_pdf
            )
            for lane in range(count):
                var id = Int(packet.path_ids[lane]) - sample_base
                if not hit.hit[lane]:
                    state.radiance[id] += (
                        beta * environment * miss_weight
                    ).get(lane)
                elif hit.emission_visible[lane] and (
                    hit.front_face[lane] or hit.emission_two_sided[lane]
                ):
                    var emission: SampledSpectrum[1]
                    if hit.emission_spectrum[lane] != NO_SPECTRUM:
                        emission = spectra.emission(
                            hit.emission_spectrum[lane], wavelengths.get(lane)
                        )
                    else:
                        emission = (
                            spectra.rgb.sample(
                                Color(
                                    hit.emission.x[lane],
                                    hit.emission.y[lane],
                                    hit.emission.z[lane],
                                ),
                                wavelengths.get(lane),
                                RgbSpectrumView.ILLUMINANT,
                            )
                            / CIE_Y_INTEGRAL
                        )
                    state.radiance[id] += (
                        beta.get(lane) * emission * hit_weight[lane]
                    )
            if bounce == settings.max_depth:
                continue
            comptime for kind in ParameterList.of[
                MaterialKind.LAMBERTIAN,
                MaterialKind.METAL,
                MaterialKind.DIELECTRIC,
                MaterialKind.COATED_DIFFUSE,
            ]():
                _shade[kind, width, integrator, world_width, instance_width](
                    settings,
                    sampling,
                    world,
                    spectra,
                    lighting,
                    packet,
                    ray,
                    hit,
                    beta,
                    wavelengths,
                    active & hit.hit,
                    count,
                    bounce,
                    sample_base,
                    state,
                    next_paths,
                )
        paths, next_paths = next_paths^, paths^
