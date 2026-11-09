"""Four-wavelength GPU wavefront transport and complete-sample film reduction."""

from std.math import abs, exp, ceildiv, pi
from max.gpu import global_idx, block_dim, grid_dim
from max.gpu.host import DeviceContext
from bajo.core import Point3f32, Vec3f32, Rayf32, dot, length
from bajo.core.color import Color
from bajo.core.sampled_spectrum import (
    SampledSpectrum,
    SampledWavelengths,
    CIE_Y_INTEGRAL,
)
from bajo.core.rgb_spectrum import RgbSpectrumView
from bajo.core.mappings import equal_area_sphere_to_square
from bajo.rt.common import (
    path_stage_rng,
    path_stage_seed,
    RUSSIAN_ROULETTE_START_DEPTH,
    RUSSIAN_ROULETTE_MIN_SURVIVAL,
    RUSSIAN_ROULETTE_MAX_SURVIVAL,
)
from bajo.rt.continuation import continuation_medium
from bajo.rt.path_values import PathContinuation
from bajo.rt.material_types import (
    MaterialKind,
    Integrator,
    SurfaceId,
    NO_TEXTURE,
    EnvironmentKind,
)
from bajo.rt.spectrum_sources import NO_SPECTRUM
from bajo.rt.shading_types import SurfaceHit
from bajo.rt.film import FilmResponse
from bajo.rt.render_types import SamplingConfig, RenderSettings
from bajo.rt.lighting import (
    _direct_light_scale,
    _emissive_hit_light_pdf,
    _emissive_hit_weight_from_pdf,
)
from bajo.rt.opacity import opacity_accepts, opacity_point_sample
from bajo.rt.wavefront_contract import (
    DeviceWavePath,
    WAVE_COUNTER,
    WAVE_STATUS,
    wavefront_rng_stage,
    wavefront_rng_roulette_stage,
)
from bajo.rt.gpu.views import (
    GpuRtSceneView,
    GpuRtTraceQueueView,
    gpu_rt_trace_queue_view,
)
from bajo.rt.gpu.config import GpuRtSceneKind, GpuRtBvhFormat
from bajo.rt.gpu.spectral_state import GpuSpectralStateView
from bajo.rt.gpu.spectral_trace import spectral_trace_surface
from bajo.rt.gpu.spectral_bsdf import load_spectral_material
from bajo.rt.gpu.spectral_lighting import sample_spectral_light
from bajo.rt.gpu.sampling import GpuRtSamplingConfig
from bajo.rt.gpu.scene_trace import _trace_scene_any, _gpu_environment_radiance
from bajo.rt.gpu.emission import gpu_emitter
from bajo.rt.gpu.opacity import gpu_opacity_value
from bajo.rt.gpu.textures import _GpuImageTextureView
from bajo.rt.gpu.wavefront_contract import (
    load_gpu_rt_path,
    store_gpu_rt_path_values,
    _reserve_slot,
    _mark_status,
    enqueue_wavefront_advance,
)
from bajo.rt.rays import spawn_surface_ray


@always_inline
def _emitter_source(scene: GpuRtSceneView, primitive: UInt32) -> UInt32:
    var begin = 0
    var end = Int(scene.emission.count)
    while begin < end:
        var middle = (begin + end) // 2
        if scene.emission.keys_and_flags[unsafe_offset=2 * middle] < primitive:
            begin = middle + 1
        else:
            end = middle
    if (
        begin < Int(scene.emission.count)
        and scene.emission.keys_and_flags[unsafe_offset=2 * begin] == primitive
    ):
        return scene.spectral.value().emitters[unsafe_offset=begin]
    return NO_SPECTRUM


@always_inline
def _environment(
    scene: GpuRtSceneView,
    direction: Vec3f32[.WORLD],
    wavelengths: SampledWavelengths[1],
) -> SampledSpectrum[1]:
    var spectral = scene.spectral.value()
    var descs = scene.texture_descs.unsafe_origin_cast[ImmutAnyOrigin]()
    var pixels = scene.texture_pixels.unsafe_origin_cast[ImmutAnyOrigin]()
    if scene.environment_kind == .BLACK:
        return SampledSpectrum[1]()
    var result: SampledSpectrum[1]
    if spectral.environment == NO_SPECTRUM:
        var rgb = _gpu_environment_radiance(scene, descs, pixels, direction)
        if scene.environment_kind == .IMAGE:
            rgb = Color(
                scene.environment_scale_x,
                scene.environment_scale_y,
                scene.environment_scale_z,
            )
        result = (
            spectral.sources.rgb.sample(
                rgb,
                wavelengths,
                RgbSpectrumView.ILLUMINANT,
            )
            / CIE_Y_INTEGRAL
        )
    else:
        result = spectral.sources.emission(spectral.environment, wavelengths)
    if scene.environment_kind == .IMAGE:
        var uv = equal_area_sphere_to_square(
            scene.environment_world_to_light.vector(direction)
        )
        var texel = _GpuImageTextureView.load(
            scene.environment_texture_index, descs, pixels
        ).sample_environment(uv.u, uv.v)
        result = result * spectral.sources.rgb.sample(
            texel, wavelengths, RgbSpectrumView.UNBOUNDED
        )
    return result


@always_inline
def _absorption(
    scene: GpuRtSceneView,
    medium: UInt32,
    distance: Float32,
    wavelengths: SampledWavelengths[1],
) -> SampledSpectrum[1]:
    if medium == 0:
        return SampledSpectrum[1](1)
    var spectral = scene.spectral.value()
    var source = spectral.media[unsafe_offset=Int(medium)]
    var sigma = spectral.sources.rgb.sample(
        Color(scene.medium_absorption.unsafe_offset(3 * Int(medium))),
        wavelengths,
        RgbSpectrumView.UNBOUNDED,
    )
    if source != NO_SPECTRUM:
        sigma = spectral.sources.sample(
            source, wavelengths, RgbSpectrumView.UNBOUNDED
        )
    var result = SampledSpectrum[1](1)
    comptime for channel in range(4):
        if sigma[channel] > 0:
            var depth = sigma[channel] * distance
            result[channel] = depth.lt(80).select(exp(-min(depth, 80)), 0)
    return result


@always_inline
def _shade[
    kind: MaterialKind,
    integrator: Integrator,
    scene_kind: GpuRtSceneKind,
    sphere_format: GpuRtBvhFormat,
    triangle_format: GpuRtBvhFormat,
    tlas_format: GpuRtBvhFormat,
    blas_format: GpuRtBvhFormat,
](
    scene: GpuRtSceneView,
    queues: GpuRtTraceQueueView,
    state: GpuSpectralStateView,
    sampling: SamplingConfig,
    path: DeviceWavePath,
    ray: Rayf32[.WORLD],
    hit: SurfaceHit[1],
    beta: SampledSpectrum[1],
    mut wavelengths: SampledWavelengths[1],
    bounce: UInt32,
):
    var index = Int(path.path_id - state.sample_base)
    var material = load_spectral_material[kind](
        scene, Int(hit.surface.index()), hit.uv_u, hit.uv_v, wavelengths
    )
    var throughput = beta
    comptime for channel in range(4):
        throughput[channel] = (
            wavelengths.pdf[channel].gt(0).select(throughput[channel], 0)
        )
    var stage = wavefront_rng_stage(bounce)
    var seed = path_stage_seed(sampling, path.path_id, stage)
    comptime if Integrator.uses_direct_lighting[integrator]:
        var light = sample_spectral_light[integrator](
            path,
            ray,
            hit.t,
            hit.normal,
            hit.surface.value,
            scene,
            sampling,
            bounce,
        )
        if light.valid:
            var shadow = spawn_surface_ray(
                ray.at(hit.t), light.direction, light.distance
            )
            if not _trace_scene_any[
                scene_kind,
                sphere_format,
                triangle_format,
                tlas_format,
                blas_format,
            ](scene, shadow, False):
                var emission: SampledSpectrum[1]
                if light.primitive == UInt32.MAX:
                    emission = _environment(scene, light.direction, wavelengths)
                else:
                    var source = _emitter_source(scene, light.primitive)
                    if source != NO_SPECTRUM:
                        emission = scene.spectral.value().sources.emission(
                            source, wavelengths
                        )
                    else:
                        emission = (
                            scene.spectral.value().sources.rgb.sample(
                                light.emission,
                                wavelengths,
                                RgbSpectrumView.ILLUMINANT,
                            )
                            / CIE_Y_INTEGRAL
                        )
                var evaluated = material.evaluate[kind](
                    ray.d,
                    hit.normal,
                    light.direction,
                    seed,
                    True,
                    hit.front_face,
                )
                var scale = _direct_light_scale[integrator, 1](
                    light.cosine, light.pdf, evaluated.pdf, True
                )
                var medium = continuation_medium(
                    SIMD[.uint32, 1](path.medium_id),
                    hit.medium_boundary,
                    hit.front_face,
                    light.direction,
                    hit.geometric_normal,
                )
                state.accumulate(
                    index,
                    throughput
                    * evaluated.value
                    * emission
                    * _absorption(scene, medium, light.distance, wavelengths)
                    * scale,
                )
    var rng = path_stage_rng(sampling, path.path_id, bounce + 1)
    var sampled = material.sample[kind](
        ray.d,
        hit.normal,
        hit.front_face,
        rng,
        path_stage_seed(sampling, path.path_id, bounce + 1),
        True,
    )
    state.store(4, index, wavelengths.pdf)
    if not sampled.ok:
        return
    var continued = throughput * sampled.weight
    var eta_offset = 16 * Int(state.capacity) + index
    var eta_scale = state.fields[unsafe_offset=eta_offset]
    comptime if kind == .DIELECTRIC:
        if dot(sampled.direction, hit.geometric_normal) < 0:
            var eta = material.parameter
            eta_scale *= eta * eta if hit.front_face else 1 / (eta * eta)
    state.fields[unsafe_offset=eta_offset] = eta_scale
    if bounce + 1 >= RUSSIAN_ROULETTE_START_DEPTH:
        var importance = wavelengths.pdf.s0.gt(0).select(continued.s0, 0)
        comptime for channel in range(1, 4):
            importance = max(
                importance,
                wavelengths.pdf[channel].gt(0).select(continued[channel], 0),
            )
        importance *= eta_scale
        var survival = importance.clamp(
            RUSSIAN_ROULETTE_MIN_SURVIVAL, RUSSIAN_ROULETTE_MAX_SURVIVAL
        )
        var roulette = path_stage_rng(
            sampling, path.path_id, wavefront_rng_roulette_stage(bounce)
        )
        if importance <= 0 or roulette.f32() >= survival:
            return
        continued = continued / survival
    state.store(8, index, continued)
    var medium = continuation_medium(
        SIMD[.uint32, 1](path.medium_id),
        hit.medium_boundary,
        hit.front_face,
        sampled.direction,
        hit.geometric_normal,
    )
    var counters = queues.counters.unsafe_origin_cast[MutAnyOrigin]()
    var slot = _reserve_slot(counters, WAVE_COUNTER.NEXT)
    if slot >= Int(state.capacity):
        _mark_status(counters, WAVE_STATUS.PATH_OVERFLOW)
        return
    store_gpu_rt_path_values[integrator](
        PathContinuation[1](
            path.path_id,
            ray.at(hit.t),
            sampled.direction,
            Color(1),
            sampled.pdf,
            sampled.delta,
            medium,
        ),
        queues.dst_path_ids.unsafe_origin_cast[MutAnyOrigin](),
        queues.dst_path_fields.unsafe_origin_cast[MutAnyOrigin](),
        Int(state.capacity),
        slot,
    )


def spectral_initialize_kernel(
    state: GpuSpectralStateView, sampling_arg: GpuRtSamplingConfig, count: Int32
):
    var index = global_idx.x
    if index >= Int(count):
        return
    var rng = path_stage_rng(
        sampling_arg.load(), state.sample_base + UInt32(index), UInt32(0)
    )
    _ = rng.f32()
    _ = rng.f32()
    _ = rng.f32()
    _ = rng.f32()
    var wavelengths = SampledWavelengths[1].visible(rng.f32())
    state.store(0, index, wavelengths.lambda_nm)
    state.store(4, index, wavelengths.pdf)
    state.store(8, index, SampledSpectrum[1](1))
    state.store(12, index, SampledSpectrum[1]())
    state.fields[unsafe_offset=16 * Int(state.capacity) + index] = 1


def spectral_bounce_kernel[
    integrator: Integrator,
    scene_kind: GpuRtSceneKind,
    sphere_format: GpuRtBvhFormat,
    triangle_format: GpuRtBvhFormat,
    tlas_format: GpuRtBvhFormat,
    blas_format: GpuRtBvhFormat,
](
    scene: GpuRtSceneView,
    queues: GpuRtTraceQueueView,
    state: GpuSpectralStateView,
    sampling_arg: GpuRtSamplingConfig,
    bounce: UInt32,
    max_depth: UInt32,
):
    var count = Int(queues.counters[unsafe_offset=WAVE_COUNTER.ACTIVE])
    var slot = global_idx.x
    var sampling = sampling_arg.load()
    while slot < count:
        var path = load_gpu_rt_path[integrator](
            queues.src_path_ids.unsafe_origin_cast[ImmutAnyOrigin](),
            queues.src_path_fields.unsafe_origin_cast[ImmutAnyOrigin](),
            Int(state.capacity),
            slot,
        )
        var index = Int(path.path_id - state.sample_base)
        var wavelengths = state.wavelengths(index)
        var ray = Rayf32[.WORLD](
            Point3f32[.WORLD](path.ox, path.oy, path.oz),
            Vec3f32[.WORLD](path.dx, path.dy, path.dz),
            path.t_min,
            path.t_max,
        )
        var hit, primitive = spectral_trace_surface[
            integrator,
            scene_kind,
            sphere_format,
            triangle_format,
            tlas_format,
            blas_format,
        ](scene, ray, sampling, path.path_id, bounce, Int(state.capacity))
        var beta = state.load(8, index) * _absorption(
            scene, path.medium_id, hit.t * length(ray.d), wavelengths
        )
        if not hit.hit:
            var rgb = _gpu_environment_radiance(
                scene,
                scene.texture_descs.unsafe_origin_cast[ImmutAnyOrigin](),
                scene.texture_pixels.unsafe_origin_cast[ImmutAnyOrigin](),
                ray.d,
            )
            if scene.environment_kind == .IMAGE:
                var uv = equal_area_sphere_to_square(
                    scene.environment_world_to_light.vector(ray.d)
                )
                rgb = _GpuImageTextureView.load(
                    scene.environment_texture_index,
                    scene.texture_descs.unsafe_origin_cast[ImmutAnyOrigin](),
                    scene.texture_pixels.unsafe_origin_cast[ImmutAnyOrigin](),
                ).sample_environment(uv.u, uv.v)
            var light_pdf = scene.spectral.value().lighting.environment_pdf(
                scene.environment_kind, rgb
            )
            var weight = _emissive_hit_weight_from_pdf[integrator](
                bounce, path.delta, path.bsdf_pdf, light_pdf
            )
            state.accumulate(
                index, beta * _environment(scene, ray.d, wavelengths) * weight
            )
        else:
            var emitter = gpu_emitter(scene, primitive, hit.surface.value)
            var source = _emitter_source(scene, primitive)
            if (hit.front_face or emitter.two_sided) and opacity_accepts(
                gpu_opacity_value(scene.opacity, primitive, hit.uv_u, hit.uv_v),
                opacity_point_sample(ray.at(hit.t)),
            ):
                var emission = (
                    scene.spectral.value().sources.rgb.sample(
                        emitter.radiance,
                        wavelengths,
                        RgbSpectrumView.ILLUMINANT,
                    )
                    / CIE_Y_INTEGRAL
                )
                if source != NO_SPECTRUM:
                    emission = scene.spectral.value().sources.emission(
                        source, wavelengths
                    )
                var importance = scene.spectral.value().lighting.emission(
                    source, emitter.radiance
                )
                var light_pdf = _emissive_hit_light_pdf(
                    ray.d,
                    hit.t,
                    hit.geometric_normal,
                    Color(importance),
                    scene.spectral.value().lighting.total_weight,
                    SIMD[.bool, 1](emitter.two_sided),
                )
                var weight = _emissive_hit_weight_from_pdf[integrator](
                    bounce, path.delta, path.bsdf_pdf, light_pdf
                )
                state.accumulate(index, beta * emission * weight)
            if bounce < max_depth:
                match hit.surface.kind():
                    case .LAMBERTIAN:
                        _shade[
                            .LAMBERTIAN,
                            integrator,
                            scene_kind,
                            sphere_format,
                            triangle_format,
                            tlas_format,
                            blas_format,
                        ](
                            scene,
                            queues,
                            state,
                            sampling,
                            path,
                            ray,
                            hit,
                            beta,
                            wavelengths,
                            bounce,
                        )
                    case .METAL:
                        _shade[
                            .METAL,
                            integrator,
                            scene_kind,
                            sphere_format,
                            triangle_format,
                            tlas_format,
                            blas_format,
                        ](
                            scene,
                            queues,
                            state,
                            sampling,
                            path,
                            ray,
                            hit,
                            beta,
                            wavelengths,
                            bounce,
                        )
                    case .DIELECTRIC:
                        _shade[
                            .DIELECTRIC,
                            integrator,
                            scene_kind,
                            sphere_format,
                            triangle_format,
                            tlas_format,
                            blas_format,
                        ](
                            scene,
                            queues,
                            state,
                            sampling,
                            path,
                            ray,
                            hit,
                            beta,
                            wavelengths,
                            bounce,
                        )
                    case .COATED_DIFFUSE:
                        _shade[
                            .COATED_DIFFUSE,
                            integrator,
                            scene_kind,
                            sphere_format,
                            triangle_format,
                            tlas_format,
                            blas_format,
                        ](
                            scene,
                            queues,
                            state,
                            sampling,
                            path,
                            ray,
                            hit,
                            beta,
                            wavelengths,
                            bounce,
                        )
                    case .EMISSIVE | .MIX:
                        pass
        slot += Int(grid_dim.x * block_dim.x)


def spectral_resolve_kernel(
    state: GpuSpectralStateView,
    weights: ImmPointer[Float32, ImmUntrackedOrigin],
    pixels: Pointer[Float32, MutUntrackedOrigin],
    spp_arg: Int32,
    count_arg: Int32,
    film: FilmResponse,
):
    var spp = Int(spp_arg)
    var count = Int(count_arg)
    var pixel = global_idx.x
    if pixel >= count // spp:
        return
    var total = Color(0)
    var weight = Float32(0)
    for sample in range(spp):
        var index = pixel * spp + sample
        var w = weights[unsafe_offset=3 * Int(state.capacity) + index]
        total += (
            film.apply_spectral_sample(
                state.load(12, index), state.wavelengths(index)
            )
            * w
        )
        weight += w
    total = total / weight if weight != 0 else Color(0)
    var output = (Int(state.sample_base) // spp + pixel) * 3
    comptime for channel in range(3):
        pixels[unsafe_offset=output + channel] = total[channel]
