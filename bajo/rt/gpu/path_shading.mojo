"""GPU surface routing, BSDF shading, and path continuation."""

from std.memory import bitcast
from max.gpu import block_dim, global_idx, grid_dim
from std.math import ceildiv
from max.gpu.host import DeviceBuffer, DeviceContext
from bajo.core import Point3f32, Vec3f32
from bajo.rt.common import path_stage_rng, path_stage_seed
from bajo.rt.continuation import finish_continuation, continuation_medium
from bajo.rt.lighting import (
    _emissive_hit_light_pdf,
    _emissive_hit_weight_from_pdf,
)
from bajo.rt.shading import (
    METAL_DELTA_FUZZ_THRESHOLD,
    _sample_coated_diffuse,
    _sample_material,
    dielectric_geometry,
)
from bajo.core.color import Color
from bajo.rt.render_types import SamplingConfig
from bajo.rt.rays import spawn_surface_ray
from bajo.core.optics import medium_transmittance
from bajo.rt.gpu.lighting import GpuDirectLightSample
from bajo.rt.gpu.views import GpuRtSceneView, GpuRtTraceQueueView
from bajo.rt.gpu.queue_order import GpuRtQueueOrderScratch
from bajo.rt.material_types import MaterialKind, Integrator, SurfaceId
from bajo.rt.wavefront_contract import (
    DeviceWavePath,
    DeviceWaveShade,
    DeviceWaveShadow,
    WAVE_COUNTER,
    WAVE_STATUS,
    WaveSampleFloatAbi,
    load_device_wave_shade,
    store_device_wave_shade,
    wavefront_rng_stage,
)
from bajo.rt.gpu.common_kernels import (
    GPU_RT_MAX_BLOCKS,
    gpu_rt_bounce_block_limit,
)
from bajo.rt.gpu.wavefront_contract import (
    GpuWavefrontArena,
    GPU_SHADE_U,
    GPU_SHADE_V,
    GPU_SHADE_BOUNDARY,
    GPU_SHADE_GNX,
    GPU_SHADE_GNY,
    GPU_SHADE_GNZ,
    GPU_SHADE_TX,
    GPU_SHADE_TY,
    GPU_SHADE_TZ,
    _mark_status,
    _reserve_slot,
    load_gpu_rt_path,
    store_gpu_rt_path_values,
    store_gpu_rt_shadow,
)
from bajo.rt.gpu.queue_layout import gpu_rt_shade_tile, gpu_rt_radiance_tile
from bajo.rt.gpu.materials import GpuRtMaterials
from bajo.rt.gpu.material_layout import (
    GpuRgbMaterialLayout,
    GpuDielectricLayout,
    GpuMetalLayout,
)
from bajo.rt.gpu.textures import (
    _sample_gpu_lambertian,
    _sample_gpu_coated_parameters,
    _sample_gpu_coated_albedo,
)
from bajo.rt.shading_types import BsdfSample


comptime GPU_RT_SHADE_MAX_BLOCKS = GPU_RT_MAX_BLOCKS
comptime GPU_RT_SHADE_BLOCK_SIZE = 128


@always_inline
def _accumulate_sample(
    sample_radiance: Pointer[Float32, MutAnyOrigin],
    capacity: Int,
    sample_base: UInt32,
    path_id: UInt32,
    value: Color,
):
    var idx = Int(path_id - sample_base)
    var radiance = gpu_rt_radiance_tile(sample_radiance, capacity)
    radiance[(WaveSampleFloatAbi.R, idx)] += value.x
    radiance[(WaveSampleFloatAbi.G, idx)] += value.y
    radiance[(WaveSampleFloatAbi.B, idx)] += value.z


@always_inline
def _append_shade(
    work: DeviceWaveShade,
    path_refs: Pointer[UInt32, MutAnyOrigin],
    surface_values: Pointer[UInt32, MutAnyOrigin],
    fields: Pointer[Float32, MutAnyOrigin],
    counters: Pointer[UInt32, MutAnyOrigin],
    capacity: Int,
) -> Int:
    var slot = _reserve_slot(counters, WAVE_COUNTER.SHADE)
    if slot >= capacity:
        _mark_status(counters, WAVE_STATUS.SHADE_OVERFLOW)
        return -1
    store_device_wave_shade(
        work, path_refs, surface_values, fields, capacity, slot
    )

    return slot


@always_inline
def _append_shadow[
    STORE_CONTRIBUTION: Bool = True
](
    work: DeviceWaveShadow,
    path_ids: Pointer[UInt32, MutAnyOrigin],
    fields: Pointer[Float32, MutAnyOrigin],
    counters: Pointer[UInt32, MutAnyOrigin],
    capacity: Int,
) -> Int:
    var slot = _reserve_slot(counters, WAVE_COUNTER.SHADOW)
    if slot >= capacity:
        _mark_status(counters, WAVE_STATUS.SHADOW_OVERFLOW)
        return -1
    store_gpu_rt_shadow[STORE_CONTRIBUTION](
        work, path_ids, fields, capacity, slot
    )
    return slot


@always_inline
def _append_direct_light_shadow(
    var direct: GpuDirectLightSample,
    path: DeviceWavePath,
    point: Point3f32[.WORLD],
    scene: GpuRtSceneView,
    queues: GpuRtTraceQueueView,
):
    if not direct.valid:
        return
    if path.medium_id != 0:
        var base = 3 * Int(path.medium_id)
        direct.contribution *= medium_transmittance(
            Color(scene.medium_absorption.unsafe_offset(base)),
            direct.shadow_t_max,
        )
    var shadow = spawn_surface_ray(point, direct.direction, direct.shadow_t_max)
    var slot = _append_shadow(
        DeviceWaveShadow(
            path.path_id,
            shadow.o.x,
            shadow.o.y,
            shadow.o.z,
            shadow.t_min,
            shadow.d.x,
            shadow.d.y,
            shadow.d.z,
            shadow.t_max,
            direct.contribution.x,
            direct.contribution.y,
            direct.contribution.z,
        ),
        queues.shadow_path_ids.unsafe_origin_cast[MutAnyOrigin](),
        queues.shadow_fields.unsafe_origin_cast[MutAnyOrigin](),
        queues.counters.unsafe_origin_cast[MutAnyOrigin](),
        Int(queues.capacity),
    )
    if slot >= 0 and scene.light_count > 0:
        GpuRtQueueOrderScratch.from_queues(queues).light_ids[
            unsafe_offset=slot
        ] = direct.light_id


@always_inline
def _continue_gpu_path[
    integrator: Integrator,
](
    path: DeviceWavePath,
    hit_t: Float32,
    sampled: BsdfSample[1],
    dst_path_ids: Pointer[UInt32, MutAnyOrigin],
    dst_path_fields: Pointer[Float32, MutAnyOrigin],
    counters: Pointer[UInt32, MutAnyOrigin],
    capacity: Int,
    sampling: SamplingConfig,
    bounce: UInt32,
):
    """Apply roulette and append a sampled continuation to the next queue.

    Material sampling owns its RNG draws; continuation uses the independent
    roulette domain and retains integrator-specific packed path storage.
    """
    if not sampled.ok:
        return
    var finished = finish_continuation[1](
        path.path_id,
        Point3f32[.WORLD](path.ox, path.oy, path.oz),
        Vec3f32[.WORLD](path.dx, path.dy, path.dz),
        hit_t,
        Color(path.tx, path.ty, path.tz),
        path.medium_id,
        sampled,
        True,
        sampling,
        bounce + UInt32(1),
    )
    if not finished.active:
        return
    var slot = _reserve_slot(counters, WAVE_COUNTER.NEXT)
    if slot >= capacity:
        _mark_status(counters, WAVE_STATUS.PATH_OVERFLOW)
        return
    store_gpu_rt_path_values[integrator](
        finished.values, dst_path_ids, dst_path_fields, capacity, slot
    )


@always_inline
def _shade_lambertian_inline[
    integrator: Integrator,
](
    path: DeviceWavePath,
    ray_direction: Vec3f32[.WORLD],
    normal: Vec3f32[.WORLD],
    uv_u: Float32,
    uv_v: Float32,
    hit_t: Float32,
    surface_value: UInt32,
    lambertians: Pointer[Float32, ImmutAnyOrigin],
    lambertian_texture_indices: Pointer[UInt32, ImmutAnyOrigin],
    texture_descs: Pointer[UInt32, ImmutAnyOrigin],
    texture_pixels: Pointer[Float32, ImmutAnyOrigin],
    dst_path_ids: Pointer[UInt32, MutAnyOrigin],
    dst_path_fields: Pointer[Float32, MutAnyOrigin],
    counters: Pointer[UInt32, MutAnyOrigin],
    capacity: Int,
    sampling: SamplingConfig,
    bounce: UInt32,
):
    """Fuse the dominant diffuse shade operation into closest-hit routing."""
    var albedo = _sample_gpu_lambertian(
        surface_value,
        uv_u,
        uv_v,
        lambertians,
        lambertian_texture_indices,
        texture_descs,
        texture_pixels,
    )
    var rng = path_stage_rng(
        sampling, path.path_id, wavefront_rng_stage(bounce)
    )
    var random_u = rng.f32()
    var random_v = rng.f32()
    var sampled = _sample_material[.LAMBERTIAN, 1](
        ray_direction,
        normal,
        albedo,
        1.0,
        True,
        random_u,
        random_v,
    )
    _continue_gpu_path[integrator](
        path,
        hit_t,
        sampled,
        dst_path_ids,
        dst_path_fields,
        counters,
        capacity,
        sampling,
        bounce,
    )


@always_inline
def _shade_coated_diffuse_inline[
    integrator: Integrator,
](
    path: DeviceWavePath,
    ray_direction: Vec3f32[.WORLD],
    normal: Vec3f32[.WORLD],
    uv_u: Float32,
    uv_v: Float32,
    hit_t: Float32,
    surface_value: UInt32,
    coated_diffuses: Pointer[Float32, ImmutAnyOrigin],
    coated_diffuse_texture_indices: Pointer[UInt32, ImmutAnyOrigin],
    texture_descs: Pointer[UInt32, ImmutAnyOrigin],
    texture_pixels: Pointer[Float32, ImmutAnyOrigin],
    dst_path_ids: Pointer[UInt32, MutAnyOrigin],
    dst_path_fields: Pointer[Float32, MutAnyOrigin],
    counters: Pointer[UInt32, MutAnyOrigin],
    capacity: Int,
    sampling: SamplingConfig,
    bounce: UInt32,
):
    var material_idx = Int(SurfaceId.index_from_raw(surface_value))
    var coated = _sample_gpu_coated_parameters(
        coated_diffuses,
        material_idx,
        uv_u,
        uv_v,
        coated_diffuse_texture_indices,
        texture_descs,
        texture_pixels,
    )
    var albedo = _sample_gpu_coated_albedo(
        surface_value,
        uv_u,
        uv_v,
        coated_diffuses,
        coated_diffuse_texture_indices,
        texture_descs,
        texture_pixels,
    )
    var rng = path_stage_rng(
        sampling, path.path_id, wavefront_rng_stage(bounce)
    )
    var sampled = _sample_coated_diffuse(
        ray_direction,
        normal,
        albedo,
        coated,
        rng.f32(),
        rng.f32(),
        estimator_seed=path_stage_seed(
            sampling, path.path_id, wavefront_rng_stage(bounce)
        ),
    )
    _continue_gpu_path[integrator](
        path,
        hit_t,
        sampled,
        dst_path_ids,
        dst_path_fields,
        counters,
        capacity,
        sampling,
        bounce,
    )


@always_inline
def _route_surface_hit[
    integrator: Integrator,
](
    active_path_idx: Int,
    path: DeviceWavePath,
    ray_direction: Vec3f32[.WORLD],
    normal: Vec3f32[.WORLD],
    front_face: Bool,
    uv_u: Float32,
    uv_v: Float32,
    hit_t: Float32,
    surface_value: UInt32,
    bounce: UInt32,
    total_light_weight: Float32,
    emissives: Pointer[Float32, ImmutAnyOrigin],
    lambertians: Pointer[Float32, ImmutAnyOrigin],
    lambertian_texture_indices: Pointer[UInt32, ImmutAnyOrigin],
    texture_descs: Pointer[UInt32, ImmutAnyOrigin],
    texture_pixels: Pointer[Float32, ImmutAnyOrigin],
    coated_diffuses: Pointer[Float32, ImmutAnyOrigin],
    coated_diffuse_texture_indices: Pointer[UInt32, ImmutAnyOrigin],
    dst_path_ids: Pointer[UInt32, MutAnyOrigin],
    dst_path_fields: Pointer[Float32, MutAnyOrigin],
    shade_path_refs: Pointer[UInt32, MutAnyOrigin],
    shade_surfaces: Pointer[UInt32, MutAnyOrigin],
    shade_fields: Pointer[Float32, MutAnyOrigin],
    counters: Pointer[UInt32, MutAnyOrigin],
    sample_radiance: Pointer[Float32, MutAnyOrigin],
    capacity: Int,
    sample_base: UInt32,
    sampling: SamplingConfig,
    medium_boundary: UInt32 = 0,
    geometric_normal: Vec3f32[.WORLD] = Vec3f32[.WORLD](0.0),
):
    """Route a geometry-independent oriented hit to output or a BSDF queue."""

    comptime if integrator == .NORMALS:
        _accumulate_sample(
            sample_radiance,
            capacity,
            sample_base,
            path.path_id,
            0.5 * (normal + Color(1.0)),
        )
        return

    var kind = SurfaceId.kind_from_raw(surface_value)
    match kind:
        case .MIX:
            debug_assert["safe"](
                False, "mix surfaces must be resolved before shading"
            )
            return
        case .EMISSIVE:
            if front_face:
                var material_idx = Int(SurfaceId.index_from_raw(surface_value))
                var base = GpuRgbMaterialLayout.STRIDE * material_idx
                var radiance = Color(emissives.unsafe_offset(base))
                var light_pdf = Float32(0.0)
                comptime if integrator == .MIS:
                    if bounce > 0 and not path.delta:
                        light_pdf = _emissive_hit_light_pdf(
                            ray_direction,
                            hit_t,
                            normal,
                            radiance,
                            total_light_weight,
                        )
                var emission_weight = _emissive_hit_weight_from_pdf[integrator](
                    bounce, path.delta, path.bsdf_pdf, light_pdf
                )
                _accumulate_sample(
                    sample_radiance,
                    capacity,
                    sample_base,
                    path.path_id,
                    Color(path.tx, path.ty, path.tz)
                    * radiance
                    * emission_weight,
                )
            return

        case .LAMBERTIAN:
            _shade_lambertian_inline[integrator](
                path,
                ray_direction,
                normal,
                uv_u,
                uv_v,
                hit_t,
                surface_value,
                lambertians,
                lambertian_texture_indices,
                texture_descs,
                texture_pixels,
                dst_path_ids,
                dst_path_fields,
                counters,
                capacity,
                sampling,
                bounce,
            )
        case .COATED_DIFFUSE | .METAL | .DIELECTRIC:
            var slot = _append_shade(
                DeviceWaveShade(
                    UInt32(active_path_idx),
                    normal.x,
                    normal.y,
                    normal.z,
                    surface_value,
                    hit_t,
                    front_face,
                ),
                shade_path_refs,
                shade_surfaces,
                shade_fields,
                counters,
                capacity,
            )
            if slot >= 0:
                var shade_tile = gpu_rt_shade_tile(shade_fields, capacity)
                shade_tile[(GPU_SHADE_U, slot)] = uv_u
                shade_tile[(GPU_SHADE_V, slot)] = uv_v
                shade_tile[(GPU_SHADE_BOUNDARY, slot)] = bitcast[.float32](
                    medium_boundary
                )
                shade_tile[(GPU_SHADE_GNX, slot)] = geometric_normal.x
                shade_tile[(GPU_SHADE_GNY, slot)] = geometric_normal.y
                shade_tile[(GPU_SHADE_GNZ, slot)] = geometric_normal.z
                shade_tile[(GPU_SHADE_TX, slot)] = path.tx
                shade_tile[(GPU_SHADE_TY, slot)] = path.ty
                shade_tile[(GPU_SHADE_TZ, slot)] = path.tz


@always_inline
def _gpu_rt_shade_one[
    integrator: Integrator,
    MATERIAL_KIND: MaterialKind,
](
    idx: Int,
    src_path_ids: Pointer[UInt32, ImmutAnyOrigin],
    src_path_fields: Pointer[Float32, ImmutAnyOrigin],
    shade_path_refs: Pointer[UInt32, ImmutAnyOrigin],
    shade_surfaces: Pointer[UInt32, ImmutAnyOrigin],
    shade_fields: Pointer[Float32, ImmutAnyOrigin],
    material_data: Pointer[Float32, ImmutAnyOrigin],
    dst_path_ids: Pointer[UInt32, MutAnyOrigin],
    dst_path_fields: Pointer[Float32, MutAnyOrigin],
    counters: Pointer[UInt32, MutAnyOrigin],
    capacity_i32: Int32,
    sampling: SamplingConfig,
    bounce: UInt32,
):
    comptime assert MaterialKind.has_bsdf[MATERIAL_KIND]
    var capacity = Int(capacity_i32)
    var shade_tile = gpu_rt_shade_tile(shade_fields, capacity)
    var work = load_device_wave_shade(
        shade_path_refs,
        shade_surfaces,
        shade_fields,
        capacity,
        idx,
    )
    var path = load_gpu_rt_path[integrator](
        src_path_ids, src_path_fields, capacity, Int(work.path_idx)
    )
    path.tx = shade_tile[(GPU_SHADE_TX, idx)]
    path.ty = shade_tile[(GPU_SHADE_TY, idx)]
    path.tz = shade_tile[(GPU_SHADE_TZ, idx)]
    var ray_direction = Vec3f32[.WORLD](path.dx, path.dy, path.dz)
    var normal = Vec3f32[.WORLD](work.nx, work.ny, work.nz)
    var albedo = Color(0.0)
    var parameter = Float32(1.0)
    var random_u = Float32(0.0)
    var random_v = Float32(0.0)
    var material_idx = Int(SurfaceId.index_from_raw(work.surface_value))
    var rng = path_stage_rng(
        sampling, path.path_id, wavefront_rng_stage(bounce)
    )

    comptime match MATERIAL_KIND:
        case .LAMBERTIAN:
            var base = GpuRgbMaterialLayout.STRIDE * material_idx
            albedo = Color(material_data.unsafe_offset(base))
            random_u = rng.f32()
            random_v = rng.f32()
        case .METAL:
            var base = GpuMetalLayout.STRIDE * material_idx
            albedo = Color(
                material_data.unsafe_offset(base + GpuMetalLayout.ALBEDO)
            )
            parameter = material_data[unsafe_offset=base + GpuMetalLayout.FUZZ]
            if parameter > METAL_DELTA_FUZZ_THRESHOLD:
                random_u = rng.f32()
                random_v = rng.f32()
        case _:
            parameter = material_data[
                unsafe_offset=GpuDielectricLayout.STRIDE * material_idx
                + GpuDielectricLayout.ETA
            ]
            var ri = Float32(1.0) / parameter if work.front_face else parameter
            var geometry = dielectric_geometry(ray_direction, normal, ri)
            if geometry[2] <= 1.0:
                random_u = rng.f32()

    var sampled = _sample_material[MATERIAL_KIND, 1](
        ray_direction,
        normal,
        albedo,
        parameter,
        work.front_face,
        random_u,
        random_v,
    )
    var boundary = bitcast[.uint32](shade_tile[(GPU_SHADE_BOUNDARY, idx)])
    var geometric = Vec3f32[.WORLD](
        shade_tile[(GPU_SHADE_GNX, idx)],
        shade_tile[(GPU_SHADE_GNY, idx)],
        shade_tile[(GPU_SHADE_GNZ, idx)],
    )
    path.medium_id = continuation_medium[1](
        path.medium_id, boundary, work.front_face, sampled.direction, geometric
    )[0]
    _continue_gpu_path[integrator](
        path,
        work.t,
        sampled,
        dst_path_ids,
        dst_path_fields,
        counters,
        capacity,
        sampling,
        bounce,
    )


def gpu_rt_shade_dispatch_kernel[
    integrator: Integrator,
](
    src_path_ids: Pointer[UInt32, ImmutAnyOrigin],
    src_path_fields: Pointer[Float32, ImmutAnyOrigin],
    shade_path_refs: Pointer[UInt32, ImmutAnyOrigin],
    shade_surfaces: Pointer[UInt32, ImmutAnyOrigin],
    shade_fields: Pointer[Float32, ImmutAnyOrigin],
    metals: Pointer[Float32, ImmutAnyOrigin],
    dielectrics: Pointer[Float32, ImmutAnyOrigin],
    dst_path_ids: Pointer[UInt32, MutAnyOrigin],
    dst_path_fields: Pointer[Float32, MutAnyOrigin],
    counters: Pointer[UInt32, MutAnyOrigin],
    capacity_i32: Int32,
    sampling: SamplingConfig,
    bounce: UInt32,
):
    var work_count = Int(counters[unsafe_offset=WAVE_COUNTER.SHADE])
    var idx = global_idx.x
    var stride = Int(grid_dim.x * block_dim.x)
    while idx < work_count:
        var surface_value = shade_surfaces[unsafe_offset=idx]
        var kind = SurfaceId.kind_from_raw(surface_value)
        match kind:
            case .METAL:
                _gpu_rt_shade_one[integrator, .METAL](
                    idx,
                    src_path_ids,
                    src_path_fields,
                    shade_path_refs,
                    shade_surfaces,
                    shade_fields,
                    metals,
                    dst_path_ids,
                    dst_path_fields,
                    counters,
                    capacity_i32,
                    sampling,
                    bounce,
                )
            case .DIELECTRIC:
                _gpu_rt_shade_one[integrator, .DIELECTRIC](
                    idx,
                    src_path_ids,
                    src_path_fields,
                    shade_path_refs,
                    shade_surfaces,
                    shade_fields,
                    dielectrics,
                    dst_path_ids,
                    dst_path_fields,
                    counters,
                    capacity_i32,
                    sampling,
                    bounce,
                )
            case _:
                pass
        idx += stride


def _enqueue_material_shading[
    integrator: Integrator,
    MAX_BLOCKS: Int = GPU_RT_SHADE_MAX_BLOCKS,
](
    ctx: DeviceContext,
    arena: GpuWavefrontArena,
    materials: GpuRtMaterials,
    src_path_ids: DeviceBuffer[.uint32],
    src_path_fields: DeviceBuffer[.float32],
    dst_path_ids: DeviceBuffer[.uint32],
    dst_path_fields: DeviceBuffer[.float32],
    sampling: SamplingConfig,
    bounce: UInt32,
) raises:
    if not materials.stages.metal_dielectric:
        return
    var blocks = min(
        ceildiv(arena.capacity, GPU_RT_SHADE_BLOCK_SIZE),
        MAX_BLOCKS,
    )
    blocks = min(
        blocks, gpu_rt_bounce_block_limit(bounce, GPU_RT_SHADE_BLOCK_SIZE)
    )
    ctx.enqueue_function[gpu_rt_shade_dispatch_kernel[integrator]](
        src_path_ids,
        src_path_fields,
        arena.shade.path_refs,
        arena.shade.surface_values,
        arena.shade.fields,
        materials.metals,
        materials.dielectrics,
        dst_path_ids,
        dst_path_fields,
        arena.counters,
        Int32(arena.capacity),
        sampling,
        bounce,
        grid_dim=blocks,
        block_dim=GPU_RT_SHADE_BLOCK_SIZE,
    )
