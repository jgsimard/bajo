"""Shared GPU RT material, lighting, routing, and shading kernels."""

from std.gpu import block_dim, global_idx, grid_dim
from std.math import ceildiv, sqrt
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh.constants import f32_max
from bajo.bvh.gpu.utils import upload_list
from bajo.core import (
    Point3f32,
    Rayf32,
    Vec3f32,
    dot,
    length2,
    normalize,
)
from bajo.core.random import random_on_hemisphere, random_unit_vector
from bajo.rt.common import path_stage_rng, russian_roulette, sky_color
from bajo.rt.lighting import (
    _direct_light_scale,
    _draw_alias_column,
    _finish_direct_light_geometry,
    _LightSurfaceSample,
    _resolve_alias_draw,
    _sample_sphere_light_surface,
    _sample_triangle_light_surface,
    _solid_angle_light_pdf,
    power_heuristic,
)
from bajo.rt.shading import _evaluate_material, _sample_material
from bajo.rt.types import (
    Color,
    MaterialKind,
    PrimitiveKind,
    Integrator,
    SURFACE_INDEX_MASK,
    SceneData,
)
from bajo.rt.wavefront_contract import (
    DeviceWavePath,
    DeviceWaveShade,
    DeviceWaveShadow,
    WAVE_COUNTER,
    WAVE_STATUS,
    WaveSampleFloatAbi,
    load_device_wave_shade,
    store_device_wave_shade,
    wavefront_plane_index,
    wavefront_rng_light_stage,
    wavefront_rng_stage,
)
from bajo.rt.gpu.common_kernels import GPU_RT_MAX_BLOCKS
from bajo.rt.gpu.wavefront_contract import (
    GpuWavefrontArena,
    _mark_status,
    _reserve_slot,
    load_gpu_rt_path,
    store_gpu_rt_path,
    store_gpu_rt_shadow,
)


comptime GPU_RT_SHADE_MAX_BLOCKS = GPU_RT_MAX_BLOCKS
comptime GPU_RT_SHADE_BLOCK_SIZE = 128


comptime GPU_RT_LIGHT_STRIDE = 14
comptime GPU_RT_LIGHT_P0_X = 0
comptime GPU_RT_LIGHT_P0_Y = 1
comptime GPU_RT_LIGHT_P0_Z = 2
comptime GPU_RT_LIGHT_P1_X = 3
comptime GPU_RT_LIGHT_P1_Y = 4
comptime GPU_RT_LIGHT_P1_Z = 5
comptime GPU_RT_LIGHT_P2_X = 6
comptime GPU_RT_LIGHT_P2_Y = 7
comptime GPU_RT_LIGHT_P2_Z = 8
comptime GPU_RT_LIGHT_RADIUS = 9
comptime GPU_RT_LIGHT_E_X = 10
comptime GPU_RT_LIGHT_E_Y = 11
comptime GPU_RT_LIGHT_E_Z = 12
# Retained in each record to preserve the proven 14-float AoS stride. The
# selection hot path reads the duplicated dense probability prefix instead.
comptime GPU_RT_LIGHT_ALIAS_PROBABILITY = 13


def _upload_nonempty[
    dtype: DType
](
    mut ctx: DeviceContext, var values: List[Scalar[dtype]]
) raises -> DeviceBuffer[dtype]:
    if len(values) == 0:
        values.append(0)
    return upload_list(ctx, values)


def _flatten_lambertians(world: SceneData) -> List[Float32]:
    var out = List[Float32](capacity=len(world.surfaces().lambertians) * 3)
    for material in world.surfaces().lambertians:
        out.append(material.albedo.x)
        out.append(material.albedo.y)
        out.append(material.albedo.z)
    return out^


def _flatten_metals(world: SceneData) -> List[Float32]:
    var out = List[Float32](capacity=len(world.surfaces().metals) * 4)
    for material in world.surfaces().metals:
        out.append(material.albedo.x)
        out.append(material.albedo.y)
        out.append(material.albedo.z)
        out.append(material.fuzz)
    return out^


def _flatten_dielectrics(world: SceneData) -> List[Float32]:
    var out = List[Float32](capacity=len(world.surfaces().dielectrics))
    for material in world.surfaces().dielectrics:
        out.append(material.refraction_index)
    return out^


def _flatten_emissives(world: SceneData) -> List[Float32]:
    var out = List[Float32](capacity=len(world.surfaces().emissives) * 3)
    for material in world.surfaces().emissives:
        out.append(material.radiance.x)
        out.append(material.radiance.y)
        out.append(material.radiance.z)
    return out^


struct GpuRtMaterials:
    """Flattened device material tables shared by all GPU geometry backends."""

    var lambertians: DeviceBuffer[.float32]
    var metals: DeviceBuffer[.float32]
    var dielectrics: DeviceBuffer[.float32]
    var emissives: DeviceBuffer[.float32]
    var has_non_lambertian: Bool

    def __init__(
        out self,
        mut ctx: DeviceContext,
        world: SceneData,
    ) raises:
        self.lambertians = _upload_nonempty(ctx, _flatten_lambertians(world))
        self.metals = _upload_nonempty(ctx, _flatten_metals(world))
        self.dielectrics = _upload_nonempty(ctx, _flatten_dielectrics(world))
        self.emissives = _upload_nonempty(ctx, _flatten_emissives(world))
        self.has_non_lambertian = (
            len(world.surfaces().metals) > 0
            or len(world.surfaces().dielectrics) > 0
        )


struct GpuRtLights:
    """Compact device light records matching the CPU power distribution."""

    var kinds: DeviceBuffer[.uint32]
    var fields: DeviceBuffer[.float32]
    var count: Int
    var total_weight: Float32
    var uniform_sampling_kind: PrimitiveKind

    def __init__(
        out self,
        mut ctx: DeviceContext,
        world: SceneData,
    ) raises:
        var kinds = List[UInt32](capacity=len(world.lights().records))
        var fields = List[Float32](
            capacity=len(world.lights().records) * (GPU_RT_LIGHT_STRIDE + 1)
        )
        debug_assert["safe", _use_compiler_assume=True](
            len(world.lights().records) < (1 << 28),
            "GPU RT alias table supports fewer than 2^28 lights",
        )
        for probability in world.lights().alias_probabilities:
            fields.append(probability)
        var uniform_kind = PrimitiveKind.UNKNOWN
        var homogeneous = True
        for light_idx, light in enumerate(world.lights().records):
            var kind = light.primitive.kind()
            var sampling_kind = (
                PrimitiveKind.SPHERE if kind
                == PrimitiveKind.SPHERE else PrimitiveKind.TRIANGLE
            )
            if light_idx == 0:
                uniform_kind = sampling_kind
            elif sampling_kind != uniform_kind:
                homogeneous = False
            var radiance = (
                world.surfaces().emissives[Int(light.surface.index())].radiance
            )
            kinds.append(
                (world.lights().alias_indices[light_idx] << UInt32(4))
                | kind.value
            )
            fields.append(light.p0.x)
            fields.append(light.p0.y)
            fields.append(light.p0.z)
            fields.append(light.p1.x)
            fields.append(light.p1.y)
            fields.append(light.p1.z)
            fields.append(light.p2.x)
            fields.append(light.p2.y)
            fields.append(light.p2.z)
            fields.append(light.radius)
            fields.append(radiance.x)
            fields.append(radiance.y)
            fields.append(radiance.z)
            fields.append(world.lights().alias_probabilities[light_idx])
        self.kinds = _upload_nonempty(ctx, kinds^)
        self.fields = _upload_nonempty(ctx, fields^)
        self.count = len(world.lights().records)
        self.total_weight = world.lights().total_weight
        self.uniform_sampling_kind = (
            uniform_kind if homogeneous else PrimitiveKind.UNKNOWN
        )


@fieldwise_init
struct GpuDirectLightSample(TrivialRegisterPassable):
    var valid: Bool
    var direction: Vec3f32[.WORLD]
    var contribution: Color
    var shadow_t_max: Float32


@always_inline
def _empty_direct_light_sample() -> GpuDirectLightSample:
    return GpuDirectLightSample(
        False,
        Vec3f32[.WORLD](0.0),
        Color(0.0),
        0.0,
    )


@always_inline
def _sample_direct_light_candidate[
    integrator: Integrator,
    light_kind: PrimitiveKind = .UNKNOWN,
](
    path: DeviceWavePath,
    incoming_ray: Rayf32[.WORLD],
    hit_t: Float32,
    normal: Vec3f32[.WORLD],
    surface_value: UInt32,
    lambertians: Pointer[Float32, ImmutAnyOrigin],
    metals: Pointer[Float32, ImmutAnyOrigin],
    dielectrics: Pointer[Float32, ImmutAnyOrigin],
    light_kinds: Pointer[UInt32, ImmutAnyOrigin],
    light_fields: Pointer[Float32, ImmutAnyOrigin],
    light_count: Int,
    total_light_weight: Float32,
    rng_seed: UInt64,
    bounce: UInt32,
) -> GpuDirectLightSample:
    comptime assert integrator in (Integrator.NEE, Integrator.MIS)
    comptime assert light_kind in (
        PrimitiveKind.UNKNOWN,
        PrimitiveKind.SPHERE,
        PrimitiveKind.TRIANGLE,
    )
    if light_count <= 0 or total_light_weight <= 0.0:
        return _empty_direct_light_sample()

    var rng = path_stage_rng(
        rng_seed, path.path_id, wavefront_rng_light_stage(bounce)
    )
    var draw = _draw_alias_column(rng.f32(), light_count)
    var packed_column = light_kinds[unsafe_offset=draw.column]
    var alias_probability = light_fields[unsafe_offset=draw.column]
    var selected_idx = _resolve_alias_draw(
        draw, alias_probability, packed_column >> UInt32(4)
    )

    var base = light_count + selected_idx * GPU_RT_LIGHT_STRIDE
    var p0 = Point3f32[.WORLD](
        light_fields[unsafe_offset=base + GPU_RT_LIGHT_P0_X],
        light_fields[unsafe_offset=base + GPU_RT_LIGHT_P0_Y],
        light_fields[unsafe_offset=base + GPU_RT_LIGHT_P0_Z],
    )
    var surface_sample: _LightSurfaceSample
    comptime if light_kind == .SPHERE:
        var radius = light_fields[unsafe_offset=base + GPU_RT_LIGHT_RADIUS]
        surface_sample = _sample_sphere_light_surface(
            p0, radius, random_unit_vector[.WORLD](rng)
        )
    elif light_kind == .TRIANGLE:
        var p1 = Point3f32[.WORLD](
            light_fields[unsafe_offset=base + GPU_RT_LIGHT_P1_X],
            light_fields[unsafe_offset=base + GPU_RT_LIGHT_P1_Y],
            light_fields[unsafe_offset=base + GPU_RT_LIGHT_P1_Z],
        )
        var p2 = Point3f32[.WORLD](
            light_fields[unsafe_offset=base + GPU_RT_LIGHT_P2_X],
            light_fields[unsafe_offset=base + GPU_RT_LIGHT_P2_Y],
            light_fields[unsafe_offset=base + GPU_RT_LIGHT_P2_Z],
        )
        surface_sample = _sample_triangle_light_surface(
            p0, p1, p2, rng.f32(), rng.f32()
        )
    else:
        var kind = PrimitiveKind(
            light_kinds[unsafe_offset=selected_idx] & UInt32(0xF)
        )
        if kind == PrimitiveKind.SPHERE:
            var radius = light_fields[unsafe_offset=base + GPU_RT_LIGHT_RADIUS]
            surface_sample = _sample_sphere_light_surface(
                p0, radius, random_unit_vector[.WORLD](rng)
            )
        else:
            var p1 = Point3f32[.WORLD](
                light_fields[unsafe_offset=base + GPU_RT_LIGHT_P1_X],
                light_fields[unsafe_offset=base + GPU_RT_LIGHT_P1_Y],
                light_fields[unsafe_offset=base + GPU_RT_LIGHT_P1_Z],
            )
            var p2 = Point3f32[.WORLD](
                light_fields[unsafe_offset=base + GPU_RT_LIGHT_P2_X],
                light_fields[unsafe_offset=base + GPU_RT_LIGHT_P2_Y],
                light_fields[unsafe_offset=base + GPU_RT_LIGHT_P2_Z],
            )
            surface_sample = _sample_triangle_light_surface(
                p0, p1, p2, rng.f32(), rng.f32()
            )

    var emission = Color(
        light_fields[unsafe_offset=base + GPU_RT_LIGHT_E_X],
        light_fields[unsafe_offset=base + GPU_RT_LIGHT_E_Y],
        light_fields[unsafe_offset=base + GPU_RT_LIGHT_E_Z],
    )
    var geometry = _finish_direct_light_geometry(
        incoming_ray.o + hit_t * incoming_ray.d,
        normal,
        surface_sample,
        emission,
        total_light_weight,
    )
    if not geometry.valid:
        return _empty_direct_light_sample()
    var surface_kind = MaterialKind(surface_value >> UInt32(28))
    var material_idx = Int(surface_value & SURFACE_INDEX_MASK)
    var value = Color(0.0)
    var bsdf_pdf = Float32(0.0)
    if surface_kind == .LAMBERTIAN:
        var material_base = 3 * material_idx
        var evaluation = _evaluate_material[.LAMBERTIAN, 1](
            incoming_ray.d,
            normal,
            Color(
                lambertians[unsafe_offset=material_base + 0],
                lambertians[unsafe_offset=material_base + 1],
                lambertians[unsafe_offset=material_base + 2],
            ),
            1.0,
            geometry.direction,
        )
        value = evaluation.value
        bsdf_pdf = evaluation.pdf
    elif surface_kind == .METAL:
        var material_base = 4 * material_idx
        var evaluation = _evaluate_material[.METAL, 1](
            incoming_ray.d,
            normal,
            Color(
                metals[unsafe_offset=material_base + 0],
                metals[unsafe_offset=material_base + 1],
                metals[unsafe_offset=material_base + 2],
            ),
            metals[unsafe_offset=material_base + 3],
            geometry.direction,
        )
        value = evaluation.value
        bsdf_pdf = evaluation.pdf
    else:
        return _empty_direct_light_sample()
    var scale = _direct_light_scale[integrator, 1](
        geometry.surface_cosine, geometry.light_pdf, bsdf_pdf, True
    )
    return GpuDirectLightSample(
        scale > 0.0,
        geometry.direction,
        Color(path.tx, path.ty, path.tz) * value * emission * scale,
        geometry.shadow_t_max,
    )


@always_inline
def _accumulate_sample(
    sample_radiance: Pointer[Float32, MutAnyOrigin],
    capacity: Int,
    sample_base: UInt32,
    path_id: UInt32,
    value: Color,
):
    var idx = Int(path_id - sample_base)
    sample_radiance[
        unsafe_offset=wavefront_plane_index(WaveSampleFloatAbi.R, capacity, idx)
    ] += value.x
    sample_radiance[
        unsafe_offset=wavefront_plane_index(WaveSampleFloatAbi.G, capacity, idx)
    ] += value.y
    sample_radiance[
        unsafe_offset=wavefront_plane_index(WaveSampleFloatAbi.B, capacity, idx)
    ] += value.z


@always_inline
def _append_shade(
    work: DeviceWaveShade,
    path_refs: Pointer[UInt32, MutAnyOrigin],
    surface_values: Pointer[UInt32, MutAnyOrigin],
    fields: Pointer[Float32, MutAnyOrigin],
    counters: Pointer[UInt32, MutAnyOrigin],
    capacity: Int,
):
    var slot = _reserve_slot(counters, WAVE_COUNTER.SHADE)
    if slot >= capacity:
        _mark_status(counters, WAVE_STATUS.SHADE_OVERFLOW)
        return
    store_device_wave_shade(
        work, path_refs, surface_values, fields, capacity, slot
    )


@always_inline
def _append_shadow[
    STORE_CONTRIBUTION: Bool = True
](
    work: DeviceWaveShadow,
    path_ids: Pointer[UInt32, MutAnyOrigin],
    fields: Pointer[Float32, MutAnyOrigin],
    counters: Pointer[UInt32, MutAnyOrigin],
    capacity: Int,
):
    var slot = _reserve_slot(counters, WAVE_COUNTER.SHADOW)
    if slot >= capacity:
        _mark_status(counters, WAVE_STATUS.SHADOW_OVERFLOW)
        return
    store_gpu_rt_shadow[STORE_CONTRIBUTION](
        work, path_ids, fields, capacity, slot
    )


@always_inline
def _shade_lambertian_inline[
    integrator: Integrator,
](
    path: DeviceWavePath,
    ray_direction: Vec3f32[.WORLD],
    normal: Vec3f32[.WORLD],
    hit_t: Float32,
    surface_value: UInt32,
    lambertians: Pointer[Float32, ImmutAnyOrigin],
    dst_path_ids: Pointer[UInt32, MutAnyOrigin],
    dst_path_fields: Pointer[Float32, MutAnyOrigin],
    counters: Pointer[UInt32, MutAnyOrigin],
    capacity: Int,
    rng_seed: UInt64,
    bounce: UInt32,
):
    """Fuse the dominant diffuse shade operation into closest-hit routing."""
    var material_idx = Int(surface_value & SURFACE_INDEX_MASK)
    var base = 3 * material_idx
    var albedo = Color(
        lambertians[unsafe_offset=base + 0],
        lambertians[unsafe_offset=base + 1],
        lambertians[unsafe_offset=base + 2],
    )
    var rng = path_stage_rng(
        rng_seed, path.path_id, wavefront_rng_stage(bounce)
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
    if not sampled.ok:
        return
    var throughput = Color(path.tx, path.ty, path.tz) * sampled.weight
    var roulette = russian_roulette(
        rng_seed, path.path_id, bounce + UInt32(1), throughput
    )
    if not roulette.survived:
        return
    var slot = _reserve_slot(counters, WAVE_COUNTER.NEXT)
    if slot >= capacity:
        _mark_status(counters, WAVE_STATUS.PATH_OVERFLOW)
        return
    store_gpu_rt_path[integrator](
        DeviceWavePath(
            path.path_id,
            path.ox + hit_t * path.dx,
            path.oy + hit_t * path.dy,
            path.oz + hit_t * path.dz,
            0.001,
            sampled.direction.x,
            sampled.direction.y,
            sampled.direction.z,
            f32_max,
            roulette.throughput.x,
            roulette.throughput.y,
            roulette.throughput.z,
            sampled.pdf,
            sampled.delta,
        ),
        dst_path_ids,
        dst_path_fields,
        capacity,
        slot,
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
    hit_t: Float32,
    surface_value: UInt32,
    bounce: UInt32,
    total_light_weight: Float32,
    emissives: Pointer[Float32, ImmutAnyOrigin],
    lambertians: Pointer[Float32, ImmutAnyOrigin],
    dst_path_ids: Pointer[UInt32, MutAnyOrigin],
    dst_path_fields: Pointer[Float32, MutAnyOrigin],
    shade_path_refs: Pointer[UInt32, MutAnyOrigin],
    shade_surfaces: Pointer[UInt32, MutAnyOrigin],
    shade_fields: Pointer[Float32, MutAnyOrigin],
    counters: Pointer[UInt32, MutAnyOrigin],
    sample_radiance: Pointer[Float32, MutAnyOrigin],
    capacity: Int,
    sample_base: UInt32,
    rng_seed: UInt64,
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

    var kind = MaterialKind(surface_value >> UInt32(28))
    if kind == .EMISSIVE:
        if front_face:
            var material_idx = Int(surface_value & SURFACE_INDEX_MASK)
            var base = 3 * material_idx
            var radiance = Color(
                emissives[unsafe_offset=base + 0],
                emissives[unsafe_offset=base + 1],
                emissives[unsafe_offset=base + 2],
            )
            var emission_weight = Float32(1.0)
            comptime if integrator == .NEE:
                if bounce > 0 and not path.delta:
                    emission_weight = 0.0
            elif integrator == .MIS:
                if bounce > 0 and not path.delta:
                    var light_cosine = max(
                        dot(normal, -normalize(ray_direction)), 0.0
                    )
                    var light_pdf = Float32(0.0)
                    if light_cosine > 0.0 and total_light_weight > 0.0:
                        var distance_squared = (
                            hit_t * hit_t * length2(ray_direction)
                        )
                        light_pdf = _solid_angle_light_pdf(
                            distance_squared,
                            light_cosine,
                            radiance,
                            total_light_weight,
                        )
                    emission_weight = power_heuristic[1](
                        path.bsdf_pdf, light_pdf
                    )
            _accumulate_sample(
                sample_radiance,
                capacity,
                sample_base,
                path.path_id,
                Color(path.tx, path.ty, path.tz) * radiance * emission_weight,
            )
        return

    if kind == .LAMBERTIAN:
        _shade_lambertian_inline[integrator](
            path,
            ray_direction,
            normal,
            hit_t,
            surface_value,
            lambertians,
            dst_path_ids,
            dst_path_fields,
            counters,
            capacity,
            rng_seed,
            bounce,
        )
    elif kind == .METAL or kind == .DIELECTRIC:
        _append_shade(
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


@always_inline
def _make_ao_ray(
    rng_seed: UInt64,
    path: DeviceWavePath,
    incoming_ray: Rayf32[.WORLD],
    hit_t: Float32,
    normal: Vec3f32[.WORLD],
) -> Rayf32[.WORLD]:
    var rng = path_stage_rng(rng_seed, path.path_id, UInt32(1))
    var direction = random_on_hemisphere[.WORLD](rng, normal)
    return Rayf32[.WORLD](
        incoming_ray.o + hit_t * incoming_ray.d,
        direction,
        0.001,
        4.0,
    )


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
    rng_seed: UInt64,
    bounce: UInt32,
):
    comptime assert MATERIAL_KIND in (
        MaterialKind.LAMBERTIAN,
        MaterialKind.METAL,
        MaterialKind.DIELECTRIC,
    )
    var capacity = Int(capacity_i32)
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
    var ray_direction = Vec3f32[.WORLD](path.dx, path.dy, path.dz)
    var normal = Vec3f32[.WORLD](work.nx, work.ny, work.nz)
    var albedo = Color(0.0)
    var parameter = Float32(1.0)
    var random_u = Float32(0.0)
    var random_v = Float32(0.0)
    var material_idx = Int(work.surface_value & SURFACE_INDEX_MASK)
    var rng = path_stage_rng(
        rng_seed, path.path_id, wavefront_rng_stage(bounce)
    )

    comptime if MATERIAL_KIND == .LAMBERTIAN:
        var base = 3 * material_idx
        albedo = Color(
            material_data[unsafe_offset=base + 0],
            material_data[unsafe_offset=base + 1],
            material_data[unsafe_offset=base + 2],
        )
        random_u = rng.f32()
        random_v = rng.f32()
    elif MATERIAL_KIND == .METAL:
        var base = 4 * material_idx
        albedo = Color(
            material_data[unsafe_offset=base + 0],
            material_data[unsafe_offset=base + 1],
            material_data[unsafe_offset=base + 2],
        )
        parameter = material_data[unsafe_offset=base + 3]
        if parameter > 1.0e-4:
            random_u = rng.f32()
            random_v = rng.f32()
    else:
        parameter = material_data[unsafe_offset=material_idx]
        var ri = Float32(1.0) / parameter if work.front_face else parameter
        var unit_direction = normalize(ray_direction)
        var cos_theta = min(dot(-unit_direction, normal), 1.0)
        var sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0))
        if ri * sin_theta <= 1.0:
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
    if not sampled.ok:
        return

    var throughput = Color(path.tx, path.ty, path.tz) * sampled.weight
    var roulette = russian_roulette(
        rng_seed, path.path_id, bounce + UInt32(1), throughput
    )
    if not roulette.survived:
        return

    var slot = _reserve_slot(counters, WAVE_COUNTER.NEXT)
    if slot >= capacity:
        _mark_status(counters, WAVE_STATUS.PATH_OVERFLOW)
        return
    store_gpu_rt_path[integrator](
        DeviceWavePath(
            path.path_id,
            path.ox + work.t * path.dx,
            path.oy + work.t * path.dy,
            path.oz + work.t * path.dz,
            0.001,
            sampled.direction.x,
            sampled.direction.y,
            sampled.direction.z,
            f32_max,
            roulette.throughput.x,
            roulette.throughput.y,
            roulette.throughput.z,
            sampled.pdf,
            sampled.delta,
        ),
        dst_path_ids,
        dst_path_fields,
        capacity,
        slot,
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
    rng_seed: UInt64,
    bounce: UInt32,
):
    var work_count = Int(counters[unsafe_offset=WAVE_COUNTER.SHADE])
    var idx = global_idx.x
    var stride = Int(grid_dim.x * block_dim.x)
    while idx < work_count:
        var surface_value = shade_surfaces[unsafe_offset=idx]
        var kind = MaterialKind(surface_value >> UInt32(28))
        if kind == .METAL:
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
                rng_seed,
                bounce,
            )
        else:
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
                rng_seed,
                bounce,
            )
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
    rng_seed: UInt64,
    bounce: UInt32,
) raises:
    if not materials.has_non_lambertian:
        return
    var blocks = min(
        ceildiv(arena.capacity, GPU_RT_SHADE_BLOCK_SIZE),
        MAX_BLOCKS,
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
        rng_seed,
        bounce,
        grid_dim=blocks,
        block_dim=GPU_RT_SHADE_BLOCK_SIZE,
    )
