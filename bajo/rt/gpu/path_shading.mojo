"""Shared GPU RT material, lighting, routing, and shading kernels."""

from max.gpu import block_dim, global_idx, grid_dim
from std.math import abs, ceildiv, floor, sqrt
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh.gpu.utils import upload_list
from bajo.core import (
    Point3f32,
    Rayf32,
    Vec3f32,
    cross,
    dot,
    normalize,
)
from bajo.core.random import random_unit_vector
from bajo.rt.common import path_stage_rng, russian_roulette
from bajo.rt.lighting import (
    _direct_light_scale,
    _draw_alias_column,
    _emissive_hit_light_pdf,
    _emissive_hit_weight_from_pdf,
    _finish_direct_light_geometry,
    _LightSurfaceSample,
    _resolve_alias_draw,
    _sample_sphere_light_surface,
    _sample_triangle_light_surface,
)
from bajo.rt.rays import spawn_surface_ray
from bajo.rt.shading import (
    _evaluate_coated_diffuse,
    _evaluate_material,
    _sample_coated_diffuse,
    _sample_material,
)
from bajo.rt.types import (
    Color,
    MaterialKind,
    NO_TEXTURE,
    PrimitiveKind,
    Integrator,
    SceneData,
    SamplingConfig,
    SurfaceId,
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


def _flatten_lambertian_texture_indices(world: SceneData) -> List[UInt32]:
    var out = List[UInt32](capacity=len(world.surfaces().lambertians))
    for material in world.surfaces().lambertians:
        out.append(material.texture_index)
    return out^


def _flatten_texture_descs(world: SceneData) -> List[UInt32]:
    var out = List[UInt32](capacity=len(world.surfaces().image_textures) * 3)
    var pixel_offset = UInt32(0)
    for texture_idx in range(len(world.surfaces().image_textures)):
        ref texture = world.surfaces().image_textures[texture_idx]
        out.append(pixel_offset)
        out.append(UInt32(texture.width))
        out.append(UInt32(texture.height))
        pixel_offset += UInt32(len(texture.pixels))
    return out^


def _flatten_texture_pixels(world: SceneData) -> List[Float32]:
    var count = 0
    for texture_idx in range(len(world.surfaces().image_textures)):
        count += len(world.surfaces().image_textures[texture_idx].pixels)
    var out = List[Float32](capacity=count)
    for texture_idx in range(len(world.surfaces().image_textures)):
        ref texture = world.surfaces().image_textures[texture_idx]
        for value in texture.pixels:
            out.append(value)
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


def _flatten_coated_diffuses(world: SceneData) -> List[Float32]:
    var out = List[Float32](capacity=len(world.surfaces().coated_diffuses) * 15)
    for material in world.surfaces().coated_diffuses:
        out.append(material.albedo.x)
        out.append(material.albedo.y)
        out.append(material.albedo.z)
        out.append(material.roughness)
        out.append(material.eta)
        out.append(material.thickness)
        out.append(material.layer_albedo.x)
        out.append(material.layer_albedo.y)
        out.append(material.layer_albedo.z)
        out.append(material.g)
        out.append(Float32(material.max_depth))
        out.append(Float32(material.n_samples))
        out.append(material.displacement_scale)
        out.append(material.displacement_u_scale)
        out.append(material.displacement_v_scale)
    return out^


def _flatten_coated_diffuse_texture_indices(
    world: SceneData,
) -> List[UInt32]:
    var out = List[UInt32](capacity=len(world.surfaces().coated_diffuses) * 2)
    for material in world.surfaces().coated_diffuses:
        out.append(material.texture_index)
        out.append(material.displacement_texture_index)
    return out^


struct GpuRtMaterials:
    """Flattened device material tables shared by all GPU geometry backends."""

    var lambertians: DeviceBuffer[.float32]
    var metals: DeviceBuffer[.float32]
    var dielectrics: DeviceBuffer[.float32]
    var emissives: DeviceBuffer[.float32]
    var coated_diffuses: DeviceBuffer[.float32]
    var lambertian_texture_indices: DeviceBuffer[.uint32]
    var coated_diffuse_texture_indices: DeviceBuffer[.uint32]
    var texture_descs: DeviceBuffer[.uint32]
    var texture_pixels: DeviceBuffer[.float32]
    var has_non_lambertian: Bool

    def __init__(
        out self,
        mut ctx: DeviceContext,
        world: SceneData,
    ) raises:
        for material in world.surfaces().metals:
            if material.texture_index != NO_TEXTURE:
                raise Error("GPU metal image textures are not yet supported")
        self.lambertians = _upload_nonempty(ctx, _flatten_lambertians(world))
        self.metals = _upload_nonempty(ctx, _flatten_metals(world))
        self.dielectrics = _upload_nonempty(ctx, _flatten_dielectrics(world))
        self.emissives = _upload_nonempty(ctx, _flatten_emissives(world))
        self.coated_diffuses = _upload_nonempty(
            ctx, _flatten_coated_diffuses(world)
        )
        self.lambertian_texture_indices = _upload_nonempty(
            ctx, _flatten_lambertian_texture_indices(world)
        )
        self.coated_diffuse_texture_indices = _upload_nonempty(
            ctx, _flatten_coated_diffuse_texture_indices(world)
        )
        self.texture_descs = _upload_nonempty(
            ctx, _flatten_texture_descs(world)
        )
        self.texture_pixels = _upload_nonempty(
            ctx, _flatten_texture_pixels(world)
        )
        self.has_non_lambertian = (
            len(world.surfaces().metals) > 0
            or len(world.surfaces().dielectrics) > 0
        )


@always_inline
def _sample_gpu_image_texture(
    texture_idx: UInt32,
    u: Float32,
    v: Float32,
    texture_descs: Pointer[UInt32, ImmutAnyOrigin],
    texture_pixels: Pointer[Float32, ImmutAnyOrigin],
) -> Color:
    var desc_base = 3 * Int(texture_idx)
    var pixel_offset = Int(texture_descs[unsafe_offset=desc_base])
    var width = Int(texture_descs[unsafe_offset=desc_base + 1])
    var height = Int(texture_descs[unsafe_offset=desc_base + 2])
    var wrapped_u = u - floor(u)
    var wrapped_v = v - floor(v)
    var x = min(Int(wrapped_u * Float32(width)), width - 1)
    var y = min(Int((Float32(1.0) - wrapped_v) * Float32(height)), height - 1)
    var base = pixel_offset + 3 * (y * width + x)
    return Color(
        texture_pixels[unsafe_offset=base],
        texture_pixels[unsafe_offset=base + 1],
        texture_pixels[unsafe_offset=base + 2],
    )


@always_inline
def _sample_gpu_lambertian(
    surface_value: UInt32,
    u: Float32,
    v: Float32,
    lambertians: Pointer[Float32, ImmutAnyOrigin],
    texture_indices: Pointer[UInt32, ImmutAnyOrigin],
    texture_descs: Pointer[UInt32, ImmutAnyOrigin],
    texture_pixels: Pointer[Float32, ImmutAnyOrigin],
) -> Color:
    var material_idx = Int(SurfaceId.index_from_raw(surface_value))
    var material_base = 3 * material_idx
    var albedo = Color(
        lambertians[unsafe_offset=material_base + 0],
        lambertians[unsafe_offset=material_base + 1],
        lambertians[unsafe_offset=material_base + 2],
    )
    var texture_idx = texture_indices[unsafe_offset=material_idx]
    if texture_idx == NO_TEXTURE:
        return albedo
    return albedo * _sample_gpu_image_texture(
        texture_idx, u, v, texture_descs, texture_pixels
    )


@always_inline
def _sample_gpu_coated_albedo(
    surface_value: UInt32,
    u: Float32,
    v: Float32,
    coated_diffuses: Pointer[Float32, ImmutAnyOrigin],
    texture_indices: Pointer[UInt32, ImmutAnyOrigin],
    texture_descs: Pointer[UInt32, ImmutAnyOrigin],
    texture_pixels: Pointer[Float32, ImmutAnyOrigin],
) -> Color:
    var material_idx = Int(SurfaceId.index_from_raw(surface_value))
    var material_base = 15 * material_idx
    var albedo = Color(
        coated_diffuses[unsafe_offset=material_base + 0],
        coated_diffuses[unsafe_offset=material_base + 1],
        coated_diffuses[unsafe_offset=material_base + 2],
    )
    var texture_idx = texture_indices[unsafe_offset=2 * material_idx]
    if texture_idx == NO_TEXTURE:
        return albedo
    return albedo * _sample_gpu_image_texture(
        texture_idx, u, v, texture_descs, texture_pixels
    )


@always_inline
def _gpu_coated_shading_normal(
    surface_value: UInt32,
    normal: Vec3f32[.WORLD],
    u: Float32,
    v: Float32,
    coated_diffuses: Pointer[Float32, ImmutAnyOrigin],
    texture_indices: Pointer[UInt32, ImmutAnyOrigin],
    texture_descs: Pointer[UInt32, ImmutAnyOrigin],
    texture_pixels: Pointer[Float32, ImmutAnyOrigin],
) -> Vec3f32[.WORLD]:
    if SurfaceId.kind_from_raw(surface_value) != .COATED_DIFFUSE:
        return normal
    var material_idx = Int(SurfaceId.index_from_raw(surface_value))
    var material_base = 15 * material_idx
    var texture_idx = texture_indices[unsafe_offset=2 * material_idx + 1]
    var scale = coated_diffuses[unsafe_offset=material_base + 12]
    if texture_idx == NO_TEXTURE or scale == 0.0:
        return normal
    var u_scale = coated_diffuses[unsafe_offset=material_base + 13]
    var v_scale = coated_diffuses[unsafe_offset=material_base + 14]
    var desc_base = 3 * Int(texture_idx)
    var width = Float32(texture_descs[unsafe_offset=desc_base + 1])
    var height = Float32(texture_descs[unsafe_offset=desc_base + 2])
    var texture_u = u * u_scale
    var texture_v = v * v_scale
    var color_u0 = _sample_gpu_image_texture(
        texture_idx,
        texture_u - Float32(0.5) / width,
        texture_v,
        texture_descs,
        texture_pixels,
    )
    var color_u1 = _sample_gpu_image_texture(
        texture_idx,
        texture_u + Float32(0.5) / width,
        texture_v,
        texture_descs,
        texture_pixels,
    )
    var color_v0 = _sample_gpu_image_texture(
        texture_idx,
        texture_u,
        texture_v - Float32(0.5) / height,
        texture_descs,
        texture_pixels,
    )
    var color_v1 = _sample_gpu_image_texture(
        texture_idx,
        texture_u,
        texture_v + Float32(0.5) / height,
        texture_descs,
        texture_pixels,
    )
    var value_u0 = (
        Float32(0.2126) * color_u0.x
        + Float32(0.7152) * color_u0.y
        + Float32(0.0722) * color_u0.z
    )
    var value_u1 = (
        Float32(0.2126) * color_u1.x
        + Float32(0.7152) * color_u1.y
        + Float32(0.0722) * color_u1.z
    )
    var value_v0 = (
        Float32(0.2126) * color_v0.x
        + Float32(0.7152) * color_v0.y
        + Float32(0.0722) * color_v0.z
    )
    var value_v1 = (
        Float32(0.2126) * color_v1.x
        + Float32(0.7152) * color_v1.y
        + Float32(0.0722) * color_v1.z
    )
    var slope_u = (value_u1 - value_u0) * width * u_scale * scale
    var slope_v = (value_v1 - value_v0) * height * v_scale * scale
    var helper = Vec3f32[.WORLD](0.0, 1.0, 0.0)
    if abs(normal.y) > 0.99:
        helper = Vec3f32[.WORLD](1.0, 0.0, 0.0)
    var tangent = normalize(cross(helper, normal))
    var bitangent = cross(normal, tangent)
    return normalize(normal - tangent * slope_u - bitangent * slope_v)


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
    uv_u: Float32,
    uv_v: Float32,
    surface_value: UInt32,
    lambertians: Pointer[Float32, ImmutAnyOrigin],
    lambertian_texture_indices: Pointer[UInt32, ImmutAnyOrigin],
    texture_descs: Pointer[UInt32, ImmutAnyOrigin],
    texture_pixels: Pointer[Float32, ImmutAnyOrigin],
    metals: Pointer[Float32, ImmutAnyOrigin],
    coated_diffuses: Pointer[Float32, ImmutAnyOrigin],
    coated_diffuse_texture_indices: Pointer[UInt32, ImmutAnyOrigin],
    light_kinds: Pointer[UInt32, ImmutAnyOrigin],
    light_fields: Pointer[Float32, ImmutAnyOrigin],
    light_count: Int,
    total_light_weight: Float32,
    sampling: SamplingConfig,
    bounce: UInt32,
) -> GpuDirectLightSample:
    comptime assert Integrator.uses_direct_lighting[integrator]
    comptime assert light_kind in (
        PrimitiveKind.UNKNOWN,
        PrimitiveKind.SPHERE,
        PrimitiveKind.TRIANGLE,
    )
    if light_count <= 0 or total_light_weight <= 0.0:
        return _empty_direct_light_sample()

    var rng = path_stage_rng(
        sampling, path.path_id, wavefront_rng_light_stage(bounce)
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
        incoming_ray.at(hit_t),
        normal,
        surface_sample,
        emission,
        total_light_weight,
    )
    if not geometry.valid:
        return _empty_direct_light_sample()
    var surface_kind = SurfaceId.kind_from_raw(surface_value)
    var material_idx = Int(SurfaceId.index_from_raw(surface_value))
    var value = Color(0.0)
    var bsdf_pdf = Float32(0.0)
    if surface_kind == .LAMBERTIAN:
        var evaluation = _evaluate_material[.LAMBERTIAN, 1](
            incoming_ray.d,
            normal,
            _sample_gpu_lambertian(
                surface_value,
                uv_u,
                uv_v,
                lambertians,
                lambertian_texture_indices,
                texture_descs,
                texture_pixels,
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
    elif surface_kind == .COATED_DIFFUSE:
        var material_base = 15 * material_idx
        var evaluation = _evaluate_coated_diffuse(
            incoming_ray.d,
            normal,
            _sample_gpu_coated_albedo(
                surface_value,
                uv_u,
                uv_v,
                coated_diffuses,
                coated_diffuse_texture_indices,
                texture_descs,
                texture_pixels,
            ),
            coated_diffuses[unsafe_offset=material_base + 3],
            coated_diffuses[unsafe_offset=material_base + 4],
            coated_diffuses[unsafe_offset=material_base + 5],
            Color(
                coated_diffuses[unsafe_offset=material_base + 6],
                coated_diffuses[unsafe_offset=material_base + 7],
                coated_diffuses[unsafe_offset=material_base + 8],
            ),
            coated_diffuses[unsafe_offset=material_base + 9],
            coated_diffuses[unsafe_offset=material_base + 10],
            coated_diffuses[unsafe_offset=material_base + 11],
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
    if not sampled.ok:
        return
    var throughput = Color(path.tx, path.ty, path.tz) * sampled.weight
    var roulette = russian_roulette(
        sampling, path.path_id, bounce + UInt32(1), throughput
    )
    if not roulette.survived:
        return
    var slot = _reserve_slot(counters, WAVE_COUNTER.NEXT)
    if slot >= capacity:
        _mark_status(counters, WAVE_STATUS.PATH_OVERFLOW)
        return
    var next_ray = spawn_surface_ray(
        Point3f32[.WORLD](
            path.ox + hit_t * path.dx,
            path.oy + hit_t * path.dy,
            path.oz + hit_t * path.dz,
        ),
        sampled.direction,
    )
    store_gpu_rt_path[integrator](
        DeviceWavePath(
            path.path_id,
            next_ray.o.x,
            next_ray.o.y,
            next_ray.o.z,
            next_ray.t_min,
            next_ray.d.x,
            next_ray.d.y,
            next_ray.d.z,
            next_ray.t_max,
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
    var material_base = 15 * material_idx
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
        coated_diffuses[unsafe_offset=material_base + 3],
        coated_diffuses[unsafe_offset=material_base + 4],
        coated_diffuses[unsafe_offset=material_base + 5],
        Color(
            coated_diffuses[unsafe_offset=material_base + 6],
            coated_diffuses[unsafe_offset=material_base + 7],
            coated_diffuses[unsafe_offset=material_base + 8],
        ),
        coated_diffuses[unsafe_offset=material_base + 9],
        coated_diffuses[unsafe_offset=material_base + 10],
        coated_diffuses[unsafe_offset=material_base + 11],
        rng.f32(),
        rng.f32(),
    )
    if not sampled.ok:
        return
    var throughput = Color(path.tx, path.ty, path.tz) * sampled.weight
    var roulette = russian_roulette(
        sampling, path.path_id, bounce + UInt32(1), throughput
    )
    if not roulette.survived:
        return
    var slot = _reserve_slot(counters, WAVE_COUNTER.NEXT)
    if slot >= capacity:
        _mark_status(counters, WAVE_STATUS.PATH_OVERFLOW)
        return
    var next_ray = spawn_surface_ray(
        Point3f32[.WORLD](
            path.ox + hit_t * path.dx,
            path.oy + hit_t * path.dy,
            path.oz + hit_t * path.dz,
        ),
        sampled.direction,
    )
    store_gpu_rt_path[integrator](
        DeviceWavePath(
            path.path_id,
            next_ray.o.x,
            next_ray.o.y,
            next_ray.o.z,
            next_ray.t_min,
            next_ray.d.x,
            next_ray.d.y,
            next_ray.d.z,
            next_ray.t_max,
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
    if kind == .EMISSIVE:
        if front_face:
            var material_idx = Int(SurfaceId.index_from_raw(surface_value))
            var base = 3 * material_idx
            var radiance = Color(
                emissives[unsafe_offset=base + 0],
                emissives[unsafe_offset=base + 1],
                emissives[unsafe_offset=base + 2],
            )
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
                Color(path.tx, path.ty, path.tz) * radiance * emission_weight,
            )
        return

    if kind == .LAMBERTIAN:
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
    elif kind == .COATED_DIFFUSE:
        _shade_coated_diffuse_inline[integrator](
            path,
            ray_direction,
            normal,
            uv_u,
            uv_v,
            hit_t,
            surface_value,
            coated_diffuses,
            coated_diffuse_texture_indices,
            texture_descs,
            texture_pixels,
            dst_path_ids,
            dst_path_fields,
            counters,
            capacity,
            sampling,
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
    var material_idx = Int(SurfaceId.index_from_raw(work.surface_value))
    var rng = path_stage_rng(
        sampling, path.path_id, wavefront_rng_stage(bounce)
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
        sampling, path.path_id, bounce + UInt32(1), throughput
    )
    if not roulette.survived:
        return

    var slot = _reserve_slot(counters, WAVE_COUNTER.NEXT)
    if slot >= capacity:
        _mark_status(counters, WAVE_STATUS.PATH_OVERFLOW)
        return
    var next_ray = spawn_surface_ray(
        Point3f32[.WORLD](
            path.ox + work.t * path.dx,
            path.oy + work.t * path.dy,
            path.oz + work.t * path.dz,
        ),
        sampled.direction,
    )
    store_gpu_rt_path[integrator](
        DeviceWavePath(
            path.path_id,
            next_ray.o.x,
            next_ray.o.y,
            next_ray.o.z,
            next_ray.t_min,
            next_ray.d.x,
            next_ray.d.y,
            next_ray.d.z,
            next_ray.t_max,
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
    sampling: SamplingConfig,
    bounce: UInt32,
):
    var work_count = Int(counters[unsafe_offset=WAVE_COUNTER.SHADE])
    var idx = global_idx.x
    var stride = Int(grid_dim.x * block_dim.x)
    while idx < work_count:
        var surface_value = shade_surfaces[unsafe_offset=idx]
        var kind = SurfaceId.kind_from_raw(surface_value)
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
                sampling,
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
                sampling,
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
    sampling: SamplingConfig,
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
        sampling,
        bounce,
        grid_dim=blocks,
        block_dim=GPU_RT_SHADE_BLOCK_SIZE,
    )
