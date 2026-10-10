"""Prepared GPU light tables and device-safe direct-light sampling."""

from std.math import pi
from max.gpu.host import DeviceBuffer, DeviceContext
from bajo.bvh.constants import f32_max
from bajo.core import Affine3f32, Point3f32, Rayf32, Vec3f32, dot, normalize
from bajo.core.random import random_unit_vector
from bajo.rt.common import (
    equal_area_square_to_sphere,
    path_stage_rng,
    sky_color,
)
from bajo.rt.lighting import (
    _direct_light_scale,
    _draw_alias_column,
    _finish_direct_light_geometry,
    _DirectLightGeometrySample,
    _LightSurfaceSample,
    _resolve_alias_draw,
    _sample_sphere_light_surface,
    _sample_triangle_light_surface,
)
from bajo.rt.shading import _evaluate_coated_diffuse, _evaluate_material
from bajo.bvh.constants import PrimitiveKind
from bajo.rt.render_types import Color, SamplingConfig
from bajo.rt.material_types import EnvironmentKind, Integrator, SurfaceId
from bajo.rt.lighting_types import _light_importance
from bajo.rt.scene_data import SceneData
from bajo.rt.wavefront_contract import DeviceWavePath, wavefront_rng_light_stage
from bajo.rt.gpu.upload import _upload_nonempty
from bajo.rt.gpu.material_layout import (
    GpuMetalLayout,
    GpuTextureDescLayout,
    _load_gpu_material_color,
    _load_gpu_coated_parameters,
)
from bajo.rt.gpu.textures import (
    _sample_gpu_lambertian,
    _sample_gpu_coated_albedo,
)


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


struct GpuRtLights:
    """Compact device light records matching the CPU power distribution."""

    var kinds: DeviceBuffer[.uint32]
    var fields: DeviceBuffer[.float32]
    var count: Int
    var total_weight: Float32
    var environment_weight: Float32
    var environment_cdf_total: Float32
    var environment_cdf: DeviceBuffer[.float32]
    var environment_cdf_count: Int
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
        self.kinds = _upload_nonempty(ctx, kinds)
        self.fields = _upload_nonempty(ctx, fields)
        self.count = len(world.lights().records)
        self.total_weight = world.total_light_weight()
        self.environment_weight = world.environment_weight()
        self.environment_cdf_total = world.environment_cdf_total()
        self.environment_cdf_count = len(world.environment_cdf())
        self.environment_cdf = _upload_nonempty(ctx, world.environment_cdf())
        self.uniform_sampling_kind = (
            uniform_kind if homogeneous
            and self.environment_weight <= 0.0 else PrimitiveKind.UNKNOWN
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
    coated_pass: Bool = False,
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
    environment_weight: Float32,
    environment_cdf_total: Float32,
    environment_cdf: Pointer[Float32, ImmutAnyOrigin],
    environment_cdf_count: Int,
    environment_kind: UInt32,
    environment_texture_index: UInt32,
    environment_scale: Color,
    environment_light_to_world: Affine3f32[.LOCAL, .WORLD],
    sampling: SamplingConfig,
    bounce: UInt32,
) -> GpuDirectLightSample:
    comptime assert Integrator.uses_direct_lighting[integrator]
    comptime assert light_kind in (
        PrimitiveKind.UNKNOWN,
        PrimitiveKind.SPHERE,
        PrimitiveKind.TRIANGLE,
    )
    if total_light_weight <= 0.0:
        return _empty_direct_light_sample()

    var surface_kind = SurfaceId.kind_from_raw(surface_value)
    comptime if coated_pass:
        if surface_kind != .COATED_DIFFUSE:
            return _empty_direct_light_sample()
    else:
        if surface_kind == .COATED_DIFFUSE:
            return _empty_direct_light_sample()
    if surface_kind == .EMISSIVE or surface_kind == .DIELECTRIC:
        return _empty_direct_light_sample()

    comptime if not coated_pass:
        if surface_kind == .METAL:
            # Delta mirrors cannot contribute to a sampled direct-light ray.
            var material_base = GpuMetalLayout.STRIDE * Int(
                SurfaceId.index_from_raw(surface_value)
            )
            if (
                metals[unsafe_offset=material_base + GpuMetalLayout.FUZZ]
                <= 1.0e-4
            ):
                return _empty_direct_light_sample()

    var rng = path_stage_rng(
        sampling, path.path_id, wavefront_rng_light_stage(bounce)
    )
    var selection_u = rng.f32()
    var environment_probability = environment_weight / total_light_weight
    var emission = Color(0.0)
    var geometry = _DirectLightGeometrySample(
        False, Vec3f32[.WORLD](0.0), 0.0, 0.0, 0.0
    )
    if selection_u < environment_probability:
        var selected_pixel = 0
        var local_direction: Vec3f32[.LOCAL]
        if environment_cdf_count > 0:
            var target = rng.f32() * environment_cdf_total
            var first = 0
            var end = environment_cdf_count
            while first < end:
                var middle = first + (end - first) / 2
                if target < environment_cdf[unsafe_offset=middle]:
                    end = middle
                else:
                    first = middle + 1
            selected_pixel = min(first, environment_cdf_count - 1)
            var desc_base = GpuTextureDescLayout.STRIDE * Int(
                environment_texture_index
            )
            var width = Int(
                texture_descs[
                    unsafe_offset=desc_base + GpuTextureDescLayout.WIDTH
                ]
            )
            var height = Int(
                texture_descs[
                    unsafe_offset=desc_base + GpuTextureDescLayout.HEIGHT
                ]
            )
            var pixel_x = selected_pixel % width
            var pixel_y = selected_pixel / width
            var sample_u = (Float32(pixel_x) + rng.f32()) / Float32(width)
            var sample_v = (Float32(pixel_y) + rng.f32()) / Float32(height)
            local_direction = equal_area_square_to_sphere[.LOCAL, 1](
                sample_u, sample_v
            )
        else:
            local_direction = random_unit_vector[.LOCAL](rng)
        var direction = normalize(
            environment_light_to_world.vector(local_direction)
        )
        __match EnvironmentKind(environment_kind):
            case .IMAGE:
                var desc_base = GpuTextureDescLayout.STRIDE * Int(
                    environment_texture_index
                )
                var pixel_offset = Int(
                    texture_descs[
                        unsafe_offset=desc_base
                        + GpuTextureDescLayout.PIXEL_OFFSET
                    ]
                )
                var base = pixel_offset + 3 * selected_pixel
                emission = environment_scale * Color(
                    max(texture_pixels[unsafe_offset=base], 0.0),
                    max(texture_pixels[unsafe_offset=base + 1], 0.0),
                    max(texture_pixels[unsafe_offset=base + 2], 0.0),
                )
            case .PROCEDURAL:
                emission = sky_color(direction)
            case .BLACK | .UNIFORM:
                emission = environment_scale
        var surface_cosine = max(dot(normal, direction), 0.0)
        var light_pdf = environment_weight / (4.0 * pi * total_light_weight)
        __match EnvironmentKind(environment_kind):
            case .IMAGE:
                light_pdf = _light_importance(emission) / total_light_weight
            case .BLACK | .PROCEDURAL | .UNIFORM:
                pass
        geometry = _DirectLightGeometrySample(
            surface_cosine > 0.0 and light_pdf > 0.0,
            direction,
            surface_cosine,
            light_pdf,
            f32_max,
        )
    else:
        if light_count <= 0:
            return _empty_direct_light_sample()
        var finite_probability = 1.0 - environment_probability
        var finite_u = (
            selection_u - environment_probability
        ) / finite_probability if finite_probability > 0.0 else 0.0
        var draw = _draw_alias_column(
            min(finite_u, Float32(0.99999994)), light_count
        )
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
        comptime __match light_kind:
            case .SPHERE:
                var radius = light_fields[
                    unsafe_offset=base + GPU_RT_LIGHT_RADIUS
                ]
                surface_sample = _sample_sphere_light_surface(
                    p0, radius, random_unit_vector[.WORLD](rng)
                )
            case .TRIANGLE:
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
            case _:
                var kind = PrimitiveKind(
                    light_kinds[unsafe_offset=selected_idx] & UInt32(0xF)
                )
                if kind == PrimitiveKind.SPHERE:
                    var radius = light_fields[
                        unsafe_offset=base + GPU_RT_LIGHT_RADIUS
                    ]
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
        emission = Color(
            light_fields[unsafe_offset=base + GPU_RT_LIGHT_E_X],
            light_fields[unsafe_offset=base + GPU_RT_LIGHT_E_Y],
            light_fields[unsafe_offset=base + GPU_RT_LIGHT_E_Z],
        )
        geometry = _finish_direct_light_geometry(
            incoming_ray.at(hit_t),
            normal,
            surface_sample,
            emission,
            total_light_weight,
        )
    if not geometry.valid:
        return _empty_direct_light_sample()
    var material_idx = Int(SurfaceId.index_from_raw(surface_value))
    var value = Color(0.0)
    var bsdf_pdf = Float32(0.0)
    if surface_kind == .LAMBERTIAN:
        comptime if not coated_pass:
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
        else:
            return _empty_direct_light_sample()
    elif surface_kind == .METAL:
        comptime if not coated_pass:
            var material_base = GpuMetalLayout.STRIDE * material_idx
            var evaluation = _evaluate_material[.METAL, 1](
                incoming_ray.d,
                normal,
                _load_gpu_material_color(
                    metals, material_base + GpuMetalLayout.ALBEDO
                ),
                metals[unsafe_offset=material_base + GpuMetalLayout.FUZZ],
                geometry.direction,
            )
            value = evaluation.value
            bsdf_pdf = evaluation.pdf
        else:
            return _empty_direct_light_sample()
    elif surface_kind == .COATED_DIFFUSE:
        comptime if coated_pass:
            var coated = _load_gpu_coated_parameters(
                coated_diffuses, material_idx
            )
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
                coated.roughness,
                coated.eta,
                coated.thickness,
                coated.layer_albedo,
                coated.g,
                coated.max_depth,
                coated.n_samples,
                geometry.direction,
            )
            value = evaluation.value
            bsdf_pdf = evaluation.pdf
        else:
            return _empty_direct_light_sample()
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
