"""Prepared GPU light tables and device-safe direct-light sampling."""

from std.utils.numerics import max_finite

from std.math import pi
from max.gpu.host import DeviceBuffer, DeviceContext
from bajo.bvh.constants import PrimitiveKind
from bajo.core import Point3f32, Rayf32, Vec3f32, dot, normalize
from bajo.core.tensor_layout import record_tile
from bajo.core.random import random_unit_vector
from bajo.core.mappings import equal_area_square_to_sphere
from bajo.rt.common import path_stage_rng, path_stage_seed, sky_color
from bajo.rt.lighting import (
    _direct_light_scale,
    _finish_direct_light_geometry,
    _DirectLightGeometrySample,
    _LightSurfaceSample,
    _sample_sphere_light_surface,
    _sample_triangle_light_surface,
)
from bajo.core.sampling import draw_alias_column, resolve_alias_draw
from bajo.rt.shading import (
    METAL_DELTA_FUZZ_THRESHOLD,
    _evaluate_coated_diffuse,
    _evaluate_material,
)

from bajo.core.color import Color
from bajo.rt.render_types import SamplingConfig
from bajo.rt.material_types import Integrator, SurfaceId
from bajo.rt.lighting_types import _light_importance
from bajo.rt.texture_math import load_environment_texel
from bajo.rt.scene_data import SceneData
from bajo.rt.wavefront_contract import (
    DeviceWavePath,
    wavefront_rng_light_stage,
    wavefront_rng_stage,
)
from bajo.rt.gpu.views import GpuRtSceneView
from bajo.rt.gpu.upload import _upload_nonempty
from bajo.rt.gpu.material_layout import GpuMetalLayout, GpuTextureDescLayout
from bajo.rt.gpu.textures import (
    _sample_gpu_lambertian,
    _sample_gpu_coated_albedo,
    _sample_gpu_coated_parameters,
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
    var environment_alias_probabilities: DeviceBuffer[.float32]
    var environment_alias_indices: DeviceBuffer[.uint32]
    var environment_alias_count: Int
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
        fields.extend(Span(world.lights().alias_probabilities))
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
        self.environment_alias_count = len(
            world.environment_alias_probabilities()
        )
        if self.environment_alias_count > Int(Int32.MAX):
            raise Error("GPU environment table exceeds Int32 indexing")
        self.environment_alias_probabilities = _upload_nonempty(
            ctx, world.environment_alias_probabilities()
        )
        self.environment_alias_indices = _upload_nonempty(
            ctx, world.environment_alias_indices()
        )
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
    var light_id: UInt32

    @staticmethod
    @always_inline
    def make_empty() -> Self:
        return Self(False, Vec3f32[.WORLD](0.0), Color(0.0), 0.0, UInt32(0))


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
    scene: GpuRtSceneView,
    sampling: SamplingConfig,
    bounce: UInt32,
) -> GpuDirectLightSample:
    comptime assert Integrator.uses_direct_lighting[integrator]
    comptime assert light_kind in (
        PrimitiveKind.UNKNOWN,
        PrimitiveKind.SPHERE,
        PrimitiveKind.TRIANGLE,
    )
    if scene.total_light_weight <= 0.0:
        return GpuDirectLightSample.make_empty()

    var surface_kind = SurfaceId.kind_from_raw(surface_value)
    comptime if coated_pass:
        if surface_kind != .COATED_DIFFUSE:
            return GpuDirectLightSample.make_empty()
    else:
        if surface_kind == .COATED_DIFFUSE:
            return GpuDirectLightSample.make_empty()
    if surface_kind == .EMISSIVE or surface_kind == .DIELECTRIC:
        return GpuDirectLightSample.make_empty()

    comptime if not coated_pass:
        if surface_kind == .METAL:
            # Delta mirrors cannot contribute to a sampled direct-light ray.
            var material_base = GpuMetalLayout.STRIDE * Int(
                SurfaceId.index_from_raw(surface_value)
            )
            if (
                scene.metals[unsafe_offset=material_base + GpuMetalLayout.FUZZ]
                <= METAL_DELTA_FUZZ_THRESHOLD
            ):
                return GpuDirectLightSample.make_empty()

    var rng = path_stage_rng(
        sampling, path.path_id, wavefront_rng_light_stage(bounce)
    )
    var selection_u = rng.f32()
    var environment_probability = (
        scene.environment_weight / scene.total_light_weight
    )
    var light_id = UInt32(Int(scene.light_count))
    var emission: Color
    var geometry: _DirectLightGeometrySample[1]
    if selection_u < environment_probability:
        var selected_pixel = 0
        var local_direction: Vec3f32[.LOCAL]
        if Int(scene.environment_alias_count) > 0:
            var draw = draw_alias_column(
                rng.f32(), Int(scene.environment_alias_count)
            )
            selected_pixel = Int(
                resolve_alias_draw(
                    draw,
                    scene.environment_alias_probabilities[
                        unsafe_offset=draw.column
                    ],
                    scene.environment_alias_indices[unsafe_offset=draw.column],
                )
            )
            var desc_base = GpuTextureDescLayout.STRIDE * Int(
                scene.environment_texture_index
            )
            var width = Int(
                scene.texture_descs[
                    unsafe_offset=desc_base + GpuTextureDescLayout.WIDTH
                ]
            )
            var height = Int(
                scene.texture_descs[
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
            scene.environment_light_to_world.vector(local_direction)
        )
        match scene.environment_kind:
            case .IMAGE:
                var desc_base = GpuTextureDescLayout.STRIDE * Int(
                    scene.environment_texture_index
                )
                var pixel_offset = Int(
                    scene.texture_descs[
                        unsafe_offset=desc_base
                        + GpuTextureDescLayout.PIXEL_OFFSET
                    ]
                )
                emission = Color(
                    scene.environment_scale_x,
                    scene.environment_scale_y,
                    scene.environment_scale_z,
                ) * load_environment_texel(
                    record_tile[.float32, 3](
                        scene.texture_pixels.unsafe_offset(pixel_offset),
                        selected_pixel + 1,
                    ),
                    Int64(selected_pixel),
                )
            case .PROCEDURAL:
                emission = sky_color(direction)
            case .BLACK | .UNIFORM:
                emission = Color(
                    scene.environment_scale_x,
                    scene.environment_scale_y,
                    scene.environment_scale_z,
                )
        var surface_cosine = max(dot(normal, direction), 0.0)
        var light_pdf = scene.environment_weight / (
            4.0 * pi * scene.total_light_weight
        )
        match scene.environment_kind:
            case .IMAGE:
                light_pdf = (
                    _light_importance(emission) / scene.total_light_weight
                )
            case .BLACK | .PROCEDURAL | .UNIFORM:
                pass
        geometry = _DirectLightGeometrySample(
            surface_cosine > 0.0 and light_pdf > 0.0,
            direction,
            surface_cosine,
            light_pdf,
            max_finite[.float32](),
        )
    else:
        if Int(scene.light_count) <= 0:
            return GpuDirectLightSample.make_empty()
        var finite_probability = 1.0 - environment_probability
        var finite_u = (
            selection_u - environment_probability
        ) / finite_probability if finite_probability > 0.0 else 0.0
        var draw = draw_alias_column(
            min(finite_u, Float32(0.99999994)), Int(scene.light_count)
        )
        var packed_column = scene.light_kinds[unsafe_offset=draw.column]
        var alias_probability = scene.light_fields[unsafe_offset=draw.column]
        var selected_idx = resolve_alias_draw(
            draw, alias_probability, packed_column >> UInt32(4)
        )
        light_id = UInt32(selected_idx)
        var base = (
            Int(scene.light_count) + Int(selected_idx) * GPU_RT_LIGHT_STRIDE
        )
        var p0 = Point3f32[.WORLD](
            scene.light_fields.unsafe_offset(base + GPU_RT_LIGHT_P0_X)
        )
        var surface_sample: _LightSurfaceSample[1]
        comptime match light_kind:
            case .SPHERE:
                var radius = scene.light_fields[
                    unsafe_offset=base + GPU_RT_LIGHT_RADIUS
                ]
                surface_sample = _sample_sphere_light_surface(
                    p0, radius, random_unit_vector[.WORLD](rng)
                )
            case .TRIANGLE:
                var p1 = Point3f32[.WORLD](
                    scene.light_fields.unsafe_offset(base + GPU_RT_LIGHT_P1_X)
                )
                var p2 = Point3f32[.WORLD](
                    scene.light_fields.unsafe_offset(base + GPU_RT_LIGHT_P2_X)
                )
                surface_sample = _sample_triangle_light_surface(
                    p0, p1, p2, rng.f32(), rng.f32()
                )
            case _:
                var kind = PrimitiveKind(
                    scene.light_kinds[unsafe_offset=selected_idx] & UInt32(0xF)
                )
                if kind == PrimitiveKind.SPHERE:
                    var radius = scene.light_fields[
                        unsafe_offset=base + GPU_RT_LIGHT_RADIUS
                    ]
                    surface_sample = _sample_sphere_light_surface(
                        p0, radius, random_unit_vector[.WORLD](rng)
                    )
                else:
                    var p1 = Point3f32[.WORLD](
                        scene.light_fields.unsafe_offset(
                            base + GPU_RT_LIGHT_P1_X
                        )
                    )
                    var p2 = Point3f32[.WORLD](
                        scene.light_fields.unsafe_offset(
                            base + GPU_RT_LIGHT_P2_X
                        )
                    )
                    surface_sample = _sample_triangle_light_surface(
                        p0, p1, p2, rng.f32(), rng.f32()
                    )
        emission = Color(
            scene.light_fields.unsafe_offset(base + GPU_RT_LIGHT_E_X)
        )
        geometry = _finish_direct_light_geometry(
            incoming_ray.at(hit_t),
            normal,
            surface_sample,
            emission,
            scene.total_light_weight,
        )
    if not geometry.valid:
        return GpuDirectLightSample.make_empty()
    var material_idx = Int(SurfaceId.index_from_raw(surface_value))
    var value = Color(0.0)
    var bsdf_pdf = Float32(0.0)
    match surface_kind:
        case .LAMBERTIAN:
            comptime if not coated_pass:
                var evaluation = _evaluate_material[.LAMBERTIAN, 1](
                    incoming_ray.d,
                    normal,
                    _sample_gpu_lambertian(
                        surface_value,
                        uv_u,
                        uv_v,
                        scene.lambertians.unsafe_origin_cast[ImmutAnyOrigin](),
                        scene.lambertian_texture_indices.unsafe_origin_cast[
                            ImmutAnyOrigin
                        ](),
                        scene.texture_descs.unsafe_origin_cast[
                            ImmutAnyOrigin
                        ](),
                        scene.texture_pixels.unsafe_origin_cast[
                            ImmutAnyOrigin
                        ](),
                    ),
                    1.0,
                    geometry.direction,
                )
                value = evaluation.value
                bsdf_pdf = evaluation.pdf
            else:
                return GpuDirectLightSample.make_empty()
        case .METAL:
            comptime if not coated_pass:
                var material_base = GpuMetalLayout.STRIDE * material_idx
                var evaluation = _evaluate_material[.METAL, 1](
                    incoming_ray.d,
                    normal,
                    Color(
                        scene.metals.unsafe_offset(
                            material_base + GpuMetalLayout.ALBEDO
                        )
                    ),
                    scene.metals[
                        unsafe_offset=material_base + GpuMetalLayout.FUZZ
                    ],
                    geometry.direction,
                )
                value = evaluation.value
                bsdf_pdf = evaluation.pdf
            else:
                return GpuDirectLightSample.make_empty()
        case .COATED_DIFFUSE:
            comptime if coated_pass:
                var coated = _sample_gpu_coated_parameters(
                    scene.coated_diffuses.unsafe_origin_cast[ImmutAnyOrigin](),
                    material_idx,
                    uv_u,
                    uv_v,
                    scene.coated_diffuse_texture_indices.unsafe_origin_cast[
                        ImmutAnyOrigin
                    ](),
                    scene.texture_descs.unsafe_origin_cast[ImmutAnyOrigin](),
                    scene.texture_pixels.unsafe_origin_cast[ImmutAnyOrigin](),
                )
                var evaluation = _evaluate_coated_diffuse(
                    incoming_ray.d,
                    normal,
                    _sample_gpu_coated_albedo(
                        surface_value,
                        uv_u,
                        uv_v,
                        scene.coated_diffuses.unsafe_origin_cast[
                            ImmutAnyOrigin
                        ](),
                        scene.coated_diffuse_texture_indices.unsafe_origin_cast[
                            ImmutAnyOrigin
                        ](),
                        scene.texture_descs.unsafe_origin_cast[
                            ImmutAnyOrigin
                        ](),
                        scene.texture_pixels.unsafe_origin_cast[
                            ImmutAnyOrigin
                        ](),
                    ),
                    coated,
                    geometry.direction,
                    estimator_seed=path_stage_seed(
                        sampling, path.path_id, wavefront_rng_stage(bounce)
                    ),
                )
                value = evaluation.value
                bsdf_pdf = evaluation.pdf
            else:
                return GpuDirectLightSample.make_empty()
        case _:
            return GpuDirectLightSample.make_empty()
    var scale = _direct_light_scale[integrator, 1](
        geometry.surface_cosine, geometry.light_pdf, bsdf_pdf, True
    )
    return GpuDirectLightSample(
        scale > 0.0,
        geometry.direction,
        Color(path.tx, path.ty, path.tz) * value * emission * scale,
        geometry.shadow_t_max,
        light_id,
    )
