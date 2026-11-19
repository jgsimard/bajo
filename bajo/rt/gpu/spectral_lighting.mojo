"""Prepared GPU light tables and device-safe direct-light sampling."""

from layout import TileTensor, Coord, Idx, row_major
from std.utils.numerics import max_finite

from std.math import abs, pi
from std.memory import bitcast
from bajo.bvh.constants import PrimitiveKind
from bajo.core import Point3f32, Rayf32, Vec3f32, dot, normalize
from bajo.core.random import random_unit_vector
from bajo.core.mappings import equal_area_square_to_sphere
from bajo.rt.common import path_stage_rng, sky_color
from bajo.rt.lighting import (
    _finish_direct_light_geometry,
    _DirectLightGeometrySample,
    _LightSurfaceSample,
    _sample_sphere_light_surface,
    _sample_triangle_light_surface,
    light_sample_uv,
)
from bajo.core.sampling import draw_alias_column, resolve_alias_draw

from bajo.core.color import Color
from bajo.rt.render_types import SamplingConfig
from bajo.rt.material_types import Integrator, SurfaceId
from bajo.rt.wavefront_contract import (
    DeviceWavePath,
    wavefront_rng_light_stage,
)
from bajo.rt.gpu.views import GpuRtSceneView
from bajo.rt.gpu.opacity import gpu_opacity_value
from bajo.rt.opacity import opacity_accepts, opacity_point_sample
from bajo.rt.gpu.material_layout import GpuTextureDescLayout
from bajo.rt.gpu.textures import _GpuImageTextureView
from bajo.rt.gpu.light_layout import GPU_RT_LIGHT


@fieldwise_init
struct SpectralLightCandidate(TrivialRegisterPassable):
    var valid: Bool
    var direction: Vec3f32[.WORLD]
    var emission: Color
    var cosine: Float32
    var pdf: Float32
    var distance: Float32
    var primitive: UInt32

    @staticmethod
    def make_empty() -> Self:
        return Self(False, Vec3f32[.WORLD](0), Color(0), 0, 0, 0, UInt32.MAX)


def sample_spectral_light[
    integrator: Integrator,
    light_kind: PrimitiveKind = .UNKNOWN,
](
    path: DeviceWavePath,
    incoming_ray: Rayf32[.WORLD],
    hit_t: Float32,
    normal: Vec3f32[.WORLD],
    surface_value: UInt32,
    scene: GpuRtSceneView,
    sampling: SamplingConfig,
    bounce: UInt32,
) -> SpectralLightCandidate:
    var lighting = scene.spectral.value().lighting
    if lighting.total_weight <= 0:
        return SpectralLightCandidate.make_empty()
    var surface_kind = SurfaceId.kind_from_raw(surface_value)
    var rng = path_stage_rng(
        sampling, path.path_id, wavefront_rng_light_stage(bounce)
    )
    var selection_u = rng.f32()
    var environment_probability = (
        lighting.environment_weight / lighting.total_weight
    )
    var light_id = UInt32(Int(scene.light_count))
    var emission: Color
    var primitive = UInt32.MAX
    var geometry: _DirectLightGeometrySample[1]
    if selection_u < environment_probability:
        var selected_pixel = 0
        var local_direction: Vec3f32[.LOCAL]
        if Int(lighting.environment_count) > 0:
            var draw = draw_alias_column(
                rng.f32(), Int(lighting.environment_count)
            )
            selected_pixel = Int(
                resolve_alias_draw(
                    draw,
                    lighting.environment_probabilities[
                        unsafe_offset=draw.column
                    ],
                    lighting.environment_aliases[unsafe_offset=draw.column],
                )
            )
            var texture_index = Int(scene.environment_texture_index)
            var descriptors = TileTensor(
                ptr=scene.texture_descs,
                layout=row_major(
                    Coord(texture_index + 1, Idx[GpuTextureDescLayout.STRIDE])
                ),
            )
            var width = Int(
                descriptors.load[invariant=False](
                    Coord(texture_index, GpuTextureDescLayout.WIDTH)
                )
            )
            var height = Int(
                descriptors.load[invariant=False](
                    Coord(texture_index, GpuTextureDescLayout.HEIGHT)
                )
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
                emission = Color(
                    scene.environment_scale_x,
                    scene.environment_scale_y,
                    scene.environment_scale_z,
                ) * _GpuImageTextureView.load(
                    scene.environment_texture_index,
                    scene.texture_descs.unsafe_origin_cast[ImmutAnyOrigin](),
                    scene.texture_pixels.unsafe_origin_cast[ImmutAnyOrigin](),
                ).environment_texel(
                    selected_pixel
                )
            case .PROCEDURAL:
                emission = sky_color(direction)
            case .BLACK | .UNIFORM:
                emission = Color(
                    scene.environment_scale_x,
                    scene.environment_scale_y,
                    scene.environment_scale_z,
                )
        var cosine = dot(normal, direction)
        var surface_cosine = abs(
            cosine
        ) if surface_kind == .DIELECTRIC else max(cosine, 0.0)
        var light_pdf = lighting.environment_weight / (
            4.0 * pi * lighting.total_weight
        )
        match scene.environment_kind:
            case .IMAGE:
                var texel = _GpuImageTextureView.load(
                    scene.environment_texture_index,
                    scene.texture_descs.unsafe_origin_cast[ImmutAnyOrigin](),
                    scene.texture_pixels.unsafe_origin_cast[ImmutAnyOrigin](),
                ).environment_texel(selected_pixel)
                light_pdf = lighting.environment_pdf(.IMAGE, texel)
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
            return SpectralLightCandidate.make_empty()
        var finite_probability = 1.0 - environment_probability
        var finite_u = (
            selection_u - environment_probability
        ) / finite_probability if finite_probability > 0.0 else 0.0
        var draw = draw_alias_column(
            min(finite_u, Float32(0.99999994)), Int(scene.light_count)
        )
        var alias_probability = lighting.probabilities[
            unsafe_offset=draw.column
        ]
        var selected_idx = resolve_alias_draw(
            draw, alias_probability, lighting.aliases[unsafe_offset=draw.column]
        )
        light_id = UInt32(selected_idx)
        var lights = TileTensor(
            ptr=scene.light_fields.unsafe_offset(Int(scene.light_count)),
            layout=row_major(
                Coord(Int(scene.light_count), Idx[GPU_RT_LIGHT.STRIDE])
            ),
        )
        var record = lights[Int(selected_idx), :]
        var p0 = Point3f32[.WORLD](
            record.ptr_at_offset(Coord(GPU_RT_LIGHT.P0_X))
        )
        var surface_sample: _LightSurfaceSample[1]
        comptime match light_kind:
            case .SPHERE:
                var radius = record.load[invariant=False](
                    Coord(GPU_RT_LIGHT.RADIUS)
                )
                surface_sample = _sample_sphere_light_surface(
                    p0, radius, random_unit_vector[.WORLD](rng)
                )
            case .TRIANGLE:
                var p1 = Point3f32[.WORLD](
                    record.ptr_at_offset(Coord(GPU_RT_LIGHT.P1_X))
                )
                var p2 = Point3f32[.WORLD](
                    record.ptr_at_offset(Coord(GPU_RT_LIGHT.P2_X))
                )
                surface_sample = _sample_triangle_light_surface(
                    p0, p1, p2, rng.f32(), rng.f32()
                )
            case _:
                var kind = PrimitiveKind(
                    scene.light_kinds[unsafe_offset=selected_idx] & UInt32(0xF)
                )
                if kind == PrimitiveKind.SPHERE:
                    var radius = record.load[invariant=False](
                        Coord(GPU_RT_LIGHT.RADIUS)
                    )
                    surface_sample = _sample_sphere_light_surface(
                        p0, radius, random_unit_vector[.WORLD](rng)
                    )
                else:
                    var p1 = Point3f32[.WORLD](
                        record.ptr_at_offset(Coord(GPU_RT_LIGHT.P1_X))
                    )
                    var p2 = Point3f32[.WORLD](
                        record.ptr_at_offset(Coord(GPU_RT_LIGHT.P2_X))
                    )
                    surface_sample = _sample_triangle_light_surface(
                        p0, p1, p2, rng.f32(), rng.f32()
                    )
        emission = Color(record.ptr_at_offset(Coord(GPU_RT_LIGHT.E_X)))
        if scene.opacity.count > 0:
            var texture_u = surface_sample.u
            var texture_v = surface_sample.v
            var kind = PrimitiveKind(
                scene.light_kinds[unsafe_offset=selected_idx] & UInt32(0xF)
            )
            if kind != PrimitiveKind.SPHERE:
                var uv = light_sample_uv(
                    record.load[invariant=False](Coord(GPU_RT_LIGHT.UV0)),
                    record.load[invariant=False](Coord(GPU_RT_LIGHT.UV0 + 1)),
                    record.load[invariant=False](Coord(GPU_RT_LIGHT.UV0 + 2)),
                    record.load[invariant=False](Coord(GPU_RT_LIGHT.UV0 + 3)),
                    record.load[invariant=False](Coord(GPU_RT_LIGHT.UV0 + 4)),
                    record.load[invariant=False](Coord(GPU_RT_LIGHT.UV0 + 5)),
                    surface_sample.u,
                    surface_sample.v,
                )
                texture_u = uv[0]
                texture_v = uv[1]
            var opacity_primitive = bitcast[.uint32](
                record.load[invariant=False](Coord(GPU_RT_LIGHT.PRIMITIVE))
            )
            surface_sample.valid &= opacity_accepts(
                gpu_opacity_value(
                    scene.opacity, opacity_primitive, texture_u, texture_v
                ),
                opacity_point_sample(surface_sample.point),
            )
        geometry = _finish_direct_light_geometry(
            incoming_ray.at(hit_t),
            normal,
            surface_sample,
            Color(lighting.weights[unsafe_offset=Int(light_id)]),
            lighting.total_weight,
            surface_kind == .DIELECTRIC,
            bitcast[.uint32](
                record.load[invariant=False](Coord(GPU_RT_LIGHT.FLAGS))
            )
            != 0,
        )
        primitive = bitcast[.uint32](
            record.load[invariant=False](Coord(GPU_RT_LIGHT.PRIMITIVE))
        )
    return SpectralLightCandidate(
        geometry.valid,
        geometry.direction,
        emission,
        geometry.surface_cosine,
        geometry.light_pdf,
        geometry.shadow_t_max,
        primitive,
    )
