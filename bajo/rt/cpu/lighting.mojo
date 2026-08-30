"""Emission and next-event estimation for CPU path integrators."""

from bajo.core import (
    Rayf32,
    Vec3,
)
from bajo.core.random import Rng, random_unit_vector
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
from bajo.rt.types import (
    Color,
    PrimitiveKind,
    Integrator,
    ShadingPoint,
    SurfaceId,
    SurfaceHit,
    SurfaceStore,
)
from .scene import CpuScene
from .bsdf import evaluate_bsdf


@fieldwise_init
struct _DirectLightSample[length: SIMDLength = 1](Copyable):
    """Visible light sample before applying the surface BSDF."""

    var valid: SIMD[.bool, Self.length]
    var direction: Vec3[.float32, .WORLD, Self.length]
    var emission: Vec3[.float32, .WORLD, Self.length]
    var surface_cosine: SIMD[.float32, Self.length]
    var light_pdf: SIMD[.float32, Self.length]
    var shadow_t_max: SIMD[.float32, Self.length]


@always_inline
def _empty_direct_light_sample[
    length: SIMDLength = 1
]() -> _DirectLightSample[length]:
    return _DirectLightSample[length](
        SIMD[.bool, length](fill=False),
        Vec3[.float32, .WORLD, length](0.0),
        Vec3[.float32, .WORLD, length](0.0),
        0.0,
        0.0,
        0.0,
    )


@always_inline
def _emissive_hit_weight[
    integrator: Integrator,
    world_bvh_width: SIMDLength,
    instance_bvh_width: SIMDLength,
](
    world: CpuScene[world_bvh_width, instance_bvh_width],
    ray: Rayf32[.WORLD],
    hit: SurfaceHit[1],
    bounce: Int,
    previous_bsdf_pdf: Float32,
    previous_delta: Bool,
) -> Float32:
    var light_pdf = Float32(0.0)
    comptime if integrator == .MIS:
        if bounce > 0 and not previous_delta:
            light_pdf = light_pdf_for_emissive_hit(world, ray, hit)
    return _emissive_hit_weight_from_pdf[integrator](
        UInt32(bounce), previous_delta, previous_bsdf_pdf, light_pdf
    )


def light_pdf_for_emissive_hit[
    world_bvh_width: SIMDLength,
    instance_bvh_width: SIMDLength,
](
    world: CpuScene[world_bvh_width, instance_bvh_width],
    ray: Rayf32[.WORLD],
    hit: SurfaceHit[1],
) -> Float32:
    """Evaluate the triangle-light distribution in solid-angle measure."""
    if hit.surface.kind() != .EMISSIVE or not hit.front_face:
        return 0.0
    var total_weight = world.scene_data().lights().total_weight
    var radiance = (
        world.scene_data()
        .surfaces()
        .emissives[Int(hit.surface.index())]
        .radiance
    )
    return _emissive_hit_light_pdf(
        ray.d, hit.t, hit.normal, radiance, total_weight
    )


def _sample_direct_light_candidate[
    world_bvh_width: SIMDLength,
    instance_bvh_width: SIMDLength,
](
    world: CpuScene[world_bvh_width, instance_bvh_width],
    point: ShadingPoint[1],
    mut rng: Rng,
) -> _DirectLightSample[1]:
    """Sample one world-space light without tracing its visibility ray.

    Lights are selected proportionally to emitted power. Keeping this scalar
    geometric/visibility work separate lets the packet renderer evaluate BSDFs
    and MIS weights with SIMD math after collecting a batch of candidates.
    """
    var total_weight = world.scene_data().lights().total_weight
    if total_weight <= 0.0:
        return _empty_direct_light_sample[1]()

    ref lights = world.scene_data().lights()
    var draw = _draw_alias_column(rng.f32(), len(lights.records))
    var selected_idx = _resolve_alias_draw(
        draw,
        lights.alias_probabilities[draw.column],
        lights.alias_indices[draw.column],
    )
    ref light = lights.records[selected_idx]
    var emission = (
        world.scene_data()
        .surfaces()
        .emissives[Int(light.surface.index())]
        .radiance
    )
    var surface_sample: _LightSurfaceSample
    if light.primitive.kind() == PrimitiveKind.SPHERE:
        surface_sample = _sample_sphere_light_surface(
            light.p0,
            light.radius,
            random_unit_vector[.WORLD](rng),
        )
    else:
        surface_sample = _sample_triangle_light_surface(
            light.p0, light.p1, light.p2, rng.f32(), rng.f32()
        )
    var geometry = _finish_direct_light_geometry(
        point.p, point.normal, surface_sample, emission, total_weight
    )
    if not geometry.valid:
        return _empty_direct_light_sample[1]()
    return _DirectLightSample[1](
        True,
        geometry.direction,
        emission,
        geometry.surface_cosine,
        geometry.light_pdf,
        geometry.shadow_t_max,
    )


def _sample_direct_light[
    world_bvh_width: SIMDLength,
    instance_bvh_width: SIMDLength,
](
    world: CpuScene[world_bvh_width, instance_bvh_width],
    point: ShadingPoint[1],
    mut rng: Rng,
) -> _DirectLightSample[1]:
    """Sample one visible world-space light without evaluating the BSDF."""
    var light = _sample_direct_light_candidate(world, point, rng)
    if not light.valid:
        return light^
    var shadow_ray = spawn_surface_ray(
        point.p, light.direction, light.shadow_t_max
    )
    if world.occluded(shadow_ray):
        return _empty_direct_light_sample[1]()
    return light^


def sample_direct_lighting[
    integrator: Integrator,
    world_bvh_width: SIMDLength,
    instance_bvh_width: SIMDLength,
](
    surface: SurfaceId[1],
    world: CpuScene[world_bvh_width, instance_bvh_width],
    incoming_ray: Rayf32[.WORLD],
    point: ShadingPoint[1],
    mut rng: Rng,
) -> Color:
    """Sample direct illumination and apply the scalar reference BSDF."""
    comptime assert Integrator.uses_direct_lighting[integrator]
    var light = _sample_direct_light(world, point, rng)
    if not light.valid:
        return Color(0.0)
    var evaluation = evaluate_bsdf(
        surface,
        world.scene_data().surfaces(),
        incoming_ray,
        point,
        light.direction,
    )
    var scale = _direct_light_scale[integrator, 1](
        light.surface_cosine,
        light.light_pdf,
        evaluation.pdf,
        light.valid,
    )
    return evaluation.value * light.emission * scale
