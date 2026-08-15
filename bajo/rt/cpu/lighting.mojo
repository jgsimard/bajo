"""Emission and next-event estimation for CPU path integrators."""

from std.math import abs, pi, sqrt

from bajo.core import (
    Frame,
    Point3f32,
    Rayf32,
    Vec3f32,
    cross,
    dot,
    length2,
    normalize,
)
from bajo.core.random import Rng, random_unit_vector
from bajo.rt.types import (
    Color,
    MAT,
    RENDER,
    ShadingPoint,
    SurfaceId,
    SurfaceHit,
    SurfaceStore,
    World,
)
from .bsdf import evaluate_bsdf


def emitted_radiance(
    surface: SurfaceId,
    surfaces: SurfaceStore,
    front_face: Bool,
) -> Color:
    """Return one-sided surface emission."""
    if surface.kind() == MAT.EMISSIVE and front_face:
        return surfaces.emissives[Int(surface.index())].radiance
    return Color(0.0)


@always_inline
def power_heuristic(pdf_a: Float32, pdf_b: Float32) -> Float32:
    var a2 = pdf_a * pdf_a
    var b2 = pdf_b * pdf_b
    if a2 + b2 <= 0.0:
        return 0.0
    return a2 / (a2 + b2)


@always_inline
def _emissive_hit_weight[
    ALGORITHM: RENDER
](
    world: World,
    ray: Rayf32[Frame.WORLD],
    hit: SurfaceHit,
    bounce: Int,
    previous_bsdf_pdf: Float32,
    previous_delta: Bool,
) -> Float32:
    comptime assert ALGORITHM in (RENDER.PATH, RENDER.NEE, RENDER.MIS)
    comptime if ALGORITHM == RENDER.NEE:
        if bounce > 0 and not previous_delta:
            return 0.0
    elif ALGORITHM == RENDER.MIS:
        if bounce > 0 and not previous_delta:
            return power_heuristic(
                previous_bsdf_pdf,
                light_pdf_for_emissive_hit(world, ray, hit),
            )
    return 1.0


@always_inline
def _triangle_area(world: World, triangle_index: Int) -> Float32:
    ref v0 = world.triangle_vertices[3 * triangle_index + 0]
    ref v1 = world.triangle_vertices[3 * triangle_index + 1]
    ref v2 = world.triangle_vertices[3 * triangle_index + 2]
    return 0.5 * sqrt(length2(cross(v1 - v0, v2 - v0)))


@always_inline
def _emission_importance(radiance: Color) -> Float32:
    return max((radiance.x + radiance.y + radiance.z) / 3.0, 0.0)


def emissive_light_weight(world: World) -> Float32:
    """Total area-times-radiance weight of triangle and sphere emitters."""
    var total_weight = Float32(0.0)
    for idx in range(len(world.triangle_surfaces)):
        ref surface = world.triangle_surfaces[idx]
        if surface.kind() == MAT.EMISSIVE:
            var radiance = world.surfaces.emissives[
                Int(surface.index())
            ].radiance
            total_weight += _triangle_area(world, idx) * _emission_importance(
                radiance
            )
    for idx in range(len(world.sphere_surfaces)):
        ref surface = world.sphere_surfaces[idx]
        if surface.kind() == MAT.EMISSIVE:
            var radiance = world.surfaces.emissives[
                Int(surface.index())
            ].radiance
            var radius = abs(world.spheres[idx].radius)
            total_weight += (
                4.0 * pi * radius * radius * _emission_importance(radiance)
            )
    return total_weight


def light_pdf_for_emissive_hit(
    world: World,
    ray: Rayf32[Frame.WORLD],
    hit: SurfaceHit,
) -> Float32:
    """Evaluate the triangle-light distribution in solid-angle measure."""
    if hit.surface.kind() != MAT.EMISSIVE or not hit.front_face:
        return 0.0
    var total_weight = emissive_light_weight(world)
    if total_weight <= 0.0:
        return 0.0
    var light_cosine = max(dot(hit.normal, -normalize(ray.d)), 0.0)
    if light_cosine <= 0.0:
        return 0.0
    var distance_squared = hit.t * hit.t * length2(ray.d)
    var radiance = world.surfaces.emissives[Int(hit.surface.index())].radiance
    return (
        distance_squared
        * _emission_importance(radiance)
        / (light_cosine * total_weight)
    )


def sample_direct_lighting[
    ALGORITHM: RENDER
](
    surface: SurfaceId,
    world: World,
    incoming_ray: Rayf32[Frame.WORLD],
    point: ShadingPoint,
    mut rng: Rng,
) -> Color:
    """Sample one world-space emissive triangle for a Lambertian hit.

    Lights are selected proportionally to emitted power. The returned value already
    includes the Lambertian BSDF, geometry term, light-selection PDF, and
    binary visibility; callers only multiply it by path throughput.
    """
    comptime assert ALGORITHM in (RENDER.NEE, RENDER.MIS)
    var total_weight = emissive_light_weight(world)
    if total_weight <= 0.0:
        return Color(0.0)

    var selected_weight = rng.f32() * total_weight
    var light_point = Point3f32[Frame.WORLD](0.0)
    var light_normal = Vec3f32[Frame.WORLD](0.0, 1.0, 0.0)
    var emission = Color(0.0)
    var found = False
    for idx in range(len(world.triangle_surfaces)):
        ref candidate = world.triangle_surfaces[idx]
        if candidate.kind() == MAT.EMISSIVE:
            var radiance = world.surfaces.emissives[
                Int(candidate.index())
            ].radiance
            var weight = _triangle_area(world, idx) * _emission_importance(
                radiance
            )
            if selected_weight <= weight:
                ref v0 = world.triangle_vertices[3 * idx + 0]
                ref v1 = world.triangle_vertices[3 * idx + 1]
                ref v2 = world.triangle_vertices[3 * idx + 2]
                var edge1 = v1 - v0
                var edge2 = v2 - v0
                var area_vector = cross(edge1, edge2)
                var twice_area_squared = length2(area_vector)
                if twice_area_squared <= 0.0:
                    return Color(0.0)
                var twice_area = sqrt(twice_area_squared)
                light_normal = area_vector / twice_area
                var root_u = sqrt(rng.f32())
                var barycentric1 = root_u * (1.0 - rng.f32())
                var barycentric2 = root_u - barycentric1
                light_point = v0 + barycentric1 * edge1 + barycentric2 * edge2
                emission = radiance
                found = True
                break
            selected_weight -= weight

    if not found:
        for idx in range(len(world.sphere_surfaces)):
            ref candidate = world.sphere_surfaces[idx]
            if candidate.kind() == MAT.EMISSIVE:
                var radiance = world.surfaces.emissives[
                    Int(candidate.index())
                ].radiance
                var radius = abs(world.spheres[idx].radius)
                var weight = (
                    4.0 * pi * radius * radius * _emission_importance(radiance)
                )
                if selected_weight <= weight:
                    light_normal = random_unit_vector[Frame.WORLD](rng)
                    light_point = world.spheres[idx].center + (
                        radius * light_normal
                    )
                    emission = radiance
                    found = True
                    break
                selected_weight -= weight

    if not found:
        return Color(0.0)
    var to_light = light_point - point.p
    var distance_squared = length2(to_light)
    if distance_squared <= 1.0e-8:
        return Color(0.0)
    var distance = sqrt(distance_squared)
    var direction = to_light / distance
    var surface_cosine = max(dot(point.normal, direction), 0.0)
    var light_cosine = max(dot(light_normal, -direction), 0.0)
    if surface_cosine <= 0.0 or light_cosine <= 0.0:
        return Color(0.0)

    var shadow_t_max = distance - 0.002
    if shadow_t_max <= 0.001:
        return Color(0.0)
    var shadow_ray = Rayf32[Frame.WORLD](
        point.p, direction, 0.001, shadow_t_max
    )
    if world.occluded(shadow_ray):
        return Color(0.0)

    var light_pdf = (
        distance_squared
        * _emission_importance(emission)
        / (light_cosine * total_weight)
    )
    var evaluation = evaluate_bsdf(
        surface, world.surfaces, incoming_ray, point, direction
    )
    if evaluation.pdf <= 0.0:
        return Color(0.0)
    var estimator_weight = Float32(1.0)
    comptime if ALGORITHM == RENDER.MIS:
        estimator_weight = power_heuristic(light_pdf, evaluation.pdf)
    return (
        evaluation.value
        * emission
        * (surface_cosine * estimator_weight / light_pdf)
    )
