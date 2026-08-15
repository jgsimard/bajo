"""Emission and next-event estimation for CPU path integrators."""

from std.math import abs, sqrt

from bajo.core import (
    Frame,
    Point3f32,
    Rayf32,
    Vec3,
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
    PRIM,
    RENDER,
    ShadingPoint,
    SurfaceId,
    SurfaceHit,
    SurfaceStore,
    World,
    _light_importance,
)
from .bsdf import evaluate_bsdf


@fieldwise_init
struct _DirectLightSample[length: SIMDLength = 1](Copyable):
    """Visible light sample before applying the surface BSDF."""

    var valid: SIMD[DType.bool, Self.length]
    var direction: Vec3[DType.float32, Frame.WORLD, Self.length]
    var emission: Vec3[DType.float32, Frame.WORLD, Self.length]
    var surface_cosine: SIMD[DType.float32, Self.length]
    var light_pdf: SIMD[DType.float32, Self.length]
    var shadow_t_max: SIMD[DType.float32, Self.length]


@always_inline
def _empty_direct_light_sample[
    length: SIMDLength = 1
]() -> _DirectLightSample[length]:
    return _DirectLightSample[length](
        SIMD[DType.bool, length](fill=False),
        Vec3[DType.float32, Frame.WORLD, length](0.0),
        Vec3[DType.float32, Frame.WORLD, length](0.0),
        0.0,
        0.0,
        0.0,
    )


def emitted_radiance(
    surface: SurfaceId[1],
    surfaces: SurfaceStore,
    front_face: Bool,
) -> Color:
    """Return one-sided surface emission."""
    if surface.kind() == MAT.EMISSIVE and front_face:
        return surfaces.emissives[Int(surface.index())].radiance
    return Color(0.0)


@always_inline
def power_heuristic[
    length: SIMDLength = 1
](
    pdf_a: SIMD[DType.float32, length],
    pdf_b: SIMD[DType.float32, length],
) -> SIMD[DType.float32, length]:
    var a2 = pdf_a * pdf_a
    var b2 = pdf_b * pdf_b
    var denominator = a2 + b2
    var nonzero = denominator.gt(0.0)
    return nonzero.select(
        a2 / nonzero.select(denominator, Float32(1.0)), Float32(0.0)
    )


@always_inline
def _direct_light_scale[
    ALGORITHM: RENDER, length: SIMDLength
](
    surface_cosine: SIMD[DType.float32, length],
    light_pdf: SIMD[DType.float32, length],
    bsdf_pdf: SIMD[DType.float32, length],
    valid: SIMD[DType.bool, length],
) -> SIMD[DType.float32, length]:
    comptime assert ALGORITHM in (RENDER.NEE, RENDER.MIS)
    var ok = valid & light_pdf.gt(0.0) & bsdf_pdf.gt(0.0)
    var safe_light_pdf = ok.select(light_pdf, Float32(1.0))
    var estimator_weight = SIMD[DType.float32, length](1.0)
    comptime if ALGORITHM == RENDER.MIS:
        estimator_weight = power_heuristic(light_pdf, bsdf_pdf)
    return ok.select(
        surface_cosine * estimator_weight / safe_light_pdf, Float32(0.0)
    )


@always_inline
def _emissive_hit_weight[
    ALGORITHM: RENDER,
    world_bvh_width: SIMDLength,
    instance_bvh_width: SIMDLength,
](
    world: World[world_bvh_width, instance_bvh_width],
    ray: Rayf32[Frame.WORLD],
    hit: SurfaceHit[1],
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
            return power_heuristic[1](
                previous_bsdf_pdf,
                light_pdf_for_emissive_hit(world, ray, hit),
            )
    return 1.0


def light_pdf_for_emissive_hit[
    world_bvh_width: SIMDLength,
    instance_bvh_width: SIMDLength,
](
    world: World[world_bvh_width, instance_bvh_width],
    ray: Rayf32[Frame.WORLD],
    hit: SurfaceHit[1],
) -> Float32:
    """Evaluate the triangle-light distribution in solid-angle measure."""
    if hit.surface.kind() != MAT.EMISSIVE or not hit.front_face:
        return 0.0
    var total_weight = world.lights.total_weight
    if total_weight <= 0.0:
        return 0.0
    var light_cosine = max(dot(hit.normal, -normalize(ray.d)), 0.0)
    if light_cosine <= 0.0:
        return 0.0
    var distance_squared = hit.t * hit.t * length2(ray.d)
    var radiance = world.surfaces.emissives[Int(hit.surface.index())].radiance
    return (
        distance_squared
        * _light_importance(radiance)
        / (light_cosine * total_weight)
    )


def _sample_direct_light_candidate[
    world_bvh_width: SIMDLength,
    instance_bvh_width: SIMDLength,
](
    world: World[world_bvh_width, instance_bvh_width],
    point: ShadingPoint[1],
    mut rng: Rng,
) -> _DirectLightSample[1]:
    """Sample one world-space light without tracing its visibility ray.

    Lights are selected proportionally to emitted power. Keeping this scalar
    geometric/visibility work separate lets the packet renderer evaluate BSDFs
    and MIS weights with SIMD math after collecting a batch of candidates.
    """
    var total_weight = world.lights.total_weight
    if total_weight <= 0.0:
        return _empty_direct_light_sample[1]()

    var selected_weight = rng.f32() * total_weight
    var light_point = Point3f32[Frame.WORLD](0.0)
    var light_normal = Vec3f32[Frame.WORLD](0.0, 1.0, 0.0)
    var emission = Color(0.0)
    var found = False
    for light in world.lights.records:
        if selected_weight <= light.weight:
            var primitive_kind = light.primitive.kind()
            var idx = Int(light.primitive.index())
            emission = world.surfaces.emissives[
                Int(light.surface.index())
            ].radiance
            if primitive_kind == PRIM.TRIANGLE:
                ref v0 = world.triangle_vertices[3 * idx + 0]
                ref v1 = world.triangle_vertices[3 * idx + 1]
                ref v2 = world.triangle_vertices[3 * idx + 2]
                var edge1 = v1 - v0
                var edge2 = v2 - v0
                var area_vector = cross(edge1, edge2)
                var twice_area_squared = length2(area_vector)
                if twice_area_squared <= 0.0:
                    return _empty_direct_light_sample[1]()
                var twice_area = sqrt(twice_area_squared)
                light_normal = area_vector / twice_area
                var root_u = sqrt(rng.f32())
                var barycentric1 = root_u * (1.0 - rng.f32())
                var barycentric2 = root_u - barycentric1
                light_point = v0 + barycentric1 * edge1 + barycentric2 * edge2
                found = True
            elif primitive_kind == PRIM.SPHERE:
                var radius = abs(world.spheres[idx].radius)
                light_normal = random_unit_vector[Frame.WORLD](rng)
                light_point = world.spheres[idx].center + radius * light_normal
                found = True
            break
        selected_weight -= light.weight

    if not found:
        return _empty_direct_light_sample[1]()
    var to_light = light_point - point.p
    var distance_squared = length2(to_light)
    if distance_squared <= 1.0e-8:
        return _empty_direct_light_sample[1]()
    var distance = sqrt(distance_squared)
    var direction = to_light / distance
    var surface_cosine = max(dot(point.normal, direction), 0.0)
    var light_cosine = max(dot(light_normal, -direction), 0.0)
    if surface_cosine <= 0.0 or light_cosine <= 0.0:
        return _empty_direct_light_sample[1]()

    var shadow_t_max = distance - 0.002
    if shadow_t_max <= 0.001:
        return _empty_direct_light_sample[1]()
    var light_pdf = (
        distance_squared
        * _light_importance(emission)
        / (light_cosine * total_weight)
    )
    return _DirectLightSample[1](
        True,
        direction,
        emission,
        surface_cosine,
        light_pdf,
        shadow_t_max,
    )


def _sample_direct_light[
    world_bvh_width: SIMDLength,
    instance_bvh_width: SIMDLength,
](
    world: World[world_bvh_width, instance_bvh_width],
    point: ShadingPoint[1],
    mut rng: Rng,
) -> _DirectLightSample[1]:
    """Sample one visible world-space light without evaluating the BSDF."""
    var light = _sample_direct_light_candidate(world, point, rng)
    if not light.valid:
        return light^
    var shadow_ray = Rayf32[Frame.WORLD](
        point.p, light.direction, 0.001, light.shadow_t_max
    )
    if world.occluded(shadow_ray):
        return _empty_direct_light_sample[1]()
    return light^


def sample_direct_lighting[
    ALGORITHM: RENDER,
    world_bvh_width: SIMDLength,
    instance_bvh_width: SIMDLength,
](
    surface: SurfaceId[1],
    world: World[world_bvh_width, instance_bvh_width],
    incoming_ray: Rayf32[Frame.WORLD],
    point: ShadingPoint[1],
    mut rng: Rng,
) -> Color:
    """Sample direct illumination and apply the scalar reference BSDF."""
    comptime assert ALGORITHM in (RENDER.NEE, RENDER.MIS)
    var light = _sample_direct_light(world, point, rng)
    if not light.valid:
        return Color(0.0)
    var evaluation = evaluate_bsdf(
        surface, world.surfaces, incoming_ray, point, light.direction
    )
    var scale = _direct_light_scale[ALGORITHM, 1](
        light.surface_cosine,
        light.light_pdf,
        evaluation.pdf,
        light.valid,
    )
    return evaluation.value * light.emission * scale
