"""Emission and next-event estimation for CPU path integrators."""

from std.memory import Pointer
from std.sys.info import size_of
from bajo.core.tensor_layout import record_tile, gather_record_vec3
from bajo.rt.lighting_types import LightRecord
from bajo.core import Point3, Ray, Vec3, dot, normalize
from bajo.core.random import Rng, unit_vector_from_samples
from bajo.bvh.constants import f32_max, PrimitiveKind
from bajo.rt.common import (
    environment_light_pdf,
    environment_pdf_from_radiance,
    environment_radiance,
)
from bajo.core.mappings import equal_area_square_to_sphere
from bajo.core.sampling import draw_alias_column, resolve_alias_draw
from bajo.rt.lighting import (
    _emissive_hit_light_pdf,
    _emissive_hit_weight_from_pdf,
    _finish_direct_light_geometry,
    _sample_sphere_light_surface,
    _sample_triangle_light_surface,
)

from bajo.rt.material_types import (
    Integrator,
    MaterialKind,
    Emissive,
    PRIMITIVE_INDEX_BITS,
    SURFACE_INDEX_MASK,
    SURFACE_INDEX_BITS,
)
from bajo.rt.shading_types import ShadingPoint, SurfaceHit
from .scene import CpuScene


@fieldwise_init
struct _DirectLightSample[length: SIMDLength = 1](Copyable):
    """Visible light sample before applying the surface BSDF."""

    var valid: SIMD[.bool, Self.length]
    var direction: Vec3[.float32, .WORLD, Self.length]
    var emission: Vec3[.float32, .WORLD, Self.length]
    var surface_cosine: SIMD[.float32, Self.length]
    var light_pdf: SIMD[.float32, Self.length]
    var shadow_t_max: SIMD[.float32, Self.length]

    @staticmethod
    @always_inline
    def make_empty() -> Self:
        return Self(
            SIMD[.bool, Self.length](fill=False),
            Vec3[.float32, .WORLD, Self.length](0.0),
            Vec3[.float32, .WORLD, Self.length](0.0),
            0.0,
            0.0,
            0.0,
        )


@always_inline
def _emissive_hit_weight[
    integrator: Integrator,
    world_bvh_width: SIMDLength,
    instance_bvh_width: SIMDLength,
    length: SIMDLength = 1,
](
    world: CpuScene[world_bvh_width, instance_bvh_width],
    ray: Ray[.float32, .WORLD, length],
    hit: SurfaceHit[length],
    emission: Vec3[.float32, .WORLD, length],
    bounce: Int,
    previous_bsdf_pdf: SIMD[.float32, length],
    previous_delta: SIMD[.bool, length],
    active: SIMD[.bool, length] = SIMD[.bool, length](fill=True),
) -> SIMD[.float32, length]:
    var light_pdf = SIMD[.float32, length](0.0)
    comptime if integrator == .MIS:
        if bounce > 0 and (active & ~previous_delta).reduce_or():
            light_pdf = _emissive_hit_light_pdf(
                ray.d,
                hit.t,
                hit.normal,
                emission,
                world.scene_data().total_light_weight(),
            )
    return _emissive_hit_weight_from_pdf[integrator](
        UInt32(bounce), previous_delta, previous_bsdf_pdf, light_pdf
    )


@always_inline
def light_pdf_for_emissive_hit[
    world_bvh_width: SIMDLength,
    instance_bvh_width: SIMDLength,
    length: SIMDLength = 1,
](
    world: CpuScene[world_bvh_width, instance_bvh_width],
    ray: Ray[.float32, .WORLD, length],
    hit: SurfaceHit[length],
    active: SIMD[.bool, length] = SIMD[.bool, length](fill=True),
) -> SIMD[.float32, length]:
    ref data = world.scene_data()
    var emitting = (
        active
        & hit.front_face
        & (hit.surface.value >> SURFACE_INDEX_BITS).eq(
            MaterialKind.EMISSIVE.value
        )
    )
    if not emitting.reduce_or():
        return 0.0
    var radiance = data.surfaces().emitted_radiance(
        hit.surface, hit.front_face, emitting
    )
    return emitting.select(
        _emissive_hit_light_pdf(
            ray.d, hit.t, hit.normal, radiance, data.total_light_weight()
        ),
        0.0,
    )


def environment_pdf_for_miss[
    world_bvh_width: SIMDLength,
    instance_bvh_width: SIMDLength,
](
    world: CpuScene[world_bvh_width, instance_bvh_width],
    direction: Vec3[.float32, .WORLD, 1],
) -> Float32:
    ref data = world.scene_data()
    return environment_light_pdf(
        data.environment(),
        data.surfaces(),
        data.environment_weight(),
        data.total_light_weight(),
        direction,
    )[0]


@always_inline
def _environment_miss_weight[
    integrator: Integrator,
    world_bvh_width: SIMDLength,
    instance_bvh_width: SIMDLength,
    length: SIMDLength = 1,
](
    world: CpuScene[world_bvh_width, instance_bvh_width],
    emission: Vec3[.float32, .WORLD, length],
    bounce: Int,
    previous_bsdf_pdf: SIMD[.float32, length],
    previous_delta: SIMD[.bool, length],
) -> SIMD[.float32, length]:
    var light_pdf = SIMD[.float32, length](0.0)
    comptime if integrator == .MIS:
        if bounce > 0 and (~previous_delta).reduce_or():
            ref data = world.scene_data()
            light_pdf = environment_pdf_from_radiance(
                data.environment(),
                data.environment_weight(),
                data.total_light_weight(),
                emission,
            )
    return _emissive_hit_weight_from_pdf[integrator](
        UInt32(bounce), previous_delta, previous_bsdf_pdf, light_pdf
    )


def _sample_environment_candidate[
    world_bvh_width: SIMDLength,
    instance_bvh_width: SIMDLength,
    length: SIMDLength = 1,
](
    world: CpuScene[world_bvh_width, instance_bvh_width],
    point: ShadingPoint[length],
    random_u: SIMD[.float32, length],
    random_v: SIMD[.float32, length],
    jitter_v: SIMD[.float32, length],
) -> _DirectLightSample[length]:
    ref data = world.scene_data()
    var local_direction: Vec3[.float32, .LOCAL, length]
    if len(data.environment_alias_probabilities()) > 0:
        var draw = draw_alias_column(
            random_u, len(data.environment_alias_probabilities())
        )
        var pixel_idx = resolve_alias_draw(
            draw,
            data.environment_alias_probabilities()
            .unsafe_ptr()
            .unsafe_gather(draw.column),
            data.environment_alias_indices()
            .unsafe_ptr()
            .unsafe_gather(draw.column),
        )
        ref texture = data.surfaces().image_textures[
            Int(data.environment().texture_index)
        ]
        var pixel_x = pixel_idx % SIMD[.int64, length](texture.width)
        var pixel_y = pixel_idx / SIMD[.int64, length](texture.width)
        var u = (pixel_x.cast[.float32]() + random_v) / Float32(texture.width)
        var v = (pixel_y.cast[.float32]() + jitter_v) / Float32(texture.height)
        local_direction = equal_area_square_to_sphere[.LOCAL, length](u, v)
    else:
        local_direction = unit_vector_from_samples[.LOCAL, length](
            random_u, random_v
        )
    ref transform = data.environment().light_to_world
    var direction = normalize(
        Vec3[.float32, .WORLD, length](
            transform.m00 * local_direction.x
            + transform.m01 * local_direction.y
            + transform.m02 * local_direction.z,
            transform.m10 * local_direction.x
            + transform.m11 * local_direction.y
            + transform.m12 * local_direction.z,
            transform.m20 * local_direction.x
            + transform.m21 * local_direction.y
            + transform.m22 * local_direction.z,
        )
    )
    var surface_cosine = max(dot(point.normal, direction), 0.0)
    var emission = environment_radiance(
        data.environment(), data.surfaces(), direction
    )
    var light_pdf = environment_pdf_from_radiance(
        data.environment(),
        data.environment_weight(),
        data.total_light_weight(),
        emission,
    )
    return _DirectLightSample[length](
        surface_cosine.gt(0.0) & light_pdf.gt(0.0),
        direction,
        emission,
        surface_cosine,
        light_pdf,
        f32_max,
    )


def _sample_direct_light_candidate[
    world_bvh_width: SIMDLength,
    instance_bvh_width: SIMDLength,
    length: SIMDLength = 1,
](
    world: CpuScene[world_bvh_width, instance_bvh_width],
    point: ShadingPoint[length],
    mut rng: Rng[length],
    active: SIMD[.bool, length] = SIMD[.bool, length](fill=True),
) -> _DirectLightSample[length]:
    """Select and sample emitters across a packet; scalar is width one."""
    ref data = world.scene_data()
    var total_weight = data.total_light_weight()
    if total_weight <= 0.0 or not active.reduce_or():
        return _DirectLightSample[length].make_empty()
    ref lights = data.lights()
    var selection_u = rng.f32()
    var environment_probability = data.environment_weight() / total_weight
    var environment_lanes = active & selection_u.lt(environment_probability)
    var result = _DirectLightSample[length].make_empty()
    var u = rng.f32()
    var v = rng.f32()
    var jitter_v = SIMD[.float32, length](0.0)
    if environment_lanes.reduce_or():
        if len(data.environment_alias_probabilities()) > 0:
            jitter_v = rng.f32()
        result = _sample_environment_candidate(world, point, u, v, jitter_v)
        result.valid &= environment_lanes
    var finite_lanes = active & ~environment_lanes
    if len(lights.records) == 0 or not finite_lanes.reduce_or():
        return result^
    var finite_probability = 1.0 - environment_probability
    var finite_u = (
        selection_u - environment_probability
    ) / finite_probability if finite_probability > 0.0 else SIMD[
        .float32, length
    ](
        0.0
    )
    var draw = draw_alias_column(
        finite_lanes.select(finite_u.clamp(0.0, 0.99999994), 0.0),
        len(lights.records),
    )
    var selected = resolve_alias_draw(
        draw,
        lights.alias_probabilities.unsafe_ptr().unsafe_gather(draw.column),
        lights.alias_indices.unsafe_ptr().unsafe_gather(draw.column),
    )
    # Read each field across records, using the field's actual address and
    # the ABI record stride. Inactive lanes always reference record zero.
    comptime stride = size_of[LightRecord]() // size_of[Float32]()
    var offsets = finite_lanes.select(selected, 0).cast[.int64]() * SIMD[
        .int64, length
    ](stride)
    var p0_fields = Pointer(to=lights.records[0].p0.x)
    var p1_fields = Pointer(to=lights.records[0].p1.x)
    var p2_fields = Pointer(to=lights.records[0].p2.x)
    var p0 = Point3[.float32, .WORLD, length](
        p0_fields.unsafe_gather(offsets),
        p0_fields.unsafe_gather(offsets + 1),
        p0_fields.unsafe_gather(offsets + 2),
    )
    var p1 = Point3[.float32, .WORLD, length](
        p1_fields.unsafe_gather(offsets),
        p1_fields.unsafe_gather(offsets + 1),
        p1_fields.unsafe_gather(offsets + 2),
    )
    var p2 = Point3[.float32, .WORLD, length](
        p2_fields.unsafe_gather(offsets),
        p2_fields.unsafe_gather(offsets + 1),
        p2_fields.unsafe_gather(offsets + 2),
    )
    var radius = Pointer(to=lights.records[0].radius).unsafe_gather(offsets)
    var primitives = Pointer(
        to=lights.records[0].primitive.value
    ).unsafe_gather(offsets)
    var spheres = (primitives >> PRIMITIVE_INDEX_BITS).eq(
        PrimitiveKind.SPHERE.value
    ) & finite_lanes
    var surfaces = Pointer(to=lights.records[0].surface.value).unsafe_gather(
        offsets
    )
    comptime assert size_of[Emissive]() == 3 * size_of[Float32]()
    var radiance = record_tile[.float32, 3](
        Pointer(to=data.surfaces().emissives[0].radiance.x),
        len(data.surfaces().emissives),
    )
    var emission = gather_record_vec3(radiance, surfaces & SURFACE_INDEX_MASK)
    # Finite emitters and uniform environments share the two draws; image
    # environments use u for alias selection, v and jitter_v for pixel jitter.
    var surface_sample = _sample_triangle_light_surface(p0, p1, p2, u, v)
    if spheres.reduce_or():
        var normal = unit_vector_from_samples[.WORLD, length](u, v)
        var sphere = _sample_sphere_light_surface(p0, radius, normal)
        surface_sample.point = Point3.select(
            spheres, sphere.point, surface_sample.point
        )
        surface_sample.normal = Vec3.select(
            spheres, sphere.normal, surface_sample.normal
        )
        surface_sample.valid |= spheres
    var geometry = _finish_direct_light_geometry(
        point.p, point.normal, surface_sample, emission, total_weight
    )
    result.valid |= finite_lanes & geometry.valid
    result.direction = Vec3.select(
        finite_lanes, geometry.direction, result.direction
    )
    result.emission = Vec3.select(finite_lanes, emission, result.emission)
    result.surface_cosine = finite_lanes.select(
        geometry.surface_cosine, result.surface_cosine
    )
    result.light_pdf = finite_lanes.select(geometry.light_pdf, result.light_pdf)
    result.shadow_t_max = finite_lanes.select(
        geometry.shadow_t_max, result.shadow_t_max
    )
    return result^
