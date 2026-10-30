"""Width-generic packet CPU wavefront kernels."""

from std.math import iota
from bajo.rt.continuation import finish_continuation, continuation_medium

from bajo.bvh.cpu.traversal_mode import CpuTraversalMode
from bajo.core import (
    Point3,
    Ray,
    Vec3,
    length as vector_length,
)
from bajo.core.color import Color
from bajo.rt.render_types import RenderSettings, SamplingConfig
from bajo.rt.material_types import (
    Environment,
    MaterialKind,
    Integrator,
    SurfaceId,
    SurfaceStore,
    SURFACE_INDEX_BITS,
)
from bajo.rt.shading_types import (
    BsdfSample,
    ShadingPoint,
)
from ..scene import CpuScene
from ..bsdf import _MaterialParameters
from bajo.rt.common import (
    environment_radiance,
    path_stage_rng,
    path_stage_seed,
)
from bajo.rt.lighting import _direct_light_scale
from bajo.rt.rays import make_ao_ray, spawn_surface_ray
from bajo.rt.shading import (
    _evaluate_coated_diffuse,
    _evaluate_material,
)
from bajo.rt.wavefront_queue import (
    PacketPathQueue,
    PacketShadeQueue,
    PathPacket,
    ShadePacket,
)
from bajo.rt.wavefront_contract import (
    wavefront_rng_light_stage,
    wavefront_rng_stage,
    wavefront_rng_material_stage,
)


from ..lighting import (
    _DirectLightSample,
    _environment_miss_weight,
    _sample_direct_light_candidate,
    _emissive_hit_weight,
)


struct _DirectLightBatch[length: SIMDLength]:
    var sample: _DirectLightSample[Self.length]
    var point: ShadingPoint[Self.length]
    var surface_kinds: SIMD[.uint32, Self.length]
    var surface_indices: SIMD[.uint32, Self.length]

    def __init__(out self):
        self.sample = _DirectLightSample[Self.length].make_empty()
        self.point = ShadingPoint[Self.length](
            Point3[.float32, .WORLD, Self.length](0.0),
            Vec3[.float32, .WORLD, Self.length](0.0),
            SIMD[.bool, Self.length](fill=False),
        )
        self.surface_kinds = 0
        self.surface_indices = 0


@always_inline
def _accumulate_direct_light_packet[
    integrator: Integrator, length: SIMDLength
](
    pixels: MutSpan[Color, _],
    paths: PathPacket[length],
    lights: _DirectLightBatch[length],
    lane_count: Int,
    surfaces: SurfaceStore,
    coated_fields: Span[Float32, _],
    samples_per_pixel: Int,
    sampling: SamplingConfig,
    stage: UInt32,
):
    """Evaluate collected NEE candidates with packet-wide BSDF and MIS math."""
    comptime assert Integrator.uses_direct_lighting[integrator]
    if not lights.sample.valid.reduce_or():
        return
    var ray_direction = Vec3[.float32, .WORLD, length](
        paths.dx, paths.dy, paths.dz
    )
    var material = _MaterialParameters[length]()
    var estimator_seeds = path_stage_seed(sampling, paths.path_ids, stage)
    var surface = SurfaceId[length](
        (lights.surface_kinds << SURFACE_INDEX_BITS) | lights.surface_indices
    )
    material.albedo = surfaces.sample_albedo(
        surface, lights.point.uv_u, lights.point.uv_v, lights.sample.valid
    )
    var coated_active = lights.sample.valid & lights.surface_kinds.eq(
        MaterialKind.COATED_DIFFUSE.value
    )
    material.gather[.METAL](
        surfaces,
        lights.surface_indices,
        lights.sample.valid & lights.surface_kinds.eq(MaterialKind.METAL.value),
    )
    material.load[.COATED_DIFFUSE, False](
        surfaces,
        lights.surface_indices,
        lights.point.uv_u,
        lights.point.uv_v,
        coated_active,
        coated_fields,
    )

    var lambertian = _evaluate_material[.LAMBERTIAN, length](
        ray_direction,
        lights.point.normal,
        material.albedo,
        material.parameter,
        lights.sample.direction,
    )
    var metal = _evaluate_material[.METAL, length](
        ray_direction,
        lights.point.normal,
        material.albedo,
        material.parameter,
        lights.sample.direction,
    )
    var coated = _evaluate_coated_diffuse(
        ray_direction,
        lights.point.normal,
        material.albedo,
        material.coated,
        lights.sample.direction,
        active=lights.sample.valid
        & lights.surface_kinds.eq(MaterialKind.COATED_DIFFUSE.value),
        estimator_seed=estimator_seeds,
    )
    var is_lambertian = lights.surface_kinds.eq(MaterialKind.LAMBERTIAN.value)
    var is_metal = lights.surface_kinds.eq(MaterialKind.METAL.value)
    var is_coated = lights.surface_kinds.eq(MaterialKind.COATED_DIFFUSE.value)
    var value = Vec3.select(
        is_coated,
        coated.value,
        Vec3.select(is_lambertian, lambertian.value, metal.value),
    )
    var pdf = is_coated.select(
        coated.pdf, is_lambertian.select(lambertian.pdf, metal.pdf)
    )
    var supported = is_lambertian | is_metal | is_coated
    var ok = lights.sample.valid & supported & pdf.gt(0.0)
    var scale = _direct_light_scale[integrator, length](
        lights.sample.surface_cosine,
        lights.sample.light_pdf,
        pdf,
        ok,
    )
    var red = paths.tx * value.x * lights.sample.emission.x * scale
    var green = paths.ty * value.y * lights.sample.emission.y * scale
    var blue = paths.tz * value.z * lights.sample.emission.z * scale
    for lane in range(lane_count):
        if ok[lane]:
            var pixel_idx = Int(paths.path_ids[lane]) / samples_per_pixel
            pixels[pixel_idx] += Color(red[lane], green[lane], blue[lane])


@always_inline
def _sample_bsdf_values[
    MATERIAL_KIND: MaterialKind, length: SIMDLength
](
    batch: ShadePacket[length],
    lane_count: Int,
    surfaces: SurfaceStore,
    coated_fields: Span[Float32, _],
    sampling: SamplingConfig,
    stage: UInt32,
) -> BsdfSample[length]:
    """Load and sample through the same setup used by scalar CPU shading."""
    var active = iota[.int, length]().lt(lane_count)
    var material = _MaterialParameters[length]()
    material.load[MATERIAL_KIND](
        surfaces,
        batch.surface_indices,
        batch.uv_u,
        batch.uv_v,
        active,
        coated_fields,
    )
    var rng = path_stage_rng(sampling, batch.path_ids, stage)
    return material.sample[MATERIAL_KIND](
        Vec3[.float32, .WORLD, length](batch.dx, batch.dy, batch.dz),
        Vec3[.float32, .WORLD, length](batch.nx, batch.ny, batch.nz),
        batch.front_faces,
        rng,
        path_stage_seed(sampling, batch.path_ids, stage),
        active,
    )


@always_inline
def _accumulate_environment_packet[
    length: SIMDLength,
    integrator: Integrator,
    world_bvh_width: SIMDLength,
    instance_bvh_width: SIMDLength,
](
    pixels: MutSpan[Color, _],
    packet: PathPacket[length],
    lane_count: Int,
    misses: SIMD[.bool, length],
    samples_per_pixel: Int,
    environment: Environment,
    surfaces: SurfaceStore,
    world: CpuScene[world_bvh_width, instance_bvh_width],
    bounce: Int,
):
    if not misses.reduce_or():
        return
    var emission = environment_radiance(
        environment,
        surfaces,
        Vec3[.float32, .WORLD, length](packet.dx, packet.dy, packet.dz),
    )
    var weight = _environment_miss_weight[integrator](
        world, emission, bounce, packet.bsdf_pdfs, packet.deltas
    )
    var red = packet.tx * emission.x * weight
    var green = packet.ty * emission.y * weight
    var blue = packet.tz * emission.z * weight
    for lane in range(lane_count):
        if misses[lane]:
            var pixel_idx = Int(packet.path_ids[lane]) / samples_per_pixel
            pixels[pixel_idx] += Color(red[lane], green[lane], blue[lane])


@always_inline
def _shade_material_packets[
    MATERIAL_KIND: MaterialKind,
    length: SIMDLength,
](
    mut next_paths: PacketPathQueue[length],
    queue: PacketShadeQueue[length],
    surfaces: SurfaceStore,
    coated_fields: Span[Float32, _],
    sampling: SamplingConfig,
    stage: UInt32,
):
    for packet_idx, batch in enumerate(queue.packets):
        var lane_count = min(length, len(queue) - packet_idx * length)
        var sampled = _sample_bsdf_values[MATERIAL_KIND, length](
            batch, lane_count, surfaces, coated_fields, sampling, stage
        )
        var media = continuation_medium(
            batch.medium_ids,
            batch.medium_boundaries,
            batch.front_faces,
            sampled.direction,
            Vec3[.float32, .WORLD, length](batch.gnx, batch.gny, batch.gnz),
        )
        var finished = finish_continuation[length](
            batch.path_ids,
            Point3[.float32, .WORLD, length](batch.ox, batch.oy, batch.oz),
            Vec3[.float32, .WORLD, length](batch.dx, batch.dy, batch.dz),
            batch.hit_t,
            Vec3[.float32, .WORLD, length](batch.tx, batch.ty, batch.tz),
            media,
            sampled,
            sampled.ok,
            sampling,
            stage,
        )
        next_paths.append_continuation(
            finished.values, finished.active, lane_count
        )


def _trace_packets[
    length: SIMDLength,
    integrator: Integrator,
    world_bvh_width: SIMDLength,
    instance_bvh_width: SIMDLength,
    traversal_mode: CpuTraversalMode,
    *adaptive_packet_sizes: SIMDLength,
](
    settings: RenderSettings,
    sampling: SamplingConfig,
    world: CpuScene[world_bvh_width, instance_bvh_width],
    pixels: MutSpan[Color, _],
    mut active_paths: PacketPathQueue[length],
    mut next_paths: PacketPathQueue[length],
    mut lambertian_queue: PacketShadeQueue[length],
    mut metal_queue: PacketShadeQueue[length],
    mut dielectric_queue: PacketShadeQueue[length],
    mut coated_diffuse_queue: PacketShadeQueue[length],
):
    var bounce_count = settings.max_depth if Integrator.is_path_tracing[
        integrator
    ] else 1
    for bounce in range(bounce_count):
        if len(active_paths) == 0:
            break
        lambertian_queue.clear()
        metal_queue.clear()
        dielectric_queue.clear()
        coated_diffuse_queue.clear()
        next_paths.clear()
        for packet_idx in range(len(active_paths.packets)):
            ref packet = active_paths.packets[packet_idx]
            var lane_count = min(
                length, len(active_paths) - packet_idx * length
            )
            var direct_lights = _DirectLightBatch[length]()
            var valid_lanes = iota[.int, length]().lt(lane_count)
            var ray_packet = Ray[.float32, .WORLD, length](
                Point3[.float32, .WORLD, length](
                    packet.ox, packet.oy, packet.oz
                ),
                Vec3[.float32, .WORLD, length](packet.dx, packet.dy, packet.dz),
                packet.t_min,
                packet.t_max,
            )
            var surface_hits = world.trace_surface_configured[
                length, traversal_mode, *adaptive_packet_sizes
            ](ray_packet, valid_lanes)
            comptime if Integrator.is_path_tracing[integrator]:
                var mixed = (
                    surface_hits.hit
                    & valid_lanes
                    & (surface_hits.surface.value >> SURFACE_INDEX_BITS).eq(
                        MaterialKind.MIX.value
                    )
                )
                if mixed.reduce_or():
                    var material_rng = path_stage_rng(
                        sampling,
                        packet.path_ids,
                        wavefront_rng_material_stage(UInt32(bounce)),
                    )
                    var mix_random = material_rng.f32()
                    for lane in range(lane_count):
                        if mixed[lane]:
                            surface_hits.surface.value[lane] = (
                                world.scene_data()
                                .surfaces()
                                .resolve_mix(
                                    surface_hits.surface.get(lane),
                                    mix_random[lane],
                                    surface_hits.uv_u[lane],
                                    surface_hits.uv_v[lane],
                                )
                                .value[0]
                            )
            surface_hits.normal = (
                world.scene_data()
                .surfaces()
                .shading_normal(
                    surface_hits.surface,
                    surface_hits.normal,
                    surface_hits.uv_u,
                    surface_hits.uv_v,
                    surface_hits.hit & valid_lanes,
                )
            )
            comptime if Integrator.is_path_tracing[integrator]:
                if (valid_lanes & packet.medium_ids.ne(0)).reduce_or():
                    var distance = surface_hits.hit.select(
                        surface_hits.t, packet.t_max
                    ) * vector_length(ray_packet.d)
                    var attenuation = (
                        world.scene_data()
                        .media()
                        .transmittance(packet.medium_ids, distance, valid_lanes)
                    )
                    packet.tx *= attenuation.x
                    packet.ty *= attenuation.y
                    packet.tz *= attenuation.z
                var misses = valid_lanes & ~surface_hits.hit
                var emitting = (
                    valid_lanes
                    & surface_hits.hit
                    & surface_hits.front_face
                    & (surface_hits.surface.value >> SURFACE_INDEX_BITS).eq(
                        MaterialKind.EMISSIVE.value
                    )
                )
                if emitting.reduce_or():
                    var emission = (
                        world.scene_data()
                        .surfaces()
                        .emitted_radiance(
                            surface_hits.surface,
                            surface_hits.front_face,
                            emitting,
                        )
                    )
                    var weight = _emissive_hit_weight[integrator](
                        world,
                        ray_packet,
                        surface_hits,
                        emission,
                        bounce,
                        packet.bsdf_pdfs,
                        packet.deltas,
                        emitting,
                    )
                    var red = packet.tx * emission.x * weight
                    var green = packet.ty * emission.y * weight
                    var blue = packet.tz * emission.z * weight
                    for lane in range(lane_count):
                        if emitting[lane]:
                            var pixel_idx = (
                                Int(packet.path_ids[lane])
                                / settings.samples_per_pixel
                            )
                            pixels[pixel_idx] += Color(
                                red[lane], green[lane], blue[lane]
                            )

                # Gather ray/throughput rows once, after per-lane attenuation;
                # append one compact tile per material, preserving lane order.
                var shade = ShadePacket[length]()
                shade.set_paths(packet)
                shade.set_hits(surface_hits)
                var valid_shade = surface_hits.hit & valid_lanes
                var kinds = surface_hits.surface.value >> SURFACE_INDEX_BITS
                lambertian_queue.append_packet_masked(
                    shade,
                    kinds.eq(MaterialKind.LAMBERTIAN.value) & valid_shade,
                    lane_count,
                )
                metal_queue.append_packet_masked(
                    shade,
                    kinds.eq(MaterialKind.METAL.value) & valid_shade,
                    lane_count,
                )
                dielectric_queue.append_packet_masked(
                    shade,
                    kinds.eq(MaterialKind.DIELECTRIC.value) & valid_shade,
                    lane_count,
                )
                coated_diffuse_queue.append_packet_masked(
                    shade,
                    kinds.eq(MaterialKind.COATED_DIFFUSE.value) & valid_shade,
                    lane_count,
                )

                comptime if Integrator.uses_direct_lighting[integrator]:
                    direct_lights.point = ShadingPoint[length].from_hit(
                        ray_packet, surface_hits
                    )
                    direct_lights.surface_kinds = kinds
                    direct_lights.surface_indices = (
                        surface_hits.surface.value
                        & ((UInt32(1) << SURFACE_INDEX_BITS) - UInt32(1))
                    )
                    var light_rng = path_stage_rng(
                        sampling,
                        packet.path_ids,
                        wavefront_rng_light_stage(UInt32(bounce)),
                    )
                    direct_lights.sample = _sample_direct_light_candidate(
                        world,
                        direct_lights.point,
                        light_rng,
                        valid_shade & kinds.ne(MaterialKind.EMISSIVE.value),
                    )
                    if (
                        direct_lights.sample.valid & packet.medium_ids.ne(0)
                    ).reduce_or():
                        var attenuation = (
                            world.scene_data()
                            .media()
                            .transmittance(
                                packet.medium_ids,
                                direct_lights.sample.shadow_t_max,
                                direct_lights.sample.valid,
                            )
                        )
                        direct_lights.sample.emission *= attenuation
                    var shadow_rays = spawn_surface_ray(
                        direct_lights.point.p,
                        direct_lights.sample.direction,
                        direct_lights.sample.shadow_t_max,
                    )
                    direct_lights.sample.valid &= ~world.occluded(
                        shadow_rays, direct_lights.sample.valid
                    )
                    _accumulate_direct_light_packet[integrator, length](
                        pixels,
                        packet,
                        direct_lights,
                        lane_count,
                        world.scene_data().surfaces(),
                        world.coated_materials(),
                        settings.samples_per_pixel,
                        sampling,
                        wavefront_rng_stage(UInt32(bounce)),
                    )
                _accumulate_environment_packet[
                    length,
                    integrator,
                    world_bvh_width,
                    instance_bvh_width,
                ](
                    pixels,
                    packet,
                    lane_count,
                    misses,
                    settings.samples_per_pixel,
                    world.scene_data().environment(),
                    world.scene_data().surfaces(),
                    world,
                    bounce,
                )
            else:
                var color = Vec3[.float32, .WORLD, length](0.0)
                comptime if integrator == .NORMALS:
                    color = Vec3.select(
                        surface_hits.hit,
                        (
                            surface_hits.normal
                            + Vec3[.float32, .WORLD, length](1.0)
                        )
                        * 0.5,
                        color,
                    )
                else:
                    comptime assert integrator == .AO
                    color = environment_radiance(
                        world.scene_data().environment(),
                        world.scene_data().surfaces(),
                        ray_packet.d,
                    )
                    var hits = surface_hits.hit & valid_lanes
                    if hits.reduce_or():
                        var rng = path_stage_rng(
                            sampling, packet.path_ids, UInt32(1)
                        )
                        var ao_ray = make_ao_ray(
                            ray_packet.at(surface_hits.t),
                            surface_hits.normal,
                            rng,
                        )
                        var blocked = world.occluded(ao_ray, hits)
                        var ao_value = blocked.select(Float32(0.08), 1.0)
                        color = Vec3.select(
                            hits,
                            Vec3[.float32, .WORLD, length](
                                ao_value, ao_value, ao_value
                            ),
                            color,
                        )
                for lane in range(lane_count):
                    var pixel_idx = (
                        Int(packet.path_ids[lane]) // settings.samples_per_pixel
                    )
                    pixels[pixel_idx] += Color(
                        color.x[lane], color.y[lane], color.z[lane]
                    )

        comptime if Integrator.is_path_tracing[integrator]:
            _shade_material_packets[.LAMBERTIAN, length](
                next_paths,
                lambertian_queue,
                world.scene_data().surfaces(),
                world.coated_materials(),
                sampling,
                UInt32(bounce + 1),
            )
            _shade_material_packets[.METAL, length](
                next_paths,
                metal_queue,
                world.scene_data().surfaces(),
                world.coated_materials(),
                sampling,
                UInt32(bounce + 1),
            )
            _shade_material_packets[.DIELECTRIC, length](
                next_paths,
                dielectric_queue,
                world.scene_data().surfaces(),
                world.coated_materials(),
                sampling,
                UInt32(bounce + 1),
            )
            _shade_material_packets[.COATED_DIFFUSE, length](
                next_paths,
                coated_diffuse_queue,
                world.scene_data().surfaces(),
                world.coated_materials(),
                sampling,
                UInt32(bounce + 1),
            )
            swap(active_paths, next_paths)
