"""Width-generic packet CPU wavefront kernels."""

from std.math import sqrt
from bajo.rt.material_parameters import CoatedParameters
from bajo.rt.material_storage import (
    coated_column_tile,
    load_coated_parameters,
    load_material_rgb,
    CoatedDiffuseLayout,
)
from bajo.rt.continuation import finish_continuation, continuation_medium
from bajo.rt.media import medium_transmittance

from bajo.bvh.cpu.traversal_mode import CpuTraversalMode
from bajo.core import (
    Point3,
    Point3f32,
    Rayf32,
    Ray,
    Vec3,
    Vec3f32,
    dot,
)
from bajo.rt.render_types import (
    Color,
    RenderSettings,
    SamplingConfig,
)
from bajo.rt.material_types import (
    NO_TEXTURE,
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
from bajo.rt.common import (
    environment_radiance,
    path_stage_rng,
    path_stage_seed,
)
from bajo.rt.lighting import _direct_light_scale
from bajo.rt.rays import spawn_surface_ray
from bajo.rt.shading import (
    dielectric_geometry,
    _evaluate_coated_diffuse,
    _evaluate_material,
    _sample_coated_diffuse,
    _sample_material,
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


struct _MaterialParameters[length: SIMDLength]:
    var albedo: Vec3[.float32, .WORLD, Self.length]
    # Fuzz/roughness, or the dielectric refraction index.
    var parameter: SIMD[.float32, Self.length]
    var coated: CoatedParameters[Self.length]

    def __init__(out self):
        self.albedo = Vec3[.float32, .WORLD, Self.length](0.0)
        self.parameter = SIMD[.float32, Self.length](1.0)
        self.coated = CoatedParameters[Self.length]()

    @always_inline
    def gather_lane[
        kind: MaterialKind
    ](
        mut self,
        surfaces: SurfaceStore,
        coated_fields: Span[Float32, _],
        surface_index: UInt32,
        lane: Int,
    ):
        comptime match kind:
            case .METAL:
                ref material = surfaces.metals[Int(surface_index)]
                self.parameter[lane] = material.fuzz
            case .COATED_DIFFUSE:
                var records = coated_column_tile(
                    coated_fields.unsafe_ptr(), len(surfaces.coated_diffuses)
                )
                self.coated.set_lane(
                    lane, load_coated_parameters(records, surface_index)
                )
            case .DIELECTRIC:
                self.parameter[lane] = surfaces.dielectrics[
                    Int(surface_index)
                ].refraction_index
            case _:
                pass


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
    for lane in range(lane_count):
        if lights.sample.valid[lane]:
            var kind = MaterialKind(lights.surface_kinds[lane])
            comptime for material_kind in [
                MaterialKind.LAMBERTIAN,
                MaterialKind.METAL,
                MaterialKind.COATED_DIFFUSE,
            ]:
                if kind == material_kind:
                    material.gather_lane[material_kind](
                        surfaces,
                        coated_fields,
                        lights.surface_indices[lane],
                        lane,
                    )
    if coated_active.reduce_or():
        material.coated.conductor_roughness = (
            surfaces.sample_conductor_roughness(
                surface, lights.point.uv_u, lights.point.uv_v, coated_active
            )
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
    """Gather material/RNG state, then sample with BSDF math."""
    comptime assert MaterialKind.has_bsdf[MATERIAL_KIND]
    var ray_direction = Vec3[.float32, .WORLD, length](
        batch.dx, batch.dy, batch.dz
    )
    var normal = Vec3[.float32, .WORLD, length](batch.nx, batch.ny, batch.nz)
    var material = _MaterialParameters[length]()
    var estimator_seeds = path_stage_seed(sampling, batch.path_ids, stage)
    var random_u = SIMD[.float32, length](0.0)
    var random_v = SIMD[.float32, length](0.0)
    var active = SIMD[.bool, length](fill=False)
    var texture_albedo = True
    var texture_roughness = True
    for lane in range(lane_count):
        active[lane] = True
    comptime if MATERIAL_KIND == .COATED_DIFFUSE:
        if lane_count > 0:
            var records = coated_column_tile(
                coated_fields.unsafe_ptr(), len(surfaces.coated_diffuses)
            )
            var ids = active.select(
                batch.surface_indices, batch.surface_indices[0]
            )
            var uniform = ids.eq(ids[0]).reduce_and()
            material.coated = load_coated_parameters(records, ids, uniform)
            material.albedo = load_material_rgb[CoatedDiffuseLayout.ALBEDO](
                records, ids, uniform
            )
            if uniform:
                ref source = surfaces.coated_diffuses[Int(ids[0])]
                texture_albedo = source.texture_index != NO_TEXTURE
                texture_roughness = source.roughness_texture_index != NO_TEXTURE
    var surface = SurfaceId[length](
        (SIMD[.uint32, length](MATERIAL_KIND.value) << SURFACE_INDEX_BITS)
        | batch.surface_indices
    )
    comptime if MATERIAL_KIND != .DIELECTRIC:
        if texture_albedo or MATERIAL_KIND != .COATED_DIFFUSE:
            material.albedo = surfaces.sample_albedo(
                surface, batch.uv_u, batch.uv_v, active
            )
    comptime if MATERIAL_KIND == .COATED_DIFFUSE:
        if texture_roughness:
            material.coated.conductor_roughness = (
                surfaces.sample_conductor_roughness(
                    surface, batch.uv_u, batch.uv_v, active
                )
            )
    else:
        for lane in range(lane_count):
            material.gather_lane[MATERIAL_KIND](
                surfaces,
                coated_fields,
                batch.surface_indices[lane],
                lane,
            )
    var rng = path_stage_rng(sampling, batch.path_ids, stage)
    comptime if MATERIAL_KIND == .DIELECTRIC:
        var ri = batch.front_faces.select(
            Float32(1.0) / material.parameter, material.parameter
        )
        var geometry = dielectric_geometry(ray_direction, normal, ri)
        var eligible = active & geometry[2].le(1.0)
        if eligible.reduce_or():
            random_u = eligible.select(rng.f32(), 0.0)
    else:
        var eligible = active
        comptime if MATERIAL_KIND == .METAL:
            eligible &= material.parameter.gt(1.0e-4)
        if eligible.reduce_or():
            random_u = eligible.select(rng.f32(), 0.0)
            random_v = eligible.select(rng.f32(), 0.0)

    var sampled: BsdfSample[length]
    comptime if MATERIAL_KIND == .COATED_DIFFUSE:
        sampled = _sample_coated_diffuse(
            ray_direction,
            normal,
            material.albedo,
            material.coated,
            random_u,
            random_v,
            active=active,
            estimator_seed=estimator_seeds,
        )
    else:
        sampled = _sample_material[MATERIAL_KIND, length](
            ray_direction,
            normal,
            material.albedo,
            material.parameter,
            batch.front_faces,
            random_u,
            random_v,
        )

    sampled.ok &= active
    return sampled^


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
    var red = packet.tx * emission.x
    var green = packet.ty * emission.y
    var blue = packet.tz * emission.z
    for lane in range(lane_count):
        if misses[lane]:
            var pixel_idx = Int(packet.path_ids[lane]) / samples_per_pixel
            var weight = _environment_miss_weight[integrator](
                world,
                Vec3f32[.WORLD](
                    emission.x[lane], emission.y[lane], emission.z[lane]
                ),
                bounce,
                packet.bsdf_pdfs[lane],
                packet.deltas[lane],
            )
            pixels[pixel_idx] += (
                Color(red[lane], green[lane], blue[lane]) * weight
            )


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


def _trace_path_packets[
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
    comptime assert Integrator.is_path_tracing[integrator]
    for bounce in range(settings.max_depth):
        if len(active_paths) == 0:
            break
        lambertian_queue.clear()
        metal_queue.clear()
        dielectric_queue.clear()
        coated_diffuse_queue.clear()
        next_paths.clear()
        for packet_idx, source_packet in enumerate(active_paths.packets):
            var packet = source_packet.copy()
            var lane_count = min(
                length, len(active_paths) - packet_idx * length
            )
            var misses = SIMD[.bool, length](fill=False)
            var direct_lights = _DirectLightBatch[length]()
            var valid_lanes = SIMD[.bool, length](fill=False)
            for lane in range(lane_count):
                valid_lanes[lane] = True
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
            for lane in range(lane_count):
                var ray = Rayf32[.WORLD](
                    Point3f32[.WORLD](
                        packet.ox[lane], packet.oy[lane], packet.oz[lane]
                    ),
                    Vec3f32[.WORLD](
                        packet.dx[lane], packet.dy[lane], packet.dz[lane]
                    ),
                    packet.t_min[lane],
                    packet.t_max[lane],
                )
                var hit = surface_hits.get(lane)
                var medium_id = Int(packet.medium_ids[lane])
                if medium_id != 0:
                    var distance = (hit.t if hit.hit else ray.t_max) * sqrt(
                        dot(ray.d, ray.d)
                    )
                    var attenuation = medium_transmittance(
                        world.scene_data().media().absorption[medium_id],
                        distance,
                    )
                    packet.tx[lane] *= attenuation.x
                    packet.ty[lane] *= attenuation.y
                    packet.tz[lane] *= attenuation.z
                if hit.hit:
                    var pixel_idx = (
                        Int(packet.path_ids[lane]) / settings.samples_per_pixel
                    )
                    if hit.surface.kind() == .EMISSIVE:
                        var emission_weight = _emissive_hit_weight[integrator](
                            world,
                            ray,
                            hit,
                            bounce,
                            packet.bsdf_pdfs[lane],
                            packet.deltas[lane],
                        )
                        pixels[pixel_idx] += (
                            Color(
                                packet.tx[lane],
                                packet.ty[lane],
                                packet.tz[lane],
                            )
                            * world.scene_data()
                            .surfaces()
                            .emitted_radiance(
                                hit.surface,
                                hit.front_face,
                            )
                            * emission_weight
                        )
                        continue

                else:
                    misses[lane] = True

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
                direct_lights.surface_indices = surface_hits.surface.value & (
                    (UInt32(1) << SURFACE_INDEX_BITS) - UInt32(1)
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
                    for lane in range(lane_count):
                        if (
                            direct_lights.sample.valid[lane]
                            and packet.medium_ids[lane] != 0
                        ):
                            var attenuation = medium_transmittance(
                                world.scene_data()
                                .media()
                                .absorption[Int(packet.medium_ids[lane])],
                                direct_lights.sample.shadow_t_max[lane],
                            )
                            direct_lights.sample.emission.x[
                                lane
                            ] *= attenuation.x
                            direct_lights.sample.emission.y[
                                lane
                            ] *= attenuation.y
                            direct_lights.sample.emission.z[
                                lane
                            ] *= attenuation.z
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
