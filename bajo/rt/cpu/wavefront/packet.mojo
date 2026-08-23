"""Width-generic packet CPU wavefront kernels."""

from std.math import sqrt

from bajo.bvh.constants import f32_max
from bajo.core import (
    Frame,
    Point3,
    Point3f32,
    Rayf32,
    Ray,
    Vec3,
    Vec3f32,
    dot,
    normalize,
)
from bajo.rt.types import (
    Color,
    MAT,
    RENDER,
    RenderSettings,
    ShadingPoint,
    SurfaceStore,
)
from ..scene import CpuScene
from bajo.rt.common import path_stage_rng, russian_roulette, sky_color
from bajo.rt.wavefront_queue import (
    PacketPathQueue,
    PacketShadeQueue,
    PathPacket,
    ShadePacket,
)
from bajo.rt.wavefront_contract import wavefront_rng_light_stage


from ..bsdf import (
    _evaluate_material,
    _sample_material,
)
from ..lighting import (
    _DirectLightSample,
    _direct_light_scale,
    _empty_direct_light_sample,
    _sample_direct_light_candidate,
    _emissive_hit_weight,
    emitted_radiance,
)


struct ScatterBatch[length: SIMDLength]:
    var paths: PathPacket[Self.length]
    var ok: SIMD[DType.bool, Self.length]

    def __init__(
        out self,
        var paths: PathPacket[Self.length],
        ok: SIMD[DType.bool, Self.length],
    ):
        self.paths = paths^
        self.ok = ok


struct _DirectLightBatch[length: SIMDLength]:
    var sample: _DirectLightSample[Self.length]
    var point: ShadingPoint[Self.length]
    var surface_kinds: SIMD[DType.uint32, Self.length]
    var surface_indices: SIMD[DType.uint32, Self.length]

    def __init__(out self):
        self.sample = _empty_direct_light_sample[Self.length]()
        self.point = ShadingPoint[Self.length](
            Point3[DType.float32, .WORLD, Self.length](0.0),
            Vec3[.float32, .WORLD, Self.length](0.0),
            SIMD[DType.bool, Self.length](fill=False),
        )
        self.surface_kinds = 0
        self.surface_indices = 0


@always_inline
def _accumulate_direct_light_packet[
    ALGORITHM: RENDER, length: SIMDLength
](
    pixels: MutSpan[Color, _],
    paths: PathPacket[length],
    lights: _DirectLightBatch[length],
    lane_count: Int,
    surfaces: SurfaceStore,
    samples_per_pixel: Int,
):
    """Evaluate collected NEE candidates with packet-wide BSDF and MIS math."""
    comptime assert ALGORITHM in (RENDER.NEE, RENDER.MIS)
    var ray_direction = Vec3[.float32, .WORLD, length](
        paths.dx, paths.dy, paths.dz
    )
    var albedo = Vec3[.float32, .WORLD, length](0.0)
    var fuzz = SIMD[DType.float32, length](1.0)
    for lane in range(lane_count):
        if not lights.sample.valid[lane]:
            continue
        if lights.surface_kinds[lane] == MAT.LAMBERTIAN.v:
            ref material = surfaces.lambertians[
                Int(lights.surface_indices[lane])
            ]
            albedo.x[lane] = material.albedo.x
            albedo.y[lane] = material.albedo.y
            albedo.z[lane] = material.albedo.z
        elif lights.surface_kinds[lane] == MAT.METAL.v:
            ref material = surfaces.metals[Int(lights.surface_indices[lane])]
            albedo.x[lane] = material.albedo.x
            albedo.y[lane] = material.albedo.y
            albedo.z[lane] = material.albedo.z
            fuzz[lane] = material.fuzz

    var lambertian = _evaluate_material[MAT.LAMBERTIAN, length](
        ray_direction,
        lights.point.normal,
        albedo,
        fuzz,
        lights.sample.direction,
    )
    var metal = _evaluate_material[MAT.METAL, length](
        ray_direction,
        lights.point.normal,
        albedo,
        fuzz,
        lights.sample.direction,
    )
    var is_lambertian = lights.surface_kinds.eq(MAT.LAMBERTIAN.v)
    var is_metal = lights.surface_kinds.eq(MAT.METAL.v)
    var value = Vec3.select(is_lambertian, lambertian.value, metal.value)
    var pdf = is_lambertian.select(lambertian.pdf, metal.pdf)
    var supported = is_lambertian | is_metal
    var ok = lights.sample.valid & supported & pdf.gt(0.0)
    var scale = _direct_light_scale[ALGORITHM, length](
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


def _sample_bsdf_batch[
    MATERIAL_KIND: MAT, length: SIMDLength
](
    batch: ShadePacket[length],
    lane_count: Int,
    surfaces: SurfaceStore,
    settings: RenderSettings,
    stage: UInt32,
) -> ScatterBatch[length]:
    """Gather material/RNG state, then sample with BSDF math."""
    comptime assert MATERIAL_KIND in (
        MAT.LAMBERTIAN,
        MAT.METAL,
        MAT.DIELECTRIC,
    )
    var ray_direction = Vec3[.float32, .WORLD, length](
        batch.dx, batch.dy, batch.dz
    )
    var normal = Vec3[.float32, .WORLD, length](
        batch.nx, batch.ny, batch.nz
    )
    var albedo = Vec3[.float32, .WORLD, length](0.0)
    var parameter = SIMD[DType.float32, length](1.0)
    var random_u = SIMD[DType.float32, length](0.0)
    var random_v = SIMD[DType.float32, length](0.0)
    var active = SIMD[DType.bool, length](fill=False)

    for lane in range(lane_count):
        active[lane] = True
        var rng = path_stage_rng(settings.rng_seed, batch.path_ids[lane], stage)
        comptime if MATERIAL_KIND == MAT.LAMBERTIAN:
            ref material = surfaces.lambertians[
                Int(batch.surface_indices[lane])
            ]
            albedo.x[lane] = material.albedo.x
            albedo.y[lane] = material.albedo.y
            albedo.z[lane] = material.albedo.z
            random_u[lane] = rng.f32()
            random_v[lane] = rng.f32()
        elif MATERIAL_KIND == MAT.METAL:
            ref material = surfaces.metals[Int(batch.surface_indices[lane])]
            albedo.x[lane] = material.albedo.x
            albedo.y[lane] = material.albedo.y
            albedo.z[lane] = material.albedo.z
            parameter[lane] = material.fuzz
            if material.fuzz > 1.0e-4:
                random_u[lane] = rng.f32()
                random_v[lane] = rng.f32()
        else:
            ref material = surfaces.dielectrics[
                Int(batch.surface_indices[lane])
            ]
            parameter[lane] = material.refraction_index

    comptime if MATERIAL_KIND == MAT.DIELECTRIC:
        var ri = batch.front_faces.select(Float32(1.0) / parameter, parameter)
        var unit_direction = normalize(ray_direction)
        var cos_theta = min(dot(-unit_direction, normal), 1.0)
        var sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0))
        var cannot_refract = (ri * sin_theta).gt(1.0)
        for lane in range(lane_count):
            if not cannot_refract[lane]:
                var rng = path_stage_rng(
                    settings.rng_seed, batch.path_ids[lane], stage
                )
                random_u[lane] = rng.f32()

    var sampled = _sample_material[MATERIAL_KIND, length](
        ray_direction,
        normal,
        albedo,
        parameter,
        batch.front_faces,
        random_u,
        random_v,
    )

    var out = PathPacket[length]()
    out.path_ids = batch.path_ids
    out.ox = batch.ox + batch.hit_t * batch.dx
    out.oy = batch.oy + batch.hit_t * batch.dy
    out.oz = batch.oz + batch.hit_t * batch.dz
    out.t_min = 0.001
    out.dx = sampled.direction.x
    out.dy = sampled.direction.y
    out.dz = sampled.direction.z
    out.t_max = f32_max
    out.tx = batch.tx * sampled.weight.x
    out.ty = batch.ty * sampled.weight.y
    out.tz = batch.tz * sampled.weight.z
    out.bsdf_pdfs = sampled.pdf
    out.deltas = sampled.delta
    return ScatterBatch[length](out^, sampled.ok & active)


@always_inline
def _accumulate_sky_packet[
    length: SIMDLength
](
    pixels: MutSpan[Color, _],
    packet: PathPacket[length],
    lane_count: Int,
    misses: SIMD[DType.bool, length],
    samples_per_pixel: Int,
):
    var sky = sky_color(
        Vec3[.float32, .WORLD, length](
            packet.dx, packet.dy, packet.dz
        )
    )
    var red = packet.tx * sky.x
    var green = packet.ty * sky.y
    var blue = packet.tz * sky.z
    for lane in range(lane_count):
        if misses[lane]:
            var pixel_idx = Int(packet.path_ids[lane]) / samples_per_pixel
            pixels[pixel_idx] += Color(red[lane], green[lane], blue[lane])


@always_inline
def _shade_material_packets[
    MATERIAL_KIND: MAT,
    length: SIMDLength,
](
    mut next_paths: PacketPathQueue[length],
    queue: PacketShadeQueue[length],
    surfaces: SurfaceStore,
    settings: RenderSettings,
    stage: UInt32,
):
    for packet_idx, batch in enumerate(queue.packets):
        var lane_count = min(length, len(queue) - packet_idx * length)
        var scattered = _sample_bsdf_batch[MATERIAL_KIND, length](
            batch, lane_count, surfaces, settings, stage
        )
        var paths = scattered.paths.copy()
        var ok = scattered.ok
        for lane in range(lane_count):
            if not ok[lane]:
                continue
            var throughput = Color(
                paths.tx[lane], paths.ty[lane], paths.tz[lane]
            )
            var roulette = russian_roulette(
                settings.rng_seed,
                paths.path_ids[lane],
                stage,
                throughput,
            )
            if roulette.survived:
                paths.tx[lane] = roulette.throughput.x
                paths.ty[lane] = roulette.throughput.y
                paths.tz[lane] = roulette.throughput.z
            else:
                ok[lane] = False
        next_paths.append_packet_masked(paths^, ok, lane_count)


def _trace_path_packets[
    length: SIMDLength,
    ALGORITHM: RENDER,
    world_bvh_width: SIMDLength,
    instance_bvh_width: SIMDLength,
](
    settings: RenderSettings,
    world: CpuScene[world_bvh_width, instance_bvh_width],
    pixels: MutSpan[Color, _],
    mut active_paths: PacketPathQueue[length],
    mut next_paths: PacketPathQueue[length],
    mut lambertian_queue: PacketShadeQueue[length],
    mut metal_queue: PacketShadeQueue[length],
    mut dielectric_queue: PacketShadeQueue[length],
):
    comptime assert ALGORITHM in (RENDER.PATH, RENDER.NEE, RENDER.MIS)
    for bounce in range(settings.max_depth):
        if len(active_paths) == 0:
            break
        lambertian_queue.clear()
        metal_queue.clear()
        dielectric_queue.clear()
        next_paths.clear()
        for packet_idx, packet in enumerate(active_paths.packets):
            var lane_count = min(
                length, len(active_paths) - packet_idx * length
            )
            var misses = SIMD[DType.bool, length](fill=False)
            var direct_lights = _DirectLightBatch[length]()
            var valid_lanes = SIMD[DType.bool, length](fill=False)
            for lane in range(lane_count):
                valid_lanes[lane] = True
            var ray_packet = Ray[.float32, .WORLD, length](
                Point3[DType.float32, .WORLD, length](
                    packet.ox, packet.oy, packet.oz
                ),
                Vec3[.float32, .WORLD, length](
                    packet.dx, packet.dy, packet.dz
                ),
                packet.t_min,
                packet.t_max,
            )
            var surface_hits = world.trace_surface(ray_packet, valid_lanes)
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
                if hit.hit:
                    var pixel_idx = (
                        Int(packet.path_ids[lane]) / settings.samples_per_pixel
                    )
                    if hit.surface.kind() == MAT.EMISSIVE:
                        var emission_weight = _emissive_hit_weight[ALGORITHM](
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
                            * emitted_radiance(
                                hit.surface,
                                world.scene_data().surfaces(),
                                hit.front_face,
                            )
                            * emission_weight
                        )
                        continue

                    comptime if ALGORITHM != .PATH:
                        var point = ShadingPoint(
                            ray.o + hit.t * ray.d,
                            hit.normal,
                            hit.front_face,
                        )
                        var light_rng = path_stage_rng(
                            settings.rng_seed,
                            packet.path_ids[lane],
                            wavefront_rng_light_stage(UInt32(bounce)),
                        )
                        var direct = _sample_direct_light_candidate(
                            world, point, light_rng
                        )
                        direct_lights.point.normal.x[lane] = hit.normal.x
                        direct_lights.point.normal.y[lane] = hit.normal.y
                        direct_lights.point.normal.z[lane] = hit.normal.z
                        direct_lights.point.front_face[lane] = hit.front_face
                        direct_lights.surface_kinds[lane] = hit.surface.kind().v
                        direct_lights.surface_indices[
                            lane
                        ] = hit.surface.index()
                        if direct.valid:
                            direct_lights.sample.valid[lane] = True
                            direct_lights.point.p.x[lane] = point.p.x
                            direct_lights.point.p.y[lane] = point.p.y
                            direct_lights.point.p.z[lane] = point.p.z
                            direct_lights.sample.direction.x[
                                lane
                            ] = direct.direction.x
                            direct_lights.sample.direction.y[
                                lane
                            ] = direct.direction.y
                            direct_lights.sample.direction.z[
                                lane
                            ] = direct.direction.z
                            direct_lights.sample.emission.x[
                                lane
                            ] = direct.emission.x
                            direct_lights.sample.emission.y[
                                lane
                            ] = direct.emission.y
                            direct_lights.sample.emission.z[
                                lane
                            ] = direct.emission.z
                            direct_lights.sample.surface_cosine[
                                lane
                            ] = direct.surface_cosine
                            direct_lights.sample.light_pdf[
                                lane
                            ] = direct.light_pdf
                            direct_lights.sample.shadow_t_max[
                                lane
                            ] = direct.shadow_t_max

                    if hit.surface.kind() == MAT.LAMBERTIAN:
                        lambertian_queue.append(packet, lane, hit)
                    elif hit.surface.kind() == MAT.METAL:
                        metal_queue.append(packet, lane, hit)
                    elif hit.surface.kind() == MAT.DIELECTRIC:
                        dielectric_queue.append(packet, lane, hit)
                else:
                    misses[lane] = True

            comptime if ALGORITHM != .PATH:
                var shadow_rays = Ray[.float32, .WORLD, length](
                    direct_lights.point.p,
                    direct_lights.sample.direction,
                    SIMD[DType.float32, length](0.001),
                    direct_lights.sample.shadow_t_max,
                )
                direct_lights.sample.valid &= ~world.occluded(
                    shadow_rays, direct_lights.sample.valid
                )
                _accumulate_direct_light_packet[ALGORITHM, length](
                    pixels,
                    packet,
                    direct_lights,
                    lane_count,
                    world.scene_data().surfaces(),
                    settings.samples_per_pixel,
                )
            _accumulate_sky_packet[length](
                pixels,
                packet,
                lane_count,
                misses,
                settings.samples_per_pixel,
            )

        _shade_material_packets[MAT.LAMBERTIAN, length](
            next_paths,
            lambertian_queue,
            world.scene_data().surfaces(),
            settings,
            UInt32(bounce + 1),
        )
        _shade_material_packets[MAT.METAL, length](
            next_paths,
            metal_queue,
            world.scene_data().surfaces(),
            settings,
            UInt32(bounce + 1),
        )
        _shade_material_packets[MAT.DIELECTRIC, length](
            next_paths,
            dielectric_queue,
            world.scene_data().surfaces(),
            settings,
            UInt32(bounce + 1),
        )
        swap(active_paths, next_paths)
