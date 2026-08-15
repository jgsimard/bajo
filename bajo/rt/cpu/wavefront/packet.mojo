"""Width-generic packet CPU wavefront kernels."""

from std.math import sqrt

from bajo.bvh.constants import f32_max
from bajo.bvh.cpu.packet import RayPacket
from bajo.core import (
    Frame,
    Point3,
    Point3f32,
    Rayf32,
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
    World,
)
from bajo.rt.wavefront_queue import (
    PacketPathQueue,
    PacketShadeQueue,
    PathPacket,
    ShadePacket,
)
from bajo.rt.wavefront_contract import wavefront_rng_light_stage


from ..bsdf import (
    _evaluate_lambertian,
    _evaluate_metal,
    _sample_material,
)
from ..common import (
    _path_stage_rng,
    _russian_roulette,
)
from ..lighting import (
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
    var origins: Point3[DType.float32, Frame.WORLD, Self.length]
    var directions: Vec3[DType.float32, Frame.WORLD, Self.length]
    var normals: Vec3[DType.float32, Frame.WORLD, Self.length]
    var emissions: Vec3[DType.float32, Frame.WORLD, Self.length]
    var surface_cosines: SIMD[DType.float32, Self.length]
    var light_pdfs: SIMD[DType.float32, Self.length]
    var shadow_t_max: SIMD[DType.float32, Self.length]
    var surface_kinds: SIMD[DType.uint32, Self.length]
    var surface_indices: SIMD[DType.uint32, Self.length]
    var valid: SIMD[DType.bool, Self.length]

    def __init__(out self):
        self.origins = Point3[DType.float32, Frame.WORLD, Self.length](0.0)
        self.directions = Vec3[DType.float32, Frame.WORLD, Self.length](0.0)
        self.normals = Vec3[DType.float32, Frame.WORLD, Self.length](0.0)
        self.emissions = Vec3[DType.float32, Frame.WORLD, Self.length](0.0)
        self.surface_cosines = 0.0
        self.light_pdfs = 0.0
        self.shadow_t_max = 0.0
        self.surface_kinds = 0
        self.surface_indices = 0
        self.valid = SIMD[DType.bool, Self.length](fill=False)


@always_inline
def _accumulate_direct_light_packet[
    ALGORITHM: RENDER, length: SIMDLength
](
    mut pixels: List[Color],
    paths: PathPacket[length],
    lights: _DirectLightBatch[length],
    lane_count: Int,
    surfaces: SurfaceStore,
    samples_per_pixel: Int,
):
    """Evaluate collected NEE candidates with packet-wide BSDF and MIS math."""
    comptime assert ALGORITHM in (RENDER.NEE, RENDER.MIS)
    var ray_direction = Vec3[DType.float32, Frame.WORLD, length](
        paths.dx, paths.dy, paths.dz
    )
    var albedo = Vec3[DType.float32, Frame.WORLD, length](0.0)
    var fuzz = SIMD[DType.float32, length](1.0)
    for lane in range(lane_count):
        if not lights.valid[lane]:
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

    var lambertian = _evaluate_lambertian(
        lights.normals, albedo, lights.directions
    )
    var metal = _evaluate_metal(
        ray_direction,
        lights.normals,
        albedo,
        fuzz,
        lights.directions,
    )
    var is_lambertian = lights.surface_kinds.eq(MAT.LAMBERTIAN.v)
    var is_metal = lights.surface_kinds.eq(MAT.METAL.v)
    var value = Vec3.select(is_lambertian, lambertian.value, metal.value)
    var pdf = is_lambertian.select(lambertian.pdf, metal.pdf)
    var supported = is_lambertian | is_metal
    var ok = lights.valid & supported & pdf.gt(0.0)
    var safe_light_pdf = ok.select(lights.light_pdfs, Float32(1.0))
    var estimator_weight = SIMD[DType.float32, length](1.0)
    comptime if ALGORITHM == RENDER.MIS:
        var light2 = lights.light_pdfs * lights.light_pdfs
        var bsdf2 = pdf * pdf
        var denominator = light2 + bsdf2
        var nonzero = denominator.gt(0.0)
        estimator_weight = nonzero.select(
            light2 / nonzero.select(denominator, Float32(1.0)),
            Float32(0.0),
        )
    var scale = lights.surface_cosines * estimator_weight / safe_light_pdf
    var red = paths.tx * value.x * lights.emissions.x * scale
    var green = paths.ty * value.y * lights.emissions.y * scale
    var blue = paths.tz * value.z * lights.emissions.z * scale
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
    var ray_direction = Vec3[DType.float32, Frame.WORLD, length](
        batch.dx, batch.dy, batch.dz
    )
    var normal = Vec3[DType.float32, Frame.WORLD, length](
        batch.nx, batch.ny, batch.nz
    )
    var albedo = Vec3[DType.float32, Frame.WORLD, length](0.0)
    var parameter = SIMD[DType.float32, length](1.0)
    var random_u = SIMD[DType.float32, length](0.0)
    var random_v = SIMD[DType.float32, length](0.0)
    var active = SIMD[DType.bool, length](fill=False)

    for lane in range(lane_count):
        active[lane] = True
        var rng = _path_stage_rng(settings, batch.path_ids[lane], stage)
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
                var rng = _path_stage_rng(settings, batch.path_ids[lane], stage)
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
    mut pixels: List[Color],
    packet: PathPacket[length],
    lane_count: Int,
    misses: SIMD[DType.bool, length],
    samples_per_pixel: Int,
):
    var ray_length = sqrt(
        packet.dx * packet.dx + packet.dy * packet.dy + packet.dz * packet.dz
    )
    var valid = ray_length.gt(1.0e-20)
    var unit_y = valid.select(packet.dy / valid.select(ray_length, 1.0), 0.0)
    var a = 0.5 * (unit_y + 1.0)
    var red = packet.tx * (1.0 - 0.5 * a)
    var green = packet.ty * (1.0 - 0.3 * a)
    var blue = packet.tz
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
    for packet_idx in range(len(queue.packets)):
        var lane_count = min(length, len(queue) - packet_idx * length)
        ref batch = queue.packets[packet_idx]
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
            var roulette = _russian_roulette(
                settings,
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
    MAX_DEPTH: Int,
    length: SIMDLength,
    ALGORITHM: RENDER = RENDER.PATH,
](
    settings: RenderSettings,
    world: World,
    mut pixels: List[Color],
    mut active_paths: PacketPathQueue[length],
    mut next_paths: PacketPathQueue[length],
    mut lambertian_queue: PacketShadeQueue[length],
    mut metal_queue: PacketShadeQueue[length],
    mut dielectric_queue: PacketShadeQueue[length],
):
    comptime assert ALGORITHM in (RENDER.PATH, RENDER.NEE, RENDER.MIS)
    for bounce in range(MAX_DEPTH):
        if len(active_paths) == 0:
            break
        lambertian_queue.clear()
        metal_queue.clear()
        dielectric_queue.clear()
        next_paths.clear()
        for packet_idx in range(len(active_paths.packets)):
            ref packet = active_paths.packets[packet_idx]
            var lane_count = min(
                length, len(active_paths) - packet_idx * length
            )
            var misses = SIMD[DType.bool, length](fill=False)
            var direct_lights = _DirectLightBatch[length]()
            var valid_lanes = SIMD[DType.bool, length](fill=False)
            for lane in range(lane_count):
                valid_lanes[lane] = True
            var ray_packet = RayPacket[Frame.WORLD, length](
                Point3[DType.float32, Frame.WORLD, length](
                    packet.ox, packet.oy, packet.oz
                ),
                Vec3[DType.float32, Frame.WORLD, length](
                    packet.dx, packet.dy, packet.dz
                ),
                packet.t_min,
                packet.t_max,
            )
            var surface_hits = world.trace_surface_packet(
                ray_packet, valid_lanes
            )
            for lane in range(lane_count):
                var ray = Rayf32[Frame.WORLD](
                    Point3f32[Frame.WORLD](
                        packet.ox[lane], packet.oy[lane], packet.oz[lane]
                    ),
                    Vec3f32[Frame.WORLD](
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
                                hit.surface, world.surfaces, hit.front_face
                            )
                            * emission_weight
                        )
                        continue

                    comptime if ALGORITHM != RENDER.PATH:
                        var point = ShadingPoint(
                            ray.o + hit.t * ray.d,
                            hit.normal,
                            hit.front_face,
                        )
                        var light_rng = _path_stage_rng(
                            settings,
                            packet.path_ids[lane],
                            wavefront_rng_light_stage(UInt32(bounce)),
                        )
                        var direct = _sample_direct_light_candidate(
                            world, point, light_rng
                        )
                        direct_lights.normals.x[lane] = hit.normal.x
                        direct_lights.normals.y[lane] = hit.normal.y
                        direct_lights.normals.z[lane] = hit.normal.z
                        direct_lights.surface_kinds[lane] = hit.surface.kind().v
                        direct_lights.surface_indices[
                            lane
                        ] = hit.surface.index()
                        if direct.valid:
                            direct_lights.valid[lane] = True
                            direct_lights.origins.x[lane] = point.p.x
                            direct_lights.origins.y[lane] = point.p.y
                            direct_lights.origins.z[lane] = point.p.z
                            direct_lights.directions.x[
                                lane
                            ] = direct.direction.x
                            direct_lights.directions.y[
                                lane
                            ] = direct.direction.y
                            direct_lights.directions.z[
                                lane
                            ] = direct.direction.z
                            direct_lights.emissions.x[lane] = direct.emission.x
                            direct_lights.emissions.y[lane] = direct.emission.y
                            direct_lights.emissions.z[lane] = direct.emission.z
                            direct_lights.surface_cosines[
                                lane
                            ] = direct.surface_cosine
                            direct_lights.light_pdfs[lane] = direct.light_pdf
                            direct_lights.shadow_t_max[
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

            comptime if ALGORITHM != RENDER.PATH:
                var shadow_rays = RayPacket[Frame.WORLD, length](
                    direct_lights.origins,
                    direct_lights.directions,
                    SIMD[DType.float32, length](0.001),
                    direct_lights.shadow_t_max,
                )
                direct_lights.valid &= ~world.occluded_packet(
                    shadow_rays, direct_lights.valid
                )
                _accumulate_direct_light_packet[ALGORITHM, length](
                    pixels,
                    packet,
                    direct_lights,
                    lane_count,
                    world.surfaces,
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
            world.surfaces,
            settings,
            UInt32(bounce + 1),
        )
        _shade_material_packets[MAT.METAL, length](
            next_paths,
            metal_queue,
            world.surfaces,
            settings,
            UInt32(bounce + 1),
        )
        _shade_material_packets[MAT.DIELECTRIC, length](
            next_paths,
            dielectric_queue,
            world.surfaces,
            settings,
            UInt32(bounce + 1),
        )
        swap(active_paths, next_paths)
