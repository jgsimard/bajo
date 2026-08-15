"""Width-generic packet CPU wavefront kernels."""

from std.math import sqrt

from bajo.core import Frame, Point3f32, Rayf32, Vec3f32
from bajo.rt.types import (
    Color,
    MAT,
    RENDER,
    RenderSettings,
    ShadingPoint,
    SurfaceId,
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


from ..bsdf import sample_bsdf
from ..common import (
    _path_stage_rng,
    _russian_roulette,
)
from ..lighting import (
    _emissive_hit_weight,
    emitted_radiance,
    sample_direct_lighting,
)


struct ScatterBatch[PACKET_LANES: SIMDLength]:
    var paths: PathPacket[Self.PACKET_LANES]
    var ok: SIMD[DType.bool, Self.PACKET_LANES]

    def __init__(
        out self,
        var paths: PathPacket[Self.PACKET_LANES],
        ok: SIMD[DType.bool, Self.PACKET_LANES],
    ):
        self.paths = paths^
        self.ok = ok


def _sample_bsdf_batch[
    MATERIAL_KIND: MAT, PACKET_LANES: SIMDLength
](
    batch: ShadePacket[PACKET_LANES],
    lane_count: Int,
    surfaces: SurfaceStore,
    settings: RenderSettings,
    stage: UInt32,
) -> ScatterBatch[PACKET_LANES]:
    """Sample the canonical scalar BSDF contract for every active lane."""
    comptime assert MATERIAL_KIND in (
        MAT.LAMBERTIAN,
        MAT.METAL,
        MAT.DIELECTRIC,
    )
    var out = PathPacket[PACKET_LANES]()
    var ok = SIMD[DType.bool, PACKET_LANES](fill=False)
    for lane in range(lane_count):
        var ray = Rayf32[Frame.WORLD](
            Point3f32[Frame.WORLD](
                batch.ox[lane], batch.oy[lane], batch.oz[lane]
            ),
            Vec3f32[Frame.WORLD](
                batch.dx[lane], batch.dy[lane], batch.dz[lane]
            ),
        )
        var point = ShadingPoint(
            ray.o + batch.hit_t[lane] * ray.d,
            Vec3f32[Frame.WORLD](
                batch.nx[lane], batch.ny[lane], batch.nz[lane]
            ),
            batch.front_faces[lane],
        )
        var rng = _path_stage_rng(settings, batch.path_ids[lane], stage)
        var sampled = sample_bsdf(
            SurfaceId(MATERIAL_KIND, batch.surface_indices[lane]),
            surfaces,
            ray,
            point,
            rng,
        )
        if sampled.ok:
            ok[lane] = True
            out.path_ids[lane] = batch.path_ids[lane]
            out.ox[lane] = sampled.ray.o.x
            out.oy[lane] = sampled.ray.o.y
            out.oz[lane] = sampled.ray.o.z
            out.t_min[lane] = sampled.ray.t_min
            out.dx[lane] = sampled.ray.d.x
            out.dy[lane] = sampled.ray.d.y
            out.dz[lane] = sampled.ray.d.z
            out.t_max[lane] = sampled.ray.t_max
            out.tx[lane] = batch.tx[lane] * sampled.weight.x
            out.ty[lane] = batch.ty[lane] * sampled.weight.y
            out.tz[lane] = batch.tz[lane] * sampled.weight.z
            out.bsdf_pdfs[lane] = sampled.pdf
            out.deltas[lane] = sampled.delta
    return ScatterBatch[PACKET_LANES](out^, ok)


@always_inline
def _accumulate_sky_packet[
    PACKET_LANES: SIMDLength
](
    mut pixels: List[Color],
    packet: PathPacket[PACKET_LANES],
    lane_count: Int,
    misses: SIMD[DType.bool, PACKET_LANES],
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
    PACKET_LANES: SIMDLength,
](
    mut next_paths: PacketPathQueue[PACKET_LANES],
    queue: PacketShadeQueue[PACKET_LANES],
    surfaces: SurfaceStore,
    settings: RenderSettings,
    stage: UInt32,
):
    for packet_idx in range(len(queue.packets)):
        var lane_count = min(
            PACKET_LANES, len(queue) - packet_idx * PACKET_LANES
        )
        ref batch = queue.packets[packet_idx]
        var scattered = _sample_bsdf_batch[MATERIAL_KIND, PACKET_LANES](
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
    PACKET_LANES: SIMDLength,
    ALGORITHM: RENDER = RENDER.PATH,
](
    settings: RenderSettings,
    world: World,
    mut pixels: List[Color],
    mut active_paths: PacketPathQueue[PACKET_LANES],
    mut next_paths: PacketPathQueue[PACKET_LANES],
    mut lambertian_queue: PacketShadeQueue[PACKET_LANES],
    mut metal_queue: PacketShadeQueue[PACKET_LANES],
    mut dielectric_queue: PacketShadeQueue[PACKET_LANES],
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
                PACKET_LANES, len(active_paths) - packet_idx * PACKET_LANES
            )
            var misses = SIMD[DType.bool, PACKET_LANES](fill=False)
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
                var hit = world.trace_surface(ray)
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
                        var direct = sample_direct_lighting[ALGORITHM](
                            hit.surface, world, ray, point, light_rng
                        )
                        pixels[pixel_idx] += (
                            Color(
                                packet.tx[lane],
                                packet.ty[lane],
                                packet.tz[lane],
                            )
                            * direct
                        )

                    if hit.surface.kind() == MAT.LAMBERTIAN:
                        lambertian_queue.append(packet, lane, hit)
                    elif hit.surface.kind() == MAT.METAL:
                        metal_queue.append(packet, lane, hit)
                    elif hit.surface.kind() == MAT.DIELECTRIC:
                        dielectric_queue.append(packet, lane, hit)
                else:
                    misses[lane] = True

            _accumulate_sky_packet[PACKET_LANES](
                pixels,
                packet,
                lane_count,
                misses,
                settings.samples_per_pixel,
            )

        _shade_material_packets[MAT.LAMBERTIAN, PACKET_LANES](
            next_paths,
            lambertian_queue,
            world.surfaces,
            settings,
            UInt32(bounce + 1),
        )
        _shade_material_packets[MAT.METAL, PACKET_LANES](
            next_paths,
            metal_queue,
            world.surfaces,
            settings,
            UInt32(bounce + 1),
        )
        _shade_material_packets[MAT.DIELECTRIC, PACKET_LANES](
            next_paths,
            dielectric_queue,
            world.surfaces,
            settings,
            UInt32(bounce + 1),
        )
        swap(active_paths, next_paths)
