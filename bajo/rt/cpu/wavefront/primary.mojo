"""Packet primary-path generation."""

from bajo.core.random import random_in_unit_disk
from bajo.bvh import Camera
from bajo.rt.types import RenderSettings, sampling_config
from bajo.rt.common import path_stage_rng
from bajo.rt.wavefront_queue import PacketPathQueue, PathPacket


def _initialize_path_packets_range[
    length: SIMDLength
](
    mut paths: PacketPathQueue[length],
    settings: RenderSettings,
    camera: Camera,
    path_begin: Int,
    path_end: Int,
):
    """Fill a reusable queue with primary paths for one render range."""
    paths.clear()
    var width = Float32(settings.image_width)
    var height = Float32(settings.image_height)
    for packet_begin in range(path_begin, path_end, length):
        var lane_count = min(length, path_end - packet_begin)
        var px = SIMD[.float32, length](0.0)
        var py = SIMD[.float32, length](0.0)
        var pixel_u = SIMD[.float32, length](0.0)
        var pixel_v = SIMD[.float32, length](0.0)
        var lens_u = SIMD[.float32, length](0.0)
        var lens_v = SIMD[.float32, length](0.0)
        var packet = PathPacket[length]()
        for lane in range(lane_count):
            var path_idx = packet_begin + lane
            var pixel_idx = path_idx / settings.samples_per_pixel
            px[lane] = Float32(pixel_idx % settings.image_width)
            py[lane] = Float32(pixel_idx / settings.image_width)
            var rng = path_stage_rng(
                sampling_config(settings), UInt32(path_idx), UInt32(0)
            )
            var lens = random_in_unit_disk[.WORLD](rng)
            lens_u[lane] = lens.x
            lens_v[lane] = lens.y
            pixel_u[lane] = rng.f32()
            pixel_v[lane] = rng.f32()
            packet.path_ids[lane] = UInt32(path_idx)

        var ray = camera.make_ray_sampled[length](
            px,
            py,
            width,
            height,
            pixel_u,
            pixel_v,
            lens_u,
            lens_v,
            0.001,
        )
        packet.ox = ray.o.x
        packet.oy = ray.o.y
        packet.oz = ray.o.z
        packet.t_min = ray.t_min
        packet.dx = ray.d.x
        packet.dy = ray.d.y
        packet.dz = ray.d.z
        packet.t_max = ray.t_max
        packet.tx = 1.0
        packet.ty = 1.0
        packet.tz = 1.0
        paths.append_packet(packet^, lane_count)
