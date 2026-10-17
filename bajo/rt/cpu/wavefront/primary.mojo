"""Packet primary-path generation."""

from std.math import iota
from bajo.core.random import random_in_unit_disk
from bajo.bvh import Camera
from bajo.rt.render_types import (
    RenderSettings,
    SamplingConfig,
)
from bajo.rt.common import path_stage_rng
from bajo.rt.rays import make_camera_ray_from_samples
from bajo.rt.wavefront_queue import PacketPathQueue, PathPacket


def _initialize_path_packets_range[
    length: SIMDLength
](
    mut paths: PacketPathQueue[length],
    settings: RenderSettings,
    sampling: SamplingConfig,
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
        var packet = PathPacket[length]()
        var ids = iota[.uint32, length]() + UInt32(packet_begin)
        var pixel_ids = ids / UInt32(settings.samples_per_pixel)
        var px = (pixel_ids % UInt32(settings.image_width)).cast[.float32]()
        var py = (pixel_ids / UInt32(settings.image_width)).cast[.float32]()
        var rng = path_stage_rng(sampling, ids, UInt32(0))
        var lens = random_in_unit_disk[.WORLD, length](rng)
        var lens_u = lens.x
        var lens_v = lens.y
        var pixel_u = rng.f32()
        var pixel_v = rng.f32()
        packet.path_ids = ids

        var ray = make_camera_ray_from_samples[length](
            camera,
            px,
            py,
            width,
            height,
            pixel_u,
            pixel_v,
            lens_u,
            lens_v,
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
