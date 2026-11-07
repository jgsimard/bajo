"""Packet primary-path generation."""

from std.math import iota
from std.memory import bitcast
from bajo.core.camera import Camera
from bajo.rt.render_types import RenderSettings, SamplingConfig
from bajo.rt.rays import make_camera_ray_from_samples
from bajo.rt.wavefront_queue import PacketPathQueue, PathPacket
from bajo.rt.camera_sampling import sample_camera


def _initialize_path_packets_range[
    length: SIMDLength
](
    mut paths: PacketPathQueue[length],
    settings: RenderSettings,
    sampling: SamplingConfig,
    camera: Camera,
    path_begin: Int,
    path_end: Int,
    weights: MutSpan[Float32, _],
):
    """Fill a reusable queue with primary paths for one render range."""
    paths.clear()
    var width = Float32(settings.image_width)
    var height = Float32(settings.image_height)
    var pinhole = (
        camera.defocus_disk_u.x == 0
        and camera.defocus_disk_u.y == 0
        and camera.defocus_disk_u.z == 0
        and camera.defocus_disk_v.x == 0
        and camera.defocus_disk_v.y == 0
        and camera.defocus_disk_v.z == 0
    )
    # Zero lens offsets can change a negative-zero camera origin. Keep the
    # original sampling math for those origins and for non-finite inputs.
    pinhole = (
        pinhole
        and camera.origin.is_finite()[0]
        and bitcast[.uint32](camera.origin.x) != 0x80000000
        and bitcast[.uint32](camera.origin.y) != 0x80000000
        and bitcast[.uint32](camera.origin.z) != 0x80000000
    )
    for packet_begin in range(path_begin, path_end, length):
        var lane_count = min(length, path_end - packet_begin)
        var packet = PathPacket[length]()
        var ids = iota[.uint32, length]() + UInt32(packet_begin)
        var pixel_ids = ids / UInt32(settings.samples_per_pixel)
        var px = (pixel_ids % UInt32(settings.image_width)).cast[.float32]()
        var py = (pixel_ids / UInt32(settings.image_width)).cast[.float32]()
        var samples = sample_camera(sampling, ids, pinhole)
        for lane in range(lane_count):
            weights[packet_begin - path_begin + lane] = samples.filter_weight[
                lane
            ]
        packet.path_ids = ids

        var ray = make_camera_ray_from_samples[length](
            camera,
            px,
            py,
            width,
            height,
            samples.pixel_u,
            samples.pixel_v,
            samples.lens_u,
            samples.lens_v,
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
