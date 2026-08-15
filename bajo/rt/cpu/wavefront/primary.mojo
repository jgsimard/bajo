"""Packet primary-path generation."""

from bajo.core import Frame, Point3, Vec3, normalize
from bajo.core.random import random_in_unit_disk
from bajo.bvh.camera import Camera
from bajo.bvh.constants import f32_max
from bajo.rt.types import RenderSettings
from bajo.rt.wavefront_queue import PacketPathQueue, PathPacket

from ..common import _path_stage_rng


def _make_initial_path_packets_range[
    PACKET_LANES: SIMDLength
](
    settings: RenderSettings,
    camera: Camera,
    path_begin: Int,
    path_end: Int,
) -> PacketPathQueue[PACKET_LANES]:
    var paths = PacketPathQueue[PACKET_LANES](path_end - path_begin)
    var width = Float32(settings.image_width)
    var height = Float32(settings.image_height)
    var aspect = width / height
    for packet_begin in range(path_begin, path_end, PACKET_LANES):
        var lane_count = min(PACKET_LANES, path_end - packet_begin)
        var px = SIMD[DType.float32, PACKET_LANES](0.0)
        var py = SIMD[DType.float32, PACKET_LANES](0.0)
        var pixel_u = SIMD[DType.float32, PACKET_LANES](0.0)
        var pixel_v = SIMD[DType.float32, PACKET_LANES](0.0)
        var lens_u = SIMD[DType.float32, PACKET_LANES](0.0)
        var lens_v = SIMD[DType.float32, PACKET_LANES](0.0)
        var packet = PathPacket[PACKET_LANES]()
        for lane in range(lane_count):
            var path_idx = packet_begin + lane
            var pixel_idx = path_idx / settings.samples_per_pixel
            px[lane] = Float32(pixel_idx % settings.image_width)
            py[lane] = Float32(pixel_idx / settings.image_width)
            var rng = _path_stage_rng(settings, UInt32(path_idx), UInt32(0))
            var lens = random_in_unit_disk[Frame.WORLD](rng)
            lens_u[lane] = lens.x
            lens_v[lane] = lens.y
            pixel_u[lane] = rng.f32()
            pixel_v[lane] = rng.f32()
            packet.path_ids[lane] = UInt32(path_idx)

        var sx = ((px + pixel_u) / width) * 2.0 - 1.0
        var sy = 1.0 - ((py + pixel_v) / height) * 2.0
        var origin = Point3[DType.float32, Frame.WORLD, PACKET_LANES](
            SIMD[DType.float32, PACKET_LANES](camera.origin.x),
            SIMD[DType.float32, PACKET_LANES](camera.origin.y),
            SIMD[DType.float32, PACKET_LANES](camera.origin.z),
        )
        var forward = Vec3[DType.float32, Frame.WORLD, PACKET_LANES](
            SIMD[DType.float32, PACKET_LANES](camera.forward.x),
            SIMD[DType.float32, PACKET_LANES](camera.forward.y),
            SIMD[DType.float32, PACKET_LANES](camera.forward.z),
        )
        var right = Vec3[DType.float32, Frame.WORLD, PACKET_LANES](
            SIMD[DType.float32, PACKET_LANES](camera.right.x),
            SIMD[DType.float32, PACKET_LANES](camera.right.y),
            SIMD[DType.float32, PACKET_LANES](camera.right.z),
        )
        var up = Vec3[DType.float32, Frame.WORLD, PACKET_LANES](
            SIMD[DType.float32, PACKET_LANES](camera.up.x),
            SIMD[DType.float32, PACKET_LANES](camera.up.y),
            SIMD[DType.float32, PACKET_LANES](camera.up.z),
        )
        var disk_u = Vec3[DType.float32, Frame.WORLD, PACKET_LANES](
            SIMD[DType.float32, PACKET_LANES](camera.defocus_disk_u.x),
            SIMD[DType.float32, PACKET_LANES](camera.defocus_disk_u.y),
            SIMD[DType.float32, PACKET_LANES](camera.defocus_disk_u.z),
        )
        var disk_v = Vec3[DType.float32, Frame.WORLD, PACKET_LANES](
            SIMD[DType.float32, PACKET_LANES](camera.defocus_disk_v.x),
            SIMD[DType.float32, PACKET_LANES](camera.defocus_disk_v.y),
            SIMD[DType.float32, PACKET_LANES](camera.defocus_disk_v.z),
        )
        var focal_point = origin + camera.focus_dist * (
            forward
            + (sx * aspect * camera.fov_scale) * right
            + (sy * camera.fov_scale) * up
        )
        var ray_origin = origin + lens_u * disk_u + lens_v * disk_v
        var direction = normalize(focal_point - ray_origin)
        packet.ox = ray_origin.x
        packet.oy = ray_origin.y
        packet.oz = ray_origin.z
        packet.t_min = 0.001
        packet.dx = direction.x
        packet.dy = direction.y
        packet.dz = direction.z
        packet.t_max = f32_max
        packet.tx = 1.0
        packet.ty = 1.0
        packet.tz = 1.0
        paths.append_packet(packet^, lane_count)
    return paths^
