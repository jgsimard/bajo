"""State layout shared by preparation owners and rendering borrowers.

Both extensions must be built with the same compiler and source revision.
The Python viewer retains the preparation module until its handles are freed.
"""

from std.math import cos, sin
from std.io.file_descriptor import FileDescriptor
from std.memory import bitcast
from std.python import PythonObject
from bajo.bvh import Camera
from bajo.core import Point3W, Vec3W
from bajo.core.random import Sampler
from bajo.core.utils import degrees_to_radians
from bajo.rt.cpu.scene import CpuScene
from bajo.rt.render_types import RenderSettings

comptime STATE_KIND_MASK = 7
comptime STATE_COMPRESSED_TRIANGLES = 8
comptime STATE_COMPRESSED_BLASES = 16


struct CpuViewerState:
    var world: CpuScene[16, 16]

    def __init__(out self, var world: CpuScene[16, 16]):
        self.world = world^


def _viewer_camera(
    origin: Point3W,
    yaw_degrees: Float32,
    pitch_degrees: Float32,
    vfov: Float32,
) -> Camera:
    var yaw = degrees_to_radians(yaw_degrees)
    var pitch = degrees_to_radians(pitch_degrees)
    var cos_pitch = cos(pitch)
    var forward = Vec3W(
        sin(yaw) * cos_pitch,
        sin(pitch),
        -cos(yaw) * cos_pitch,
    )
    return Camera.from_vfov(
        origin,
        origin + forward,
        Vec3W(0.0, 1.0, 0.0),
        vfov,
        10.0,
        0.0,
    )


def _write_linear_frame(path: String, pixels: ImmSpan[Vec3W, _]) raises:
    var bytes = List[UInt8](length=len(pixels) * 12, fill=0)
    var out_idx = 0
    for pixel in pixels:
        var values = SIMD[.float32, 4](pixel.x, pixel.y, pixel.z, 0.0)
        for channel in range(3):
            var value = values[channel]
            var word = bitcast[.uint32](value)
            bytes[out_idx + 0] = UInt8(word & UInt32(0xFF))
            bytes[out_idx + 1] = UInt8((word >> UInt32(8)) & UInt32(0xFF))
            bytes[out_idx + 2] = UInt8((word >> UInt32(16)) & UInt32(0xFF))
            bytes[out_idx + 3] = UInt8(word >> UInt32(24))
            out_idx += 4
    with open(path, "w") as f:
        var fd = FileDescriptor(f)
        fd.write_bytes(bytes)


def _settings(config: PythonObject) raises -> RenderSettings:
    return RenderSettings(
        Int(py=config["width"]),
        Int(py=config["height"]),
        Int(py=config["samples"]),
        UInt64(1234),
        Int(py=config["max_depth"]),
        Sampler(Int(py=config["sampler"])),
        Int(py=config["sample_offset"]),
        Int(py=config["sample_sequence_length"]),
    )


def _camera(config: PythonObject) raises -> Camera:
    return _viewer_camera(
        Point3W(
            Float32(py=config["x"]),
            Float32(py=config["y"]),
            Float32(py=config["z"]),
        ),
        Float32(py=config["yaw"]),
        Float32(py=config["pitch"]),
        Float32(py=config["vfov"]),
    )
