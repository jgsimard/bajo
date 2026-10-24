"""State layout shared by preparation owners and rendering borrowers.

Both extensions must be built with the same compiler and source revision.
The Python viewer retains the preparation module until its handles are freed.
"""

from std.memory import Pointer
from std.python import Python, PythonObject
from bajo.bvh import Camera
from bajo.core import Point3W, Vec3W
from bajo.core.random import Sampler
from bajo.rt.render_types import RenderSettings
from .core import CpuViewerState, viewer_camera as _viewer_camera
from .output import write_linear_frame as _write_linear_frame


def _linear_frame(
    width: Int, height: Int, pixels: ImmSpan[Vec3W, _]
) raises -> PythonObject:
    """Copy linear RGB into a Python-owned array that outlives the render result.
    """
    if width <= 0 or height <= 0 or len(pixels) != width * height:
        raise Error("viewer frame dimensions do not match its pixels")
    var numpy = Python.import_module("numpy")
    var frame = numpy.empty(Python.tuple(height, width, 3), dtype="float32")
    # NumPy owns a contiguous float32 allocation. Keep frame alive while filling
    # it; returning the array transfers ownership independently of Mojo pixels.
    var output = Pointer[Float32, MutUntrackedOrigin](
        unsafe_from_address=Int(py=frame.ctypes.data)
    )
    for index in range(len(pixels)):
        output[unsafe_offset=3 * index] = pixels[index].x
        output[unsafe_offset=3 * index + 1] = pixels[index].y
        output[unsafe_offset=3 * index + 2] = pixels[index].z
    return frame^


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
