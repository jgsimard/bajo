from bajo.bvh.types import Ray
from bajo.bvh.constants import f32_max
from bajo.core.vec import Vec3f32, normalize


comptime CAMERA_STRIDE = 13
comptime CAMERA_ORIGIN = 0
comptime CAMERA_FORWARD = 3
comptime CAMERA_RIGHT = 6
comptime CAMERA_UP = 9
comptime CAMERA_FOV = 12


@fieldwise_init
struct Camera(TrivialRegisterPassable, Writable):
    var origin: Vec3f32
    var forward: Vec3f32
    var right: Vec3f32
    var up: Vec3f32
    var fov_scale: Float32

    def __init__(
        out self,
        ptr: UnsafePointer[Float32, MutAnyOrigin],
        base: Int = 0,
    ):
        self.origin = Vec3f32.load(ptr, base + CAMERA_ORIGIN)
        self.forward = Vec3f32.load(ptr, base + CAMERA_FORWARD)
        self.right = Vec3f32.load(ptr, base + CAMERA_RIGHT)
        self.up = Vec3f32.load(ptr, base + CAMERA_UP)
        self.fov_scale = ptr[base + CAMERA_FOV]

    def make_ray(
        self,
        px_i: Int,
        py_i: Int,
        width: Int,
        height: Int,
    ) -> Ray:
        var aspect = Float32(width) / Float32(height)

        var sx = ((Float32(px_i) + 0.5) / Float32(width)) * 2.0 - 1.0
        var sy = 1.0 - ((Float32(py_i) + 0.5) / Float32(height)) * 2.0

        var dir = (
            self.forward
            + self.right * (sx * aspect * self.fov_scale)
            + self.up * (sy * self.fov_scale)
        )

        return Ray(
            self.origin,
            0.0,
            normalize(dir),
            f32_max,
        )
