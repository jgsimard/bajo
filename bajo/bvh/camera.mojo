from bajo.bvh.types import Ray
from bajo.bvh.constants import f32_max
from bajo.core import Vec3f32, normalize, cross


@fieldwise_init
struct Camera(TrivialRegisterPassable, Writable):
    comptime STRIDE = 13
    comptime ORIGIN = 0
    comptime FORWARD = 3
    comptime RIGHT = 6
    comptime UP = 9
    comptime FOV = 12

    var origin: Vec3f32
    var forward: Vec3f32
    var right: Vec3f32
    var up: Vec3f32
    var fov_scale: Float32

    def __init__[
        origin: ImmutOrigin
    ](out self, ptr: UnsafePointer[Float32, origin], base: Int = 0):
        self.origin = Vec3f32.load(ptr, base + Camera.ORIGIN)
        self.forward = Vec3f32.load(ptr, base + Camera.FORWARD)
        self.right = Vec3f32.load(ptr, base + Camera.RIGHT)
        self.up = Vec3f32.load(ptr, base + Camera.UP)
        self.fov_scale = ptr[base + Camera.FOV]

    def __init__(
        out self,
        origin: Vec3f32,
        target: Vec3f32,
        world_up: Vec3f32,
        fov_scale: Float32,
    ):
        self.origin = origin
        self.forward = normalize(target - origin)
        self.right = normalize(cross(self.forward, world_up))
        self.up = normalize(cross(self.right, self.forward))
        self.fov_scale = fov_scale

    def make_ray(
        self,
        px_i: Int,
        py_i: Int,
        width: SIMDSize,
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

    def flatten(self) -> List[Float32]:
        return [
            self.origin.x,
            self.origin.y,
            self.origin.z,
            self.forward.x,
            self.forward.y,
            self.forward.z,
            self.right.x,
            self.right.y,
            self.right.z,
            self.up.x,
            self.up.y,
            self.up.z,
            self.fov_scale,
        ]
