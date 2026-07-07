from std.math import tan

from bajo.bvh.types import Ray
from bajo.bvh.constants import f32_max
from bajo.core import Vec3f32, normalize, cross, Point3f32
from bajo.core.utils import degrees_to_radians


@fieldwise_init
struct Camera(TrivialRegisterPassable, Writable):
    comptime STRIDE = 20
    comptime ORIGIN = 0
    comptime FORWARD = 3
    comptime RIGHT = 6
    comptime UP = 9
    comptime FOV = 12
    comptime FOCUS_DIST = 13
    comptime DEFOCUS_DISK_U = 14
    comptime DEFOCUS_DISK_V = 17

    var origin: Point3f32
    var forward: Vec3f32
    var right: Vec3f32
    var up: Vec3f32
    var fov_scale: Float32
    var focus_dist: Float32
    var defocus_disk_u: Vec3f32
    var defocus_disk_v: Vec3f32

    def __init__[
        origin: ImmutOrigin
    ](out self, ptr: UnsafePointer[Float32, origin], base: Int = 0):
        self.origin = Point3f32.load(ptr, base + Camera.ORIGIN)
        self.forward = Vec3f32.load(ptr, base + Camera.FORWARD)
        self.right = Vec3f32.load(ptr, base + Camera.RIGHT)
        self.up = Vec3f32.load(ptr, base + Camera.UP)
        self.fov_scale = ptr[base + Camera.FOV]
        self.focus_dist = ptr[base + Camera.FOCUS_DIST]
        self.defocus_disk_u = Vec3f32.load(ptr, base + Camera.DEFOCUS_DISK_U)
        self.defocus_disk_v = Vec3f32.load(ptr, base + Camera.DEFOCUS_DISK_V)

    def __init__(
        out self,
        origin: Point3f32,
        target: Point3f32,
        world_up: Vec3f32,
        fov_scale: Float32,
        focus_dist: Float32 = 1.0,
        defocus_angle: Float32 = 0.0,
    ):
        self.origin = origin
        self.forward = normalize(target - origin)
        self.right = normalize(cross(self.forward, world_up))
        self.up = normalize(cross(self.right, self.forward))
        self.fov_scale = fov_scale
        self.focus_dist = focus_dist

        var defocus_radius = focus_dist * tan(
            degrees_to_radians(defocus_angle / 2.0)
        )
        self.defocus_disk_u = self.right * defocus_radius
        self.defocus_disk_v = self.up * defocus_radius

    @staticmethod
    def from_vfov(
        origin: Point3f32,
        target: Point3f32,
        world_up: Vec3f32,
        vfov: Float32,
        focus_dist: Float32 = 1.0,
        defocus_angle: Float32 = 0.0,
    ) -> Self:
        var theta = degrees_to_radians(vfov)
        return Self(
            origin,
            target,
            world_up,
            tan(theta / 2.0),
            focus_dist,
            defocus_angle,
        )

    def make_ray(
        self,
        px_i: Int,
        py_i: Int,
        width: Int,
        height: Int,
    ) -> Ray:
        return self.make_ray_sampled(
            px_i,
            py_i,
            width,
            height,
            0.5,
            0.5,
            0.0,
            0.0,
            0.0,
        )

    def make_ray_sampled(
        self,
        px_i: Int,
        py_i: Int,
        width: Int,
        height: Int,
        pixel_u: Float32,
        pixel_v: Float32,
        lens_u: Float32 = 0.0,
        lens_v: Float32 = 0.0,
        t_min: Float32 = 0.0,
    ) -> Ray:
        var aspect = Float32(width) / Float32(height)

        var sx = ((Float32(px_i) + pixel_u) / Float32(width)) * 2.0 - 1.0
        var sy = 1.0 - ((Float32(py_i) + pixel_v) / Float32(height)) * 2.0

        var focal_point = self.origin + self.focus_dist * (
            self.forward
            + self.right * (sx * aspect * self.fov_scale)
            + self.up * (sy * self.fov_scale)
        )
        var ray_origin = (
            self.origin
            + lens_u * self.defocus_disk_u
            + lens_v * self.defocus_disk_v
        )
        var dir = focal_point - ray_origin

        return Ray(
            ray_origin,
            normalize(dir),
            t_min,
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
            self.focus_dist,
            self.defocus_disk_u.x,
            self.defocus_disk_u.y,
            self.defocus_disk_u.z,
            self.defocus_disk_v.x,
            self.defocus_disk_v.y,
            self.defocus_disk_v.z,
        ]
