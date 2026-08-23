from std.math import tan

from bajo.bvh.constants import f32_max
from bajo.core import (
    Vec3,
    Vec3f32,
    normalize,
    cross,
    Point3,
    Point3f32,
    Frame,
    Ray,
    Rayf32,
)
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

    var origin: Point3f32[.WORLD]
    var forward: Vec3f32[.WORLD]
    var right: Vec3f32[.WORLD]
    var up: Vec3f32[.WORLD]
    var fov_scale: Float32
    var focus_dist: Float32
    var defocus_disk_u: Vec3f32[.WORLD]
    var defocus_disk_v: Vec3f32[.WORLD]

    def __init__(out self, data: ImmSpan[Float32, _], base: Int = 0):
        debug_assert["safe", _use_compiler_assume=True](
            base >= 0 and base <= len(data) - Self.STRIDE,
            "Camera load is outside the input span",
        )
        self.origin = [
            data.unsafe_get(base + Camera.ORIGIN + 0),
            data.unsafe_get(base + Camera.ORIGIN + 1),
            data.unsafe_get(base + Camera.ORIGIN + 2),
        ]
        self.forward = [
            data.unsafe_get(base + Camera.FORWARD + 0),
            data.unsafe_get(base + Camera.FORWARD + 1),
            data.unsafe_get(base + Camera.FORWARD + 2),
        ]
        self.right = [
            data.unsafe_get(base + Camera.RIGHT + 0),
            data.unsafe_get(base + Camera.RIGHT + 1),
            data.unsafe_get(base + Camera.RIGHT + 2),
        ]
        self.up = [
            data.unsafe_get(base + Camera.UP + 0),
            data.unsafe_get(base + Camera.UP + 1),
            data.unsafe_get(base + Camera.UP + 2),
        ]
        self.fov_scale = data.unsafe_get(base + Camera.FOV)
        self.focus_dist = data.unsafe_get(base + Camera.FOCUS_DIST)
        self.defocus_disk_u = [
            data.unsafe_get(base + Camera.DEFOCUS_DISK_U + 0),
            data.unsafe_get(base + Camera.DEFOCUS_DISK_U + 1),
            data.unsafe_get(base + Camera.DEFOCUS_DISK_U + 2),
        ]
        self.defocus_disk_v = [
            data.unsafe_get(base + Camera.DEFOCUS_DISK_V + 0),
            data.unsafe_get(base + Camera.DEFOCUS_DISK_V + 1),
            data.unsafe_get(base + Camera.DEFOCUS_DISK_V + 2),
        ]

    def __init__(
        out self,
        origin: Point3f32[.WORLD],
        target: Point3f32[.WORLD],
        world_up: Vec3f32[.WORLD],
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
        origin: Point3f32[.WORLD],
        target: Point3f32[.WORLD],
        world_up: Vec3f32[.WORLD],
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
    ) -> Rayf32[.WORLD]:
        return self.make_ray_sampled(
            Float32(px_i),
            Float32(py_i),
            Float32(width),
            Float32(height),
            0.5,
            0.5,
            0.0,
            0.0,
            0.0,
        )

    def make_ray_raster(
        self,
        px_i: Int,
        py_i: Int,
        width: Int,
        inv_height: Float32,
    ) -> Rayf32[.WORLD]:
        var screen_x = (
            2.0 * (Float32(px_i) + 0.5) - Float32(width)
        ) * inv_height
        var screen_y = 1.0 - 2.0 * (Float32(py_i) + 0.5) * inv_height

        var direction = normalize(
            self.forward
            + self.right * (screen_x * self.fov_scale)
            + self.up * (screen_y * self.fov_scale)
        )
        return Rayf32[.WORLD](self.origin, direction, 0.0, f32_max)

    def make_ray_sampled[
        length: SIMDLength = 1
    ](
        self,
        px: SIMD[.float32, length],
        py: SIMD[.float32, length],
        width: Float32,
        height: Float32,
        pixel_u: SIMD[.float32, length],
        pixel_v: SIMD[.float32, length],
        lens_u: SIMD[.float32, length] = 0.0,
        lens_v: SIMD[.float32, length] = 0.0,
        t_min: SIMD[.float32, length] = 0.0,
    ) -> Ray[.float32, .WORLD, length]:
        var aspect = width / height
        var sx = ((px + pixel_u) / width) * 2.0 - 1.0
        var sy = 1.0 - ((py + pixel_v) / height) * 2.0
        var origin = Point3[.float32, .WORLD, length](
            self.origin.x, self.origin.y, self.origin.z
        )
        var forward = Vec3[.float32, .WORLD, length](
            self.forward.x, self.forward.y, self.forward.z
        )
        var right = Vec3[.float32, .WORLD, length](
            self.right.x, self.right.y, self.right.z
        )
        var up = Vec3[.float32, .WORLD, length](
            self.up.x, self.up.y, self.up.z
        )
        var disk_u = Vec3[.float32, .WORLD, length](
            self.defocus_disk_u.x,
            self.defocus_disk_u.y,
            self.defocus_disk_u.z,
        )
        var disk_v = Vec3[.float32, .WORLD, length](
            self.defocus_disk_v.x,
            self.defocus_disk_v.y,
            self.defocus_disk_v.z,
        )
        var focal_point = origin + self.focus_dist * (
            forward
            + right * (sx * aspect * self.fov_scale)
            + up * (sy * self.fov_scale)
        )
        var ray_origin = origin + lens_u * disk_u + lens_v * disk_v
        var dir = focal_point - ray_origin

        return Ray[.float32, .WORLD, length](
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
