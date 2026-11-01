"""Pinhole and thin-lens camera geometry and ray generation."""

from std.math import atan, cos, sin, tan
from std.utils.numerics import max_finite
from .vec import Vec3, Point3, normalize, cross, dot, length
from .ray import Ray
from .transform import Affine3
from .utils import degrees_to_radians, radians_to_degrees


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

    var origin: Point3[.float32, .WORLD]
    var forward: Vec3[.float32, .WORLD]
    var right: Vec3[.float32, .WORLD]
    var up: Vec3[.float32, .WORLD]
    var fov_scale: Float32
    var focus_dist: Float32
    var defocus_disk_u: Vec3[.float32, .WORLD]
    var defocus_disk_v: Vec3[.float32, .WORLD]

    def __init__(out self, data: ImmSpan[Float32, _], base: Int = 0):
        debug_assert["safe", _use_compiler_assume=True](
            base >= 0 and base <= len(data) - Self.STRIDE,
            "Camera load is outside the input span",
        )
        var fields = data.unsafe_ptr().unsafe_offset(base)
        self.origin = Point3[.float32, .WORLD](
            fields.unsafe_offset(Self.ORIGIN)
        )
        self.forward = Vec3[.float32, .WORLD](
            fields.unsafe_offset(Self.FORWARD)
        )
        self.right = Vec3[.float32, .WORLD](fields.unsafe_offset(Self.RIGHT))
        self.up = Vec3[.float32, .WORLD](fields.unsafe_offset(Self.UP))
        self.fov_scale = fields[unsafe_offset=Self.FOV]
        self.focus_dist = fields[unsafe_offset=Self.FOCUS_DIST]
        self.defocus_disk_u = Vec3[.float32, .WORLD](
            fields.unsafe_offset(Self.DEFOCUS_DISK_U)
        )
        self.defocus_disk_v = Vec3[.float32, .WORLD](
            fields.unsafe_offset(Self.DEFOCUS_DISK_V)
        )

    def __init__(
        out self,
        origin: Point3[.float32, .WORLD],
        target: Point3[.float32, .WORLD],
        world_up: Vec3[.float32, .WORLD],
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
        origin: Point3[.float32, .WORLD],
        target: Point3[.float32, .WORLD],
        world_up: Vec3[.float32, .WORLD],
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

    @staticmethod
    def from_yaw_pitch(
        origin: Point3[.float32, .WORLD],
        yaw_degrees: Float32,
        pitch_degrees: Float32,
        vfov: Float32,
        focus_dist: Float32 = 1.0,
    ) -> Self:
        """Construct a Y-up camera; zero yaw/pitch faces world -Z."""
        var yaw = degrees_to_radians(yaw_degrees)
        var pitch = degrees_to_radians(pitch_degrees)
        var cos_pitch = cos(pitch)
        var forward = Vec3[.float32, .WORLD](
            sin(yaw) * cos_pitch, sin(pitch), -cos(yaw) * cos_pitch
        )
        return Self.from_vfov(
            origin,
            origin + forward,
            Vec3[.float32, .WORLD](0, 1, 0),
            vfov,
            focus_dist,
        )

    def vertical_fov_degrees(self) -> Float32:
        return radians_to_degrees[.float32, 1](2 * atan(self.fov_scale))

    def fit_fov_to_film(mut self, width: Int, height: Int):
        """Interpret the current FOV as the narrower film dimension's FOV."""
        if width < height:
            self.fov_scale *= Float32(height) / Float32(width)

    def set_lens_radius(mut self, radius: Float32):
        self.defocus_disk_u = self.right * radius
        self.defocus_disk_v = self.up * radius

    def camera_to_world(self) -> Affine3[.float32, .CAMERA, .WORLD]:
        """Return the authored basis with camera -Z facing forward."""
        return Affine3[.float32, .CAMERA, .WORLD](
            self.right.x,
            self.up.x,
            -self.forward.x,
            self.origin.x,
            self.right.y,
            self.up.y,
            -self.forward.y,
            self.origin.y,
            self.right.z,
            self.up.z,
            -self.forward.z,
            self.origin.z,
        )

    def world_to_camera(self) raises -> Affine3[.float32, .WORLD, .CAMERA]:
        var inverse = self.camera_to_world().inverse()
        if not inverse.mask[0] or not inverse.inv.is_finite()[0]:
            raise Error("camera basis must be finite and invertible")
        return inverse.inv.copy()

    @staticmethod
    def from_world_to_camera(
        view: Affine3[.float32, .WORLD, .CAMERA],
        vfov: Float32,
        focus_dist: Float32 = 1.0,
        *,
        positive_z_forward: Bool = False,
    ) raises -> Self:
        """Construct a camera from a finite, invertible authored view matrix.

        Set positive_z_forward for conventions whose camera forward axis is +Z.
        Basis scale, shear and handedness are preserved.
        """
        var inverse = view.inverse()
        if not inverse.mask[0] or not inverse.inv.is_finite()[0]:
            raise Error("camera transform must be invertible and finite")
        var origin = inverse.inv.point(Point3[.float32, .CAMERA](0))
        var right = inverse.inv.vector(Vec3[.float32, .CAMERA](1, 0, 0))
        var up = inverse.inv.vector(Vec3[.float32, .CAMERA](0, 1, 0))
        var forward = inverse.inv.vector(
            Vec3[.float32, .CAMERA](
                0, 0, Float32(1 if positive_z_forward else -1)
            )
        )
        return Self(
            origin,
            forward,
            right,
            up,
            tan(degrees_to_radians(vfov) / 2),
            focus_dist,
            Vec3[.float32, .WORLD](0),
            Vec3[.float32, .WORLD](0),
        )

    def rotate(
        mut self, yaw: Float32, pitch: Float32, vertical_limit: Float32 = 0.9998
    ):
        """Rotate in radians while preserving handedness, roll, lens size and FOV.
        """
        var handedness: Float32 = (
            -1 if dot(cross(self.right, self.forward), self.up) < 0 else 1
        )
        var angle = yaw
        var c = cos(angle)
        var s = sin(angle)
        var forward = Vec3[.float32, .WORLD](
            c * self.forward.x + s * self.forward.z,
            self.forward.y,
            -s * self.forward.x + c * self.forward.z,
        )
        var right = Vec3[.float32, .WORLD](
            c * self.right.x + s * self.right.z,
            self.right.y,
            -s * self.right.x + c * self.right.z,
        )
        angle = pitch * handedness
        var candidate = forward * cos(angle) + cross(right, forward) * sin(
            angle
        )
        if abs(candidate.y) < vertical_limit:
            forward = candidate
        var lens_u = length(self.defocus_disk_u)
        var lens_v = length(self.defocus_disk_v)
        self.forward = normalize(forward)
        self.right = normalize(right)
        self.up = normalize(cross(self.right, self.forward)) * handedness
        self.defocus_disk_u = self.right * lens_u
        self.defocus_disk_v = self.up * lens_v

    def make_ray(
        self,
        px_i: Int,
        py_i: Int,
        width: Int,
        height: Int,
    ) -> Ray[.float32, .WORLD]:
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
    ) -> Ray[.float32, .WORLD]:
        var screen_x = (
            2.0 * (Float32(px_i) + 0.5) - Float32(width)
        ) * inv_height
        var screen_y = 1.0 - 2.0 * (Float32(py_i) + 0.5) * inv_height

        var direction = normalize(
            self.forward
            + self.right * (screen_x * self.fov_scale)
            + self.up * (screen_y * self.fov_scale)
        )
        return Ray[.float32, .WORLD](
            self.origin, direction, 0.0, max_finite[.float32]()
        )

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
        var origin = self.origin.broadcast[length]()
        var forward = self.forward.broadcast[length]()
        var right = self.right.broadcast[length]()
        var up = self.up.broadcast[length]()
        var disk_u = self.defocus_disk_u.broadcast[length]()
        var disk_v = self.defocus_disk_v.broadcast[length]()
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
            max_finite[.float32](),
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
