from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.math import asin, atan2, cos, pi, sin, sqrt

from bajo.core.vec import (
    Vec3,
    normalize as vnormalize,
    length as vlength,
)
from bajo.core.mat import Mat


def length2[
    dtype: DType,
    width: Int,
](q: Quaternion[dtype, width]) -> SIMD[dtype, width]:
    return q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w


def length[
    dtype: DType,
    width: Int,
](q: Quaternion[dtype, width]) -> SIMD[dtype, width]:
    return sqrt(length2(q))


def normalize[
    dtype: DType,
    width: Int,
](q: Quaternion[dtype, width]) -> Quaternion[dtype, width]:
    return q * (1.0 / length(q))


comptime Quat = Quaternion[DType.float32]


@fieldwise_init
struct Quaternion[dtype: DType, width: Int = 1](
    DevicePassable, TrivialRegisterPassable, Writable
):
    var x: SIMD[Self.dtype, Self.width]
    var y: SIMD[Self.dtype, Self.width]
    var z: SIMD[Self.dtype, Self.width]
    var w: SIMD[Self.dtype, Self.width]

    # DevicePassable stuff
    comptime device_type: AnyType = Self

    def _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self.copy()

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return String(
            t"Quaternion[{reflect[Scalar[Self.dtype]].name()},{Self.width}]"
        )

    def __init__(
        out self,
        xyz: Vec3[Self.dtype, Self.width],
        w: SIMD[Self.dtype, Self.width],
    ):
        self.x = xyz.x
        self.y = xyz.y
        self.z = xyz.z
        self.w = w

    @staticmethod
    def identity() -> Self:
        return Self(0, 0, 0, 1)

    def xyz(self) -> Vec3[Self.dtype, Self.width]:
        return Vec3[Self.dtype, Self.width](self.x, self.y, self.z)

    def conjugate(self) -> Self:
        return Self(-self.x, -self.y, -self.z, self.w)

    def inverse_unit(self) -> Self:
        return self.conjugate()

    def inverse(self) -> Self:
        return self.conjugate() * (1.0 / length2(self))

    def __eq__(self, rhs: Self) -> SIMD[DType.bool, Self.width]:
        return (
            self.x.eq(rhs.x)
            & self.y.eq(rhs.y)
            & self.z.eq(rhs.z)
            & self.w.eq(rhs.w)
        )

    def rotate(
        self, v: Vec3[Self.dtype, Self.width]
    ) -> Vec3[Self.dtype, Self.width]:
        var c = 2.0 * self.w * self.w - 1.0
        var d = 2.0 * (self.x * v.x + self.y * v.y + self.z * v.z)
        var w2 = 2.0 * self.w

        var rx = v.x * c + self.x * d + (self.y * v.z - self.z * v.y) * w2
        var ry = v.y * c + self.y * d + (self.z * v.x - self.x * v.z) * w2
        var rz = v.z * c + self.z * d + (self.x * v.y - self.y * v.x) * w2

        return Vec3[Self.dtype, Self.width](rx, ry, rz)

    def rotate_inverse(
        self, v: Vec3[Self.dtype, Self.width]
    ) -> Vec3[Self.dtype, Self.width]:
        var c = 2.0 * self.w * self.w - 1.0
        var d = 2.0 * (self.x * v.x + self.y * v.y + self.z * v.z)
        var w2 = 2.0 * self.w

        var rx = v.x * c + self.x * d - (self.y * v.z - self.z * v.y) * w2
        var ry = v.y * c + self.y * d - (self.z * v.x - self.x * v.z) * w2
        var rz = v.z * c + self.z * d - (self.x * v.y - self.y * v.x) * w2

        return Vec3[Self.dtype, Self.width](rx, ry, rz)

    @staticmethod
    def from_axis_angle(
        axis: Vec3[Self.dtype, Self.width],
        angle: SIMD[Self.dtype, Self.width],
    ) -> Self where Self.dtype.is_floating_point():
        half_angle = angle * 0.5
        w = cos(half_angle)
        xyz = axis * sin(half_angle)
        return Self(xyz, w)

    def to_axis_angle(
        self,
    ) -> Tuple[
        Vec3[Self.dtype, Self.width],
        SIMD[Self.dtype, Self.width],
    ] where Self.dtype.is_floating_point():
        v = self.xyz()
        sign = self.w.lt(0).select(Scalar[Self.dtype](-1.0), 1.0)
        axis = vnormalize(v) * sign
        angle = 2.0 * atan2(vlength(v), abs(self.w))
        return (axis, angle)

    def to_matrix(self) -> Mat[Self.dtype, 3, 3, Self.width]:
        c0 = self.rotate(Vec3[Self.dtype, Self.width](1, 0, 0))
        c1 = self.rotate(Vec3[Self.dtype, Self.width](0, 1, 0))
        c2 = self.rotate(Vec3[Self.dtype, Self.width](0, 0, 1))
        return Mat[Self.dtype, 3, 3, Self.width].from_cols(c0, c1, c2)

    @staticmethod
    def _from_rotation_matrix_elements(
        m00: SIMD[Self.dtype, Self.width],
        m01: SIMD[Self.dtype, Self.width],
        m02: SIMD[Self.dtype, Self.width],
        m10: SIMD[Self.dtype, Self.width],
        m11: SIMD[Self.dtype, Self.width],
        m12: SIMD[Self.dtype, Self.width],
        m20: SIMD[Self.dtype, Self.width],
        m21: SIMD[Self.dtype, Self.width],
        m22: SIMD[Self.dtype, Self.width],
    ) -> Self where Self.dtype.is_floating_point():
        # traces = 4 * component^2.
        var trace_x = 1.0 + m00 - m11 - m22
        var trace_y = 1.0 - m00 + m11 - m22
        var trace_z = 1.0 - m00 - m11 + m22
        var trace_w = 1.0 + m00 + m11 + m22

        var best_trace = trace_w

        var use_x = trace_x.gt(best_trace)
        best_trace = use_x.select(trace_x, best_trace)

        var use_y = trace_y.gt(best_trace)
        best_trace = use_y.select(trace_y, best_trace)

        var use_z = trace_z.gt(best_trace)
        best_trace = use_z.select(trace_z, best_trace)

        var v = sqrt(best_trace) * 0.5
        var mult = 0.25 / v

        # w is largest
        var out_x = (m21 - m12) * mult
        var out_y = (m02 - m20) * mult
        var out_z = (m10 - m01) * mult
        var out_w = v

        # x is largest
        out_x = use_x.select(v, out_x)
        out_y = use_x.select((m01 + m10) * mult, out_y)
        out_z = use_x.select((m02 + m20) * mult, out_z)
        out_w = use_x.select((m21 - m12) * mult, out_w)

        # y is largest
        out_x = use_y.select((m01 + m10) * mult, out_x)
        out_y = use_y.select(v, out_y)
        out_z = use_y.select((m12 + m21) * mult, out_z)
        out_w = use_y.select((m02 - m20) * mult, out_w)

        # z is largest
        out_x = use_z.select((m02 + m20) * mult, out_x)
        out_y = use_z.select((m12 + m21) * mult, out_y)
        out_z = use_z.select(v, out_z)
        out_w = use_z.select((m10 - m01) * mult, out_w)

        return normalize(Self(out_x, out_y, out_z, out_w))

    @staticmethod
    def from_matrix[
        rows: Int,
        cols: Int,
    ](m: Mat[Self.dtype, rows, cols, Self.width]) -> Self where (
        (rows == cols) and (rows == 3) and (Self.dtype.is_floating_point())
    ):
        return Self._from_rotation_matrix_elements(
            m[0][0],
            m[0][1],
            m[0][2],
            m[1][0],
            m[1][1],
            m[1][2],
            m[2][0],
            m[2][1],
            m[2][2],
        )

    @staticmethod
    def from_basis[
        version: Int = 0,
    ](
        a: Vec3[Self.dtype, Self.width],
        b: Vec3[Self.dtype, Self.width],
        c: Vec3[Self.dtype, Self.width],
    ) -> Self where Self.dtype.is_floating_point():
        return Self._from_rotation_matrix_elements(
            a.x,
            b.x,
            c.x,
            a.y,
            b.y,
            c.y,
            a.z,
            b.z,
            c.z,
        )

    comptime from_rpy = Self.from_euler

    @staticmethod
    def from_euler(
        roll: SIMD[Self.dtype, Self.width],
        pitch: SIMD[Self.dtype, Self.width],
        yaw: SIMD[Self.dtype, Self.width],
    ) -> Self where Self.dtype.is_floating_point():
        """
        Converts Euler angles (Roll, Pitch, Yaw) to a Quaternion.
        Order: Z (Yaw) -> Y (Pitch) -> X (Roll).
        """
        cr = cos(roll * 0.5)
        sr = sin(roll * 0.5)
        cp = cos(pitch * 0.5)
        sp = sin(pitch * 0.5)
        cy = cos(yaw * 0.5)
        sy = sin(yaw * 0.5)

        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        w = cr * cp * cy + sr * sp * sy

        return Self(x, y, z, w)

    comptime to_rpy = Self.to_euler

    def to_euler(
        self,
    ) -> Vec3[Self.dtype, Self.width] where Self.dtype.is_floating_point():
        var sinr_cosp = 2.0 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1 - 2 * (self.x * self.x + self.y * self.y)
        roll = atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (self.w * self.y - self.z * self.x)
        abs_sinp = abs(sinp)
        comptime pi_2 = Scalar[Self.dtype](pi / 2.0)
        pitch_sat = sinp.lt(0.0).select(-pi_2, pi_2)
        saturated = abs_sinp.gt(1.0) | abs_sinp.eq(1.0)
        pitch = saturated.select(pitch_sat, asin(sinp))

        siny_cosp = 2 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1 - 2 * (self.y * self.y + self.z * self.z)
        yaw = atan2(siny_cosp, cosy_cosp)

        return Vec3[Self.dtype, Self.width](roll, pitch, yaw)

    def __neg__(self) -> Self:
        return Self(-self.x, -self.y, -self.z, -self.w)

    def __add__(self, o: Self) -> Self:
        return Self(self.x + o.x, self.y + o.y, self.z + o.z, self.w + o.w)

    def __iadd__(mut self, o: Self):
        self.x += o.x
        self.y += o.y
        self.z += o.z
        self.w += o.w

    def __sub__(self, o: Self) -> Self:
        return Self(self.x - o.x, self.y - o.y, self.z - o.z, self.w - o.w)

    def __isub__(mut self, o: Self):
        self.x -= o.x
        self.y -= o.y
        self.z -= o.z
        self.w -= o.w

    def __mul__(self, b: Self) -> Self:
        # hamilton product, simplest version
        # might optimize this later
        return Self(
            self.w * b.x + self.x * b.w + self.y * b.z - self.z * b.y,
            self.w * b.y - self.x * b.z + self.y * b.w + self.z * b.x,
            self.w * b.z + self.x * b.y - self.y * b.x + self.z * b.w,
            self.w * b.w - self.x * b.x - self.y * b.y - self.z * b.z,
        )

    def __mul__(self, s: SIMD[Self.dtype, Self.width]) -> Self:
        return Self(self.x * s, self.y * s, self.z * s, self.w * s)

    def __rmul__(self, s: SIMD[Self.dtype, Self.width]) -> Self:
        return Self(self.x * s, self.y * s, self.z * s, self.w * s)

    def __imul__(mut self, s: SIMD[Self.dtype, Self.width]):
        self.x *= s
        self.y *= s
        self.z *= s
        self.w *= s


def slerp[
    dtype: DType,
    width: Int,
](
    q0: Quaternion[dtype, width],
    q1: Quaternion[dtype, width],
    t: SIMD[dtype, width],
) -> Quaternion[dtype, width] where dtype.is_floating_point():
    """Spherical Linear Interpolation."""
    var axis_angle = (q0.inverse_unit() * q1).to_axis_angle()

    return q0 * Quaternion[dtype, width].from_axis_angle(
        axis_angle[0],
        t * axis_angle[1],
    )
