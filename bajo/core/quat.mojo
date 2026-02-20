from math import acos, asin, atan2, clamp, cos, fma, pi, sin, sqrt, tan

from bajo.core.vec import Vec3, cross


fn length2[
    type: DType where type.is_floating_point()
](q: Quaternion[type]) -> Scalar[type]:
    return (q.data * q.data).reduce_add()


fn length[
    type: DType where type.is_floating_point()
](q: Quaternion[type]) -> Scalar[type]:
    return sqrt(length2(q))


fn inv_length[
    type: DType where type.is_floating_point()
](q: Quaternion[type]) -> Scalar[type]:
    return 1.0 / length(q)


fn normalize[
    type: DType where type.is_floating_point()
](q: Quaternion[type]) -> Quaternion[type]:
    return Quaternion[type](q.data * inv_length(q))


comptime Quat = Quaternion[DType.float32]


@fieldwise_init
struct Quaternion[type: DType where type.is_floating_point()](
    Copyable, Equatable, TrivialRegisterPassable, Writable
):
    var data: SIMD[Self.type, 4]

    fn __init__(
        out self,
        x: Scalar[Self.type],
        y: Scalar[Self.type],
        z: Scalar[Self.type],
        w: Scalar[Self.type],
    ):
        self.data = SIMD[Self.type, 4](x, y, z, w)

    @staticmethod
    fn identity() -> Self:
        return Self([1, 0, 0, 0])

    fn w(self) -> Scalar[Self.type]:
        return self.data[0]

    fn x(self) -> Scalar[Self.type]:
        return self.data[1]

    fn y(self) -> Scalar[Self.type]:
        return self.data[2]

    fn z(self) -> Scalar[Self.type]:
        return self.data[3]

    fn xyz(self) -> Vec3[Self.type]:
        return Vec3[Self.type](self.x(), self.y(), self.z())

    fn inv(self) -> Self:
        return Self(self.w(), -self.x(), -self.y(), -self.z())

    fn rotate_vec(self, v: Vec3[Self.type]) -> Vec3[Self.type]:
        var pure = self.xyz()
        var scalar = self.w()
        var pure_x_v = cross(pure, v)
        var pure_x_pure_x_v = cross(pure, pure_x_v)

        return v + (pure_x_v * scalar + pure_x_pure_x_v) * 2.0

    @staticmethod
    fn angle_axis(angle: Scalar[Self.type], normal: Vec3[Self.type]) -> Self:
        var half_angle = angle / 2.0
        var w = cos(half_angle)
        var xyz = normal * sin(half_angle)
        return Self(w, xyz.x(), xyz.y(), xyz.z())

    @staticmethod
    fn from_basis[
        version: Int = 0
    ](a: Vec3[Self.type], b: Vec3[Self.type], c: Vec3[Self.type]) -> Self:
        @parameter
        if version == 0:
            var four_x_sq_m1 = a.x() - b.y() - c.z()
            var four_y_sq_m1 = b.y() - a.x() - c.z()
            var four_z_sq_m1 = c.z() - a.x() - b.y()
            var four_w_sq_m1 = a.x() + b.y() + c.z()

            var biggest_index = 0
            var four_biggest_sq_m1 = four_w_sq_m1

            if four_x_sq_m1 > four_biggest_sq_m1:
                four_biggest_sq_m1 = four_x_sq_m1
                biggest_index = 1

            if four_y_sq_m1 > four_biggest_sq_m1:
                four_biggest_sq_m1 = four_y_sq_m1
                biggest_index = 2

            if four_z_sq_m1 > four_biggest_sq_m1:
                four_biggest_sq_m1 = four_z_sq_m1
                biggest_index = 3

            var biggest_val = sqrt(four_biggest_sq_m1 + 1.0) * 0.5
            var mult = 0.25 / biggest_val

            var w: Scalar[Self.type]
            var x: Scalar[Self.type]
            var y: Scalar[Self.type]
            var z: Scalar[Self.type]
            if biggest_index == 0:
                w = biggest_val
                x = (b.z() - c.y()) * mult
                y = (c.x() - a.z()) * mult
                z = (a.y() - b.x()) * mult
            elif biggest_index == 1:
                w = (b.z() - c.y()) * mult
                x = biggest_val
                y = (a.y() + b.x()) * mult
                z = (c.x() + a.z()) * mult
            elif biggest_index == 2:
                w = (c.x() - a.z()) * mult
                x = (a.y() + b.x()) * mult
                y = biggest_val
                z = (b.z() + c.y()) * mult
            else:
                w = (a.y() - b.x()) * mult
                x = (c.x() + a.z()) * mult
                y = (b.z() + c.y()) * mult
                z = biggest_val
            return Self(w, x, y, z)

        else:
            comptime S = SIMD[Self.type, 4]

            # We compute all 4 traces (4w^2, 4x^2, 4y^2, 4z^2)
            var diag = S(a.x(), b.y(), c.z(), 0.0)
            comptime s_w = S(1, 1, 1, 1)
            comptime s_x = S(1, -1, -1, 1)
            comptime s_y = S(-1, 1, -1, 1)
            comptime s_z = S(-1, -1, 1, 1)

            # traces = [w_trace, x_trace, y_trace, z_trace]
            var traces = (
                1.0 + (diag[0] * s_x) + (diag[1] * s_y) + (diag[2] * s_z)
            )
            var max_trace = traces.reduce_max()

            # compute the chosen component value and multiplier
            var v = sqrt(max_trace) * 0.5
            var mult = 0.25 / v

            # Bulk Off-Diagonal Arithmetic
            # indices: m_rc where r=row, c=col.
            # m21 is a.z(), m02 is c.x(), m10 is b.x() etc.
            var m_left = S(b.z(), c.x(), a.y(), 0.0)  # [m12, m20, m01, 0]
            var m_right = S(c.y(), a.z(), b.x(), 0.0)  # [m21, m02, m10, 0]

            # d = [ (m21-m12), (m02-m20), (m10-m01), 0 ]
            # s = [ (m21+m12), (m02+m20), (m10+m01), 0 ]
            var diff = (m_right - m_left) * mult
            var sum = (m_right + m_left) * mult

            # construct candidates
            var qw = S(v, diff[0], diff[1], diff[2])
            var qx = S(diff[0], v, sum[2], sum[1])
            var qy = S(diff[1], sum[2], v, sum[0])
            var qz = S(diff[2], sum[1], sum[0], v)

            # branchless Selection
            var mask = traces.eq(max_trace)
            var res = qz
            res = SIMD[DType.bool, 4](mask[2]).select(qy, res)
            res = SIMD[DType.bool, 4](mask[1]).select(qx, res)
            res = SIMD[DType.bool, 4](mask[0]).select(qw, res)

            return Self(res)

    @staticmethod
    fn from_euler(
        roll: Scalar[Self.type],
        pitch: Scalar[Self.type],
        yaw: Scalar[Self.type],
    ) -> Self:
        """
        Converts Euler angles (Roll, Pitch, Yaw) to a Quaternion.
        Order: Z (Yaw) -> Y (Pitch) -> X (Roll).
        """
        var cr = cos(roll * 0.5)
        var sr = sin(roll * 0.5)
        var cp = cos(pitch * 0.5)
        var sp = sin(pitch * 0.5)
        var cy = cos(yaw * 0.5)
        var sy = sin(yaw * 0.5)

        var w = cr * cp * cy + sr * sp * sy
        var x = sr * cp * cy - cr * sp * sy
        var y = cr * sp * cy + sr * cp * sy
        var z = cr * cp * sy - sr * sp * cy

        return Self(w, x, y, z)

    fn to_euler(self) -> Vec3[Self.type]:
        """
        Converts Quaternion back to Euler angles (Roll, Pitch, Yaw).
        """
        var sinr_cosp = 2 * (self.w() * self.x() + self.y() * self.z())
        var cosr_cosp = 1 - 2 * (self.x() * self.x() + self.y() * self.y())
        var roll = atan2(sinr_cosp, cosr_cosp)

        var sinp = 2 * (self.w() * self.y() - self.z() * self.x())
        var pitch: Scalar[Self.type]
        if abs(sinp) >= 1:
            comptime pi_2 = pi / 2.0
            pitch = Scalar[Self.type](1.0 if sinp > 0 else -1.0) * pi_2
        else:
            pitch = asin(sinp)

        var siny_cosp = 2 * (self.w() * self.z() + self.x() * self.y())
        var cosy_cosp = 1 - 2 * (self.y() * self.y() + self.z() * self.z())
        var yaw = atan2(siny_cosp, cosy_cosp)

        return Vec3[Self.type](roll, pitch, yaw)

    fn __add__(self, o: Self) -> Self:
        return Self(self.data + o.data)

    fn __iadd__(mut self, o: Self):
        self.data += o.data

    fn __sub__(self, o: Self) -> Self:
        return Self(self.data - o.data)

    fn __isub__(mut self, o: Self):
        self.data -= o.data

    @always_inline
    fn __mul__(self, b: Self) -> Self:
        comptime s1 = SIMD[Self.type, 4](-1.0, 1.0, -1.0, 1.0)
        comptime s2 = SIMD[Self.type, 4](-1.0, 1.0, 1.0, -1.0)
        comptime s3 = SIMD[Self.type, 4](-1.0, -1.0, 1.0, 1.0)

        # two independent branches to maximize Instruction Level Parallelism (ILP).
        # Branch A
        var a_w = self.data[0]
        var a_x_signed = self.data[1] * s1
        var res_a = fma(a_x_signed, b.data.shuffle[1, 0, 3, 2](), a_w * b.data)

        # Branch B
        var a_y_signed = self.data[2] * s2
        var a_z_signed = self.data[3] * s3
        var res_b = fma(
            a_y_signed,
            b.data.shuffle[2, 3, 0, 1](),
            a_z_signed * b.data.shuffle[3, 2, 1, 0](),
        )

        return Self(res_a + res_b)

    fn __mul__(self, f: Scalar[Self.type]) -> Self:
        return Self(self.data * f)

    fn __imul__(mut self, f: Scalar[Self.type]):
        self.data *= f
