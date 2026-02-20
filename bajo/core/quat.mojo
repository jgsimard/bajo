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
    var data: SIMD[Self.type, 4]  # layout: [x, y, z, w]

    fn __init__(
        out self,
        x: Scalar[Self.type],
        y: Scalar[Self.type],
        z: Scalar[Self.type],
        w: Scalar[Self.type],
    ):
        self.data = SIMD[Self.type, 4](x, y, z, w)

    fn __init__(
        out self,
        xyz: Vec3[Self.type],
        w: Scalar[Self.type],
    ):
        self.data = SIMD[Self.type, 4](xyz.x(), xyz.y(), xyz.z(), w)

    @staticmethod
    fn identity() -> Self:
        return Self([0, 0, 0, 1])

    fn x(self) -> Scalar[Self.type]:
        return self.data[0]

    fn y(self) -> Scalar[Self.type]:
        return self.data[1]

    fn z(self) -> Scalar[Self.type]:
        return self.data[2]

    fn w(self) -> Scalar[Self.type]:
        return self.data[3]

    fn xyz(self) -> Vec3[Self.type]:
        return Vec3[Self.type](self.x(), self.y(), self.z())

    fn inv(self) -> Self:
        return Self(-self.x(), -self.y(), -self.z(), self.w())

    fn rotate_vec(self, v: Vec3[Self.type]) -> Vec3[Self.type]:
        p = self.xyz()
        w = self.w()
        pv = cross(p, v)
        ppv = cross(p, pv)

        return v + (pv * w + ppv) * 2.0

    @staticmethod
    fn angle_axis(angle: Scalar[Self.type], normal: Vec3[Self.type]) -> Self:
        half_angle = angle / 2.0
        w = cos(half_angle)
        xyz = normal * sin(half_angle)
        return Self(xyz, w)

    @staticmethod
    fn from_basis[
        version: Int = 0
    ](a: Vec3[Self.type], b: Vec3[Self.type], c: Vec3[Self.type]) -> Self:
        @parameter
        if version == 0:
            # tx, ty, tz, tw represent 4*q^2 - 1 for each component
            tx = a.x() - b.y() - c.z()
            ty = b.y() - a.x() - c.z()
            tz = c.z() - a.x() - b.y()
            tw = a.x() + b.y() + c.z()

            biggest_index = 0
            max_val = tw

            if tx > max_val:
                max_val = tx
                biggest_index = 1

            if ty > max_val:
                max_val = ty
                biggest_index = 2

            if tz > max_val:
                max_val = tz
                biggest_index = 3

            v = sqrt(max_val + 1.0) * 0.5
            mult = 0.25 / v

            if biggest_index == 0:  # W is biggest
                x = (b.z() - c.y()) * mult
                y = (c.x() - a.z()) * mult
                z = (a.y() - b.x()) * mult
                w = v
            elif biggest_index == 1:  # X is biggest
                x = v
                y = (a.y() + b.x()) * mult
                z = (c.x() + a.z()) * mult
                w = (b.z() - c.y()) * mult
            elif biggest_index == 2:  # Y is biggest
                x = (a.y() + b.x()) * mult
                y = v
                z = (b.z() + c.y()) * mult
                w = (c.x() - a.z()) * mult
            else:  # Z is biggest
                x = (c.x() + a.z()) * mult
                y = (b.z() + c.y()) * mult
                z = v
                w = (a.y() - b.x()) * mult
            return Self(x, y, z, w)

        else:
            comptime S = SIMD[Self.type, 4]

            # We compute all 4 traces (4w^2, 4x^2, 4y^2, 4z^2)
            diag = S(a.x(), b.y(), c.z(), 0.0)
            comptime s_x = S(1, -1, -1, 1)
            comptime s_y = S(-1, 1, -1, 1)
            comptime s_z = S(-1, -1, 1, 1)

            # traces = [x_trace, y_trace, z_trace, w_trace]
            traces = 1.0 + (diag[0] * s_x) + (diag[1] * s_y) + (diag[2] * s_z)
            max_trace = traces.reduce_max()

            # compute the chosen component value and multiplier
            v = sqrt(max_trace) * 0.5
            mult = 0.25 / v

            # Bulk Off-Diagonal Arithmetic
            # indices: m_rc where r=row, c=col.
            # m21 is a.z(), m02 is c.x(), m10 is b.x() etc.
            m_left = S(b.z(), c.x(), a.y(), 0.0)  # [m12, m20, m01, 0]
            m_right = S(c.y(), a.z(), b.x(), 0.0)  # [m21, m02, m10, 0]

            # d = [ (m21-m12), (m02-m20), (m10-m01), 0 ]
            # s = [ (m21+m12), (m02+m20), (m10+m01), 0 ]
            diff = (m_right - m_left) * mult
            sum = (m_right + m_left) * mult

            # construct candidates
            qx = S(v, diff[0], sum[1], sum[0])
            qy = S(sum[0], v, diff[2], sum[1])
            qz = S(sum[1], sum[2], v, diff[2])
            qw = S(diff[0], diff[1], diff[2], v)

            # branchless Selection
            mask = traces.eq(max_trace)
            res = qw
            res = SIMD[DType.bool, 4](mask[0]).select(qx, res)
            res = SIMD[DType.bool, 4](mask[1]).select(qy, res)
            res = SIMD[DType.bool, 4](mask[2]).select(qz, res)

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

    fn to_euler(self) -> Vec3[Self.type]:
        """
        Converts Quaternion back to Euler angles (Roll, Pitch, Yaw).
        """
        sinr_cosp = 2 * (self.w() * self.x() + self.y() * self.z())
        cosr_cosp = 1 - 2 * (self.x() * self.x() + self.y() * self.y())
        roll = atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (self.w() * self.y() - self.z() * self.x())
        pitch: Scalar[Self.type]
        if abs(sinp) >= 1:
            comptime pi_2 = pi / 2.0
            pitch = Scalar[Self.type](1.0 if sinp > 0 else -1.0) * pi_2
        else:
            pitch = asin(sinp)

        siny_cosp = 2 * (self.w() * self.z() + self.x() * self.y())
        cosy_cosp = 1 - 2 * (self.y() * self.y() + self.z() * self.z())
        yaw = atan2(siny_cosp, cosy_cosp)

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
        # Layout: [x, y, z, w]
        # Hamiltonian product formula derived for this layout:
        # res.x =  x1*w2 + y1*z2 - z1*y2 + w1*x2
        # res.y = -x1*z2 + y1*w2 + z1*x2 + w1*y2
        # res.z =  x1*y2 - y1*x2 + z1*w2 + w1*z2
        # res.w = -x1*x2 - y1*y2 - z1*z2 + w1*w2
        comptime s1 = SIMD[Self.type, 4](1.0, -1.0, 1.0, -1.0)
        comptime s2 = SIMD[Self.type, 4](1.0, 1.0, -1.0, -1.0)
        comptime s3 = SIMD[Self.type, 4](-1.0, 1.0, 1.0, -1.0)

        # two independent branches to maximize Instruction Level Parallelism (ILP).
        # Branch A
        a_w = self.data[3]
        a_x_signed = self.data[0] * s1
        # Shuffling B to [w2, z2, y2, x2]
        res_a = fma(a_x_signed, b.data.shuffle[3, 2, 1, 0](), a_w * b.data)

        # Branch B
        a_y_signed = self.data[1] * s2
        a_z_signed = self.data[2] * s3
        # Shuffle B for y1: [z2, w2, x2, y2]
        # Shuffle B for z1: [y2, x2, w2, z2]
        res_b = fma(
            a_y_signed,
            b.data.shuffle[2, 3, 0, 1](),
            a_z_signed * b.data.shuffle[1, 0, 3, 2](),
        )

        return Self(res_a + res_b)

    fn __mul__(self, f: Scalar[Self.type]) -> Self:
        return Self(self.data * f)

    fn __imul__(mut self, f: Scalar[Self.type]):
        self.data *= f
