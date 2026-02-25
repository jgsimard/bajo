from math import asin, atan2, cos, fma, pi, sin, sqrt, tan, copysign

from bajo.core.vec import (
    Vec3,
    cross,
    normalize as vnormalize,
    length as vlength,
)
from bajo.core.mat import Mat, Mat33, Mat44


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

    fn inverse(self) -> Self:
        return Self(-self.x(), -self.y(), -self.z(), self.w())

    # TODO: benchmark the two version to choose the fastest
    fn rotate(self, v: Vec3[Self.type]) -> Vec3[Self.type]:
        # p = self.xyz()
        # w = self.w()
        # pv = cross(p, v)
        # ppv = cross(p, pv)
        # return v + (pv * w + ppv) * 2.0
        qw = self.w()
        qx = self.x()
        qy = self.y()
        qz = self.z()

        vx = v[0]
        vy = v[1]
        vz = v[2]

        c = 2 * qw * qw - 1
        d = 2 * (qx * vx + qy * vy + qz * vz)
        w2 = 2 * qw

        rx = vx * c + qx * d + (qy * vz - qz * vy) * w2
        ry = vy * c + qy * d + (qz * vx - qx * vz) * w2
        rz = vz * c + qz * d + (qx * vy - qy * vx) * w2

        return Vec3[Self.type](rx, ry, rz)

    fn rotate_inverse(self, v: Vec3[Self.type]) -> Vec3[Self.type]:
        qw = self.w()
        qx = self.x()
        qy = self.y()
        qz = self.z()

        vx = v[0]
        vy = v[1]
        vz = v[2]

        c = 2 * qw * qw - 1
        d = 2 * (qx * vx + qy * vy + qz * vz)
        w2 = 2 * qw

        # Calcul des composantes avec le signe '-' pour le produit vectoriel
        # Cela correspond à tourner par le quaternion conjugué (inverse)
        rx = vx * c + qx * d - (qy * vz - qz * vy) * w2
        ry = vy * c + qy * d - (qz * vx - qx * vz) * w2
        rz = vz * c + qz * d - (qx * vy - qy * vx) * w2

        return Vec3[Self.type](rx, ry, rz)

    @staticmethod
    fn from_axis_angle(axis: Vec3[Self.type], angle: Scalar[Self.type]) -> Self:
        half_angle = angle * 0.5
        w = cos(half_angle)
        xyz = axis * sin(half_angle)
        return Self(xyz, w)

    fn to_axis_angle(self) -> Tuple[Vec3[Self.type], Scalar[Self.type]]:
        v = self.xyz()
        axis = vnormalize(v) * copysign(Scalar[Self.type](1.0), self.w())
        angle = 2 * atan2(vlength(v), abs(self.w()))
        return (axis^, angle)

    fn to_matrix(self) -> Mat33[Self.type]:
        c1 = self.rotate(Vec3[Self.type](1, 0, 0))
        c2 = self.rotate(Vec3[Self.type](0, 1, 0))
        c3 = self.rotate(Vec3[Self.type](0, 0, 1))
        return Mat33[Self.type].from_cols(c1, c2, c3)

    @staticmethod
    fn from_matrix[
        rows: Int where rows >= 1, cols: Int where cols >= 1
    ](m: Mat[Self.type, rows, cols]) -> Self where (rows == cols) and (
        rows == 3 or rows == 4
    ):
        """Only accepts 3x3 and 4x4 matrices."""

        comptime zero = Scalar[Self.type](0)
        comptime one = Scalar[Self.type](1)
        comptime half = Scalar[Self.type](0.5)
        tr = m[0][0] + m[1][1] + m[2][2]
        x: Scalar[Self.type]
        y: Scalar[Self.type]
        z: Scalar[Self.type]
        w: Scalar[Self.type]
        h: Scalar[Self.type]

        if tr >= zero:
            h = sqrt(tr + 1)
            w = half * h
            h = half / h

            x = (m[2][1] - m[1][2]) * h
            y = (m[0][2] - m[2][0]) * h
            z = (m[1][0] - m[0][1]) * h
        else:
            max_diag = 0
            if m[1][1] > m[0][0]:
                max_diag = 1
            if m[2][2] > m[max_diag][max_diag]:
                max_diag = 2

            if max_diag == 0:
                h = sqrt((m[0][0] - (m[1][1] + m[2][2])) + one)
                x = half * h
                h = half / h

                y = (m[0][1] + m[1][0]) * h
                z = (m[2][0] + m[0][2]) * h
                w = (m[2][1] - m[1][2]) * h
            elif max_diag == 1:
                h = sqrt((m[1][1] - (m[2][2] + m[0][0])) + one)
                y = half * h
                h = half / h

                z = (m[1][2] + m[2][1]) * h
                x = (m[0][1] + m[1][0]) * h
                w = (m[0][2] - m[2][0]) * h
            else:  # max_diag == 2
                h = sqrt((m[2][2] - (m[0][0] + m[1][1])) + one)
                z = half * h
                h = half / h

                x = (m[2][0] + m[0][2]) * h
                y = (m[1][2] + m[2][1]) * h
                w = (m[1][0] - m[0][1]) * h

        return normalize(Self(x, y, z, w))

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

    comptime from_rpy = Self.from_euler

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

    comptime to_rpy = Self.to_euler

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

    fn __imul__(mut self, s: Scalar[Self.type]):
        self.data *= s


fn slerp[
    dtype: DType where dtype.is_floating_point()
](q0: Quaternion[dtype], q1: Quaternion[dtype], t: Scalar[dtype]) -> Quaternion[
    dtype
]:
    """Spherical Linear Interpolation."""
    axis_angle = (q0.inverse() * q1).to_axis_angle()
    return q0 * Quaternion.from_axis_angle(axis_angle[0], t * axis_angle[1])


fn main():
    print("hello bajo.core.quat")
    comptime T = DType.float32
    m = Mat[T, 4, 4](1)
    q = Quaternion.from_matrix(m)
    print(q)
