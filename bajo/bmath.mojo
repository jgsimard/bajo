from math import pi, sqrt, sin, cos, tan, acos, atan2, clamp
from std.utils.numerics import max_finite, min_finite
from math import fma
from std.bit import next_power_of_two
from random import random_float64

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

comptime pi_d2 = pi / 2.0
comptime pi_d180 = pi / 180.0
comptime pi_m2 = pi * 2.0


comptime Vec2f = Vector2[DType.float32]
comptime Vec3f = Vector3[DType.float32]
comptime Vec4f = Vector4[DType.float32]

comptime Vec2i = Vector2[DType.int32]
comptime Vec3i = Vector3[DType.int32]

comptime Vec4b = Vector4[DType.uint8]


comptime Diag3f = Diag3x3[DType.float32]
comptime Mat3f = Mat3x3[DType.float32]
comptime Mat3x4f = Mat3x4[DType.float32]
comptime Mat4f = Mat4x4[DType.float32]

comptime Quat = Quaternion[DType.float32]

comptime AABB = AxisAlignedBoundingBox[DType.float32]


# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------
fn degrees_to_radians[
    type: DType, size: Int
](degrees: SIMD[type, size]) -> SIMD[type, size]:
    return degrees * pi_d180


@fieldwise_init
@register_passable("trivial")
struct QuadraticSolutions[type: DType, size: Int](Copyable):
    var roots_0: SIMD[Self.type, Self.size]
    var roots_1: SIMD[Self.type, Self.size]
    var mask: SIMD[DType.bool, Self.size]


fn solve_quadratic[
    type: DType, size: Int
](
    a: SIMD[type, size], b: SIMD[type, size], c: SIMD[type, size]
) -> QuadraticSolutions[type, size]:
    """Solves the quadratic equation ax^2 + bx + c = 0 element-wise for SIMD vectors.

    This function uses a numerically stable implementation (Citardauq's formula)
    to prevent catastrophic cancellation when 'b' is much larger than 'ac'.
    see: https://en.wikipedia.org/wiki/Quadratic_formula#Square_root_in_the_denominator

    Args:
        a: The quadratic coefficients.
        b: The linear coefficients.
        c: The constant terms.

    Returns:
        QuadraticSolutions[roots_0, roots_1, mask].
    """
    var det = b * b - 4.0 * a * c
    var mask = det.ge(0.0)  # Element-wise >= 0.0

    var sqrt_det = sqrt(max(det, 0.0))

    var q = -0.5 * (b + b.ge(0.0).select(sqrt_det, -sqrt_det))

    var t0 = q / a
    var t1 = c / q

    return QuadraticSolutions(t0, t1, mask)


# ----------------------------------------------------------------------
# Vector
# ----------------------------------------------------------------------
comptime Vector2[type: DType] = Vector[type, 2]
comptime Vector3[type: DType] = Vector[type, 3]
comptime Vector4[type: DType] = Vector[type, 4]


fn dot[t: DType, s: Int](a: Vector[t, s], b: Vector[t, s]) -> Scalar[t]:
    return (a.data * b.data).reduce_add()


fn length2[t: DType, s: Int](a: Vector[t, s]) -> Scalar[t]:
    return (a.data * a.data).reduce_add()


fn length[t: DType, s: Int](a: Vector[t, s]) -> Scalar[t]:
    return sqrt(length2(a))


fn inv_length[t: DType, s: Int](a: Vector[t, s]) -> Scalar[t]:
    return 1.0 / length(a)


fn normalize[t: DType, s: Int](a: Vector[t, s]) -> Vector[t, s]:
    return a * inv_length(a)


fn distance[t: DType, s: Int](a: Vector[t, s], b: Vector[t, s]) -> Scalar[t]:
    return length(a - b)


fn distance2[t: DType, s: Int](a: Vector[t, s], b: Vector[t, s]) -> Scalar[t]:
    return length2(a - b)


fn cross[type: DType](a: Vector3[type], b: Vector3[type]) -> Vector3[type]:
    return Vector3[type](
        a.y() * b.z() - a.z() * b.y(),
        a.z() * b.x() - a.x() * b.z(),
        a.x() * b.y() - a.y() * b.x(),
    )


fn cross[type: DType](a: Vector2[type], b: Vector2[type]) -> Scalar[type]:
    return a.x() * b.y() - a.y() * b.x()


fn lerp[
    t: DType, s: Int
](a: Vector[t, s], b: Vector[t, s], u: Scalar[t]) -> Vector[t, s]:
    return a * (Scalar[t](1.0) - u) + b * u


@fieldwise_init
@register_passable("trivial")
struct Vector[type: DType, size: Int](
    Copyable,
    Equatable,
    Powable,
    Stringable,
    Writable,
):
    """A wrapper around SIMD."""

    comptime storage_size = next_power_of_two(Self.size)
    comptime data_type = SIMD[Self.type, Self.storage_size]

    var data: Self.data_type

    fn __init__(out self, v: Scalar[Self.type]):
        self.data = Self.data_type(v)

    # ugly but it works, TODO: change this
    fn __init__(out self: Vec3f, v: Int):
        self.data = rebind[Vector[DType.float32, 3].data_type](
            SIMD[DType.float32, 4](Float32(v), Float32(v), Float32(v), 0)
        )

    fn __init__(out self: Vec3f, v: Float64):
        self.data = rebind[Vector[DType.float32, 3].data_type](
            SIMD[DType.float32, 4](Float32(v), Float32(v), Float32(v), 0)
        )

    fn __init__(
        out self,
        x: Scalar[Self.type],
        y: Scalar[Self.type],
    ):
        __comptime_assert Self.size == 2
        self.data = Self.data_type(x, y)

    fn __init__(
        out self,
        x: Scalar[Self.type],
        y: Scalar[Self.type],
        z: Scalar[Self.type],
    ):
        __comptime_assert Self.size == 3
        self.data = Self.data_type(x, y, z, Scalar[Self.type](0))

    fn __init__(
        out self,
        x: Scalar[Self.type],
        y: Scalar[Self.type],
        z: Scalar[Self.type],
        w: Scalar[Self.type],
    ):
        __comptime_assert Self.size == 4
        self.data = Self.data_type(x, y, z, w)

    @staticmethod
    fn zeros() -> Self:
        return Self(Scalar[Self.type](0))

    @staticmethod
    fn ones() -> Self:
        return Self(Scalar[Self.type](1))

    @staticmethod
    fn max(a: Self, b: Self) -> Self:
        return Self(max(a.data, b.data))

    @staticmethod
    fn min(a: Self, b: Self) -> Self:
        return Self(min(a.data, b.data))

    fn __pow__(a: Self, b: Self) -> Self:
        return Self(pow(a.data, b.data))

    fn mean(self) -> Scalar[Self.type]:
        var data = self.data

        @parameter
        if not Self.size.is_power_of_two():
            data[-1] = 0
        return data.reduce_add() / Scalar[Self.type](Self.size)

    fn clamp(self, min: Scalar[Self.type], max: Scalar[Self.type]) -> Self:
        return Self(clamp(self.data, min, max))

    @staticmethod
    fn random(
        min: Scalar[Self.type] = 0,
        max: Scalar[Self.type] = 1,
    ) -> Self:
        var rng = Self.zeros()

        @parameter
        for i in range(Self.size):
            rng[i] = Scalar[Self.type](
                random_float64(Float64(min), Float64(max))
            )
        return rng

    fn near_zero(self) -> Bool:
        comptime s = 1e-8
        return abs(self.data).le(s).reduce_and()

    fn x(self) -> Scalar[Self.type]:
        return self.data[0]

    fn y(self) -> Scalar[Self.type]:
        return self.data[1]

    fn z(self) -> Scalar[Self.type]:
        __comptime_assert Self.size >= 3
        return self.data[2]

    fn w(self) -> Scalar[Self.type]:
        __comptime_assert Self.size >= 4
        return self.data[3]

    # Swizzles
    fn xy(self) -> Vector2[Self.type]:
        return Vector2[self.type](self.x(), self.y())

    fn yz(self) -> Vector2[Self.type]:
        __comptime_assert Self.size >= 3
        return Vector2[self.type](self.y(), self.z())

    fn xz(self) -> Vector2[Self.type]:
        __comptime_assert Self.size >= 3
        return Vector2[self.type](self.x(), self.z())

    fn xyz(self) -> Vector[Self.type, 3]:
        __comptime_assert Self.size >= 3
        return Vector[Self.type, 3](self.x(), self.y(), self.z())

    # accessors
    fn __getitem__(self, i: Int) -> Scalar[Self.type]:
        return self.data[i]

    fn __setitem__(mut self, i: Int, v: Scalar[Self.type]):
        self.data[i] = v

    # operators
    fn __neg__(self) -> Self:
        return Self(-self.data)

    fn __add__(self, other: Self) -> Self:
        return Self(self.data + other.data)

    fn __iadd__(mut self, other: Self):
        self.data += other.data

    fn __sub__(self, other: Self) -> Self:
        return Self(self.data - other.data)

    fn __isub__(mut self, other: Self):
        self.data -= other.data

    fn __mul__(self, other: Self) -> Self:
        return Self(self.data * other.data)

    fn __imul__(mut self, other: Self):
        self.data *= other.data

    fn __truediv__(self, other: Self) -> Self:
        return Self(self.data / other.data)

    # scalar
    fn __add__(self, s: Scalar[Self.type]) -> Self:
        return Self(self.data + s)

    fn __iadd__(mut self, s: Scalar[Self.type]):
        self.data += s

    fn __sub__(self, s: Scalar[Self.type]) -> Self:
        return Self(self.data - s)

    fn __isub__(mut self, s: Scalar[Self.type]):
        self.data -= s

    fn __mul__(self, s: Scalar[Self.type]) -> Self:
        return Self(self.data * s)

    fn __imul__(mut self, s: Scalar[Self.type]):
        self.data *= s

    fn __truediv__(self, s: Scalar[Self.type]) -> Self:
        return Self(self.data / s)

    fn __itruediv__(mut self, s: Scalar[Self.type]):
        self.data /= s

    # reverse scalar
    fn __radd__(self, s: Scalar[Self.type]) -> Self:
        return Self(self.data + s)

    fn __riadd__(mut self, s: Scalar[Self.type]):
        self.data += s

    fn __rsub__(self, s: Scalar[Self.type]) -> Self:
        return Self(self.data - s)

    fn __risub__(mut self, s: Scalar[Self.type]):
        self.data -= s

    fn __rmul__(self, s: Scalar[Self.type]) -> Self:
        return Self(self.data * s)

    fn __rimul__(mut self, s: Scalar[Self.type]):
        self.data *= s

    fn __eq__(self, o: Self) -> Bool:
        return self.data == o.data

    # print
    fn __str__(self) -> String:
        @parameter
        if Self.size == 3:  # to not print the 0 padding
            return "Vec3[{}, {}, {}]".format(self.x(), self.y(), self.z())
        else:
            return "Vec{}{}".format(Self.size, self.data)

    fn write_to(self, mut writer: Some[Writer]):
        writer.write(String(self))


# ----------------------------------------------------------------------
# Quaternion
# ----------------------------------------------------------------------
fn length2[type: DType](q: Quaternion[type]) -> Scalar[type]:
    return (q.data * q.data).reduce_add()


fn length[type: DType](q: Quaternion[type]) -> Scalar[type]:
    return sqrt(length2(q))


fn inv_length[type: DType](q: Quaternion[type]) -> Scalar[type]:
    return 1.0 / length(q)


fn normalize[type: DType](q: Quaternion[type]) -> Quaternion[type]:
    return Quaternion[type](q.data * inv_length(q))


@fieldwise_init
@register_passable("trivial")
struct Quaternion[type: DType]:
    """TODO: Should we use wxyz or xyzw ? i heard gpu expect xyzw. but eigen uses wxyz.
    """

    var data: SIMD[Self.type, 4]

    fn __init__(
        out self,
        w: Scalar[Self.type],
        x: Scalar[Self.type],
        y: Scalar[Self.type],
        z: Scalar[Self.type],
    ):
        self.data = SIMD[Self.type, 4](w, x, y, z)

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

    fn xyz(self) -> Vector3[Self.type]:
        return Vector3[Self.type](self.x(), self.y(), self.z())

    fn inv(self) -> Self:
        return Self(self.w(), -self.x(), -self.y(), -self.z())

    fn rotate_vec(self, v: Vector3[Self.type]) -> Vector3[Self.type]:
        var pure = self.xyz()
        var scalar = self.w()
        var pure_x_v = cross(pure, v)
        var pure_x_pure_x_v = cross(pure, pure_x_v)

        return v + (pure_x_v * scalar + pure_x_pure_x_v) * 2.0

    @staticmethod
    fn angle_axis(angle: Scalar[Self.type], normal: Vector3[Self.type]) -> Self:
        var half_angle = angle / 2.0
        var w = cos(half_angle)
        var xyz = normal * sin(half_angle)
        return Self(w, xyz.x(), xyz.y(), xyz.z())

    @staticmethod
    fn from_basis[
        version: Int = 0
    ](
        a: Vector3[Self.type], b: Vector3[Self.type], c: Vector3[Self.type]
    ) -> Self:
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


# ----------------------------------------------------------------------
# Diag3x3
# ----------------------------------------------------------------------
@fieldwise_init
@register_passable("trivial")
struct Diag3x3[type: DType]:
    var d0: Scalar[Self.type]
    var d1: Scalar[Self.type]
    var d2: Scalar[Self.type]

    @staticmethod
    fn identity() -> Self:
        return Self(1, 1, 1)

    @staticmethod
    fn uniform(scale: Scalar[Self.type]) -> Self:
        return Diag3x3[Self.type](scale, scale, scale)

    @staticmethod
    fn from_vec(v: Vector3[Self.type]) -> Self:
        return Self(v.x(), v.y(), v.z())

    fn inv(self) -> Self:
        return Self(1.0 / self.d0, 1.0 / self.d1, 1.0 / self.d2)

    fn __mul__(self, o: Self) -> Self:
        return Self(self.d0 * o.d0, self.d1 * o.d1, self.d2 * o.d2)

    fn __mul__(self, v: Vector3[Self.type]) -> Vector3[Self.type]:
        return Vector3[Self.type](
            self.d0 * v.x(), self.d1 * v.y(), self.d2 * v.z()
        )

    fn __mul__(self, f: Scalar[Self.type]) -> Diag3x3[Self.type]:
        return Diag3x3[Self.type](self.d0 * f, self.d1 * f, self.d2 * f)


# ----------------------------------------------------------------------
# Mat3x3
# ----------------------------------------------------------------------
@fieldwise_init
@register_passable("trivial")
struct Mat3x3[type: DType]:
    var c0: Vector3[Self.type]
    var c1: Vector3[Self.type]
    var c2: Vector3[Self.type]

    fn __getitem__(self, i: Int) -> Vector3[Self.type]:
        if i == 0:
            return self.c0
        elif i == 1:
            return self.c1
        else:
            return self.c2

    fn determinant(self) -> Scalar[Self.type]:
        return (
            self.c0.x()
            * (self.c1.y() * self.c2.z() - self.c2.y() * self.c1.z())
            - self.c0.y()
            * (self.c1.x() * self.c2.z() - self.c2.x() * self.c1.z())
            + self.c0.z()
            * (self.c1.x() * self.c2.y() - self.c2.x() * self.c1.y())
        )

    fn transpose(self) -> Self:
        return Self(
            Vector3[Self.type](self.c0.x(), self.c1.x(), self.c2.x()),
            Vector3[Self.type](self.c0.y(), self.c1.y(), self.c2.y()),
            Vector3[Self.type](self.c0.z(), self.c1.z(), self.c2.z()),
        )

    @staticmethod
    fn from_quat(r: Quaternion[Self.type]) -> Self:
        var x2 = r.x() * r.x()
        var y2 = r.y() * r.y()
        var z2 = r.z() * r.z()

        var xz = r.x() * r.z()
        var xy = r.x() * r.y()
        var yz = r.y() * r.z()

        var r_rw = r * r.w()
        var wx = r_rw.x()
        var wy = r_rw.y()
        var wz = r_rw.z()

        comptime v3 = Vector3[Self.type]
        return Self(
            v3(1.0 - 2.0 * (y2 + z2), 2.0 * (xy + wz), 2.0 * (xz - wy)),
            v3(2.0 * (xy - wz), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz + wx)),
            v3(2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (x2 + y2)),
        )

    @staticmethod
    fn from_rs(r: Quaternion[Self.type], s: Diag3x3[Self.type]) -> Self:
        var x2 = r.x() * r.x()
        var y2 = r.y() * r.y()
        var z2 = r.z() * r.z()

        var xz = r.x() * r.z()
        var xy = r.x() * r.y()
        var yz = r.y() * r.z()

        var r_rw = r * r.w()
        var wx = r_rw.x()
        var wy = r_rw.y()
        var wz = r_rw.z()

        var ds = s * 2.0

        comptime v3 = Vector3[Self.type]
        return Self(
            v3(s.d0 - ds.d0 * (y2 + z2), ds.d0 * (xy + wz), ds.d0 * (xz - wy)),
            v3(ds.d1 * (xy - wz), s.d1 - ds.d1 * (x2 + z2), ds.d1 * (yz + wx)),
            v3(ds.d2 * (xz + wy), ds.d2 * (yz - wx), s.d2 - ds.d2 * (x2 + y2)),
        )

    fn __add__(self, o: Self) -> Self:
        return Self(self.c0 + o.c0, self.c1 + o.c1, self.c2 + o.c2)

    fn __sub__(self, o: Self) -> Self:
        return Self(self.c0 - o.c0, self.c1 - o.c1, self.c2 - o.c2)

    fn __mul__(self, v: Vector3[Self.type]) -> Vector3[Self.type]:
        return self.c0 * v.x() + self.c1 * v.y() + self.c2 * v.z()

    fn __mul__(self, o: Self) -> Self:
        return Self(self * o.c0, self * o.c1, self * o.c2)

    fn __mul__(self, s: Scalar[Self.type]) -> Self:
        return Self(self.c0 * s, self.c1 * s, self.c2 * s)


# ----------------------------------------------------------------------
# Mat3x4
# ----------------------------------------------------------------------
@fieldwise_init
@register_passable("trivial")
struct Mat3x4[type: DType]:
    var c0: Vector3[Self.type]
    var c1: Vector3[Self.type]
    var c2: Vector3[Self.type]
    var c3: Vector3[Self.type]

    @staticmethod
    fn identity() -> Self:
        return Self(
            Vector3[Self.type](1, 0, 0),
            Vector3[Self.type](0, 1, 0),
            Vector3[Self.type](0, 0, 1),
            Vector3[Self.type](0, 0, 0),
        )

    fn txfm_point(self, p: Vector3[Self.type]) -> Vector3[Self.type]:
        return self.c0 * p.x() + self.c1 * p.y() + self.c2 * p.z() + self.c3

    fn txfm_dir(self, p: Vector3[Self.type]) -> Vector3[Self.type]:
        return self.c0 * p.x() + self.c1 * p.y() + self.c2 * p.z()

    fn compose(self, o: Self) -> Self:
        return Self(
            self.txfm_dir(o.c0),
            self.txfm_dir(o.c1),
            self.txfm_dir(o.c2),
            self.txfm_point(o.c3),
        )

    fn decompose(
        self,
    ) -> Tuple[Vector3[Self.type], Quaternion[Self.type], Diag3x3[Self.type]]:
        var scale = Diag3x3[Self.type](
            length(self.c0), length(self.c1), length(self.c2)
        )

        if dot(cross(self.c0, self.c1), self.c2) < 0.0:
            scale.d0 *= -1.0

        var v1 = self.c0 / scale.d0
        var v2 = self.c1 / scale.d1
        var v3 = self.c2 / scale.d2

        v2 = normalize((v2 - v1 * dot(v2, v1)))
        v3 = v3 - v1 * dot(v3, v1)
        v3 = v3 - v2 * dot(v3, v2)
        v3 = normalize(v3)

        var rot = Quaternion[Self.type].from_basis(v1, v2, v3)
        return self.c3, rot, scale

    @staticmethod
    fn from_trs(
        t: Vector3[Self.type], r: Quaternion[Self.type], s: Diag3x3[Self.type]
    ) -> Self:
        var x2 = r.x() * r.x()
        var y2 = r.y() * r.y()
        var z2 = r.z() * r.z()
        var xz = r.x() * r.z()
        var xy = r.x() * r.y()
        var yz = r.y() * r.z()
        var wx = r.w() * r.x()
        var wy = r.w() * r.y()
        var wz = r.w() * r.z()
        var ds = s * 2.0

        comptime v3 = Vector3[Self.type]
        return Self(
            v3(s.d0 - ds.d0 * (y2 + z2), ds.d0 * (xy + wz), ds.d0 * (xz - wy)),
            v3(ds.d1 * (xy - wz), s.d1 - ds.d1 * (x2 + z2), ds.d1 * (yz + wx)),
            v3(ds.d2 * (xz + wy), ds.d2 * (yz - wx), s.d2 - ds.d2 * (x2 + y2)),
            t,
        )


# ----------------------------------------------------------------------
# Mat4x4
# ----------------------------------------------------------------------
@fieldwise_init
@register_passable("trivial")
struct Mat4x4[type: DType]:
    var c0: Vector4[Self.type]
    var c1: Vector4[Self.type]
    var c2: Vector4[Self.type]
    var c3: Vector4[Self.type]

    @staticmethod
    fn identity() -> Self:
        return Self(
            Vector4[Self.type](1, 0, 0, 0),
            Vector4[Self.type](0, 1, 0, 0),
            Vector4[Self.type](0, 0, 1, 0),
            Vector4[Self.type](0, 0, 0, 1),
        )

    fn txfm_point(self, p: Vector4[Self.type]) -> Vector4[Self.type]:
        return (
            self.c0 * p.x()
            + self.c1 * p.y()
            + self.c2 * p.z()
            + self.c3 * p.w()
        )

    fn compose(self, o: Mat4x4[Self.type]) -> Self:
        return Mat4x4[Self.type](
            self.txfm_point(o.c0),
            self.txfm_point(o.c1),
            self.txfm_point(o.c2),
            self.txfm_point(o.c3),
        )


# ----------------------------------------------------------------------
# AABB
# ----------------------------------------------------------------------
@fieldwise_init
@register_passable("trivial")
struct AxisAlignedBoundingBox[type: DType where type.is_floating_point()]:
    var pMin: Vector3[Self.type]
    var pMax: Vector3[Self.type]

    @staticmethod
    fn invalid() -> Self:
        comptime flt_max = max_finite[Self.type]()
        comptime flt_min = min_finite[Self.type]()
        return Self(
            Vector3[Self.type](flt_max),
            Vector3[Self.type](flt_min),
        )

    @staticmethod
    fn point(p: Vector3[Self.type]) -> Self:
        return Self(p, p)

    @staticmethod
    fn merge(a: Self, b: Self) -> Self:
        return Self(
            Vector.min(a.pMin, b.pMin),
            Vector.max(a.pMax, b.pMax),
        )

    fn surface_area(self) -> Scalar[Self.type]:
        var d = self.pMax - self.pMin
        return 2.0 * (d.x() * d.y() + d.x() * d.z() + d.y() * d.z())

    fn centroid(self) -> Vector3[Self.type]:
        return (self.pMin + self.pMax) * 0.5

    fn overlaps(self, o: Self) -> Bool:
        return (
            self.pMin.x() < o.pMax.x()
            and o.pMin.x() < self.pMax.x()
            and self.pMin.y() < o.pMax.y()
            and o.pMin.y() < self.pMax.y()
            and self.pMin.z() < o.pMax.z()
            and o.pMin.z() < self.pMax.z()
        )

    fn contains(self, p: Vector3[Self.type]) -> Bool:
        return (
            self.pMin.x() <= p.x()
            and self.pMin.y() <= p.y()
            and self.pMin.z() <= p.z()
            and self.pMax.x() >= p.x()
            and self.pMax.y() >= p.y()
            and self.pMax.z() >= p.z()
        )

    fn ray_intersects(
        self,
        ray_o: Vector3[Self.type],
        inv_ray_d: Diag3x3[Self.type],
        ray_t_min: Scalar[Self.type],
        ray_t_max: Scalar[Self.type],
    ) -> Bool:
        var t_lower = inv_ray_d * (self.pMin - ray_o)
        var t_upper = inv_ray_d * (self.pMax - ray_o)

        var t_min_vec = Vector.min(t_lower, t_upper)
        var t_max_vec = Vector.max(t_lower, t_upper)

        var t_box_min = max(
            t_min_vec.x(), t_min_vec.y(), t_min_vec.z(), ray_t_min
        )

        var t_box_max = min(
            t_max_vec.x(), t_max_vec.y(), t_max_vec.z(), ray_t_max
        )

        return t_box_min <= t_box_max

    fn apply_trs(
        self,
        translation: Vector3[Self.type],
        rotation: Quaternion[Self.type],
        scale: Diag3x3[Self.type],
    ) -> Self:
        """
        Transforms the AABB using Jim Arvo algorithm (from "Graphics Gems", Academic Press, 1990) with explicit SIMD.
        """
        var mat = Mat3x3[Self.type].from_rs(rotation, scale)

        var new_min = translation
        var new_max = translation

        # X
        var c0_a = mat.c0 * self.pMin.x()
        var c0_b = mat.c0 * self.pMax.x()
        new_min += Vector.min(c0_a, c0_b)
        new_max += Vector.max(c0_a, c0_b)

        # Y
        var c1_a = mat.c1 * self.pMin.y()
        var c1_b = mat.c1 * self.pMax.y()
        new_min += Vector.min(c1_a, c1_b)
        new_max += Vector.max(c1_a, c1_b)

        # Z
        var c2_a = mat.c2 * self.pMin.z()
        var c2_b = mat.c2 * self.pMax.z()
        new_min += Vector.min(c2_a, c2_b)
        new_max += Vector.max(c2_a, c2_b)

        return AxisAlignedBoundingBox(new_min, new_max)


fn main() raises:
    print("hello bmath")
