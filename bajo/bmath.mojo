from math import pi, sqrt, sin, cos, tan, acos, atan2
from std.utils.numerics import max_finite, min_finite
from builtin.math import max as builtin_max, min as builtin_min
from std.bit import next_power_of_two

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

comptime Diag3 = Diag3x3[DType.float32]
comptime Mat33 = Mat3x3[DType.float32]
comptime Mat34 = Mat3x4[DType.float32]
comptime Mat44 = Mat4x4[DType.float32]

comptime Quat = Quaternion[DType.float32]

comptime AABB = AxisAlignedBoundingBox[DType.float32]

# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------


fn deg_to_radians[
    type: DType, size: Int
](degrees: SIMD[type, size]) -> SIMD[type, size]:
    return degrees * pi_d180


fn solve_quadratic[
    type: DType, size: Int
](a: SIMD[type, size], b: SIMD[type, size], c: SIMD[type, size]) -> Optional[
    Tuple[SIMD[type, size], SIMD[type, size]]
]:
    var det = b * b - 4.0 * a * c
    if det < 0.0:
        return None

    var sqrt_det = sqrt(det)
    var rcp_2a = 1.0 / (2.0 * a)

    var t1 = (-b - sqrt_det) * rcp_2a
    var t2 = (-b + sqrt_det) * rcp_2a

    return Optional(Tuple(t1, t2))


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


fn max[t: DType, s: Int](a: Vector[t, s], b: Vector[t, s]) -> Vector[t, s]:
    return Vector[t, s](builtin_max(a.data, b.data))


fn min[t: DType, s: Int](a: Vector[t, s], b: Vector[t, s]) -> Vector[t, s]:
    return Vector[t, s](builtin_min(a.data, b.data))


fn cross[type: DType](a: Vector3[type], b: Vector3[type]) -> Vector3[type]:
    return Vector3[type](
        a.y() * b.z() - a.z() * b.y(),
        a.z() * b.x() - a.x() * b.z(),
        a.x() * b.y() - a.y() * b.x(),
    )


fn lerp[
    t: DType, s: Int
](a: Vector[t, s], b: Vector[t, s], u: Scalar[t]) -> Vector[t, s]:
    return a * (Scalar[t](1.0) - u) + b * u


@fieldwise_init
@register_passable("trivial")
struct Vector[type: DType, size: Int](
    Copyable, Representable, Stringable, Writable
):
    """A wrapper around SIMD."""

    comptime storage_size = next_power_of_two(Self.size)
    comptime data_type = SIMD[Self.type, Self.storage_size]

    var data: Self.data_type

    fn __init__(out self, v: Scalar[Self.type]):
        self.data = Self.data_type(v)

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
        self.data = Self.data_type(x, y, z)

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
    fn zero() -> Self:
        return Self(Scalar[Self.type](0))

    @staticmethod
    fn one() -> Self:
        return Self(Scalar[Self.type](1))

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

    # print
    fn __str__(self) -> String:
        @parameter
        if Self.size == 3:  # to not print the 0 padding
            return "Vec3[{}, {}, {}]".format(self.x(), self.y(), self.z())
        else:
            return "Vec{}{}".format(Self.size, self.data)

    fn __repr__(self) -> String:
        return String(self)

    fn write_to(self, mut writer: Some[Writer]):
        writer.write(String(self))


# ----------------------------------------------------------------------
# Quat
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
    fn id() -> Self:
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
        return Self([self.w(), -self.x(), -self.y(), -self.z()])

    fn rotate_vec(self, v: Vector3[Self.type]) -> Vector3[Self.type]:
        var pure = self.xyz()
        var scalar = self.w()
        var pure_x_v = cross(pure, v)
        var pure_x_pure_x_v = cross(pure, pure_x_v)

        return v + (pure_x_v * scalar + pure_x_pure_x_v) * 2.0

    @staticmethod
    fn angle_axis(angle: Scalar[Self.type], normal: Vector3[Self.type]) -> Self:
        var half_angle = angle / 2.0
        var coshalf = cos(half_angle)
        var sinhalf = sin(half_angle)
        return Self(
            coshalf,
            normal.x() * sinhalf,
            normal.y() * sinhalf,
            normal.z() * sinhalf,
        )

    @staticmethod
    fn from_basis(
        a: Vector3[Self.type], b: Vector3[Self.type], c: Vector3[Self.type]
    ) -> Self:
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

        if biggest_index == 0:
            return Self(
                biggest_val,
                (b.z() - c.y()) * mult,
                (c.x() - a.z()) * mult,
                (a.y() - b.x()) * mult,
            )
        elif biggest_index == 1:
            return Self(
                (b.z() - c.y()) * mult,
                biggest_val,
                (a.y() + b.x()) * mult,
                (c.x() + a.z()) * mult,
            )
        elif biggest_index == 2:
            return Self(
                (c.x() - a.z()) * mult,
                (a.y() + b.x()) * mult,
                biggest_val,
                (b.z() + c.y()) * mult,
            )
        else:
            return Self(
                (a.y() - b.x()) * mult,
                (c.x() + a.z()) * mult,
                (b.z() + c.y()) * mult,
                biggest_val,
            )

    fn __add__(self, o: Self) -> Self:
        return Self(self.data + o.data)

    fn __iadd__(mut self, o: Self):
        self.data += o.data

    fn __sub__(self, o: Self) -> Self:
        return Self(self.data - o.data)

    fn __isub__(mut self, o: Self):
        self.data -= o.data

    fn __mul__(self, b: Self) -> Self:
        return Self(
            self.w() * b.w()
            - self.x() * b.x()
            - self.y() * b.y()
            - self.z() * b.z(),
            self.w() * b.x()
            + self.x() * b.w()
            + self.y() * b.z()
            - self.z() * b.y(),
            self.w() * b.y()
            - self.x() * b.z()
            + self.y() * b.w()
            + self.z() * b.x(),
            self.w() * b.z()
            + self.x() * b.y()
            - self.y() * b.x()
            + self.z() * b.w(),
        )

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
    fn id() -> Self:
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

        var wx = r.w() * r.x()
        var wy = r.w() * r.y()
        var wz = r.w() * r.z()

        return Self(
            Vector3[Self.type](
                1.0 - 2.0 * (y2 + z2), 2.0 * (xy + wz), 2.0 * (xz - wy)
            ),
            Vector3[Self.type](
                2.0 * (xy - wz), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz + wx)
            ),
            Vector3[Self.type](
                2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (x2 + y2)
            ),
        )

    @staticmethod
    fn from_rs(r: Quaternion[Self.type], s: Diag3x3[Self.type]) -> Self:
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

        return Self(
            Vector3[Self.type](
                s.d0 - ds.d0 * (y2 + z2), ds.d0 * (xy + wz), ds.d0 * (xz - wy)
            ),
            Vector3[Self.type](
                ds.d1 * (xy - wz), s.d1 - ds.d1 * (x2 + z2), ds.d1 * (yz + wx)
            ),
            Vector3[Self.type](
                ds.d2 * (xz + wy), ds.d2 * (yz - wx), s.d2 - ds.d2 * (x2 + y2)
            ),
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

        return Self(
            Vector3[Self.type](
                s.d0 - ds.d0 * (y2 + z2), ds.d0 * (xy + wz), ds.d0 * (xz - wy)
            ),
            Vector3[Self.type](
                ds.d1 * (xy - wz), s.d1 - ds.d1 * (x2 + z2), ds.d1 * (yz + wx)
            ),
            Vector3[Self.type](
                ds.d2 * (xz + wy), ds.d2 * (yz - wx), s.d2 - ds.d2 * (x2 + y2)
            ),
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
            Vector3[Self.type](flt_max, flt_max, flt_max),
            Vector3[Self.type](flt_min, flt_min, flt_min),
        )

    @staticmethod
    fn point(p: Vector3[Self.type]) -> Self:
        return Self(p, p)

    @staticmethod
    fn merge(a: Self, b: Self) -> Self:
        return Self(
            min(a.pMin, b.pMin),
            max(a.pMax, b.pMax),
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

        var t_min_vec = min(t_lower, t_upper)
        var t_max_vec = max(t_lower, t_upper)

        var t_box_min = builtin_max(
            t_min_vec.x(), t_min_vec.y(), t_min_vec.z(), ray_t_min
        )

        var t_box_max = builtin_min(
            t_max_vec.x(), t_max_vec.y(), t_max_vec.z(), ray_t_max
        )

        return t_box_min <= t_box_max

    fn apply_trs(
        self,
        translation: Vector3[Self.type],
        rotation: Quaternion[Self.type],
        scale: Diag3x3[Self.type],
    ) -> Self:
        var rot_mat = Mat3x3[Self.type].from_rs(rotation, scale)
        var txfmed = Self(translation, translation)

        for i in range(3):
            for j in range(3):
                # Manual indexing since we use fields
                var val_min: Scalar[Self.type]
                var val_max: Scalar[Self.type]
                if j == 0:
                    val_min = self.pMin.x()
                    val_max = self.pMax.x()
                elif j == 1:
                    val_min = self.pMin.y()
                    val_max = self.pMax.y()
                else:
                    val_min = self.pMin.z()
                    val_max = self.pMax.z()

                # rot_mat is col major, so rot_mat[j][i]
                var col_j = rot_mat[j]
                var mat_val: Scalar[Self.type]
                if i == 0:
                    mat_val = col_j.x()
                elif i == 1:
                    mat_val = col_j.y()
                else:
                    mat_val = col_j.z()

                var e = mat_val * val_min
                var f = mat_val * val_max

                if e < f:
                    txfmed.pMin[i] += e
                    txfmed.pMax[i] += f
                else:
                    txfmed.pMin[i] += f
                    txfmed.pMax[i] += e
        return txfmed


fn main() raises:
    a = Vec2f(0.1, 0.2)
    print(a)
    print("hello")
