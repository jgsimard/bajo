from math import sqrt, cos, sin
from utils.numerics import max_finite, min_finite

# Constants
comptime pi = 3.14159265358979323846264338327950288
comptime pi_d2 = pi / 2.0
comptime pi_d180 = pi / 180.0
comptime pi_m2 = pi * 2.0


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
# Vector2
# ----------------------------------------------------------------------
@fieldwise_init
@register_passable("trivial")
struct Vector2[type: DType]:
    var x: Scalar[Self.type]
    var y: Scalar[Self.type]

    @always_inline
    fn dot(self, o: Vector2[Self.type]) -> Scalar[Self.type]:
        return self.x * o.x + self.y * o.y

    @always_inline
    fn length2(self) -> Scalar[Self.type]:
        return self.x * self.x + self.y * self.y

    @always_inline
    fn length(self) -> Scalar[Self.type]:
        return math.sqrt(self.length2())

    @always_inline
    fn inv_length(self) -> Scalar[Self.type]:
        return 1.0 / self.length()

    fn __getitem__(self, i: Int) -> Scalar[Self.type]:
        return self.x if i == 0 else self.y

    fn __setitem__(mut self, i: Int, val: Scalar[Self.type]):
        if i == 0:
            self.x = val
        else:
            self.y = val

    # Operators
    fn __add__(self, o: Vector2[Self.type]) -> Vector2[Self.type]:
        return Vector2[Self.type](self.x + o.x, self.y + o.y)

    fn __iadd__(mut self, o: Vector2[Self.type]):
        self.x += o.x
        self.y += o.y

    fn __sub__(self, o: Vector2[Self.type]) -> Vector2[Self.type]:
        return Vector2[Self.type](self.x - o.x, self.y - o.y)

    fn __isub__(mut self, o: Vector2[Self.type]):
        self.x -= o.x
        self.y -= o.y

    # Scalar ops
    fn __add__(self, o: Scalar[Self.type]) -> Vector2[Self.type]:
        return Vector2[Self.type](self.x + o, self.y + o)

    fn __sub__(self, o: Scalar[Self.type]) -> Vector2[Self.type]:
        return Vector2[Self.type](self.x - o, self.y - o)

    fn __mul__(self, o: Scalar[Self.type]) -> Vector2[Self.type]:
        return Vector2[Self.type](self.x * o, self.y * o)

    fn __truediv__(self, o: Scalar[Self.type]) -> Vector2[Self.type]:
        var inv = 1.0 / o
        return self * inv

    fn __neg__(self) -> Vector2[Self.type]:
        return Vector2[Self.type](-self.x, -self.y)

    @staticmethod
    fn min(a: Vector2[Self.type], b: Vector2[Self.type]) -> Vector2[Self.type]:
        return Vector2[Self.type](min(a.x, b.x), min(a.y, b.y))

    @staticmethod
    fn max(a: Vector2[Self.type], b: Vector2[Self.type]) -> Vector2[Self.type]:
        return Vector2[Self.type](max(a.x, b.x), max(a.y, b.y))


# ----------------------------------------------------------------------
# Vector3
# ----------------------------------------------------------------------
@fieldwise_init
@register_passable("trivial")
struct Vector3[type: DType]:
    var x: Scalar[Self.type]
    var y: Scalar[Self.type]
    var z: Scalar[Self.type]

    @staticmethod
    fn zero() -> Vector3[Self.type]:
        return Vector3[Self.type](0, 0, 0)

    @staticmethod
    fn one() -> Vector3[Self.type]:
        return Vector3[Self.type](1, 1, 1)

    @staticmethod
    fn all(v: Scalar[Self.type]) -> Vector3[Self.type]:
        return Vector3[Self.type](v, v, v)

    @always_inline
    fn dot(self, o: Vector3[Self.type]) -> Scalar[Self.type]:
        return self.x * o.x + self.y * o.y + self.z * o.z

    @always_inline
    fn cross(self, o: Vector3[Self.type]) -> Vector3[Self.type]:
        return Vector3[Self.type](
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )

    fn frame(self) -> Tuple[Vector3[Self.type], Vector3[Self.type]]:
        var arbitrary: Vector3[Self.type]
        if math.abs(self.x) < 0.8:
            arbitrary = Vector3[Self.type](1, 0, 0)
        else:
            arbitrary = Vector3[Self.type](0, 1, 0)

        var a = self.cross(arbitrary)
        var b = self.cross(a)
        return a, b

    @always_inline
    fn length2(self) -> Scalar[Self.type]:
        return self.x * self.x + self.y * self.y + self.z * self.z

    @always_inline
    fn length(self) -> Scalar[Self.type]:
        return math.sqrt(self.length2())

    @always_inline
    fn inv_length(self) -> Scalar[Self.type]:
        return 1.0 / self.length()

    @always_inline
    fn distance(self, o: Vector3[Self.type]) -> Scalar[Self.type]:
        return (self - o).length()

    @always_inline
    fn distance2(self, o: Vector3[Self.type]) -> Scalar[Self.type]:
        return (self - o).length2()

    @always_inline
    fn normalize(self) -> Vector3[Self.type]:
        return self * self.inv_length()

    # Swizzles
    fn xy(self) -> Vector2[Self.type]:
        return Vector2[self.type](self.x, self.y)

    fn yz(self) -> Vector2[Self.type]:
        return Vector2[self.type](self.y, self.z)

    fn xz(self) -> Vector2[Self.type]:
        return Vector2[self.type](self.x, self.z)

    fn __getitem__(self, i: Int) -> Scalar[Self.type]:
        if i == 0:
            return self.x
        elif i == 1:
            return self.y
        else:
            return self.z

    fn __setitem__(mut self, i: Int, val: Scalar[Self.type]):
        if i == 0:
            self.x = val
        elif i == 1:
            self.y = val
        else:
            self.z = val

    # Operators
    fn __add__(self, o: Vector3[Self.type]) -> Vector3[Self.type]:
        return Vector3[Self.type](self.x + o.x, self.y + o.y, self.z + o.z)

    fn __iadd__(mut self, o: Vector3[Self.type]):
        self.x += o.x
        self.y += o.y
        self.z += o.z

    fn __sub__(self, o: Vector3[Self.type]) -> Vector3[Self.type]:
        return Vector3[Self.type](self.x - o.x, self.y - o.y, self.z - o.z)

    fn __isub__(mut self, o: Vector3[Self.type]):
        self.x -= o.x
        self.y -= o.y
        self.z -= o.z

    # Scalar ops
    fn __add__(self, o: Scalar[Self.type]) -> Vector3[Self.type]:
        return Vector3[Self.type](self.x + o, self.y + o, self.z + o)

    fn __sub__(self, o: Scalar[Self.type]) -> Vector3[Self.type]:
        return Vector3[Self.type](self.x - o, self.y - o, self.z - o)

    fn __mul__(self, o: Scalar[Self.type]) -> Vector3[Self.type]:
        return Vector3[Self.type](self.x * o, self.y * o, self.z * o)

    fn __truediv__(self, o: Scalar[Self.type]) -> Vector3[Self.type]:
        var inv = 1.0 / o
        return self * inv

    fn __neg__(self) -> Vector3[Self.type]:
        return Vector3[Self.type](-self.x, -self.y, -self.z)

    @staticmethod
    fn min(a: Vector3[Self.type], b: Vector3[Self.type]) -> Vector3[Self.type]:
        return Vector3[Self.type](min(a.x, b.x), min(a.y, b.y), min(a.z, b.z))

    @staticmethod
    fn max(a: Vector3[Self.type], b: Vector3[Self.type]) -> Vector3[Self.type]:
        return Vector3[Self.type](max(a.x, b.x), max(a.y, b.y), max(a.z, b.z))


# ----------------------------------------------------------------------
# Vector4
# ----------------------------------------------------------------------
@fieldwise_init
@register_passable("trivial")
struct Vector4[type: DType]:
    var xyzw: SIMD[Self.type, 4]

    fn __init__(
        out self,
        x: Scalar[Self.type],
        y: Scalar[Self.type],
        z: Scalar[Self.type],
        w: Scalar[Self.type],
    ):
        self.xyzw = SIMD[Self.type, 4](x, y, z, w)

    @staticmethod
    fn zero() -> Vector4[Self.type]:
        return Vector4[Self.type](0)

    @staticmethod
    fn one() -> Vector4[Self.type]:
        return Vector4[Self.type](1)

    fn x(self) -> Scalar[Self.type]:
        return self.xyzw[0]

    fn y(self) -> Scalar[Self.type]:
        return self.xyzw[1]

    fn z(self) -> Scalar[Self.type]:
        return self.xyzw[2]

    fn w(self) -> Scalar[Self.type]:
        return self.xyzw[3]

    @staticmethod
    fn from_vec3_w(
        v: Vector3[Self.type], w: Scalar[Self.type]
    ) -> Vector4[Self.type]:
        return Vector4[Self.type]([v.x, v.y, v.z, w])

    fn xyz(self) -> Vector3[Self.type]:
        return Vector3[Self.type](self.xyzw[0], self.xyzw[1], self.xyzw[2])

    fn __getitem__(self, i: Int) -> Scalar[Self.type]:
        return self.xyzw[i]

    fn __mul__(self, s: Scalar[Self.type]) -> Vector4[Self.type]:
        return Vector4[Self.type](self.xyzw * s)

    fn __add__(self, o: Vector4[Self.type]) -> Vector4[Self.type]:
        return Vector4[Self.type](self.xyzw + o.xyzw)


# ----------------------------------------------------------------------
# Quat
# ----------------------------------------------------------------------
@fieldwise_init
@register_passable("trivial")
struct Quat[type: DType]:
    var wxyz: SIMD[Self.type, 4]

    fn __init__(
        out self,
        w: Scalar[Self.type],
        x: Scalar[Self.type],
        y: Scalar[Self.type],
        z: Scalar[Self.type],
    ):
        self.wxyz = SIMD[Self.type, 4](w, x, y, z)

    @staticmethod
    fn id() -> Quat[Self.type]:
        return Quat[Self.type]([1, 0, 0, 0])

    @always_inline
    fn length2(self) -> Scalar[Self.type]:
        return (self.wxyz * self.wxyz).reduce_add()

    @always_inline
    fn length(self) -> Scalar[Self.type]:
        return sqrt(self.length2())

    @always_inline
    fn inv_length(self) -> Scalar[Self.type]:
        return 1.0 / self.length()

    fn normalize(self) -> Quat[Self.type]:
        var inv = self.inv_length()
        return Quat[Self.type](self.wxyz * inv)

    fn w(self) -> Scalar[Self.type]:
        return self.wxyz[0]

    fn x(self) -> Scalar[Self.type]:
        return self.wxyz[1]

    fn y(self) -> Scalar[Self.type]:
        return self.wxyz[2]

    fn z(self) -> Scalar[Self.type]:
        return self.wxyz[3]

    fn xyz(self) -> Vector3[Self.type]:
        return Vector3[Self.type](self.wxyz[1], self.wxyz[2], self.wxyz[3])

    fn inv(self) -> Quat[Self.type]:
        return Quat[Self.type]([self.w(), -self.x(), -self.y(), -self.z()])

    fn rotate_vec(self, v: Vector3[Self.type]) -> Vector3[Self.type]:
        var pure = self.xyz()
        var scalar = self.w()
        var pure_x_v = pure.cross(v)
        var pure_x_pure_x_v = pure.cross(pure_x_v)

        return v + (pure_x_v * scalar + pure_x_pure_x_v) * 2.0

    @staticmethod
    fn angle_axis(
        angle: Scalar[Self.type], normal: Vector3[Self.type]
    ) -> Quat[Self.type]:
        var half_angle = angle / 2.0
        var coshalf = cos(half_angle)
        var sinhalf = sin(half_angle)
        return Quat[Self.type](
            coshalf, normal.x * sinhalf, normal.y * sinhalf, normal.z * sinhalf
        )

    @staticmethod
    fn from_basis(
        a: Vector3[Self.type], b: Vector3[Self.type], c: Vector3[Self.type]
    ) -> Quat[Self.type]:
        var four_x_sq_m1 = a.x - b.y - c.z
        var four_y_sq_m1 = b.y - a.x - c.z
        var four_z_sq_m1 = c.z - a.x - b.y
        var four_w_sq_m1 = a.x + b.y + c.z

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

        var biggest_val = math.sqrt(four_biggest_sq_m1 + 1.0) * 0.5
        var mult = 0.25 / biggest_val

        if biggest_index == 0:
            return Quat[Self.type](
                biggest_val,
                (b.z - c.y) * mult,
                (c.x - a.z) * mult,
                (a.y - b.x) * mult,
            )
        elif biggest_index == 1:
            return Quat[Self.type](
                (b.z - c.y) * mult,
                biggest_val,
                (a.y + b.x) * mult,
                (c.x + a.z) * mult,
            )
        elif biggest_index == 2:
            return Quat[Self.type](
                (c.x - a.z) * mult,
                (a.y + b.x) * mult,
                biggest_val,
                (b.z + c.y) * mult,
            )
        else:
            return Quat[Self.type](
                (a.y - b.x) * mult,
                (c.x + a.z) * mult,
                (b.z + c.y) * mult,
                biggest_val,
            )

    fn __add__(self, o: Quat[Self.type]) -> Quat[Self.type]:
        return Quat[Self.type](self.wxyz + o.wxyz)

    fn __iadd__(mut self, o: Quat[Self.type]):
        self.wxyz += o.wxyz

    fn __sub__(self, o: Quat[Self.type]) -> Quat[Self.type]:
        return Quat[Self.type](self.wxyz - o.wxyz)

    fn __isub__(mut self, o: Quat[Self.type]):
        self.wxyz -= o.wxyz

    fn __mul__(self, b: Quat[Self.type]) -> Quat[Self.type]:
        return Quat[Self.type](
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

    fn __mul__(self, f: Scalar[Self.type]) -> Quat[Self.type]:
        return Quat[Self.type](self.wxyz * f)

    fn __imul__(mut self, f: Scalar[Self.type]):
        self.wxyz *= f


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
    fn id() -> Diag3x3[Self.type]:
        return Diag3x3[Self.type](1, 1, 1)

    @staticmethod
    fn uniform(scale: Scalar[Self.type]) -> Diag3x3[Self.type]:
        return Diag3x3[Self.type](scale, scale, scale)

    @staticmethod
    fn from_vec(v: Vector3[Self.type]) -> Diag3x3[Self.type]:
        return Diag3x3[Self.type](v.x, v.y, v.z)

    fn inv(self) -> Diag3x3[Self.type]:
        return Diag3x3[Self.type](1.0 / self.d0, 1.0 / self.d1, 1.0 / self.d2)

    fn __mul__(self, o: Diag3x3[Self.type]) -> Diag3x3[Self.type]:
        return Diag3x3[Self.type](
            self.d0 * o.d0, self.d1 * o.d1, self.d2 * o.d2
        )

    fn __mul__(self, v: Vector3[Self.type]) -> Vector3[Self.type]:
        return Vector3[Self.type](self.d0 * v.x, self.d1 * v.y, self.d2 * v.z)

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
            self.c0.x * (self.c1.y * self.c2.z - self.c2.y * self.c1.z)
            - self.c0.y * (self.c1.x * self.c2.z - self.c2.x * self.c1.z)
            + self.c0.z * (self.c1.x * self.c2.y - self.c2.x * self.c1.y)
        )

    fn transpose(self) -> Mat3x3[Self.type]:
        return Mat3x3[Self.type](
            Vector3[Self.type](self.c0.x, self.c1.x, self.c2.x),
            Vector3[Self.type](self.c0.y, self.c1.y, self.c2.y),
            Vector3[Self.type](self.c0.z, self.c1.z, self.c2.z),
        )

    @staticmethod
    fn from_quat(r: Quat[Self.type]) -> Mat3x3[Self.type]:
        var x2 = r.x() * r.x()
        var y2 = r.y() * r.y()
        var z2 = r.z() * r.z()

        var xz = r.x() * r.z()
        var xy = r.x() * r.y()
        var yz = r.y() * r.z()

        var wx = r.w() * r.x()
        var wy = r.w() * r.y()
        var wz = r.w() * r.z()

        return Mat3x3[Self.type](
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
    fn from_rs(r: Quat[Self.type], s: Diag3x3[Self.type]) -> Mat3x3[Self.type]:
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

        return Mat3x3[Self.type](
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

    fn __add__(self, o: Mat3x3[Self.type]) -> Mat3x3[Self.type]:
        return Mat3x3[Self.type](self.c0 + o.c0, self.c1 + o.c1, self.c2 + o.c2)

    fn __sub__(self, o: Mat3x3[Self.type]) -> Mat3x3[Self.type]:
        return Mat3x3[Self.type](self.c0 - o.c0, self.c1 - o.c1, self.c2 - o.c2)

    fn __mul__(self, v: Vector3[Self.type]) -> Vector3[Self.type]:
        return self.c0 * v.x + self.c1 * v.y + self.c2 * v.z

    fn __mul__(self, o: Mat3x3[Self.type]) -> Mat3x3[Self.type]:
        return Mat3x3[Self.type](self * o.c0, self * o.c1, self * o.c2)

    fn __mul__(self, s: Scalar[Self.type]) -> Mat3x3[Self.type]:
        return Mat3x3[Self.type](self.c0 * s, self.c1 * s, self.c2 * s)


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
    fn identity() -> Mat3x4[Self.type]:
        return Mat3x4[Self.type](
            Vector3[Self.type](1, 0, 0),
            Vector3[Self.type](0, 1, 0),
            Vector3[Self.type](0, 0, 1),
            Vector3[Self.type](0, 0, 0),
        )

    fn txfm_point(self, p: Vector3[Self.type]) -> Vector3[Self.type]:
        return self.c0 * p.x + self.c1 * p.y + self.c2 * p.z + self.c3

    fn txfm_dir(self, p: Vector3[Self.type]) -> Vector3[Self.type]:
        return self.c0 * p.x + self.c1 * p.y + self.c2 * p.z

    fn compose(self, o: Mat3x4[Self.type]) -> Mat3x4[Self.type]:
        return Mat3x4[Self.type](
            self.txfm_dir(o.c0),
            self.txfm_dir(o.c1),
            self.txfm_dir(o.c2),
            self.txfm_point(o.c3),
        )

    fn decompose(
        self,
    ) -> Tuple[Vector3[Self.type], Quat[Self.type], Diag3x3[Self.type]]:
        var scale = Diag3x3[Self.type](
            self.c0.length(), self.c1.length(), self.c2.length()
        )

        if self.c0.cross(self.c1).dot(self.c2) < 0.0:
            scale.d0 *= -1.0

        var v1 = self.c0 / scale.d0
        var v2 = self.c1 / scale.d1
        var v3 = self.c2 / scale.d2

        v2 = (v2 - v1 * v2.dot(v1)).normalize()
        v3 = v3 - v1 * v3.dot(v1)
        v3 = v3 - v2 * v3.dot(v2)
        v3 = v3.normalize()

        var rot = Quat[Self.type].from_basis(v1, v2, v3)
        return self.c3, rot, scale

    @staticmethod
    fn from_trs(
        t: Vector3[Self.type], r: Quat[Self.type], s: Diag3x3[Self.type]
    ) -> Mat3x4[Self.type]:
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

        return Mat3x4[Self.type](
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
    fn identity() -> Mat4x4[Self.type]:
        return Mat4x4[Self.type](
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

    fn compose(self, o: Mat4x4[Self.type]) -> Mat4x4[Self.type]:
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
struct AABB[type: DType where type.is_floating_point()]:
    var pMin: Vector3[Self.type]
    var pMax: Vector3[Self.type]

    @staticmethod
    fn invalid() -> AABB[Self.type]:
        alias flt_max = max_finite[Self.type]()
        alias flt_min = min_finite[Self.type]()
        return AABB[Self.type](
            Vector3[Self.type](flt_max, flt_max, flt_max),
            Vector3[Self.type](flt_min, flt_min, flt_min),
        )

    @staticmethod
    fn point(p: Vector3[Self.type]) -> AABB[Self.type]:
        return AABB[Self.type](p, p)

    @staticmethod
    fn merge(a: AABB[Self.type], b: AABB[Self.type]) -> AABB[Self.type]:
        return AABB[Self.type](
            Vector3[Self.type].min(a.pMin, b.pMin),
            Vector3[Self.type].max(a.pMax, b.pMax),
        )

    fn surface_area(self) -> Scalar[Self.type]:
        var d = self.pMax - self.pMin
        return 2.0 * (d.x * d.y + d.x * d.z + d.y * d.z)

    fn centroid(self) -> Vector3[Self.type]:
        return (self.pMin + self.pMax) * 0.5

    fn overlaps(self, o: AABB[Self.type]) -> Bool:
        return (
            self.pMin.x < o.pMax.x
            and o.pMin.x < self.pMax.x
            and self.pMin.y < o.pMax.y
            and o.pMin.y < self.pMax.y
            and self.pMin.z < o.pMax.z
            and o.pMin.z < self.pMax.z
        )

    fn contains(self, p: Vector3[Self.type]) -> Bool:
        return (
            self.pMin.x <= p.x
            and self.pMin.y <= p.y
            and self.pMin.z <= p.z
            and self.pMax.x >= p.x
            and self.pMax.y >= p.y
            and self.pMax.z >= p.z
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

        var t_min_vec = Vector3[Self.type].min(t_lower, t_upper)
        var t_max_vec = Vector3[Self.type].max(t_lower, t_upper)

        var t_box_min = math.max(
            t_min_vec.x, math.max(t_min_vec.y, math.max(t_min_vec.z, ray_t_min))
        )
        var t_box_max = math.min(
            t_max_vec.x, math.min(t_max_vec.y, math.min(t_max_vec.z, ray_t_max))
        )

        return t_box_min <= t_box_max

    fn apply_trs(
        self,
        translation: Vector3[Self.type],
        rotation: Quat[Self.type],
        scale: Diag3x3[Self.type],
    ) -> AABB[Self.type]:
        var rot_mat = Mat3x3[Self.type].from_rs(rotation, scale)
        var txfmed = AABB[Self.type](translation, translation)

        for i in range(3):
            for j in range(3):
                # Manual indexing since we use fields
                var val_min: Scalar[Self.type]
                var val_max: Scalar[Self.type]
                if j == 0:
                    val_min = self.pMin.x
                    val_max = self.pMax.x
                elif j == 1:
                    val_min = self.pMin.y
                    val_max = self.pMax.y
                else:
                    val_min = self.pMin.z
                    val_max = self.pMax.z

                # rot_mat is col major, so rot_mat[j][i]
                var col_j = rot_mat[j]
                var mat_val: Scalar[Self.type]
                if i == 0:
                    mat_val = col_j.x
                elif i == 1:
                    mat_val = col_j.y
                else:
                    mat_val = col_j.z

                var e = mat_val * val_min
                var f = mat_val * val_max

                if e < f:
                    txfmed.pMin[i] += e
                    txfmed.pMax[i] += f
                else:
                    txfmed.pMin[i] += f
                    txfmed.pMax[i] += e
        return txfmed
