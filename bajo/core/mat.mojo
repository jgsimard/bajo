from math import acos, asin, atan2, clamp, cos, fma, pi, sin, sqrt, tan
from random import random_float64, Random
from std.bit import next_power_of_two
from std.utils.numerics import max_finite, min_finite

from bajo.core.quat import Quaternion
from bajo.core.vec import Vec3, Vec4, length, dot, cross, normalize


comptime Mat3f = Mat3x3[DType.float32]
comptime Mat3x4f = Mat3x4[DType.float32]
comptime Mat4f = Mat4x4[DType.float32]


# ----------------------------------------------------------------------
# Mat3x3
# ----------------------------------------------------------------------
@fieldwise_init
struct Mat3x3[type: DType where type.is_floating_point()](Copyable, Writable):
    var c0: Vec3[Self.type]
    var c1: Vec3[Self.type]
    var c2: Vec3[Self.type]

    fn __getitem__(self, i: Int) -> Vec3[Self.type]:
        if i == 0:
            return self.c0.copy()
        elif i == 1:
            return self.c1.copy()
        else:
            return self.c2.copy()

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
            Vec3[Self.type](self.c0.x(), self.c1.x(), self.c2.x()),
            Vec3[Self.type](self.c0.y(), self.c1.y(), self.c2.y()),
            Vec3[Self.type](self.c0.z(), self.c1.z(), self.c2.z()),
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

        comptime v3 = Vec3[Self.type]
        return Self(
            v3(1.0 - 2.0 * (y2 + z2), 2.0 * (xy + wz), 2.0 * (xz - wy)),
            v3(2.0 * (xy - wz), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz + wx)),
            v3(2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (x2 + y2)),
        )

    @staticmethod
    fn from_rs(r: Quaternion[Self.type], s: Vec3[Self.type]) -> Self:
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

        comptime v3 = Vec3[Self.type]
        return Self(
            v3(s[0] - ds[0] * (y2 + z2), ds[0] * (xy + wz), ds[0] * (xz - wy)),
            v3(ds[1] * (xy - wz), s[1] - ds[1] * (x2 + z2), ds[1] * (yz + wx)),
            v3(ds[2] * (xz + wy), ds[2] * (yz - wx), s[2] - ds[2] * (x2 + y2)),
        )

    fn __add__(self, o: Self) -> Self:
        return Self(self.c0 + o.c0, self.c1 + o.c1, self.c2 + o.c2)

    fn __sub__(self, o: Self) -> Self:
        return Self(self.c0 - o.c0, self.c1 - o.c1, self.c2 - o.c2)

    fn __mul__(self, v: Vec3[Self.type]) -> Vec3[Self.type]:
        return self.c0 * v.x() + self.c1 * v.y() + self.c2 * v.z()

    fn __mul__(self, o: Self) -> Self:
        return Self(self * o.c0, self * o.c1, self * o.c2)

    fn __mul__(self, s: Scalar[Self.type]) -> Self:
        return Self(self.c0 * s, self.c1 * s, self.c2 * s)


# ----------------------------------------------------------------------
# Mat3x4
# ----------------------------------------------------------------------
@fieldwise_init
struct Mat3x4[type: DType where type.is_floating_point()](Copyable, Writable):
    var c0: Vec3[Self.type]
    var c1: Vec3[Self.type]
    var c2: Vec3[Self.type]
    var c3: Vec3[Self.type]

    @staticmethod
    fn identity() -> Self:
        return Self(
            Vec3[Self.type](1, 0, 0),
            Vec3[Self.type](0, 1, 0),
            Vec3[Self.type](0, 0, 1),
            Vec3[Self.type](0, 0, 0),
        )

    fn txfm_point(self, p: Vec3[Self.type]) -> Vec3[Self.type]:
        return self.c0 * p.x() + self.c1 * p.y() + self.c2 * p.z() + self.c3

    fn txfm_dir(self, p: Vec3[Self.type]) -> Vec3[Self.type]:
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
    ) -> Tuple[Vec3[Self.type], Quaternion[Self.type], Vec3[Self.type]]:
        var scale = Vec3[Self.type](
            length(self.c0), length(self.c1), length(self.c2)
        )

        if dot(cross(self.c0, self.c1), self.c2) < 0.0:
            scale[0] *= -1.0

        var v1 = self.c0 / scale[0]
        var v2 = self.c1 / scale[1]
        var v3 = self.c2 / scale[2]

        v2 = normalize((v2 - v1 * dot(v2, v1)))
        v3 = v3 - v1 * dot(v3, v1)
        v3 = v3 - v2 * dot(v3, v2)
        v3 = normalize(v3)

        var rot = Quaternion[Self.type].from_basis(v1, v2, v3)
        return self.c3.copy(), rot, scale^

    @staticmethod
    fn from_trs(
        t: Vec3[Self.type], r: Quaternion[Self.type], s: Vec3[Self.type]
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

        comptime v3 = Vec3[Self.type]
        return Self(
            v3(s[0] - ds[0] * (y2 + z2), ds[0] * (xy + wz), ds[0] * (xz - wy)),
            v3(ds[1] * (xy - wz), s[1] - ds[1] * (x2 + z2), ds[1] * (yz + wx)),
            v3(ds[2] * (xz + wy), ds[2] * (yz - wx), s[2] - ds[2] * (x2 + y2)),
            t.copy(),
        )


# ----------------------------------------------------------------------
# Mat4x4
# ----------------------------------------------------------------------
@fieldwise_init
struct Mat4x4[type: DType](TrivialRegisterPassable):
    var c0: Vec4[Self.type]
    var c1: Vec4[Self.type]
    var c2: Vec4[Self.type]
    var c3: Vec4[Self.type]

    @staticmethod
    fn identity() -> Self:
        return Self(
            Vec4[Self.type](1, 0, 0, 0),
            Vec4[Self.type](0, 1, 0, 0),
            Vec4[Self.type](0, 0, 1, 0),
            Vec4[Self.type](0, 0, 0, 1),
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
