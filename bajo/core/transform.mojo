from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.math import abs, fma

from bajo.core.vec import Vec3
from bajo.core.quat import Quaternion


comptime Affine3f32 = Affine3[DType.float32]
comptime Affine3f64 = Affine3[DType.float64]


@fieldwise_init
struct Affine3InverseResult[
    dtype: DType,
    width: Int,
](Movable):
    var mask: SIMD[DType.bool, Self.width]
    var inv: Affine3[Self.dtype, Self.width]


@fieldwise_init
struct Affine3[dtype: DType, width: Int = 1](
    Copyable, DevicePassable, Writable
):
    """3D affine transform.

    Points  : p_out = M * p_in + t

    Vectors : v_out = M * v_in
    """

    var m00: SIMD[Self.dtype, Self.width]
    var m01: SIMD[Self.dtype, Self.width]
    var m02: SIMD[Self.dtype, Self.width]
    var tx: SIMD[Self.dtype, Self.width]

    var m10: SIMD[Self.dtype, Self.width]
    var m11: SIMD[Self.dtype, Self.width]
    var m12: SIMD[Self.dtype, Self.width]
    var ty: SIMD[Self.dtype, Self.width]

    var m20: SIMD[Self.dtype, Self.width]
    var m21: SIMD[Self.dtype, Self.width]
    var m22: SIMD[Self.dtype, Self.width]
    var tz: SIMD[Self.dtype, Self.width]

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
            t"Affine3[{reflect[Scalar[Self.dtype]].name()},{Self.width}]"
        )

    def __init__(out self, v: Scalar[Self.dtype]):
        self.m00 = v
        self.m01 = v
        self.m02 = v
        self.tx = v

        self.m10 = v
        self.m11 = v
        self.m12 = v
        self.ty = v

        self.m20 = v
        self.m21 = v
        self.m22 = v
        self.tz = v

    @staticmethod
    def identity() -> Self:
        var m = Self(Scalar[Self.dtype](0))

        m.m00 = 1.0
        m.m11 = 1.0
        m.m22 = 1.0

        return m^

    @staticmethod
    def from_translation(t: Vec3[Self.dtype, Self.width]) -> Self:
        var m = Self.identity()

        m.tx = t.x
        m.ty = t.y
        m.tz = t.z

        return m^

    @staticmethod
    def from_scale(s: Vec3[Self.dtype, Self.width]) -> Self:
        var m = Self(Scalar[Self.dtype](0))

        m.m00 = s.x
        m.m11 = s.y
        m.m22 = s.z

        return m^

    @staticmethod
    def from_rotation_scale(
        r: Quaternion[Self.dtype, Self.width],
        s: Vec3[Self.dtype, Self.width],
    ) -> Self where Self.dtype.is_floating_point():
        return Self.from_rotation_scale_translation(
            r,
            s,
            Vec3[Self.dtype, Self.width](0),
        )

    @staticmethod
    def from_rotation_scale_translation(
        r: Quaternion[Self.dtype, Self.width],
        s: Vec3[Self.dtype, Self.width],
        t: Vec3[Self.dtype, Self.width],
    ) -> Self where Self.dtype.is_floating_point():
        var x2 = r.x * r.x
        var y2 = r.y * r.y
        var z2 = r.z * r.z

        var xy = r.x * r.y
        var xz = r.x * r.z
        var yz = r.y * r.z

        var wx = r.w * r.x
        var wy = r.w * r.y
        var wz = r.w * r.z

        var sx = s.x
        var sy = s.y
        var sz = s.z

        var dsx = sx * 2.0
        var dsy = sy * 2.0
        var dsz = sz * 2.0
        # fmt: off
        return Self(
            sx - dsx * (y2 + z2), dsy * (xy - wz),      dsz * (xz + wy),      t.x,
            dsx * (xy + wz),      sy - dsy * (x2 + z2), dsz * (yz - wx),      t.y,
            dsx * (xz - wy),      dsy * (yz + wx),      sz - dsz * (x2 + y2), t.z,
        )
        # fmt: on

    def transform_point(
        self, p: Vec3[Self.dtype, Self.width]
    ) -> Vec3[Self.dtype, Self.width]:
        """Formula : p_out = M * p_in + t."""
        return Vec3[Self.dtype, Self.width](
            fma(self.m00, p.x, fma(self.m01, p.y, fma(self.m02, p.z, self.tx))),
            fma(self.m10, p.x, fma(self.m11, p.y, fma(self.m12, p.z, self.ty))),
            fma(self.m20, p.x, fma(self.m21, p.y, fma(self.m22, p.z, self.tz))),
        )

    def transform_vector(
        self, v: Vec3[Self.dtype, Self.width]
    ) -> Vec3[Self.dtype, Self.width]:
        """Formula : v_out = M * v_in."""
        return Vec3[Self.dtype, Self.width](
            fma(self.m00, v.x, fma(self.m01, v.y, self.m02 * v.z)),
            fma(self.m10, v.x, fma(self.m11, v.y, self.m12 * v.z)),
            fma(self.m20, v.x, fma(self.m21, v.y, self.m22 * v.z)),
        )

    def translation(self) -> Vec3[Self.dtype, Self.width]:
        return Vec3[Self.dtype, Self.width](self.tx, self.ty, self.tz)

    @staticmethod
    def load[
        origin: Origin
    ](ptr: UnsafePointer[Scalar[Self.dtype], origin], base: Int) -> Self:
        comptime assert Self.width == 1
        # fmt: off
        return Self(
            ptr[base + 0], ptr[base + 1], ptr[base + 2], ptr[base + 3],
            ptr[base + 4], ptr[base + 5], ptr[base + 6], ptr[base + 7],
            ptr[base + 8], ptr[base + 9], ptr[base + 10], ptr[base + 11],
        )
        # fmt: on

    def store[
        origin: Origin[mut=True]
    ](self, ptr: UnsafePointer[Scalar[Self.dtype], origin], base: Int):
        comptime assert Self.width == 1
        # fmt: off
        (ptr + base + 0).store[width=4]([self.m00[0], self.m01[0], self.m02[0], self.tx[0]])
        (ptr + base + 4).store[width=4]([self.m10[0], self.m11[0], self.m12[0], self.ty[0]])
        (ptr + base + 8).store[width=4]([self.m20[0], self.m21[0], self.m22[0], self.tz[0]])
        # fmt: on

    def inverse(
        self,
        eps: Scalar[Self.dtype] = 1.0e-8,
    ) -> Affine3InverseResult[
        Self.dtype, Self.width
    ] where Self.dtype.is_floating_point():
        # full fma version = harder to read, but asm is the same length
        # inverse of the 3x3 linear part : inv(Mat33)
        var inv00 = self.m11 * self.m22 - self.m12 * self.m21
        var inv01 = self.m02 * self.m21 - self.m01 * self.m22
        var inv02 = self.m01 * self.m12 - self.m02 * self.m11

        var inv10 = self.m12 * self.m20 - self.m10 * self.m22
        var inv11 = self.m00 * self.m22 - self.m02 * self.m20
        var inv12 = self.m02 * self.m10 - self.m00 * self.m12

        var inv20 = self.m10 * self.m21 - self.m11 * self.m20
        var inv21 = self.m01 * self.m20 - self.m00 * self.m21
        var inv22 = self.m00 * self.m11 - self.m01 * self.m10

        var det = self.m00 * inv00 + self.m01 * inv10 + self.m02 * inv20

        var mask = abs(det).gt(eps)

        var safe_det = mask.select(det, 1.0)
        var rcp_det = 1.0 / safe_det

        inv00 *= rcp_det
        inv01 *= rcp_det
        inv02 *= rcp_det

        inv10 *= rcp_det
        inv11 *= rcp_det
        inv12 *= rcp_det

        inv20 *= rcp_det
        inv21 *= rcp_det
        inv22 *= rcp_det

        # inverse of the translation: -inv_M * t
        var inv_tx = -(inv00 * self.tx + inv01 * self.ty + inv02 * self.tz)
        var inv_ty = -(inv10 * self.tx + inv11 * self.ty + inv12 * self.tz)
        var inv_tz = -(inv20 * self.tx + inv21 * self.ty + inv22 * self.tz)

        return Affine3InverseResult(
            mask,
            # fmt: off
            Self(
                inv00, inv01, inv02, inv_tx,
                inv10, inv11, inv12, inv_ty,
                inv20, inv21, inv22, inv_tz,
            ),
            # fmt: on
        )

    def flatten(self) -> List[SIMD[Self.dtype, Self.width]]:
        return [
            self.m00,
            self.m01,
            self.m02,
            self.tx,
            self.m10,
            self.m11,
            self.m12,
            self.ty,
            self.m20,
            self.m21,
            self.m22,
            self.tz,
        ]
