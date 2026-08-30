from std.utils.numerics import max_finite

from bajo.core.frame import Frame
from bajo.core.vec import Point3, Vec3


struct Ray[dtype: DType, frame: Frame, length: SIMDLength = 1](
    TrivialRegisterPassable, Writable
):
    comptime STRIDE = 8
    comptime ORIGIN = 0  # 0, 1, 2
    comptime T_MIN = 3
    comptime DIRECTION = 4  # 4, 5, 6
    comptime T_MAX = 7

    var o: Point3[Self.dtype, Self.frame, Self.length]
    var t_min: SIMD[Self.dtype, Self.length]
    var d: Vec3[Self.dtype, Self.frame, Self.length]
    var t_max: SIMD[Self.dtype, Self.length]

    def __init__(
        out self,
        origin: Point3[Self.dtype, Self.frame, Self.length],
        direction: Vec3[Self.dtype, Self.frame, Self.length],
        t_min: SIMD[Self.dtype, Self.length] = 0.0,
        t_max: SIMD[Self.dtype, Self.length] = max_finite[Self.dtype](),
    ):
        self.o = origin
        self.d = direction
        self.t_min = t_min
        self.t_max = t_max

    def __init__(
        out self,
        rays: ImmSpan[Scalar[Self.dtype], _],
        ray_idx: Int,
    ):
        comptime assert Self.length == 1
        debug_assert["safe", _use_compiler_assume=True](
            ray_idx >= 0 and ray_idx < len(rays) / Ray.STRIDE,
            "Ray load is outside the input span",
        )
        var base = ray_idx * Ray.STRIDE
        self.o = Point3[Self.dtype, Self.frame, Self.length](
            rays.unsafe_get(base + Ray.ORIGIN + 0),
            rays.unsafe_get(base + Ray.ORIGIN + 1),
            rays.unsafe_get(base + Ray.ORIGIN + 2),
        )
        self.t_min = rays.unsafe_get(base + Ray.T_MIN)
        self.d = Vec3[Self.dtype, Self.frame, Self.length](
            rays.unsafe_get(base + Ray.DIRECTION + 0),
            rays.unsafe_get(base + Ray.DIRECTION + 1),
            rays.unsafe_get(base + Ray.DIRECTION + 2),
        )
        self.t_max = rays.unsafe_get(base + Ray.T_MAX)

    @always_inline
    def at(
        self, t: SIMD[Self.dtype, Self.length]
    ) -> Point3[Self.dtype, Self.frame, Self.length]:
        return self.o + t * self.d

    def flatten(self) -> List[Scalar[Self.dtype]]:
        comptime assert Self.length == 1
        return [
            self.o.x[0],
            self.o.y[0],
            self.o.z[0],
            self.t_min[0],
            self.d.x[0],
            self.d.y[0],
            self.d.z[0],
            self.t_max[0],
        ]

    def origin[
        width: SIMDLength
    ](self) -> Point3[Self.dtype, Self.frame, width]:
        comptime assert Self.length == 1
        return Point3[Self.dtype, Self.frame, width](
            self.o.x[0], self.o.y[0], self.o.z[0]
        )

    def direction[
        width: SIMDLength
    ](self) -> Vec3[Self.dtype, Self.frame, width]:
        comptime assert Self.length == 1
        return Vec3[Self.dtype, Self.frame, width](
            self.d.x[0], self.d.y[0], self.d.z[0]
        )

    def reciprocal_direction[
        width: SIMDLength = Self.length
    ](self, eps: Scalar[Self.dtype] = 1.0e-9) -> Vec3[
        Self.dtype, Self.frame, width
    ] where Self.dtype.is_floating_point():
        comptime assert Self.length == 1 or width == Self.length
        var direction = Vec3[Self.dtype, Self.frame, width](0.0)
        comptime for lane in range(width):
            comptime if Self.length == 1:
                direction.x[lane] = self.d.x[0]
                direction.y[lane] = self.d.y[0]
                direction.z[lane] = self.d.z[0]
            else:
                direction.x[lane] = self.d.x[lane]
                direction.y[lane] = self.d.y[lane]
                direction.z[lane] = self.d.z[lane]

        var e = SIMD[Self.dtype, width](eps)
        var large = SIMD[Self.dtype, width](1.0 / eps)
        var one = SIMD[Self.dtype, width](1.0)

        var mx = abs(direction.x).gt(e)
        var my = abs(direction.y).gt(e)
        var mz = abs(direction.z).gt(e)

        var sx = direction.x.lt(0.0).select(-large, large)
        var sy = direction.y.lt(0.0).select(-large, large)
        var sz = direction.z.lt(0.0).select(-large, large)

        var dx = mx.select(direction.x, one)
        var dy = my.select(direction.y, one)
        var dz = mz.select(direction.z, one)

        return Vec3[Self.dtype, Self.frame, width](
            mx.select(one / dx, sx),
            my.select(one / dy, sy),
            mz.select(one / dz, sz),
        )
