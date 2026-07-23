from std.utils.numerics import max_finite

from bajo.core.frame import Frame
from bajo.core.vec import Point3, Vec3


struct Ray[dtype: DType, frame: Frame](TrivialRegisterPassable, Writable):
    comptime STRIDE = 8
    comptime ORIGIN = 0  # 0, 1, 2
    comptime T_MIN = 3
    comptime DIRECTION = 4  # 4, 5, 6
    comptime T_MAX = 7

    var o: Point3[Self.dtype, Self.frame]
    var t_min: Scalar[Self.dtype]
    var d: Vec3[Self.dtype, Self.frame]
    var t_max: Scalar[Self.dtype]

    def __init__(
        out self,
        origin: Point3[Self.dtype, Self.frame],
        direction: Vec3[Self.dtype, Self.frame],
        t_min: Scalar[Self.dtype] = 0.0,
        t_max: Scalar[Self.dtype] = max_finite[Self.dtype](),
    ):
        self.o = origin
        self.d = direction
        self.t_min = t_min
        self.t_max = t_max

    def __init__[
        origin: ImmOrigin
    ](out self, rays: UnsafePointer[Scalar[Self.dtype], origin], ray_idx: Int,):
        var base = ray_idx * Ray.STRIDE
        self.o = Point3[Self.dtype, Self.frame].load(rays, base + Ray.ORIGIN)
        self.t_min = rays[base + Ray.T_MIN]
        self.d = Vec3[Self.dtype, Self.frame].load(rays, base + Ray.DIRECTION)
        self.t_max = rays[base + Ray.T_MAX]

    def flatten(self) -> List[Scalar[Self.dtype]]:
        return [
            self.o.x,
            self.o.y,
            self.o.z,
            self.t_min,
            self.d.x,
            self.d.y,
            self.d.z,
            self.t_max,
        ]

    def origin[
        width: SIMDLength
    ](self) -> Point3[Self.dtype, Self.frame, width]:
        return Point3[Self.dtype, Self.frame, width](
            self.o.x, self.o.y, self.o.z
        )

    def direction[
        width: SIMDLength
    ](self) -> Vec3[Self.dtype, Self.frame, width]:
        return Vec3[Self.dtype, Self.frame, width](self.d.x, self.d.y, self.d.z)

    def rcp_direction[
        width: SIMDLength
    ](self, eps: Scalar[Self.dtype] = 1.0e-9) -> Vec3[
        Self.dtype, Self.frame, width
    ] where Self.dtype.is_floating_point():
        var d = self.direction[width]()
        var e = SIMD[Self.dtype, width](eps)
        var large = SIMD[Self.dtype, width](1.0 / eps)
        var one = SIMD[Self.dtype, width](1.0)

        var mx = abs(d.x).gt(e)
        var my = abs(d.y).gt(e)
        var mz = abs(d.z).gt(e)

        var sx = d.x.lt(0.0).select(-large, large)
        var sy = d.y.lt(0.0).select(-large, large)
        var sz = d.z.lt(0.0).select(-large, large)

        var dx = mx.select(d.x, one)
        var dy = my.select(d.y, one)
        var dz = mz.select(d.z, one)

        return Vec3[Self.dtype, Self.frame, width](
            mx.select(one / dx, sx),
            my.select(one / dy, sy),
            mz.select(one / dz, sz),
        )
