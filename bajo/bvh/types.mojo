from std.math import clamp
from std.utils.numerics import max_finite, min_finite

from bajo.core.aabb import AABB
from bajo.core.vec import Vec3f32, vmin, vmax, Vec3
from bajo.core.mat import Mat44f32, transform_point

comptime f32_max = max_finite[DType.float32]()
comptime f32_min = min_finite[DType.float32]()
comptime INV_3 = Float32(1.0 / 3.0)
comptime BVH_BINS = 16
comptime EMPTY_LANE = UInt32(0xFFFFFFFF)

comptime RAY_FLAT_STRIDE = 12
comptime RAY_O = 0  # 0, 1, 2
comptime RAY_D = 3  # 3, 4, 5
comptime RAY_RD = 6  # 6, 7, 8
comptime RAY_T_MIN = 9
comptime RAY_T_MAX = 10
comptime RAY_MASK = 11


struct Hit(TrivialRegisterPassable, Writable):
    var t: Float32
    var u: Float32
    var v: Float32
    var prim: UInt32
    var inst: UInt32
    var occluded: UInt32

    @always_inline
    def __init__(out self):
        self.t = f32_max
        self.u = 0.0
        self.v = 0.0
        self.prim = EMPTY_LANE
        self.inst = EMPTY_LANE
        self.occluded = UInt32(0)

    @always_inline
    def __init__(
        out self,
        t: Float32,
        u: Float32,
        v: Float32,
        prim: UInt32,
        inst: UInt32 = EMPTY_LANE,
        occluded: UInt32 = UInt32(0),
    ):
        self.t = t
        self.u = u
        self.v = v
        self.prim = prim
        self.inst = inst
        self.occluded = occluded

    @always_inline
    @staticmethod
    def miss() -> Self:
        return Self(f32_max, 0.0, 0.0, EMPTY_LANE, EMPTY_LANE, 0)

    @always_inline
    @staticmethod
    def shadow_hit() -> Self:
        return Self(0.0, 0.0, 0.0, EMPTY_LANE, EMPTY_LANE, 1)

    @always_inline
    def is_hit(self) -> Bool:
        return self.prim != EMPTY_LANE and self.t < f32_max

    @always_inline
    def is_occluded(self) -> Bool:
        return self.occluded != UInt32(0)


@fieldwise_init
struct Ray(TrivialRegisterPassable, Writable):
    var o: Vec3f32
    var d: Vec3f32
    var rd: Vec3f32
    var t_min: Float32
    var t_max: Float32
    var mask: UInt32

    def __init__(
        out self,
        origin: Vec3f32,
        direction: Vec3f32,
        t_min: Float32 = 0.0,
        t_max: Float32 = f32_max,
        mask: UInt32 = UInt32(0xFFFFFFFF),
    ):
        self.o = origin
        self.d = direction
        self.rd = Vec3f32(
            clamp(Float32(1.0) / direction.x, f32_min, f32_max),
            clamp(Float32(1.0) / direction.y, f32_min, f32_max),
            clamp(Float32(1.0) / direction.z, f32_min, f32_max),
        )
        self.t_min = t_min
        self.t_max = t_max
        self.mask = mask

    def __init__(
        out self,
        rays: UnsafePointer[Float32, MutAnyOrigin],
        ray_idx: Int,
    ):
        var base = ray_idx * RAY_FLAT_STRIDE
        self.o = Vec3f32.load(rays, base + RAY_O)
        self.d = Vec3f32.load(rays, base + RAY_D)
        self.rd = Vec3f32.load(rays, base + RAY_RD)
        self.t_min = rays[base + RAY_T_MIN]
        self.t_max = rays[base + RAY_T_MAX]
        self.mask = UInt32(rays[base + RAY_MASK])


@fieldwise_init
struct Sphere(TrivialRegisterPassable):
    var center: Vec3f32
    var radius: Float32

    @always_inline
    def bounds(self) -> AABB:
        var r = Vec3f32(self.radius)
        return AABB(self.center - r, self.center + r)


@fieldwise_init
struct SphereLeafBlock[width: Int](Copyable):
    var center: Vec3[DType.float32, Self.width]
    var radius: SIMD[DType.float32, Self.width]
    var prim_indices: SIMD[DType.uint32, Self.width]
    var valid_lane: SIMD[DType.bool, Self.width]

    @always_inline
    def __init__(out self):
        self.center = Vec3[DType.float32, Self.width](0.0)
        self.radius = SIMD[DType.float32, Self.width](0.0)
        self.prim_indices = SIMD[DType.uint32, Self.width](EMPTY_LANE)
        self.valid_lane = SIMD[DType.bool, Self.width](fill=False)


@fieldwise_init
struct TriangleLeafBlock[width: Int](Copyable):
    var v0: Vec3[DType.float32, Self.width]
    var v1: Vec3[DType.float32, Self.width]
    var v2: Vec3[DType.float32, Self.width]
    var prim_indices: SIMD[DType.uint32, Self.width]
    var valid_lane: SIMD[DType.bool, Self.width]

    @always_inline
    def __init__(out self):
        self.v0 = Vec3[DType.float32, Self.width](0.0)
        self.v1 = Vec3[DType.float32, Self.width](0.0)
        self.v2 = Vec3[DType.float32, Self.width](0.0)
        self.prim_indices = SIMD[DType.uint32, Self.width](EMPTY_LANE)
        self.valid_lane = SIMD[DType.bool, Self.width](fill=False)


struct Instance(Copyable):
    """Instance of a BLAS in world space.

    - `transform` maps BLAS-local points/vectors to world space.
    - `inv_transform` maps world-space rays to BLAS-local space.
    - `bounds` is the transformed world-space root AABB.
    - `blas_idx` indexes the BLAS array passed to traversal.
    """

    var transform: Mat44f32
    var inv_transform: Mat44f32
    var bounds: AABB
    var blas_idx: UInt32

    @always_inline
    def __init__(out self):
        self.transform = Mat44f32.identity()
        self.inv_transform = Mat44f32.identity()
        self.bounds = AABB.invalid()
        self.blas_idx = UInt32(0)

    @always_inline
    def __init__(
        out self,
        transform: Mat44f32,
        inv_transform: Mat44f32,
        blas_idx: UInt32,
        blas_bounds: AABB,
    ):
        self.transform = transform.copy()
        self.inv_transform = inv_transform.copy()
        self.blas_idx = blas_idx
        self.bounds = transform_bounds(transform, blas_bounds)


@always_inline
def transform_bounds(transform: Mat44f32, bounds: AABB) -> AABB:
    var corners = InlineArray[Vec3f32, 8](fill=Vec3f32(0.0))

    corners[0] = Vec3f32(bounds._min.x, bounds._min.y, bounds._min.z)
    corners[1] = Vec3f32(bounds._max.x, bounds._min.y, bounds._min.z)
    corners[2] = Vec3f32(bounds._min.x, bounds._max.y, bounds._min.z)
    corners[3] = Vec3f32(bounds._max.x, bounds._max.y, bounds._min.z)
    corners[4] = Vec3f32(bounds._min.x, bounds._min.y, bounds._max.z)
    corners[5] = Vec3f32(bounds._max.x, bounds._min.y, bounds._max.z)
    corners[6] = Vec3f32(bounds._min.x, bounds._max.y, bounds._max.z)
    corners[7] = Vec3f32(bounds._max.x, bounds._max.y, bounds._max.z)

    var out = AABB.invalid()

    comptime for i in range(8):
        out.grow(transform_point(transform, corners[i]))

    return out
