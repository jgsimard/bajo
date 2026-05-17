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


@fieldwise_init
struct Intersection(TrivialRegisterPassable, Writable):
    var t: Float32
    var u: Float32
    var v: Float32
    var inst: UInt32
    var prim: UInt32

    @always_inline
    def __init__(out self):
        self.t = f32_max
        self.u = 0.0
        self.v = 0.0
        self.inst = 0
        self.prim = 0


@fieldwise_init
struct Ray(Copyable, Writable):
    var O: Vec3f32
    var mask: UInt32
    var D: Vec3f32
    var rD: Vec3f32
    var hit: Intersection

    def __init__(out self, O: Vec3f32, D: Vec3f32, t_max: Float32 = f32_max):
        self.O = O
        self.D = D
        var rDx = clamp(Float32(1.0) / D.x, f32_min, f32_max)
        var rDy = clamp(Float32(1.0) / D.y, f32_min, f32_max)
        var rDz = clamp(Float32(1.0) / D.z, f32_min, f32_max)
        self.rD = Vec3f32(rDx, rDy, rDz)
        self.mask = 0xFFFFFFFF
        self.hit = Intersection()
        self.hit.t = t_max


@fieldwise_init
struct RayFlat(TrivialRegisterPassable):
    var o: Vec3f32
    var d: Vec3f32
    var rd: Vec3f32
    var t_max: Float32

    def __init__(
        out self, rays: UnsafePointer[Float32, MutAnyOrigin], ray_idx: Int
    ):
        var base = ray_idx * 10
        self.o = Vec3f32.load(rays, base)
        self.d = Vec3f32.load(rays, base + 3)
        self.rd = Vec3f32.load(rays, base + 6)
        self.t_max = rays[base + 9]


@fieldwise_init
struct Hit(TrivialRegisterPassable):
    var t: Float32
    var u: Float32
    var v: Float32
    var prim: UInt32
    var occluded: UInt32


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


@fieldwise_init
struct BvhInstance(Copyable):
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
