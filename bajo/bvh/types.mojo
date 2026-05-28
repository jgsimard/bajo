from std.math import clamp
from std.utils.numerics import max_finite, min_finite
from std.gpu import DeviceBuffer

from bajo.core.aabb import AABB
from bajo.bvh.constants import f32_max, EMPTY_LANE, Primitive, TRACE
from bajo.core.vec import Vec3f32, vmin, vmax, Vec3
from bajo.core.transform import Affine3f32

comptime RAY_STRIDE = 8
comptime RAY_O = 0  # 0, 1, 2
comptime RAY_T_MIN = 3
comptime RAY_D = 4  # 4, 5, 6
comptime RAY_T_MAX = 7


@fieldwise_init
struct Hit(TrivialRegisterPassable, Writable):
    var u: Float32
    var v: Float32
    var prim: UInt32
    var inst: UInt32
    var normal: Vec3f32
    var t: Float32

    @staticmethod
    def miss(t: Float32 = f32_max) -> Self:
        return Self(0.0, 0.0, EMPTY_LANE, EMPTY_LANE, Vec3f32(0), t)

    @staticmethod
    def shadow_hit() -> Self:
        return Self(0.0, 0.0, EMPTY_LANE, EMPTY_LANE, Vec3f32(0), 0.0)

    def is_hit(self) -> Bool:
        return self.prim != EMPTY_LANE and self.t < f32_max

    def is_occluded(self) -> Bool:
        return self.t < f32_max


@fieldwise_init
struct Ray(TrivialRegisterPassable, Writable):
    var o: Vec3f32
    var t_min: Float32
    var d: Vec3f32
    var t_max: Float32

    def __init__(
        out self,
        origin: Vec3f32,
        direction: Vec3f32,
        t_min: Float32 = 0.0,
        t_max: Float32 = f32_max,
    ):
        self.o = origin
        self.d = direction
        self.t_min = t_min
        self.t_max = t_max

    def __init__(
        out self,
        rays: UnsafePointer[Float32, MutAnyOrigin],
        ray_idx: Int,
    ):
        var base = ray_idx * RAY_STRIDE
        self.o = Vec3f32.load(rays, base + RAY_O)
        self.t_min = rays[base + RAY_T_MIN]
        self.d = Vec3f32.load(rays, base + RAY_D)
        self.t_max = rays[base + RAY_T_MAX]

    def flatten(self) -> List[Float32]:
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


@fieldwise_init
struct Sphere(TrivialRegisterPassable):
    comptime STRIDE = 4
    var center: Vec3f32
    var radius: Float32

    def bounds(self) -> AABB:
        var r = Vec3f32(self.radius)
        return AABB(self.center - r, self.center + r)


@fieldwise_init
struct SphereLeafBlock[width: Int](Copyable):
    var center: Vec3[DType.float32, Self.width]
    var radius: SIMD[DType.float32, Self.width]
    var prim_indices: SIMD[DType.uint32, Self.width]
    var valid_lane: SIMD[DType.bool, Self.width]

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

    var transform: Affine3f32
    var inv_transform: Affine3f32
    var bounds: AABB
    var blas_idx: UInt32
    var kind: Primitive

    def __init__(out self):
        self.transform = Affine3f32.identity()
        self.inv_transform = Affine3f32.identity()
        self.bounds = AABB.invalid()
        self.blas_idx = UInt32(0)
        self.kind = Primitive.UNKNOWN

    def __init__(
        out self,
        transform: Affine3f32,
        inv_transform: Affine3f32,
        blas_idx: UInt32,
        blas_bounds: AABB,
        kind: Primitive,
    ):
        self.transform = transform.copy()
        self.inv_transform = inv_transform.copy()
        self.blas_idx = blas_idx
        self.bounds = transform_bounds(transform, blas_bounds)
        self.kind = kind


def transform_bounds(transform: Affine3f32, bounds: AABB) -> AABB:
    var out = AABB.invalid()
    out.grow(
        transform.point(Vec3f32(bounds._min.x, bounds._min.y, bounds._min.z)),
        transform.point(Vec3f32(bounds._max.x, bounds._min.y, bounds._min.z)),
        transform.point(Vec3f32(bounds._min.x, bounds._max.y, bounds._min.z)),
        transform.point(Vec3f32(bounds._max.x, bounds._max.y, bounds._min.z)),
        transform.point(Vec3f32(bounds._min.x, bounds._min.y, bounds._max.z)),
        transform.point(Vec3f32(bounds._max.x, bounds._min.y, bounds._max.z)),
        transform.point(Vec3f32(bounds._min.x, bounds._max.y, bounds._max.z)),
        transform.point(Vec3f32(bounds._max.x, bounds._max.y, bounds._max.z)),
    )
    return out


trait TypedBvh:
    def trace[mode: TRACE](self, ray: Ray) -> Hit:
        ...


@fieldwise_init
struct BlasSet[width: Int]:
    comptime WIDE_BOUNDS_BASE = 0
    comptime WIDE_LANE_BASE = 1
    comptime LEAF_F32_BASE = 2
    comptime LEAF_U32_BASE = 3
    comptime ROOT_IDX = 4
    comptime NODE_COUNT = 5
    comptime LEAF_BLOCK_COUNT = 6
    comptime PRIM_COUNT = 7
    comptime STRIDE = 8

    var descs: DeviceBuffer[DType.uint32]
    var wide_bounds: DeviceBuffer[DType.float32]
    var wide_data: DeviceBuffer[DType.uint32]
    var wide_counts: DeviceBuffer[DType.uint32]
    var leaf_data_f32: DeviceBuffer[DType.float32]
    var leaf_prims: DeviceBuffer[DType.uint32]
    var blas_count: Int
