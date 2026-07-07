from std.math import clamp
from std.utils.numerics import max_finite, min_finite
from std.gpu import DeviceBuffer

from bajo.core import AABB, Vec3f32, Affine3f32, Point3f32
from bajo.bvh.constants import f32_max, EMPTY_LANE, Primitive, TRACE
from bajo.core.vec import Vec3, Point3, Normal3


@fieldwise_init
struct Hit(TrivialRegisterPassable, Writable):
    comptime U = 0
    comptime V = 1
    comptime PRIM = 2
    comptime INST = 3
    comptime NORMAL = 4
    comptime T = 7
    comptime STRIDE = 8

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

    def store[
        origin: MutOrigin,
        address_space: AddressSpace,
    ](
        self,
        hits: UnsafePointer[Float32, origin, address_space=address_space],
        idx: Int,
    ):
        var base = idx * Hit.STRIDE

        hits[base + Hit.U] = self.u
        hits[base + Hit.V] = self.v
        hits[base + Hit.NORMAL + 0] = self.normal.x
        hits[base + Hit.NORMAL + 1] = self.normal.y
        hits[base + Hit.NORMAL + 2] = self.normal.z
        hits[base + Hit.T] = self.t

        var hits_u32 = hits.bitcast[UInt32]()
        hits_u32[base + Hit.PRIM] = self.prim
        hits_u32[base + Hit.INST] = self.inst

    @staticmethod
    def load[
        origin: ImmutOrigin,
        address_space: AddressSpace,
    ](
        hits: UnsafePointer[Float32, origin, address_space=address_space],
        idx: Int,
    ) -> Hit:
        var base = idx * Hit.STRIDE
        var hits_u32 = hits.bitcast[UInt32]()

        return Hit(
            hits[base + Hit.U],
            hits[base + Hit.V],
            hits_u32[base + Hit.PRIM],
            hits_u32[base + Hit.INST],
            Vec3f32(
                hits[base + Hit.NORMAL + 0],
                hits[base + Hit.NORMAL + 1],
                hits[base + Hit.NORMAL + 2],
            ),
            hits[base + Hit.T],
        )


struct Ray(TrivialRegisterPassable, Writable):
    comptime STRIDE = 8
    comptime ORIGIN = 0  # 0, 1, 2
    comptime T_MIN = 3
    comptime DIRECTION = 4  # 4, 5, 6
    comptime T_MAX = 7

    var o: Point3f32
    var t_min: Float32
    var d: Vec3f32
    var t_max: Float32

    def __init__(
        out self,
        origin: Point3f32,
        direction: Vec3f32,
        t_min: Float32 = 0.0,
        t_max: Float32 = f32_max,
    ):
        self.o = origin
        self.d = direction
        self.t_min = t_min
        self.t_max = t_max

    def __init__[
        origin: ImmutOrigin
    ](out self, rays: UnsafePointer[Float32, origin], ray_idx: Int):
        var base = ray_idx * Ray.STRIDE
        self.o = Point3f32.load(rays, base + Ray.ORIGIN)
        self.t_min = rays[base + Ray.T_MIN]
        self.d = Vec3f32.load(rays, base + Ray.DIRECTION)
        self.t_max = rays[base + Ray.T_MAX]

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

    def origin[width: SIMDSize](self) -> Point3[DType.float32, width]:
        return Point3[DType.float32, width](self.o.x, self.o.y, self.o.z)

    def direction[width: SIMDSize](self) -> Vec3[DType.float32, width]:
        return Vec3[DType.float32, width](self.d.x, self.d.y, self.d.z)

    def rcp_direction[
        width: SIMDSize
    ](self, eps: Float32 = 1.0e-9) -> Vec3[DType.float32, width]:
        var d = self.direction[width]()
        var e = SIMD[DType.float32, width](eps)
        var large = SIMD[DType.float32, width](1.0 / eps)
        var one = SIMD[DType.float32, width](1.0)

        var mx = abs(d.x).gt(e)
        var my = abs(d.y).gt(e)
        var mz = abs(d.z).gt(e)

        var sx = d.x.lt(0.0).select(-large, large)
        var sy = d.y.lt(0.0).select(-large, large)
        var sz = d.z.lt(0.0).select(-large, large)

        var dx = mx.select(d.x, one)
        var dy = my.select(d.y, one)
        var dz = mz.select(d.z, one)

        return Vec3[DType.float32, width](
            mx.select(one / dx, sx),
            my.select(one / dy, sy),
            mz.select(one / dz, sz),
        )


@fieldwise_init
struct Sphere(TrivialRegisterPassable):
    comptime STRIDE = 4
    var center: Point3f32
    var radius: Float32

    def bounds(self) -> AABB:
        var r = Vec3f32(self.radius)
        return AABB(self.center - r, self.center + r)


@fieldwise_init
struct SphereLeafBlock[width: SIMDSize](Copyable):
    var center: Point3[DType.float32, Self.width]
    var radius: SIMD[DType.float32, Self.width]
    var prim_indices: SIMD[DType.uint32, Self.width]

    def __init__(out self):
        self.center = Point3[DType.float32, Self.width](0.0)
        self.radius = SIMD[DType.float32, Self.width](0.0)
        self.prim_indices = SIMD[DType.uint32, Self.width](EMPTY_LANE)


@fieldwise_init
struct TriangleLeafBlock[width: SIMDSize](Copyable):
    var v0: Point3[DType.float32, Self.width]
    var v1: Point3[DType.float32, Self.width]
    var v2: Point3[DType.float32, Self.width]
    var prim_indices: SIMD[DType.uint32, Self.width]

    def __init__(out self):
        self.v0 = Point3[DType.float32, Self.width](0.0)
        self.v1 = Point3[DType.float32, Self.width](0.0)
        self.v2 = Point3[DType.float32, Self.width](0.0)
        self.prim_indices = SIMD[DType.uint32, Self.width](EMPTY_LANE)


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
        self.blas_idx = 0
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
        self.bounds = blas_bounds.apply_transform(transform)
        self.kind = kind


trait TypedBvh:
    def trace[mode: TRACE](self, ray: Ray) -> Hit:
        ...


@fieldwise_init
struct BlasSet[width: SIMDSize]:
    comptime WIDE_NODE_BASE = 0
    comptime LEAF_F32_BASE = 1
    comptime ROOT_IDX = 2
    comptime NODE_COUNT = 3
    comptime LEAF_BLOCK_COUNT = 4
    comptime PRIM_COUNT = 5
    comptime STRIDE = 6

    var descs: DeviceBuffer[DType.uint32]
    var wide_nodes: DeviceBuffer[DType.float32]
    var leaves: DeviceBuffer[DType.float32]
    var blas_count: Int
