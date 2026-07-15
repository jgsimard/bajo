from std.gpu import DeviceBuffer

from bajo.core import (
    AABB,
    Vec3f32,
    Normal3f32,
    Affine3f32,
    Point3f32,
    Frame,
    Ray,
    Rayf32,
)
from bajo.bvh.constants import f32_max, EMPTY_LANE, Primitive, TRACE
from bajo.core.vec import Vec3, Point3


@fieldwise_init
struct Hit[frame: Frame = Frame.WORLD](TrivialRegisterPassable, Writable):
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
    var normal: Normal3f32[Self.frame]
    var t: Float32

    @staticmethod
    def miss(t: Float32 = f32_max) -> Self:
        return Self(
            0.0,
            0.0,
            EMPTY_LANE,
            EMPTY_LANE,
            Normal3f32[Self.frame](0),
            t,
        )

    @staticmethod
    def shadow_hit() -> Self:
        return Self(
            0.0,
            0.0,
            EMPTY_LANE,
            EMPTY_LANE,
            Normal3f32[Self.frame](0),
            0.0,
        )

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
    ) -> Self:
        var base = idx * Hit.STRIDE
        var hits_u32 = hits.bitcast[UInt32]()

        return Self(
            hits[base + Hit.U],
            hits[base + Hit.V],
            hits_u32[base + Hit.PRIM],
            hits_u32[base + Hit.INST],
            Normal3f32[Self.frame](
                hits[base + Hit.NORMAL + 0],
                hits[base + Hit.NORMAL + 1],
                hits[base + Hit.NORMAL + 2],
            ),
            hits[base + Hit.T],
        )


@fieldwise_init
struct Sphere[frame: Frame = Frame.WORLD](TrivialRegisterPassable):
    comptime STRIDE = 4
    var center: Point3f32[Self.frame]
    var radius: Float32

    def bounds(self) -> AABB[Self.frame]:
        var r = Vec3f32[Self.frame](self.radius)
        return AABB[Self.frame](self.center - r, self.center + r)


@fieldwise_init
struct SphereLeafBlock[frame: Frame, width: SIMDSize](Copyable):
    var center: Point3[DType.float32, Self.frame, Self.width]
    var radius: SIMD[DType.float32, Self.width]
    var prim_indices: SIMD[DType.uint32, Self.width]

    def __init__(out self):
        self.center = Point3[DType.float32, Self.frame, Self.width](0.0)
        self.radius = SIMD[DType.float32, Self.width](0.0)
        self.prim_indices = SIMD[DType.uint32, Self.width](EMPTY_LANE)


@fieldwise_init
struct TriangleLeafBlock[frame: Frame, width: SIMDSize](Copyable):
    var v0: Point3[DType.float32, Self.frame, Self.width]
    var v1: Point3[DType.float32, Self.frame, Self.width]
    var v2: Point3[DType.float32, Self.frame, Self.width]
    var prim_indices: SIMD[DType.uint32, Self.width]

    def __init__(out self):
        self.v0 = Point3[DType.float32, Self.frame, Self.width](0.0)
        self.v1 = Point3[DType.float32, Self.frame, Self.width](0.0)
        self.v2 = Point3[DType.float32, Self.frame, Self.width](0.0)
        self.prim_indices = SIMD[DType.uint32, Self.width](EMPTY_LANE)


struct Instance(Copyable):
    """Instance of a BLAS in world space.

    - `transform` maps BLAS-local points/vectors to world space.
    - `inv_transform` maps world-space rays to BLAS-local space.
    - `bounds` is the transformed world-space root AABB.
    - `blas_idx` indexes the BLAS array passed to traversal.
    """

    var transform: Affine3f32[Frame.LOCAL, Frame.WORLD]
    var inv_transform: Affine3f32[Frame.WORLD, Frame.LOCAL]
    var bounds: AABB[Frame.WORLD]
    var blas_idx: UInt32
    var kind: Primitive

    def __init__(out self):
        self.transform = Affine3f32[Frame.LOCAL, Frame.WORLD].identity()
        self.inv_transform = Affine3f32[Frame.WORLD, Frame.LOCAL].identity()
        self.bounds = AABB[Frame.WORLD].invalid()
        self.blas_idx = 0
        self.kind = Primitive.UNKNOWN

    def __init__(
        out self,
        transform: Affine3f32[Frame.LOCAL, Frame.WORLD],
        inv_transform: Affine3f32[Frame.WORLD, Frame.LOCAL],
        blas_idx: UInt32,
        blas_bounds: AABB[Frame.LOCAL],
        kind: Primitive,
    ):
        self.transform = transform.copy()
        self.inv_transform = inv_transform.copy()
        self.blas_idx = blas_idx
        self.bounds = blas_bounds.apply_transform(transform)
        self.kind = kind


# TODO: use paramatric traits when they will be finally introduced
trait TypedBvh:
    comptime bvh_frame: Frame

    def trace[
        mode: TRACE
    ](self, ray: Rayf32[Self.bvh_frame]) -> Hit[Self.bvh_frame]:
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
