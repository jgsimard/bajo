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
from bajo.bvh.constants import f32_max, EMPTY_LANE, PrimitiveKind
from bajo.core.vec import Vec3, Point3, Normal3


@fieldwise_init
struct Hit[frame: Frame = .WORLD, length: SIMDLength = 1](
    TrivialRegisterPassable, Writable
):
    comptime U = 0
    comptime V = 1
    comptime PrimitiveKind = 2
    comptime INST = 3
    comptime NORMAL = 4
    comptime T = 7
    comptime STRIDE = 8

    var u: SIMD[.float32, Self.length]
    var v: SIMD[.float32, Self.length]
    var prim: SIMD[DType.uint32, Self.length]
    var inst: SIMD[DType.uint32, Self.length]
    var normal: Normal3[DType.float32, Self.frame, Self.length]
    var t: SIMD[.float32, Self.length]

    @staticmethod
    def miss(t: SIMD[.float32, Self.length] = f32_max) -> Self:
        return Self(
            0.0,
            0.0,
            EMPTY_LANE,
            EMPTY_LANE,
            Normal3[DType.float32, Self.frame, Self.length](0),
            t,
        )

    @staticmethod
    def shadow_hit() -> Self:
        return Self(
            0.0,
            0.0,
            EMPTY_LANE,
            EMPTY_LANE,
            Normal3[DType.float32, Self.frame, Self.length](0),
            0.0,
        )

    def is_hit(self) -> SIMD[.bool, Self.length]:
        return self.prim.ne(EMPTY_LANE) & self.t.lt(f32_max)

    @always_inline
    def is_occluded(self) -> SIMD[.bool, Self.length]:
        return self.prim.eq(EMPTY_LANE) & self.t.eq(0.0)

    def store(self, hits: MutSpan[Float32, _], idx: Int):
        comptime assert Self.length == 1
        debug_assert["safe", _use_compiler_assume=True](
            idx >= 0 and idx < len(hits) / Self.STRIDE,
            "Hit store is outside the output span",
        )
        self._store_unchecked(hits, idx)

    def _store_unchecked(self, hits: MutSpan[Float32, _], idx: Int):
        """Store after the caller has validated the complete Hit block."""
        comptime assert Self.length == 1
        var base = idx * Hit.STRIDE
        var ptr = hits.unsafe_ptr()

        ptr[unsafe_offset=base + Hit.U] = self.u[0]
        ptr[unsafe_offset=base + Hit.V] = self.v[0]
        ptr[unsafe_offset=base + Hit.NORMAL + 0] = self.normal.x[0]
        ptr[unsafe_offset=base + Hit.NORMAL + 1] = self.normal.y[0]
        ptr[unsafe_offset=base + Hit.NORMAL + 2] = self.normal.z[0]
        ptr[unsafe_offset=base + Hit.T] = self.t[0]

        var hits_u32 = ptr.unsafe_bitcast[UInt32]()
        hits_u32[unsafe_offset=base + Hit.PrimitiveKind] = self.prim[0]
        hits_u32[unsafe_offset=base + Hit.INST] = self.inst[0]

    @staticmethod
    def load(hits: ImmSpan[Float32, _], idx: Int) -> Self:
        comptime assert Self.length == 1
        debug_assert["safe", _use_compiler_assume=True](
            idx >= 0 and idx < len(hits) / Self.STRIDE,
            "Hit load is outside the input span",
        )
        var base = idx * Hit.STRIDE
        var ptr = hits.unsafe_ptr()
        var hits_u32 = ptr.unsafe_bitcast[UInt32]()

        return Self(
            ptr[unsafe_offset=base + Hit.U],
            ptr[unsafe_offset=base + Hit.V],
            hits_u32[unsafe_offset=base + Hit.PrimitiveKind],
            hits_u32[unsafe_offset=base + Hit.INST],
            Normal3[DType.float32, Self.frame, Self.length](
                ptr[unsafe_offset=base + Hit.NORMAL + 0],
                ptr[unsafe_offset=base + Hit.NORMAL + 1],
                ptr[unsafe_offset=base + Hit.NORMAL + 2],
            ),
            ptr[unsafe_offset=base + Hit.T],
        )


@fieldwise_init
struct Sphere[frame: Frame = .WORLD](TrivialRegisterPassable):
    comptime STRIDE = 4
    var center: Point3f32[Self.frame]
    var radius: Float32

    def bounds(self) -> AABB[Self.frame]:
        var r = Vec3f32[Self.frame](self.radius)
        return AABB[Self.frame](self.center - r, self.center + r)


@fieldwise_init
struct SphereLeafBlock[frame: Frame, width: SIMDLength](Copyable):
    var center: Point3[DType.float32, Self.frame, Self.width]
    var radius: SIMD[.float32, Self.width]
    var prim_indices: SIMD[DType.uint32, Self.width]

    def __init__(out self):
        self.center = Point3[DType.float32, Self.frame, Self.width](0.0)
        self.radius = SIMD[.float32, Self.width](0.0)
        self.prim_indices = SIMD[DType.uint32, Self.width](EMPTY_LANE)


@fieldwise_init
struct TriangleLeafBlock[frame: Frame, width: SIMDLength](Copyable):
    var v0: Point3[DType.float32, Self.frame, Self.width]
    var e1: Vec3[.float32, Self.frame, Self.width]
    var e2: Vec3[.float32, Self.frame, Self.width]
    var prim_indices: SIMD[DType.uint32, Self.width]

    def __init__(out self):
        self.v0 = Point3[DType.float32, Self.frame, Self.width](0.0)
        self.e1 = Vec3[.float32, Self.frame, Self.width](0.0)
        self.e2 = Vec3[.float32, Self.frame, Self.width](0.0)
        self.prim_indices = SIMD[DType.uint32, Self.width](EMPTY_LANE)


struct Instance(Copyable):
    """Instance of a BLAS in world space.

    - `transform` maps BLAS-local points/vectors to world space.
    - `inv_transform`
    - `bounds` is the transformed world-space root AABB.
    - `blas_idx` indexes the BLAS array passed to traversal.
    """

    var transform: Affine3f32[.LOCAL, .WORLD]
    var inv_transform: Affine3f32[.WORLD, .LOCAL]
    var bounds: AABB[.WORLD]
    var blas_idx: UInt32
    var kind: PrimitiveKind

    def __init__(out self):
        self.transform = Affine3f32[.LOCAL, .WORLD].identity()
        self.inv_transform = Affine3f32[.WORLD, .LOCAL].identity()
        self.bounds = AABB[.WORLD].invalid()
        self.blas_idx = 0
        self.kind = .UNKNOWN

    def __init__(
        out self,
        transform: Affine3f32[.LOCAL, .WORLD],
        blas_idx: UInt32,
        blas_bounds: AABB[.LOCAL],
        kind: PrimitiveKind,
    ):
        var inverse = transform.inverse()
        debug_assert["safe", _use_compiler_assume=True](
            inverse.mask[0], "instance transform must be invertible"
        )
        self.transform = transform.copy()
        self.inv_transform = inverse.inv.copy()
        self.blas_idx = blas_idx
        self.bounds = blas_bounds.apply_transform(transform)
        self.kind = kind


struct BlasDescLayout:
    comptime NODE_F32_BASE = 0
    comptime LEAF_F32_BASE = 1
    comptime ROOT_IDX = 2
    comptime NODE_COUNT = 3
    comptime LEAF_BLOCK_COUNT = 4
    comptime PRIM_COUNT = 5
    comptime STRIDE = 6

    @staticmethod
    def base(blas_idx: Int) -> Int:
        return blas_idx * Self.STRIDE


@fieldwise_init
struct BlasDesc(TrivialRegisterPassable):
    var node_f32_base: UInt32
    var leaf_f32_base: UInt32
    var root_idx: UInt32
    var node_count: UInt32
    var leaf_block_count: UInt32
    var prim_count: UInt32

    @staticmethod
    def empty(node_f32_base: UInt32, leaf_f32_base: UInt32) -> Self:
        return Self(node_f32_base, leaf_f32_base, 0, 0, 0, 0)

    @staticmethod
    def load(descs: ImmPointer[UInt32, _], blas_idx: UInt32) -> Self:
        var base = BlasDescLayout.base(Int(blas_idx))
        return Self(
            descs[unsafe_offset=base + BlasDescLayout.NODE_F32_BASE],
            descs[unsafe_offset=base + BlasDescLayout.LEAF_F32_BASE],
            descs[unsafe_offset=base + BlasDescLayout.ROOT_IDX],
            descs[unsafe_offset=base + BlasDescLayout.NODE_COUNT],
            descs[unsafe_offset=base + BlasDescLayout.LEAF_BLOCK_COUNT],
            descs[unsafe_offset=base + BlasDescLayout.PRIM_COUNT],
        )

    @always_inline
    def store(self, descs: MutPointer[UInt32, _], blas_idx: Int):
        var base = BlasDescLayout.base(blas_idx)
        descs[
            unsafe_offset=base + BlasDescLayout.NODE_F32_BASE
        ] = self.node_f32_base
        descs[
            unsafe_offset=base + BlasDescLayout.LEAF_F32_BASE
        ] = self.leaf_f32_base
        descs[unsafe_offset=base + BlasDescLayout.ROOT_IDX] = self.root_idx
        descs[unsafe_offset=base + BlasDescLayout.NODE_COUNT] = self.node_count
        descs[
            unsafe_offset=base + BlasDescLayout.LEAF_BLOCK_COUNT
        ] = self.leaf_block_count
        descs[unsafe_offset=base + BlasDescLayout.PRIM_COUNT] = self.prim_count
