from std.utils.numerics import max_finite, min_finite

from bajo.core.vec import Vec3, cross, length, vmin, vmax


struct AABB[dtype: DType](Copyable):
    var _min: Vec3[Self.dtype]
    var _max: Vec3[Self.dtype]

    fn __init__(out self, v0: Vec3[Self.dtype], v1: Vec3[Self.dtype]):
        self._min = vmin(v0, v1)
        self._max = vmax(v0, v1)

    fn __init__(
        out self,
        v0: Vec3[Self.dtype],
        v1: Vec3[Self.dtype],
        v2: Vec3[Self.dtype],
    ):
        self._min = vmin(vmin(v0, v1), v2)
        self._max = vmax(vmax(v0, v1), v2)

    fn grow(mut self, v: Vec3[Self.dtype]):
        self._min = vmin(self._min, v)
        self._max = vmax(self._max, v)

    fn grow(mut self, other: Self):
        self._min = vmin(self._min, other._min)
        self._max = vmax(self._max, other._max)

    fn clear(mut self):
        comptime _min = min_finite[Self.dtype]()
        comptime _max = max_finite[Self.dtype]()
        self._min = Vec3[Self.dtype](_min)
        self._max = Vec3[Self.dtype](_max)

    fn centroid(self) -> Vec3[Self.dtype]:
        return 0.5 * (self._min + self._max)

    fn area(self) -> Scalar[Self.dtype]:
        diff = self._max - self._min
        return diff.x() * diff.y() + diff.y() * diff.z() + diff.z() * diff.x()


@fieldwise_init
struct Triangle[dtype: DType]:
    var v0: Vec3[Self.dtype]
    var v1: Vec3[Self.dtype]
    var v2: Vec3[Self.dtype]

    fn centroid(self) -> Vec3[Self.dtype]:
        return 0.5 * (self.v0 + self.v1 + self.v2)

    fn bounds(self) -> AABB[Self.dtype]:
        return AABB(self.v0, self.v1, self.v2)

    fn normal(self) -> Vec3[Self.dtype]:
        edge0 = self.v1 - self.v0
        edge1 = self.v2 - self.v0
        return cross(edge0, edge1)

    fn area(self) -> Scalar[Self.dtype]:
        normal = self.normal()
        return 0.5 * length(normal)


struct BuildConfig:
    var prioritize_speed: Bool
    """Wether to use 64-bit or 32-bit Morton keys for positional encoding.
    When prioritizeSpeed is set to true, sorting is faster but positional
    encoding has a limited accuracy which results in a lower BVH quality.
    """


@fieldwise_init
struct PrimType:
    var val: UInt8

    comptime AABB = Self(0)
    comptime Triangle = Self(1)


@fieldwise_init
struct Node2(Copyable):
    var bounds: AABB[DType.float32]
    var left_child: Int
    var right_child: Int


struct BVH2[mut: Bool, //, origin: Origin[mut=mut]](Copyable):
    var bounds: AABB[DType.float32]
    var nodes: UnsafePointer[Node2, Self.origin]
    var node_count: Int
    var prim_count: Int


@fieldwise_init
struct NodeExplicit(Copyable):
    """Explicit layout for easier field access."""

    var p: Vec3[DType.float32]
    """Origin point of the local grid (12 bytes)."""

    var e: InlineArray[UInt8, 3]
    """Scale of the grid (3 bytes)."""

    var imask: UInt8
    """8-bit mask to indicate which of the children are internal nodes (1 byte)."""

    var child_base_idx: UInt32
    """Index of the first child (4 bytes)."""

    var prim_base_idx: UInt32
    """Index of the first triangle (4 bytes)."""

    var meta: InlineArray[UInt8, 8]
    """Field encoding the indexing information of every child (8 bytes)."""

    # Quantized origin of the childs' AABBs (8 bytes each)
    var qlox: InlineArray[UInt8, 8]
    var qloy: InlineArray[UInt8, 8]
    var qloz: InlineArray[UInt8, 8]

    # Quantized end point of the childs' AABBs (8 bytes each)
    var qhix: InlineArray[UInt8, 8]
    var qhiy: InlineArray[UInt8, 8]
    var qhiz: InlineArray[UInt8, 8]


@fieldwise_init
struct Node8(Copyable):
    """Packed layout for SIMD processing (80 bytes total)."""

    var p_e_imask: SIMD[DType.float32, 4]
    """P (12 bytes), e (3 bytes), imask (1 byte)."""

    var childidx_tridx_meta: SIMD[DType.float32, 4]
    """Child base index (4B), triangle base index (4B), meta (8B)."""

    var qlox_qloy: SIMD[DType.float32, 4]
    """Qlox (8 bytes), qloy (8 bytes)."""

    var qloz_qhix: SIMD[DType.float32, 4]
    """Qloz (8 bytes), qhix (8 bytes)."""

    var qhiy_qhiz: SIMD[DType.float32, 4]
    """Qhiy (8 bytes), qhiz (8 bytes)."""


struct BVH8[mut: Bool, //, origin: Origin[mut=mut]](Copyable):
    var nodes: UnsafePointer[Node8, Self.origin]
    var node_count: Int

    var prim_idx: UnsafePointer[UInt32, Self.origin]
    var prim_count: Int

    # Root bounds
    var bounds: AABB[DType.float32]


def main() raises:
    print("hello bajo.bvh.hploc")
