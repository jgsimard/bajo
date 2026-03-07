from std.utils.numerics import max_finite, min_finite

from bajo.core.vec import Vec3, Vec3f32, cross, length, vmin, vmax
from bajo.core.aabb import AABB


comptime float4 = SIMD[DType.float32, 4]


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
    var bounds: AABB
    var left_child: Int
    var right_child: Int


struct BVH2[mut: Bool, //, origin: Origin[mut=mut]](Copyable):
    var bounds: AABB
    var nodes: UnsafePointer[Node2, Self.origin]
    var node_count: Int
    var prim_count: Int


@fieldwise_init
struct NodeExplicit(Copyable):
    """Explicit layout for easier field access."""

    var p: Vec3f32
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

    var p_e_imask: float4
    """P (12 bytes), e (3 bytes), imask (1 byte)."""

    var childidx_tridx_meta: float4
    """Child base index (4B), triangle base index (4B), meta (8B)."""

    var qlox_qloy: float4
    """Qlox (8 bytes), qloy (8 bytes)."""

    var qloz_qhix: float4
    """Qloz (8 bytes), qhix (8 bytes)."""

    var qhiy_qhiz: float4
    """Qhiy (8 bytes), qhiz (8 bytes)."""


struct BVH8[mut: Bool, //, origin: Origin[mut=mut]](Copyable):
    var nodes: UnsafePointer[Node8, Self.origin]
    var node_count: Int

    var prim_idx: UnsafePointer[UInt32, Self.origin]
    var prim_count: Int

    # Root bounds
    var bounds: AABB


fn divide_round_up(x: UInt32, y: UInt32) -> UInt32:
    return 1 + (x - 1) / y


@fieldwise_init
struct BVH2BuildState[mut: Bool, //, origin: Origin[mut=mut]](Copyable):
    var scene_bounds: UnsafePointer[AABB, Self.origin]
    """Scene bounds."""

    var nodes: UnsafePointer[Node2, Self.origin]
    """BVH2 nodes."""

    var cluster_idx: UnsafePointer[UInt32, Self.origin]
    """Cluster indices."""

    var parent_idx: UnsafePointer[UInt32, Self.origin]
    """BVH2 parent indices."""

    var prim_count: UInt32
    """Number of primitives."""

    var cluster_count: UnsafePointer[UInt32, Self.origin]
    """Number of merged clusters."""


@fieldwise_init
struct BVH8BuildState[mut: Bool, //, origin: Origin[mut=mut]](Copyable):
    var bvh2_nodes: UnsafePointer[Node2, Self.origin]
    """BVH2 nodes."""

    var bvh8_nodes: UnsafePointer[Node8, Self.origin]
    """BVH8 nodes."""

    var prim_idx: UnsafePointer[UInt32, Self.origin]
    """Primitive indices."""

    var prim_count: UInt32
    """Number of primitives."""

    var node_counter: UnsafePointer[UInt32, Self.origin]
    """Number of BVH8 nodes (Atomic)."""

    var leaf_counter: UnsafePointer[UInt32, Self.origin]
    """Number of leaves (Atomic)."""

    var index_pairs: UnsafePointer[UInt64, Self.origin]
    """Index pairs."""

    var work_counter: UnsafePointer[UInt32, Self.origin]
    """Atomic counter to dispatch work items."""

    var work_alloc_counter: UnsafePointer[UInt32, Self.origin]
    """Atomic counter to dispatch work items."""


@always_inline
fn interleave_bits_32(val: UInt32) -> UInt32:
    x = val
    x = (x | (x << 16)) & 0x30000FF
    x = (x | (x << 8)) & 0x300F00F
    x = (x | (x << 4)) & 0x30C30C3
    x = (x | (x << 2)) & 0x9249249
    return x


@always_inline
fn interleave_bits_64(val: UInt64) -> UInt64:
    x = val
    x = (x | (x << 32)) & 0x1F00000000FFFF
    x = (x | (x << 16)) & 0x1F0000FF0000FF
    x = (x | (x << 8)) & 0x100F00F00F00F00F
    x = (x | (x << 4)) & 0x10C30C30C30C30C3
    x = (x | (x << 2)) & 0x1249249249249249
    return x


fn morton_code_32(v: Vec3[DType.float32]) -> UInt32:
    x = UInt32(v.x() * 0x3FF)
    y = UInt32(v.y() * 0x3FF)
    z = UInt32(v.z() * 0x3FF)
    return (
        interleave_bits_32(x)
        | (interleave_bits_32(y) << 1)
        | (interleave_bits_32(z) << 2)
    )


fn morton_code_64(v: Vec3[DType.float32]) -> UInt64:
    x = UInt32(v.x() * 0x1FFFFF)
    y = UInt32(v.y() * 0x1FFFFF)
    z = UInt32(v.z() * 0x1FFFFF)
    return (
        interleave_bits_64(UInt64(x))
        | (interleave_bits_64(UInt64(y)) << 1)
        | (interleave_bits_64(UInt64(z)) << 2)
    )


def main() raises:
    print("hello bajo.bvh.hploc")
