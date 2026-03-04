# Mojo implementation of tinybvh
from std.sys.info import size_of
from std.reflection import get_type_name
from ffi import UnsafeUnion

from bajo.core.vec import Vec3f32, dot
from bajo.core.mat import Mat44f32


comptime BVHBINS = 8
comptime HQBVHBINS = 8
comptime MAXHQBINS = 256
comptime AVXBINS = 8  # // must stay at 8.
comptime INST_IDX_BITS = 32
"""Use 4..~12 to use prim field bits for instance id, or set to 32 to store index in separate field."""


@fieldwise_init
struct AABB:
    var minBounds: Vec3f32
    var dummy1: UInt32
    var maxBounds: Vec3f32
    var dummy2: UInt32


comptime userChar = InlineArray[UInt8, 56]
comptime userFloat = InlineArray[Float32, 14]
comptime userInt32 = InlineArray[UInt32, 14]
comptime userDouble = InlineArray[Float64, 7]
comptime userInt64 = InlineArray[UInt64, 7]

comptime IntersectionData = UnsafeUnion[
    userChar, userFloat, userInt32, userDouble, userInt64
]


@fieldwise_init
struct Intersection(Copyable):
    """An intersection result is designed to fit in no more than
    four 32-bit values. This allows efficient storage of a result in
    GPU code. The obvious missing result is an instance id; consider
    squeezing this in the 'prim' field in some way.
    Using this data and the original triangle data, all other info for
    shading (such as normal, texture color etc.) can be reconstructed.
    """

    var inst: Int
    """Instance index. Stored in top bits of prim if INST_IDX_BITS != 32."""
    var t: Float32  # Distance along ray
    var u: Float32  # Barycentric U
    var v: Float32  # Barycentric V
    var prim: Int
    """Primitive index."""

    # 64 byte of custom data -
    # assuming struct Ray is aligned, this starts at a cache line boundary.
    var auxData: UnsafePointer[Int, MutAnyOrigin]
    var userData: IntersectionData


comptime RAY_MASK_INTERSECT_ALL = 0xFFFF


struct Ray(Copyable):
    var O: Vec3f32
    var mask: Int
    var D: Vec3f32
    var instIdx: Int
    var rD: Vec3f32
    # var dummy2: UInt32
    # var hit: Intersection


def print_size_of[type: AnyType]():
    comptime name = get_type_name[type]()
    size_bytes = size_of[type]()
    size_32 = size_bytes / 4
    print(t"{name}: {size_bytes} bytes, {size_32} x 32 bits")


def main() raises:
    print("tinybvh hello")
    print_size_of[AABB]()
    print_size_of[Intersection]()
    print_size_of[IntersectionData]()
    print_size_of[Ray]()
