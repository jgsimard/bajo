from bit import count_leading_zeros
from math import clamp
from std.utils.numerics import max_finite, min_finite


from bajo.core.vec import Vec3, Vec3f32, vmin, vmax


comptime BVH_SHARED_STACK = False

comptime SAH_NUM_BUCKETS = 16
comptime USE_LOAD4 = True
comptime BVH_QUERY_STACK_SIZE = 32
comptime BVH_BLOCK_DIM = 256


@fieldwise_init
struct BvhConstructor(
    Equatable,
    Hashable,
    ImplicitlyCopyable,
    KeyElement,
    TrivialRegisterPassable,
    Writable,
):
    var v: Int

    comptime SAH = Self(0)
    comptime MEDIAN = Self(1)
    comptime LBVH = Self(2)


@fieldwise_init
struct Bounds3[dtype: DType](Copyable, Defaultable, Writable):
    var lower: Vec3[Self.dtype]
    var upper: Vec3[Self.dtype]

    fn __init__(out self):
        comptime max_val = max_finite[Self.dtype]()
        comptime min_val = min_finite[Self.dtype]()
        self.lower = Vec3[Self.dtype](max_val)
        self.upper = Vec3[Self.dtype](min_val)

    fn center(self) -> Vec3[Self.dtype]:
        return (self.lower + self.upper) * 0.5

    fn edges(self) -> Vec3[Self.dtype]:
        return self.upper - self.lower

    fn expand(mut self, r: Scalar[Self.dtype]):
        self.lower = self.lower - r
        self.upper = self.upper + r

    fn expand(mut self, r: Vec3[Self.dtype]):
        self.lower = self.lower - r
        self.upper = self.upper + r

    fn empty(self) -> Bool:
        comptime for i in range(3):
            if self.lower[i] >= self.upper[i]:
                return True
        return False

    fn overlaps(self, p: Vec3[Self.dtype]) -> Bool:
        comptime for i in range(3):
            if p[i] < self.lower[i] or p[i] > self.upper[i]:
                return False
        return True

    fn overlaps(self, b: Bounds3[Self.dtype]) -> Bool:
        comptime for i in range(3):
            if self.lower[i] > b.upper[i] or self.upper[i] < b.lower[i]:
                return False
        return True

    fn overlaps(
        self, b_lower: Vec3[Self.dtype], b_upper: Vec3[Self.dtype]
    ) -> Bool:
        comptime for i in range(3):
            if self.lower[i] > b_upper[i] or self.upper[i] < b_lower[i]:
                return False
        return True

    fn add_point(mut self, p: Vec3[Self.dtype]):
        self.lower = vmin(self.lower, p)
        self.upper = vmax(self.upper, p)

    fn add_bounds(
        mut self, lower_other: Vec3[Self.dtype], upper_other: Vec3[Self.dtype]
    ):
        self.lower = vmin(self.lower, lower_other)
        self.upper = vmax(self.upper, upper_other)

    fn area(self) -> Scalar[Self.dtype]:
        e = self.edges()
        return Scalar[Self.dtype](2.0) * (
            e[0] * e[1] + e[0] * e[2] + e[1] * e[2]
        )


fn bounds_union[
    dtype: DType
](a: Bounds3[dtype], b: Vec3[dtype]) -> Bounds3[dtype]:
    return Bounds3(vmin(a.lower, b), vmax(a.upper, b))


fn bounds_union[
    dtype: DType
](a: Bounds3[dtype], b: Bounds3[dtype]) -> Bounds3[dtype]:
    return Bounds3(vmin(a.lower, b.lower), vmax(a.upper, b.upper))


fn bounds_intersection[
    dtype: DType
](a: Bounds3[dtype], b: Bounds3[dtype]) -> Bounds3[dtype]:
    return Bounds3(vmax(a.lower, b.lower), vmin(a.upper, b.upper))


struct BVHPackedNodeHalf(Copyable, TrivialRegisterPassable):
    """For non-leaf nodes:
    - 'lower.i' represents the index of the left child node.
    - 'upper.i' represents the index of the right child node.

    For leaf nodes:
    - 'lower.i' indicates the start index of the primitives in 'primitive_indices'.
    - 'upper.i' indicates the index just after the last primitive in 'primitive_indices'.

    TODO: might be cool to use SIMD[DType.float32, 4] and rebind the last float to ib
    """

    var x: Float32
    var y: Float32
    var z: Float32
    var ib: UInt32
    """Two parts : i = 31 first bits, b = last bit."""

    comptime INDEX_MASK: UInt32 = 0x7FFFFFFF
    comptime LEAF_BIT: UInt32 = 0x80000000

    fn index(self) -> UInt32:
        return self.ib & Self.INDEX_MASK

    fn is_leaf(self) -> Bool:
        return (self.ib & Self.LEAF_BIT) != 0

    fn set_leaf(mut self, is_leaf: Bool):
        leaf_val = Self.LEAF_BIT if is_leaf else UInt32(0)
        self.ib = (self.ib & Self.INDEX_MASK) | leaf_val

    fn set_index(mut self, idx: UInt32):
        self.ib = (idx & Self.INDEX_MASK) | (self.ib & Self.LEAF_BIT)

    @staticmethod
    fn set_index_and_leaf(idx: UInt32, is_leaf: Bool) -> UInt32:
        leaf_val = Self.LEAF_BIT if is_leaf else UInt32(0)
        return (idx & Self.INDEX_MASK) | leaf_val

    fn __init__(out self, bound: Vec3f32, child: Int, leaf: Bool):
        self.x = bound.x()
        self.y = bound.y()
        self.z = bound.z()
        self.ib = Self.set_index_and_leaf(UInt32(child), leaf)


fn part1by2(n: UInt32) -> UInt32:
    m = n
    m = (m ^ (m << 16)) & 0xFF0000FF
    m = (m ^ (m << 8)) & 0x0300F00F
    m = (m ^ (m << 4)) & 0x030C30C3
    m = (m ^ (m << 2)) & 0x09249249
    return m


fn morton3[dim: UInt32](x: Float32, y: Float32, z: Float32) -> UInt32:
    """Takes values in the range [0, 1] and assigns an index based Morton codes of length 3*dim bits.
    """
    comptime dimf = Float32(dim)
    ux = clamp(UInt32(x * dimf), 0, dim - 1)
    uy = clamp(UInt32(y * dimf), 0, dim - 1)
    uz = clamp(UInt32(z * dimf), 0, dim - 1)
    return (part1by2(uz) << 2) | (part1by2(uy) << 1) | part1by2(ux)


@fieldwise_init
struct BVH[origin: Origin](Copyable):
    var node_lowers: UnsafePointer[BVHPackedNodeHalf, Self.origin]
    var node_uppers: UnsafePointer[BVHPackedNodeHalf, Self.origin]

    var primitive_indices: UnsafePointer[Int, Self.origin]
    """Reordered primitive indices corresponds to the ordering of leaf nodes."""

    # Refit / Topology pointers
    var node_parents: UnsafePointer[Int, Self.origin]
    var node_counts: UnsafePointer[Int, Self.origin]

    # Hierarchy Metadata
    var max_depth: Int
    var max_nodes: Int
    var num_nodes: Int
    var num_leaf_nodes: Int
    """Since we use packed leaf nodes, the number of them is no longer the number of items, but variable."""

    var root: UnsafePointer[Int, Self.origin]
    """Pointer (CPU or GPU) to a single integer index in node_lowers, node_uppers
       representing the root of the tree, this is not always the first node
       for bottom-up builders."""

    var item_lowers: UnsafePointer[Vec3f32, Self.origin]
    var item_uppers: UnsafePointer[Vec3f32, Self.origin]
    var item_groups: UnsafePointer[Int, Self.origin]
    var num_items: Int
    var leaf_size: Int

    # var context: AnyPointer
    # """gpu context"""


@fieldwise_init
struct BvhQuery:
    """Object used to track state during BVH traversal. AKA bvh_query_t."""

    ...
    var v: Int


@fieldwise_init
struct BvhQueryTiled:
    """Object used to track state during thread-block parallel BVH traversal. AKA bvh_query_thread_block_t
    """

    ...
    var v: Int


fn main():
    print("hello warp.bvh")
