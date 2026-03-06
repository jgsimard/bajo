from bit import count_leading_zeros
from math import clamp
from std.utils.numerics import max_finite, min_finite
from sys.info import is_gpu
from os import abort

from bajo.core.intersect import intersect_ray_aabb, intersect_aabb_aabb
from bajo.core.sort import nth_element
from std.builtin.sort import _quicksort_partition_right
from bajo.core.vec import Vec3, Vec3f32, vmin, vmax, longest_axis

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


comptime Bounds3f32 = Bounds3[DType.float32]


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

    fn add_bounds(mut self, other: Self):
        self.lower = vmin(self.lower, other.lower)
        self.upper = vmax(self.upper, other.upper)

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

    fn index(self) -> Int:
        return Int(self.ib & Self.INDEX_MASK)

    fn is_leaf(self) -> Bool:
        return (self.ib & Self.LEAF_BIT) != 0

    fn set_leaf(mut self, is_leaf: Bool):
        leaf_val = Self.LEAF_BIT if is_leaf else UInt32(0)
        self.ib = (self.ib & Self.INDEX_MASK) | leaf_val

    fn set_index(mut self, idx: UInt32):
        self.ib = (idx & Self.INDEX_MASK) | (self.ib & Self.LEAF_BIT)

    @staticmethod
    fn index_and_leaf(idx: UInt32, is_leaf: Bool) -> UInt32:
        leaf_val = Self.LEAF_BIT if is_leaf else UInt32(0)
        return (idx & Self.INDEX_MASK) | leaf_val

    fn __init__(out self, bound: Vec3f32, child: Int, leaf: Bool):
        self.x = bound.x()
        self.y = bound.y()
        self.z = bound.z()
        self.ib = Self.index_and_leaf(UInt32(child), leaf)


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

    fn num_bounds(self) -> Int:
        return self.num_items

    fn leaf_group(self, leaf: Int) -> Int:
        if not self.item_groups:
            return 0
        idx_p = self.node_lowers[leaf].index()
        idx_g = self.primitive_indices[idx_p]
        return self.item_groups[idx_g]

    fn lower_bound_group(self, group: Int) -> Int:
        lo = 0
        hi = self.num_leaf_nodes

        while lo < hi:
            mid = (lo + hi) >> 1
            if self.leaf_group(mid) < group:
                lo = mid + 1
            else:
                hi = mid

        if lo == self.num_leaf_nodes or (self.leaf_group(lo) != group):
            return -1
        return lo

    fn lca(self, node_a: Int, node_b: Int) -> Int:
        """Lowest Common Ancestor."""
        da = 0
        db = 0

        t = node_a
        while t != -1:
            da += 1
            t = self.node_parents[t]

        t = node_b
        while t != -1:
            db += 1
            t = self.node_parents[t]

        curr_a = node_a
        curr_b = node_b
        if da > db:
            diff = da - db
            for _ in range(diff):
                if curr_a == -1:
                    break
                curr_a = self.node_parents[curr_a]
        elif db > da:
            diff = db - da
            for _ in range(diff):
                if curr_b == -1:
                    break
                curr_b = self.node_parents[curr_b]

        while curr_a != curr_b:
            if curr_a == -1 or curr_b == -1:
                return -1
            curr_a = self.node_parents[curr_a]
            curr_b = self.node_parents[curr_b]

        return curr_a  # either the LCA or -1

    fn group_root(self, group_id: Int) -> Int:
        # this function requires all the leaf nodes to be stored as the first bvh.num_leaf_nodes nodes
        # and sorted by their group ids

        # locate first leaf of the current group
        first = self.lower_bound_group(group_id)
        if first < 0:
            return -1

        # find first leaf of next group to find the last leaf of the current group
        next_group = group_id + 1
        next = self.lower_bound_group(next_group)
        last = (self.num_leaf_nodes if next < 0 else next) - 1

        return self.lca(first, last)


fn calc_bounds(
    lowers: UnsafePointer[Vec3f32, ImmutAnyOrigin],
    uppers: UnsafePointer[Vec3f32, ImmutAnyOrigin],
    indices: UnsafePointer[Int, ImmutAnyOrigin],
    start: Int,
    end: Int,
) -> Bounds3f32:
    u = Bounds3f32()
    for i in range(start, end):
        idx = indices[i]
        u.add_bounds(lowers[idx], uppers[idx])
    return u^


fn partition_median[
    origin: MutOrigin
](
    lowers: UnsafePointer[Vec3f32, origin],
    uppers: UnsafePointer[Vec3f32, origin],
    indices: UnsafePointer[Int, origin],
    start: Int,
    end: Int,
    range_bounds: Bounds3[DType.float32],
) -> Int:
    len_span = end - start
    debug_assert["safe"](len_span >= 2)

    axis = longest_axis(range_bounds.edges())
    k = (start + end) // 2

    fn compare_centers(a: Int, b: Int) capturing -> Bool:
        center_a = 0.5 * (lowers[a] + uppers[a])
        center_b = 0.5 * (lowers[b] + uppers[b])
        return center_a[axis] < center_b[axis]

    indices_span = Span[Int, origin](ptr=indices + start, length=len_span)

    # Use relative index `k - start` because Span starts at index 0
    nth_element[compare_centers](indices_span, k - start)

    return k


@fieldwise_init
struct SahIndice(TrivialRegisterPassable):
    var split_axis: Int
    var split_point: Float32


fn partition_sah_indices(
    lowers: UnsafePointer[Vec3f32, ImmutAnyOrigin],
    uppers: UnsafePointer[Vec3f32, ImmutAnyOrigin],
    indices: UnsafePointer[Int, ImmutAnyOrigin],
    start: Int,
    end: Int,
    range_bounds: Bounds3[DType.float32],
) -> SahIndice:
    """
    Computes the optimal split point and axis using the Surface Area Heuristic.
    """
    # 1. Compute centroid bounds
    centroid_bounds = Bounds3[DType.float32]()
    for i in range(start, end):
        idx = indices[i]
        item_center = (lowers[idx] + uppers[idx]) * 0.5
        centroid_bounds.add_point(item_center)

    edges = centroid_bounds.edges()
    split_axis = longest_axis(edges)

    range_start = centroid_bounds.lower[split_axis]
    range_end = centroid_bounds.upper[split_axis]

    # Guard against zero extent along the split axis
    if range_end <= range_start:
        return SahIndice(split_axis, range_start)

    # binning
    buckets_counts = InlineArray[Int, SAH_NUM_BUCKETS](fill=0)
    buckets = InlineArray[Bounds3[DType.float32], SAH_NUM_BUCKETS](
        fill=Bounds3[DType.float32]()
    )

    inv_range = Float32(SAH_NUM_BUCKETS) / (range_end - range_start)

    for i in range(start, end):
        idx = indices[i]
        center_val = 0.5 * (lowers[idx][split_axis] + uppers[idx][split_axis])

        bucket_idx = clamp(
            Int((center_val - range_start) * inv_range), 0, SAH_NUM_BUCKETS - 1
        )

        # contains default value so no need to check
        buckets[bucket_idx].add_bounds(lowers[idx], uppers[idx])
        buckets_counts[bucket_idx] += 1

    # n-1 split planes for n buckets
    left_areas = InlineArray[Float32, SAH_NUM_BUCKETS - 1](fill=0.0)
    right_areas = InlineArray[Float32, SAH_NUM_BUCKETS - 1](fill=0.0)
    counts_l = InlineArray[Int, SAH_NUM_BUCKETS - 1](fill=0)
    counts_r = InlineArray[Int, SAH_NUM_BUCKETS - 1](fill=0)

    left = Bounds3[DType.float32]()
    right = Bounds3[DType.float32]()
    count_l = 0
    count_r = 0

    for i in range(SAH_NUM_BUCKETS - 1):
        ref bound_start = buckets[i]
        ref bound_end = buckets[SAH_NUM_BUCKETS - i - 1]

        left = bounds_union(left, bound_start)
        right = bounds_union(right, bound_end)

        left_areas[i] = left.area()
        right_areas[SAH_NUM_BUCKETS - i - 2] = right.area()

        count_l += buckets_counts[i]
        count_r += buckets_counts[SAH_NUM_BUCKETS - i - 1]

        counts_l[i] = count_l
        counts_r[SAH_NUM_BUCKETS - i - 2] = count_r

    inv_total_area = 1.0 / range_bounds.area()

    # find split point i that minimizes area(left[i]) * count[left[i]] + area(right[i]) * count[right[i]]
    min_cost = max_finite[DType.float32]()
    min_split = 0
    for i in range(SAH_NUM_BUCKETS - 1):
        p_below = left_areas[i] * inv_total_area
        p_above = right_areas[i] * inv_total_area

        cost = p_below * Float32(counts_l[i]) + p_above * Float32(counts_r[i])

        if cost < min_cost:
            min_cost = cost
            min_split = i

    debug_assert["safe"](min_split >= 0 and min_split < SAH_NUM_BUCKETS - 1)

    split_point = range_start + (Float32(min_split) + 1.0) * (
        range_end - range_start
    ) / Float32(SAH_NUM_BUCKETS)

    return SahIndice(split_axis, split_point)


struct BvhStack[origin: Origin, device: String](Copyable, Writable):
    # represents a strided stack in shared memory
    # so each level of the stack is stored contiguously
    # across the block

    # cpu
    comptime local_size = BVH_QUERY_STACK_SIZE if Self.device == "cpu" else 0
    var _local: InlineArray[Int, Self.local_size]

    # # gpu
    var _ptr: UnsafePointer[Int, Self.origin]

    fn __init__(out self):
        comptime assert Self.device == "cpu"

        self._ptr = UnsafePointer[Int, Self.origin]()
        self._local = InlineArray[Int, Self.local_size](fill=0)

    fn __init__(out self, ptr: UnsafePointer[Int, Self.origin], root: Int):
        self._ptr = ptr
        self._local = InlineArray[Int, Self.local_size](fill=0)
        self._local[0] = root

    @always_inline
    fn __getitem__(self, depth: Int) -> Int:
        comptime if self.device == "cpu":
            return self._local[depth]
        else:
            comptime assert False, "GPU not supported yet"
            # return self._ptr[depth * BVH_BLOCK_DIM]

    @always_inline
    fn __setitem__(mut self, depth: Int, value: Int):
        comptime if self.device == "cpu":
            self._local[depth] = value
        else:
            comptime assert False, "GPU not supported yet"
            # self._ptr[depth * BVH_BLOCK_DIM] = value


@fieldwise_init
struct BvhQuery[origin: Origin, device: String](Copyable, Writable):
    """Object used to track state during BVH traversal. AKA bvh_query_t."""

    var bvh: UnsafePointer[BVH[Self.origin], Self.origin]
    var stack: BvhStack[Self.origin, Self.device]
    var count: Int
    var primitive_counter: Int
    var input_lower: Vec3f32
    var input_upper: Vec3f32
    var bounds_nr: Int
    var is_ray: Bool

    fn __init__(
        out self,
        bvh: UnsafePointer[BVH[Self.origin], Self.origin],
        is_ray: Bool,
        lower: Vec3f32,
        upper: Vec3f32,
        root: Int,
    ):
        self.bvh = bvh
        self.stack = BvhStack[Self.origin, self.device](bvh[].root, root)
        self.count = 1
        self.primitive_counter = 0
        self.input_lower = lower.copy()
        self.input_upper = upper.copy()
        self.bounds_nr = -1
        self.is_ray = is_ray

    @staticmethod
    fn from_aabb(
        bvh: UnsafePointer[BVH[Self.origin], Self.origin],
        lower: Vec3f32,
        upper: Vec3f32,
        root: Int,
    ) -> Self:
        return Self(bvh, False, lower, upper, root)

    @staticmethod
    fn from_ray(
        bvh: UnsafePointer[BVH[Self.origin], Self.origin],
        start: Vec3f32,
        dir: Vec3f32,
        root: Int,
    ) -> Self:
        return Self(bvh, True, start, 1.0 / dir, root)

    fn intersection_test(
        self, node_lower: Vec3f32, node_upper: Vec3f32, mut t: Float32
    ) -> Bool:
        if self.is_ray:
            return intersect_ray_aabb(
                self.input_lower, self.input_upper, node_lower, node_upper, t
            )
        else:
            return intersect_aabb_aabb(
                self.input_lower, self.input_upper, node_lower, node_upper
            )

    fn next(mut self, mut index: Int, max_dist: Float32) -> Bool:
        comptime flt_max = max_finite[DType.float32]()
        comptime flt_min = min_finite[DType.float32]()

        # Extract the struct from the pointer to easily access fields
        ref bvh = self.bvh[]

        # Navigate through the BVH, find the first overlapping leaf node.
        while self.count > 0:
            self.count -= 1
            var node_index = self.stack[self.count]

            ref node_lower = bvh.node_lowers[node_index]
            ref node_upper = bvh.node_uppers[node_index]

            if self.primitive_counter == 0:
                t = flt_max
                lower = Vec3f32(node_lower.x, node_lower.y, node_lower.z)
                upper = Vec3f32(node_upper.x, node_upper.y, node_upper.z)
                var hit = self.intersection_test(lower, upper, t)
                if not hit or (self.is_ray and t >= max_dist):
                    continue

            left_index = node_lower.index()
            right_index = node_upper.index()

            if node_lower.is_leaf():
                start = left_index
                end = right_index

                # Fast path when the actual leaf range contains exactly one primitive
                if end - start == 1:
                    primitive_index = bvh.primitive_indices[start]
                    index = primitive_index
                    self.bounds_nr = primitive_index
                    return True

                else:
                    primitive_index = bvh.primitive_indices[
                        start + self.primitive_counter
                    ]
                    self.primitive_counter += 1

                    # if already visited the last primitive in the leaf node
                    # move to the next node and reset the primitive counter to 0
                    if start + self.primitive_counter == end:
                        self.primitive_counter = 0
                    else:
                        # otherwise we need to keep this leaf node in stack for a future visit
                        self.stack[self.count] = node_index
                        self.count += 1

                    t = flt_max
                    lower = bvh.item_lowers[primitive_index].copy()
                    upper = bvh.item_uppers[primitive_index].copy()
                    var hit = self.intersection_test(lower, upper, t)
                    if not hit or (self.is_ray and t >= max_dist):
                        continue

                    index = primitive_index
                    self.bounds_nr = primitive_index
                    return True
            else:
                # if it's not a leaf node we treat it as if we have visited the last primitive
                self.primitive_counter = 0
                self.stack[self.count] = left_index
                self.count += 1
                self.stack[self.count] = right_index
                self.count += 1

        return False


@fieldwise_init
struct BvhQueryTiled:
    """Object used to track state during thread-block parallel BVH traversal. AKA bvh_query_thread_block_t.
    """

    var v: Int


fn main():
    print("hello warp.bvh")
