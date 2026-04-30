from std.algorithm import parallelize
from std.math import abs, min, max, clamp
from std.memory import UnsafePointer
from std.atomic import Atomic
from std.utils.numerics import max_finite, min_finite
from std.bit import count_leading_zeros

from bajo.core.aabb import AABB
from bajo.core.intersect import intersect_ray_tri_moller, intersect_ray_aabb
from bajo.core.mat import Mat44f32, transform_point, transform_vector, inverse
from bajo.core.vec import Vec3f32, vmin, vmax, longest_axis, InlineArray
from bajo.core.morton import morton3

comptime f32_max = max_finite[DType.float32]()
comptime f32_min = min_finite[DType.float32]()
comptime INV_3 = Float32(1.0) / Float32(3.0)
comptime BVH_BINS = 16


@fieldwise_init
struct Intersection(Copyable):
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
struct Ray(Copyable):
    var O: Vec3f32
    var mask: UInt32
    var D: Vec3f32
    var rD: Vec3f32
    var hit: Intersection

    def __init__(out self, O: Vec3f32, D: Vec3f32, t_max: Float32 = f32_max):
        self.O = O.copy()
        self.D = D.copy()
        var rDx = clamp(Float32(1.0) / D.x(), f32_min, f32_max)
        var rDy = clamp(Float32(1.0) / D.y(), f32_min, f32_max)
        var rDz = clamp(Float32(1.0) / D.z(), f32_min, f32_max)
        self.rD = Vec3f32(rDx, rDy, rDz)
        self.mask = 0xFFFFFFFF
        self.hit = Intersection()
        self.hit.t = t_max


@fieldwise_init
struct BVHNode(Copyable):
    var aabb: AABB
    var leftFirst: UInt32
    var triCount: UInt32

    @always_inline
    def __init__(out self):
        self.aabb = AABB.invalid()
        self.leftFirst = 0
        self.triCount = 0

    @always_inline
    def is_leaf(self) -> Bool:
        return self.triCount > 0

    @always_inline
    def surface_area(self) -> Float32:
        return self.aabb.surface_area()


@fieldwise_init
struct Bin(Copyable):
    var bounds: AABB
    var tri_count: UInt32

    def __init__(out self):
        self.bounds = AABB.invalid()
        self.tri_count = 0


@fieldwise_init
struct Fragment(Copyable):
    """Cached primitive bounds used by the builder.

    prim_indices stores indices into this Fragment array. `prim_idx` stores the
    original triangle id, so traversal still reports the mesh primitive id.
    """

    var bmin: Vec3f32
    var prim_idx: UInt32
    var bmax: Vec3f32
    var _pad: UInt32

    @always_inline
    def __init__(out self):
        self.bmin = Vec3f32(f32_max)
        self.prim_idx = 0
        self.bmax = Vec3f32(f32_min)
        self._pad = 0

    @always_inline
    def __init__(
        out self, prim_idx: UInt32, v0: Vec3f32, v1: Vec3f32, v2: Vec3f32
    ):
        self.bmin = vmin(v0, v1, v2)
        self.prim_idx = prim_idx
        self.bmax = vmax(v0, v1, v2)
        self._pad = 0

    @always_inline
    def center_axis(self, axis: Int) -> Float32:
        return (self.bmin[axis] + self.bmax[axis]) * 0.5

    @always_inline
    def grow_into(self, mut aabb: AABB):
        aabb._min = vmin(aabb._min, self.bmin)
        aabb._max = vmax(aabb._max, self.bmax)


@fieldwise_init
struct SplitResult(Copyable):
    """Result of the binned SAH sweep.

    `bin`, `bin_min`, and `bin_scale` describe the exact binning rule used
    during evaluation. Partitioning by these fields keeps the partition in sync
    with the cost and child bounds computed by `_sah`.
    """

    var axis: Int
    var bin: Int
    var pos: Float32
    var cost: Float32
    var bin_min: Float32
    var bin_scale: Float32
    var left_bounds: AABB
    var right_bounds: AABB

    @always_inline
    def __init__(out self):
        self.axis = -1
        self.bin = -1
        self.pos = 0.0
        self.cost = f32_max
        self.bin_min = 0.0
        self.bin_scale = 0.0
        self.left_bounds = AABB.invalid()
        self.right_bounds = AABB.invalid()

    @always_inline
    def valid(self) -> Bool:
        return self.axis >= 0 and self.bin >= 0


@fieldwise_init
struct MortonPrim(Copyable):
    """Morton-code / fragment-index pair used by the CPU LBVH builder."""

    var code: UInt32
    var frag_idx: UInt32

    @always_inline
    def __init__(out self):
        self.code = 0
        self.frag_idx = 0


struct BVH(Copyable):
    var bvh_nodes: List[BVHNode]
    var prim_indices: List[UInt32]
    var fragments: List[Fragment]
    var vertices: UnsafePointer[Vec3f32, MutAnyOrigin]
    var tri_count: UInt32
    var nodes_used: UInt32

    def __init__(
        out self,
        vertices: UnsafePointer[Vec3f32, MutAnyOrigin],
        tri_count: UInt32,
    ):
        self.vertices = vertices
        self.tri_count = tri_count
        self.nodes_used = 1

        var max_nodes = Int(tri_count * 2 - 1)
        self.bvh_nodes = List[BVHNode](capacity=max_nodes)
        for _ in range(max_nodes):
            self.bvh_nodes.append(BVHNode())

        self.fragments = List[Fragment](capacity=Int(tri_count))
        self.prim_indices = List[UInt32](capacity=Int(tri_count))
        for i in range(Int(tri_count)):
            ref v0 = vertices[i * 3 + 0]
            ref v1 = vertices[i * 3 + 1]
            ref v2 = vertices[i * 3 + 2]
            self.fragments.append(Fragment(UInt32(i), v0, v1, v2))
            self.prim_indices.append(UInt32(i))

        ref root = self.bvh_nodes[0]
        root.leftFirst = 0
        root.triCount = tri_count

        self.update_node_bounds(0)

    @always_inline
    def update_node_bounds(mut self, node_idx: UInt32):
        ref node = self.bvh_nodes[Int(node_idx)]
        node.aabb = AABB.invalid()

        var first = Int(node.leftFirst)
        for i in range(Int(node.triCount)):
            var frag_idx = Int(self.prim_indices[first + i])
            self.fragments[frag_idx].grow_into(node.aabb)

    @always_inline
    def _split_node[
        split_method: String
    ](
        mut self,
        node_idx: UInt32,
        atomic_nodes: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    ) -> Optional[Tuple[UInt32, UInt32]]:
        comptime MAX_LEAF_SIZE = 4  # Set to 4, 8, or 16 depending on your target WideBVH

        var nodes_ptr = self.bvh_nodes.unsafe_ptr()
        ref node = nodes_ptr[Int(node_idx)]

        var axis: Int
        var pos: Float32
        var use_sah_bounds = False
        var split_bin = -1
        var split_bin_min = Float32(0.0)
        var split_bin_scale = Float32(0.0)
        var cached_left_bounds = AABB.invalid()
        var cached_right_bounds = AABB.invalid()

        comptime if split_method == "median":
            var extent = node.aabb._max - node.aabb._min
            axis = longest_axis(extent)
            pos = node.aabb._min[axis] + extent[axis] * 0.5
        elif split_method == "sah":
            var split = _sah(
                node,
                self.prim_indices.unsafe_ptr(),
                self.fragments.unsafe_ptr(),
            )

            # Compare in the same unnormalized units as the raw split cost:
            # split leaf cost = area * N, internal split cost = traversal area + child costs.
            var leaf_cost = node.surface_area() * Float32(node.triCount)
            var split_cost = split.cost + node.surface_area()

            if (not split.valid()) or split_cost >= leaf_cost:
                if node.triCount > MAX_LEAF_SIZE:
                    # Force a spatial median split to keep breaking the node down.
                    var extent = node.aabb._max - node.aabb._min
                    axis = longest_axis(extent)
                    pos = node.aabb._min[axis] + extent[axis] * 0.5
                else:
                    return None  # Safely make a leaf.
            else:
                axis = split.axis
                pos = split.pos
                split_bin = split.bin
                split_bin_min = split.bin_min
                split_bin_scale = split.bin_scale
                cached_left_bounds = split.left_bounds.copy()
                cached_right_bounds = split.right_bounds.copy()
                use_sah_bounds = True

        else:
            comptime assert False, "Unknown split method"

        # Termination Criteria

        # 2. Stop splitting if we are small enough
        if node.triCount <= MAX_LEAF_SIZE:
            return None

        var split_idx: Int
        comptime if split_method == "sah":
            if use_sah_bounds:
                split_idx = _partition_fragments_by_bin(
                    self.prim_indices.unsafe_ptr(),
                    self.fragments.unsafe_ptr(),
                    Int(node.leftFirst),
                    Int(node.triCount),
                    axis,
                    split_bin,
                    split_bin_min,
                    split_bin_scale,
                )
            else:
                split_idx = _partition_fragments(
                    self.prim_indices.unsafe_ptr(),
                    self.fragments.unsafe_ptr(),
                    Int(node.leftFirst),
                    Int(node.triCount),
                    axis,
                    pos,
                )
        else:
            split_idx = _partition_fragments(
                self.prim_indices.unsafe_ptr(),
                self.fragments.unsafe_ptr(),
                Int(node.leftFirst),
                Int(node.triCount),
                axis,
                pos,
            )

        var left_count = UInt32(split_idx - Int(node.leftFirst))
        if left_count == 0 or left_count == node.triCount:
            use_sah_bounds = False
            comptime if split_method == "median":
                # Fallback: If spatial center failed, just split the indices 50/50.
                left_count = node.triCount // 2
                split_idx = Int(node.leftFirst) + Int(left_count)
            else:
                if node.triCount > MAX_LEAF_SIZE:
                    left_count = node.triCount // 2
                    split_idx = Int(node.leftFirst) + Int(left_count)
                else:
                    # Only now is it safe to give up and make a leaf.
                    return None

        # Atomic allocation (Safe for both ST and MT)
        var left_child_idx = Atomic.fetch_add(atomic_nodes, 2)

        nodes_ptr[Int(left_child_idx)].leftFirst = node.leftFirst
        nodes_ptr[Int(left_child_idx)].triCount = left_count
        nodes_ptr[Int(left_child_idx + 1)].leftFirst = UInt32(split_idx)
        nodes_ptr[Int(left_child_idx + 1)].triCount = node.triCount - left_count

        node.leftFirst = left_child_idx
        node.triCount = 0  # Internal node

        if use_sah_bounds:
            nodes_ptr[Int(left_child_idx)].aabb = cached_left_bounds.copy()
            nodes_ptr[Int(left_child_idx + 1)].aabb = cached_right_bounds.copy()
        else:
            self.update_node_bounds(left_child_idx)
            self.update_node_bounds(left_child_idx + 1)

        return (left_child_idx, left_child_idx + 1)

    @always_inline
    def _build_iterative[
        split_method: String
    ](
        mut self,
        root_idx: UInt32,
        atomic_nodes: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    ):
        var stack = [root_idx]
        while len(stack) > 0:
            var children = self._split_node[split_method](
                stack.pop(), atomic_nodes
            )
            if children:
                var res = children.value()
                stack.append(res[1])  # Push Right
                stack.append(res[0])  # Push Left

    def _build_lbvh_recursive(
        mut self,
        pairs: UnsafePointer[MortonPrim, ImmutAnyOrigin],
        node_idx: UInt32,
        first: Int,
        count: Int,
    ):
        comptime MAX_LEAF_SIZE = 4

        ref node = self.bvh_nodes[Int(node_idx)]
        node.leftFirst = UInt32(first)
        node.triCount = UInt32(count)
        self.update_node_bounds(node_idx)

        if count <= MAX_LEAF_SIZE:
            return

        var split = _find_lbvh_split(pairs, first, first + count)
        var left_count = split - first
        if left_count <= 0 or left_count >= count:
            split = first + count // 2
            left_count = split - first

        var left_child_idx = self.nodes_used
        self.nodes_used += 2

        self.bvh_nodes[Int(left_child_idx)].leftFirst = UInt32(first)
        self.bvh_nodes[Int(left_child_idx)].triCount = UInt32(left_count)
        self.bvh_nodes[Int(left_child_idx + 1)].leftFirst = UInt32(split)
        self.bvh_nodes[Int(left_child_idx + 1)].triCount = UInt32(
            count - left_count
        )

        node.leftFirst = left_child_idx
        node.triCount = 0

        self._build_lbvh_recursive(pairs, left_child_idx, first, left_count)
        self._build_lbvh_recursive(
            pairs, left_child_idx + 1, split, count - left_count
        )

    def build_lbvh(mut self):
        """Build a binary LBVH using sorted Morton codes over cached fragments.

        This is a CPU reference builder. It produces the same BVHNode /
        prim_indices layout as the median and SAH builders, so scalar traversal,
        WideBVH, and BVHGPU conversion can all consume it unchanged.
        """
        self.nodes_used = 1

        if self.tri_count == 0:
            return

        var centroid_min = Vec3f32(f32_max, f32_max, f32_max)
        var centroid_max = Vec3f32(f32_min, f32_min, f32_min)

        for i in range(Int(self.tri_count)):
            ref frag = self.fragments[i]
            var c = Vec3f32(
                frag.center_axis(0),
                frag.center_axis(1),
                frag.center_axis(2),
            )
            centroid_min = vmin(centroid_min, c)
            centroid_max = vmax(centroid_max, c)

        var extent = centroid_max - centroid_min
        var inv_x = Float32(0.0)
        var inv_y = Float32(0.0)
        var inv_z = Float32(0.0)
        if extent.x() > 1.0e-20:
            inv_x = 1.0 / extent.x()
        if extent.y() > 1.0e-20:
            inv_y = 1.0 / extent.y()
        if extent.z() > 1.0e-20:
            inv_z = 1.0 / extent.z()

        var pairs = List[MortonPrim](capacity=Int(self.tri_count))
        for i in range(Int(self.tri_count)):
            ref frag = self.fragments[i]
            var x = (frag.center_axis(0) - centroid_min.x()) * inv_x
            var y = (frag.center_axis(1) - centroid_min.y()) * inv_y
            var z = (frag.center_axis(2) - centroid_min.z()) * inv_z
            var code = _morton3_scalar(x, y, z)
            pairs.append(MortonPrim(code, UInt32(i)))

        span = Span(ptr=pairs.unsafe_ptr(), length=len(pairs))

        sort[_morton_pair_less](span)

        for i in range(len(pairs)):
            self.prim_indices[i] = pairs[i].frag_idx

        ref root = self.bvh_nodes[0]
        root.leftFirst = 0
        root.triCount = self.tri_count
        self.update_node_bounds(0)
        self._build_lbvh_recursive(
            pairs.unsafe_ptr(), 0, 0, Int(self.tri_count)
        )

    def build[split_method: String, is_mt: Bool](mut self):
        comptime if split_method == "lbvh":
            self.build_lbvh()
        else:
            self.nodes_used = 1
            self.update_node_bounds(0)

            # All build modes use an atomic counter for unified API
            var atomic_nodes = alloc[Scalar[DType.uint32]](1)
            atomic_nodes[0] = self.nodes_used

            comptime if not is_mt:
                # Single-threaded: Just run the loop once from the root
                self._build_iterative[split_method](0, atomic_nodes)

            else:
                # Multi-threaded: breadth-first seed to generate independent tasks.
                var tasks: List[UInt32] = [0]
                while len(tasks) < 64:
                    var largest_idx = -1
                    var max_tris = UInt32(0)

                    for i in range(len(tasks)):
                        var count = self.bvh_nodes[Int(tasks[i])].triCount
                        if count > max_tris:
                            max_tris = count
                            largest_idx = i

                    if largest_idx == -1:
                        break

                    var target = tasks.pop(largest_idx)
                    var children = self._split_node[split_method](
                        target, atomic_nodes
                    )
                    if children:
                        var c = children.value()
                        tasks.append(c[0])
                        tasks.append(c[1])

                    if len(tasks) == 0:
                        break

                # Parallelize the sub-trees
                @parameter
                def worker(i: Int):
                    self._build_iterative[split_method](tasks[i], atomic_nodes)

                parallelize[worker](len(tasks))

            self.nodes_used = atomic_nodes[0]
            atomic_nodes.free()

    def sah_cost(self, node_idx: UInt32 = 0) -> Float32:
        """Recursively calculates the unnormalized SAH cost of the subtree."""
        ref node = self.bvh_nodes[Int(node_idx)]
        var area = node.surface_area()

        # Standard SAH constants
        comptime C_traverse = 1.0
        comptime C_intersect = 1.0

        cost_aabb = area * C_traverse
        if node.is_leaf():
            # Cost of evaluating the AABB + cost of evaluating the primitives
            return cost_aabb + area * Float32(node.triCount) * C_intersect
        else:
            # Cost of evaluating this AABB + expected cost of traversing children
            var left_cost = self.sah_cost(node.leftFirst)
            var right_cost = self.sah_cost(node.leftFirst + 1)
            return cost_aabb + left_cost + right_cost

    def tree_quality(self) -> Float32:
        """
        Returns the normalized SAH cost of the BVH.
        This number mathematically represents the 'Expected number of ray operations'
        (AABB checks + Triangle checks) a random ray will perform. Lower is better.
        """
        var root_area = self.bvh_nodes[0].surface_area()
        var total_cost = self.sah_cost(0)
        return total_cost / root_area

    @always_inline
    def _intersect_tri[
        is_shadow: Bool
    ](self, mut ray: Ray, prim_idx: UInt32) -> Bool:
        # Local scalar Möller-Trumbore. This keeps BVH nearest-hit semantics
        # independent from the lower-level helper's exact sign / culling policy,
        # and matches the SIMD leaf path below.
        ref v0 = self.vertices[Int(prim_idx) * 3]
        ref v1 = self.vertices[Int(prim_idx) * 3 + 1]
        ref v2 = self.vertices[Int(prim_idx) * 3 + 2]

        var e1x = v1.x() - v0.x()
        var e1y = v1.y() - v0.y()
        var e1z = v1.z() - v0.z()
        var e2x = v2.x() - v0.x()
        var e2y = v2.y() - v0.y()
        var e2z = v2.z() - v0.z()

        var px = ray.D.y() * e2z - ray.D.z() * e2y
        var py = ray.D.z() * e2x - ray.D.x() * e2z
        var pz = ray.D.x() * e2y - ray.D.y() * e2x
        var det = e1x * px + e1y * py + e1z * pz

        if det > -1e-12 and det < 1e-12:
            return False

        var inv_det = 1.0 / det
        var tx = ray.O.x() - v0.x()
        var ty = ray.O.y() - v0.y()
        var tz = ray.O.z() - v0.z()

        var u = (tx * px + ty * py + tz * pz) * inv_det
        if u < 0.0 or u > 1.0:
            return False

        var qx = ty * e1z - tz * e1y
        var qy = tz * e1x - tx * e1z
        var qz = tx * e1y - ty * e1x

        var v = (ray.D.x() * qx + ray.D.y() * qy + ray.D.z() * qz) * inv_det
        if v < 0.0 or u + v > 1.0:
            return False

        var t = (e2x * qx + e2y * qy + e2z * qz) * inv_det
        if t > 1e-4 and t < ray.hit.t:
            comptime if is_shadow:
                return True
            ray.hit.t = t
            ray.hit.u = u
            ray.hit.v = v
            ray.hit.prim = prim_idx
            return True

        return False

    @always_inline
    def _traverse[is_shadow: Bool](self, mut ray: Ray) -> Bool:
        var stack = InlineArray[UInt32, 64](fill=0)
        var stack_ptr = 0
        var node_idx = UInt32(0)

        while True:
            ref node = self.bvh_nodes[Int(node_idx)]

            if node.is_leaf():
                for i in range(Int(node.triCount)):
                    var frag_idx = self.prim_indices[Int(node.leftFirst) + i]
                    var p_idx = self.fragments[Int(frag_idx)].prim_idx
                    if self._intersect_tri[is_shadow](ray, p_idx):
                        comptime if is_shadow:
                            return True

                if stack_ptr == 0:
                    break
                stack_ptr -= 1
                node_idx = stack[stack_ptr]
                continue

            var child1_idx = node.leftFirst
            var child2_idx = node.leftFirst + 1
            var dist1, dist2 = Float32(f32_max), Float32(f32_max)

            # Intersection checks
            var h1 = intersect_ray_aabb(
                ray.O,
                ray.rD,
                self.bvh_nodes[Int(child1_idx)].aabb._min,
                self.bvh_nodes[Int(child1_idx)].aabb._max,
                dist1,
            )
            var h2 = intersect_ray_aabb(
                ray.O,
                ray.rD,
                self.bvh_nodes[Int(child2_idx)].aabb._min,
                self.bvh_nodes[Int(child2_idx)].aabb._max,
                dist2,
            )

            # Cull by current ray distance
            if h1 and dist1 >= ray.hit.t:
                h1 = False
            if h2 and dist2 >= ray.hit.t:
                h2 = False

            if not h1 and not h2:
                if stack_ptr == 0:
                    break
                stack_ptr -= 1
                node_idx = stack[stack_ptr]
            elif h1 and not h2:
                node_idx = child1_idx
            elif not h1 and h2:
                node_idx = child2_idx
            else:
                # BOTH HIT: Handle ordering
                var near = child1_idx
                var far = child2_idx

                # Nested if to separate comptime from runtime
                comptime if not is_shadow:
                    if dist1 > dist2:
                        near = child2_idx
                        far = child1_idx

                stack[stack_ptr] = far
                stack_ptr += 1
                node_idx = near

        return False

    def traverse(self, mut ray: Ray):
        _ = self._traverse[False](ray)

    def is_occluded(self, mut ray: Ray) -> Bool:
        return self._traverse[True](ray)


@always_inline
def _partition_fragments(
    prims: UnsafePointer[UInt32, MutAnyOrigin],
    fragments: UnsafePointer[Fragment, ImmutAnyOrigin],
    first: Int,
    count: Int,
    axis: Int,
    pos: Float32,
) -> Int:
    var i = first
    var j = first + count - 1

    while i <= j:
        var frag_idx = Int(prims[i])
        var c = fragments[frag_idx].center_axis(axis)

        if c < pos:
            i += 1
        else:
            prims[i], prims[j] = prims[j], prims[i]
            j -= 1

    return i


@always_inline
def _fragment_bin(
    fragments: UnsafePointer[Fragment, ImmutAnyOrigin],
    frag_idx: Int,
    axis: Int,
    bin_min: Float32,
    bin_scale: Float32,
) -> Int:
    var c = fragments[frag_idx].center_axis(axis)
    var b_idx = Int((c - bin_min) * bin_scale)
    if b_idx < 0:
        return 0
    if b_idx >= BVH_BINS:
        return BVH_BINS - 1
    return b_idx


@always_inline
def _partition_fragments_by_bin(
    prims: UnsafePointer[UInt32, MutAnyOrigin],
    fragments: UnsafePointer[Fragment, ImmutAnyOrigin],
    first: Int,
    count: Int,
    axis: Int,
    split_bin: Int,
    bin_min: Float32,
    bin_scale: Float32,
) -> Int:
    var i = first
    var j = first + count - 1

    while i <= j:
        var frag_idx = Int(prims[i])
        var b_idx = _fragment_bin(fragments, frag_idx, axis, bin_min, bin_scale)

        if b_idx <= split_bin:
            i += 1
        else:
            prims[i], prims[j] = prims[j], prims[i]
            j -= 1

    return i


@always_inline
def _sah(
    node: BVHNode,
    prims: UnsafePointer[UInt32, ImmutAnyOrigin],
    fragments: UnsafePointer[Fragment, ImmutAnyOrigin],
) -> SplitResult:
    var best = SplitResult()

    for axis in range(3):
        var min_c = f32_max
        var max_c = f32_min

        # 1. Find centroid range for this node/axis.
        for i in range(Int(node.triCount)):
            var frag_idx = Int(prims[Int(node.leftFirst) + i])
            var c = fragments[frag_idx].center_axis(axis)
            min_c = min(min_c, c)
            max_c = max(max_c, c)

        if min_c == max_c:
            continue

        # 2. Bin cached primitive bounds.
        var bins = InlineArray[Bin, BVH_BINS](fill=Bin())
        var scale = Float32(BVH_BINS) / (max_c - min_c)

        for i in range(Int(node.triCount)):
            var frag_idx = Int(prims[Int(node.leftFirst) + i])
            ref frag = fragments[frag_idx]

            var b_idx = _fragment_bin(fragments, frag_idx, axis, min_c, scale)
            bins[b_idx].tri_count += 1
            frag.grow_into(bins[b_idx].bounds)

        # 3. Left sweep: store both area/count and exact bounds at each split.
        var left_areas = InlineArray[Float32, BVH_BINS](fill=0.0)
        var left_counts = InlineArray[UInt32, BVH_BINS](fill=0)
        var left_bounds = InlineArray[AABB, BVH_BINS](fill=AABB.invalid())

        var left_box = AABB.invalid()
        var left_sum = UInt32(0)

        for i in range(BVH_BINS - 1):
            left_sum += bins[i].tri_count
            left_counts[i] = left_sum
            left_box.grow(bins[i].bounds)
            left_bounds[i] = left_box.copy()
            left_areas[i] = left_box.surface_area()

        # 4. Right sweep + split cost.
        var right_box = AABB.invalid()
        var right_sum = UInt32(0)

        for i in range(BVH_BINS - 1, 0, -1):
            right_sum += bins[i].tri_count
            right_box.grow(bins[i].bounds)

            var left_count = left_counts[i - 1]
            var right_count = right_sum

            if left_count == 0 or right_count == 0:
                continue

            var left_cost = left_areas[i - 1] * Float32(left_count)
            var right_cost = right_box.surface_area() * Float32(right_count)
            var cost = left_cost + right_cost

            if cost < best.cost:
                best.axis = axis
                best.bin = i - 1
                best.pos = min_c + Float32(i) / scale
                best.cost = cost
                best.bin_min = min_c
                best.bin_scale = scale
                best.left_bounds = left_bounds[i - 1].copy()
                best.right_bounds = right_box.copy()

    return best^


@always_inline
def _morton3_scalar(x: Float32, y: Float32, z: Float32) -> UInt32:
    var vx = SIMD[DType.float32, 1](clamp(x, 0.0, 1.0))
    var vy = SIMD[DType.float32, 1](clamp(y, 0.0, 1.0))
    var vz = SIMD[DType.float32, 1](clamp(z, 0.0, 1.0))
    var code = morton3[1](vx, vy, vz)
    return code[0]


@always_inline
def _morton_pair_less(a: MortonPrim, b: MortonPrim) capturing -> Bool:
    if a.code < b.code:
        return True
    if a.code > b.code:
        return False
    return a.frag_idx < b.frag_idx


@always_inline
def _highest_set_bit(v: UInt32) -> Int:
    if v == 0:
        return -1
    return 31 - Int(count_leading_zeros(v))


@always_inline
def _find_lbvh_split(
    pairs: UnsafePointer[MortonPrim, ImmutAnyOrigin],
    first: Int,
    last: Int,
) -> Int:
    var first_code = pairs[first].code
    var last_code = pairs[last - 1].code

    if first_code == last_code:
        return (first + last) // 2

    var bit = _highest_set_bit(first_code ^ last_code)
    if bit < 0:
        return (first + last) // 2

    var mask = UInt32(1) << UInt32(bit)
    var left_bit = first_code & mask

    for i in range(first + 1, last):
        if (pairs[i].code & mask) != left_bit:
            return i

    return (first + last) // 2


@fieldwise_init
struct WideLeaf[width: Int](Copyable):
    var v0x: SIMD[DType.float32, Self.width]
    var v0y: SIMD[DType.float32, Self.width]
    var v0z: SIMD[DType.float32, Self.width]
    var v1x: SIMD[DType.float32, Self.width]
    var v1y: SIMD[DType.float32, Self.width]
    var v1z: SIMD[DType.float32, Self.width]
    var v2x: SIMD[DType.float32, Self.width]
    var v2y: SIMD[DType.float32, Self.width]
    var v2z: SIMD[DType.float32, Self.width]
    var prim_indices: SIMD[DType.uint32, Self.width]

    def __init__(out self):
        self.v0x = SIMD[DType.float32, Self.width](0)
        self.v0y = SIMD[DType.float32, Self.width](0)
        self.v0z = SIMD[DType.float32, Self.width](0)
        self.v1x = SIMD[DType.float32, Self.width](0)
        self.v1y = SIMD[DType.float32, Self.width](0)
        self.v1z = SIMD[DType.float32, Self.width](0)
        self.v2x = SIMD[DType.float32, Self.width](0)
        self.v2y = SIMD[DType.float32, Self.width](0)
        self.v2z = SIMD[DType.float32, Self.width](0)
        self.prim_indices = SIMD[DType.uint32, Self.width](0xFFFFFFFF)


@fieldwise_init
struct WideBVHNode[width: Int](Copyable):
    var min_x: SIMD[DType.float32, Self.width]
    var min_y: SIMD[DType.float32, Self.width]
    var min_z: SIMD[DType.float32, Self.width]

    var max_x: SIMD[DType.float32, Self.width]
    var max_y: SIMD[DType.float32, Self.width]
    var max_z: SIMD[DType.float32, Self.width]

    var data: SIMD[DType.uint32, Self.width]
    var counts: SIMD[DType.uint32, Self.width]

    @always_inline
    def __init__(out self):
        self.min_x = SIMD[DType.float32, Self.width](f32_max)
        self.min_y = SIMD[DType.float32, Self.width](f32_max)
        self.min_z = SIMD[DType.float32, Self.width](f32_max)

        self.max_x = SIMD[DType.float32, Self.width](f32_min)
        self.max_y = SIMD[DType.float32, Self.width](f32_min)
        self.max_z = SIMD[DType.float32, Self.width](f32_min)

        self.data = SIMD[DType.uint32, Self.width](0)
        self.counts = SIMD[DType.uint32, Self.width](0xFFFFFFFF)


struct WideBVH[width: Int](Copyable):
    var nodes: List[WideBVHNode[Self.width]]
    var leaves: List[WideLeaf[Self.width]]
    var vertices: UnsafePointer[Vec3f32, MutAnyOrigin]

    def __init__(out self, mut binary_bvh: BVH):
        self.nodes = List[WideBVHNode[Self.width]]()
        self.leaves = List[WideLeaf[Self.width]]()
        self.vertices = binary_bvh.vertices
        _ = self._collapse(binary_bvh, 0)

    def _collapse(mut self, binary_bvh: BVH, bin_idx: UInt32) -> UInt32:
        var wide_idx = UInt32(len(self.nodes))
        self.nodes.append(WideBVHNode[Self.width]())

        # Pull up children to fill SIMD width
        var pool = InlineArray[UInt32, Self.width](fill=bin_idx)
        var p_size = 1
        while p_size < Self.width:
            var best_a: Float32 = -1.0
            var best_i: Int = -1
            for i in range(p_size):
                if not binary_bvh.bvh_nodes[Int(pool[i])].is_leaf():
                    var a = binary_bvh.bvh_nodes[Int(pool[i])].surface_area()
                    if a > best_a:
                        best_a = a
                        best_i = i
            if best_i == -1:
                break
            ref n = binary_bvh.bvh_nodes[Int(pool[best_i])]
            pool[best_i] = n.leftFirst
            pool[p_size] = n.leftFirst + 1
            p_size += 1

        var node = WideBVHNode[Self.width]()
        comptime for i in range(Self.width):
            if i < p_size:
                ref n = binary_bvh.bvh_nodes[Int(pool[i])]
                node.min_x[i] = n.aabb._min.x()
                node.max_x[i] = n.aabb._max.x()
                node.min_y[i] = n.aabb._min.y()
                node.max_y[i] = n.aabb._max.y()
                node.min_z[i] = n.aabb._min.z()
                node.max_z[i] = n.aabb._max.z()
                if n.is_leaf():
                    # PACK TRIANGLES INTO SIMD LEAF
                    var l_idx = UInt32(len(self.leaves))
                    var packed = WideLeaf[Self.width]()
                    for tri in range(min(Int(n.triCount), Self.width)):
                        var frag_idx = Int(
                            binary_bvh.prim_indices[Int(n.leftFirst) + tri]
                        )
                        var p_idx = Int(binary_bvh.fragments[frag_idx].prim_idx)
                        ref v0 = self.vertices[p_idx * 3]
                        ref v1 = self.vertices[p_idx * 3 + 1]
                        ref v2 = self.vertices[p_idx * 3 + 2]
                        packed.v0x[tri] = v0.x()
                        packed.v0y[tri] = v0.y()
                        packed.v0z[tri] = v0.z()
                        packed.v1x[tri] = v1.x()
                        packed.v1y[tri] = v1.y()
                        packed.v1z[tri] = v1.z()
                        packed.v2x[tri] = v2.x()
                        packed.v2y[tri] = v2.y()
                        packed.v2z[tri] = v2.z()
                        packed.prim_indices[tri] = UInt32(p_idx)
                    # Sentinel for empty lanes. Keep a separate valid-lane mask,
                    # because only poisoning v0x can still create NaNs in SIMD math.
                    for tri in range(Int(n.triCount), Self.width):
                        packed.prim_indices[tri] = 0xFFFFFFFF
                    self.leaves.append(packed^)
                    node.data[i] = l_idx
                    node.counts[i] = 1  # SIMD Leaf
                else:
                    node.data[i] = self._collapse(binary_bvh, pool[i])
                    node.counts[i] = 0
            else:
                node.counts[i] = 0xFFFFFFFF

        self.nodes[Int(wide_idx)] = node^
        return wide_idx

    @always_inline
    def _intersect_leaf[
        is_occlusion: Bool
    ](self, mut ray: Ray, leaf_idx: UInt32) -> Bool:
        ref leaf = self.leaves[Int(leaf_idx)]
        # Möller-Trumbore SIMD math
        var e1x = leaf.v1x - leaf.v0x
        var e1y = leaf.v1y - leaf.v0y
        var e1z = leaf.v1z - leaf.v0z
        var e2x = leaf.v2x - leaf.v0x
        var e2y = leaf.v2y - leaf.v0y
        var e2z = leaf.v2z - leaf.v0z
        var px = ray.D.y() * e2z - ray.D.z() * e2y
        var py = ray.D.z() * e2x - ray.D.x() * e2z
        var pz = ray.D.x() * e2y - ray.D.y() * e2x
        var det = e1x * px + e1y * py + e1z * pz
        var inv_det = 1.0 / det
        var tx = ray.O.x() - leaf.v0x
        var ty = ray.O.y() - leaf.v0y
        var tz = ray.O.z() - leaf.v0z
        var u = (tx * px + ty * py + tz * pz) * inv_det
        var qx = ty * e1z - tz * e1y
        var qy = tz * e1x - tx * e1z
        var qz = tx * e1y - ty * e1x
        var v = (ray.D.x() * qx + ray.D.y() * qy + ray.D.z() * qz) * inv_det
        var t = (e2x * qx + e2y * qy + e2z * qz) * inv_det

        var det_ok = det.gt(1e-12) | det.lt(-1e-12)
        var valid_lane = ~leaf.prim_indices.eq(0xFFFFFFFF)
        var hit_mask = (
            valid_lane
            & det_ok
            & (t.gt(1e-4))
            & (t.lt(ray.hit.t))
            & (u.ge(0.0))
            & (v.ge(0.0))
            & ((u + v).le(1.0))
        )
        if hit_mask.reduce_or():
            comptime if is_occlusion:
                return True
            else:
                var min_t = hit_mask.select(t, f32_max).reduce_min()
                comptime for i in range(Self.width):
                    if hit_mask[i] and t[i] == min_t:
                        ray.hit.t = min_t
                        ray.hit.u = u[i]
                        ray.hit.v = v[i]
                        ray.hit.prim = leaf.prim_indices[i]
                return True
        return False

    @always_inline
    def _traverse_generic[is_occlusion: Bool](self, mut ray: Ray) -> Bool:
        var stack = InlineArray[UInt32, 64](fill=0)
        var s_ptr = 0
        var n_idx = UInt32(0)

        while True:
            ref node = self.nodes[Int(n_idx)]
            # SIMD AABB Check
            var tmin = max(
                max(
                    min(
                        (node.min_x - ray.O.x()) * ray.rD.x(),
                        (node.max_x - ray.O.x()) * ray.rD.x(),
                    ),
                    min(
                        (node.min_y - ray.O.y()) * ray.rD.y(),
                        (node.max_y - ray.O.y()) * ray.rD.y(),
                    ),
                ),
                max(
                    min(
                        (node.min_z - ray.O.z()) * ray.rD.z(),
                        (node.max_z - ray.O.z()) * ray.rD.z(),
                    ),
                    0.0,
                ),
            )
            var tmax = min(
                min(
                    max(
                        (node.min_x - ray.O.x()) * ray.rD.x(),
                        (node.max_x - ray.O.x()) * ray.rD.x(),
                    ),
                    max(
                        (node.min_y - ray.O.y()) * ray.rD.y(),
                        (node.max_y - ray.O.y()) * ray.rD.y(),
                    ),
                ),
                min(
                    max(
                        (node.min_z - ray.O.z()) * ray.rD.z(),
                        (node.max_z - ray.O.z()) * ray.rD.z(),
                    ),
                    ray.hit.t,
                ),
            )

            # Use <= here: triangle AABBs are often flat on one axis, so
            # a ray can touch them with tmin == tmax. The scalar AABB helper
            # accepts these hits; the wide path must do the same.
            var mask = tmin.le(tmax) & (~node.counts.eq(0xFFFFFFFF))

            if mask.reduce_or():
                for i in range(Self.width):
                    if mask[i]:
                        if node.counts[i] == 0:
                            stack[s_ptr] = node.data[i]
                            s_ptr += 1
                        else:
                            if self._intersect_leaf[is_occlusion](
                                ray, node.data[i]
                            ):
                                comptime if is_occlusion:
                                    return True

            if s_ptr == 0:
                break
            s_ptr -= 1
            n_idx = stack[s_ptr]
        return False

    def traverse(self, mut ray: Ray):
        _ = self._traverse_generic[False](ray)

    def is_occluded(self, mut ray: Ray) -> Bool:
        return self._traverse_generic[True](ray)


struct BVHGPUNode(Copyable):
    """TinyBVH-style Aila-Laine GPU node.

    This mirrors TinyBVH's BVH_GPU node semantics:
    - If triCount > 0, this node is a leaf and firstTri points into prim_indices.
    - If triCount == 0, this node is internal:
        lmin/lmax bound the left child, and left is the left child node index.
        rmin/rmax bound the right child, and right is the right child node index.
    """

    var lmin: Vec3f32
    var left: UInt32
    var lmax: Vec3f32
    var right: UInt32

    var rmin: Vec3f32
    var triCount: UInt32
    var rmax: Vec3f32
    var firstTri: UInt32

    @always_inline
    def __init__(out self):
        self.lmin = Vec3f32(f32_max)
        self.left = 0
        self.lmax = Vec3f32(f32_min)
        self.right = 0

        self.rmin = Vec3f32(f32_max)
        self.triCount = 0
        self.rmax = Vec3f32(f32_min)
        self.firstTri = 0

    @always_inline
    def is_leaf(self) -> Bool:
        return self.triCount > 0


struct BVHGPU(Copyable):
    """TinyBVH-compatible GPU layout with CPU reference traversal.

    This layout follows TinyBVH's BVH_GPU / Aila-Laine layout more closely than
    the earlier child-count encoding. Leaves are real nodes (`triCount > 0`),
    while internal nodes store both child bounds directly plus child node indices.

    `prim_indices` stores original triangle ids, not fragment ids. This keeps
    traversal independent from the builder's fragment reordering and makes the
    buffers directly suitable for a future GPU kernel.
    """

    var nodes: List[BVHGPUNode]
    var prim_indices: List[UInt32]
    var vertices: UnsafePointer[Vec3f32, MutAnyOrigin]

    def __init__(out self, mut binary_bvh: BVH):
        self.nodes = List[BVHGPUNode](capacity=Int(binary_bvh.nodes_used))
        self.prim_indices = List[UInt32](capacity=len(binary_bvh.prim_indices))
        self.vertices = binary_bvh.vertices

        # Convert fragment indices to original primitive ids once. GPU leaves
        # point into this array using the source BVH's leftFirst/count ranges.
        for i in range(len(binary_bvh.prim_indices)):
            var frag_idx = Int(binary_bvh.prim_indices[i])
            self.prim_indices.append(binary_bvh.fragments[frag_idx].prim_idx)

        if binary_bvh.nodes_used > 0:
            _ = self._convert_node(binary_bvh, UInt32(0))

    def _convert_node(mut self, binary_bvh: BVH, binary_idx: UInt32) -> UInt32:
        var gpu_idx = UInt32(len(self.nodes))
        self.nodes.append(BVHGPUNode())

        ref bnode = binary_bvh.bvh_nodes[Int(binary_idx)]
        var out = BVHGPUNode()

        if bnode.is_leaf():
            out.triCount = bnode.triCount
            out.firstTri = bnode.leftFirst
            # Leaf bounds are not needed by traversal once the leaf is entered,
            # but storing them makes debug inspection easier and keeps the node
            # self-describing.
            out.lmin = bnode.aabb._min.copy()
            out.lmax = bnode.aabb._max.copy()
            out.rmin = bnode.aabb._min.copy()
            out.rmax = bnode.aabb._max.copy()
        else:
            var left_binary_idx = bnode.leftFirst
            var right_binary_idx = bnode.leftFirst + 1

            ref left = binary_bvh.bvh_nodes[Int(left_binary_idx)]
            ref right = binary_bvh.bvh_nodes[Int(right_binary_idx)]

            out.lmin = left.aabb._min.copy()
            out.lmax = left.aabb._max.copy()
            out.rmin = right.aabb._min.copy()
            out.rmax = right.aabb._max.copy()
            out.triCount = 0
            out.firstTri = 0

            out.left = self._convert_node(binary_bvh, left_binary_idx)
            out.right = self._convert_node(binary_bvh, right_binary_idx)

        self.nodes[Int(gpu_idx)] = out^
        return gpu_idx

    @always_inline
    def _intersect_tri[
        is_shadow: Bool
    ](self, mut ray: Ray, prim_idx: UInt32) -> Bool:
        ref v0 = self.vertices[Int(prim_idx) * 3]
        ref v1 = self.vertices[Int(prim_idx) * 3 + 1]
        ref v2 = self.vertices[Int(prim_idx) * 3 + 2]

        var e1x = v1.x() - v0.x()
        var e1y = v1.y() - v0.y()
        var e1z = v1.z() - v0.z()
        var e2x = v2.x() - v0.x()
        var e2y = v2.y() - v0.y()
        var e2z = v2.z() - v0.z()

        var px = ray.D.y() * e2z - ray.D.z() * e2y
        var py = ray.D.z() * e2x - ray.D.x() * e2z
        var pz = ray.D.x() * e2y - ray.D.y() * e2x
        var det = e1x * px + e1y * py + e1z * pz

        if det > -1e-12 and det < 1e-12:
            return False

        var inv_det = 1.0 / det
        var tx = ray.O.x() - v0.x()
        var ty = ray.O.y() - v0.y()
        var tz = ray.O.z() - v0.z()

        var u = (tx * px + ty * py + tz * pz) * inv_det
        if u < 0.0 or u > 1.0:
            return False

        var qx = ty * e1z - tz * e1y
        var qy = tz * e1x - tx * e1z
        var qz = tx * e1y - ty * e1x

        var v = (ray.D.x() * qx + ray.D.y() * qy + ray.D.z() * qz) * inv_det
        if v < 0.0 or u + v > 1.0:
            return False

        var t = (e2x * qx + e2y * qy + e2z * qz) * inv_det
        if t > 1e-4 and t < ray.hit.t:
            comptime if is_shadow:
                return True
            ray.hit.t = t
            ray.hit.u = u
            ray.hit.v = v
            ray.hit.prim = prim_idx
            return True

        return False

    @always_inline
    def _intersect_leaf[
        is_shadow: Bool
    ](self, mut ray: Ray, first: UInt32, count: UInt32) -> Bool:
        var any_hit = False
        for i in range(Int(count)):
            var prim_idx = self.prim_indices[Int(first) + i]
            if self._intersect_tri[is_shadow](ray, prim_idx):
                comptime if is_shadow:
                    return True
                any_hit = True
        return any_hit

    @always_inline
    def _traverse[is_shadow: Bool](self, mut ray: Ray) -> Bool:
        if len(self.nodes) == 0:
            return False

        var stack = InlineArray[UInt32, 64](fill=0)
        var stack_ptr = 0
        var node_idx = UInt32(0)

        while True:
            ref node = self.nodes[Int(node_idx)]

            if node.is_leaf():
                if self._intersect_leaf[is_shadow](
                    ray, node.firstTri, node.triCount
                ):
                    comptime if is_shadow:
                        return True

                if stack_ptr == 0:
                    break
                stack_ptr -= 1
                node_idx = stack[stack_ptr]
                continue

            var dist_left = Float32(f32_max)
            var dist_right = Float32(f32_max)

            var hit_left = intersect_ray_aabb(
                ray.O,
                ray.rD,
                node.lmin,
                node.lmax,
                dist_left,
            )
            var hit_right = intersect_ray_aabb(
                ray.O,
                ray.rD,
                node.rmin,
                node.rmax,
                dist_right,
            )

            if hit_left and dist_left >= ray.hit.t:
                hit_left = False
            if hit_right and dist_right >= ray.hit.t:
                hit_right = False

            if not hit_left and not hit_right:
                if stack_ptr == 0:
                    break
                stack_ptr -= 1
                node_idx = stack[stack_ptr]
            elif hit_left and not hit_right:
                node_idx = node.left
            elif not hit_left and hit_right:
                node_idx = node.right
            else:
                var near = node.left
                var far = node.right

                # Closest-hit rays benefit from near-first traversal. Shadow rays
                # do not require ordering for correctness, but the same order is
                # still a reasonable CPU reference for the future GPU kernel.
                if dist_left > dist_right:
                    near = node.right
                    far = node.left

                stack[stack_ptr] = far
                stack_ptr += 1
                node_idx = near

        return False

    def traverse(self, mut ray: Ray):
        _ = self._traverse[False](ray)

    def is_occluded(self, mut ray: Ray) -> Bool:
        return self._traverse[True](ray)


# TLAS & INSTANCING (2-LAYER BVH)


@fieldwise_init
struct Instance(Copyable):
    var transform: Mat44f32
    var inv_transform: Mat44f32
    var bounds_min: Vec3f32
    var bounds_max: Vec3f32
    var blas_idx: UInt32

    @always_inline
    def __init__(
        out self,
        transform: Mat44f32,
        inv_transform: Mat44f32,
        blas_idx: UInt32,
        blas_min: Vec3f32,
        blas_max: Vec3f32,
    ):
        self.transform = transform.copy()
        self.inv_transform = inv_transform.copy()
        self.blas_idx = blas_idx

        # To find the World Space AABB, we must transform all 8 corners of the Local BLAS AABB
        var corners = InlineArray[Vec3f32, 8](fill=Vec3f32(0))
        corners[0] = Vec3f32(blas_min.x(), blas_min.y(), blas_min.z())
        corners[1] = Vec3f32(blas_max.x(), blas_min.y(), blas_min.z())
        corners[2] = Vec3f32(blas_min.x(), blas_max.y(), blas_min.z())
        corners[3] = Vec3f32(blas_max.x(), blas_max.y(), blas_min.z())
        corners[4] = Vec3f32(blas_min.x(), blas_min.y(), blas_max.z())
        corners[5] = Vec3f32(blas_max.x(), blas_min.y(), blas_max.z())
        corners[6] = Vec3f32(blas_min.x(), blas_max.y(), blas_max.z())
        corners[7] = Vec3f32(blas_max.x(), blas_max.y(), blas_max.z())

        var w_min = Vec3f32(f32_max)
        var w_max = Vec3f32(f32_min)
        comptime for i in range(8):
            var transformed = transform_point(transform, corners[i])
            w_min = vmin(w_min, transformed)
            w_max = vmax(w_max, transformed)

        self.bounds_min = w_min^
        self.bounds_max = w_max^


struct TLAS:
    var tlas_nodes: List[BVHNode]
    var inst_indices: List[UInt32]
    var instances: UnsafePointer[Instance, MutAnyOrigin]
    var inst_count: UInt32
    var nodes_used: UInt32

    def __init__(
        out self,
        instances: UnsafePointer[Instance, MutAnyOrigin],
        inst_count: UInt32,
    ):
        self.instances = instances
        self.inst_count = inst_count
        self.nodes_used = 1

        # Capacity is 2N - 1
        var max_nodes = Int(inst_count * 2 - 1) if inst_count > 0 else 1
        self.tlas_nodes = List[BVHNode](capacity=max_nodes)
        for _ in range(max_nodes):
            self.tlas_nodes.append(BVHNode())

        self.inst_indices = List[UInt32](capacity=Int(inst_count))
        for i in range(Int(inst_count)):
            self.inst_indices.append(UInt32(i))

        if inst_count > 0:
            ref root = self.tlas_nodes[0]
            root.leftFirst = 0
            root.triCount = inst_count
            self.update_node_bounds(0)

    @always_inline
    def update_node_bounds(mut self, node_idx: UInt32):
        ref node = self.tlas_nodes[Int(node_idx)]
        node.aabb._min = Vec3f32(f32_max)
        node.aabb._max = Vec3f32(f32_min)

        var first = Int(node.leftFirst)
        for i in range(Int(node.triCount)):
            var inst_idx = Int(self.inst_indices[first + i])
            ref inst = self.instances[inst_idx]

            node.aabb._min = vmin(node.aabb._min, inst.bounds_min)
            node.aabb._max = vmax(node.aabb._max, inst.bounds_max)

    def build(mut self):
        """TLAS uses the Quick/Spatial Median builder since instance count is usually low (<10,000).
        """
        if self.inst_count > 0:
            self.subdivide(0)

    def subdivide(mut self, node_idx: UInt32):
        ref node = self.tlas_nodes[Int(node_idx)]

        if node.triCount <= 2:
            return

        var extent = node.aabb._max - node.aabb._min
        var axis = longest_axis(extent)
        var split_pos = node.aabb._min[axis] + extent[axis] * 0.5

        var i = Int(node.leftFirst)
        var j = i + Int(node.triCount) - 1

        while i <= j:
            var inst_idx = Int(self.inst_indices[i])
            ref inst = self.instances[inst_idx]

            var centroid = (inst.bounds_min[axis] + inst.bounds_max[axis]) * 0.5

            if centroid < split_pos:
                i += 1
            else:
                var tmp = self.inst_indices[i]
                self.inst_indices[i] = self.inst_indices[j]
                self.inst_indices[j] = tmp
                j -= 1

        var left_count = UInt32(i - Int(node.leftFirst))
        if left_count == 0 or left_count == node.triCount:
            left_count = node.triCount // 2
            i = Int(node.leftFirst) + Int(left_count)

        var left_child_idx = self.nodes_used
        self.nodes_used += 2

        ref left_child = self.tlas_nodes[Int(left_child_idx)]
        ref right_child = self.tlas_nodes[Int(left_child_idx + 1)]

        left_child.leftFirst = node.leftFirst
        left_child.triCount = left_count
        right_child.leftFirst = UInt32(i)
        right_child.triCount = node.triCount - left_count

        node.leftFirst = left_child_idx
        node.triCount = 0

        self.update_node_bounds(left_child_idx)
        self.update_node_bounds(left_child_idx + 1)

        self.subdivide(left_child_idx)
        self.subdivide(left_child_idx + 1)

    def traverse(self, mut ray: Ray, blases: UnsafePointer[BVH, MutAnyOrigin]):
        """
        Traverses the Top Level tree. When an instance leaf is hit, the ray is transformed
        into local space and passed to the respective Bottom Level Acceleration Structure.
        """
        var stack = InlineArray[UInt32, 64](fill=0)
        var stack_ptr = 0
        var node_idx = UInt32(0)

        while True:
            ref node = self.tlas_nodes[Int(node_idx)]
            if node.is_leaf():
                for i in range(Int(node.triCount)):
                    var inst_idx = self.inst_indices[Int(node.leftFirst) + i]
                    ref inst = self.instances[Int(inst_idx)]

                    # 1. Transform World Ray -> Local Space
                    var local_O = transform_point(inst.inv_transform, ray.O)
                    var local_D = transform_vector(inst.inv_transform, ray.D)
                    var local_ray = Ray(local_O, local_D, ray.hit.t)

                    # 2. Traverse the BLAS
                    blases[Int(inst.blas_idx)].traverse(local_ray)

                    # 3. If a closer hit is found, save it!
                    # Due to linear matrix math, local_ray.hit.t is identical to World Space t!
                    if local_ray.hit.t < ray.hit.t:
                        ray.hit.t = local_ray.hit.t
                        ray.hit.u = local_ray.hit.u
                        ray.hit.v = local_ray.hit.v
                        ray.hit.prim = local_ray.hit.prim
                        ray.hit.inst = inst_idx  # Tag the instance ID so shaders know what mesh was hit

                if stack_ptr == 0:
                    break
                stack_ptr -= 1
                node_idx = stack[stack_ptr]
                continue

            var child1_idx = node.leftFirst
            var child2_idx = node.leftFirst + 1
            ref child1 = self.tlas_nodes[Int(child1_idx)]
            ref child2 = self.tlas_nodes[Int(child2_idx)]

            var dist1 = Float32(f32_max)
            var dist2 = Float32(f32_max)
            var hit1 = intersect_ray_aabb(
                ray.O, ray.rD, child1.aabb._min, child1.aabb._max, dist1
            )
            var hit2 = intersect_ray_aabb(
                ray.O, ray.rD, child2.aabb._min, child2.aabb._max, dist2
            )

            if hit1 and dist1 >= ray.hit.t:
                hit1 = False
            if hit2 and dist2 >= ray.hit.t:
                hit2 = False

            if hit1 and hit2 and (dist1 > dist2):
                var tmp = child1_idx
                child1_idx = child2_idx
                child2_idx = tmp

            if not hit1 and not hit2:
                if stack_ptr == 0:
                    break
                stack_ptr -= 1
                node_idx = stack[stack_ptr]
            elif hit1 and not hit2:
                node_idx = child1_idx
            elif not hit1 and hit2:
                node_idx = child2_idx
            else:
                stack[stack_ptr] = child2_idx
                stack_ptr += 1
                node_idx = child1_idx
