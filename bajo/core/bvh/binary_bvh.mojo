from std.algorithm import parallelize
from std.bit import count_leading_zeros
from std.math import abs, min, max, clamp
from std.memory import UnsafePointer
from std.atomic import Atomic
from std.utils.numerics import max_finite, min_finite

from bajo.core.aabb import AABB
from bajo.core.intersect import intersect_ray_aabb, intersect_ray_tri
from bajo.core.mat import Mat44f32, transform_point, transform_vector, inverse
from bajo.core.vec import Vec3f32, vmin, vmax, longest_axis
from bajo.core.morton import morton3
from bajo.core.bvh.types import (
    Intersection,
    Ray,
    BVHNode,
    Bin,
    Fragment,
    MortonPrim,
)
from bajo.core.bvh.build import (
    _sah,
    _partition_fragments_by_bin,
    _partition_fragments,
    _find_lbvh_split,
    _morton3_scalar,
    _morton_pair_less,
)

comptime f32_max = max_finite[DType.float32]()
comptime f32_min = min_finite[DType.float32]()


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
        self.bvh_nodes = List[BVHNode](length=max_nodes, fill=BVHNode())

        self.fragments = List[Fragment](capacity=Int(tri_count))
        self.prim_indices = List[UInt32](capacity=Int(tri_count))
        for i in range(Int(tri_count)):
            ref v0 = vertices[i * 3 + 0]
            ref v1 = vertices[i * 3 + 1]
            ref v2 = vertices[i * 3 + 2]
            self.fragments.append(Fragment(UInt32(i), v0, v1, v2))
            self.prim_indices.append(UInt32(i))

        ref root = self.bvh_nodes[0]
        root.left_first = 0
        root.tri_count = tri_count

        self.update_node_bounds(0)

    @always_inline
    def update_node_bounds(mut self, node_idx: UInt32):
        ref node = self.bvh_nodes[Int(node_idx)]
        node.aabb = AABB.invalid()

        var first = Int(node.left_first)
        for i in range(Int(node.tri_count)):
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
            var leaf_cost = node.surface_area() * Float32(node.tri_count)
            var split_cost = split.cost + node.surface_area()

            if (not split.valid()) or split_cost >= leaf_cost:
                if node.tri_count > MAX_LEAF_SIZE:
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
        if node.tri_count <= MAX_LEAF_SIZE:
            return None

        var split_idx: Int
        comptime if split_method == "sah":
            if use_sah_bounds:
                split_idx = _partition_fragments_by_bin(
                    self.prim_indices.unsafe_ptr(),
                    self.fragments.unsafe_ptr(),
                    Int(node.left_first),
                    Int(node.tri_count),
                    axis,
                    split_bin,
                    split_bin_min,
                    split_bin_scale,
                )
            else:
                split_idx = _partition_fragments(
                    self.prim_indices.unsafe_ptr(),
                    self.fragments.unsafe_ptr(),
                    Int(node.left_first),
                    Int(node.tri_count),
                    axis,
                    pos,
                )
        else:
            split_idx = _partition_fragments(
                self.prim_indices.unsafe_ptr(),
                self.fragments.unsafe_ptr(),
                Int(node.left_first),
                Int(node.tri_count),
                axis,
                pos,
            )

        var left_count = UInt32(split_idx - Int(node.left_first))
        if left_count == 0 or left_count == node.tri_count:
            use_sah_bounds = False
            comptime if split_method == "median":
                # Fallback: If spatial center failed, just split the indices 50/50.
                left_count = node.tri_count // 2
                split_idx = Int(node.left_first) + Int(left_count)
            else:
                if node.tri_count > MAX_LEAF_SIZE:
                    left_count = node.tri_count // 2
                    split_idx = Int(node.left_first) + Int(left_count)
                else:
                    # Only now is it safe to give up and make a leaf.
                    return None

        # Atomic allocation (Safe for both ST and MT)
        var left_child_idx = Atomic.fetch_add(atomic_nodes, 2)

        nodes_ptr[Int(left_child_idx)].left_first = node.left_first
        nodes_ptr[Int(left_child_idx)].tri_count = left_count
        nodes_ptr[Int(left_child_idx + 1)].left_first = UInt32(split_idx)
        nodes_ptr[Int(left_child_idx + 1)].tri_count = (
            node.tri_count - left_count
        )

        node.left_first = left_child_idx
        node.tri_count = 0  # Internal node

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
    ) -> AABB:
        comptime MAX_LEAF_SIZE = 4

        if count <= MAX_LEAF_SIZE:
            ref leaf = self.bvh_nodes[Int(node_idx)]
            leaf.left_first = UInt32(first)
            leaf.tri_count = UInt32(count)
            leaf.aabb = AABB.invalid()

            for i in range(count):
                var frag_idx = Int(self.prim_indices[first + i])
                self.fragments[frag_idx].grow_into(leaf.aabb)

            return leaf.aabb.copy()

        var split = _find_lbvh_split(pairs, first, first + count)
        var left_count = split - first
        if left_count <= 0 or left_count >= count:
            split = first + count // 2
            left_count = split - first

        var left_child_idx = self.nodes_used
        self.nodes_used += 2

        var left_bounds = self._build_lbvh_recursive(
            pairs, left_child_idx, first, left_count
        )
        var right_bounds = self._build_lbvh_recursive(
            pairs, left_child_idx + 1, split, count - left_count
        )

        ref node = self.bvh_nodes[Int(node_idx)]
        node.left_first = left_child_idx
        node.tri_count = 0
        node.aabb = AABB.invalid()
        node.aabb.grow(left_bounds)
        node.aabb.grow(right_bounds)

        return node.aabb.copy()

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

        _ = self._build_lbvh_recursive(
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
                        var count = self.bvh_nodes[Int(tasks[i])].tri_count
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
            return cost_aabb + area * Float32(node.tri_count) * C_intersect
        else:
            # Cost of evaluating this AABB + expected cost of traversing children
            var left_cost = self.sah_cost(node.left_first)
            var right_cost = self.sah_cost(node.left_first + 1)
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
        ref v0 = self.vertices[Int(prim_idx) * 3]
        ref v1 = self.vertices[Int(prim_idx) * 3 + 1]
        ref v2 = self.vertices[Int(prim_idx) * 3 + 2]

        h = intersect_ray_tri(
            ray.O.x(),
            ray.O.y(),
            ray.O.z(),
            ray.D.x(),
            ray.D.y(),
            ray.D.z(),
            v0.x(),
            v0.y(),
            v0.z(),
            v1.x(),
            v1.y(),
            v1.z(),
            v2.x(),
            v2.y(),
            v2.z(),
            ray.hit.t,
        )
        if not h.mask[0]:
            return False

        comptime if is_shadow:
            return True

        ray.hit.t = h.t[0]
        ray.hit.u = h.u[0]
        ray.hit.v = h.v[0]
        ray.hit.prim = prim_idx
        return True

    @always_inline
    def _traverse[is_shadow: Bool](self, mut ray: Ray) -> Bool:
        var stack = InlineArray[UInt32, 64](fill=0)
        var stack_ptr = 0
        var node_idx = UInt32(0)

        while True:
            ref node = self.bvh_nodes[Int(node_idx)]

            if node.is_leaf():
                for i in range(Int(node.tri_count)):
                    var frag_idx = self.prim_indices[Int(node.left_first) + i]
                    var p_idx = self.fragments[Int(frag_idx)].prim_idx
                    if self._intersect_tri[is_shadow](ray, p_idx):
                        comptime if is_shadow:
                            return True

                if stack_ptr == 0:
                    break
                stack_ptr -= 1
                node_idx = stack[stack_ptr]
                continue

            var child1_idx = node.left_first
            var child2_idx = node.left_first + 1
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
