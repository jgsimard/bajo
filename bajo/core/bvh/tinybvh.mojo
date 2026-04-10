from std.algorithm import parallelize
from std.math import abs, min, max, clamp
from std.memory import UnsafePointer
from std.os.atomic import Atomic
from std.utils.numerics import max_finite, min_finite

from bajo.core.aabb import AABB
from bajo.core.intersect import intersect_ray_tri_moller, intersect_ray_aabb
from bajo.core.mat import Mat44f32, transform_point, transform_vector, inverse
from bajo.core.vec import Vec3f32, vmin, vmax, longest_axis, InlineArray

comptime f32_max = max_finite[DType.float32]()
comptime f32_min = min_finite[DType.float32]()
comptime INV_3 = Float32(1.0) / Float32(3.0)


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


struct BVH(Copyable):
    var bvh_nodes: List[BVHNode]
    var prim_indices: List[UInt32]
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

        self.prim_indices = List[UInt32](capacity=Int(tri_count))
        for i in range(Int(tri_count)):
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
            var leaf_tri_idx = Int(self.prim_indices[first + i])
            ref v0 = self.vertices[leaf_tri_idx * 3 + 0]
            ref v1 = self.vertices[leaf_tri_idx * 3 + 1]
            ref v2 = self.vertices[leaf_tri_idx * 3 + 2]

            node.aabb._min = vmin(node.aabb._min, v0, v1, v2)
            node.aabb._max = vmax(node.aabb._max, v0, v1, v2)

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
        comptime if split_method == "median":
            var extent = node.aabb._max - node.aabb._min
            axis = longest_axis(extent)
            pos = node.aabb._min[axis] + extent[axis] * 0.5
        elif split_method == "sah":
            axis, pos, cost = _sah(
                node,
                self.prim_indices.unsafe_ptr(),
                self.vertices,
            )
            if cost >= node.surface_area() * Float32(node.triCount):
                if node.triCount > MAX_LEAF_SIZE:
                    # Force a spatial median split to keep breaking the node down
                    var extent = node.aabb._max - node.aabb._min
                    axis = longest_axis(extent)
                    pos = node.aabb._min[axis] + extent[axis] * 0.5
                else:
                    return None  # Safely make a leaf

        else:
            comptime assert False, "Unknown split method"

        # Termination Criteria

        # 2. Stop splitting if we are small enough
        if node.triCount <= MAX_LEAF_SIZE:
            return None

        var split_idx = _partition_tris(
            self.prim_indices.unsafe_ptr(),
            self.vertices,
            Int(node.leftFirst),
            Int(node.triCount),
            axis,
            pos,
        )

        var left_count = UInt32(split_idx - Int(node.leftFirst))
        if left_count == 0 or left_count == node.triCount:
            comptime if split_method == "median":
                # Fallback: If spatial center failed, just split the indices 50/50
                left_count = node.triCount // 2
                split_idx = Int(node.leftFirst) + Int(left_count)
            else:
                if node.triCount > MAX_LEAF_SIZE:
                    left_count = node.triCount // 2
                    split_idx = Int(node.leftFirst) + Int(left_count)
                else:
                    # Only now is it safe to give up and make a leaf
                    return None

        # Atomic allocation (Safe for both ST and MT)
        var left_child_idx = Atomic.fetch_add(atomic_nodes, 2)

        nodes_ptr[Int(left_child_idx)].leftFirst = node.leftFirst
        nodes_ptr[Int(left_child_idx)].triCount = left_count
        nodes_ptr[Int(left_child_idx + 1)].leftFirst = UInt32(split_idx)
        nodes_ptr[Int(left_child_idx + 1)].triCount = node.triCount - left_count

        node.leftFirst = left_child_idx
        node.triCount = 0  # Internal node

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

    def build[split_method: String, is_mt: Bool](mut self):
        self.nodes_used = 1
        self.update_node_bounds(0)

        # All build modes use an atomic counter for unified API
        var atomic_nodes = alloc[Scalar[DType.uint32]](1)
        atomic_nodes[0] = self.nodes_used

        comptime if not is_mt:
            # Single-threaded: Just run the loop once from the root
            self._build_iterative[split_method](0, atomic_nodes)

        else:
            # Multi-threaded: Breadth-first "seed" to generate tasks
            var tasks: List[UInt32] = [0]
            while len(tasks) < 64:
                # Simple logic: split the largest remaining task
                var largest_idx = -1
                var max_tris = UInt32(0)
                for i in range(len(tasks)):
                    var count = self.bvh_nodes[Int(tasks[i])].triCount
                    if count > max_tris:
                        max_tris = count
                        largest_idx = i

                # If we can't split further or have no tasks, break
                if largest_idx == -1 and len(tasks) > 0:
                    break

                var target = tasks.pop(
                    largest_idx
                ) if largest_idx != -1 else UInt32(0)
                var children = self._split_node[split_method](
                    target, atomic_nodes
                )
                if children:
                    tasks.append(children.value()[0])
                    tasks.append(children.value()[1])
                else:
                    # If root couldn't split, we're done
                    if largest_idx == -1:
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
        ref v0 = self.vertices[Int(prim_idx) * 3]
        ref v1 = self.vertices[Int(prim_idx) * 3 + 1]
        ref v2 = self.vertices[Int(prim_idx) * 3 + 2]

        var t = f32_max
        var u = Float32(0)
        var v = Float32(0)
        var w = Float32(0)
        var sign = Float32(0)

        var hit = intersect_ray_tri_moller(
            ray.O,
            ray.D,
            v0,
            v1,
            v2,
            t,
            u,
            v,
            w,
            sign,
            UnsafePointer[Vec3f32, MutAnyOrigin](),
        )

        if hit and t < ray.hit.t:
            comptime if is_shadow:
                return True
            ray.hit.t = t
            ray.hit.u = u
            ray.hit.v = v
            ray.hit.prim = prim_idx
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
                    var p_idx = self.prim_indices[Int(node.leftFirst) + i]
                    if self._intersect_tri[is_shadow](ray, p_idx):
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
def _partition_tris(
    prims: UnsafePointer[UInt32, MutAnyOrigin],
    verts: UnsafePointer[Vec3f32, MutAnyOrigin],
    first: Int,
    count: Int,
    axis: Int,
    pos: Float32,
) -> Int:
    var i = first
    var j = i + count - 1
    while i <= j:
        var p_idx = Int(prims[i])
        # centroid
        var c = (
            verts[p_idx * 3].data[axis]
            + verts[p_idx * 3 + 1].data[axis]
            + verts[p_idx * 3 + 2].data[axis]
        ) * INV_3
        if c < pos:
            i += 1
        else:
            prims[i], prims[j] = prims[j], prims[i]
            j -= 1
    return i


@always_inline
def _sah(
    node: BVHNode,
    prims: UnsafePointer[UInt32, ImmutAnyOrigin],
    verts: UnsafePointer[Vec3f32, ImmutAnyOrigin],
) -> Tuple[Int, Float32, Float32]:
    comptime NB_BINS = 16

    var best_axis: Int = -1
    var best_pos: Float32 = 0
    var best_cost: Float32 = f32_max

    for axis in range(3):
        var min_c = f32_max
        var max_c = f32_min

        # 1. Find centroid range
        for i in range(Int(node.triCount)):
            var p_idx = Int(prims[Int(node.leftFirst) + i])
            ref v0 = verts[p_idx * 3 + 0]
            ref v1 = verts[p_idx * 3 + 1]
            ref v2 = verts[p_idx * 3 + 2]

            var c = (v0[axis] + v1[axis] + v2[axis]) * INV_3
            min_c = min(min_c, c)
            max_c = max(max_c, c)

        if min_c == max_c:
            continue

        # 2. Binning
        var bins = InlineArray[Bin, NB_BINS](fill=Bin())
        var scale = Float32(NB_BINS) / (max_c - min_c)
        for i in range(Int(node.triCount)):
            var p_idx = Int(prims[Int(node.leftFirst) + i])

            ref v0 = verts[p_idx * 3 + 0]
            ref v1 = verts[p_idx * 3 + 1]
            ref v2 = verts[p_idx * 3 + 2]

            var c = (v0[axis] + v1[axis] + v2[axis]) * INV_3
            var b_idx = min(NB_BINS - 1, Int((c - min_c) * scale))
            bins[b_idx].tri_count += 1
            bins[b_idx].bounds.grow(v0, v1, v2)

        # 3. Sweep SAH Costs
        var left_areas = InlineArray[Float32, NB_BINS](fill=0.0)
        var left_counts = InlineArray[UInt32, NB_BINS](fill=0)
        var left_box = AABB.invalid()
        var left_sum = UInt32(0)

        for i in range(NB_BINS - 1):
            left_sum += bins[i].tri_count
            left_counts[i] = left_sum
            left_box.grow(bins[i].bounds)
            left_areas[i] = left_box.surface_area()

        var right_box = AABB.invalid()
        var right_sum = UInt32(0)
        for i in range(NB_BINS - 1, 0, -1):
            right_sum += bins[i].tri_count
            right_box.grow(bins[i].bounds)

            var left_cost = left_areas[i - 1] * Float32(left_counts[i - 1])
            var right_cost = right_box.surface_area() * Float32(right_sum)
            var cost = left_cost + right_cost

            if cost < best_cost:
                best_cost = cost
                best_axis = axis
                best_pos = min_c + (Float32(i) / scale)

    return (best_axis, best_pos, best_cost)


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
        self.prim_indices = SIMD[DType.uint32, Self.width](0)


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
                        var p_idx = Int(
                            binary_bvh.prim_indices[Int(n.leftFirst) + tri]
                        )
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
                    # Sentinel for empty lanes
                    for tri in range(Int(n.triCount), Self.width):
                        packed.v0x[tri] = f32_max
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

        var hit_mask = (
            (t.gt(1e-4))
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

            var mask = tmin.lt(tmax) & (~node.counts.eq(0xFFFFFFFF))

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


def main() raises:
    print("--- TinyBVH Mojo Port: Parametrized SIMD Test ---")

    var vertices = List[Vec3f32]()

    # Triangle 0 (Z=2)
    vertices.append(Vec3f32(-1.0, -1.0, 2.0))
    vertices.append(Vec3f32(1.0, -1.0, 2.0))
    vertices.append(Vec3f32(0.0, 1.0, 2.0))

    # Triangle 1 (Z=4, behind Triangle 0)
    vertices.append(Vec3f32(-1.0, -1.0, 4.0))
    vertices.append(Vec3f32(1.0, -1.0, 4.0))
    vertices.append(Vec3f32(0.0, 1.0, 4.0))

    # Triangle 2 (Off to the right)
    vertices.append(Vec3f32(3.0, -1.0, 2.0))
    vertices.append(Vec3f32(5.0, -1.0, 2.0))
    vertices.append(Vec3f32(4.0, 1.0, 2.0))

    var tri_count = UInt32(len(vertices) // 3)

    print("\n[1] Building Base Binary BVH for", tri_count, "triangles...")
    var bvh = BVH(vertices.unsafe_ptr(), tri_count)
    bvh.build["median", False]()
    print("Binary Nodes used:", bvh.nodes_used)

    print("\n[2] Collapsing to Wide SIMD BVH4...")
    var bvh4 = WideBVH[4](bvh)
    print("BVH4 Nodes used:", len(bvh4.nodes))

    print("\n[3] Collapsing to Wide SIMD BVH8...")
    var bvh8 = WideBVH[8](bvh)
    print("BVH8 Nodes used:", len(bvh8.nodes))

    # Fire a Ray! Origin at (0, 0, 0), looking down +Z
    var ray_base = Ray(Vec3f32(0.0, 0.0, 0.0), Vec3f32(0.0, 0.0, 1.0))

    print("\n--- Testing Traversal (BVH4) ---")
    var ray_bvh4 = ray_base.copy()
    bvh4.traverse(ray_bvh4)
    if ray_bvh4.hit.t < 1e29:
        print(
            t"HIT! t: {ray_bvh4.hit.t}| tri: {ray_bvh4.hit.prim} | bary:"
            t" u={ray_bvh4.hit.u}, v={ray_bvh4.hit.v}"
        )
    else:
        print("MISS")

    print("\n--- Testing Traversal (BVH8) ---")
    var ray_bvh8 = ray_base.copy()
    bvh8.traverse(ray_bvh8)
    if ray_bvh8.hit.t < 1e29:
        print(
            t"HIT! t: {ray_bvh8.hit.t}| tri: {ray_bvh8.hit.prim} | bary:"
            t" u={ray_bvh8.hit.u}, v={ray_bvh8.hit.v}"
        )
    else:
        print("MISS")

    print("\n\n--- 2-Layer TLAS/BLAS Demo ---")
    # 1. Generate cube geometry
    var chair_vertices = create_cube_mesh()
    tri_count = UInt32(len(chair_vertices) // 3)

    # 2. Build the BLAS (The "Blueprint")
    # We cast to MutAnyOrigin because the BVH expects to be able to sort the indices
    var chair_blas = BVH(
        chair_vertices.unsafe_ptr(),
        tri_count,
    )
    chair_blas.build["sah", True]()

    # Pointer for TLAS traversal
    var blases = List[BVH]()
    blases.append(chair_blas.copy())
    var blases_ptr = blases.unsafe_ptr()

    # 3. Create Scene Instances (The "Map")
    var instances = List[Instance]()
    var b_min = chair_blas.bvh_nodes[0].aabb._min.copy()
    var b_max = chair_blas.bvh_nodes[0].aabb._max.copy()

    # Instance 0: Original Cube at origin
    var transform1 = Mat44f32.identity()
    instances.append(Instance(transform1, inverse(transform1), 0, b_min, b_max))

    # Instance 1: Cube shifted to X=20, Scaled 5x
    from bajo.core.mat import _matmul

    var t_mat = translation_matrix(Vec3f32(20.0, 0.0, 0.0))
    var s_mat = scaling_matrix(Vec3f32(5.0, 5.0, 5.0))
    var transform2 = _matmul(t_mat, s_mat)

    instances.append(Instance(transform2, inverse(transform2), 0, b_min, b_max))

    # 4. Build the TLAS
    var tlas = TLAS(
        instances.unsafe_ptr(),
        UInt32(len(instances)),
    )
    tlas.build()

    # 5. Raytest
    # Aiming at the big cube at X=20
    var ray = Ray(Vec3f32(20.0, 0.0, -50.0), Vec3f32(0.0, 0.0, 1.0))
    tlas.traverse(ray, blases_ptr)

    if ray.hit.t < 1e29:
        print(t"Hit Instance: {ray.hit.inst}, Expected: 1")
        # Expected: ~47.5 (Origin -50 hitting front face at Z= -2.5 (0.5 * 5 scale))
        print(t"Distance t:  {ray.hit.t}, Expected 47.5")


def translation_matrix(v: Vec3f32) -> Mat44f32:
    var m = Mat44f32.identity()
    m[0][3] = v.x()
    m[1][3] = v.y()
    m[2][3] = v.z()
    return m^


def scaling_matrix(s: Vec3f32) -> Mat44f32:
    var m = Mat44f32.identity()
    m[0][0] = s.x()
    m[1][1] = s.y()
    m[2][2] = s.z()
    return m^


def create_cube_mesh() -> List[Vec3f32]:
    var v = List[Vec3f32]()

    # Helper to add a quad (2 triangles)
    @parameter
    def add_quad(p1: Vec3f32, p2: Vec3f32, p3: Vec3f32, p4: Vec3f32):
        v.append(p1.copy())
        v.append(p2.copy())
        v.append(p3.copy())
        v.append(p1.copy())
        v.append(p3.copy())
        v.append(p4.copy())

    # Corner coordinates
    var c0 = Vec3f32(-0.5, -0.5, -0.5)
    var c1 = Vec3f32(0.5, -0.5, -0.5)
    var c2 = Vec3f32(0.5, 0.5, -0.5)
    var c3 = Vec3f32(-0.5, 0.5, -0.5)
    var c4 = Vec3f32(-0.5, -0.5, 0.5)
    var c5 = Vec3f32(0.5, -0.5, 0.5)
    var c6 = Vec3f32(0.5, 0.5, 0.5)
    var c7 = Vec3f32(-0.5, 0.5, 0.5)

    # Add the 6 faces
    add_quad(c0, c3, c2, c1)  # Back
    add_quad(c5, c6, c7, c4)  # Front
    add_quad(c4, c7, c3, c0)  # Left
    add_quad(c1, c2, c6, c5)  # Right
    add_quad(c0, c1, c5, c4)  # Bottom
    add_quad(c3, c7, c6, c2)  # Top

    return v^
