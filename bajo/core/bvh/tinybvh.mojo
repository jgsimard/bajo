from std.math import abs, min, max, clamp
from std.memory import UnsafePointer
from std.algorithm import parallelize
from std.os.atomic import Atomic


from bajo.core.vec import Vec3f32, vmin, vmax, longest_axis, InlineArray
from bajo.core.intersect import intersect_ray_tri_moller, intersect_ray_aabb


@fieldwise_init
struct Intersection(Copyable):
    var t: Float32
    var u: Float32
    var v: Float32
    var inst: UInt32
    var prim: UInt32

    @always_inline
    def __init__(out self):
        self.t = 1e30
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

    @always_inline
    def __init__(out self):
        self.O = Vec3f32(0.0)
        self.mask = 0xFFFFFFFF
        self.D = Vec3f32(0.0, 0.0, 1.0)
        self.rD = Vec3f32(1e30)
        self.hit = Intersection()

    @always_inline
    def __init__(out self, O: Vec3f32, D: Vec3f32, t_max: Float32 = 1e30):
        self.O = O.copy()
        self.D = D.copy()
        var rDx = Float32(1.0) / D.x() if abs(D.x()) > Float32(1e-12) else (
            Float32(1e30) if D.x() >= 0.0 else Float32(-1e30)
        )
        var rDy = Float32(1.0) / D.y() if abs(D.y()) > Float32(1e-12) else (
            Float32(1e30) if D.y() >= 0.0 else Float32(-1e30)
        )
        var rDz = Float32(1.0) / D.z() if abs(D.z()) > Float32(1e-12) else (
            Float32(1e30) if D.z() >= 0.0 else Float32(-1e30)
        )
        self.rD = Vec3f32(rDx, rDy, rDz)
        self.mask = 0xFFFFFFFF
        self.hit = Intersection()
        self.hit.t = t_max


@fieldwise_init
struct BVHNode(Copyable):
    var aabbMin: Vec3f32
    var leftFirst: UInt32
    var aabbMax: Vec3f32
    var triCount: UInt32

    @always_inline
    def __init__(out self):
        self.aabbMin = Vec3f32(1e30)
        self.leftFirst = 0
        self.aabbMax = Vec3f32(-1e30)
        self.triCount = 0

    @always_inline
    def is_leaf(self) -> Bool:
        return self.triCount > 0

    @always_inline
    def surface_area(self) -> Float32:
        var e = self.aabbMax - self.aabbMin
        return e.x() * e.y() + e.y() * e.z() + e.z() * e.x()


@fieldwise_init
struct Bin(Copyable):
    var bounds: BVHNode  # We repurpose BVHNode for its AABB storage
    var tri_count: UInt32

    def __init__(out self):
        self.bounds = BVHNode()  # Initializes to infinity
        self.tri_count = 0


struct BVH:
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
        node.aabbMin = Vec3f32(1e30)
        node.aabbMax = Vec3f32(-1e30)

        var first = Int(node.leftFirst)
        for i in range(Int(node.triCount)):
            var leaf_tri_idx = Int(self.prim_indices[first + i])
            ref v0 = self.vertices[leaf_tri_idx * 3 + 0]
            ref v1 = self.vertices[leaf_tri_idx * 3 + 1]
            ref v2 = self.vertices[leaf_tri_idx * 3 + 2]

            node.aabbMin = vmin(node.aabbMin, v0, v1, v2)
            node.aabbMax = vmax(node.aabbMax, v0, v1, v2)

    def build_quick(mut self):
        self.subdivide_quick(0)

    def subdivide_quick(mut self, node_idx: UInt32):
        var node = self.bvh_nodes[Int(node_idx)].copy()

        if node.triCount <= 2:
            return

        var extent = node.aabbMax - node.aabbMin
        var axis = longest_axis(extent)
        var split_pos = node.aabbMin[axis] + extent[axis] * 0.5

        var i = Int(node.leftFirst)
        var j = i + Int(node.triCount) - 1

        while i <= j:
            var prim_idx = Int(self.prim_indices[i])
            var v0 = self.vertices[prim_idx * 3 + 0].copy()
            var v1 = self.vertices[prim_idx * 3 + 1].copy()
            var v2 = self.vertices[prim_idx * 3 + 2].copy()

            var centroid = (v0[axis] + v1[axis] + v2[axis]) * Float32(0.3333333)

            if centroid < split_pos:
                i += 1
            else:
                var tmp = self.prim_indices[i]
                self.prim_indices[i] = self.prim_indices[j]
                self.prim_indices[j] = tmp
                j -= 1

        var left_count = UInt32(i - Int(node.leftFirst))

        if left_count == 0 or left_count == node.triCount:
            left_count = node.triCount // 2
            i = Int(node.leftFirst) + Int(left_count)

        var left_child_idx = self.nodes_used
        self.nodes_used += 2

        var left_child = self.bvh_nodes[Int(left_child_idx)].copy()
        var right_child = self.bvh_nodes[Int(left_child_idx + 1)].copy()

        left_child.leftFirst = node.leftFirst
        left_child.triCount = left_count
        right_child.leftFirst = UInt32(i)
        right_child.triCount = node.triCount - left_count

        self.bvh_nodes[Int(left_child_idx)] = left_child^
        self.bvh_nodes[Int(left_child_idx + 1)] = right_child^

        node.leftFirst = left_child_idx
        node.triCount = 0
        self.bvh_nodes[Int(node_idx)] = node^

        self.update_node_bounds(left_child_idx)
        self.update_node_bounds(left_child_idx + 1)

        self.subdivide_quick(left_child_idx)
        self.subdivide_quick(left_child_idx + 1)

    def build_sah(mut self):
        """Entry point for the high-quality SAH builder."""
        self.subdivide_sah(0)

    def subdivide_sah(mut self, node_idx: UInt32):
        var node = self.bvh_nodes[Int(node_idx)].copy()

        # 1. Find the best split using SAH
        var best_axis: Int = -1
        var best_pos: Float32 = 0
        var best_cost: Float32 = 1e30

        # We test 16 bins per axis
        comptime BINS = 16

        for axis in range(3):
            # Find the range of centroids on this axis
            var min_c = Float32(1e30)
            var max_c = Float32(-1e30)
            for i in range(Int(node.triCount)):
                var p_idx = Int(self.prim_indices[Int(node.leftFirst) + i])
                var centroid = (
                    self.vertices[p_idx * 3].data[axis]
                    + self.vertices[p_idx * 3 + 1].data[axis]
                    + self.vertices[p_idx * 3 + 2].data[axis]
                ) * 0.3333333
                min_c = min(min_c, centroid)
                max_c = max(max_c, centroid)

            if min_c == max_c:
                continue  # Skip flat axis

            # Populate Bins
            var bins = InlineArray[Bin, BINS](fill=Bin())
            var scale = Float32(BINS) / (max_c - min_c)

            for i in range(Int(node.triCount)):
                var p_idx = Int(self.prim_indices[Int(node.leftFirst) + i])
                var centroid = (
                    self.vertices[p_idx * 3].data[axis]
                    + self.vertices[p_idx * 3 + 1].data[axis]
                    + self.vertices[p_idx * 3 + 2].data[axis]
                ) * 0.3333333
                var bin_idx = min(BINS - 1, Int((centroid - min_c) * scale))

                bins[bin_idx].tri_count += 1
                # Grow bin bounds using triangle vertices
                for v_off in range(3):
                    var v = self.vertices[p_idx * 3 + v_off].copy()
                    bins[bin_idx].bounds.aabbMin = vmin(
                        bins[bin_idx].bounds.aabbMin, v
                    )
                    bins[bin_idx].bounds.aabbMax = vmax(
                        bins[bin_idx].bounds.aabbMax, v
                    )

            # Evaluate Costs (Sweep from left and right)
            # Area(Left) * Count(Left) + Area(Right) * Count(Right)
            var left_areas = InlineArray[Float32, BINS](fill=0.0)
            var left_counts = InlineArray[UInt32, BINS](fill=0)
            var left_box = BVHNode()
            var left_sum = UInt32(0)

            for i in range(BINS - 1):
                left_sum += bins[i].tri_count
                left_counts[i] = left_sum
                left_box.aabbMin = vmin(
                    left_box.aabbMin, bins[i].bounds.aabbMin
                )
                left_box.aabbMax = vmax(
                    left_box.aabbMax, bins[i].bounds.aabbMax
                )
                left_areas[i] = left_box.surface_area()

            var right_box = BVHNode()
            var right_sum = UInt32(0)
            for i in range(BINS - 1, 0, -1):
                right_sum += bins[i].tri_count
                right_box.aabbMin = vmin(
                    right_box.aabbMin, bins[i].bounds.aabbMin
                )
                right_box.aabbMax = vmax(
                    right_box.aabbMax, bins[i].bounds.aabbMax
                )

                var cost = left_areas[i - 1] * Float32(
                    left_counts[i - 1]
                ) + right_box.surface_area() * Float32(right_sum)

                if cost < best_cost:
                    best_cost = cost
                    best_axis = axis
                    best_pos = min_c + (Float32(i) / scale)

        # 2. Compare SAH cost vs Leaf cost
        var leaf_cost = node.surface_area() * Float32(node.triCount)
        if best_cost >= leaf_cost:
            return  # Stopping: Building a leaf is cheaper than splitting

        # 3. Perform the Partition based on the best SAH candidate
        var i = Int(node.leftFirst)
        var j = i + Int(node.triCount) - 1
        while i <= j:
            var p_idx = Int(self.prim_indices[i])
            var centroid = (
                self.vertices[p_idx * 3].data[best_axis]
                + self.vertices[p_idx * 3 + 1].data[best_axis]
                + self.vertices[p_idx * 3 + 2].data[best_axis]
            ) * 0.3333333
            if centroid < best_pos:
                i += 1
            else:
                var tmp = self.prim_indices[i]
                self.prim_indices[i] = self.prim_indices[j]
                self.prim_indices[j] = tmp
                j -= 1

        # 4. Finalize and Recurse (Same logic as Quick build)
        var left_count = UInt32(i - Int(node.leftFirst))
        if left_count == 0 or left_count == node.triCount:
            return

        var left_child_idx = self.nodes_used
        self.nodes_used += 2

        self.bvh_nodes[Int(left_child_idx)].leftFirst = node.leftFirst
        self.bvh_nodes[Int(left_child_idx)].triCount = left_count
        self.bvh_nodes[Int(left_child_idx + 1)].leftFirst = UInt32(i)
        self.bvh_nodes[Int(left_child_idx + 1)].triCount = (
            node.triCount - left_count
        )

        node.leftFirst = left_child_idx
        node.triCount = 0
        self.bvh_nodes[Int(node_idx)] = node^

        self.update_node_bounds(left_child_idx)
        self.update_node_bounds(left_child_idx + 1)

        self.subdivide_sah(left_child_idx)
        self.subdivide_sah(left_child_idx + 1)

    def build_sah_mt(mut self):
        """Multithreaded SAH Builder."""
        # 1. Extract raw pointers to bypass the borrow-checker during parallelization
        var nodes_ptr = self.bvh_nodes.unsafe_ptr().unsafe_origin_cast[
            MutAnyOrigin
        ]()
        var prims_ptr = self.prim_indices.unsafe_ptr().unsafe_origin_cast[
            MutAnyOrigin
        ]()
        var verts_ptr = self.vertices

        var queue = List[UInt32]()
        queue.append(0)
        var sub_roots = List[UInt32]()

        # 2. Single-threaded Breadth-First split to generate independent tasks
        while len(queue) > 0 and (len(queue) + len(sub_roots)) < 64:
            var node_idx = queue.pop(0)
            var node = nodes_ptr[Int(node_idx)].copy()

            if node.triCount < 1000:  # Too small to be worth a new thread task
                sub_roots.append(node_idx)
                continue

            var best_axis = -1
            var best_pos = Float32(0)
            var best_cost = Float32(1e30)
            comptime BINS = 16

            for axis in range(3):
                var min_c = Float32(1e30)
                var max_c = Float32(-1e30)
                for i in range(Int(node.triCount)):
                    var p_idx = Int(prims_ptr[Int(node.leftFirst) + i])
                    var c = (
                        verts_ptr[p_idx * 3].data[axis]
                        + verts_ptr[p_idx * 3 + 1].data[axis]
                        + verts_ptr[p_idx * 3 + 2].data[axis]
                    ) * 0.3333333
                    min_c = min(min_c, c)
                    max_c = max(max_c, c)
                if min_c == max_c:
                    continue

                var bins = InlineArray[Bin, BINS](fill=Bin())
                var scale = Float32(BINS) / (max_c - min_c)
                for i in range(Int(node.triCount)):
                    var p_idx = Int(prims_ptr[Int(node.leftFirst) + i])
                    var c = (
                        verts_ptr[p_idx * 3].data[axis]
                        + verts_ptr[p_idx * 3 + 1].data[axis]
                        + verts_ptr[p_idx * 3 + 2].data[axis]
                    ) * 0.3333333
                    var bin_idx = min(BINS - 1, Int((c - min_c) * scale))
                    bins[bin_idx].tri_count += 1
                    for v_off in range(3):
                        var v = verts_ptr[p_idx * 3 + v_off].copy()
                        bins[bin_idx].bounds.aabbMin = vmin(
                            bins[bin_idx].bounds.aabbMin, v
                        )
                        bins[bin_idx].bounds.aabbMax = vmax(
                            bins[bin_idx].bounds.aabbMax, v
                        )

                var left_areas = InlineArray[Float32, BINS](fill=0.0)
                var left_counts = InlineArray[UInt32, BINS](fill=0)
                var left_box = BVHNode()
                var left_sum = UInt32(0)
                for i in range(BINS - 1):
                    left_sum += bins[i].tri_count
                    left_counts[i] = left_sum
                    left_box.aabbMin = vmin(
                        left_box.aabbMin, bins[i].bounds.aabbMin
                    )
                    left_box.aabbMax = vmax(
                        left_box.aabbMax, bins[i].bounds.aabbMax
                    )
                    left_areas[i] = left_box.surface_area()

                var right_box = BVHNode()
                var right_sum = UInt32(0)
                for i in range(BINS - 1, 0, -1):
                    right_sum += bins[i].tri_count
                    right_box.aabbMin = vmin(
                        right_box.aabbMin, bins[i].bounds.aabbMin
                    )
                    right_box.aabbMax = vmax(
                        right_box.aabbMax, bins[i].bounds.aabbMax
                    )
                    var cost = left_areas[i - 1] * Float32(
                        left_counts[i - 1]
                    ) + right_box.surface_area() * Float32(right_sum)
                    if cost < best_cost:
                        best_cost = cost
                        best_axis = axis
                        best_pos = min_c + (Float32(i) / scale)

            var leaf_cost = node.surface_area() * Float32(node.triCount)
            if best_cost >= leaf_cost:
                sub_roots.append(node_idx)
                continue

            var i = Int(node.leftFirst)
            var j = i + Int(node.triCount) - 1
            while i <= j:
                var p_idx = Int(prims_ptr[i])
                var c = (
                    verts_ptr[p_idx * 3].data[best_axis]
                    + verts_ptr[p_idx * 3 + 1].data[best_axis]
                    + verts_ptr[p_idx * 3 + 2].data[best_axis]
                ) * 0.3333333
                if c < best_pos:
                    i += 1
                else:
                    var tmp = prims_ptr[i]
                    prims_ptr[i] = prims_ptr[j]
                    prims_ptr[j] = tmp
                    j -= 1

            var left_count = UInt32(i - Int(node.leftFirst))
            if left_count == 0 or left_count == node.triCount:
                sub_roots.append(node_idx)
                continue

            var left_child_idx = self.nodes_used
            self.nodes_used += 2

            nodes_ptr[Int(left_child_idx)].leftFirst = node.leftFirst
            nodes_ptr[Int(left_child_idx)].triCount = left_count
            nodes_ptr[Int(left_child_idx + 1)].leftFirst = UInt32(i)
            nodes_ptr[Int(left_child_idx + 1)].triCount = (
                node.triCount - left_count
            )

            node.leftFirst = left_child_idx
            node.triCount = 0
            nodes_ptr[Int(node_idx)] = node^

            for child in range(2):
                var c_idx = left_child_idx + UInt32(child)
                var c_node = nodes_ptr[Int(c_idx)].copy()
                c_node.aabbMin = Vec3f32(1e30)
                c_node.aabbMax = Vec3f32(-1e30)
                for t_idx in range(Int(c_node.triCount)):
                    var leaf_tri_idx = Int(
                        prims_ptr[Int(c_node.leftFirst) + t_idx]
                    )
                    c_node.aabbMin = vmin(
                        c_node.aabbMin,
                        vmin(
                            verts_ptr[leaf_tri_idx * 3].copy(),
                            vmin(
                                verts_ptr[leaf_tri_idx * 3 + 1].copy(),
                                verts_ptr[leaf_tri_idx * 3 + 2].copy(),
                            ),
                        ),
                    )
                    c_node.aabbMax = vmax(
                        c_node.aabbMax,
                        vmax(
                            verts_ptr[leaf_tri_idx * 3].copy(),
                            vmax(
                                verts_ptr[leaf_tri_idx * 3 + 1].copy(),
                                verts_ptr[leaf_tri_idx * 3 + 2].copy(),
                            ),
                        ),
                    )
                nodes_ptr[Int(c_idx)] = c_node^
                queue.append(c_idx)

        for i in range(len(queue)):
            sub_roots.append(queue[i])

        var sub_roots_ptr = sub_roots.unsafe_ptr()

        # 3. Setup Shared Atomic Counter
        var atomic_nodes = alloc[Scalar[DType.uint32]](1)
        atomic_nodes[0] = self.nodes_used

        # 4. Multithreaded Processing
        @parameter
        def worker(idx: Int):
            var root_idx = sub_roots_ptr[idx]
            BVH._subdivide_sah_worker(
                root_idx,
                nodes_ptr,
                prims_ptr,
                verts_ptr,
                atomic_nodes.unsafe_origin_cast[MutAnyOrigin](),
            )

        parallelize[worker](len(sub_roots))

        # 5. Cleanup
        self.nodes_used = atomic_nodes[0]
        atomic_nodes.free()

    @staticmethod
    def _subdivide_sah_worker(
        node_idx: UInt32,
        nodes_ptr: UnsafePointer[BVHNode, MutAnyOrigin],
        prims_ptr: UnsafePointer[UInt32, MutAnyOrigin],
        verts_ptr: UnsafePointer[Vec3f32, MutAnyOrigin],
        atomic_nodes: UnsafePointer[Scalar[DType.uint32], MutAnyOrigin],
    ):
        var node = nodes_ptr[Int(node_idx)].copy()

        var best_axis = -1
        var best_pos = Float32(0)
        var best_cost = Float32(1e30)
        comptime BINS = 16

        for axis in range(3):
            var min_c = Float32(1e30)
            var max_c = Float32(-1e30)
            for i in range(Int(node.triCount)):
                var p_idx = Int(prims_ptr[Int(node.leftFirst) + i])
                var c = (
                    verts_ptr[p_idx * 3].data[axis]
                    + verts_ptr[p_idx * 3 + 1].data[axis]
                    + verts_ptr[p_idx * 3 + 2].data[axis]
                ) * 0.3333333
                min_c = min(min_c, c)
                max_c = max(max_c, c)
            if min_c == max_c:
                continue

            var bins = InlineArray[Bin, BINS](fill=Bin())
            var scale = Float32(BINS) / (max_c - min_c)
            for i in range(Int(node.triCount)):
                var p_idx = Int(prims_ptr[Int(node.leftFirst) + i])
                var c = (
                    verts_ptr[p_idx * 3].data[axis]
                    + verts_ptr[p_idx * 3 + 1].data[axis]
                    + verts_ptr[p_idx * 3 + 2].data[axis]
                ) * 0.3333333
                var bin_idx = min(BINS - 1, Int((c - min_c) * scale))
                bins[bin_idx].tri_count += 1
                for v_off in range(3):
                    var v = verts_ptr[p_idx * 3 + v_off].copy()
                    bins[bin_idx].bounds.aabbMin = vmin(
                        bins[bin_idx].bounds.aabbMin, v
                    )
                    bins[bin_idx].bounds.aabbMax = vmax(
                        bins[bin_idx].bounds.aabbMax, v
                    )

            var left_areas = InlineArray[Float32, BINS](fill=0.0)
            var left_counts = InlineArray[UInt32, BINS](fill=0)
            var left_box = BVHNode()
            var left_sum = UInt32(0)
            for i in range(BINS - 1):
                left_sum += bins[i].tri_count
                left_counts[i] = left_sum
                left_box.aabbMin = vmin(
                    left_box.aabbMin, bins[i].bounds.aabbMin
                )
                left_box.aabbMax = vmax(
                    left_box.aabbMax, bins[i].bounds.aabbMax
                )
                left_areas[i] = left_box.surface_area()

            var right_box = BVHNode()
            var right_sum = UInt32(0)
            for i in range(BINS - 1, 0, -1):
                right_sum += bins[i].tri_count
                right_box.aabbMin = vmin(
                    right_box.aabbMin, bins[i].bounds.aabbMin
                )
                right_box.aabbMax = vmax(
                    right_box.aabbMax, bins[i].bounds.aabbMax
                )
                var cost = left_areas[i - 1] * Float32(
                    left_counts[i - 1]
                ) + right_box.surface_area() * Float32(right_sum)
                if cost < best_cost:
                    best_cost = cost
                    best_axis = axis
                    best_pos = min_c + (Float32(i) / scale)

        var leaf_cost = node.surface_area() * Float32(node.triCount)
        if best_cost >= leaf_cost:
            return

        var i = Int(node.leftFirst)
        var j = i + Int(node.triCount) - 1
        while i <= j:
            var p_idx = Int(prims_ptr[i])
            var c = (
                verts_ptr[p_idx * 3].data[best_axis]
                + verts_ptr[p_idx * 3 + 1].data[best_axis]
                + verts_ptr[p_idx * 3 + 2].data[best_axis]
            ) * 0.3333333
            if c < best_pos:
                i += 1
            else:
                var tmp = prims_ptr[i]
                prims_ptr[i] = prims_ptr[j]
                prims_ptr[j] = tmp
                j -= 1

        var left_count = UInt32(i - Int(node.leftFirst))
        if left_count == 0 or left_count == node.triCount:
            return

        # Atomic fetch-add requesting 2 nodes (returns the old value)
        var left_child_idx = Atomic.fetch_add(atomic_nodes, 2)

        nodes_ptr[Int(left_child_idx)].leftFirst = node.leftFirst
        nodes_ptr[Int(left_child_idx)].triCount = left_count
        nodes_ptr[Int(left_child_idx + 1)].leftFirst = UInt32(i)
        nodes_ptr[Int(left_child_idx + 1)].triCount = node.triCount - left_count

        node.leftFirst = left_child_idx
        node.triCount = 0
        nodes_ptr[Int(node_idx)] = node^

        for child in range(2):
            var c_idx = left_child_idx + UInt32(child)
            var c_node = nodes_ptr[Int(c_idx)].copy()
            c_node.aabbMin = Vec3f32(1e30)
            c_node.aabbMax = Vec3f32(-1e30)
            for t_idx in range(Int(c_node.triCount)):
                var leaf_tri_idx = Int(prims_ptr[Int(c_node.leftFirst) + t_idx])
                c_node.aabbMin = vmin(
                    c_node.aabbMin,
                    vmin(
                        verts_ptr[leaf_tri_idx * 3].copy(),
                        vmin(
                            verts_ptr[leaf_tri_idx * 3 + 1].copy(),
                            verts_ptr[leaf_tri_idx * 3 + 2].copy(),
                        ),
                    ),
                )
                c_node.aabbMax = vmax(
                    c_node.aabbMax,
                    vmax(
                        verts_ptr[leaf_tri_idx * 3].copy(),
                        vmax(
                            verts_ptr[leaf_tri_idx * 3 + 1].copy(),
                            verts_ptr[leaf_tri_idx * 3 + 2].copy(),
                        ),
                    ),
                )
            nodes_ptr[Int(c_idx)] = c_node^

        BVH._subdivide_sah_worker(
            left_child_idx, nodes_ptr, prims_ptr, verts_ptr, atomic_nodes
        )
        BVH._subdivide_sah_worker(
            left_child_idx + 1, nodes_ptr, prims_ptr, verts_ptr, atomic_nodes
        )

    def sah_cost(self, node_idx: UInt32 = 0) -> Float32:
        """Recursively calculates the unnormalized SAH cost of the subtree."""
        var node = self.bvh_nodes[Int(node_idx)].copy()
        var area = node.surface_area()

        # Standard SAH constants
        var C_traverse = Float32(1.0)
        var C_intersect = Float32(1.0)

        if node.is_leaf():
            # Cost of evaluating the AABB + cost of evaluating the primitives
            return (
                area * C_traverse + area * Float32(node.triCount) * C_intersect
            )
        else:
            # Cost of evaluating this AABB + expected cost of traversing children
            var left_cost = self.sah_cost(node.leftFirst)
            var right_cost = self.sah_cost(node.leftFirst + 1)
            return area * C_traverse + left_cost + right_cost

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
    def get_indices_ptr(mut self) -> UnsafePointer[UInt32, MutAnyOrigin]:
        return self.prim_indices.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()

    @always_inline
    def intersect_tri(self, mut ray: Ray, prim_idx: UInt32):
        ref v0 = self.vertices[Int(prim_idx) * 3 + 0]
        ref v1 = self.vertices[Int(prim_idx) * 3 + 1]
        ref v2 = self.vertices[Int(prim_idx) * 3 + 2]
        var t = Float32(1e30)
        var u = Float32(0.0)
        var v = Float32(0.0)
        var w = Float32(0.0)
        var sign = Float32(0.0)

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
            ray.hit.t = t
            ray.hit.u = u
            ray.hit.v = v
            ray.hit.prim = prim_idx

    def traverse(self, mut ray: Ray):
        var stack = InlineArray[UInt32, 64](fill=0)
        var stack_ptr = 0
        var node_idx = UInt32(0)

        while True:
            var node = self.bvh_nodes[Int(node_idx)].copy()
            if node.is_leaf():
                for i in range(Int(node.triCount)):
                    var prim_idx = self.prim_indices[Int(node.leftFirst) + i]
                    self.intersect_tri(ray, prim_idx)

                if stack_ptr == 0:
                    break
                stack_ptr -= 1
                node_idx = stack[stack_ptr]
                continue

            var child1_idx = node.leftFirst
            var child2_idx = node.leftFirst + 1
            var child1 = self.bvh_nodes[Int(child1_idx)].copy()
            var child2 = self.bvh_nodes[Int(child2_idx)].copy()

            var dist1 = Float32(1e30)
            var dist2 = Float32(1e30)
            var hit1 = intersect_ray_aabb(
                ray.O, ray.rD, child1.aabbMin, child1.aabbMax, dist1
            )
            var hit2 = intersect_ray_aabb(
                ray.O, ray.rD, child2.aabbMin, child2.aabbMax, dist2
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

    def is_occluded(self, mut ray: Ray) -> Bool:
        var stack = InlineArray[UInt32, 64](fill=0)
        var stack_ptr = 0
        var node_idx = UInt32(0)

        while True:
            var node = self.bvh_nodes[Int(node_idx)].copy()
            if node.is_leaf():
                for i in range(Int(node.triCount)):
                    var prim_idx = self.prim_indices[Int(node.leftFirst) + i]
                    ref v0 = self.vertices[Int(prim_idx) * 3 + 0]
                    ref v1 = self.vertices[Int(prim_idx) * 3 + 1]
                    ref v2 = self.vertices[Int(prim_idx) * 3 + 2]
                    var t = Float32(1e30)
                    var u = Float32(0.0)
                    var v = Float32(0.0)
                    var w = Float32(0.0)
                    var sign = Float32(0.0)
                    if intersect_ray_tri_moller(
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
                    ):
                        if t < ray.hit.t:
                            return True
                if stack_ptr == 0:
                    break
                stack_ptr -= 1
                node_idx = stack[stack_ptr]
                continue

            var child1_idx = node.leftFirst
            var child2_idx = node.leftFirst + 1
            var child1 = self.bvh_nodes[Int(child1_idx)].copy()
            var child2 = self.bvh_nodes[Int(child2_idx)].copy()

            var dist1 = Float32(1e30)
            var dist2 = Float32(1e30)
            var hit1 = intersect_ray_aabb(
                ray.O, ray.rD, child1.aabbMin, child1.aabbMax, dist1
            )
            var hit2 = intersect_ray_aabb(
                ray.O, ray.rD, child2.aabbMin, child2.aabbMax, dist2
            )

            if hit1 and dist1 >= ray.hit.t:
                hit1 = False
            if hit2 and dist2 >= ray.hit.t:
                hit2 = False

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
        return False


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
        self.min_x = SIMD[DType.float32, Self.width](1e30)
        self.min_y = SIMD[DType.float32, Self.width](1e30)
        self.min_z = SIMD[DType.float32, Self.width](1e30)

        self.max_x = SIMD[DType.float32, Self.width](-1e30)
        self.max_y = SIMD[DType.float32, Self.width](-1e30)
        self.max_z = SIMD[DType.float32, Self.width](-1e30)

        self.data = SIMD[DType.uint32, Self.width](0)
        self.counts = SIMD[DType.uint32, Self.width](0xFFFFFFFF)


struct WideBVH[width: Int]:
    var nodes: List[WideBVHNode[Self.width]]
    var prim_indices: UnsafePointer[UInt32, MutAnyOrigin]
    var vertices: UnsafePointer[Vec3f32, MutAnyOrigin]

    def __init__(out self, mut binary_bvh: BVH):
        comptime assert Self.width in [4, 8, 16]
        self.nodes = List[WideBVHNode[Self.width]]()
        self.prim_indices = binary_bvh.get_indices_ptr()
        self.vertices = binary_bvh.vertices

        _ = self._collapse(binary_bvh, 0)

    def _collapse(mut self, binary_bvh: BVH, bin_node_idx: UInt32) -> UInt32:
        var wide_idx = UInt32(len(self.nodes))
        self.nodes.append(WideBVHNode[Self.width]())

        var pool = InlineArray[UInt32, Self.width](fill=0)
        pool[0] = bin_node_idx
        var pool_size = 1

        # Pull up children based on surface area heuristic until pool is full
        while pool_size < Self.width:
            var largest_area = Float32(-1.0)
            var largest_idx = -1

            for i in range(pool_size):
                var n = binary_bvh.bvh_nodes[Int(pool[i])].copy()
                if not n.is_leaf():
                    var area = n.surface_area()
                    if area > largest_area:
                        largest_area = area
                        largest_idx = i

            if largest_idx == -1:
                break

            var n = binary_bvh.bvh_nodes[Int(pool[largest_idx])].copy()
            pool[largest_idx] = n.leftFirst
            pool[pool_size] = n.leftFirst + 1
            pool_size += 1

        var wide_node = WideBVHNode[Self.width]()

        # Compile-time loop unrolling!
        comptime for i in range(Self.width):
            if i < pool_size:
                var n = binary_bvh.bvh_nodes[Int(pool[i])].copy()
                wide_node.min_x[i] = n.aabbMin.x()
                wide_node.min_y[i] = n.aabbMin.y()
                wide_node.min_z[i] = n.aabbMin.z()
                wide_node.max_x[i] = n.aabbMax.x()
                wide_node.max_y[i] = n.aabbMax.y()
                wide_node.max_z[i] = n.aabbMax.z()

                if n.is_leaf():
                    wide_node.data[i] = n.leftFirst
                    wide_node.counts[i] = n.triCount
                else:
                    var child_idx = self._collapse(binary_bvh, pool[i])
                    wide_node.data[i] = child_idx
                    wide_node.counts[i] = 0
            else:
                wide_node.counts[i] = 0xFFFFFFFF

        self.nodes[Int(wide_idx)] = wide_node^
        return wide_idx

    @always_inline
    def intersect_tri(self, mut ray: Ray, prim_idx: UInt32):
        ref v0 = self.vertices[Int(prim_idx) * 3 + 0]
        ref v1 = self.vertices[Int(prim_idx) * 3 + 1]
        ref v2 = self.vertices[Int(prim_idx) * 3 + 2]

        var t = Float32(1e30)
        var u = Float32(0.0)
        var v = Float32(0.0)
        var w = Float32(0.0)
        var sign = Float32(0.0)

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
            ray.hit.t = t
            ray.hit.u = u
            ray.hit.v = v
            ray.hit.prim = prim_idx

    def traverse(self, mut ray: Ray):
        var stack = InlineArray[UInt32, 32](fill=0)
        var stack_ptr = 0
        var node_idx = UInt32(0)

        while True:
            ref node = self.nodes[Int(node_idx)]

            # Massive parallel SIMD intersection
            var t1_x = (node.min_x - ray.O.x()) * ray.rD.x()
            var t2_x = (node.max_x - ray.O.x()) * ray.rD.x()
            var tmin_x = min(t1_x, t2_x)
            var tmax_x = max(t1_x, t2_x)

            var t1_y = (node.min_y - ray.O.y()) * ray.rD.y()
            var t2_y = (node.max_y - ray.O.y()) * ray.rD.y()
            var tmin_y = min(t1_y, t2_y)
            var tmax_y = max(t1_y, t2_y)

            var t1_z = (node.min_z - ray.O.z()) * ray.rD.z()
            var t2_z = (node.max_z - ray.O.z()) * ray.rD.z()
            var tmin_z = min(t1_z, t2_z)
            var tmax_z = max(t1_z, t2_z)

            var tmin = max(
                max(tmin_x, tmin_y),
                max(tmin_z, SIMD[DType.float32, Self.width](0.0)),
            )
            var tmax = min(
                min(tmax_x, tmax_y),
                min(tmax_z, SIMD[DType.float32, Self.width](ray.hit.t)),
            )

            var hit_mask = tmin.le(tmax)

            var hit_dists = InlineArray[Float32, Self.width](fill=1e30)
            var hit_indices = InlineArray[Int, Self.width](fill=0)
            var hit_count = 0

            comptime for i in range(Self.width):
                if hit_mask[i] and node.counts[i] != UInt32(0xFFFFFFFF):
                    hit_dists[hit_count] = tmin[i]
                    hit_indices[hit_count] = i
                    hit_count += 1

            # Insertion sort (descending) to pop the closest children first
            for i in range(1, hit_count):
                for j in range(i, 0, -1):
                    if hit_dists[j] > hit_dists[j - 1]:
                        var tmp_d = hit_dists[j]
                        hit_dists[j] = hit_dists[j - 1]
                        hit_dists[j - 1] = tmp_d

                        var tmp_i = hit_indices[j]
                        hit_indices[j] = hit_indices[j - 1]
                        hit_indices[j - 1] = tmp_i

            for i in range(hit_count):
                var child_idx = hit_indices[i]
                var count = node.counts[child_idx]

                if count == 0:
                    stack[stack_ptr] = node.data[child_idx]
                    stack_ptr += 1
                else:
                    for j in range(Int(count)):
                        var prim_idx = self.prim_indices[
                            Int(node.data[child_idx]) + j
                        ]
                        self.intersect_tri(ray, prim_idx)

            if stack_ptr == 0:
                break
            stack_ptr -= 1
            node_idx = stack[stack_ptr]

    def is_occluded(self, mut ray: Ray) -> Bool:
        var stack = InlineArray[UInt32, 64](fill=0)
        var stack_ptr = 0
        var node_idx = UInt32(0)

        while True:
            ref node = self.nodes[Int(node_idx)]

            var t1_x = (node.min_x - ray.O.x()) * ray.rD.x()
            var t2_x = (node.max_x - ray.O.x()) * ray.rD.x()
            var tmin_x = min(t1_x, t2_x)
            var tmax_x = max(t1_x, t2_x)

            var t1_y = (node.min_y - ray.O.y()) * ray.rD.y()
            var t2_y = (node.max_y - ray.O.y()) * ray.rD.y()
            var tmin_y = min(t1_y, t2_y)
            var tmax_y = max(t1_y, t2_y)

            var t1_z = (node.min_z - ray.O.z()) * ray.rD.z()
            var t2_z = (node.max_z - ray.O.z()) * ray.rD.z()
            var tmin_z = min(t1_z, t2_z)
            var tmax_z = max(t1_z, t2_z)

            var tmin = max(
                max(tmin_x, tmin_y),
                max(tmin_z, SIMD[DType.float32, Self.width](0.0)),
            )
            var tmax = min(
                min(tmax_x, tmax_y),
                min(tmax_z, SIMD[DType.float32, Self.width](ray.hit.t)),
            )
            var hit_mask = tmin.le(tmax)

            for i in range(Self.width):
                if hit_mask[i] and node.counts[i] != UInt32(0xFFFFFFFF):
                    if node.counts[i] == 0:
                        stack[stack_ptr] = node.data[i]
                        stack_ptr += 1
                    else:
                        for j in range(Int(node.counts[i])):
                            var prim_idx = self.prim_indices[
                                Int(node.data[i]) + j
                            ]
                            ref v0 = self.vertices[Int(prim_idx) * 3 + 0]
                            ref v1 = self.vertices[Int(prim_idx) * 3 + 1]
                            ref v2 = self.vertices[Int(prim_idx) * 3 + 2]
                            var t = Float32(1e30)
                            var u = Float32(0.0)
                            var v = Float32(0.0)
                            var w = Float32(0.0)
                            var sign = Float32(0.0)
                            if intersect_ray_tri_moller(
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
                            ):
                                if t < ray.hit.t:
                                    return True

            if stack_ptr == 0:
                break
            stack_ptr -= 1
            node_idx = stack[stack_ptr]

        return False


def main():
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
    bvh.build_quick()
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
