from std.utils.numerics import max_finite, min_finite

from bajo.core.intersect import intersect_ray_aabb, intersect_ray_tri

comptime f32_max = max_finite[DType.float32]()
comptime f32_min = min_finite[DType.float32]()


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
        var h = intersect_ray_tri[DType.float32, Self.width](
            SIMD[DType.float32, Self.width](ray.O.x()),
            SIMD[DType.float32, Self.width](ray.O.y()),
            SIMD[DType.float32, Self.width](ray.O.z()),
            SIMD[DType.float32, Self.width](ray.D.x()),
            SIMD[DType.float32, Self.width](ray.D.y()),
            SIMD[DType.float32, Self.width](ray.D.z()),
            leaf.v0x,
            leaf.v0y,
            leaf.v0z,
            leaf.v1x,
            leaf.v1y,
            leaf.v1z,
            leaf.v2x,
            leaf.v2y,
            leaf.v2z,
            SIMD[DType.float32, Self.width](ray.hit.t),
        )

        var valid_lane = ~leaf.prim_indices.eq(0xFFFFFFFF)
        var hit_mask = h.mask & valid_lane

        if hit_mask.reduce_or():
            comptime if is_occlusion:
                return True
            else:
                var min_t = hit_mask.select(h.t, f32_max).reduce_min()

                comptime for i in range(Self.width):
                    if hit_mask[i] and h.t[i] == min_t:
                        ray.hit.t = min_t
                        ray.hit.u = h.u[i]
                        ray.hit.v = h.v[i]
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
