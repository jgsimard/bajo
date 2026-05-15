from std.utils.numerics import max_finite, min_finite

from bajo.core.intersect import intersect_ray_tri, intersect_ray_aabb
from bajo.core.vec import Vec3, Vec3f32
from bajo.core.bvh.cpu.binary_bvh import BinaryBvh
from bajo.core.bvh.types import Ray

comptime f32_max = max_finite[DType.float32]()
comptime f32_min = min_finite[DType.float32]()


@fieldwise_init
struct WideLeaf[width: Int](Copyable):
    var v0: Vec3[DType.float32, Self.width]
    var v1: Vec3[DType.float32, Self.width]
    var v2: Vec3[DType.float32, Self.width]
    var prim_indices: SIMD[DType.uint32, Self.width]

    def __init__(out self):
        self.v0 = Vec3[DType.float32, Self.width](0.0)
        self.v1 = Vec3[DType.float32, Self.width](0.0)
        self.v2 = Vec3[DType.float32, Self.width](0.0)
        self.prim_indices = 0xFFFFFFFF


@fieldwise_init
struct WideBvhNode[width: Int](Copyable):
    var _min: Vec3[DType.float32, Self.width]
    var _max: Vec3[DType.float32, Self.width]
    var data: SIMD[DType.uint32, Self.width]
    var counts: SIMD[DType.uint32, Self.width]

    @always_inline
    def __init__(out self):
        self._min = Vec3[DType.float32, Self.width](f32_max)
        self._max = Vec3[DType.float32, Self.width](f32_min)

        self.data = 0
        self.counts = 0xFFFFFFFF


struct WideBvh[width: Int](Copyable):
    var nodes: List[WideBvhNode[Self.width]]
    var leaves: List[WideLeaf[Self.width]]
    var vertices: UnsafePointer[Vec3f32, MutAnyOrigin]

    def __init__(out self, mut binary_bvh: BinaryBvh):
        self.nodes = List[WideBvhNode[Self.width]]()
        self.leaves = List[WideLeaf[Self.width]]()
        self.vertices = binary_bvh.vertices
        _ = self._collapse(binary_bvh, 0)

    def _collapse(mut self, binary_bvh: BinaryBvh, bin_idx: UInt32) -> UInt32:
        var wide_idx = UInt32(len(self.nodes))
        self.nodes.append(WideBvhNode[Self.width]())

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
            pool[best_i] = n.left_first
            pool[p_size] = n.left_first + 1
            p_size += 1

        var node = WideBvhNode[Self.width]()
        comptime for i in range(Self.width):
            if i < p_size:
                ref n = binary_bvh.bvh_nodes[Int(pool[i])]
                node._min.x[i] = n.aabb._min.x
                node._min.y[i] = n.aabb._min.y
                node._min.z[i] = n.aabb._min.z
                node._max.x[i] = n.aabb._max.x
                node._max.y[i] = n.aabb._max.y
                node._max.z[i] = n.aabb._max.z
                if n.is_leaf():
                    # PACK TRIANGLES INTO SIMD LEAF
                    var l_idx = UInt32(len(self.leaves))
                    var packed = WideLeaf[Self.width]()
                    for tri in range(min(Int(n.item_count), Self.width)):
                        var frag_idx = Int(
                            binary_bvh.prim_indices[Int(n.left_first) + tri]
                        )
                        var p_idx = Int(binary_bvh.fragments[frag_idx].prim_idx)
                        ref v0 = self.vertices[p_idx * 3]
                        ref v1 = self.vertices[p_idx * 3 + 1]
                        ref v2 = self.vertices[p_idx * 3 + 2]
                        packed.v0.x[tri] = v0.x
                        packed.v0.y[tri] = v0.y
                        packed.v0.z[tri] = v0.z
                        packed.v1.x[tri] = v1.x
                        packed.v1.y[tri] = v1.y
                        packed.v1.z[tri] = v1.z
                        packed.v2.x[tri] = v2.x
                        packed.v2.y[tri] = v2.y
                        packed.v2.z[tri] = v2.z
                        packed.prim_indices[tri] = UInt32(p_idx)
                    # Sentinel for empty lanes. Keep a separate valid-lane mask,
                    # because only poisoning v0x can still create NaNs in SIMD math.
                    for tri in range(Int(n.item_count), Self.width):
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
        var O = Vec3[DType.float32, Self.width](ray.O.x, ray.O.y, ray.O.z)
        var D = Vec3[DType.float32, Self.width](ray.D.x, ray.D.y, ray.D.z)
        var h = intersect_ray_tri[DType.float32, Self.width](
            O,
            D,
            leaf.v0,
            leaf.v1,
            leaf.v2,
            ray.hit.t,
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

        O = Vec3[DType.float32, Self.width](ray.O.x, ray.O.y, ray.O.z)
        rD = Vec3[DType.float32, Self.width](ray.rD.x, ray.rD.y, ray.rD.z)
        while True:
            ref node = self.nodes[Int(n_idx)]
            hit = intersect_ray_aabb(O, rD, node._min, node._max, ray.hit.t)
            var valid_lane = ~node.counts.eq(0xFFFFFFFF)
            var mask = hit.mask & valid_lane

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
