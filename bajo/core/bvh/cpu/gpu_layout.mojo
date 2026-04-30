from std.utils.numerics import max_finite, min_finite

from bajo.core.aabb import AABB
from bajo.core.intersect import intersect_ray_aabb, intersect_ray_tri

comptime f32_max = max_finite[DType.float32]()
comptime f32_min = min_finite[DType.float32]()


struct BVHGPUNode(Copyable):
    """TinyBVH-style Aila-Laine GPU node.

    This mirrors TinyBVH's BVH_GPU node semantics:
    - If tri_count > 0, this node is a leaf and first_tri points into prim_indices.
    - If tri_count == 0, this node is internal:
        lmin/lmax bound the left child, and left is the left child node index.
        rmin/rmax bound the right child, and right is the right child node index.
    """

    var lmin: Vec3f32
    var left: UInt32
    var lmax: Vec3f32
    var right: UInt32

    var rmin: Vec3f32
    var tri_count: UInt32
    var rmax: Vec3f32
    var first_tri: UInt32

    @always_inline
    def __init__(out self):
        self.lmin = Vec3f32(f32_max)
        self.left = 0
        self.lmax = Vec3f32(f32_min)
        self.right = 0

        self.rmin = Vec3f32(f32_max)
        self.tri_count = 0
        self.rmax = Vec3f32(f32_min)
        self.first_tri = 0

    @always_inline
    def is_leaf(self) -> Bool:
        return self.tri_count > 0


struct BvhGpuLayout(Copyable):
    """TinyBVH-compatible GPU layout with CPU reference traversal.

    This layout follows TinyBVH's BVH_GPU / Aila-Laine layout more closely than
    the earlier child-count encoding. Leaves are real nodes (`tri_count > 0`),
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
        # point into this array using the source BVH's left_first/count ranges.
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
            out.tri_count = bnode.tri_count
            out.first_tri = bnode.left_first
            # Leaf bounds are not needed by traversal once the leaf is entered,
            # but storing them makes debug inspection easier and keeps the node
            # self-describing.
            out.lmin = bnode.aabb._min.copy()
            out.lmax = bnode.aabb._max.copy()
            out.rmin = bnode.aabb._min.copy()
            out.rmax = bnode.aabb._max.copy()
        else:
            var left_binary_idx = bnode.left_first
            var right_binary_idx = bnode.left_first + 1

            ref left = binary_bvh.bvh_nodes[Int(left_binary_idx)]
            ref right = binary_bvh.bvh_nodes[Int(right_binary_idx)]

            out.lmin = left.aabb._min.copy()
            out.lmax = left.aabb._max.copy()
            out.rmin = right.aabb._min.copy()
            out.rmax = right.aabb._max.copy()
            out.tri_count = 0
            out.first_tri = 0

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
                    ray, node.first_tri, node.tri_count
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
