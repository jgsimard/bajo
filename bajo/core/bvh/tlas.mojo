from std.utils.numerics import max_finite, min_finite

from bajo.core.bvh.types import BVHNode
from bajo.core.intersect import intersect_ray_aabb
from bajo.core.mat import Mat44f32, transform_point, transform_vector, inverse
from bajo.core.vec import Vec3f32, vmin, vmax, longest_axis


comptime f32_max = max_finite[DType.float32]()
comptime f32_min = min_finite[DType.float32]()


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
