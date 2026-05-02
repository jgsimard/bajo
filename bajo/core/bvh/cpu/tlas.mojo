from std.memory import UnsafePointer
from std.utils.numerics import max_finite, min_finite

from bajo.core.aabb import AABB
from bajo.core.bvh.cpu.binary_bvh import BinaryBvh
from bajo.core.bvh.types import BvhNode, Ray
from bajo.core.intersect import intersect_ray_aabb
from bajo.core.mat import Mat44f32, transform_point, transform_vector
from bajo.core.vec import Vec3f32, longest_axis, vmin, vmax


comptime f32_max = max_finite[DType.float32]()
comptime f32_min = min_finite[DType.float32]()
comptime TLAS_LEAF_SIZE = UInt32(2)


@fieldwise_init
struct BvhInstance(Copyable):
    """Instance of a BLAS in world space.

    Phase A intentionally keeps this small:
    - `transform` maps BLAS-local points/vectors to world space.
    - `inv_transform` maps world-space rays to BLAS-local space.
    - `bounds_min/max` store the transformed world-space BLAS root AABB.
    - `blas_idx` indexes the BLAS array passed to `Tlas.traverse`.

    Traversal assumes rigid transforms or uniform scale, so the local-space hit
    `t` can be compared directly with the world-space ray `t`. General affine
    transforms need explicit world-space distance reconstruction.
    """

    var transform: Mat44f32
    var inv_transform: Mat44f32
    var bounds_min: Vec3f32
    var bounds_max: Vec3f32
    var blas_idx: UInt32

    @always_inline
    def __init__(out self):
        self.transform = Mat44f32.identity()
        self.inv_transform = Mat44f32.identity()
        self.bounds_min = Vec3f32(f32_max)
        self.bounds_max = Vec3f32(f32_min)
        self.blas_idx = 0

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

        # Transform all 8 local BLAS AABB corners and re-bound them in world
        # space. This is correct for arbitrary affine transforms, even though
        # Phase A traversal only relies on rigid/uniform-scale t semantics.
        var corners = InlineArray[Vec3f32, 8](fill=Vec3f32(0.0))
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
            var p = transform_point(transform, corners[i])
            w_min = vmin(w_min, p)
            w_max = vmax(w_max, p)

        self.bounds_min = w_min^
        self.bounds_max = w_max^

    @staticmethod
    def from_blas(
        transform: Mat44f32,
        inv_transform: Mat44f32,
        blas_idx: UInt32,
        blas: BinaryBvh,
    ) -> BvhInstance:
        ref root = blas.bvh_nodes[0]
        return BvhInstance(
            transform,
            inv_transform,
            blas_idx,
            root.aabb._min,
            root.aabb._max,
        )

    @always_inline
    def centroid_axis(self, axis: Int) -> Float32:
        return (self.bounds_min[axis] + self.bounds_max[axis]) * 0.5


struct Tlas(Copyable):
    """CPU top-level acceleration structure over BLAS instances.

    It owns its instance list and uses the same `BvhNode` layout as `BinaryBvh`: leaves
    store a range in `inst_indices`; internal nodes store two children at
    `left_first` and `left_first + 1`.
    """

    var tlas_nodes: List[BvhNode]
    var inst_indices: List[UInt32]
    var instances: List[BvhInstance]
    var inst_count: UInt32
    var nodes_used: UInt32

    def __init__(out self):
        self.tlas_nodes = List[BvhNode]()
        self.inst_indices = List[UInt32]()
        self.instances = List[BvhInstance]()
        self.inst_count = 0
        self.nodes_used = 0
        self._reset_build_state()

    def __init__(out self, instances: List[BvhInstance]):
        self.tlas_nodes = List[BvhNode]()
        self.inst_indices = List[UInt32]()
        self.instances = List[BvhInstance](capacity=len(instances))
        self.inst_count = 0
        self.nodes_used = 0

        for i in range(len(instances)):
            self.instances.append(instances[i].copy())

        self._reset_build_state()

    def add_instance(mut self, instance: BvhInstance):
        self.instances.append(instance.copy())
        self._reset_build_state()

    def _reset_build_state(mut self):
        self.inst_count = UInt32(len(self.instances))
        if self.inst_count > 0:
            self.nodes_used = 1
        else:
            self.nodes_used = 0

        var max_nodes = 1
        if self.inst_count > 0:
            max_nodes = Int(self.inst_count * 2 - 1)

        self.tlas_nodes = List[BvhNode](capacity=max_nodes)
        for _ in range(max_nodes):
            self.tlas_nodes.append(BvhNode())

        self.inst_indices = List[UInt32](capacity=Int(self.inst_count))
        for i in range(Int(self.inst_count)):
            self.inst_indices.append(UInt32(i))

        if self.inst_count > 0:
            ref root = self.tlas_nodes[0]
            root.left_first = 0
            root.tri_count = self.inst_count
            self.update_node_bounds(0)

    @always_inline
    def update_node_bounds(mut self, node_idx: UInt32):
        ref node = self.tlas_nodes[Int(node_idx)]
        node.aabb = AABB.invalid()

        var first = Int(node.left_first)
        for i in range(Int(node.tri_count)):
            var inst_idx = Int(self.inst_indices[first + i])
            ref inst = self.instances[inst_idx]
            node.aabb._min = vmin(node.aabb._min, inst.bounds_min)
            node.aabb._max = vmax(node.aabb._max, inst.bounds_max)

    def build(mut self):
        """Build a simple median TLAS over instance bounds.

        Instance counts are usually much smaller than primitive counts, so Phase
        A deliberately uses a small and deterministic median builder instead of
        a SAH builder.
        """
        self._reset_build_state()
        if self.inst_count > 0:
            self._subdivide(0)

    def _subdivide(mut self, node_idx: UInt32):
        ref node = self.tlas_nodes[Int(node_idx)]

        if node.tri_count <= TLAS_LEAF_SIZE:
            return

        var extent = node.aabb._max - node.aabb._min
        var axis = longest_axis(extent)
        var split_pos = node.aabb._min[axis] + extent[axis] * 0.5

        var i = Int(node.left_first)
        var j = i + Int(node.tri_count) - 1

        while i <= j:
            var inst_idx = Int(self.inst_indices[i])
            ref inst = self.instances[inst_idx]
            var centroid = inst.centroid_axis(axis)

            if centroid < split_pos:
                i += 1
            else:
                var tmp = self.inst_indices[i]
                self.inst_indices[i] = self.inst_indices[j]
                self.inst_indices[j] = tmp
                j -= 1

        var left_count = UInt32(i - Int(node.left_first))
        if left_count == 0 or left_count == node.tri_count:
            left_count = node.tri_count // 2
            i = Int(node.left_first) + Int(left_count)

        var left_child_idx = self.nodes_used
        self.nodes_used += 2

        ref left_child = self.tlas_nodes[Int(left_child_idx)]
        ref right_child = self.tlas_nodes[Int(left_child_idx + 1)]

        left_child.left_first = node.left_first
        left_child.tri_count = left_count
        right_child.left_first = UInt32(i)
        right_child.tri_count = node.tri_count - left_count

        node.left_first = left_child_idx
        node.tri_count = 0

        self.update_node_bounds(left_child_idx)
        self.update_node_bounds(left_child_idx + 1)

        self._subdivide(left_child_idx)
        self._subdivide(left_child_idx + 1)

    def traverse(
        self,
        mut ray: Ray,
        blases: UnsafePointer[BinaryBvh, MutAnyOrigin],
    ):
        """Traverse TLAS in world space, then BLASes in local space.

        On hit:
        - `ray.hit.prim` is the primitive id inside the hit BLAS.
        - `ray.hit.inst` is the instance id inside this TLAS.
        """
        if self.inst_count == 0:
            return

        var stack = InlineArray[UInt32, 64](fill=0)
        var stack_ptr = 0
        var node_idx = UInt32(0)

        while True:
            ref node = self.tlas_nodes[Int(node_idx)]

            if node.is_leaf():
                for i in range(Int(node.tri_count)):
                    var inst_idx = self.inst_indices[Int(node.left_first) + i]
                    ref inst = self.instances[Int(inst_idx)]

                    var local_origin = transform_point(
                        inst.inv_transform, ray.O
                    )
                    var local_dir = transform_vector(inst.inv_transform, ray.D)
                    var local_ray = Ray(local_origin, local_dir, ray.hit.t)

                    blases[Int(inst.blas_idx)].traverse(local_ray)

                    if local_ray.hit.t < ray.hit.t:
                        ray.hit.t = local_ray.hit.t
                        ray.hit.u = local_ray.hit.u
                        ray.hit.v = local_ray.hit.v
                        ray.hit.prim = local_ray.hit.prim
                        ray.hit.inst = inst_idx

                if stack_ptr == 0:
                    break
                stack_ptr -= 1
                node_idx = stack[stack_ptr]
                continue

            var child1_idx = node.left_first
            var child2_idx = node.left_first + 1
            ref child1 = self.tlas_nodes[Int(child1_idx)]
            ref child2 = self.tlas_nodes[Int(child2_idx)]

            var dist1 = Float32(f32_max)
            var dist2 = Float32(f32_max)
            var hit1 = intersect_ray_aabb(
                ray.O,
                ray.rD,
                child1.aabb._min,
                child1.aabb._max,
                dist1,
            )
            var hit2 = intersect_ray_aabb(
                ray.O,
                ray.rD,
                child2.aabb._min,
                child2.aabb._max,
                dist2,
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
                var near = child1_idx
                var far = child2_idx
                if dist1 > dist2:
                    near = child2_idx
                    far = child1_idx

                debug_assert["safe"](
                    stack_ptr < 64, "TLAS traversal stack overflow"
                )
                stack[stack_ptr] = far
                stack_ptr += 1
                node_idx = near
