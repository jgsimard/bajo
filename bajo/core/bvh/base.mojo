from bajo.core.aabb import AABB
from bajo.core.primitives import Trianglef32
from bajo.core.vec import longest_axis, Vec3f32, dot


@fieldwise_init
struct Ray(Copyable, Writable):
    var origin: Vec3f32  # 16 -> 16
    var direction: Vec3f32  # 16 -> 32
    var inv_direction: Vec3f32  # 16 -> 48
    var time: Float32  #  4 -> 52
    var _pad: InlineArray[Float32, 3]  # 3*4=12-> 64

    fn __init__(
        out self, origin: Vec3f32, direction: Vec3f32, time: Float32 = 0.0
    ):
        self.origin = origin.copy()
        self.direction = direction.copy()
        self.inv_direction = 1.0 / self.direction
        self.time = time
        self._pad = InlineArray[Float32, 3](fill=0.0)

    fn at(self, t: Float32) -> Vec3f32:
        return self.origin + t * self.direction


struct HitRecord(Copyable):
    var p: Vec3f32  # 4*4 = 16 => 16
    var normal: Vec3f32  # 4*4 = 16 => 32
    var t: Float32  # 4 => 40
    var u: Float32  # 4 => 44
    var v: Float32  # 4 => 48
    var front_face: Bool  # 1 -> 4 => 52

    fn __init__(
        out self,
        p: Vec3f32,
        normal: Vec3f32,
        material_id: Int,
        t: Float32,
        r: Ray,
    ):
        self.p = p.copy()
        self.t = t
        self.u = 0.0
        self.v = 0.0
        self.front_face = dot(r.direction, normal) < 0
        self.normal = normal * Float32(1.0 if self.front_face else -1.0)


@fieldwise_init
struct BvhNode(Copyable):
    """Bounding Volume Hierarchy Node."""

    var bbox: AABB
    var left_idx: Int
    """Index in 'nodes' list. -1 if leaf."""
    var right_idx: Int
    """Index in 'nodes' list. -1 if leaf."""
    var object_idx: Int
    """Index in 'objects' list. -1 if internal node."""


@fieldwise_init
struct Bvh:
    """Bounding Volume Hierarchy."""

    var nodes: List[BvhNode]
    var objects: List[Trianglef32]
    var root_idx: Int

    fn __init__(
        out self,
        var objects: List[Trianglef32],
    ):
        self.nodes = List[BvhNode]()
        self.objects = objects^
        self.root_idx = -1

        if len(self.objects) > 0:
            self.root_idx = self._build(0, len(self.objects))

    fn _build(mut self, start: Int, end: Int) -> Int:
        var span_len = end - start

        # leaf node
        if span_len == 1:
            box = self.objects[start].bounds()
            node = BvhNode(box^, -1, -1, start)
            self.nodes.append(node^)
            return len(self.nodes) - 1

        aabb = self.objects[start].bounds()
        for i in range(start + 1, end):
            aabb.grow(self.objects[i].bounds())
        axis = longest_axis(aabb.edges())

        # internal node
        fn cmp_fn(a: Trianglef32, b: Trianglef32) capturing -> Bool:
            var box_a = a.bounds()
            var box_b = b.bounds()
            return box_a._min[axis] < box_b._min[axis]

        # SIMD version
        sort[cmp_fn=cmp_fn, stable=True](self.objects[start : start + span_len])
        # # not SIMD version
        # sort[cmp_fn=cmp_fn](self.objects[start : start + span_len])

        mid = start + span_len // 2

        # Recursively build children
        var left_idx = self._build(start, mid)
        var right_idx = self._build(mid, end)

        # Compute combined bounding box
        ref box_l = self.nodes[left_idx].bbox
        ref box_r = self.nodes[right_idx].bbox
        var combined_box = AABB(box_l, box_r)

        node = BvhNode(combined_box^, left_idx, right_idx, -1)
        self.nodes.append(node^)
        return len(self.nodes) - 1

    fn hit(
        self, ray: Ray, ray_t_min: Float32, ray_t_max: Float32
    ) -> Optional[HitRecord]:
        if self.root_idx == -1:
            return None

        var closest_so_far = ray_t_max
        var hit_anything: Optional[HitRecord] = None

        var node_stack = InlineArray[Int, 32](fill=0)
        var stack_ptr = 0

        # Push root
        node_stack[stack_ptr] = self.root_idx
        stack_ptr += 1

        while stack_ptr > 0:
            # Pop
            stack_ptr -= 1
            var node_idx = node_stack[stack_ptr]
            ref node = self.nodes[node_idx]

            # check AABB
            if not node.bbox.ray_intersects(
                ray.origin, ray.inv_direction, ray_t_min, closest_so_far
            ):
                continue

            # leaf node = object
            if node.object_idx != -1:
                ref obj = self.objects[node.object_idx]
                var hit_res: Optional[HitRecord]

                if obj.isa[Sphere]():
                    hit_res = obj[Sphere].hit(
                        ray, Interval(ray_t.min, closest_so_far)
                    )
                else:
                    print("ooooooops")
                    abort()

                if hit_res:
                    ref rec = hit_res.value()
                    closest_so_far = rec.t
                    hit_anything = hit_res

            # internal node = check children
            else:
                # push children to stack
                node_stack[stack_ptr] = node.right_idx
                stack_ptr += 1
                node_stack[stack_ptr] = node.left_idx
                stack_ptr += 1

        return hit_anything


def main() raises:
    from bajo.core.utils import print_size_of

    print_size_of[AABB]()
    print_size_of[BvhNode]()
