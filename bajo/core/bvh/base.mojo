from bajo.core.aabb import AABB
from bajo.core.primitives import Trianglef32
from bajo.core.vec import longest_axis


@fieldwise_init
struct BVHNode(Copyable):
    """Bounding Volume Hierarchy Node."""

    var bbox: AABB
    var left_idx: Int
    """Index in 'nodes' list. -1 if leaf."""
    var right_idx: Int
    """Index in 'nodes' list. -1 if leaf."""
    var object_idx: Int
    """Index in 'objects' list. -1 if internal node."""


@fieldwise_init
struct BVH:
    """Bounding Volume Hierarchy."""

    var nodes: List[BVHNode]
    var objects: List[Trianglef32]
    var root_idx: Int

    fn __init__(
        out self,
        var objects: List[Trianglef32],
    ):
        self.nodes = List[BVHNode]()
        self.objects = objects^
        self.root_idx = -1

        if len(self.objects) > 0:
            self.root_idx = self._build(0, len(self.objects))

    fn _build(mut self, start: Int, end: Int) -> Int:
        var span_len = end - start

        # leaf node
        if span_len == 1:
            box = self.objects[start].bounds()
            node = BVHNode(box^, -1, -1, start)
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

        node = BVHNode(combined_box^, left_idx, right_idx, -1)
        self.nodes.append(node^)
        return len(self.nodes) - 1
