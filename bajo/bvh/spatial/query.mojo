"""Borrowed, stackless bounds queries shared by CPU and GPU kernels.

Nodes use preorder storage: six floats of bounds and two UInt32 words per
node (subtree escape index, primitive ID or EMPTY_LANE for an internal node).
Touching boundaries count as overlap. Invalid queries produce no candidates.
"""

from bajo.core.intersect import (
    ray_overlaps_bounds_cached,
    sphere_overlaps_bounds,
)

from std.math import min, iota
from std.sys.intrinsics import strided_load
from std.sys.info import is_gpu
from std.utils.numerics import isfinite
from bajo.core import (
    AABB,
    Frame,
    Point3f32,
    Rayf32,
    Vec3f32,
    Vec3,
    Point3,
    AxisAlignedBoundingBox,
)
from bajo.bvh.constants import EMPTY_LANE, f32_max
from .policy import BATCH_WIDTH


struct BoundsQuery[origin: ImmOrigin, frame: Frame](Copyable):
    """A resumable candidate iterator borrowing its BVH buffers.

    `next` preserves the output index after exhaustion. Ray/capsule cutoffs can
    only shrink: increasing a later cutoff never reopens rejected subtrees.
    Candidate order is unspecified. A capsule query is conservative broad phase.
    """

    var bounds: ImmSpan[Float32, Self.origin]
    var nodes: ImmSpan[UInt32, Self.origin]
    var cursor: UInt32
    var end: UInt32
    var kind: UInt8
    var box: AABB[Self.frame]
    var ray: Rayf32[Self.frame]
    var radius: Float32
    var cutoff: Float32
    var valid: Bool
    var inverse_direction: Vec3[.float64, Self.frame]

    @always_inline
    def __init__(
        out self,
        bounds: ImmSpan[Float32, Self.origin],
        nodes: ImmSpan[UInt32, Self.origin],
        root: UInt32,
        kind: UInt8,
        box: AABB[Self.frame],
        ray: Rayf32[Self.frame],
        radius: Float32,
        valid: Bool,
    ):
        self.bounds = bounds
        self.nodes = nodes
        self.cursor = root
        self.end = 0
        self.kind = kind
        self.box = box
        self.ray = ray
        self.radius = radius
        self.cutoff = ray.t_max
        self.valid = valid and Int(root) < len(nodes) // 2
        var inverse = Vec3[.float64, Self.frame](0.0)
        if self.valid and (kind == 1 or kind == 3):
            comptime for axis in range(3):
                var direction = Float64(ray.d[axis])
                inverse[axis] = 1.0 / direction if direction != 0.0 else 0.0
        self.inverse_direction = inverse
        if self.valid:
            self.end = nodes[Int(root) * 2]

    @always_inline
    def _begin(mut self, max_dist: Float32) -> Bool:
        if not self.valid:
            return False
        if self.kind == 1 or self.kind == 3:
            if not isfinite(max_dist):
                return False
            self.cutoff = min(self.cutoff, max_dist)
        return True

    @always_inline
    def _node_overlaps[
        width: SIMDLength, full: Bool = False
    ](self, current: Int) -> SIMD[.bool, width]:
        var active = SIMD[.bool, width](fill=True)
        comptime if not full:
            active = (SIMD[.int, width](current) + iota[.int, width]()).lt(
                Int(self.end)
            )
        var ptr = self.bounds.unsafe_ptr().unsafe_offset(current * 6)
        var lo = Point3[.float32, Self.frame, width](0.0)
        var hi = Point3[.float32, Self.frame, width](0.0)
        comptime for axis in range(3):
            lo[axis] = strided_load(ptr.unsafe_offset(axis), 6, active)
            hi[axis] = strided_load(ptr.unsafe_offset(axis + 3), 6, active)
        var box = AxisAlignedBoundingBox[.float32, Self.frame, width](lo, hi)
        if self.kind == 0:
            var overlap = active
            comptime for axis in range(3):
                overlap &= lo[axis].le(self.box._max[axis]) & hi[axis].ge(
                    self.box._min[axis]
                )
            return overlap
        elif self.kind == 2:
            return active & sphere_overlaps_bounds(
                self.box._min, self.radius, box
            )
        return ray_overlaps_bounds_cached(
            self.ray,
            box,
            self.cutoff,
            self.radius,
            self.inverse_direction,
            active,
        )

    @always_inline
    def _advance_node(mut self) -> UInt32:
        var current = Int(self.cursor)
        if not self._node_overlaps[1, True](current)[0]:
            self.cursor = self.nodes.unsafe_ptr().unsafe_load(current * 2)
            return EMPTY_LANE
        self.cursor += 1
        return self.nodes.unsafe_ptr().unsafe_load(current * 2 + 1)

    @always_inline
    def next(mut self, mut index: UInt32, max_dist: Float32 = f32_max) -> Bool:
        if not self._begin(max_dist):
            return False
        while self.cursor < self.end:
            var primitive = self._advance_node()
            if primitive != EMPTY_LANE:
                index = primitive
                return True
        return False

    @always_inline
    def next_batch[
        width: SIMDLength = BATCH_WIDTH
    ](mut self, output: MutSpan[UInt32, _], max_dist: Float32 = f32_max) -> Int:
        if len(output) == 0 or not self._begin(max_dist):
            return 0
        var count = 0
        while self.cursor < self.end and count < len(output):
            comptime if is_gpu() or width == 1:
                var primitive = self._advance_node()
                if primitive != EMPTY_LANE:
                    output[count] = primitive
                    count += 1
            else:
                var start = Int(self.cursor)
                var overlaps: SIMD[.bool, width]
                if start + width <= Int(self.end):
                    overlaps = self._node_overlaps[width, True](start)
                else:
                    overlaps = self._node_overlaps[width](start)
                while (
                    self.cursor < self.end
                    and Int(self.cursor) < start + width
                    and count < len(output)
                ):
                    var current = Int(self.cursor)
                    if overlaps[current - start]:
                        self.cursor += 1
                        var primitive = self.nodes.unsafe_ptr().unsafe_load(
                            current * 2 + 1
                        )
                        if primitive != EMPTY_LANE:
                            output[count] = primitive
                            count += 1
                    else:
                        self.cursor = self.nodes.unsafe_ptr().unsafe_load(
                            current * 2
                        )
        return count


@fieldwise_init
struct BoundsQueryView[origin: ImmOrigin, frame: Frame](Copyable):
    """A non-owning view usable in host code or GPU kernels."""

    var bounds: ImmSpan[Float32, Self.origin]
    var nodes: ImmSpan[UInt32, Self.origin]

    @always_inline
    def query_aabb(
        self, box: AABB[Self.frame], root: UInt32 = 0
    ) -> BoundsQuery[Self.origin, Self.frame]:
        return BoundsQuery(
            self.bounds,
            self.nodes,
            root,
            UInt8(0),
            box,
            Rayf32[Self.frame](
                Point3f32[Self.frame](0.0), Vec3f32[Self.frame](0.0)
            ),
            0.0,
            box.is_valid()[0],
        )

    @always_inline
    def query_ray(
        self, ray: Rayf32[Self.frame], root: UInt32 = 0
    ) -> BoundsQuery[Self.origin, Self.frame]:
        var valid = (
            ray.o.is_finite()[0]
            and ray.d.is_finite()[0]
            and isfinite(ray.t_min)
            and isfinite(ray.t_max)
            and ray.t_min <= ray.t_max
        )
        return BoundsQuery(
            self.bounds,
            self.nodes,
            root,
            UInt8(1),
            AABB[Self.frame].invalid(),
            ray,
            0.0,
            valid,
        )

    @always_inline
    def query_sphere(
        self, center: Point3f32[Self.frame], radius: Float32, root: UInt32 = 0
    ) -> BoundsQuery[Self.origin, Self.frame]:
        return BoundsQuery(
            self.bounds,
            self.nodes,
            root,
            UInt8(2),
            AABB[Self.frame].point(center),
            Rayf32[Self.frame](center, Vec3f32[Self.frame](0.0)),
            radius,
            center.is_finite()[0] and isfinite(radius) and radius >= 0.0,
        )

    @always_inline
    def query_capsule(
        self,
        start: Point3f32[Self.frame],
        direction: Vec3f32[Self.frame],
        radius: Float32,
        root: UInt32 = 0,
    ) -> BoundsQuery[Self.origin, Self.frame]:
        var ray = Rayf32[Self.frame](start, direction, 0.0, 1.0)
        return BoundsQuery(
            self.bounds,
            self.nodes,
            root,
            UInt8(3),
            AABB[Self.frame].invalid(),
            ray,
            radius,
            start.is_finite()[0]
            and direction.is_finite()[0]
            and isfinite(radius)
            and radius >= 0.0,
        )
