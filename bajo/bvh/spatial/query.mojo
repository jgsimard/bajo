"""Borrowed, stackless bounds queries shared by CPU and GPU kernels.

Nodes use preorder storage: six floats of bounds and two UInt32 words per
node (subtree escape index, primitive ID or EMPTY_LANE for an internal node).
Touching boundaries count as overlap. Invalid queries produce no candidates.
"""

from std.math import max, min
from std.utils.numerics import isfinite
from bajo.core import AABB, Frame, Point3f32, Rayf32, Vec3f32
from bajo.bvh.constants import EMPTY_LANE, f32_max


def ray_overlaps_bounds[
    frame: Frame
](
    ray: Rayf32[frame],
    bounds: AABB[frame],
    cutoff: Float32,
    radius: Float32 = 0.0,
) -> Bool:
    """Conservative expanded-box slab test; handles parallel and zero rays."""
    var lower = Float64(ray.t_min)
    var upper = Float64(min(ray.t_max, cutoff))
    if lower > upper:
        return False
    for axis in range(3):
        var origin = Float64(ray.o[axis])
        var direction = Float64(ray.d[axis])
        var lo = Float64(bounds._min[axis]) - Float64(radius)
        var hi = Float64(bounds._max[axis]) + Float64(radius)
        if direction == 0.0:
            if origin < lo or origin > hi:
                return False
        else:
            var a = (lo - origin) / direction
            var b = (hi - origin) / direction
            lower = max(lower, min(a, b))
            upper = min(upper, max(a, b))
            if lower > upper:
                return False
    return True


def sphere_overlaps_bounds[
    frame: Frame
](center: Point3f32[frame], radius: Float32, bounds: AABB[frame],) -> Bool:
    var distance2 = Float64(0.0)
    for axis in range(3):
        var p = Float64(center[axis])
        var delta = max(
            max(Float64(bounds._min[axis]) - p, p - Float64(bounds._max[axis])),
            0.0,
        )
        distance2 += delta * delta
    return distance2 <= Float64(radius) * Float64(radius)


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
        if self.valid:
            self.end = nodes[Int(root) * 2]

    def next(mut self, mut index: UInt32, max_dist: Float32 = f32_max) -> Bool:
        if not self.valid:
            return False
        if self.kind == 1 or self.kind == 3:
            if not isfinite(max_dist):
                return False
            self.cutoff = min(self.cutoff, max_dist)
        while self.cursor < self.end:
            var current = Int(self.cursor)
            var base = current * 6
            var box = AABB[Self.frame](
                Point3f32[Self.frame](
                    self.bounds[base],
                    self.bounds[base + 1],
                    self.bounds[base + 2],
                ),
                Point3f32[Self.frame](
                    self.bounds[base + 3],
                    self.bounds[base + 4],
                    self.bounds[base + 5],
                ),
            )
            var overlap: Bool
            if self.kind == 0:
                overlap = self.box.overlaps(box)[0]
            elif self.kind == 2:
                overlap = sphere_overlaps_bounds(
                    self.box._min, self.radius, box
                )
            else:
                overlap = ray_overlaps_bounds(
                    self.ray, box, self.cutoff, self.radius
                )
            var primitive = self.nodes[current * 2 + 1]
            if not overlap:
                self.cursor = self.nodes[current * 2]
            else:
                self.cursor += 1
                if primitive != EMPTY_LANE:
                    index = primitive
                    return True
        return False


@fieldwise_init
struct BoundsQueryView[origin: ImmOrigin, frame: Frame](Copyable):
    """A non-owning view usable in host code or GPU kernels."""

    var bounds: ImmSpan[Float32, Self.origin]
    var nodes: ImmSpan[UInt32, Self.origin]

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
