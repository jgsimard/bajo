"""Borrowed, stackless bounds queries shared by CPU and GPU kernels.

Nodes use preorder storage: six floats of bounds and two UInt32 words per
node (subtree escape index, primitive ID or EMPTY_LANE for an internal node).
Touching boundaries count as overlap. Invalid queries produce no candidates.
"""

from std.math import max, min, abs, iota
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


@always_inline
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
    comptime for axis in range(3):
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


@always_inline
def _gpu_ray_bounds_interval[
    frame: Frame
](
    ray: Rayf32[frame],
    bounds: AABB[frame],
    cutoff: Float32,
    radius: Float32,
    inverse: Vec3[.float64, frame],
) -> Tuple[Int, Float32]:
    """Return 0/1 for separated intervals, -1 when Float64 is needed.

    Bound both interval endpoints by the largest per-axis error. Strict
    comparisons use twice that error; uncertain intervals retain Float64.
    The generous margin
    covers Float32 subtraction, reciprocal conversion and multiplication;
    overflow, underflow and touching intervals use the original exact test.
    """
    var lower = ray.t_min
    var upper = min(ray.t_max, cutoff)
    var interval_error = Float32(0.0)
    if lower > upper:
        return (0, 0.0)
    comptime for axis in range(3):
        var origin = ray.o[axis]
        var lo = bounds._min[axis] - radius
        var hi = bounds._max[axis] + radius
        var scale = max(abs(bounds._min[axis]), abs(bounds._max[axis]))
        var error = (scale + abs(radius) + abs(origin)) * 0.000002
        if not isfinite(error) or error < 1e-30:
            return (-1, 0.0)
        if ray.d[axis] == 0.0:
            if origin < lo - error or origin > hi + error:
                return (0, 0.0)
            if origin <= lo + error or origin >= hi - error:
                return (-1, 0.0)
        else:
            var direction = abs(ray.d[axis])
            if direction < 1e-30 or direction > 1e30:
                return (-1, 0.0)
            var inv = Float32(inverse[axis])
            var a = (lo - origin) * inv
            var b = (hi - origin) * inv
            error = error * abs(inv) + max(abs(a), abs(b)) * 0.000002
            if not isfinite(error) or error < 1e-30:
                return (-1, 0.0)
            interval_error = max(interval_error, error)
            lower = max(lower, min(a, b))
            upper = min(upper, max(a, b))
            if lower > upper + interval_error * 2.0:
                return (0, 0.0)
    if lower < upper - interval_error * 2.0:
        return (1, lower - interval_error)
    return (-1, 0.0)


@always_inline
def _ray_overlaps_bounds_cached_impl[
    frame: Frame, width: SIMDLength = 1
](
    ray: Rayf32[frame],
    bounds: AxisAlignedBoundingBox[.float32, frame, width],
    cutoff: Float32,
    radius: Float32,
    inverse: Vec3[.float64, frame],
    active: SIMD[.bool, width] = SIMD[.bool, width](fill=True),
) -> SIMD[.bool, width]:
    var lower = SIMD[.float64, width](Float64(ray.t_min))
    var upper = SIMD[.float64, width](Float64(min(ray.t_max, cutoff)))
    var valid = active & lower.le(upper)
    comptime if width == 1:
        if not valid[0]:
            return valid
    comptime for axis in range(3):
        var origin = SIMD[.float64, width](Float64(ray.o[axis]))
        var lo = bounds._min[axis].cast[.float64]() - Float64(radius)
        var hi = bounds._max[axis].cast[.float64]() + Float64(radius)
        if ray.d[axis] == 0.0:
            valid &= origin.ge(lo) & origin.le(hi)
            comptime if width == 1:
                if not valid[0]:
                    return valid
        else:
            var a = (lo - origin) * inverse[axis]
            var b = (hi - origin) * inverse[axis]
            lower = max(lower, min(a, b))
            upper = min(upper, max(a, b))
            comptime if width == 1:
                if lower[0] > upper[0]:
                    if (
                        lower[0] - upper[0]
                        <= max(abs(lower[0]), abs(upper[0])) * 1e-14
                    ):
                        var box = AABB[frame](
                            Point3f32[frame](
                                bounds._min.x[0],
                                bounds._min.y[0],
                                bounds._min.z[0],
                            ),
                            Point3f32[frame](
                                bounds._max.x[0],
                                bounds._max.y[0],
                                bounds._max.z[0],
                            ),
                        )
                        return SIMD[.bool, width](
                            fill=ray_overlaps_bounds(ray, box, cutoff, radius)
                        )
                    return SIMD[.bool, width](fill=False)
    var result = valid & lower.le(upper)
    var boundary = abs(upper - lower).le(max(abs(lower), abs(upper)) * 1e-14)
    comptime for lane in range(width):
        if active[lane] and boundary[lane]:
            var box = AABB[frame](
                Point3f32[frame](
                    bounds._min.x[lane],
                    bounds._min.y[lane],
                    bounds._min.z[lane],
                ),
                Point3f32[frame](
                    bounds._max.x[lane],
                    bounds._max.y[lane],
                    bounds._max.z[lane],
                ),
            )
            result[lane] = ray_overlaps_bounds(ray, box, cutoff, radius)
    return result


@no_inline
def _ray_overlaps_bounds_cached_exact[
    frame: Frame, width: SIMDLength = 1
](
    ray: Rayf32[frame],
    bounds: AxisAlignedBoundingBox[.float32, frame, width],
    cutoff: Float32,
    radius: Float32,
    inverse: Vec3[.float64, frame],
    active: SIMD[.bool, width] = SIMD[.bool, width](fill=True),
) -> SIMD[.bool, width]:
    # Keep uncertain GPU slabs in a separate function; the common Float32
    # traversal path need not inline the full Float64 boundary calculation.
    return _ray_overlaps_bounds_cached_impl(
        ray, bounds, cutoff, radius, inverse, active
    )


@always_inline
def ray_overlaps_bounds_cached[
    frame: Frame, width: SIMDLength = 1
](
    ray: Rayf32[frame],
    bounds: AxisAlignedBoundingBox[.float32, frame, width],
    cutoff: Float32,
    radius: Float32,
    inverse: Vec3[.float64, frame],
    active: SIMD[.bool, width] = SIMD[.bool, width](fill=True),
) -> SIMD[.bool, width]:
    """Cached scalar/SIMD slabs with division fallback near interval boundaries.
    """
    comptime if is_gpu() and width == 1:
        if not active[0]:
            return active
        var box = AABB[frame](
            Point3f32[frame](
                bounds._min.x[0], bounds._min.y[0], bounds._min.z[0]
            ),
            Point3f32[frame](
                bounds._max.x[0], bounds._max.y[0], bounds._max.z[0]
            ),
        )
        var filtered = _gpu_ray_bounds_interval(
            ray, box, cutoff, radius, inverse
        )[0]
        if filtered >= 0:
            return SIMD[.bool, width](fill=filtered == 1)
        return _ray_overlaps_bounds_cached_exact(
            ray, bounds, cutoff, radius, inverse, active
        )
    return _ray_overlaps_bounds_cached_impl(
        ray, bounds, cutoff, radius, inverse, active
    )


@no_inline
def _sphere_overlaps_exact[
    frame: Frame, width: SIMDLength
](
    center: Point3f32[frame],
    radius: Float32,
    bounds: AxisAlignedBoundingBox[.float32, frame, width],
) -> SIMD[.bool, width]:
    # Preserve touching and extreme-scale results without inlining this
    # calculation into every node visited by the GPU Float32 filter.
    return bounds.distance2(center).le(Float64(radius) * Float64(radius))


@always_inline
def sphere_overlaps_bounds[
    frame: Frame, width: SIMDLength = 1
](
    center: Point3f32[frame],
    radius: Float32,
    bounds: AxisAlignedBoundingBox[.float32, frame, width],
) -> SIMD[.bool, width]:
    comptime if is_gpu() and width == 1:
        var distance2 = Float32(0.0)
        comptime for axis in range(3):
            var delta = max(
                max(
                    bounds._min[axis][0] - center[axis],
                    center[axis] - bounds._max[axis][0],
                ),
                Float32(0.0),
            )
            distance2 += delta * delta
        var radius2 = radius * radius
        var scale = max(distance2, radius2)
        # Normal finite arithmetic away from tangency is unambiguous. Retain
        # Float64 for tiny/overflowing distances and touching spheres.
        if isfinite(scale) and scale > 1e-30:
            if abs(distance2 - radius2) > scale * 0.000002:
                return SIMD[.bool, width](fill=distance2 < radius2)
        return _sphere_overlaps_exact(center, radius, bounds)
    return bounds.distance2(center).le(Float64(radius) * Float64(radius))


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
