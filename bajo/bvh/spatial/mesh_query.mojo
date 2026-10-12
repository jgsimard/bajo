"""Borrowed indexed mesh queries executable on CPU and GPU.

Barycentrics are weights of vertices A and B; C has weight 1-u-v. Ray intervals
are inclusive. Counts report intersected faces, including multiple faces on a
shared edge. Candidate iteration order is unspecified; nearest/furthest ties
choose the lowest original face ID. Sign classification assumes an oriented,
closed surface; open/nonmanifold surfaces do not define a universal inside.
"""

from std.sys.intrinsics import strided_load
from std.math import abs, sqrt, max, min, pi
from std.memory import stack_allocation, bitcast
from std.sys.info import is_gpu
from std.utils.numerics import isfinite
from bajo.core import AABB, Frame, Point3f32, Rayf32, Vec3f32, Vec3, cross, dot
from bajo.core import Point3, AxisAlignedBoundingBox
from bajo.bvh.constants import EMPTY_LANE, f32_max
from .query import BoundsQueryView, BoundsQuery, _gpu_ray_bounds_interval
from .policy import BATCH_WIDTH, RAY_COUNT_WIDTH
from .winding import WindingView, DeviceWindingView
from .ordered_traversal import traverse_ordered
from .gpu_geometry import (
    ray_triangle_filter,
    ray_triangle_interval,
    project_triangle_filtered,
    project_triangle_filtered_with_exactness,
    triangle_normal_f32,
    normal_sign_filter,
)
from bajo.core.intersect import (
    project_triangle,
    TriangleProjection,
    intersect_triangle,
    triangle_normal,
    solid_angle,
    vector64,
    angle_atan2,
)

comptime INV_4_PI = 1.0 / (4.0 * pi)


@no_inline
def _furthest_corner_exact[
    frame: Frame
](
    point: Point3f32[frame],
    a: Point3f32[frame],
    b: Point3f32[frame],
    c: Point3f32[frame],
) -> Tuple[Float64, Int]:
    var d0 = vector64(a) - vector64(point)
    var d1 = vector64(b) - vector64(point)
    var d2 = vector64(c) - vector64(point)
    var distance0 = dot(d0, d0)
    var distance1 = dot(d1, d1)
    var distance2 = dot(d2, d2)
    var first = distance1 > distance0
    var selected_distance = distance1 if first else distance0
    var second = distance2 > selected_distance
    selected_distance = distance2 if second else selected_distance
    return selected_distance, (2 if second else (1 if first else 0))


@fieldwise_init
struct MeshPointResult[frame: Frame](TrivialRegisterPassable):
    var result: Bool
    var face: UInt32
    var u: Float32
    var v: Float32
    var distance: Float64
    var point: Point3f32[Self.frame]
    var sign: Float32

    @staticmethod
    def miss() -> Self:
        return Self(
            False,
            EMPTY_LANE,
            0.0,
            0.0,
            Float64(f32_max),
            Point3f32[Self.frame](0.0),
            0.0,
        )


@fieldwise_init
struct MeshRayResult[frame: Frame](TrivialRegisterPassable):
    var result: Bool
    var face: UInt32
    var u: Float32
    var v: Float32
    var t: Float64
    var normal: Vec3f32[Self.frame]
    var front_face: Bool

    @staticmethod
    def miss() -> Self:
        return Self(
            False,
            EMPTY_LANE,
            0.0,
            0.0,
            Float64(f32_max),
            Vec3f32[Self.frame](0.0),
            False,
        )


def load_mesh_point[
    frame: Frame
](points: ImmSpan[Float32, _], index: UInt32) -> Point3f32[frame]:
    var base = Int(index) * 3
    return Point3f32[frame].load(points, base)


def _load_mesh_triangle[
    frame: Frame
](
    points: ImmSpan[Float32, _], indices: ImmSpan[UInt32, _], face: UInt32
) -> Tuple[Point3f32[frame], Point3f32[frame], Point3f32[frame]]:
    # Internal traversal supplies a valid original face ID from a validated
    # mesh. Public vertex/face evaluation retains checked span access.
    var vertices = points.unsafe_ptr()
    var faces = indices.unsafe_ptr().unsafe_offset(Int(face) * 3)
    var a = Int(faces.unsafe_load(0)) * 3
    var b = Int(faces.unsafe_load(1)) * 3
    var c = Int(faces.unsafe_load(2)) * 3
    return (
        Point3f32[frame](vertices.unsafe_offset(a)),
        Point3f32[frame](vertices.unsafe_offset(b)),
        Point3f32[frame](vertices.unsafe_offset(c)),
    )


@fieldwise_init
struct MeshSphereQuery[origin: ImmOrigin, frame: Frame](Copyable):
    var candidates: BoundsQuery[Self.origin, Self.frame]
    var points: ImmSpan[Float32, Self.origin]
    var indices: ImmSpan[UInt32, Self.origin]
    var center: Point3f32[Self.frame]
    var radius: Float32

    @always_inline
    def next(mut self, mut face: UInt32) -> Bool:
        var candidate = UInt32(0)
        while self.candidates.next(candidate):
            var a: Point3f32[Self.frame]
            var b: Point3f32[Self.frame]
            var c: Point3f32[Self.frame]
            comptime if is_gpu():
                var triangle = _load_mesh_triangle[Self.frame](
                    self.points, self.indices, candidate
                )
                a = triangle[0]
                b = triangle[1]
                c = triangle[2]
            else:
                var base = Int(candidate) * 3
                a = load_mesh_point[Self.frame](self.points, self.indices[base])
                b = load_mesh_point[Self.frame](
                    self.points, self.indices[base + 1]
                )
                c = load_mesh_point[Self.frame](
                    self.points, self.indices[base + 2]
                )
            var radius2 = Float64(self.radius) * Float64(self.radius)
            var overlaps: Bool
            comptime if is_gpu():
                var hit, error, exact = (
                    project_triangle_filtered_with_exactness(
                        self.center, a, b, c
                    )
                )
                # Certified vertex distances still require canonical projection
                # when coordinate cancellation can affect Float64 endpoints.
                if not exact and error == 0.0:
                    hit = project_triangle(self.center, a, b, c)
                    overlaps = hit.distance2 <= radius2
                elif hit.distance2 + error <= radius2:
                    overlaps = True
                elif hit.distance2 - error > radius2:
                    overlaps = False
                else:
                    overlaps = (
                        project_triangle(self.center, a, b, c).distance2
                        <= radius2
                    )
            else:
                overlaps = (
                    project_triangle(self.center, a, b, c).distance2 <= radius2
                )
            if overlaps:
                face = candidate
                return True
        return False


struct MeshQueryView[
    origin: ImmOrigin, frame: Frame, storage_width: SIMDLength = 8
](Copyable):
    var points: ImmSpan[Float32, Self.origin]
    var indices: ImmSpan[UInt32, Self.origin]
    var velocities: ImmSpan[Float32, Self.origin]
    var bvh: BoundsQueryView[Self.origin, Self.frame]
    var packed_triangles: ImmSpan[Float32, Self.origin]
    var winding: WindingView[Self.origin, Self.frame]
    var device_winding: DeviceWindingView[Self.origin, Self.frame]

    @always_inline
    def __init__(
        out self,
        points: ImmSpan[Float32, Self.origin],
        indices: ImmSpan[UInt32, Self.origin],
        velocities: ImmSpan[Float32, Self.origin],
        var bvh: BoundsQueryView[Self.origin, Self.frame],
    ):
        self = Self(
            points, indices, velocities, bvh^, ImmSpan[Float32, Self.origin]()
        )

    @always_inline
    def __init__(
        out self,
        points: ImmSpan[Float32, Self.origin],
        indices: ImmSpan[UInt32, Self.origin],
        velocities: ImmSpan[Float32, Self.origin],
        var bvh: BoundsQueryView[Self.origin, Self.frame],
        packed_triangles: ImmSpan[Float32, Self.origin],
    ):
        self.points = points
        self.indices = indices
        self.velocities = velocities
        self.bvh = bvh^
        self.packed_triangles = packed_triangles
        self.winding = WindingView[Self.origin, Self.frame]()
        self.device_winding = DeviceWindingView[Self.origin, Self.frame]()

    @always_inline
    def __init__(
        out self,
        points: ImmSpan[Float32, Self.origin],
        indices: ImmSpan[UInt32, Self.origin],
        velocities: ImmSpan[Float32, Self.origin],
        var bvh: BoundsQueryView[Self.origin, Self.frame],
        packed_triangles: ImmSpan[Float32, Self.origin],
        var winding: WindingView[Self.origin, Self.frame],
    ):
        self = Self(points, indices, velocities, bvh^, packed_triangles)
        self.winding = winding^

    @always_inline
    def __init__(
        out self,
        points: ImmSpan[Float32, Self.origin],
        indices: ImmSpan[UInt32, Self.origin],
        velocities: ImmSpan[Float32, Self.origin],
        var bvh: BoundsQueryView[Self.origin, Self.frame],
        var device_winding: DeviceWindingView[Self.origin, Self.frame],
    ):
        self = Self(points, indices, velocities, bvh^)
        self.device_winding = device_winding^

    def vertex(self, index: UInt32) -> Point3f32[Self.frame]:
        return load_mesh_point[Self.frame](self.points, index)

    def face_vertex(self, face: UInt32, corner: Int) -> Point3f32[Self.frame]:
        return self.vertex(self.indices[Int(face) * 3 + corner])

    @always_inline
    def eval_position(
        self, face: UInt32, u: Float32, v: Float32
    ) -> Point3f32[Self.frame]:
        var a = vector64(self.face_vertex(face, 0))
        var b = vector64(self.face_vertex(face, 1))
        var c = vector64(self.face_vertex(face, 2))
        var p = (
            a * Float64(u)
            + b * Float64(v)
            + c * (1.0 - Float64(u) - Float64(v))
        )
        return Point3f32[Self.frame](Float32(p.x), Float32(p.y), Float32(p.z))

    @always_inline
    def eval_velocity(
        self, face: UInt32, u: Float32, v: Float32
    ) -> Vec3f32[Self.frame]:
        if len(self.velocities) == 0:
            return Vec3f32[Self.frame](0.0)
        var a = vector64(
            load_mesh_point[Self.frame](
                self.velocities, self.indices[Int(face) * 3]
            )
        )
        var b = vector64(
            load_mesh_point[Self.frame](
                self.velocities, self.indices[Int(face) * 3 + 1]
            )
        )
        var c = vector64(
            load_mesh_point[Self.frame](
                self.velocities, self.indices[Int(face) * 3 + 2]
            )
        )
        var p = (
            a * Float64(u)
            + b * Float64(v)
            + c * (1.0 - Float64(u) - Float64(v))
        )
        return Vec3f32[Self.frame](Float32(p.x), Float32(p.y), Float32(p.z))

    def eval_normal(self, face: UInt32) -> Vec3f32[Self.frame]:
        comptime if is_gpu():
            return triangle_normal_f32(
                self.face_vertex(face, 0),
                self.face_vertex(face, 1),
                self.face_vertex(face, 2),
            )
        var n = triangle_normal(
            self.face_vertex(face, 0),
            self.face_vertex(face, 1),
            self.face_vertex(face, 2),
        )
        return Vec3f32[Self.frame](Float32(n.x), Float32(n.y), Float32(n.z))

    @always_inline
    def query_aabb(
        self, box: AABB[Self.frame], root: UInt32 = 0
    ) -> BoundsQuery[Self.origin, Self.frame]:
        """Enumerate faces whose bounds overlap; this is broad phase."""
        return self.bvh.query_aabb(box, root)

    @always_inline
    def query_sphere(
        self, center: Point3f32[Self.frame], radius: Float32, root: UInt32 = 0
    ) -> MeshSphereQuery[Self.origin, Self.frame]:
        """Enumerate exact triangle/sphere intersections, including tangency."""
        return MeshSphereQuery(
            self.bvh.query_sphere(center, radius, root),
            self.points,
            self.indices,
            center,
            radius,
        )

    @always_inline
    def query_ray[
        width: SIMDLength = BATCH_WIDTH
    ](self, ray: Rayf32[Self.frame], root: UInt32 = 0) -> MeshRayResult[
        Self.frame
    ]:
        return self._query_ray_ordered[1 if is_gpu() else width](ray, root)

    @always_inline
    def _ray_node_entries[
        width: SIMDLength, full: Bool = False
    ](
        self,
        ray: Rayf32[Self.frame],
        inverse: Vec3[.float64, Self.frame],
        nodes: SIMD[.uint32, width],
    ) -> SIMD[.float64, width]:
        var active = SIMD[.bool, width](fill=True)
        comptime if not full:
            active = nodes.ne(EMPTY_LANE)
        var stride = 0
        comptime if width > 1:
            stride = (Int(nodes[1]) - Int(nodes[0])) * 6
        var bounds = self.bvh.bounds.unsafe_ptr().unsafe_offset(
            Int(nodes[0]) * 6
        )
        comptime if is_gpu() and width == 1:
            var box = AABB[Self.frame].load6(bounds)
            var filtered, entry = _gpu_ray_bounds_interval(
                ray, box, ray.t_max, 0.0, inverse
            )
            if filtered >= 0:
                return SIMD[.float64, width](
                    Float64(entry) if filtered == 1 else 1e300
                )
        var lower = SIMD[.float64, width](Float64(ray.t_min))
        var upper = SIMD[.float64, width](Float64(ray.t_max))
        comptime for axis in range(3):
            var lo = strided_load(
                bounds.unsafe_offset(axis), stride, active
            ).cast[.float64]()
            var hi = strided_load(
                bounds.unsafe_offset(axis + 3), stride, active
            ).cast[.float64]()
            var origin = Float64(ray.o[axis])
            if ray.d[axis] == 0.0:
                active &= lo.le(origin) & hi.ge(origin)
            else:
                var a = (lo - origin) * inverse[axis]
                var b = (hi - origin) * inverse[axis]
                lower = max(lower, min(a, b))
                upper = min(upper, max(a, b))
        # Conservatively retain near-touching slabs; exact triangle tests decide
        # hits. The same margin protects pruning against reciprocal rounding.
        active &= lower.le(upper + max(abs(lower), abs(upper)) * 1e-14)
        return active.select(lower, SIMD[.float64, width](1e300))

    @always_inline
    def _ray_node_lower_f32(
        self,
        ray: Rayf32[Self.frame],
        inverse: Vec3f32[Self.frame],
        node: UInt32,
    ) -> Float32:
        # Independent axis calculations avoid serial slab branches. Parallel
        # axes contribute the ray interval and retain exact box membership.
        # The outward margins and uncertain-node retention are unchanged.
        var bounds = self.bvh.bounds.unsafe_ptr().unsafe_offset(Int(node) * 6)
        var lo = SIMD[.float32, 4](
            bounds.unsafe_load(0),
            bounds.unsafe_load(1),
            bounds.unsafe_load(2),
            0.0,
        )
        var hi = SIMD[.float32, 4](
            bounds.unsafe_load(3),
            bounds.unsafe_load(4),
            bounds.unsafe_load(5),
            0.0,
        )
        var ray_origin = SIMD[.float32, 4](ray.o.x, ray.o.y, ray.o.z, 0.0)
        var inv = SIMD[.float32, 4](inverse.x, inverse.y, inverse.z, 0.0)
        var direction = SIMD[.float32, 4](ray.d.x, ray.d.y, ray.d.z, 0.0)
        var parallel = direction.eq(0.0)
        var outside = parallel & (ray_origin.lt(lo) | ray_origin.gt(hi))
        var a = (lo - ray_origin) * inv
        var b = (hi - ray_origin) * inv
        var error = (
            (max(abs(lo), abs(hi)) + abs(ray_origin)) * 0.000004 * abs(inv)
        )
        error += max(abs(a), abs(b)) * 0.000004 + Float32(1e-30)
        if (
            not isfinite(error[0])
            or not isfinite(error[1])
            or not isfinite(error[2])
        ):
            return ray.t_min
        var entries = parallel.select(
            SIMD[.float32, 4](ray.t_min), min(a, b) - error
        )
        var exits = parallel.select(
            SIMD[.float32, 4](ray.t_max), max(a, b) + error
        )
        var lower = max(ray.t_min, max(entries[0], max(entries[1], entries[2])))
        var upper = min(ray.t_max, min(exits[0], min(exits[1], exits[2])))
        var miss = bitcast[.float32](UInt32(0x7F800000))
        return (
            miss if outside[0]
            or outside[1]
            or outside[2]
            or lower > upper else lower
        )

    @always_inline
    def _ray_children_lower_f32(
        self,
        ray: Rayf32[Self.frame],
        inverse: Vec3f32[Self.frame],
        left: UInt32,
        right: UInt32,
    ) -> SIMD[.float32, 2]:
        # Evaluate both children together, retaining the scalar slab margins.
        var left_bounds = self.bvh.bounds.unsafe_ptr().unsafe_offset(
            Int(left) * 6
        )
        var right_bounds = self.bvh.bounds.unsafe_ptr().unsafe_offset(
            Int(right) * 6
        )
        var miss = bitcast[.float32](UInt32(0x7F800000))
        var lo0 = SIMD[.float32, 2](
            left_bounds.unsafe_load(0), right_bounds.unsafe_load(0)
        )
        var hi0 = SIMD[.float32, 2](
            left_bounds.unsafe_load(3), right_bounds.unsafe_load(3)
        )
        var parallel0 = SIMD[.bool, 2](fill=ray.d[0] == 0.0)
        var a0 = (lo0 - ray.o[0]) * inverse[0]
        var b0 = (hi0 - ray.o[0]) * inverse[0]
        var error0 = (
            (max(abs(lo0), abs(hi0)) + abs(ray.o[0]))
            * 0.000004
            * abs(inverse[0])
        )
        error0 += max(abs(a0), abs(b0)) * 0.000004 + Float32(1e-30)
        var entry0 = parallel0.select(
            SIMD[.float32, 2](ray.t_min), min(a0, b0) - error0
        )
        var exit0 = parallel0.select(
            SIMD[.float32, 2](ray.t_max), max(a0, b0) + error0
        )
        var outside0 = parallel0 & (~lo0.le(ray.o[0]) | ~hi0.ge(ray.o[0]))
        var lo1 = SIMD[.float32, 2](
            left_bounds.unsafe_load(1), right_bounds.unsafe_load(1)
        )
        var hi1 = SIMD[.float32, 2](
            left_bounds.unsafe_load(4), right_bounds.unsafe_load(4)
        )
        var parallel1 = SIMD[.bool, 2](fill=ray.d[1] == 0.0)
        var a1 = (lo1 - ray.o[1]) * inverse[1]
        var b1 = (hi1 - ray.o[1]) * inverse[1]
        var error1 = (
            (max(abs(lo1), abs(hi1)) + abs(ray.o[1]))
            * 0.000004
            * abs(inverse[1])
        )
        error1 += max(abs(a1), abs(b1)) * 0.000004 + Float32(1e-30)
        var entry1 = parallel1.select(
            SIMD[.float32, 2](ray.t_min), min(a1, b1) - error1
        )
        var exit1 = parallel1.select(
            SIMD[.float32, 2](ray.t_max), max(a1, b1) + error1
        )
        var outside1 = parallel1 & (~lo1.le(ray.o[1]) | ~hi1.ge(ray.o[1]))
        var lo2 = SIMD[.float32, 2](
            left_bounds.unsafe_load(2), right_bounds.unsafe_load(2)
        )
        var hi2 = SIMD[.float32, 2](
            left_bounds.unsafe_load(5), right_bounds.unsafe_load(5)
        )
        var parallel2 = SIMD[.bool, 2](fill=ray.d[2] == 0.0)
        var a2 = (lo2 - ray.o[2]) * inverse[2]
        var b2 = (hi2 - ray.o[2]) * inverse[2]
        var error2 = (
            (max(abs(lo2), abs(hi2)) + abs(ray.o[2]))
            * 0.000004
            * abs(inverse[2])
        )
        error2 += max(abs(a2), abs(b2)) * 0.000004 + Float32(1e-30)
        var entry2 = parallel2.select(
            SIMD[.float32, 2](ray.t_min), min(a2, b2) - error2
        )
        var exit2 = parallel2.select(
            SIMD[.float32, 2](ray.t_max), max(a2, b2) + error2
        )
        var outside2 = parallel2 & (~lo2.le(ray.o[2]) | ~hi2.ge(ray.o[2]))
        var lower = max(
            SIMD[.float32, 2](ray.t_min), max(entry0, max(entry1, entry2))
        )
        var upper = min(
            SIMD[.float32, 2](ray.t_max), min(exit0, min(exit1, exit2))
        )
        var uncertain = ~SIMD[.bool, 2](
            isfinite(error0[0]) and isfinite(error1[0]) and isfinite(error2[0]),
            isfinite(error0[1]) and isfinite(error1[1]) and isfinite(error2[1]),
        )
        var active = ~(outside0 | outside1 | outside2) & lower.le(upper)
        return uncertain.select(
            SIMD[.float32, 2](ray.t_min),
            active.select(lower, SIMD[.float32, 2](miss)),
        )

    @always_inline
    def _query_ray_f32_bounds(
        self,
        ray: Rayf32[Self.frame],
        root: UInt32,
        inverse: Vec3f32[Self.frame],
    ) -> MeshRayResult[Self.frame]:
        var result = MeshRayResult[Self.frame].miss()
        var miss = bitcast[.float32](UInt32(0x7F800000))
        var best = Float64(ray.t_max)
        var best_error = Float64(0.0)
        var best_upper = ray.t_max + abs(ray.t_max) * 0.000004 + Float32(1e-30)

        @always_inline
        def accepts(value: Float32, cutoff: Float32) {imm} -> Bool:
            return value != miss and value <= cutoff

        @always_inline
        def child_bounds(
            left: UInt32, right: UInt32
        ) {imm} -> Tuple[Float32, Float32]:
            var distances = self._ray_children_lower_f32(
                ray, inverse, left, right
            )
            var ld = distances[0]
            var rd = distances[1]
            return (ld, rd)

        @always_inline
        def visit_face(
            face: UInt32,
        ) {
            imm, mut result, mut best, mut best_error, mut best_upper
        } -> Float32:
            var triangle = _load_mesh_triangle[Self.frame](
                self.points, self.indices, face
            )
            var filtered, candidate_t, candidate_error = ray_triangle_interval(
                ray, triangle[0], triangle[1], triangle[2]
            )
            if filtered != 0:
                var hit_t = Float64(candidate_t)
                var error = Float64(candidate_error)
                var eligible = True
                var hit_u = Float64(0.0)
                var hit_v = Float64(0.0)
                if filtered < 0:
                    var hit = intersect_triangle(
                        ray, triangle[0], triangle[1], triangle[2]
                    )
                    eligible = hit.result
                    hit_t = hit.t
                    error = 0.0
                    hit_u = hit.u
                    hit_v = hit.v
                eligible &= hit_t - error <= best + best_error
                if eligible and hit_t + error >= best - best_error:
                    # Overlapping intervals require the existing exact
                    # distances before choosing a face or breaking a tie.
                    if result.result and best_error > 0.0:
                        var previous = _load_mesh_triangle[Self.frame](
                            self.points, self.indices, result.face
                        )
                        var exact = intersect_triangle(
                            ray, previous[0], previous[1], previous[2]
                        )
                        best = exact.t
                        best_error = 0.0
                        result.u = Float32(exact.u)
                        result.v = Float32(exact.v)
                    if error > 0.0:
                        var exact = intersect_triangle(
                            ray, triangle[0], triangle[1], triangle[2]
                        )
                        eligible = exact.result
                        hit_t = exact.t
                        hit_u = exact.u
                        hit_v = exact.v
                        error = 0.0
                if (
                    eligible
                    and hit_t <= best + best_error
                    and (
                        not result.result
                        or hit_t < best
                        or (hit_t == best and face < result.face)
                    )
                ):
                    best = hit_t
                    best_error = error
                    result.result = True
                    result.face = face
                    result.u = Float32(hit_u)
                    result.v = Float32(hit_v)
                # A provisional winner's upper endpoint must be retained
                # until its final exact projection is computed.
                var rounded = Float32(best + best_error)
                best_upper = rounded + abs(rounded) * 0.000004 + Float32(1e-30)
            return best_upper

        traverse_ordered[.float32, False](
            self.bvh.nodes,
            root,
            self._ray_node_lower_f32(ray, inverse, root),
            best_upper,
            accepts,
            child_bounds,
            visit_face,
        )
        if result.result:
            var triangle = _load_mesh_triangle[Self.frame](
                self.points, self.indices, result.face
            )
            if best_error > 0.0:
                var exact = intersect_triangle(
                    ray, triangle[0], triangle[1], triangle[2]
                )
                best = exact.t
                result.u = Float32(exact.u)
                result.v = Float32(exact.v)
            result.t = best
            var n = cross(
                vector64(triangle[1]) - vector64(triangle[0]),
                vector64(triangle[2]) - vector64(triangle[0]),
            )
            if n.x != 0.0 and n.y == 0.0 and n.z == 0.0:
                n.x = Float64(-1.0 if n.x < 0.0 else 1.0)
            elif n.y != 0.0 and n.x == 0.0 and n.z == 0.0:
                n.y = Float64(-1.0 if n.y < 0.0 else 1.0)
            elif n.z != 0.0 and n.x == 0.0 and n.y == 0.0:
                n.z = Float64(-1.0 if n.z < 0.0 else 1.0)
            else:
                var length2 = dot(n, n)
                if length2 > 0.0:
                    n = n / sqrt(length2)
                else:
                    n = Vec3[.float64, Self.frame](0.0)
            var direction = Vec3[.float64, Self.frame](
                Float64(ray.d.x), Float64(ray.d.y), Float64(ray.d.z)
            )
            result.normal = Vec3f32[Self.frame](
                Float32(n.x), Float32(n.y), Float32(n.z)
            )
            result.front_face = dot(n, direction) < 0.0
        return result

    @always_inline
    def _ray_f32_inverse(
        self, ray: Rayf32[Self.frame], root: UInt32
    ) -> Tuple[Vec3f32[Self.frame], Bool]:
        var regular = True
        var inverse32 = Vec3f32[Self.frame](0.0)
        var bounds = self.bvh.bounds.unsafe_ptr().unsafe_offset(Int(root) * 6)
        comptime for axis in range(3):
            regular &= (
                max(
                    abs(bounds.unsafe_load(axis)),
                    abs(bounds.unsafe_load(axis + 3)),
                )
                <= 1e20
            )
            regular &= abs(ray.o[axis]) <= 1e20
            var direction = abs(ray.d[axis])
            if (bitcast[.uint32](ray.d[axis]) & UInt32(0x7FFFFFFF)) != 0:
                if direction < 1e-10 or direction > 1e10:
                    regular = False
                else:
                    inverse32[axis] = Float32(1.0) / ray.d[axis]
        return (inverse32, regular)

    @always_inline
    def _query_ray_ordered[
        width: SIMDLength
    ](self, ray: Rayf32[Self.frame], root: UInt32) -> MeshRayResult[Self.frame]:
        comptime assert width > 0
        var result = MeshRayResult[Self.frame].miss()
        if (
            not ray.o.is_finite()[0]
            or not ray.d.is_finite()[0]
            or not isfinite(ray.t_min)
            or not isfinite(ray.t_max)
            or ray.t_min > ray.t_max
            or Int(root) >= len(self.bvh.nodes) // 2
        ):
            return result
        comptime if is_gpu():
            var inverse32, regular = self._ray_f32_inverse(ray, root)
            if regular:
                return self._query_ray_f32_bounds(ray, root, inverse32)
        var inverse = Vec3[.float64, Self.frame](0.0)
        comptime for axis in range(3):
            var direction = Float64(ray.d[axis])
            inverse[axis] = 1.0 / direction if direction != 0.0 else 0.0
        var best = Float64(ray.t_max)

        @always_inline
        def accepts(value: Float64, cutoff: Float64) {imm} -> Bool:
            return value <= cutoff

        @always_inline
        def child_bounds(
            left: UInt32, right: UInt32
        ) {imm} -> Tuple[Float64, Float64]:
            var ld: Float64
            var rd: Float64
            comptime if width == 1:
                ld = self._ray_node_entries[1, True](
                    ray, inverse, SIMD[.uint32, 1](left)
                )
                rd = self._ray_node_entries[1, True](
                    ray, inverse, SIMD[.uint32, 1](right)
                )
            else:
                var children = SIMD[.uint32, width](EMPTY_LANE)
                children[0] = left
                children[1] = right
                var child_entries = self._ray_node_entries[width, width == 2](
                    ray, inverse, children
                )
                ld = child_entries[0]
                rd = child_entries[1]
            return (ld, rd)

        @always_inline
        def visit_face(face: UInt32) {imm, mut result, mut best} -> Float64:
            var triangle = _load_mesh_triangle[Self.frame](
                self.points, self.indices, face
            )
            var hit = intersect_triangle(
                ray, triangle[0], triangle[1], triangle[2]
            )
            if hit.result and (
                not result.result
                or hit.t < best
                or (hit.t == best and face < result.face)
            ):
                best = hit.t
                result.result = True
                result.face = face
                result.u = Float32(hit.u)
                result.v = Float32(hit.v)
                result.t = hit.t
            return best + abs(best) * 1e-14

        traverse_ordered[.float64, False](
            self.bvh.nodes,
            root,
            self._ray_node_entries[1, True](
                ray, inverse, SIMD[.uint32, 1](root)
            ),
            best + abs(best) * 1e-14,
            accepts,
            child_bounds,
            visit_face,
        )
        if result.result:
            var triangle = _load_mesh_triangle[Self.frame](
                self.points, self.indices, result.face
            )
            var n = triangle_normal(triangle[0], triangle[1], triangle[2])
            var direction = Vec3[.float64, Self.frame](
                Float64(ray.d.x), Float64(ray.d.y), Float64(ray.d.z)
            )
            result.normal = Vec3f32[Self.frame](
                Float32(n.x), Float32(n.y), Float32(n.z)
            )
            result.front_face = dot(n, direction) < 0.0
        return result

    @always_inline
    def query_ray_anyhit[
        width: SIMDLength = BATCH_WIDTH
    ](self, ray: Rayf32[Self.frame], root: UInt32 = 0) -> Bool:
        return (
            self._query_ray_hits[1 if is_gpu() else width, True](ray, root)[0]
            != 0
        )

    @always_inline
    def query_ray_count_intersections[
        width: SIMDLength = RAY_COUNT_WIDTH
    ](self, ray: Rayf32[Self.frame], root: UInt32 = 0) -> UInt32:
        return self._query_ray_hits[1 if is_gpu() else width, False](ray, root)[
            0
        ]

    @always_inline
    def _query_ray_any_f32(
        self,
        ray: Rayf32[Self.frame],
        root: UInt32,
        inverse: Vec3f32[Self.frame],
    ) -> Bool:
        # Visit the nearer child first to reach an accepted hit sooner.
        # A pending child already passed the conservative ray bounds test.
        var pending = stack_allocation[64, UInt32]()
        var count = 0
        var node = root
        var entry = self._ray_node_lower_f32(ray, inverse, node)
        var miss = bitcast[.float32](UInt32(0x7F800000))
        while True:
            if entry != miss:
                var face = self.bvh.nodes.unsafe_ptr().unsafe_load(
                    Int(node) * 2 + 1
                )
                if face == EMPTY_LANE:
                    var left = node + 1
                    var right = self.bvh.nodes.unsafe_ptr().unsafe_load(
                        Int(left) * 2
                    )
                    var distances = self._ray_children_lower_f32(
                        ray, inverse, left, right
                    )
                    var ld = distances[0]
                    var rd = distances[1]
                    if rd < ld:
                        var temp = left
                        left = right
                        right = temp
                        var td = ld
                        ld = rd
                        rd = td
                    if rd != miss:
                        pending[unsafe_offset=count] = right
                        count += 1
                    node = left
                    entry = ld
                    continue
                var triangle = _load_mesh_triangle[Self.frame](
                    self.points, self.indices, face
                )
                var filtered = ray_triangle_filter(
                    ray, triangle[0], triangle[1], triangle[2]
                )
                if filtered == 1:
                    return True
                if (
                    filtered < 0
                    and intersect_triangle(
                        ray, triangle[0], triangle[1], triangle[2]
                    ).result
                ):
                    return True
            if count == 0:
                return False
            count -= 1
            node = pending[unsafe_offset=count]
            entry = ray.t_min

    @always_inline
    def _query_ray_hits_f32[
        anyhit: Bool, parity: Bool
    ](
        self,
        ray: Rayf32[Self.frame],
        root: UInt32,
        inverse: Vec3f32[Self.frame],
    ) -> Tuple[UInt32, Bool]:
        comptime if anyhit and not parity:
            return (UInt32(self._query_ray_any_f32(ray, root, inverse)), False)
        # Preorder escape links prune complete subtrees without a candidate
        # iterator or traversal stack. Every face retains its original ID.
        var node = root
        var end = self.bvh.nodes.unsafe_ptr().unsafe_load(Int(root) * 2)
        var miss = bitcast[.float32](UInt32(0x7F800000))
        var hits = UInt32(0)
        while node < end:
            var escape = self.bvh.nodes.unsafe_ptr().unsafe_load(Int(node) * 2)
            if self._ray_node_lower_f32(ray, inverse, node) == miss:
                node = escape
                continue
            var face = self.bvh.nodes.unsafe_ptr().unsafe_load(
                Int(node) * 2 + 1
            )
            node += 1
            if face == EMPTY_LANE:
                continue
            var triangle = _load_mesh_triangle[Self.frame](
                self.points, self.indices, face
            )
            var filtered = ray_triangle_filter(
                ray, triangle[0], triangle[1], triangle[2]
            )
            if filtered == 0:
                continue
            if filtered == 1:
                comptime if anyhit:
                    return (1, False)
                else:
                    hits += 1
                    continue
            var hit = intersect_triangle(
                ray, triangle[0], triangle[1], triangle[2]
            )
            if hit.result:
                comptime if parity:
                    if (
                        hit.u <= 1e-12
                        or hit.v <= 1e-12
                        or 1.0 - hit.u - hit.v <= 1e-12
                    ):
                        return (0, True)
                comptime if anyhit:
                    return (1, False)
                else:
                    hits += 1
        return (hits, False)

    @always_inline
    def _query_ray_hits[
        width: SIMDLength, anyhit: Bool, parity: Bool = False
    ](self, ray: Rayf32[Self.frame], root: UInt32) -> Tuple[UInt32, Bool]:
        comptime assert width > 0
        comptime if is_gpu():
            if (
                not ray.o.is_finite()[0]
                or not ray.d.is_finite()[0]
                or not isfinite(ray.t_min)
                or not isfinite(ray.t_max)
                or ray.t_min > ray.t_max
                or Int(root) >= len(self.bvh.nodes) // 2
            ):
                return (0, False)
            var inverse32, regular = self._ray_f32_inverse(ray, root)
            if regular:
                return self._query_ray_hits_f32[anyhit, parity](
                    ray, root, inverse32
                )

        var candidates = self.bvh.query_ray(ray, root)
        var faces = stack_allocation[width, UInt32]()
        # Early-exit queries test the first candidate before traversing farther.
        # Bounds lookahead retains the caller's SIMD width.
        var face_span = MutSpan(unsafe_ptr=faces, length=1 if anyhit else width)
        var hits = UInt32(0)
        while True:
            var count = candidates.next_batch[width](face_span)
            if count == 0:
                return (hits, False)
            for lane in range(count):
                var face = faces[unsafe_offset=lane]
                var triangle = _load_mesh_triangle[Self.frame](
                    self.points, self.indices, face
                )
                comptime if is_gpu():
                    var filtered = ray_triangle_filter(
                        ray, triangle[0], triangle[1], triangle[2]
                    )
                    if filtered == 0:
                        continue
                    if filtered == 1:
                        comptime if anyhit:
                            return (1, False)
                        else:
                            hits += 1
                            continue
                var hit = intersect_triangle(
                    ray, triangle[0], triangle[1], triangle[2]
                )
                if hit.result:
                    comptime if parity:
                        if (
                            hit.u <= 1e-12
                            or hit.v <= 1e-12
                            or 1.0 - hit.u - hit.v <= 1e-12
                        ):
                            return (0, True)
                    comptime if anyhit:
                        return (1, False)
                    else:
                        hits += 1

    @always_inline
    def _point_node_distances2[
        width: SIMDLength, furthest: Bool = False
    ](self, point: Point3f32[Self.frame], nodes: SIMD[.uint32, width]) -> SIMD[
        .float64, width
    ]:
        # Callers supply one node or a child pair, padding wider packets with
        # EMPTY_LANE. Two addresses define a runtime stride even in preorder.
        var active = nodes.ne(EMPTY_LANE)
        var stride = 0
        comptime if width > 1:
            stride = (Int(nodes[1]) - Int(nodes[0])) * 6
        var bounds = self.bvh.bounds.unsafe_ptr().unsafe_offset(
            Int(nodes[0]) * 6
        )
        var lower = Point3[.float32, Self.frame, width](0.0)
        var upper = Point3[.float32, Self.frame, width](0.0)
        comptime for axis in range(3):
            lower[axis] = strided_load(
                bounds.unsafe_offset(axis), stride, active
            )
            upper[axis] = strided_load(
                bounds.unsafe_offset(axis + 3), stride, active
            )
        comptime if is_gpu() and width == 1:
            var d2 = Float32(0.0)
            comptime for axis in range(3):
                var delta: Float32
                comptime if furthest:
                    delta = max(
                        abs(lower[axis][0] - point[axis]),
                        abs(upper[axis][0] - point[axis]),
                    )
                else:
                    delta = max(
                        max(
                            lower[axis][0] - point[axis],
                            point[axis] - upper[axis][0],
                        ),
                        Float32(0.0),
                    )
                d2 += delta * delta
            if isfinite(d2) and d2 > 1e-30:
                # Outward bounds retain all possible improving/tied faces.
                comptime if furthest:
                    return SIMD[.float64, width](Float64(d2) * 1.000002)
                else:
                    return SIMD[.float64, width](Float64(d2) * 0.999998)
            if d2 == 0.0:
                comptime if not furthest:
                    return SIMD[.float64, width](0.0)
        var distance2 = AxisAlignedBoundingBox[.float32, Self.frame, width](
            lower, upper
        ).distance2[furthest](point)
        return nodes.ne(EMPTY_LANE).select(
            distance2, SIMD[.float64, width](-1.0 if furthest else 1e300)
        )

    @always_inline
    def _furthest_node_upper_f32(
        self, point: Point3f32[Self.frame], node: UInt32
    ) -> Float32:
        var bounds = self.bvh.bounds.unsafe_ptr().unsafe_offset(Int(node) * 6)
        var distance2 = Float32(0.0)
        comptime for axis in range(3):
            var delta = max(
                abs(bounds.unsafe_load(axis) - point[axis]),
                abs(bounds.unsafe_load(axis + 3) - point[axis]),
            )
            distance2 += delta * delta
        # Cover subtraction/square/sum rounding and flushed tiny differences.
        # Overflow gives +infinity, which conservatively retains the node.
        return distance2 * Float32(1.000002) + Float32(1e-30)

    @always_inline
    def _query_furthest_f32_bounds(
        self,
        point: Point3f32[Self.frame],
        cutoff: Float32,
        root: UInt32,
        root_upper: Float32,
    ) -> MeshPointResult[Self.frame]:
        var best_face = EMPTY_LANE
        var best_corner = 0
        var best2 = Float64(cutoff) * Float64(cutoff)
        var lower = Float32(best2) * Float32(0.999998)

        @always_inline
        def accepts(value: Float32, cutoff: Float32) {imm} -> Bool:
            return value >= cutoff

        @always_inline
        def child_bounds(
            left: UInt32, right: UInt32
        ) {imm} -> Tuple[Float32, Float32]:
            var ld = self._furthest_node_upper_f32(point, left)
            var rd = self._furthest_node_upper_f32(point, right)
            return (ld, rd)

        @always_inline
        def visit_face(
            face: UInt32,
        ) {
            imm, mut best_face, mut best_corner, mut best2, mut lower
        } -> Float32:
            var triangle = _load_mesh_triangle[Self.frame](
                self.points, self.indices, face
            )
            # Conservative vertex intervals avoid exact work for losers.
            # The finite root bound guards the ordinary-range fast path.
            var v0 = triangle[0] - point
            var v1 = triangle[1] - point
            var v2 = triangle[2] - point
            var f0 = dot(v0, v0)
            var f1 = dot(v1, v1)
            var f2 = dot(v2, v2)
            var e0 = f0 * Float32(0.000002) + Float32(1e-30)
            var e1 = f1 * Float32(0.000002) + Float32(1e-30)
            var e2 = f2 * Float32(0.000002) + Float32(1e-30)
            if max(max(f0 + e0, f1 + e1), f2 + e2) >= lower:
                var selected_corner = -1
                if f0 - e0 > max(f1 + e1, f2 + e2):
                    selected_corner = 0
                elif f1 - e1 > max(f0 + e0, f2 + e2):
                    selected_corner = 1
                elif f2 - e2 > max(f0 + e0, f1 + e1):
                    selected_corner = 2
                var selected_distance: Float64
                if selected_corner >= 0:
                    var vertex = triangle[0]
                    if selected_corner == 1:
                        vertex = triangle[1]
                    elif selected_corner == 2:
                        vertex = triangle[2]
                    var delta = vector64(vertex) - vector64(point)
                    selected_distance = dot(delta, delta)
                else:
                    selected_distance, selected_corner = _furthest_corner_exact[
                        Self.frame
                    ](point, triangle[0], triangle[1], triangle[2])
                if selected_distance >= best2 and (
                    best_face == EMPTY_LANE
                    or selected_distance > best2
                    or face < best_face
                ):
                    best2 = selected_distance
                    lower = Float32(best2) * Float32(0.999998)
                    best_face = face
                    best_corner = selected_corner
            return lower

        traverse_ordered[.float32, True](
            self.bvh.nodes,
            root,
            root_upper,
            lower,
            accepts,
            child_bounds,
            visit_face,
        )
        if best_face == EMPTY_LANE:
            return MeshPointResult[Self.frame].miss()
        var triangle = _load_mesh_triangle[Self.frame](
            self.points, self.indices, best_face
        )
        var vertex = triangle[0]
        if best_corner == 1:
            vertex = triangle[1]
        elif best_corner == 2:
            vertex = triangle[2]
        return MeshPointResult(
            True,
            best_face,
            Float32(1.0 if best_corner == 0 else 0.0),
            Float32(1.0 if best_corner == 1 else 0.0),
            sqrt(best2),
            vertex,
            0.0,
        )

    @always_inline
    def _nearest_node_lower_f32(
        self, point: Point3f32[Self.frame], node: UInt32
    ) -> Float32:
        var bounds = self.bvh.bounds.unsafe_ptr().unsafe_offset(Int(node) * 6)
        var distance2 = Float32(0.0)
        comptime for axis in range(3):
            var delta = max(
                max(
                    bounds.unsafe_load(axis) - point[axis],
                    point[axis] - bounds.unsafe_load(axis + 3),
                ),
                Float32(0.0),
            )
            distance2 += delta * delta
        return max(distance2 * Float32(0.999998) - Float32(1e-30), Float32(0.0))

    @always_inline
    def _query_nearest_f32_bounds(
        self,
        point: Point3f32[Self.frame],
        cutoff: Float32,
        root: UInt32,
    ) -> MeshPointResult[Self.frame]:
        # Keep only winner identity/distance in the traversal. Construct all
        # canonical fields together after the final face has been selected.
        var best_face = EMPTY_LANE
        var best2 = Float64(cutoff) * Float64(cutoff)
        var best_error = Float64(0.0)
        var upper = Float32(best2) * Float32(1.000002) + Float32(1e-30)

        @always_inline
        def accepts(value: Float32, cutoff: Float32) {imm} -> Bool:
            return value <= cutoff

        @always_inline
        def child_bounds(
            left: UInt32, right: UInt32
        ) {imm} -> Tuple[Float32, Float32]:
            var ld = self._nearest_node_lower_f32(point, left)
            var rd = self._nearest_node_lower_f32(point, right)
            return (ld, rd)

        @always_inline
        def visit_face(
            face: UInt32,
        ) {imm, mut best_face, mut best2, mut best_error, mut upper} -> Float32:
            var triangle = _load_mesh_triangle[Self.frame](
                self.points, self.indices, face
            )
            var hit, error, _ = project_triangle_filtered_with_exactness(
                point, triangle[0], triangle[1], triangle[2]
            )
            var eligible = hit.distance2 - error <= best2 + best_error
            if (
                eligible
                and (error > 0.0 or best_error > 0.0)
                and hit.distance2 + error >= best2 - best_error
            ):
                if best_face != EMPTY_LANE and best_error > 0.0:
                    var previous = _load_mesh_triangle[Self.frame](
                        self.points, self.indices, best_face
                    )
                    best2 = project_triangle(
                        point, previous[0], previous[1], previous[2]
                    ).distance2
                    best_error = 0.0
                hit = project_triangle(
                    point, triangle[0], triangle[1], triangle[2]
                )
                error = 0.0
            if (
                eligible
                and hit.distance2 <= best2
                and (
                    best_face == EMPTY_LANE
                    or hit.distance2 < best2
                    or face < best_face
                )
            ):
                best2 = hit.distance2
                best_error = error
                best_face = face
            # Recompute even if only the previous provisional winner was
            # refined, so pruning retains its current uncertainty interval.
            upper = Float32(best2 + best_error) * Float32(1.000002) + Float32(
                1e-30
            )
            return upper

        traverse_ordered[.float32, False](
            self.bvh.nodes,
            root,
            self._nearest_node_lower_f32(point, root),
            upper,
            accepts,
            child_bounds,
            visit_face,
        )
        if best_face == EMPTY_LANE:
            return MeshPointResult[Self.frame].miss()
        var triangle = _load_mesh_triangle[Self.frame](
            self.points, self.indices, best_face
        )
        var exact = project_triangle(
            point, triangle[0], triangle[1], triangle[2]
        )
        return MeshPointResult(
            True,
            best_face,
            Float32(exact.u),
            Float32(exact.v),
            sqrt(exact.distance2),
            Point3f32[Self.frame](
                Float32(exact.point.x),
                Float32(exact.point.y),
                Float32(exact.point.z),
            ),
            0.0,
        )

    @always_inline
    def _query_point_distance[
        width: SIMDLength, furthest: Bool
    ](
        self,
        point: Point3f32[Self.frame],
        cutoff: Float32,
        root: UInt32,
    ) -> MeshPointResult[Self.frame]:
        comptime assert width > 0
        var result = MeshPointResult[Self.frame].miss()
        if (
            not point.is_finite()[0]
            or not isfinite(cutoff)
            or cutoff < 0.0
            or Int(root) >= len(self.bvh.nodes) // 2
        ):
            return result
        comptime if is_gpu() and furthest:
            var root_upper = self._furthest_node_upper_f32(point, root)
            if (
                isfinite(root_upper)
                and root_upper > 1e-28
                and isfinite(cutoff * cutoff)
            ):
                return self._query_furthest_f32_bounds(
                    point, cutoff, root, root_upper
                )
        comptime if is_gpu() and not furthest:
            var root_upper = self._furthest_node_upper_f32(point, root)
            if isfinite(root_upper) and root_upper > 1e-28:
                return self._query_nearest_f32_bounds(point, cutoff, root)
        var best2 = Float64(cutoff) * Float64(cutoff)
        var best_error = Float64(0.0)

        # Balanced trees have depth <= 32 for UInt32 node indices. Device rebuild
        # retains that skeleton. One pending far child per level fits here.
        @always_inline
        def accepts(value: Float64, cutoff: Float64) {imm} -> Bool:
            return value >= cutoff if furthest else value <= cutoff

        @always_inline
        def child_bounds(
            left: UInt32, right: UInt32
        ) {imm} -> Tuple[Float64, Float64]:
            var ld: Float64
            var rd: Float64
            comptime if width == 1:
                ld = self._point_node_distances2[1, furthest](
                    point, SIMD[.uint32, 1](left)
                )
                rd = self._point_node_distances2[1, furthest](
                    point, SIMD[.uint32, 1](right)
                )
            else:
                var children = SIMD[.uint32, width](EMPTY_LANE)
                children[0] = left
                children[1] = right
                var bounds_distance2 = self._point_node_distances2[
                    width, furthest
                ](point, children)
                ld = bounds_distance2[0]
                rd = bounds_distance2[1]
            return (ld, rd)

        @always_inline
        def visit_face(
            face: UInt32,
        ) {imm, mut result, mut best2, mut best_error} -> Float64:
            comptime if furthest:
                var triangle = _load_mesh_triangle[Self.frame](
                    self.points, self.indices, face
                )
                comptime for corner in range(3):
                    var vertex = triangle[corner]
                    var delta = vector64(vertex) - vector64(point)
                    var vertex_distance2 = dot(delta, delta)
                    if vertex_distance2 >= best2 and (
                        not result.result
                        or vertex_distance2 > best2
                        or face < result.face
                    ):
                        best2 = vertex_distance2
                        result = MeshPointResult(
                            True,
                            face,
                            Float32(1.0 if corner == 0 else 0.0),
                            Float32(1.0 if corner == 1 else 0.0),
                            0.0,
                            vertex,
                            0.0,
                        )

            else:
                var triangle = _load_mesh_triangle[Self.frame](
                    self.points, self.indices, face
                )
                var hit: TriangleProjection[Self.frame]
                var error = Float64(0.0)
                var eligible = True
                comptime if is_gpu():
                    hit, error = project_triangle_filtered(
                        point, triangle[0], triangle[1], triangle[2]
                    )
                    eligible = hit.distance2 - error <= best2 + best_error
                    if (
                        eligible
                        and (error > 0.0 or best_error > 0.0)
                        and hit.distance2 + error >= best2 - best_error
                    ):
                        if result.result and best_error > 0.0:
                            var previous = _load_mesh_triangle[Self.frame](
                                self.points, self.indices, result.face
                            )
                            best2 = project_triangle(
                                point, previous[0], previous[1], previous[2]
                            ).distance2
                            best_error = 0.0
                        hit = project_triangle(
                            point, triangle[0], triangle[1], triangle[2]
                        )
                        error = 0.0
                else:
                    hit = project_triangle(
                        point, triangle[0], triangle[1], triangle[2]
                    )
                if (
                    eligible
                    and hit.distance2 <= best2
                    and (
                        not result.result
                        or hit.distance2 < best2
                        or face < result.face
                    )
                ):
                    best2 = hit.distance2
                    best_error = error
                    result = MeshPointResult(
                        True,
                        face,
                        Float32(hit.u),
                        Float32(hit.v),
                        0.0,
                        Point3f32[Self.frame](
                            Float32(hit.point.x),
                            Float32(hit.point.y),
                            Float32(hit.point.z),
                        ),
                        0.0,
                    )
            return (
                (best2 - best_error)
                * 0.99999999999999 if furthest else (best2 + best_error)
                * 1.00000000000001
            )

        traverse_ordered[.float64, furthest](
            self.bvh.nodes,
            root,
            self._point_node_distances2[1, furthest](
                point, SIMD[.uint32, 1](root)
            ),
            (best2 - best_error)
            * 0.99999999999999 if furthest else (best2 + best_error)
            * 1.00000000000001,
            accepts,
            child_bounds,
            visit_face,
        )
        if result.result:
            comptime if is_gpu() and not furthest:
                var triangle = _load_mesh_triangle[Self.frame](
                    self.points, self.indices, result.face
                )
                var exact = project_triangle(
                    point, triangle[0], triangle[1], triangle[2]
                )
                best2 = exact.distance2
                result.u = Float32(exact.u)
                result.v = Float32(exact.v)
                result.point = Point3f32[Self.frame](
                    Float32(exact.point.x),
                    Float32(exact.point.y),
                    Float32(exact.point.z),
                )
            comptime if is_gpu() and furthest:
                var delta = vector64(result.point) - vector64(point)
                best2 = dot(delta, delta)
            result.distance = sqrt(best2)
        return result

    def query_point_no_sign[
        width: SIMDLength = 2
    ](
        self,
        point: Point3f32[Self.frame],
        max_dist: Float32 = f32_max,
        root: UInt32 = 0,
    ) -> MeshPointResult[Self.frame]:
        return self._query_point_distance[1 if is_gpu() else width, False](
            point, max_dist, root
        )

    def _all_faces(self, root: UInt32) -> BoundsQuery[Self.origin, Self.frame]:
        return self.query_aabb(
            AABB(
                Point3f32[Self.frame](-f32_max), Point3f32[Self.frame](f32_max)
            ),
            root,
        )

    def query_furthest_point_no_sign[
        width: SIMDLength = 2
    ](
        self,
        point: Point3f32[Self.frame],
        min_dist: Float32 = 0.0,
        root: UInt32 = 0,
    ) -> MeshPointResult[Self.frame]:
        return self._query_point_distance[1 if is_gpu() else width, True](
            point, min_dist, root
        )

    def winding_number[
        width: SIMDLength = Self.storage_width
    ](self, point: Point3f32[Self.frame], root: UInt32 = 0) -> Float64:
        # GPU lanes process independent queries; CPU packets process faces.
        comptime target_width = 1 if is_gpu() else width
        return self._winding_number[target_width](point, root)

    @always_inline
    def _accumulate_packed_angles[
        width: SIMDLength
    ](
        self,
        point: Point3f32[Self.frame],
        begin: Int,
        end: Int,
        mut sum: SIMD[.float64, width],
        mut compensation: SIMD[.float64, width],
    ):
        var triangles = self.packed_triangles.unsafe_ptr()
        for block in range(begin, end):
            var base = block * 9 * width
            var a = Vec3[.float64, Self.frame, width](
                triangles.unsafe_load[width=width](base).cast[.float64](),
                triangles.unsafe_load[width=width](base + width).cast[
                    .float64
                ](),
                triangles.unsafe_load[width=width](base + 2 * width).cast[
                    .float64
                ](),
            )
            var b = Vec3[.float64, Self.frame, width](
                triangles.unsafe_load[width=width](base + 3 * width).cast[
                    .float64
                ](),
                triangles.unsafe_load[width=width](base + 4 * width).cast[
                    .float64
                ](),
                triangles.unsafe_load[width=width](base + 5 * width).cast[
                    .float64
                ](),
            )
            var c = Vec3[.float64, Self.frame, width](
                triangles.unsafe_load[width=width](base + 6 * width).cast[
                    .float64
                ](),
                triangles.unsafe_load[width=width](base + 7 * width).cast[
                    .float64
                ](),
                triangles.unsafe_load[width=width](base + 8 * width).cast[
                    .float64
                ](),
            )
            var adjusted = solid_angle(point, a, b, c) - compensation
            var updated = sum + adjusted
            compensation = (updated - sum) - adjusted
            sum = updated

    @always_inline
    def _accumulate_device_angles(
        self,
        point: Point3f32[Self.frame],
        begin: UInt32,
        end: UInt32,
        mut sum: Float64,
        mut compensation: Float64,
    ):
        for slot in range(Int(begin), Int(end)):
            var triangle = _load_mesh_triangle[Self.frame](
                self.points, self.indices, self.device_winding.faces[slot]
            )
            var adjusted = (
                solid_angle(
                    point,
                    vector64(triangle[0]),
                    vector64(triangle[1]),
                    vector64(triangle[2]),
                )
                - compensation
            )
            var updated = sum + adjusted
            compensation = (updated - sum) - adjusted
            sum = updated

    def _winding_number[
        width: SIMDLength
    ](self, point: Point3f32[Self.frame], root: UInt32) -> Float64:
        if not point.is_finite()[0] or Int(root) >= len(self.bvh.nodes) // 2:
            return 0.0
        var sum = SIMD[.float64, width](0.0)
        var compensation = SIMD[.float64, width](0.0)
        comptime if is_gpu() and width == 1:
            if root == 0 and len(self.device_winding.offsets) >= 2:
                var device_sum = Float64(0.0)
                var device_compensation = Float64(0.0)
                var count = len(self.device_winding.offsets) - 2
                self._accumulate_device_angles(
                    point,
                    self.device_winding.offsets[count],
                    self.device_winding.offsets[count + 1],
                    device_sum,
                    device_compensation,
                )
                var candidates = self.device_winding.bvh.query_aabb(
                    AABB[Self.frame](point, point)
                )
                var component = UInt32(0)
                while candidates.next(component):
                    self._accumulate_device_angles(
                        point,
                        self.device_winding.offsets[Int(component)],
                        self.device_winding.offsets[Int(component) + 1],
                        device_sum,
                        device_compensation,
                    )
                return device_sum * INV_4_PI
        comptime if width == Self.storage_width:
            if root == 0 and len(self.packed_triangles) > 0:
                if len(self.winding.offsets) > 2:
                    # Open or uncertified faces always contribute. Closed
                    # components contribute zero outside their AABBs, exactly.
                    var count = len(self.winding.offsets) - 2
                    self._accumulate_packed_angles(
                        point,
                        Int(self.winding.offsets[count]),
                        Int(self.winding.offsets[count + 1]),
                        sum,
                        compensation,
                    )
                    var candidates = self.winding.bvh.query_aabb(
                        AABB[Self.frame](point, point)
                    )
                    var component = UInt32(0)
                    while candidates.next(component):
                        self._accumulate_packed_angles(
                            point,
                            Int(self.winding.offsets[Int(component)]),
                            Int(self.winding.offsets[Int(component) + 1]),
                            sum,
                            compensation,
                        )
                else:
                    self._accumulate_packed_angles(
                        point,
                        0,
                        len(self.packed_triangles) // (9 * width),
                        sum,
                        compensation,
                    )
                return sum.reduce_add() * INV_4_PI
        var node = Int(root)
        var end = Int(self.bvh.nodes[node * 2])
        while node < end:
            var a = Vec3[.float64, Self.frame, width](
                SIMD[.float64, width](0.0),
                SIMD[.float64, width](0.0),
                SIMD[.float64, width](0.0),
            )
            var b = a
            var c = a
            comptime for lane in range(width):
                var face = EMPTY_LANE
                while node < end and face == EMPTY_LANE:
                    face = self.bvh.nodes[node * 2 + 1]
                    node += 1
                if face != EMPTY_LANE:
                    var av = self.face_vertex(face, 0)
                    var bv = self.face_vertex(face, 1)
                    var cv = self.face_vertex(face, 2)
                    a.x[lane] = Float64(av.x)
                    a.y[lane] = Float64(av.y)
                    a.z[lane] = Float64(av.z)
                    b.x[lane] = Float64(bv.x)
                    b.y[lane] = Float64(bv.y)
                    b.z[lane] = Float64(bv.z)
                    c.x[lane] = Float64(cv.x)
                    c.y[lane] = Float64(cv.y)
                    c.z[lane] = Float64(cv.z)
            var angle = solid_angle(point, a, b, c)
            var adjusted = angle - compensation
            var updated = sum + adjusted
            compensation = (updated - sum) - adjusted
            sum = updated
        return sum.reduce_add() * INV_4_PI

    def query_point_sign_winding_number(
        self,
        point: Point3f32[Self.frame],
        max_dist: Float32 = f32_max,
        root: UInt32 = 0,
        threshold: Float32 = 0.5,
    ) -> MeshPointResult[Self.frame]:
        var result = self.query_point_no_sign(point, max_dist, root)
        if not isfinite(threshold) or threshold < 0.0:
            return MeshPointResult[Self.frame].miss()
        if result.result:
            result.sign = (
                -1.0 if abs(self.winding_number(point, root))
                > Float64(threshold) else 1.0
            )
            if result.distance == 0.0:
                result.sign = 0.0
        return result

    @always_inline
    def _component_parity_faces(
        self, ray: Rayf32[Self.frame], begin: UInt32, end: UInt32
    ) -> Tuple[UInt32, Bool]:
        var hits = UInt32(0)
        for slot in range(Int(begin), Int(end)):
            var triangle = _load_mesh_triangle[Self.frame](
                self.points, self.indices, self.device_winding.faces[slot]
            )
            var filtered = ray_triangle_filter(
                ray, triangle[0], triangle[1], triangle[2]
            )
            if filtered == 0:
                continue
            if filtered == 1:
                hits += 1
                continue
            var hit = intersect_triangle(
                ray, triangle[0], triangle[1], triangle[2]
            )
            if hit.result:
                if (
                    hit.u <= 1e-12
                    or hit.v <= 1e-12
                    or 1.0 - hit.u - hit.v <= 1e-12
                ):
                    return (0, True)
                hits += 1
        return (hits, False)

    def _component_parity(self, ray: Rayf32[Self.frame]) -> Tuple[UInt32, Bool]:
        var count = len(self.device_winding.offsets) - 2
        var hits, ambiguous = self._component_parity_faces(
            ray,
            self.device_winding.offsets[count],
            self.device_winding.offsets[count + 1],
        )
        if ambiguous:
            return (0, True)
        var components = self.device_winding.bvh.query_aabb(
            AABB[Self.frame](ray.o, ray.o)
        )
        var component = UInt32(0)
        while components.next(component):
            var result = self._component_parity_faces(
                ray,
                self.device_winding.offsets[Int(component)],
                self.device_winding.offsets[Int(component) + 1],
            )
            if result[1]:
                return (0, True)
            hits += result[0]
        return (hits, False)

    def query_point_sign_parity[
        width: SIMDLength = BATCH_WIDTH
    ](
        self,
        point: Point3f32[Self.frame],
        max_dist: Float32 = f32_max,
        root: UInt32 = 0,
    ) -> MeshPointResult[Self.frame]:
        return self.apply_point_sign_parity[width](
            point, self.query_point_no_sign(point, max_dist, root), root
        )

    def apply_point_sign_parity[
        width: SIMDLength = RAY_COUNT_WIDTH
    ](
        self,
        point: Point3f32[Self.frame],
        var result: MeshPointResult[Self.frame],
        root: UInt32 = 0,
    ) -> MeshPointResult[Self.frame]:
        if result.result and result.distance == 0.0:
            return result
        if result.result:
            var inside_votes = 0
            var valid_votes = 0
            # Face counts at a shared edge/vertex do not define crossing parity.
            # Retry boundary hits with different, noncoplanar directions.
            for sample in range(16):
                var direction = Vec3f32[Self.frame](
                    1.0,
                    0.173 + Float32((sample * 37 + 11) % 101) * 0.013,
                    0.219 + Float32((sample * sample * 19 + 23) % 103) * 0.017,
                )
                var ray = Rayf32[Self.frame](point, direction)
                var votes: Tuple[UInt32, Bool]
                comptime if is_gpu():
                    if root == 0 and len(self.device_winding.offsets) >= 2:
                        # Off a closed surface, components whose bounds exclude
                        # the point contribute an even number of crossings.
                        votes = self._component_parity(ray)
                    else:
                        votes = self._query_ray_hits[1, False, True](ray, root)
                else:
                    votes = self._query_ray_hits[width, False, True](ray, root)
                var count = Int(votes[0])
                var ambiguous = votes[1]
                if not ambiguous:
                    inside_votes += count % 2
                    valid_votes += 1
                    # Two agreeing votes already determine the majority of
                    # three; retry only when the first two disagree.
                    if valid_votes == 3 or (
                        valid_votes == 2 and inside_votes != 1
                    ):
                        break
            if valid_votes > 0:
                result.sign = -1.0 if inside_votes * 2 > valid_votes else 1.0
            else:
                # All rays met boundary features. For the documented oriented
                # closed surface contract, exact winding resolves this case.
                result.sign = (
                    -1.0 if abs(self.winding_number(point, root)) > 0.5 else 1.0
                )
            if result.distance == 0.0:
                result.sign = 0.0
        return result

    def query_point_sign_normal(
        self,
        point: Point3f32[Self.frame],
        max_dist: Float32 = f32_max,
        root: UInt32 = 0,
        epsilon: Float32 = 1.0e-6,
    ) -> MeshPointResult[Self.frame]:
        return self.apply_point_sign_normal(
            point,
            self.query_point_no_sign(point, max_dist, root),
            epsilon,
            root,
        )

    def apply_point_sign_normal(
        self,
        point: Point3f32[Self.frame],
        var result: MeshPointResult[Self.frame],
        epsilon: Float32 = 1e-6,
        root: UInt32 = 0,
    ) -> MeshPointResult[Self.frame]:
        if not isfinite(epsilon) or epsilon < 0.0 or epsilon >= 0.5:
            return MeshPointResult[Self.frame].miss()
        if not result.result:
            return result
        var vertex_corner = -1
        var edge_opposite = -1
        for corner in range(3):
            var weight = (
                result.u if corner
                == 0 else result.v if corner
                == 1 else Float32(1.0 - result.u - result.v)
            )
            if weight >= 1.0 - epsilon:
                vertex_corner = corner
            if weight <= epsilon:
                edge_opposite = corner
        comptime if is_gpu():
            if vertex_corner < 0 and edge_opposite < 0:
                var filtered = normal_sign_filter(
                    point,
                    result.point,
                    self.face_vertex(result.face, 0),
                    self.face_vertex(result.face, 1),
                    self.face_vertex(result.face, 2),
                )
                if filtered != 0:
                    result.sign = Float32(filtered)
                    if result.distance == 0.0:
                        result.sign = 0.0
                    return result
        var n = triangle_normal(
            self.face_vertex(result.face, 0),
            self.face_vertex(result.face, 1),
            self.face_vertex(result.face, 2),
        )
        if vertex_corner >= 0 or edge_opposite >= 0:
            n = Vec3[.float64, Self.frame](0.0)
            var target_a = self.indices[
                Int(result.face) * 3
                + (
                    vertex_corner if vertex_corner
                    >= 0 else (edge_opposite + 1) % 3
                )
            ]
            var target_b = (
                self.indices[
                    Int(result.face) * 3 + (edge_opposite + 2) % 3
                ] if edge_opposite
                >= 0 else target_a
            )
            var faces = self._all_faces(root)
            var face = UInt32(0)
            while faces.next(face):
                var matched_a = -1
                var matched_b = False
                for corner in range(3):
                    var index = self.indices[Int(face) * 3 + corner]
                    if index == target_a:
                        matched_a = corner
                    matched_b |= index == target_b
                if matched_a < 0 or (vertex_corner < 0 and not matched_b):
                    continue
                var normal = triangle_normal(
                    self.face_vertex(face, 0),
                    self.face_vertex(face, 1),
                    self.face_vertex(face, 2),
                )
                var weight = Float64(1.0)
                if vertex_corner >= 0:
                    var center = vector64(self.face_vertex(face, matched_a))
                    var left = (
                        vector64(self.face_vertex(face, (matched_a + 1) % 3))
                        - center
                    )
                    var right = (
                        vector64(self.face_vertex(face, (matched_a + 2) % 3))
                        - center
                    )
                    var perpendicular = cross(left, right)
                    weight = angle_atan2(
                        sqrt(dot(perpendicular, perpendicular)),
                        dot(left, right),
                    )
                n += normal * weight
        var delta = vector64(point) - vector64(result.point)
        result.sign = -1.0 if dot(n, delta) < 0.0 else 1.0
        if result.distance == 0.0:
            result.sign = 0.0
        return result

    def query_point(
        self,
        point: Point3f32[Self.frame],
        max_dist: Float32 = f32_max,
        root: UInt32 = 0,
    ) -> MeshPointResult[Self.frame]:
        return self.query_point_sign_normal(point, max_dist, root)
