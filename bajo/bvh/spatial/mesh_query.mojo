"""Borrowed indexed mesh queries executable on CPU and GPU.

Barycentrics are weights of vertices A and B; C has weight 1-u-v. Ray intervals
are inclusive. Counts report intersected faces, including multiple faces on a
shared edge. Candidate iteration order is unspecified; nearest/furthest ties
choose the lowest original face ID. Sign classification assumes an oriented,
closed surface; open/nonmanifold surfaces do not define a universal inside.
"""

from std.math import abs, sqrt
from std.utils.numerics import isfinite
from bajo.core import AABB, Frame, Point3f32, Rayf32, Vec3f32, Vec3, cross, dot
from bajo.bvh.constants import EMPTY_LANE, f32_max
from .query import BoundsQueryView, BoundsQuery
from .geometry import (
    project_triangle,
    intersect_triangle,
    triangle_normal,
    solid_angle,
    vector64,
    angle_atan2,
)


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
    return Point3f32[frame](points[base], points[base + 1], points[base + 2])


@fieldwise_init
struct MeshSphereQuery[origin: ImmOrigin, frame: Frame](Copyable):
    var candidates: BoundsQuery[Self.origin, Self.frame]
    var points: ImmSpan[Float32, Self.origin]
    var indices: ImmSpan[UInt32, Self.origin]
    var center: Point3f32[Self.frame]
    var radius: Float32

    def next(mut self, mut face: UInt32) -> Bool:
        var candidate = UInt32(0)
        while self.candidates.next(candidate):
            var base = Int(candidate) * 3
            var a = load_mesh_point[Self.frame](self.points, self.indices[base])
            var b = load_mesh_point[Self.frame](
                self.points, self.indices[base + 1]
            )
            var c = load_mesh_point[Self.frame](
                self.points, self.indices[base + 2]
            )
            if project_triangle(self.center, a, b, c).distance2 <= Float64(
                self.radius
            ) * Float64(self.radius):
                face = candidate
                return True
        return False


@fieldwise_init
struct MeshQueryView[origin: ImmOrigin, frame: Frame](Copyable):
    var points: ImmSpan[Float32, Self.origin]
    var indices: ImmSpan[UInt32, Self.origin]
    var velocities: ImmSpan[Float32, Self.origin]
    var bvh: BoundsQueryView[Self.origin, Self.frame]

    def vertex(self, index: UInt32) -> Point3f32[Self.frame]:
        return load_mesh_point[Self.frame](self.points, index)

    def face_vertex(self, face: UInt32, corner: Int) -> Point3f32[Self.frame]:
        return self.vertex(self.indices[Int(face) * 3 + corner])

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
        var n = triangle_normal(
            self.face_vertex(face, 0),
            self.face_vertex(face, 1),
            self.face_vertex(face, 2),
        )
        return Vec3f32[Self.frame](Float32(n.x), Float32(n.y), Float32(n.z))

    def query_aabb(
        self, box: AABB[Self.frame], root: UInt32 = 0
    ) -> BoundsQuery[Self.origin, Self.frame]:
        """Enumerate faces whose bounds overlap; this is broad phase."""
        return self.bvh.query_aabb(box, root)

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

    def query_ray(
        self, ray: Rayf32[Self.frame], root: UInt32 = 0
    ) -> MeshRayResult[Self.frame]:
        var result = MeshRayResult[Self.frame].miss()
        var candidates = self.bvh.query_ray(ray, root)
        var face = UInt32(0)
        while candidates.next(face):
            var a = self.face_vertex(face, 0)
            var b = self.face_vertex(face, 1)
            var c = self.face_vertex(face, 2)
            var hit = intersect_triangle(ray, a, b, c)
            if hit.result and (
                not result.result
                or hit.t < result.t
                or (hit.t == result.t and face < result.face)
            ):
                var n = triangle_normal(a, b, c)
                var direction = Vec3[.float64, Self.frame](
                    Float64(ray.d.x), Float64(ray.d.y), Float64(ray.d.z)
                )
                result = MeshRayResult(
                    True,
                    face,
                    Float32(hit.u),
                    Float32(hit.v),
                    hit.t,
                    Vec3f32[Self.frame](
                        Float32(n.x), Float32(n.y), Float32(n.z)
                    ),
                    dot(n, direction) < 0.0,
                )
        return result

    def query_ray_anyhit(
        self, ray: Rayf32[Self.frame], root: UInt32 = 0
    ) -> Bool:
        var candidates = self.bvh.query_ray(ray, root)
        var face = UInt32(0)
        while candidates.next(face):
            if intersect_triangle(
                ray,
                self.face_vertex(face, 0),
                self.face_vertex(face, 1),
                self.face_vertex(face, 2),
            ).result:
                return True
        return False

    def query_ray_count_intersections(
        self, ray: Rayf32[Self.frame], root: UInt32 = 0
    ) -> UInt32:
        var count = UInt32(0)
        var candidates = self.bvh.query_ray(ray, root)
        var face = UInt32(0)
        while candidates.next(face):
            if intersect_triangle(
                ray,
                self.face_vertex(face, 0),
                self.face_vertex(face, 1),
                self.face_vertex(face, 2),
            ).result:
                count += 1
        return count

    def query_point_no_sign(
        self,
        point: Point3f32[Self.frame],
        max_dist: Float32 = f32_max,
        root: UInt32 = 0,
    ) -> MeshPointResult[Self.frame]:
        var result = MeshPointResult[Self.frame].miss()
        var best2 = Float64(max_dist) * Float64(max_dist)
        var candidates = self.bvh.query_sphere(point, max_dist, root)
        var face = UInt32(0)
        while candidates.next(face):
            var hit = project_triangle(
                point,
                self.face_vertex(face, 0),
                self.face_vertex(face, 1),
                self.face_vertex(face, 2),
            )
            if hit.distance2 <= best2 and (
                not result.result or hit.distance2 < best2 or face < result.face
            ):
                best2 = hit.distance2
                result = MeshPointResult(
                    True,
                    face,
                    Float32(hit.u),
                    Float32(hit.v),
                    sqrt(best2),
                    Point3f32[Self.frame](
                        Float32(hit.point.x),
                        Float32(hit.point.y),
                        Float32(hit.point.z),
                    ),
                    0.0,
                )
        return result

    def _all_faces(self, root: UInt32) -> BoundsQuery[Self.origin, Self.frame]:
        return self.query_aabb(
            AABB(
                Point3f32[Self.frame](-f32_max), Point3f32[Self.frame](f32_max)
            ),
            root,
        )

    def query_furthest_point_no_sign(
        self,
        point: Point3f32[Self.frame],
        min_dist: Float32 = 0.0,
        root: UInt32 = 0,
    ) -> MeshPointResult[Self.frame]:
        var result = MeshPointResult[Self.frame].miss()
        if not point.is_finite()[0] or not isfinite(min_dist) or min_dist < 0.0:
            return result
        var best2 = Float64(min_dist) * Float64(min_dist)
        var faces = self._all_faces(root)
        var face = UInt32(0)
        while faces.next(face):
            for corner in range(3):
                var vertex = self.face_vertex(face, corner)
                var delta = vector64(vertex) - vector64(point)
                var distance2 = dot(delta, delta)
                if distance2 >= best2 and (
                    not result.result or distance2 > best2 or face < result.face
                ):
                    best2 = distance2
                    result = MeshPointResult(
                        True,
                        face,
                        Float32(1.0 if corner == 0 else 0.0),
                        Float32(1.0 if corner == 1 else 0.0),
                        sqrt(best2),
                        vertex,
                        0.0,
                    )
        return result

    def winding_number(
        self, point: Point3f32[Self.frame], root: UInt32 = 0
    ) -> Float64:
        if not point.is_finite()[0]:
            return 0.0
        var sum = Float64(0.0)
        var compensation = Float64(0.0)
        var faces = self._all_faces(root)
        var face = UInt32(0)
        while faces.next(face):
            var angle = solid_angle(
                point,
                self.face_vertex(face, 0),
                self.face_vertex(face, 1),
                self.face_vertex(face, 2),
            )
            var adjusted = angle - compensation
            var updated = sum + adjusted
            compensation = (updated - sum) - adjusted
            sum = updated
        return sum / (4.0 * Float64(3.141592653589793))

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

    def query_point_sign_parity(
        self,
        point: Point3f32[Self.frame],
        max_dist: Float32 = f32_max,
        root: UInt32 = 0,
    ) -> MeshPointResult[Self.frame]:
        var result = self.query_point_no_sign(point, max_dist, root)
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
                var candidates = self.bvh.query_ray(ray, root)
                var face = UInt32(0)
                var count = 0
                var ambiguous = False
                while candidates.next(face):
                    var hit = intersect_triangle(
                        ray,
                        self.face_vertex(face, 0),
                        self.face_vertex(face, 1),
                        self.face_vertex(face, 2),
                    )
                    if hit.result:
                        count += 1
                        ambiguous |= (
                            hit.u <= 1e-12
                            or hit.v <= 1e-12
                            or 1.0 - hit.u - hit.v <= 1e-12
                        )
                if not ambiguous:
                    inside_votes += count % 2
                    valid_votes += 1
                    if valid_votes == 3:
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
        var result = self.query_point_no_sign(point, max_dist, root)
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
