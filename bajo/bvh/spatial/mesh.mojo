"""Owning indexed mesh for shared CPU/GPU spatial queries."""

from bajo.core import AABB, Frame, Point3f32, Vec3f32
from .bounds import SpatialBoundsBvh
from .query import BoundsQueryView
from .mesh_query import MeshQueryView


struct SpatialMesh[frame: Frame = .WORLD](Copyable):
    """Snapshot indexed geometry; update positions without changing topology.

    Indices are three original vertex IDs per face. Optional groups are one
    label per face; optional velocities are one vector per vertex. Views borrow
    the owner and must be discarded before updates. Published buffers retain
    their addresses through refit/rebuild and velocity updates.
    """

    var points: List[Float32]
    var indices: List[UInt32]
    var velocities: List[Float32]
    var bvh: SpatialBoundsBvh[Self.frame]

    def __init__(
        out self,
        points: ImmSpan[Point3f32[Self.frame], _],
        indices: ImmSpan[UInt32, _],
    ) raises:
        self = Self(
            points, indices, List[UInt32](), List[Vec3f32[Self.frame]]()
        )

    def __init__(
        out self,
        points: ImmSpan[Point3f32[Self.frame], _],
        indices: ImmSpan[UInt32, _],
        groups: ImmSpan[UInt32, _],
    ) raises:
        self = Self(points, indices, groups, List[Vec3f32[Self.frame]]())

    def __init__(
        out self,
        points: ImmSpan[Point3f32[Self.frame], _],
        indices: ImmSpan[UInt32, _],
        groups: ImmSpan[UInt32, _],
        velocities: ImmSpan[Vec3f32[Self.frame], _],
    ) raises:
        if len(indices) % 3 != 0:
            raise Error("Mesh indices must contain three vertex IDs per face")
        if len(velocities) != 0 and len(velocities) != len(points):
            raise Error("Mesh velocity count must match vertex count")
        for point in points:
            if not point.is_finite()[0]:
                raise Error("Mesh vertices must be finite")
        for velocity in velocities:
            if not velocity.is_finite()[0]:
                raise Error("Mesh velocities must be finite")
        for index in indices:
            if Int(index) >= len(points):
                raise Error("Mesh vertex index is out of range")
        self.points = List[Float32](capacity=len(points) * 3)
        self.velocities = List[Float32](capacity=len(points) * 3)
        self.indices = List[UInt32](capacity=len(indices))
        for i in range(len(points)):
            self.points.extend([points[i].x, points[i].y, points[i].z])
            var velocity = Vec3f32[Self.frame](0.0)
            if len(velocities) > 0:
                velocity = velocities[i]
            self.velocities.extend([velocity.x, velocity.y, velocity.z])
        for index in indices:
            self.indices.append(index)
        self.bvh = SpatialBoundsBvh(
            Self._face_bounds(points, self.indices), groups
        )

    @staticmethod
    def _face_bounds(
        points: ImmSpan[Point3f32[Self.frame], _], indices: ImmSpan[UInt32, _]
    ) -> List[AABB[Self.frame]]:
        var boxes = List[AABB[Self.frame]](capacity=len(indices) // 3)
        for face in range(len(indices) // 3):
            var box = AABB[Self.frame].invalid()
            for corner in range(3):
                box = AABB[Self.frame].merge(
                    box,
                    AABB[Self.frame].point(
                        points[Int(indices[face * 3 + corner])]
                    ),
                )
            boxes.append(box)
        return boxes^

    def view(self) -> MeshQueryView[origin_of(self), Self.frame]:
        return MeshQueryView[origin_of(self), Self.frame](
            ImmSpan(
                unsafe_ptr=self.points.unsafe_ptr().unsafe_origin_cast[
                    origin_of(self)
                ](),
                length=len(self.points),
            ),
            ImmSpan(
                unsafe_ptr=self.indices.unsafe_ptr().unsafe_origin_cast[
                    origin_of(self)
                ](),
                length=len(self.indices),
            ),
            ImmSpan(
                unsafe_ptr=self.velocities.unsafe_ptr().unsafe_origin_cast[
                    origin_of(self)
                ](),
                length=len(self.velocities),
            ),
            BoundsQueryView[origin_of(self), Self.frame](
                ImmSpan(
                    unsafe_ptr=self.bvh.bounds.unsafe_ptr().unsafe_origin_cast[
                        origin_of(self)
                    ](),
                    length=len(self.bvh.bounds),
                ),
                ImmSpan(
                    unsafe_ptr=self.bvh.nodes.unsafe_ptr().unsafe_origin_cast[
                        origin_of(self)
                    ](),
                    length=len(self.bvh.nodes),
                ),
            ),
        )

    def group_root(self, group: UInt32) -> UInt32:
        return self.bvh.group_root(group)

    def _validate_points(
        self, points: ImmSpan[Point3f32[Self.frame], _]
    ) raises:
        if len(points) * 3 != len(self.points):
            raise Error("Mesh update requires unchanged vertex count")
        for point in points:
            if not point.is_finite()[0]:
                raise Error("Mesh vertices must be finite")

    def _store_points(mut self, points: ImmSpan[Point3f32[Self.frame], _]):
        for i in range(len(points)):
            self.points[i * 3] = points[i].x
            self.points[i * 3 + 1] = points[i].y
            self.points[i * 3 + 2] = points[i].z

    def refit(mut self, points: ImmSpan[Point3f32[Self.frame], _]) raises:
        self._validate_points(points)
        self.bvh.refit(Self._face_bounds(points, self.indices))
        self._store_points(points)

    def rebuild(mut self, points: ImmSpan[Point3f32[Self.frame], _]) raises:
        self._validate_points(points)
        self.bvh.rebuild(Self._face_bounds(points, self.indices))
        self._store_points(points)

    def set_velocities(
        mut self, velocities: ImmSpan[Vec3f32[Self.frame], _]
    ) raises:
        """Replace velocities in place; an empty span resets them to zero."""
        if len(velocities) != 0 and len(velocities) * 3 != len(self.points):
            raise Error("Mesh velocity count must match vertex count")
        for velocity in velocities:
            if not velocity.is_finite()[0]:
                raise Error("Mesh velocities must be finite")
        for i in range(len(self.points) // 3):
            var velocity = Vec3f32[Self.frame](0.0)
            if len(velocities) > 0:
                velocity = velocities[i]
            self.velocities[i * 3] = velocity.x
            self.velocities[i * 3 + 1] = velocity.y
            self.velocities[i * 3 + 2] = velocity.z
