"""Owning indexed mesh for shared CPU/GPU spatial queries."""

from std.memory import unsafe_memcpy
from bajo.core import AABB, Frame, Point3f32, Vec3f32
from bajo.bvh.constants import EMPTY_LANE
from .bounds import SpatialBoundsBvh
from .query import BoundsQueryView
from .mesh_query import MeshQueryView
from .winding import WindingLayout


struct SpatialMesh[frame: Frame = .WORLD, storage_width: SIMDLength = 8](
    Copyable
):
    """Snapshot indexed geometry; update positions without changing topology.

    Indices are three original vertex IDs per face. Optional groups are one
    label per face; optional velocities are one vector per vertex. Views borrow
    the owner and must be discarded before updates. Published buffers retain
    their addresses through refit/rebuild and velocity updates.
    cache_winding=False skips the optional winding layout/packing; exact queries
    still work through the indexed Float64 fallback.
    """

    var points: List[Float32]
    var indices: List[UInt32]
    var velocities: List[Float32]
    var bvh: SpatialBoundsBvh[Self.frame]
    var packed_triangles: List[Float32]
    var winding: WindingLayout[Self.frame, Self.storage_width]

    def __init__(
        out self,
        points: ImmSpan[Point3f32[Self.frame], _],
        indices: ImmSpan[UInt32, _],
        cache_winding: Bool = True,
    ) raises:
        self = Self(
            points,
            indices,
            List[UInt32](),
            List[Vec3f32[Self.frame]](),
            cache_winding,
        )

    def __init__(
        out self,
        points: ImmSpan[Point3f32[Self.frame], _],
        indices: ImmSpan[UInt32, _],
        groups: ImmSpan[UInt32, _],
        cache_winding: Bool = True,
    ) raises:
        self = Self(
            points, indices, groups, List[Vec3f32[Self.frame]](), cache_winding
        )

    def __init__(
        out self,
        points: ImmSpan[Point3f32[Self.frame], _],
        indices: ImmSpan[UInt32, _],
        groups: ImmSpan[UInt32, _],
        velocities: ImmSpan[Vec3f32[Self.frame], _],
        cache_winding: Bool = True,
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
        self.points = List[Float32](length=len(points) * 3, fill=0.0)
        self.velocities = List[Float32](length=len(points) * 3, fill=0.0)
        self.indices = List[UInt32](length=len(indices), fill=0)
        var positions = self.points.unsafe_ptr()
        var speeds = self.velocities.unsafe_ptr()
        for i in range(len(points)):
            comptime for axis in range(3):
                positions.unsafe_store(i * 3 + axis, points[i][axis])
        if len(velocities) > 0:
            for i in range(len(velocities)):
                comptime for axis in range(3):
                    speeds.unsafe_store(i * 3 + axis, velocities[i][axis])
        if len(indices) > 0:
            unsafe_memcpy(
                dest=self.indices.unsafe_ptr(),
                src=indices.unsafe_ptr(),
                count=len(indices),
            )
        self.bvh = SpatialBoundsBvh(
            Self._face_bounds_validated(points, self.indices), groups
        )

        comptime assert Self.storage_width > 0
        self.winding = WindingLayout[Self.frame, Self.storage_width](
            List[UInt32](), List[AABB[Self.frame]]()
        )
        self.packed_triangles = List[Float32]()
        if cache_winding:
            self._ensure_winding_cache()

    def _ensure_winding_cache(mut self) raises:
        if len(self.winding.face_components) == len(self.indices) // 3:
            return
        self.winding = WindingLayout[Self.frame, Self.storage_width](
            self.indices, self.bvh.item_bounds
        )
        self.packed_triangles = List[Float32](
            length=Int(self.winding.offsets[len(self.winding.offsets) - 1])
            * Self.storage_width
            * 9,
            fill=0.0,
        )
        self._pack_triangles()

    def _pack_triangles[update_bounds: Bool = False](mut self):
        var has_cache = len(self.packed_triangles) > 0
        comptime if not update_bounds:
            if not has_cache:
                return
        var points = self.points.unsafe_ptr()
        var indices = self.indices.unsafe_ptr()
        var packed = self.packed_triangles.unsafe_ptr()
        var boxes = self.bvh.item_bounds.unsafe_ptr().unsafe_bitcast[Float32]()
        var slots = self.winding.face_slots.unsafe_ptr()
        var components = self.winding.face_components.unsafe_ptr()
        var winding_boxes = self.winding.bvh.item_bounds.unsafe_ptr()
        comptime if update_bounds:
            for component in range(len(self.winding.bvh.item_bounds)):
                winding_boxes[unsafe_offset=component] = AABB[
                    Self.frame
                ].invalid()
        for face in range(len(self.indices) // 3):
            var av = Int(indices.unsafe_load(face * 3)) * 3
            var bv = Int(indices.unsafe_load(face * 3 + 1)) * 3
            var cv = Int(indices.unsafe_load(face * 3 + 2)) * 3
            var a = Point3f32[Self.frame](points.unsafe_offset(av))
            var b = Point3f32[Self.frame](points.unsafe_offset(bv))
            var c = Point3f32[Self.frame](points.unsafe_offset(cv))
            if has_cache:
                var slot = Int(slots.unsafe_load(face))
                var block = slot // Self.storage_width
                var lane = slot % Self.storage_width
                comptime for corner in range(3):
                    var vertex = a if corner == 0 else b if corner == 1 else c
                    comptime for axis in range(3):
                        packed.unsafe_store(
                            block * 9 * Self.storage_width
                            + (corner * 3 + axis) * Self.storage_width
                            + lane,
                            vertex[axis],
                        )
            comptime if update_bounds:
                var box = AABB[Self.frame](a, b, c)
                comptime for axis in range(3):
                    boxes.unsafe_store(face * 6 + axis, box._min[axis])
                    boxes.unsafe_store(face * 6 + axis + 3, box._max[axis])
                if has_cache:
                    var component = components.unsafe_load(face)
                    if component != EMPTY_LANE:
                        winding_boxes[unsafe_offset=Int(component)] = AABB[
                            Self.frame
                        ].merge(
                            winding_boxes[unsafe_offset=Int(component)], box
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

    @staticmethod
    def _face_bounds_validated(
        points: ImmSpan[Point3f32[Self.frame], _], indices: ImmSpan[UInt32, _]
    ) -> List[AABB[Self.frame]]:
        # Construction already validated all immutable input vertex indexes.
        var count = len(indices) // 3
        var boxes = List[AABB[Self.frame]](capacity=count)
        boxes.resize(unsafe_uninit_length=count)
        for face in range(count):
            var box = AABB[Self.frame].invalid()
            comptime for corner in range(3):
                box = AABB[Self.frame].merge(
                    box,
                    AABB[Self.frame].point(
                        points.unsafe_get(
                            Int(indices.unsafe_get(face * 3 + corner))
                        )
                    ),
                )
            boxes.unsafe_get(face) = box
        return boxes^

    def view(
        self,
    ) -> MeshQueryView[origin_of(self), Self.frame, Self.storage_width]:
        return MeshQueryView[origin_of(self), Self.frame, Self.storage_width](
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
            ImmSpan(
                unsafe_ptr=self.packed_triangles.unsafe_ptr().unsafe_origin_cast[
                    origin_of(self)
                ](),
                length=len(self.packed_triangles),
            ),
            self.winding._view[origin_of(self)](),
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
        self._store_points(points)
        self._pack_triangles[True]()
        self.bvh._refit_nodes()
        self.winding.bvh._refit_nodes()

    def rebuild(mut self, points: ImmSpan[Point3f32[Self.frame], _]) raises:
        self._validate_points(points)
        self.bvh.rebuild(Self._face_bounds(points, self.indices))
        self._store_points(points)
        self._pack_triangles[True]()
        self.winding.bvh._refit_nodes()

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
