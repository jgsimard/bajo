"""Scene light records and sampling distributions."""

from bajo.core import Point3f32
from bajo.rt.material_types import PrimitiveId, SurfaceId
from bajo.rt.render_types import Color


struct LightRecord(Copyable, Writable):
    """Finalized world-space emitter geometry and power-distribution entry."""

    var primitive: PrimitiveId
    var surface: SurfaceId[1]
    var weight: Float32
    var p0: Point3f32[.WORLD]
    var p1: Point3f32[.WORLD]
    var p2: Point3f32[.WORLD]
    var radius: Float32

    def __init__(
        out self,
        primitive: PrimitiveId,
        surface: SurfaceId[1],
        weight: Float32,
        p0: Point3f32[.WORLD],
        p1: Point3f32[.WORLD],
        p2: Point3f32[.WORLD],
        radius: Float32,
    ):
        self.primitive = primitive.copy()
        self.surface = surface.copy()
        self.weight = weight
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.radius = radius

    @staticmethod
    def triangle(
        primitive: PrimitiveId,
        surface: SurfaceId[1],
        weight: Float32,
        p0: Point3f32[.WORLD],
        p1: Point3f32[.WORLD],
        p2: Point3f32[.WORLD],
    ) -> Self:
        return Self(primitive, surface, weight, p0, p1, p2, 0.0)

    @staticmethod
    def sphere(
        primitive: PrimitiveId,
        surface: SurfaceId[1],
        weight: Float32,
        center: Point3f32[.WORLD],
        radius: Float32,
    ) -> Self:
        return Self(
            primitive,
            surface,
            weight,
            center,
            Point3f32[.WORLD](0.0),
            Point3f32[.WORLD](0.0),
            radius,
        )


struct LightStore:
    var records: List[LightRecord]
    var total_weight: Float32
    var alias_probabilities: List[Float32]
    var alias_indices: List[UInt32]

    def __init__(out self):
        self.records = List[LightRecord]()
        self.total_weight = 0.0
        self.alias_probabilities = List[Float32]()
        self.alias_indices = List[UInt32]()

    @always_inline
    def append(mut self, var record: LightRecord):
        self.total_weight += record.weight
        self.records.append(record^)

    def build_alias_table(mut self):
        """Build a reusable Walker-Vose power distribution in linear time."""
        var count = len(self.records)
        self.alias_probabilities = List[Float32](length=count, fill=1.0)
        self.alias_indices = List[UInt32](length=count, fill=UInt32(0))
        if count == 0 or self.total_weight <= 0.0:
            return

        var scaled = List[Float32](length=count, fill=0.0)
        var small = List[Int](capacity=count)
        var large = List[Int](capacity=count)
        for i in range(count):
            self.alias_indices[i] = UInt32(i)
            scaled[i] = (
                self.records[i].weight * Float32(count) / self.total_weight
            )
            if scaled[i] < 1.0:
                small.append(i)
            else:
                large.append(i)

        while len(small) > 0 and len(large) > 0:
            var small_idx = small.pop()
            var large_idx = large.pop()
            self.alias_probabilities[small_idx] = scaled[small_idx]
            self.alias_indices[small_idx] = UInt32(large_idx)
            scaled[large_idx] = scaled[large_idx] + scaled[small_idx] - 1.0
            if scaled[large_idx] < 1.0:
                small.append(large_idx)
            else:
                large.append(large_idx)

        while len(small) > 0:
            var idx = small.pop()
            self.alias_probabilities[idx] = 1.0
            self.alias_indices[idx] = UInt32(idx)
        while len(large) > 0:
            var idx = large.pop()
            self.alias_probabilities[idx] = 1.0
            self.alias_indices[idx] = UInt32(idx)


def _light_importance(radiance: Color) -> Float32:
    return max((radiance.x + radiance.y + radiance.z) / 3.0, 0.0)
