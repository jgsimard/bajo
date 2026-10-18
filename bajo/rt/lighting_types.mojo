"""Scene light records and sampling distributions."""

from bajo.core import Point3f32, Vec3
from bajo.rt.material_types import PrimitiveId, SurfaceId


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
        var weights = List[Float32](capacity=len(self.records))
        for record in self.records:
            weights.append(record.weight)
        _build_alias_distribution(
            Span(weights),
            Float64(self.total_weight),
            self.alias_probabilities,
            self.alias_indices,
        )


def _build_alias_distribution(
    weights: Span[Float32, _],
    total: Float64,
    mut probabilities: List[Float32],
    mut indices: List[UInt32],
):
    """Shared Walker-Vose construction for finite emitters and image lights.

    Float64 workspace avoids overflow during normalization and cancellation
    while balancing columns. Zero-total distributions stay empty.
    """
    var count = len(weights)
    probabilities = List[Float32]()
    indices = List[UInt32]()
    if count == 0 or total <= 0.0:
        return
    probabilities = List[Float32](length=count, fill=1.0)
    indices = List[UInt32](
        length=count, fill_with=lambda (i: Int) -> UInt32: UInt32(i)
    )
    var scaled = List[Float64](length=count, fill=0.0)
    var small = List[Int](capacity=count)
    var large = List[Int](capacity=count)
    for i in range(count):
        scaled[i] = Float64(weights[i]) / total * Float64(count)
        if scaled[i] < 1.0:
            small.append(i)
        else:
            large.append(i)
    while len(small) > 0 and len(large) > 0:
        var small_idx = small.pop()
        var large_idx = large.pop()
        probabilities[small_idx] = Float32(scaled[small_idx]).clamp(0.0, 1.0)
        indices[small_idx] = UInt32(large_idx)
        scaled[large_idx] = scaled[large_idx] + scaled[small_idx] - 1.0
        if scaled[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)
    # Unpaired columns already have probability one and alias themselves.


@always_inline
def _light_importance[
    length: SIMDLength = 1
](radiance: Vec3[.float32, .WORLD, length]) -> SIMD[.float32, length]:
    return max((radiance.x + radiance.y + radiance.z) / 3.0, 0.0)
