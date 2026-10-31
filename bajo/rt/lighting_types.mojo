"""Scene light records and sampling distributions."""

from bajo.core.sampling import build_alias_distribution

from bajo.core import Point3f32, Vec3
from bajo.core.color import Color
from bajo.rt.material_types import PrimitiveId, SurfaceId


@fieldwise_init
struct LightRecord(Copyable, Writable):
    """Finalized world-space emitter geometry and power-distribution entry."""

    var primitive: PrimitiveId
    var surface: SurfaceId[1]
    var weight: Float32
    var p0: Point3f32[.WORLD]
    var p1: Point3f32[.WORLD]
    var p2: Point3f32[.WORLD]
    var radius: Float32
    var uv0_u: Float32
    var uv0_v: Float32
    var uv1_u: Float32
    var uv1_v: Float32
    var uv2_u: Float32
    var uv2_v: Float32
    var radiance: Color
    var flags: UInt32

    @staticmethod
    def triangle(
        primitive: PrimitiveId,
        surface: SurfaceId[1],
        weight: Float32,
        p0: Point3f32[.WORLD],
        p1: Point3f32[.WORLD],
        p2: Point3f32[.WORLD],
        uv0_u: Float32 = 0.0,
        uv0_v: Float32 = 0.0,
        uv1_u: Float32 = 1.0,
        uv1_v: Float32 = 0.0,
        uv2_u: Float32 = 1.0,
        uv2_v: Float32 = 1.0,
        radiance: Color = Color(0.0),
        two_sided: Bool = False,
    ) -> Self:
        return Self(
            primitive.copy(),
            surface.copy(),
            weight,
            p0,
            p1,
            p2,
            0.0,
            uv0_u,
            uv0_v,
            uv1_u,
            uv1_v,
            uv2_u,
            uv2_v,
            radiance,
            UInt32(two_sided),
        )

    @staticmethod
    def sphere(
        primitive: PrimitiveId,
        surface: SurfaceId[1],
        weight: Float32,
        center: Point3f32[.WORLD],
        radius: Float32,
        radiance: Color = Color(0.0),
        two_sided: Bool = False,
    ) -> Self:
        return Self(
            primitive.copy(),
            surface.copy(),
            weight,
            center,
            Point3f32[.WORLD](0.0),
            Point3f32[.WORLD](0.0),
            radius,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            radiance,
            UInt32(two_sided),
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
        build_alias_distribution(
            Span(weights),
            Float64(self.total_weight),
            self.alias_probabilities,
            self.alias_indices,
        )


@always_inline
def _light_importance[
    length: SIMDLength = 1
](radiance: Vec3[.float32, .WORLD, length]) -> SIMD[.float32, length]:
    return max((radiance.x + radiance.y + radiance.z) / 3.0, 0.0)
