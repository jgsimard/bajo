"""Hit records and backend-neutral shading values."""

from bajo.core import Point3, Point3f32, Ray, Vec3, Vec3f32
from bajo.rt.material_types import PrimitiveId, SurfaceId


@fieldwise_init
struct HitRecord(Copyable, Writable):
    var primitive: PrimitiveId
    var p: Point3f32[.WORLD]
    var normal: Vec3f32[.WORLD]
    var surface: SurfaceId[1]
    var t: Float32
    var front_face: Bool


@fieldwise_init
struct SurfaceHit[length: SIMDLength = 1](Copyable, Writable):
    """Renderer hit without primitive identity or position."""

    var normal: Vec3[.float32, .WORLD, Self.length]
    var surface: SurfaceId[Self.length]
    var uv_u: SIMD[.float32, Self.length]
    var uv_v: SIMD[.float32, Self.length]
    var t: SIMD[.float32, Self.length]
    var front_face: SIMD[.bool, Self.length]
    var hit: SIMD[.bool, Self.length]

    def __init__(out self, t_max: SIMD[.float32, Self.length]):
        self.normal = Vec3[.float32, .WORLD, Self.length](0.0)
        self.surface = SurfaceId[Self.length](SIMD[.uint32, Self.length](0))
        self.uv_u = 0.0
        self.uv_v = 0.0
        self.t = t_max
        self.front_face = SIMD[.bool, Self.length](fill=True)
        self.hit = SIMD[.bool, Self.length](fill=False)

    def __init__(
        out self,
        normal: Vec3[.float32, .WORLD, Self.length],
        surface: SurfaceId[Self.length],
        t: SIMD[.float32, Self.length],
        front_face: SIMD[.bool, Self.length],
        hit: SIMD[.bool, Self.length],
    ):
        self.normal = normal
        self.surface = surface.copy()
        self.uv_u = 0.0
        self.uv_v = 0.0
        self.t = t
        self.front_face = front_face
        self.hit = hit

    @always_inline
    def get(self, lane: Int) -> SurfaceHit[1]:
        return SurfaceHit[1](
            Vec3f32[.WORLD](
                self.normal.x[lane],
                self.normal.y[lane],
                self.normal.z[lane],
            ),
            self.surface.get(lane),
            self.uv_u[lane],
            self.uv_v[lane],
            self.t[lane],
            self.front_face[lane],
            self.hit[lane],
        )


struct ShadingPoint[length: SIMDLength = 1](Copyable, Writable):
    var p: Point3[.float32, .WORLD, Self.length]
    var normal: Vec3[.float32, .WORLD, Self.length]
    var front_face: SIMD[.bool, Self.length]
    var uv_u: SIMD[.float32, Self.length]
    var uv_v: SIMD[.float32, Self.length]

    def __init__(
        out self,
        p: Point3[.float32, .WORLD, Self.length],
        normal: Vec3[.float32, .WORLD, Self.length],
        front_face: SIMD[.bool, Self.length],
        uv_u: SIMD[.float32, Self.length] = 0.0,
        uv_v: SIMD[.float32, Self.length] = 0.0,
    ):
        self.p = p
        self.normal = normal
        self.front_face = front_face
        self.uv_u = uv_u
        self.uv_v = uv_v

    @staticmethod
    def from_hit(
        ray: Ray[.float32, .WORLD, Self.length], hit: SurfaceHit[Self.length]
    ) -> Self:
        return Self(
            ray.at(hit.t),
            hit.normal,
            hit.front_face,
            hit.uv_u,
            hit.uv_v,
        )


@fieldwise_init
struct BsdfSample[length: SIMDLength = 1](Copyable, Writable):
    """Sampled direction and throughput/PDF metadata."""

    var direction: Vec3[.float32, .WORLD, Self.length]
    var weight: Vec3[.float32, .WORLD, Self.length]
    var pdf: SIMD[.float32, Self.length]
    var delta: SIMD[.bool, Self.length]
    var ok: SIMD[.bool, Self.length]


@fieldwise_init
struct BsdfEvaluation[length: SIMDLength = 1](Copyable, Writable):
    """BSDF value and solid-angle PDF."""

    var value: Vec3[.float32, .WORLD, Self.length]
    var pdf: SIMD[.float32, Self.length]
    var delta: SIMD[.bool, Self.length]
