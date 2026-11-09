"""Hit records and backend-neutral shading values."""

from bajo.core import Point3, Point3f32, Ray, Vec3, Vec3f32
from bajo.rt.material_types import PrimitiveId, SurfaceId


@fieldwise_init
struct RefractionGeometry[length: SIMDLength](TrivialRegisterPassable):
    var unit_direction: Vec3[.float32, .WORLD, Self.length]
    var incident_cosine: SIMD[.float32, Self.length]
    var transmitted_sine: SIMD[.float32, Self.length]


@fieldwise_init
struct HitRecord(Copyable, Writable):
    var primitive: PrimitiveId
    var p: Point3f32[.WORLD]
    var normal: Vec3f32[.WORLD]
    var surface: SurfaceId[1]
    var t: Float32
    var front_face: Bool


struct SurfaceHit[length: SIMDLength = 1](Copyable, Writable):
    """Renderer hit without primitive identity or position."""

    var normal: Vec3[.float32, .WORLD, Self.length]
    var surface: SurfaceId[Self.length]
    var uv_u: SIMD[.float32, Self.length]
    var uv_v: SIMD[.float32, Self.length]
    var t: SIMD[.float32, Self.length]
    var front_face: SIMD[.bool, Self.length]
    var hit: SIMD[.bool, Self.length]
    var medium_boundary: SIMD[.uint32, Self.length]
    var geometric_normal: Vec3[.float32, .WORLD, Self.length]
    var dpdu: Vec3[.float32, .WORLD, Self.length]
    var emission_visible: SIMD[.bool, Self.length]
    var emission: Vec3[.float32, .WORLD, Self.length]
    var emission_two_sided: SIMD[.bool, Self.length]
    var emission_spectrum: SIMD[.uint32, Self.length]

    def __init__(out self, t_max: SIMD[.float32, Self.length]):
        self = Self(
            Vec3[.float32, .WORLD, Self.length](0.0),
            SurfaceId[Self.length](SIMD[.uint32, Self.length](0)),
            t_max,
            SIMD[.bool, Self.length](fill=True),
            SIMD[.bool, Self.length](fill=False),
        )

    def __init__(
        out self,
        normal: Vec3[.float32, .WORLD, Self.length],
        surface: SurfaceId[Self.length],
        t: SIMD[.float32, Self.length],
        front_face: SIMD[.bool, Self.length],
        hit: SIMD[.bool, Self.length],
    ):
        self = Self(
            normal,
            surface,
            0.0,
            0.0,
            t,
            front_face,
            hit,
            geometric_normal=normal,
        )

    def __init__(
        out self,
        normal: Vec3[.float32, .WORLD, Self.length],
        surface: SurfaceId[Self.length],
        uv_u: SIMD[.float32, Self.length],
        uv_v: SIMD[.float32, Self.length],
        t: SIMD[.float32, Self.length],
        front_face: SIMD[.bool, Self.length],
        hit: SIMD[.bool, Self.length],
        medium_boundary: SIMD[.uint32, Self.length] = 0,
        geometric_normal: Vec3[.float32, .WORLD, Self.length] = Vec3[
            .float32, .WORLD, Self.length
        ](0.0),
        dpdu: Vec3[.float32, .WORLD, Self.length] = Vec3[
            .float32, .WORLD, Self.length
        ](0.0),
        emission_visible: SIMD[.bool, Self.length] = SIMD[.bool, Self.length](
            fill=True
        ),
        emission: Vec3[.float32, .WORLD, Self.length] = Vec3[
            .float32, .WORLD, Self.length
        ](0.0),
        emission_two_sided: SIMD[.bool, Self.length] = SIMD[.bool, Self.length](
            fill=False
        ),
        emission_spectrum: SIMD[.uint32, Self.length] = UInt32.MAX,
    ):
        self.normal = normal
        self.surface = surface.copy()
        self.uv_u = uv_u
        self.uv_v = uv_v
        self.t = t
        self.front_face = front_face
        self.hit = hit
        self.medium_boundary = medium_boundary
        self.geometric_normal = geometric_normal
        self.dpdu = dpdu
        self.emission_visible = emission_visible
        self.emission = emission
        self.emission_two_sided = emission_two_sided
        self.emission_spectrum = emission_spectrum

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
            self.medium_boundary[lane],
            Vec3f32[.WORLD](
                self.geometric_normal.x[lane],
                self.geometric_normal.y[lane],
                self.geometric_normal.z[lane],
            ),
            Vec3f32[.WORLD](
                self.dpdu.x[lane], self.dpdu.y[lane], self.dpdu.z[lane]
            ),
            self.emission_visible[lane],
            Vec3f32[.WORLD](
                self.emission.x[lane],
                self.emission.y[lane],
                self.emission.z[lane],
            ),
            self.emission_two_sided[lane],
            self.emission_spectrum[lane],
        )

    @always_inline
    def set(mut self, lane: Int, value: SurfaceHit[1]):
        self.normal.x[lane] = value.normal.x
        self.normal.y[lane] = value.normal.y
        self.normal.z[lane] = value.normal.z
        self.geometric_normal.x[lane] = value.geometric_normal.x
        self.geometric_normal.y[lane] = value.geometric_normal.y
        self.geometric_normal.z[lane] = value.geometric_normal.z
        self.dpdu.x[lane] = value.dpdu.x
        self.dpdu.y[lane] = value.dpdu.y
        self.dpdu.z[lane] = value.dpdu.z
        self.surface.value[lane] = value.surface.value
        self.uv_u[lane] = value.uv_u
        self.uv_v[lane] = value.uv_v
        self.t[lane] = value.t
        self.front_face[lane] = value.front_face
        self.hit[lane] = value.hit
        self.medium_boundary[lane] = value.medium_boundary
        self.emission_visible[lane] = value.emission_visible
        self.emission.x[lane] = value.emission.x
        self.emission.y[lane] = value.emission.y
        self.emission.z[lane] = value.emission.z
        self.emission_two_sided[lane] = value.emission_two_sided
        self.emission_spectrum[lane] = value.emission_spectrum


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
