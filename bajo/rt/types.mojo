from std.math import abs, pi, sqrt

from bajo.core import (
    AABB,
    Affine3f32,
    Frame,
    Vec3f32,
    cross,
    dot,
    length2,
    Point3f32,
    GeoKind,
    Rayf32,
)
from bajo.bvh.constants import EMPTY_LANE, Primitive, TRACE, f32_max
from bajo.bvh.cpu.sphere_bvh import SphereBvh
from bajo.bvh.cpu.tlas import Tlas
from bajo.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.bvh.types import Instance, Sphere


comptime Color = Vec3f32[Frame.WORLD]
comptime BVH_WIDTH = 16


@fieldwise_init
struct MAT(Equatable, TrivialRegisterPassable, Writable):
    var v: UInt32
    comptime LAMBERTIAN = Self(0)
    comptime METAL = Self(1)
    comptime DIELECTRIC = Self(2)
    comptime EMISSIVE = Self(3)


comptime SURFACE_KIND_BITS = UInt32(4)
comptime SURFACE_INDEX_BITS = 32 - SURFACE_KIND_BITS
comptime SURFACE_INDEX_MASK = UInt32((1 << SURFACE_INDEX_BITS) - 1)


@fieldwise_init
struct PRIM(Equatable, TrivialRegisterPassable, Writable):
    var v: UInt32
    comptime SPHERE = Self(0)
    comptime TRIANGLE = Self(1)
    comptime TRIANGLE_INSTANCE = Self(2)


comptime PRIMITIVE_KIND_BITS = UInt32(4)
comptime PRIMITIVE_INDEX_BITS = 32 - PRIMITIVE_KIND_BITS
comptime PRIMITIVE_INDEX_MASK = UInt32((1 << PRIMITIVE_INDEX_BITS) - 1)


@fieldwise_init
struct RENDER(Equatable, TrivialRegisterPassable, Writable):
    var v: UInt32
    comptime PATH = Self(0)
    comptime NORMALS = Self(1)
    comptime AO = Self(2)
    comptime NEE = Self(3)
    comptime MIS = Self(4)


@fieldwise_init
struct PrimitiveId(Copyable, Writable):
    var value: UInt32

    def __init__(out self, kind: PRIM, index: UInt32):
        debug_assert["safe", _use_compiler_assume=True](
            kind.v < (UInt32(1) << PRIMITIVE_KIND_BITS)
        )
        debug_assert["safe", _use_compiler_assume=True](
            index < (UInt32(1) << PRIMITIVE_INDEX_BITS)
        )
        self.value = (kind.v << PRIMITIVE_INDEX_BITS) | index

    def kind(self) -> PRIM:
        return PRIM(self.value >> PRIMITIVE_INDEX_BITS)

    def index(self) -> UInt32:
        return self.value & PRIMITIVE_INDEX_MASK


@fieldwise_init
struct SurfaceId(Copyable, Writable):
    var value: UInt32

    def __init__(out self, kind: MAT, index: UInt32):
        debug_assert["safe", _use_compiler_assume=True](
            kind.v < (UInt32(1) << SURFACE_KIND_BITS)
        )
        debug_assert["safe", _use_compiler_assume=True](
            index < (UInt32(1) << SURFACE_INDEX_BITS)
        )
        self.value = (kind.v << SURFACE_INDEX_BITS) | index

    def kind(self) -> MAT:
        return MAT(self.value >> SURFACE_INDEX_BITS)

    def index(self) -> UInt32:
        return self.value & SURFACE_INDEX_MASK


@fieldwise_init
struct Lambertian(Copyable, Writable):
    var albedo: Color


@fieldwise_init
struct Metal(Copyable, Writable):
    var albedo: Color
    var fuzz: Float32


@fieldwise_init
struct Dielectric(Copyable, Writable):
    var refraction_index: Float32


@fieldwise_init
struct Emissive(Copyable, Writable):
    var radiance: Color


struct SurfaceStore:
    var lambertians: List[Lambertian]
    var metals: List[Metal]
    var dielectrics: List[Dielectric]
    var emissives: List[Emissive]

    def __init__(out self):
        self.lambertians = List[Lambertian]()
        self.metals = List[Metal]()
        self.dielectrics = List[Dielectric]()
        self.emissives = List[Emissive]()

    def validate(self, surface: SurfaceId) -> Bool:
        if surface.kind() == MAT.LAMBERTIAN:
            return surface.index() < UInt32(len(self.lambertians))
        elif surface.kind() == MAT.METAL:
            return surface.index() < UInt32(len(self.metals))
        elif surface.kind() == MAT.DIELECTRIC:
            return surface.index() < UInt32(len(self.dielectrics))
        elif surface.kind() == MAT.EMISSIVE:
            return surface.index() < UInt32(len(self.emissives))

        return False

    def add_lambertian(mut self, albedo: Color) -> SurfaceId:
        var index = UInt32(len(self.lambertians))
        self.lambertians.append(Lambertian(albedo))
        return SurfaceId(MAT.LAMBERTIAN, index)

    def add_metal(mut self, albedo: Color, fuzz: Float32) -> SurfaceId:
        debug_assert["safe", _use_compiler_assume=True](fuzz >= 0.0)
        debug_assert["safe", _use_compiler_assume=True](fuzz <= 1.0)
        var index = UInt32(len(self.metals))
        self.metals.append(Metal(albedo, fuzz))
        return SurfaceId(MAT.METAL, index)

    def add_dielectric(mut self, refraction_index: Float32) -> SurfaceId:
        debug_assert["safe", _use_compiler_assume=True](refraction_index > 0.0)
        var index = UInt32(len(self.dielectrics))
        self.dielectrics.append(Dielectric(refraction_index))
        return SurfaceId(MAT.DIELECTRIC, index)

    def add_emissive(mut self, radiance: Color) -> SurfaceId:
        debug_assert["safe", _use_compiler_assume=True](
            radiance.x >= 0.0 and radiance.y >= 0.0 and radiance.z >= 0.0,
            "emissive radiance must be non-negative",
        )
        var index = UInt32(len(self.emissives))
        self.emissives.append(Emissive(radiance))
        return SurfaceId(MAT.EMISSIVE, index)


@fieldwise_init
struct LightRecord(Copyable, Writable):
    """Compact emissive-primitive entry suitable for device packing."""

    var primitive: PrimitiveId
    var surface: SurfaceId
    var weight: Float32


struct LightStore:
    var records: List[LightRecord]
    var total_weight: Float32

    def __init__(out self):
        self.records = List[LightRecord]()
        self.total_weight = 0.0

    @always_inline
    def append(mut self, var record: LightRecord):
        self.total_weight += record.weight
        self.records.append(record^)


@fieldwise_init
struct HitRecord(Copyable, Writable):
    var primitive: PrimitiveId
    var p: Point3f32[Frame.WORLD]
    var normal: Vec3f32[Frame.WORLD]
    var surface: SurfaceId
    var t: Float32
    var front_face: Bool


@fieldwise_init
struct SurfaceHit(Copyable, Writable):
    """Compact renderer hit without public primitive identity or position."""

    var normal: Vec3f32[Frame.WORLD]
    var surface: SurfaceId
    var t: Float32
    var front_face: Bool
    var hit: Bool

    @staticmethod
    def miss(t: Float32 = f32_max) -> Self:
        return Self(
            Vec3f32[Frame.WORLD](0.0),
            SurfaceId(MAT.LAMBERTIAN, UInt32(0)),
            t,
            True,
            False,
        )


@fieldwise_init
struct _WorldHit(Copyable, Writable):
    """Canonical closest-hit record shared by public and renderer queries."""

    var primitive: PrimitiveId
    var normal: Vec3f32[Frame.WORLD]
    var surface: SurfaceId
    var t: Float32
    var front_face: Bool
    var hit: Bool

    @staticmethod
    def miss(t: Float32 = f32_max) -> Self:
        return Self(
            PrimitiveId(PRIM.SPHERE, UInt32(0)),
            Vec3f32[Frame.WORLD](0.0),
            SurfaceId(MAT.LAMBERTIAN, UInt32(0)),
            t,
            True,
            False,
        )


@fieldwise_init
struct ShadingPoint(Copyable, Writable):
    var p: Point3f32[Frame.WORLD]
    var normal: Vec3f32[Frame.WORLD]
    var front_face: Bool


@fieldwise_init
struct BsdfSample(Copyable, Writable):
    """One sampled continuation and its throughput/PDF metadata."""

    var ok: Bool
    var ray: Rayf32[Frame.WORLD]
    var weight: Color
    var pdf: Float32
    var delta: Bool


@fieldwise_init
struct BsdfEvaluation(Copyable, Writable):
    """Evaluated BSDF value and solid-angle PDF for one direction."""

    var value: Color
    var pdf: Float32
    var delta: Bool


struct RenderSettings(Copyable, Writable):
    var image_width: Int
    var image_height: Int
    var samples_per_pixel: Int
    var rng_seed: UInt64

    def __init__(
        out self,
        image_width: Int,
        image_height: Int,
        samples_per_pixel: Int,
        rng_seed: UInt64,
    ):
        debug_assert["safe", _use_compiler_assume=True](
            image_width > 0, "image width must be positive"
        )
        debug_assert["safe", _use_compiler_assume=True](
            image_height > 0, "image height must be positive"
        )
        debug_assert["safe", _use_compiler_assume=True](
            samples_per_pixel > 0, "samples per pixel must be positive"
        )

        self.image_width = image_width
        self.image_height = image_height
        self.samples_per_pixel = samples_per_pixel
        self.rng_seed = rng_seed


@fieldwise_init
struct RenderTimings(Copyable, Writable):
    var total_ns: Int
    var init_ns: Int
    var render_ns: Int
    var pixel_count: Int
    var sample_count: Int
    var max_depth: Int


struct RenderResult:
    var pixels: List[Color]
    var timings: RenderTimings

    def __init__(
        out self,
        var pixels: List[Color],
        timings: RenderTimings,
    ):
        self.pixels = pixels^
        self.timings = timings.copy()


struct World:
    var sphere_bvh: Optional[SphereBvh[Frame.WORLD, BVH_WIDTH]]
    var triangle_bvh: Optional[TriangleBvh[Frame.WORLD, BVH_WIDTH]]
    var triangle_tlas: Optional[Tlas[BVH_WIDTH]]
    var spheres: List[Sphere[Frame.WORLD]]
    var sphere_surfaces: List[SurfaceId]
    var triangle_vertices: List[Point3f32[Frame.WORLD]]
    var triangle_surfaces: List[SurfaceId]
    var triangle_meshes: List[List[Point3f32[Frame.LOCAL]]]
    var triangle_mesh_blases: List[TriangleBvh[Frame.LOCAL, BVH_WIDTH]]
    var triangle_instances: List[Instance]
    var triangle_instance_surfaces: List[SurfaceId]
    var surfaces: SurfaceStore
    var lights: LightStore

    def __init__(
        out self,
        var spheres: List[Sphere[Frame.WORLD]],
        var sphere_surfaces: List[SurfaceId],
        var triangle_vertices: List[Point3f32[Frame.WORLD]],
        var triangle_surfaces: List[SurfaceId],
        var triangle_meshes: List[List[Point3f32[Frame.LOCAL]]],
        var triangle_instances: List[Instance],
        var triangle_instance_surfaces: List[SurfaceId],
        var surfaces: SurfaceStore,
    ):
        debug_assert["safe", _use_compiler_assume=True](
            len(spheres) > 0
            or len(triangle_vertices) > 0
            or len(triangle_instances) > 0,
            "world requires at least one primitive",
        )
        debug_assert["safe", _use_compiler_assume=True](
            len(spheres) == len(sphere_surfaces),
            "sphere and surface sidecar lengths must match",
        )
        debug_assert["safe", _use_compiler_assume=True](
            len(triangle_vertices) % 3 == 0,
            "triangle vertex count must be a multiple of three",
        )
        debug_assert["safe", _use_compiler_assume=True](
            len(triangle_vertices) / 3 == len(triangle_surfaces),
            "triangle and surface sidecar lengths must match",
        )
        debug_assert["safe", _use_compiler_assume=True](
            len(triangle_instances) == len(triangle_instance_surfaces),
            "triangle instance and surface sidecar lengths must match",
        )

        self.spheres = spheres^
        self.sphere_surfaces = sphere_surfaces^
        self.triangle_vertices = triangle_vertices^
        self.triangle_surfaces = triangle_surfaces^
        self.triangle_meshes = triangle_meshes^
        self.triangle_instances = triangle_instances^
        self.triangle_instance_surfaces = triangle_instance_surfaces^
        self.surfaces = surfaces^
        self.lights = LightStore()

        self.sphere_bvh = Optional[SphereBvh[Frame.WORLD, BVH_WIDTH]]()
        self.triangle_bvh = Optional[TriangleBvh[Frame.WORLD, BVH_WIDTH]]()
        self.triangle_tlas = Optional[Tlas[BVH_WIDTH]]()
        self.triangle_mesh_blases = List[TriangleBvh[Frame.LOCAL, BVH_WIDTH]]()

        if len(self.spheres) > 0:
            var bvh_spheres = List[Sphere[Frame.WORLD]](
                capacity=len(self.spheres)
            )
            for i in range(len(self.spheres)):
                ref s = self.spheres[i]
                debug_assert["safe", _use_compiler_assume=True](
                    s.radius != 0.0, "sphere radius must be non-zero"
                )
                debug_assert["safe", _use_compiler_assume=True](
                    self.surfaces.validate(self.sphere_surfaces[i]),
                    "sphere surface id is out of range",
                )
                bvh_spheres.append(Sphere[Frame.WORLD](s.center, abs(s.radius)))

            self.sphere_bvh = Optional[SphereBvh[Frame.WORLD, BVH_WIDTH]](
                SphereBvh[Frame.WORLD, BVH_WIDTH].__init__["sah"](bvh_spheres^)
            )

        if len(self.triangle_vertices) > 0:
            for i in range(len(self.triangle_surfaces)):
                debug_assert["safe", _use_compiler_assume=True](
                    self.surfaces.validate(self.triangle_surfaces[i]),
                    "triangle surface id is out of range",
                )

            self.triangle_bvh = Optional[TriangleBvh[Frame.WORLD, BVH_WIDTH]](
                TriangleBvh[Frame.WORLD, BVH_WIDTH].__init__["sah"](
                    self.triangle_vertices
                )
            )

        if len(self.triangle_instances) > 0:
            for mesh_idx in range(len(self.triangle_meshes)):
                ref vertices = self.triangle_meshes[mesh_idx]
                debug_assert["safe", _use_compiler_assume=True](
                    len(vertices) > 0 and len(vertices) % 3 == 0,
                    (
                        "triangle mesh vertex count must be a positive multiple"
                        " of three"
                    ),
                )
                self.triangle_mesh_blases.append(
                    TriangleBvh[Frame.LOCAL, BVH_WIDTH].__init__["sah"](
                        vertices
                    )
                )

            for i in range(len(self.triangle_instances)):
                ref inst = self.triangle_instances[i]
                debug_assert["safe", _use_compiler_assume=True](
                    inst.kind == Primitive.TRIANGLE,
                    "triangle instance must have triangle primitive kind",
                )
                debug_assert["safe", _use_compiler_assume=True](
                    inst.blas_idx < UInt32(len(self.triangle_meshes)),
                    "triangle instance blas_idx is out of range",
                )
                debug_assert["safe", _use_compiler_assume=True](
                    self.surfaces.validate(self.triangle_instance_surfaces[i]),
                    "triangle instance surface id is out of range",
                )

            self.triangle_tlas = Optional[Tlas[BVH_WIDTH]](
                Tlas[BVH_WIDTH](self.triangle_instances)
            )

        self._build_light_store()

    def _build_light_store(mut self):
        for idx in range(len(self.triangle_surfaces)):
            ref surface = self.triangle_surfaces[idx]
            if surface.kind() == MAT.EMISSIVE:
                var radiance = self.surfaces.emissives[
                    Int(surface.index())
                ].radiance
                var weight = _world_triangle_area(self, idx) * (
                    _light_importance(radiance)
                )
                if weight > 0.0:
                    self.lights.append(
                        LightRecord(
                            PrimitiveId(PRIM.TRIANGLE, UInt32(idx)),
                            surface.copy(),
                            weight,
                        )
                    )

        for idx in range(len(self.sphere_surfaces)):
            ref surface = self.sphere_surfaces[idx]
            if surface.kind() == MAT.EMISSIVE:
                var radiance = self.surfaces.emissives[
                    Int(surface.index())
                ].radiance
                var radius = abs(self.spheres[idx].radius)
                var weight = (
                    4.0 * pi * radius * radius * _light_importance(radiance)
                )
                if weight > 0.0:
                    self.lights.append(
                        LightRecord(
                            PrimitiveId(PRIM.SPHERE, UInt32(idx)),
                            surface.copy(),
                            weight,
                        )
                    )

    def occluded(self, ray: Rayf32[Frame.WORLD]) -> Bool:
        """Return as soon as any world primitive intersects `ray`.

        Visibility queries do not need the closest primitive, surface data,
        hit point, or normal. Keep them on the BVHs' any-hit paths and
        short-circuit between geometry classes as well.
        """
        if self.sphere_bvh:
            var hit = self.sphere_bvh.value().trace[TRACE.ANY_HIT](ray)
            if hit.is_occluded():
                return True

        if self.triangle_bvh:
            var hit = self.triangle_bvh.value().trace[TRACE.ANY_HIT](ray)
            if hit.is_occluded():
                return True

        if self.triangle_tlas:
            var hit = self.triangle_tlas.value().trace[
                TriangleBvh[Frame.LOCAL, BVH_WIDTH],
                TRACE.ANY_HIT,
            ](ray, Span(self.triangle_mesh_blases))
            if hit.is_occluded():
                return True

        return False

    def _trace_closest(self, ray: Rayf32[Frame.WORLD]) -> _WorldHit:
        var closest = self._trace_spheres(ray)
        var triangle_hit = self._trace_triangles(ray)
        if triangle_hit.hit and (not closest.hit or triangle_hit.t < closest.t):
            closest = triangle_hit^

        var instance_hit = self._trace_triangle_instances(ray)
        if instance_hit.hit and (not closest.hit or instance_hit.t < closest.t):
            closest = instance_hit^
        return closest^

    def trace(self, ray: Rayf32[Frame.WORLD]) -> Optional[HitRecord]:
        var hit = self._trace_closest(ray)
        if not hit.hit:
            return None
        return HitRecord(
            hit.primitive.copy(),
            ray_at(ray, hit.t),
            hit.normal,
            hit.surface.copy(),
            hit.t,
            hit.front_face,
        )

    def trace_surface(self, ray: Rayf32[Frame.WORLD]) -> SurfaceHit:
        """Trace the compact hit consumed by render integrators."""
        var hit = self._trace_closest(ray)
        return SurfaceHit(
            hit.normal,
            hit.surface.copy(),
            hit.t,
            hit.front_face,
            hit.hit,
        )

    def _trace_spheres(self, ray: Rayf32[Frame.WORLD]) -> _WorldHit:
        if not self.sphere_bvh:
            return _WorldHit.miss(ray.t_max)

        var bvh_hit = self.sphere_bvh.value().trace[TRACE.CLOSEST_HIT](ray)
        if not bvh_hit.is_hit():
            return _WorldHit.miss(ray.t_max)

        var sphere_idx = Int(bvh_hit.prim)
        debug_assert["safe", _use_compiler_assume=True](
            sphere_idx >= 0 and sphere_idx < len(self.spheres),
            "BVH returned an out-of-range sphere index",
        )
        ref sphere = self.spheres[sphere_idx]
        var p = ray_at(ray, bvh_hit.t)
        var outward_normal = (p - sphere.center) / sphere.radius
        var front_face = dot(ray.d, outward_normal) < 0.0
        var normal = outward_normal if front_face else -outward_normal
        return _WorldHit(
            PrimitiveId(PRIM.SPHERE, bvh_hit.prim),
            normal,
            self.sphere_surfaces[sphere_idx].copy(),
            bvh_hit.t,
            front_face,
            True,
        )

    def _trace_triangles(self, ray: Rayf32[Frame.WORLD]) -> _WorldHit:
        if not self.triangle_bvh:
            return _WorldHit.miss(ray.t_max)

        var bvh_hit = self.triangle_bvh.value().trace[TRACE.CLOSEST_HIT](ray)
        if not bvh_hit.is_hit():
            return _WorldHit.miss(ray.t_max)

        var tri_idx = Int(bvh_hit.prim)
        debug_assert["safe", _use_compiler_assume=True](
            tri_idx >= 0 and tri_idx < len(self.triangle_surfaces),
            "BVH returned an out-of-range triangle index",
        )
        var outward_normal = bvh_hit.normal.unsafe_convert[
            new_kind=GeoKind.VECTOR
        ]()
        var front_face = dot(ray.d, outward_normal) < 0.0
        var normal = outward_normal if front_face else -outward_normal
        return _WorldHit(
            PrimitiveId(PRIM.TRIANGLE, bvh_hit.prim),
            normal,
            self.triangle_surfaces[tri_idx].copy(),
            bvh_hit.t,
            front_face,
            True,
        )

    def _trace_triangle_instances(self, ray: Rayf32[Frame.WORLD]) -> _WorldHit:
        if not self.triangle_tlas:
            return _WorldHit.miss(ray.t_max)

        var bvh_hit = self.triangle_tlas.value().trace[
            TriangleBvh[Frame.LOCAL, BVH_WIDTH],
            TRACE.CLOSEST_HIT,
        ](ray, Span(self.triangle_mesh_blases))
        if not bvh_hit.is_hit() or bvh_hit.inst == EMPTY_LANE:
            return _WorldHit.miss(ray.t_max)

        var instance_idx = Int(bvh_hit.inst)
        debug_assert["safe", _use_compiler_assume=True](
            instance_idx >= 0 and instance_idx < len(self.triangle_instances),
            "TLAS returned an out-of-range triangle instance index",
        )
        var outward_normal = bvh_hit.normal.unsafe_convert[
            new_kind=GeoKind.VECTOR
        ]()
        var front_face = dot(ray.d, outward_normal) < 0.0
        var normal = outward_normal if front_face else -outward_normal
        return _WorldHit(
            PrimitiveId(PRIM.TRIANGLE_INSTANCE, bvh_hit.inst),
            normal,
            self.triangle_instance_surfaces[instance_idx].copy(),
            bvh_hit.t,
            front_face,
            True,
        )


@always_inline
def _light_importance(radiance: Color) -> Float32:
    return max((radiance.x + radiance.y + radiance.z) / 3.0, 0.0)


@always_inline
def _world_triangle_area(world: World, triangle_index: Int) -> Float32:
    ref v0 = world.triangle_vertices[3 * triangle_index + 0]
    ref v1 = world.triangle_vertices[3 * triangle_index + 1]
    ref v2 = world.triangle_vertices[3 * triangle_index + 2]
    return 0.5 * sqrt(length2(cross(v1 - v0, v2 - v0)))


def ray_at(ray: Rayf32[Frame.WORLD], t: Float32) -> Point3f32[Frame.WORLD]:
    return ray.o + t * ray.d


def add_sphere(
    mut spheres: List[Sphere[Frame.WORLD]],
    mut sphere_surfaces: List[SurfaceId],
    center: Point3f32[Frame.WORLD],
    radius: Float32,
    surface: SurfaceId,
):
    debug_assert["safe", _use_compiler_assume=True](
        radius != 0.0, "sphere radius must be non-zero"
    )
    spheres.append(Sphere[Frame.WORLD](center, radius))
    sphere_surfaces.append(surface.copy())


def add_triangle(
    mut triangle_vertices: List[Point3f32[Frame.WORLD]],
    mut triangle_surfaces: List[SurfaceId],
    v0: Point3f32[Frame.WORLD],
    v1: Point3f32[Frame.WORLD],
    v2: Point3f32[Frame.WORLD],
    surface: SurfaceId,
):
    triangle_vertices.append(v0)
    triangle_vertices.append(v1)
    triangle_vertices.append(v2)
    triangle_surfaces.append(surface.copy())


def add_triangle_mesh(
    mut triangle_vertices: List[Point3f32[Frame.WORLD]],
    mut triangle_surfaces: List[SurfaceId],
    vertices: List[Point3f32[Frame.WORLD]],
    surface: SurfaceId,
):
    debug_assert["safe", _use_compiler_assume=True](
        len(vertices) % 3 == 0,
        "triangle mesh vertex count must be a multiple of three",
    )
    for v in vertices:
        triangle_vertices.append(v)
    for _ in range(len(vertices) / 3):
        triangle_surfaces.append(surface.copy())


def add_triangle_mesh_instance(
    mut triangle_meshes: List[List[Point3f32[Frame.LOCAL]]],
    mut triangle_instances: List[Instance],
    mut triangle_instance_surfaces: List[SurfaceId],
    vertices: List[Point3f32[Frame.LOCAL]],
    transform: Affine3f32[Frame.LOCAL, Frame.WORLD],
    bounds: AABB[Frame.LOCAL],
    surface: SurfaceId,
) -> UInt32:
    debug_assert["safe", _use_compiler_assume=True](
        len(vertices) > 0 and len(vertices) % 3 == 0,
        "triangle mesh vertex count must be a positive multiple of three",
    )
    var mesh_idx = UInt32(len(triangle_meshes))
    triangle_meshes.append(vertices.copy())
    triangle_instances.append(
        Instance(
            transform,
            mesh_idx,
            bounds,
            Primitive.TRIANGLE,
        )
    )
    triangle_instance_surfaces.append(surface.copy())
    return mesh_idx


def add_triangle_instance(
    mut triangle_instances: List[Instance],
    mut triangle_instance_surfaces: List[SurfaceId],
    mesh_idx: UInt32,
    transform: Affine3f32[Frame.LOCAL, Frame.WORLD],
    mesh_bounds: AABB[Frame.LOCAL],
    surface: SurfaceId,
):
    triangle_instances.append(
        Instance(
            transform,
            mesh_idx,
            mesh_bounds,
            Primitive.TRIANGLE,
        )
    )
    triangle_instance_surfaces.append(surface.copy())
