from std.math import abs

from bajo.core import (
    AABB,
    Affine3f32,
    Frame,
    Vec3f32,
    cross,
    dot,
    normalize,
    Point3f32,
    GeoKind,
    Rayf32,
)
from bajo.bvh.constants import EMPTY_LANE, Primitive, TRACE
from bajo.bvh.cpu.sphere_bvh import SphereBvh
from bajo.bvh.cpu.tlas import Tlas
from bajo.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.bvh.types import Instance, Sphere


comptime Color = Vec3f32[Frame.WORLD]
comptime BVH_WIDTH = 8

comptime MAT_LAMBERTIAN = UInt32(0)
comptime MAT_METAL = UInt32(1)
comptime MAT_DIELECTRIC = UInt32(2)
comptime SURFACE_KIND_BITS = UInt32(4)
comptime SURFACE_INDEX_BITS = UInt32(28)
comptime SURFACE_INDEX_MASK = UInt32((1 << 28) - 1)
comptime PRIM_SPHERE = UInt32(0)
comptime PRIM_TRIANGLE = UInt32(1)
comptime PRIM_TRIANGLE_INSTANCE = UInt32(2)
comptime PRIMITIVE_KIND_BITS = UInt32(4)
comptime PRIMITIVE_INDEX_BITS = UInt32(28)
comptime PRIMITIVE_INDEX_MASK = UInt32((1 << 28) - 1)
comptime RENDER_PATH = UInt32(0)
comptime RENDER_NORMALS = UInt32(1)
comptime RENDER_AO = UInt32(2)
comptime RENDER_NEE = UInt32(3)
comptime RENDER_MIS = UInt32(4)


@fieldwise_init
struct PrimitiveId(Copyable, Writable):
    var value: UInt32

    def __init__(out self, kind: UInt32, index: UInt32):
        debug_assert["safe"](kind < (UInt32(1) << PRIMITIVE_KIND_BITS))
        debug_assert["safe"](index < (UInt32(1) << PRIMITIVE_INDEX_BITS))
        self.value = (kind << PRIMITIVE_INDEX_BITS) | index

    def kind(self) -> UInt32:
        return self.value >> PRIMITIVE_INDEX_BITS

    def index(self) -> UInt32:
        return self.value & PRIMITIVE_INDEX_MASK


@fieldwise_init
struct SurfaceId(Copyable, Writable):
    var value: UInt32

    def __init__(out self, kind: UInt32, index: UInt32):
        debug_assert["safe"](kind < (UInt32(1) << SURFACE_KIND_BITS))
        debug_assert["safe"](index < (UInt32(1) << SURFACE_INDEX_BITS))
        self.value = (kind << SURFACE_INDEX_BITS) | index

    def kind(self) -> UInt32:
        return self.value >> SURFACE_INDEX_BITS

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


struct SurfaceStore(Movable):
    var lambertians: List[Lambertian]
    var metals: List[Metal]
    var dielectrics: List[Dielectric]

    def __init__(out self):
        self.lambertians = List[Lambertian]()
        self.metals = List[Metal]()
        self.dielectrics = List[Dielectric]()

    def validate(self, surface: SurfaceId) -> Bool:
        if surface.kind() == MAT_LAMBERTIAN:
            return surface.index() < UInt32(len(self.lambertians))
        elif surface.kind() == MAT_METAL:
            return surface.index() < UInt32(len(self.metals))
        elif surface.kind() == MAT_DIELECTRIC:
            return surface.index() < UInt32(len(self.dielectrics))

        return False

    def add_lambertian(mut self, albedo: Color) -> SurfaceId:
        var index = UInt32(len(self.lambertians))
        self.lambertians.append(Lambertian(albedo))
        return SurfaceId(MAT_LAMBERTIAN, index)

    def add_metal(mut self, albedo: Color, fuzz: Float32) -> SurfaceId:
        debug_assert["safe"](fuzz >= 0.0)
        debug_assert["safe"](fuzz <= 1.0)
        var index = UInt32(len(self.metals))
        self.metals.append(Metal(albedo, fuzz))
        return SurfaceId(MAT_METAL, index)

    def add_dielectric(mut self, refraction_index: Float32) -> SurfaceId:
        debug_assert["safe"](refraction_index > 0.0)
        var index = UInt32(len(self.dielectrics))
        self.dielectrics.append(Dielectric(refraction_index))
        return SurfaceId(MAT_DIELECTRIC, index)


@fieldwise_init
struct HitRecord(Copyable, Writable):
    var primitive: PrimitiveId
    var p: Point3f32[Frame.WORLD]
    var normal: Vec3f32[Frame.WORLD]
    var surface: SurfaceId
    var t: Float32
    var front_face: Bool


@fieldwise_init
struct ScatterResult(Copyable, Writable):
    var ok: Bool
    var ray: Rayf32[Frame.WORLD]
    var attenuation: Color


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
        debug_assert["safe"](image_width > 0, "image width must be positive")
        debug_assert["safe"](image_height > 0, "image height must be positive")
        debug_assert["safe"](
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


struct RenderResult(Movable):
    var pixels: List[Color]
    var timings: RenderTimings

    def __init__(
        out self,
        var pixels: List[Color],
        timings: RenderTimings,
    ):
        self.pixels = pixels^
        self.timings = timings.copy()


struct World(Movable):
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
        debug_assert["safe"](
            len(spheres) > 0
            or len(triangle_vertices) > 0
            or len(triangle_instances) > 0,
            "world requires at least one primitive",
        )
        debug_assert["safe"](
            len(spheres) == len(sphere_surfaces),
            "sphere and surface sidecar lengths must match",
        )
        debug_assert["safe"](
            len(triangle_vertices) % 3 == 0,
            "triangle vertex count must be a multiple of three",
        )
        debug_assert["safe"](
            len(triangle_vertices) / 3 == len(triangle_surfaces),
            "triangle and surface sidecar lengths must match",
        )
        debug_assert["safe"](
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
                debug_assert["safe"](
                    s.radius != 0.0, "sphere radius must be non-zero"
                )
                debug_assert["safe"](
                    self.surfaces.validate(self.sphere_surfaces[i]),
                    "sphere surface id is out of range",
                )
                bvh_spheres.append(Sphere[Frame.WORLD](s.center, abs(s.radius)))

            self.sphere_bvh = Optional[SphereBvh[Frame.WORLD, BVH_WIDTH]](
                SphereBvh[Frame.WORLD, BVH_WIDTH].__init__["lbvh"](bvh_spheres^)
            )

        if len(self.triangle_vertices) > 0:
            for i in range(len(self.triangle_surfaces)):
                debug_assert["safe"](
                    self.surfaces.validate(self.triangle_surfaces[i]),
                    "triangle surface id is out of range",
                )

            self.triangle_bvh = Optional[TriangleBvh[Frame.WORLD, BVH_WIDTH]](
                TriangleBvh[Frame.WORLD, BVH_WIDTH].__init__["lbvh"](
                    self.triangle_vertices.copy()
                )
            )

        if len(self.triangle_instances) > 0:
            for mesh_idx in range(len(self.triangle_meshes)):
                ref vertices = self.triangle_meshes[mesh_idx]
                debug_assert["safe"](
                    len(vertices) > 0 and len(vertices) % 3 == 0,
                    (
                        "triangle mesh vertex count must be a positive multiple"
                        " of three"
                    ),
                )
                self.triangle_mesh_blases.append(
                    TriangleBvh[Frame.LOCAL, BVH_WIDTH].__init__["lbvh"](
                        vertices.copy()
                    )
                )

            for i in range(len(self.triangle_instances)):
                ref inst = self.triangle_instances[i]
                debug_assert["safe"](
                    inst.kind == Primitive.TRIANGLE,
                    "triangle instance must have triangle primitive kind",
                )
                debug_assert["safe"](
                    inst.blas_idx < UInt32(len(self.triangle_meshes)),
                    "triangle instance blas_idx is out of range",
                )
                debug_assert["safe"](
                    self.surfaces.validate(self.triangle_instance_surfaces[i]),
                    "triangle instance surface id is out of range",
                )

            self.triangle_tlas = Optional[Tlas[BVH_WIDTH]](
                Tlas[BVH_WIDTH](self.triangle_instances)
            )

    def hit(self, ray: Rayf32[Frame.WORLD]) -> Optional[HitRecord]:
        return self.trace(ray)

    def trace(self, ray: Rayf32[Frame.WORLD]) -> Optional[HitRecord]:
        var sphere_hit = self._trace_spheres(ray)
        var triangle_hit = self._trace_triangles(ray)
        var instance_hit = self._trace_triangle_instances(ray)

        if sphere_hit:
            if triangle_hit:
                if instance_hit:
                    return _closest_hit3(
                        sphere_hit.value(),
                        triangle_hit.value(),
                        instance_hit.value(),
                    )
                return _closest_hit2(sphere_hit.value(), triangle_hit.value())
            if instance_hit:
                return _closest_hit2(sphere_hit.value(), instance_hit.value())
            return Optional[HitRecord](sphere_hit.value().copy())

        if triangle_hit:
            if instance_hit:
                return _closest_hit2(triangle_hit.value(), instance_hit.value())
            return Optional[HitRecord](triangle_hit.value().copy())

        if instance_hit:
            return Optional[HitRecord](instance_hit.value().copy())

        return None

    def _trace_spheres(self, ray: Rayf32[Frame.WORLD]) -> Optional[HitRecord]:
        if not self.sphere_bvh:
            return None

        var bvh_hit = self.sphere_bvh.value().trace[TRACE.CLOSEST_HIT](ray)
        if not bvh_hit.is_hit():
            return None

        var primitive = PrimitiveId(PRIM_SPHERE, bvh_hit.prim)
        var sphere_idx = Int(primitive.index())
        debug_assert["safe"](
            sphere_idx >= 0 and sphere_idx < len(self.spheres),
            "BVH returned an out-of-range sphere index",
        )
        ref sphere = self.spheres[sphere_idx]
        debug_assert["safe"](
            self.surfaces.validate(self.sphere_surfaces[sphere_idx]),
            "hit sphere surface id is out of range",
        )
        var p = ray_at(ray, bvh_hit.t)
        var outward_normal = (p - sphere.center) / sphere.radius
        var front_face = dot(ray.d, outward_normal) < 0.0
        var normal = outward_normal if front_face else -outward_normal

        return HitRecord(
            primitive.copy(),
            p,
            normal,
            self.sphere_surfaces[sphere_idx].copy(),
            bvh_hit.t,
            front_face,
        )

    def _trace_triangle_instances(
        self, ray: Rayf32[Frame.WORLD]
    ) -> Optional[HitRecord]:
        if not self.triangle_tlas:
            return None

        var bvh_hit = self.triangle_tlas.value().trace[
            TriangleBvh[Frame.LOCAL, BVH_WIDTH],
            TRACE.CLOSEST_HIT,
        ](ray, self.triangle_mesh_blases.unsafe_ptr())
        if not bvh_hit.is_hit() or bvh_hit.inst == EMPTY_LANE:
            return None

        var instance_idx = Int(bvh_hit.inst)
        debug_assert["safe"](
            instance_idx >= 0 and instance_idx < len(self.triangle_instances),
            "TLAS returned an out-of-range triangle instance index",
        )
        debug_assert["safe"](
            self.surfaces.validate(
                self.triangle_instance_surfaces[instance_idx]
            ),
            "hit triangle instance surface id is out of range",
        )

        var primitive = PrimitiveId(PRIM_TRIANGLE_INSTANCE, bvh_hit.inst)
        var p = ray_at(ray, bvh_hit.t)
        var outward_normal = bvh_hit.normal.unsafe_convert[
            new_kind=GeoKind.VECTOR
        ]()
        var front_face = dot(ray.d, outward_normal) < 0.0
        var normal = outward_normal if front_face else -outward_normal

        return HitRecord(
            primitive.copy(),
            p,
            normal,
            self.triangle_instance_surfaces[instance_idx].copy(),
            bvh_hit.t,
            front_face,
        )

    def _trace_triangles(self, ray: Rayf32[Frame.WORLD]) -> Optional[HitRecord]:
        if not self.triangle_bvh:
            return None

        var bvh_hit = self.triangle_bvh.value().trace[TRACE.CLOSEST_HIT](ray)
        if not bvh_hit.is_hit():
            return None

        var primitive = PrimitiveId(PRIM_TRIANGLE, bvh_hit.prim)
        var tri_idx = Int(primitive.index())
        debug_assert["safe"](
            tri_idx >= 0 and tri_idx < len(self.triangle_surfaces),
            "BVH returned an out-of-range triangle index",
        )
        debug_assert["safe"](
            self.surfaces.validate(self.triangle_surfaces[tri_idx]),
            "hit triangle surface id is out of range",
        )

        var base = tri_idx * 3
        ref v0 = self.triangle_vertices[base + 0]
        ref v1 = self.triangle_vertices[base + 1]
        ref v2 = self.triangle_vertices[base + 2]

        var p = ray_at(ray, bvh_hit.t)
        var outward_normal = normalize(cross(v1 - v0, v2 - v0))
        var front_face = dot(ray.d, outward_normal) < 0.0
        var normal = outward_normal if front_face else -outward_normal

        return HitRecord(
            primitive.copy(),
            p,
            normal,
            self.triangle_surfaces[tri_idx].copy(),
            bvh_hit.t,
            front_face,
        )


def ray_at(ray: Rayf32[Frame.WORLD], t: Float32) -> Point3f32[Frame.WORLD]:
    return ray.o + t * ray.d


def _closest_hit2(a: HitRecord, b: HitRecord) -> Optional[HitRecord]:
    if a.t <= b.t:
        return Optional[HitRecord](a.copy())
    return Optional[HitRecord](b.copy())


def _closest_hit3(
    a: HitRecord, b: HitRecord, c: HitRecord
) -> Optional[HitRecord]:
    if a.t <= b.t and a.t <= c.t:
        return Optional[HitRecord](a.copy())
    elif b.t <= c.t:
        return Optional[HitRecord](b.copy())
    return Optional[HitRecord](c.copy())


def add_sphere(
    mut spheres: List[Sphere[Frame.WORLD]],
    mut sphere_surfaces: List[SurfaceId],
    center: Point3f32[Frame.WORLD],
    radius: Float32,
    surface: SurfaceId,
):
    debug_assert["safe"](radius != 0.0, "sphere radius must be non-zero")
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
    debug_assert["safe"](
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
    debug_assert["safe"](
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
