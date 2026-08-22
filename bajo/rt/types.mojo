from std.math import pi, sqrt

from bajo.core import (
    AABB,
    Affine3f32,
    Frame,
    Vec3,
    Vec3f32,
    cross,
    length2,
    Point3,
    Point3f32,
    Rayf32,
)
from bajo.bvh.constants import Primitive, f32_max
from bajo.bvh.types import Instance, Sphere
from bajo.rt.geometry import sphere_unsigned_radius


comptime Color = Vec3f32[Frame.WORLD]


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
struct SurfaceId[length: SIMDLength = 1](Copyable, Writable):
    var value: SIMD[DType.uint32, Self.length]

    def __init__(out self, kind: MAT, index: UInt32):
        debug_assert["safe", _use_compiler_assume=True](
            kind.v < (UInt32(1) << SURFACE_KIND_BITS)
        )
        debug_assert["safe", _use_compiler_assume=True](
            index < (UInt32(1) << SURFACE_INDEX_BITS)
        )
        self.value = (kind.v << SURFACE_INDEX_BITS) | index

    @always_inline
    def kind(self) -> MAT:
        comptime assert Self.length == 1
        return MAT(self.value[0] >> SURFACE_INDEX_BITS)

    @always_inline
    def index(self) -> UInt32:
        comptime assert Self.length == 1
        return self.value[0] & SURFACE_INDEX_MASK

    @always_inline
    def get(self, lane: Int) -> SurfaceId[1]:
        return SurfaceId[1](self.value[lane])


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

    def validate(self, surface: SurfaceId[1]) -> Bool:
        if surface.kind() == MAT.LAMBERTIAN:
            return surface.index() < UInt32(len(self.lambertians))
        elif surface.kind() == MAT.METAL:
            return surface.index() < UInt32(len(self.metals))
        elif surface.kind() == MAT.DIELECTRIC:
            return surface.index() < UInt32(len(self.dielectrics))
        elif surface.kind() == MAT.EMISSIVE:
            return surface.index() < UInt32(len(self.emissives))

        return False

    def add_lambertian(mut self, albedo: Color) -> SurfaceId[1]:
        var index = UInt32(len(self.lambertians))
        self.lambertians.append(Lambertian(albedo))
        return SurfaceId(MAT.LAMBERTIAN, index)

    def add_metal(mut self, albedo: Color, fuzz: Float32) -> SurfaceId[1]:
        debug_assert["safe", _use_compiler_assume=True](fuzz >= 0.0)
        debug_assert["safe", _use_compiler_assume=True](fuzz <= 1.0)
        var index = UInt32(len(self.metals))
        self.metals.append(Metal(albedo, fuzz))
        return SurfaceId(MAT.METAL, index)

    def add_dielectric(mut self, refraction_index: Float32) -> SurfaceId[1]:
        debug_assert["safe", _use_compiler_assume=True](refraction_index > 0.0)
        var index = UInt32(len(self.dielectrics))
        self.dielectrics.append(Dielectric(refraction_index))
        return SurfaceId(MAT.DIELECTRIC, index)

    def add_emissive(mut self, radiance: Color) -> SurfaceId[1]:
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
    var surface: SurfaceId[1]
    var weight: Float32


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


@fieldwise_init
struct HitRecord(Copyable, Writable):
    var primitive: PrimitiveId
    var p: Point3f32[Frame.WORLD]
    var normal: Vec3f32[Frame.WORLD]
    var surface: SurfaceId[1]
    var t: Float32
    var front_face: Bool


@fieldwise_init
struct SurfaceHit[length: SIMDLength = 1](Copyable, Writable):
    """Renderer hit without primitive identity or position."""

    var normal: Vec3[DType.float32, Frame.WORLD, Self.length]
    var surface: SurfaceId[Self.length]
    var t: SIMD[DType.float32, Self.length]
    var front_face: SIMD[DType.bool, Self.length]
    var hit: SIMD[DType.bool, Self.length]

    def __init__(out self, t_max: SIMD[DType.float32, Self.length]):
        self.normal = Vec3[DType.float32, Frame.WORLD, Self.length](0.0)
        self.surface = SurfaceId[Self.length](
            SIMD[DType.uint32, Self.length](0)
        )
        self.t = t_max
        self.front_face = SIMD[DType.bool, Self.length](fill=True)
        self.hit = SIMD[DType.bool, Self.length](fill=False)

    @staticmethod
    def miss(t: SIMD[DType.float32, Self.length] = f32_max) -> Self:
        return Self(t)

    @always_inline
    def get(self, lane: Int) -> SurfaceHit[1]:
        return SurfaceHit[1](
            Vec3f32[Frame.WORLD](
                self.normal.x[lane],
                self.normal.y[lane],
                self.normal.z[lane],
            ),
            self.surface.get(lane),
            self.t[lane],
            self.front_face[lane],
            self.hit[lane],
        )


@fieldwise_init
struct ShadingPoint[length: SIMDLength = 1](Copyable, Writable):
    var p: Point3[DType.float32, Frame.WORLD, Self.length]
    var normal: Vec3[DType.float32, Frame.WORLD, Self.length]
    var front_face: SIMD[DType.bool, Self.length]


@fieldwise_init
struct BsdfSample[length: SIMDLength = 1](Copyable, Writable):
    """Sampled direction and throughput/PDF metadata."""

    var direction: Vec3[DType.float32, Frame.WORLD, Self.length]
    var weight: Vec3[DType.float32, Frame.WORLD, Self.length]
    var pdf: SIMD[DType.float32, Self.length]
    var delta: SIMD[DType.bool, Self.length]
    var ok: SIMD[DType.bool, Self.length]


@fieldwise_init
struct BsdfEvaluation[length: SIMDLength = 1](Copyable, Writable):
    """BSDF value and solid-angle PDF."""

    var value: Vec3[DType.float32, Frame.WORLD, Self.length]
    var pdf: SIMD[DType.float32, Self.length]
    var delta: SIMD[DType.bool, Self.length]


struct RenderSettings(Copyable, Writable):
    var image_width: Int
    var image_height: Int
    var samples_per_pixel: Int
    var rng_seed: UInt64
    var max_depth: Int

    def __init__(
        out self,
        image_width: Int,
        image_height: Int,
        samples_per_pixel: Int,
        rng_seed: UInt64,
        max_depth: Int = 8,
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
        debug_assert["safe", _use_compiler_assume=True](
            max_depth >= 0, "max depth must be non-negative"
        )

        self.image_width = image_width
        self.image_height = image_height
        self.samples_per_pixel = samples_per_pixel
        self.rng_seed = rng_seed
        self.max_depth = max_depth


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


struct SceneData:
    """Mutable backend-neutral scene authoring data.

    Build or edit this data before preparation. `CpuScene` consumes it into an
    immutable CPU snapshot, while GPU preparation uploads a device snapshot.
    Later authoring changes are not reflected in prepared scenes: rebuild a
    prepared scene to observe them. No implicit refit or dirty tracking exists.

    `SceneData` performs no acceleration-structure construction, so it can be
    prepared independently for either backend without paying for the other.
    """

    var spheres: List[Sphere[Frame.WORLD]]
    var sphere_surfaces: List[SurfaceId[1]]
    var triangle_vertices: List[Point3f32[Frame.WORLD]]
    var triangle_surfaces: List[SurfaceId[1]]
    var triangle_meshes: List[List[Point3f32[Frame.LOCAL]]]
    var triangle_instances: List[Instance]
    var triangle_instance_surfaces: List[SurfaceId[1]]
    var surfaces: SurfaceStore
    var lights: LightStore

    def __init__(
        out self,
        var spheres: List[Sphere[Frame.WORLD]],
        var sphere_surfaces: List[SurfaceId[1]],
        var triangle_vertices: List[Point3f32[Frame.WORLD]],
        var triangle_surfaces: List[SurfaceId[1]],
        var triangle_meshes: List[List[Point3f32[Frame.LOCAL]]],
        var triangle_instances: List[Instance],
        var triangle_instance_surfaces: List[SurfaceId[1]],
        var surfaces: SurfaceStore,
    ):
        debug_assert["safe", _use_compiler_assume=True](
            len(spheres) > 0
            or len(triangle_vertices) > 0
            or len(triangle_instances) > 0,
            "scene requires at least one primitive",
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

        for i, sphere in enumerate(self.spheres):
            debug_assert["safe", _use_compiler_assume=True](
                sphere.radius != 0.0,
                "sphere radius must be non-zero",
            )
            debug_assert["safe", _use_compiler_assume=True](
                self.surfaces.validate(self.sphere_surfaces[i]),
                "sphere surface id is out of range",
            )

        for surface in self.triangle_surfaces:
            debug_assert["safe", _use_compiler_assume=True](
                self.surfaces.validate(surface),
                "triangle surface id is out of range",
            )

        for vertices in self.triangle_meshes:
            debug_assert["safe", _use_compiler_assume=True](
                len(vertices) > 0 and len(vertices) % 3 == 0,
                (
                    "triangle mesh vertex count must be a positive multiple of"
                    " three"
                ),
            )

        for i, inst in enumerate(self.triangle_instances):
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

        self._build_light_store()
        self.lights.build_alias_table()

    def _build_light_store(mut self):
        for idx, surface in enumerate(self.triangle_surfaces):
            if surface.kind() == MAT.EMISSIVE:
                var radiance = self.surfaces.emissives[
                    Int(surface.index())
                ].radiance
                var weight = _scene_triangle_area(self, idx) * (
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

        for idx, surface in enumerate(self.sphere_surfaces):
            if surface.kind() == MAT.EMISSIVE:
                var radiance = self.surfaces.emissives[
                    Int(surface.index())
                ].radiance
                var radius = sphere_unsigned_radius(self.spheres[idx])
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


@always_inline
def _light_importance(radiance: Color) -> Float32:
    return max((radiance.x + radiance.y + radiance.z) / 3.0, 0.0)


@always_inline
def _scene_triangle_area(scene: SceneData, triangle_index: Int) -> Float32:
    ref v0 = scene.triangle_vertices[3 * triangle_index + 0]
    ref v1 = scene.triangle_vertices[3 * triangle_index + 1]
    ref v2 = scene.triangle_vertices[3 * triangle_index + 2]
    return 0.5 * sqrt(length2(cross(v1 - v0, v2 - v0)))


def ray_at(ray: Rayf32[Frame.WORLD], t: Float32) -> Point3f32[Frame.WORLD]:
    return ray.o + t * ray.d


def add_sphere(
    mut spheres: List[Sphere[Frame.WORLD]],
    mut sphere_surfaces: List[SurfaceId[1]],
    center: Point3f32[Frame.WORLD],
    radius: Float32,
    surface: SurfaceId[1],
):
    debug_assert["safe", _use_compiler_assume=True](
        radius != 0.0, "sphere radius must be non-zero"
    )
    spheres.append(Sphere[Frame.WORLD](center, radius))
    sphere_surfaces.append(surface.copy())


def add_triangle(
    mut triangle_vertices: List[Point3f32[Frame.WORLD]],
    mut triangle_surfaces: List[SurfaceId[1]],
    v0: Point3f32[Frame.WORLD],
    v1: Point3f32[Frame.WORLD],
    v2: Point3f32[Frame.WORLD],
    surface: SurfaceId[1],
):
    triangle_vertices.append(v0)
    triangle_vertices.append(v1)
    triangle_vertices.append(v2)
    triangle_surfaces.append(surface.copy())


def add_triangle_mesh(
    mut triangle_vertices: List[Point3f32[Frame.WORLD]],
    mut triangle_surfaces: List[SurfaceId[1]],
    vertices: ImmSpan[Point3f32[Frame.WORLD], _],
    surface: SurfaceId[1],
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
    mut triangle_instance_surfaces: List[SurfaceId[1]],
    vertices: ImmSpan[Point3f32[Frame.LOCAL], _],
    transform: Affine3f32[Frame.LOCAL, Frame.WORLD],
    bounds: AABB[Frame.LOCAL],
    surface: SurfaceId[1],
) -> UInt32:
    debug_assert["safe", _use_compiler_assume=True](
        len(vertices) > 0 and len(vertices) % 3 == 0,
        "triangle mesh vertex count must be a positive multiple of three",
    )
    var mesh_idx = UInt32(len(triangle_meshes))
    var owned_vertices = List[Point3f32[Frame.LOCAL]](capacity=len(vertices))
    owned_vertices.extend(vertices)
    triangle_meshes.append(owned_vertices^)
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
    mut triangle_instance_surfaces: List[SurfaceId[1]],
    mesh_idx: UInt32,
    transform: Affine3f32[Frame.LOCAL, Frame.WORLD],
    mesh_bounds: AABB[Frame.LOCAL],
    surface: SurfaceId[1],
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
