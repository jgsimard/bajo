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
from bajo.bvh import Instance, Sphere
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


struct SceneBuilder(
    Deinitable where (False, "call finish() to validate and finalize the scene")
):
    var spheres: List[Sphere[Frame.WORLD]]
    var sphere_surfaces: List[SurfaceId[1]]
    var triangle_vertices: List[Point3f32[Frame.WORLD]]
    var triangle_surfaces: List[SurfaceId[1]]
    var triangle_meshes: List[List[Point3f32[Frame.LOCAL]]]
    var triangle_instances: List[Instance]
    var triangle_instance_surfaces: List[SurfaceId[1]]
    var surfaces: SurfaceStore

    def __init__(out self):
        self.spheres = List[Sphere[Frame.WORLD]]()
        self.sphere_surfaces = List[SurfaceId[1]]()
        self.triangle_vertices = List[Point3f32[Frame.WORLD]]()
        self.triangle_surfaces = List[SurfaceId[1]]()
        self.triangle_meshes = List[List[Point3f32[Frame.LOCAL]]]()
        self.triangle_instances = List[Instance]()
        self.triangle_instance_surfaces = List[SurfaceId[1]]()
        self.surfaces = SurfaceStore()

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
        self.spheres = spheres^
        self.sphere_surfaces = sphere_surfaces^
        self.triangle_vertices = triangle_vertices^
        self.triangle_surfaces = triangle_surfaces^
        self.triangle_meshes = triangle_meshes^
        self.triangle_instances = triangle_instances^
        self.triangle_instance_surfaces = triangle_instance_surfaces^
        self.surfaces = surfaces^

    def add_lambertian(mut self, albedo: Color) -> SurfaceId[1]:
        return self.surfaces.add_lambertian(albedo)

    def add_metal(mut self, albedo: Color, fuzz: Float32) -> SurfaceId[1]:
        return self.surfaces.add_metal(albedo, fuzz)

    def add_dielectric(mut self, refraction_index: Float32) -> SurfaceId[1]:
        return self.surfaces.add_dielectric(refraction_index)

    def add_emissive(mut self, radiance: Color) -> SurfaceId[1]:
        return self.surfaces.add_emissive(radiance)

    def add_sphere(
        mut self,
        center: Point3f32[Frame.WORLD],
        radius: Float32,
        surface: SurfaceId[1],
    ):
        self.spheres.append(Sphere[Frame.WORLD](center, radius))
        self.sphere_surfaces.append(surface.copy())

    def add_triangle(
        mut self,
        v0: Point3f32[Frame.WORLD],
        v1: Point3f32[Frame.WORLD],
        v2: Point3f32[Frame.WORLD],
        surface: SurfaceId[1],
    ):
        self.triangle_vertices.append(v0)
        self.triangle_vertices.append(v1)
        self.triangle_vertices.append(v2)
        self.triangle_surfaces.append(surface.copy())

    def add_quad(
        mut self,
        a: Point3f32[Frame.WORLD],
        b: Point3f32[Frame.WORLD],
        c: Point3f32[Frame.WORLD],
        d: Point3f32[Frame.WORLD],
        surface: SurfaceId[1],
    ):
        """Append two consistently wound triangles: `(a, b, c)` and `(a, c, d)`.
        """
        self.add_triangle(a, b, c, surface)
        self.add_triangle(a, c, d, surface)

    def add_triangle_mesh(
        mut self,
        vertices: ImmSpan[Point3f32[Frame.WORLD], _],
        surface: SurfaceId[1],
    ):
        for v in vertices:
            self.triangle_vertices.append(v)
        for _ in range(len(vertices) / 3):
            self.triangle_surfaces.append(surface.copy())

    def add_triangle_mesh_instance(
        mut self,
        vertices: ImmSpan[Point3f32[Frame.LOCAL], _],
        transform: Affine3f32[Frame.LOCAL, Frame.WORLD],
        bounds: AABB[Frame.LOCAL],
        surface: SurfaceId[1],
    ) -> UInt32:
        var mesh_idx = UInt32(len(self.triangle_meshes))
        var owned_vertices = List[Point3f32[Frame.LOCAL]](
            capacity=len(vertices)
        )
        owned_vertices.extend(vertices)
        self.triangle_meshes.append(owned_vertices^)
        self.triangle_instances.append(
            Instance(transform, mesh_idx, bounds, Primitive.TRIANGLE)
        )
        self.triangle_instance_surfaces.append(surface.copy())
        return mesh_idx

    def add_triangle_instance(
        mut self,
        mesh_idx: UInt32,
        transform: Affine3f32[Frame.LOCAL, Frame.WORLD],
        mesh_bounds: AABB[Frame.LOCAL],
        surface: SurfaceId[1],
    ):
        self.triangle_instances.append(
            Instance(transform, mesh_idx, mesh_bounds, Primitive.TRIANGLE)
        )
        self.triangle_instance_surfaces.append(surface.copy())

    def finish(deinit self) raises -> SceneData:
        """Consume the builder and produce one validated immutable snapshot."""
        return SceneData(
            self.spheres^,
            self.sphere_surfaces^,
            self.triangle_vertices^,
            self.triangle_surfaces^,
            self.triangle_meshes^,
            self.triangle_instances^,
            self.triangle_instance_surfaces^,
            self.surfaces^,
        )


struct SceneData:
    """Validated backend-neutral scene snapshot.

    All geometry and material mutation happens in `SceneBuilder`. This type owns
    the finalized buffers plus the matching derived light distribution. CPU and
    GPU preparation may read it independently; neither can observe stale
    sidecars or alias weights.
    """

    var _spheres: List[Sphere[Frame.WORLD]]
    var _sphere_surfaces: List[SurfaceId[1]]
    var _triangle_vertices: List[Point3f32[Frame.WORLD]]
    var _triangle_surfaces: List[SurfaceId[1]]
    var _triangle_meshes: List[List[Point3f32[Frame.LOCAL]]]
    var _triangle_instances: List[Instance]
    var _triangle_instance_surfaces: List[SurfaceId[1]]
    var _surfaces: SurfaceStore
    var _lights: LightStore

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
    ) raises:
        self._spheres = spheres^
        self._sphere_surfaces = sphere_surfaces^
        self._triangle_vertices = triangle_vertices^
        self._triangle_surfaces = triangle_surfaces^
        self._triangle_meshes = triangle_meshes^
        self._triangle_instances = triangle_instances^
        self._triangle_instance_surfaces = triangle_instance_surfaces^
        self._surfaces = surfaces^
        self._lights = LightStore()
        self._validate()
        self._build_light_store()
        self._lights.build_alias_table()

    def spheres(self) -> ref[self._spheres] List[Sphere[Frame.WORLD]]:
        return self._spheres

    def sphere_surfaces(
        self,
    ) -> ref[self._sphere_surfaces] List[SurfaceId[1]]:
        return self._sphere_surfaces

    def triangle_vertices(
        self,
    ) -> ref[self._triangle_vertices] List[Point3f32[Frame.WORLD]]:
        return self._triangle_vertices

    def triangle_surfaces(
        self,
    ) -> ref[self._triangle_surfaces] List[SurfaceId[1]]:
        return self._triangle_surfaces

    def triangle_meshes(
        self,
    ) -> ref[self._triangle_meshes] List[List[Point3f32[Frame.LOCAL]]]:
        return self._triangle_meshes

    def triangle_instances(
        self,
    ) -> ref[self._triangle_instances] List[Instance]:
        return self._triangle_instances

    def triangle_instance_surfaces(
        self,
    ) -> ref[self._triangle_instance_surfaces] List[SurfaceId[1]]:
        return self._triangle_instance_surfaces

    def surfaces(self) -> ref[self._surfaces] SurfaceStore:
        return self._surfaces

    def lights(self) -> ref[self._lights] LightStore:
        return self._lights

    def _validate(self) raises:
        if (
            len(self._spheres) == 0
            and len(self._triangle_vertices) == 0
            and len(self._triangle_instances) == 0
        ):
            raise Error("scene requires at least one primitive")
        if len(self._spheres) != len(self._sphere_surfaces):
            raise Error("sphere and surface sidecar lengths must match")
        if len(self._triangle_vertices) % 3 != 0:
            raise Error("triangle vertex count must be a multiple of three")
        if len(self._triangle_vertices) / 3 != len(self._triangle_surfaces):
            raise Error("triangle and surface sidecar lengths must match")
        if len(self._triangle_instances) != len(
            self._triangle_instance_surfaces
        ):
            raise Error(
                "triangle instance and surface sidecar lengths must match"
            )

        for i, sphere in enumerate(self._spheres):
            if sphere.radius == 0.0:
                raise Error("sphere radius must be non-zero")
            if not self._surfaces.validate(self._sphere_surfaces[i]):
                raise Error("sphere surface id is out of range")

        for surface in self._triangle_surfaces:
            if not self._surfaces.validate(surface):
                raise Error("triangle surface id is out of range")

        for vertices in self._triangle_meshes:
            if len(vertices) == 0 or len(vertices) % 3 != 0:
                raise Error(
                    "triangle mesh vertex count must be a positive multiple of"
                    " three"
                )

        for i, inst in enumerate(self._triangle_instances):
            if inst.kind != Primitive.TRIANGLE:
                raise Error(
                    "triangle instance must have triangle primitive kind"
                )
            if inst.blas_idx >= UInt32(len(self._triangle_meshes)):
                raise Error("triangle instance blas_idx is out of range")
            var surface = self._triangle_instance_surfaces[i].copy()
            if not self._surfaces.validate(surface):
                raise Error("triangle instance surface id is out of range")
            if surface.kind() == MAT.EMISSIVE:
                raise Error(
                    "emissive triangle instances are not supported by the light"
                    " sampler"
                )

    def _build_light_store(mut self):
        for idx, surface in enumerate(self._triangle_surfaces):
            if surface.kind() == MAT.EMISSIVE:
                var radiance = self._surfaces.emissives[
                    Int(surface.index())
                ].radiance
                var weight = _scene_triangle_area(self, idx) * (
                    _light_importance(radiance)
                )
                if weight > 0.0:
                    self._lights.append(
                        LightRecord(
                            PrimitiveId(PRIM.TRIANGLE, UInt32(idx)),
                            surface.copy(),
                            weight,
                        )
                    )

        for idx, surface in enumerate(self._sphere_surfaces):
            if surface.kind() == MAT.EMISSIVE:
                var radiance = self._surfaces.emissives[
                    Int(surface.index())
                ].radiance
                var radius = sphere_unsigned_radius(self._spheres[idx])
                var weight = (
                    4.0 * pi * radius * radius * _light_importance(radiance)
                )
                if weight > 0.0:
                    self._lights.append(
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
    ref v0 = scene.triangle_vertices()[3 * triangle_index + 0]
    ref v1 = scene.triangle_vertices()[3 * triangle_index + 1]
    ref v2 = scene.triangle_vertices()[3 * triangle_index + 2]
    return 0.5 * sqrt(length2(cross(v1 - v0, v2 - v0)))


def ray_at(ray: Rayf32[Frame.WORLD], t: Float32) -> Point3f32[Frame.WORLD]:
    return ray.o + t * ray.d
