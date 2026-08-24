from std.math import pi, sqrt
from std.utils.numerics import isfinite

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
from bajo.bvh.constants import PrimitiveKind, f32_max
from bajo.bvh import Instance, Sphere
from bajo.rt.geometry import sphere_unsigned_radius


comptime Color = Vec3f32[.WORLD]


@fieldwise_init
struct MaterialKind(Equatable, TrivialRegisterPassable, Writable):
    var value: UInt32
    comptime LAMBERTIAN = Self(0)
    comptime METAL = Self(1)
    comptime DIELECTRIC = Self(2)
    comptime EMISSIVE = Self(3)


comptime SURFACE_KIND_BITS = UInt32(4)
comptime SURFACE_INDEX_BITS = 32 - SURFACE_KIND_BITS
comptime SURFACE_INDEX_MASK = UInt32((1 << SURFACE_INDEX_BITS) - 1)


comptime PRIMITIVE_KIND_BITS = UInt32(4)
comptime PRIMITIVE_INDEX_BITS = 32 - PRIMITIVE_KIND_BITS
comptime PRIMITIVE_INDEX_MASK = UInt32((1 << PRIMITIVE_INDEX_BITS) - 1)


@fieldwise_init
struct Integrator(Equatable, TrivialRegisterPassable, Writable):
    var value: UInt32
    comptime PATH = Self(0)
    comptime NORMALS = Self(1)
    comptime AO = Self(2)
    comptime NEE = Self(3)
    comptime MIS = Self(4)

    def is_valid(self) -> Bool:
        return self in (
            Integrator.PATH,
            Integrator.NORMALS,
            Integrator.AO,
            Integrator.NEE,
            Integrator.MIS,
        )


@fieldwise_init
struct PrimitiveId(Copyable, Writable):
    var value: UInt32

    def __init__(out self, kind: PrimitiveKind, index: UInt32):
        debug_assert["safe", _use_compiler_assume=True](
            kind.value < (UInt32(1) << PRIMITIVE_KIND_BITS)
        )
        debug_assert["safe", _use_compiler_assume=True](
            index < (UInt32(1) << PRIMITIVE_INDEX_BITS)
        )
        self.value = (kind.value << PRIMITIVE_INDEX_BITS) | index

    def kind(self) -> PrimitiveKind:
        return PrimitiveKind(self.value >> PRIMITIVE_INDEX_BITS)

    def index(self) -> UInt32:
        return self.value & PRIMITIVE_INDEX_MASK


@fieldwise_init
struct SurfaceId[length: SIMDLength = 1](Copyable, Writable):
    var value: SIMD[.uint32, Self.length]

    def __init__(out self, kind: MaterialKind, index: UInt32):
        debug_assert["safe", _use_compiler_assume=True](
            kind.value < (UInt32(1) << SURFACE_KIND_BITS)
        )
        debug_assert["safe", _use_compiler_assume=True](
            index < (UInt32(1) << SURFACE_INDEX_BITS)
        )
        self.value = (kind.value << SURFACE_INDEX_BITS) | index

    @always_inline
    def kind(self) -> MaterialKind:
        comptime assert Self.length == 1
        return MaterialKind(self.value[0] >> SURFACE_INDEX_BITS)

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
        if surface.kind() == .LAMBERTIAN:
            return surface.index() < UInt32(len(self.lambertians))
        elif surface.kind() == .METAL:
            return surface.index() < UInt32(len(self.metals))
        elif surface.kind() == .DIELECTRIC:
            return surface.index() < UInt32(len(self.dielectrics))
        elif surface.kind() == .EMISSIVE:
            return surface.index() < UInt32(len(self.emissives))

        return False

    def add_lambertian(mut self, albedo: Color) -> SurfaceId[1]:
        var index = UInt32(len(self.lambertians))
        self.lambertians.append(Lambertian(albedo))
        return SurfaceId(.LAMBERTIAN, index)

    def add_metal(mut self, albedo: Color, fuzz: Float32) -> SurfaceId[1]:
        var index = UInt32(len(self.metals))
        self.metals.append(Metal(albedo, fuzz))
        return SurfaceId(.METAL, index)

    def add_dielectric(mut self, refraction_index: Float32) -> SurfaceId[1]:
        var index = UInt32(len(self.dielectrics))
        self.dielectrics.append(Dielectric(refraction_index))
        return SurfaceId(.DIELECTRIC, index)

    def add_emissive(mut self, radiance: Color) -> SurfaceId[1]:
        var index = UInt32(len(self.emissives))
        self.emissives.append(Emissive(radiance))
        return SurfaceId(.EMISSIVE, index)


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
    var t: SIMD[.float32, Self.length]
    var front_face: SIMD[.bool, Self.length]
    var hit: SIMD[.bool, Self.length]

    def __init__(out self, t_max: SIMD[.float32, Self.length]):
        self.normal = Vec3[.float32, .WORLD, Self.length](0.0)
        self.surface = SurfaceId[Self.length](SIMD[.uint32, Self.length](0))
        self.t = t_max
        self.front_face = SIMD[.bool, Self.length](fill=True)
        self.hit = SIMD[.bool, Self.length](fill=False)

    @staticmethod
    def miss(t: SIMD[.float32, Self.length] = f32_max) -> Self:
        return Self(t)

    @always_inline
    def get(self, lane: Int) -> SurfaceHit[1]:
        return SurfaceHit[1](
            Vec3f32[.WORLD](
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
    var p: Point3[.float32, .WORLD, Self.length]
    var normal: Vec3[.float32, .WORLD, Self.length]
    var front_face: SIMD[.bool, Self.length]


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
    var spheres: List[Sphere[.WORLD]]
    var sphere_surfaces: List[SurfaceId[1]]
    var triangle_vertices: List[Point3f32[.WORLD]]
    var triangle_surfaces: List[SurfaceId[1]]
    var triangle_meshes: List[List[Point3f32[.LOCAL]]]
    var triangle_instances: List[Instance]
    var triangle_instance_surfaces: List[SurfaceId[1]]
    var surfaces: SurfaceStore

    def __init__(out self):
        self.spheres = List[Sphere[.WORLD]]()
        self.sphere_surfaces = List[SurfaceId[1]]()
        self.triangle_vertices = List[Point3f32[.WORLD]]()
        self.triangle_surfaces = List[SurfaceId[1]]()
        self.triangle_meshes = List[List[Point3f32[.LOCAL]]]()
        self.triangle_instances = List[Instance]()
        self.triangle_instance_surfaces = List[SurfaceId[1]]()
        self.surfaces = SurfaceStore()

    def __init__(
        out self,
        var spheres: List[Sphere[.WORLD]],
        var sphere_surfaces: List[SurfaceId[1]],
        var triangle_vertices: List[Point3f32[.WORLD]],
        var triangle_surfaces: List[SurfaceId[1]],
        var triangle_meshes: List[List[Point3f32[.LOCAL]]],
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
        center: Point3f32[.WORLD],
        radius: Float32,
        surface: SurfaceId[1],
    ):
        self.spheres.append(Sphere[.WORLD](center, radius))
        self.sphere_surfaces.append(surface.copy())

    def add_triangle(
        mut self,
        v0: Point3f32[.WORLD],
        v1: Point3f32[.WORLD],
        v2: Point3f32[.WORLD],
        surface: SurfaceId[1],
    ):
        self.triangle_vertices.append(v0)
        self.triangle_vertices.append(v1)
        self.triangle_vertices.append(v2)
        self.triangle_surfaces.append(surface.copy())

    def add_quad(
        mut self,
        a: Point3f32[.WORLD],
        b: Point3f32[.WORLD],
        c: Point3f32[.WORLD],
        d: Point3f32[.WORLD],
        surface: SurfaceId[1],
    ):
        """Append two consistently wound triangles: `(a, b, c)` and `(a, c, d)`.
        """
        self.add_triangle(a, b, c, surface)
        self.add_triangle(a, c, d, surface)

    def add_triangle_mesh(
        mut self,
        vertices: ImmSpan[Point3f32[.WORLD], _],
        surface: SurfaceId[1],
    ):
        for v in vertices:
            self.triangle_vertices.append(v)
        for _ in range(len(vertices) / 3):
            self.triangle_surfaces.append(surface.copy())

    def add_triangle_mesh_instance(
        mut self,
        vertices: ImmSpan[Point3f32[.LOCAL], _],
        transform: Affine3f32[.LOCAL, .WORLD],
        bounds: AABB[.LOCAL],
        surface: SurfaceId[1],
    ) -> UInt32:
        var mesh_idx = UInt32(len(self.triangle_meshes))
        var owned_vertices = List[Point3f32[.LOCAL]](capacity=len(vertices))
        owned_vertices.extend(vertices)
        self.triangle_meshes.append(owned_vertices^)
        self._add_triangle_instance_unchecked(mesh_idx, transform, bounds)
        self.triangle_instance_surfaces.append(surface.copy())
        return mesh_idx

    def add_triangle_instance(
        mut self,
        mesh_idx: UInt32,
        transform: Affine3f32[.LOCAL, .WORLD],
        mesh_bounds: AABB[.LOCAL],
        surface: SurfaceId[1],
    ):
        self._add_triangle_instance_unchecked(mesh_idx, transform, mesh_bounds)
        self.triangle_instance_surfaces.append(surface.copy())

    def _add_triangle_instance_unchecked(
        mut self,
        mesh_idx: UInt32,
        transform: Affine3f32[.LOCAL, .WORLD],
        mesh_bounds: AABB[.LOCAL],
    ):
        var instance = Instance()
        instance.transform = transform.copy()
        instance.bounds = mesh_bounds.apply_transform(transform)
        instance.blas_idx = mesh_idx
        instance.kind = .TRIANGLE
        self.triangle_instances.append(instance^)

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
    """Validated backend-neutral scene snapshot."""

    var _spheres: List[Sphere[.WORLD]]
    var _sphere_surfaces: List[SurfaceId[1]]
    var _triangle_vertices: List[Point3f32[.WORLD]]
    var _triangle_surfaces: List[SurfaceId[1]]
    var _triangle_meshes: List[List[Point3f32[.LOCAL]]]
    var _triangle_instances: List[Instance]
    var _triangle_instance_surfaces: List[SurfaceId[1]]
    var _surfaces: SurfaceStore
    var _lights: LightStore

    def __init__(
        out self,
        var spheres: List[Sphere[.WORLD]],
        var sphere_surfaces: List[SurfaceId[1]],
        var triangle_vertices: List[Point3f32[.WORLD]],
        var triangle_surfaces: List[SurfaceId[1]],
        var triangle_meshes: List[List[Point3f32[.LOCAL]]],
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

    def spheres(self) -> ref[self._spheres] List[Sphere[.WORLD]]:
        return self._spheres

    def sphere_surfaces(
        self,
    ) -> ref[self._sphere_surfaces] List[SurfaceId[1]]:
        return self._sphere_surfaces

    def triangle_vertices(
        self,
    ) -> ref[self._triangle_vertices] List[Point3f32[.WORLD]]:
        return self._triangle_vertices

    def triangle_surfaces(
        self,
    ) -> ref[self._triangle_surfaces] List[SurfaceId[1]]:
        return self._triangle_surfaces

    def triangle_meshes(
        self,
    ) -> ref[self._triangle_meshes] List[List[Point3f32[.LOCAL]]]:
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

    def _validate(mut self) raises:
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

        self._validate_materials()

        for i, sphere in enumerate(self._spheres):
            if not _point_is_finite(sphere.center):
                raise Error("sphere center must be finite")
            if not isfinite(sphere.radius):
                raise Error("sphere radius must be finite")
            if sphere.radius == 0.0:
                raise Error("sphere radius must be non-zero")
            var radius = sphere_unsigned_radius(sphere)
            if not (
                isfinite(sphere.center.x[0] - radius)
                and isfinite(sphere.center.y[0] - radius)
                and isfinite(sphere.center.z[0] - radius)
                and isfinite(sphere.center.x[0] + radius)
                and isfinite(sphere.center.y[0] + radius)
                and isfinite(sphere.center.z[0] + radius)
            ):
                raise Error("sphere bounds must be finite")
            if not self._surfaces.validate(self._sphere_surfaces[i]):
                raise Error("sphere surface id is out of range")

        for triangle_idx, surface in enumerate(self._triangle_surfaces):
            if not self._surfaces.validate(surface):
                raise Error("triangle surface id is out of range")
            var base = 3 * triangle_idx
            if not _triangle_is_valid(
                self._triangle_vertices[base],
                self._triangle_vertices[base + 1],
                self._triangle_vertices[base + 2],
            ):
                raise Error(
                    "triangle vertices must be finite and non-degenerate"
                )

        for vertices in self._triangle_meshes:
            if len(vertices) == 0 or len(vertices) % 3 != 0:
                raise Error(
                    "triangle mesh vertex count must be a positive multiple of"
                    " three"
                )
            for triangle_idx in range(len(vertices) / 3):
                var base = 3 * triangle_idx
                if not _triangle_is_valid(
                    vertices[base], vertices[base + 1], vertices[base + 2]
                ):
                    raise Error(
                        "triangle mesh vertices must be finite and"
                        " non-degenerate"
                    )

        for i, inst in enumerate(self._triangle_instances):
            if inst.kind != .TRIANGLE:
                raise Error(
                    "triangle instance must have triangle primitive kind"
                )
            if inst.blas_idx >= UInt32(len(self._triangle_meshes)):
                raise Error("triangle instance blas_idx is out of range")
            var surface = self._triangle_instance_surfaces[i].copy()
            if not self._surfaces.validate(surface):
                raise Error("triangle instance surface id is out of range")

            if not _transform_is_finite(inst.transform):
                raise Error("triangle instance transform must be finite")
            var inverse = inst.transform.inverse()
            if not inverse.mask[0] or not _transform_is_finite(inverse.inv):
                raise Error("triangle instance transform must be invertible")

            ref vertices = self._triangle_meshes[Int(inst.blas_idx)]
            var local_bounds = AABB[.LOCAL].invalid()
            for vertex in vertices:
                local_bounds.grow(vertex)
            var world_bounds = local_bounds.apply_transform(inst.transform)
            if not _bounds_are_valid(world_bounds):
                raise Error(
                    "triangle instance transformed bounds must be finite"
                )

            ref finalized_instance = self._triangle_instances[i]
            finalized_instance.inv_transform = inverse.inv.copy()
            finalized_instance.bounds = world_bounds

    def _validate_materials(self) raises:
        for material in self._surfaces.lambertians:
            if not _color_is_unit_interval(material.albedo):
                raise Error(
                    "lambertian albedo must be finite and within [0, 1]"
                )

        for material in self._surfaces.metals:
            if not _color_is_unit_interval(material.albedo):
                raise Error("metal albedo must be finite and within [0, 1]")
            if not isfinite(material.fuzz) or not (
                material.fuzz >= 0.0 and material.fuzz <= 1.0
            ):
                raise Error("metal fuzz must be finite and within [0, 1]")

        for material in self._surfaces.dielectrics:
            if not isfinite(material.refraction_index) or (
                material.refraction_index <= 0.0
            ):
                raise Error(
                    "dielectric refraction index must be finite and positive"
                )

        for material in self._surfaces.emissives:
            if not _color_is_finite_nonnegative(material.radiance):
                raise Error("emissive radiance must be finite and non-negative")

    def _build_light_store(mut self) raises:
        for idx, surface in enumerate(self._triangle_surfaces):
            if surface.kind() == .EMISSIVE:
                var radiance = self._surfaces.emissives[
                    Int(surface.index())
                ].radiance
                ref p0 = self._triangle_vertices[3 * idx + 0]
                ref p1 = self._triangle_vertices[3 * idx + 1]
                ref p2 = self._triangle_vertices[3 * idx + 2]
                var weight = _triangle_area(p0, p1, p2) * _light_importance(
                    radiance
                )
                if not isfinite(weight):
                    raise Error("triangle light weight must be finite")
                if weight > 0.0:
                    self._append_light(
                        LightRecord.triangle(
                            PrimitiveId(PrimitiveKind.TRIANGLE, UInt32(idx)),
                            surface.copy(),
                            weight,
                            p0,
                            p1,
                            p2,
                        )
                    )

        for idx, surface in enumerate(self._sphere_surfaces):
            if surface.kind() == .EMISSIVE:
                var radiance = self._surfaces.emissives[
                    Int(surface.index())
                ].radiance
                var radius = sphere_unsigned_radius(self._spheres[idx])
                var weight = (
                    4.0 * pi * radius * radius * _light_importance(radiance)
                )
                if not isfinite(weight):
                    raise Error("sphere light weight must be finite")
                if weight > 0.0:
                    self._append_light(
                        LightRecord.sphere(
                            PrimitiveId(PrimitiveKind.SPHERE, UInt32(idx)),
                            surface.copy(),
                            weight,
                            self._spheres[idx].center,
                            radius,
                        )
                    )

        for instance_idx, surface in enumerate(
            self._triangle_instance_surfaces
        ):
            if surface.kind() != .EMISSIVE:
                continue
            var radiance = self._surfaces.emissives[
                Int(surface.index())
            ].radiance
            var transform = self._triangle_instances[
                instance_idx
            ].transform.copy()
            var mesh_idx = Int(self._triangle_instances[instance_idx].blas_idx)
            var reverses_orientation = _transform_reverses_orientation(
                transform
            )
            var triangle_count = len(self._triangle_meshes[mesh_idx]) / 3
            for triangle_idx in range(triangle_count):
                var base = 3 * triangle_idx
                var p0 = transform.point(
                    self._triangle_meshes[mesh_idx][base + 0]
                )
                var p1 = transform.point(
                    self._triangle_meshes[mesh_idx][base + 1]
                )
                var p2 = transform.point(
                    self._triangle_meshes[mesh_idx][base + 2]
                )
                if reverses_orientation:
                    var tmp = p1
                    p1 = p2
                    p2 = tmp
                var weight = _triangle_area(p0, p1, p2) * (
                    _light_importance(radiance)
                )
                if not isfinite(weight):
                    raise Error("triangle instance light weight must be finite")
                if weight > 0.0:
                    self._append_light(
                        LightRecord.triangle(
                            PrimitiveId(
                                PrimitiveKind.TRIANGLE_INSTANCE,
                                UInt32(instance_idx),
                            ),
                            surface.copy(),
                            weight,
                            p0,
                            p1,
                            p2,
                        )
                    )

    def _append_light(mut self, var light: LightRecord) raises:
        self._lights.append(light^)
        if not isfinite(self._lights.total_weight):
            raise Error("total light weight must be finite")


@always_inline
def _point_is_finite[frame: Frame](point: Point3f32[frame]) -> Bool:
    return (
        isfinite(point.x[0]) and isfinite(point.y[0]) and isfinite(point.z[0])
    )


@always_inline
def _color_is_unit_interval(color: Color) -> Bool:
    return (
        isfinite(color.x[0])
        and isfinite(color.y[0])
        and isfinite(color.z[0])
        and color.x[0] >= 0.0
        and color.x[0] <= 1.0
        and color.y[0] >= 0.0
        and color.y[0] <= 1.0
        and color.z[0] >= 0.0
        and color.z[0] <= 1.0
    )


@always_inline
def _color_is_finite_nonnegative(color: Color) -> Bool:
    return (
        isfinite(color.x[0])
        and isfinite(color.y[0])
        and isfinite(color.z[0])
        and color.x[0] >= 0.0
        and color.y[0] >= 0.0
        and color.z[0] >= 0.0
    )


@always_inline
def _triangle_is_valid[
    frame: Frame
](v0: Point3f32[frame], v1: Point3f32[frame], v2: Point3f32[frame],) -> Bool:
    if not (
        _point_is_finite(v0) and _point_is_finite(v1) and _point_is_finite(v2)
    ):
        return False
    var twice_area_squared = length2(cross(v1 - v0, v2 - v0))[0]
    return isfinite(twice_area_squared) and twice_area_squared > 0.0


@always_inline
def _transform_is_finite[
    From: Frame, To: Frame
](transform: Affine3f32[From, To]) -> Bool:
    return (
        isfinite(transform.m00[0])
        and isfinite(transform.m01[0])
        and isfinite(transform.m02[0])
        and isfinite(transform.tx[0])
        and isfinite(transform.m10[0])
        and isfinite(transform.m11[0])
        and isfinite(transform.m12[0])
        and isfinite(transform.ty[0])
        and isfinite(transform.m20[0])
        and isfinite(transform.m21[0])
        and isfinite(transform.m22[0])
        and isfinite(transform.tz[0])
    )


@always_inline
def _bounds_are_valid[frame: Frame](bounds: AABB[frame]) -> Bool:
    return (
        _point_is_finite(bounds._min)
        and _point_is_finite(bounds._max)
        and bounds._min.x[0] <= bounds._max.x[0]
        and bounds._min.y[0] <= bounds._max.y[0]
        and bounds._min.z[0] <= bounds._max.z[0]
    )


@always_inline
def _light_importance(radiance: Color) -> Float32:
    return max((radiance.x + radiance.y + radiance.z) / 3.0, 0.0)


@always_inline
def _triangle_area[
    frame: Frame
](v0: Point3f32[frame], v1: Point3f32[frame], v2: Point3f32[frame]) -> Float32:
    return 0.5 * sqrt(length2(cross(v1 - v0, v2 - v0)))


@always_inline
def _transform_reverses_orientation(
    transform: Affine3f32[.LOCAL, .WORLD],
) -> Bool:
    var determinant = (
        transform.m00
        * (transform.m11 * transform.m22 - transform.m12 * transform.m21)
        - transform.m01
        * (transform.m10 * transform.m22 - transform.m12 * transform.m20)
        + transform.m02
        * (transform.m10 * transform.m21 - transform.m11 * transform.m20)
    )
    return determinant[0] < 0.0


def ray_at(ray: Rayf32[.WORLD], t: Float32) -> Point3f32[.WORLD]:
    return ray.o + t * ray.d
