"""Material, texture, environment, and surface storage types."""

from std.math import abs, floor
from std.utils.numerics import isfinite

from bajo.core import Affine3f32, Vec3f32, cross, normalize
from bajo.bvh.constants import PrimitiveKind
from bajo.rt.render_types import Color

comptime NO_TEXTURE = UInt32.MAX


struct ImageTexture:
    """Linear RGB image used by host-side material sampling."""

    var width: Int
    var height: Int
    var pixels: List[Float32]

    def __init__(out self, width: Int, height: Int, var pixels: List[Float32]):
        self.width = width
        self.height = height
        self.pixels = pixels^

    def sample(self, u: Float32, v: Float32) -> Color:
        # PBRT repeats UV image maps. PNG scanlines are top-to-bottom while
        # PBRT's texture coordinate origin is at the lower left.
        var wrapped_u = u - floor(u)
        var wrapped_v = v - floor(v)
        var x = min(Int(wrapped_u * Float32(self.width)), self.width - 1)
        var y = min(
            Int((Float32(1.0) - wrapped_v) * Float32(self.height)),
            self.height - 1,
        )
        var base = 3 * (y * self.width + x)
        return Color(
            self.pixels[base],
            self.pixels[base + 1],
            self.pixels[base + 2],
        )

    def sample_scalar(self, u: Float32, v: Float32) -> Float32:
        var color = self.sample(u, v)
        return (
            Float32(0.2126) * color.x[0]
            + Float32(0.7152) * color.y[0]
            + Float32(0.0722) * color.z[0]
        )

    def sample_environment(self, u: Float32, v: Float32) -> Color:
        """Nearest lookup in PBRT's top-to-bottom equal-area image domain."""
        var x = min(max(Int(u * Float32(self.width)), 0), self.width - 1)
        var y = min(max(Int(v * Float32(self.height)), 0), self.height - 1)
        var base = 3 * (y * self.width + x)
        return Color(
            max(self.pixels[base], 0.0),
            max(self.pixels[base + 1], 0.0),
            max(self.pixels[base + 2], 0.0),
        )


@fieldwise_init
struct EnvironmentKind(Equatable, TrivialRegisterPassable, Writable):
    var value: UInt32
    comptime BLACK = Self(0)
    comptime PROCEDURAL = Self(1)
    comptime UNIFORM = Self(2)
    comptime IMAGE = Self(3)


struct Environment(Copyable, Writable):
    """Scene-owned radiance for rays that leave the world."""

    var kind: EnvironmentKind
    var texture_index: UInt32
    var scale: Color
    var world_to_light: Affine3f32[.WORLD, .LOCAL]
    var light_to_world: Affine3f32[.LOCAL, .WORLD]

    def __init__(out self):
        self.kind = .PROCEDURAL
        self.texture_index = NO_TEXTURE
        self.scale = Color(1.0)
        self.world_to_light = Affine3f32[.WORLD, .LOCAL].identity()
        self.light_to_world = Affine3f32[.LOCAL, .WORLD].identity()

    @staticmethod
    def black() -> Self:
        var result = Self()
        result.kind = .BLACK
        result.scale = Color(0.0)
        return result^

    @staticmethod
    def uniform(radiance: Color) -> Self:
        var result = Self()
        result.kind = .UNIFORM
        result.scale = radiance
        return result^

    @staticmethod
    def image(
        texture_index: UInt32,
        scale: Color,
        world_to_light: Affine3f32[.WORLD, .LOCAL],
        light_to_world: Affine3f32[.LOCAL, .WORLD],
    ) -> Self:
        var result = Self()
        result.kind = .IMAGE
        result.texture_index = texture_index
        result.scale = scale
        result.world_to_light = world_to_light.copy()
        result.light_to_world = light_to_world.copy()
        return result^


@fieldwise_init
struct MaterialKind(EnumLike, Equatable, TrivialRegisterPassable, Writable):
    var value: UInt32
    comptime LAMBERTIAN = Self(0)
    comptime METAL = Self(1)
    comptime DIELECTRIC = Self(2)
    comptime EMISSIVE = Self(3)
    comptime COATED_DIFFUSE = Self(4)
    comptime _enum_case_names = ParameterList.of[
        "LAMBERTIAN".value,
        "METAL".value,
        "DIELECTRIC".value,
        "EMISSIVE".value,
        "COATED_DIFFUSE".value,
    ].values
    comptime _enum_case_types = TypeList.of[
        Trait=AnyType, NoneType, NoneType, NoneType, NoneType, NoneType
    ].values
    comptime has_bsdf[kind: Self] = (
        kind.value == Self.LAMBERTIAN.value
        or kind.value == Self.METAL.value
        or kind.value == Self.DIELECTRIC.value
        or kind.value == Self.COATED_DIFFUSE.value
    )

    def _get_enum_discriminant(self) -> Int:
        return Int(self.value)

    def _unsafe_get_enum_payload[
        id: Int
    ](ref self) -> ref[self] TypeList[Trait=AnyType, Self._enum_case_types]()[
        id
    ]:
        while True:
            pass


comptime SURFACE_KIND_BITS = UInt32(4)
comptime SURFACE_INDEX_BITS = 32 - SURFACE_KIND_BITS
comptime SURFACE_INDEX_MASK = UInt32((1 << SURFACE_INDEX_BITS) - 1)


comptime PRIMITIVE_KIND_BITS = UInt32(4)
comptime PRIMITIVE_INDEX_BITS = 32 - PRIMITIVE_KIND_BITS
comptime PRIMITIVE_INDEX_MASK = UInt32((1 << PRIMITIVE_INDEX_BITS) - 1)


@fieldwise_init
struct Integrator(EnumLike, Equatable, TrivialRegisterPassable, Writable):
    var value: UInt32
    comptime PATH = Self(0)
    comptime NORMALS = Self(1)
    comptime AO = Self(2)
    comptime NEE = Self(3)
    comptime MIS = Self(4)
    comptime _enum_case_names = ParameterList.of[
        "PATH".value,
        "NORMALS".value,
        "AO".value,
        "NEE".value,
        "MIS".value,
    ].values
    comptime _enum_case_types = TypeList.of[
        Trait=AnyType, NoneType, NoneType, NoneType, NoneType, NoneType
    ].values
    comptime is_path_tracing[integrator: Self] = (
        integrator.value == Self.PATH.value
        or integrator.value == Self.NEE.value
        or integrator.value == Self.MIS.value
    )
    comptime uses_direct_lighting[integrator: Self] = (
        integrator.value == Self.NEE.value or integrator.value == Self.MIS.value
    )
    comptime uses_visibility[integrator: Self] = (
        integrator.value == Self.AO.value
        or Self.uses_direct_lighting[integrator]
    )

    def _get_enum_discriminant(self) -> Int:
        return Int(self.value)

    def _unsafe_get_enum_payload[
        id: Int
    ](ref self) -> ref[self] TypeList[Trait=AnyType, Self._enum_case_types]()[
        id
    ]:
        while True:
            pass

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
        self.value = Self.pack_raw(kind, index)

    @staticmethod
    @always_inline
    def pack_raw(kind: MaterialKind, index: UInt32) -> UInt32:
        return (kind.value << SURFACE_INDEX_BITS) | index

    @always_inline
    def kind(self) -> MaterialKind:
        comptime assert Self.length == 1
        return Self.kind_from_raw(self.value[0])

    @always_inline
    def index(self) -> UInt32:
        comptime assert Self.length == 1
        return Self.index_from_raw(self.value[0])

    @staticmethod
    @always_inline
    def from_raw(value: UInt32) -> SurfaceId[1]:
        return SurfaceId[1](SIMD[.uint32, 1](value))

    @staticmethod
    @always_inline
    def kind_from_raw(value: UInt32) -> MaterialKind:
        return MaterialKind(value >> SURFACE_INDEX_BITS)

    @staticmethod
    @always_inline
    def index_from_raw(value: UInt32) -> UInt32:
        return value & SURFACE_INDEX_MASK

    @always_inline
    def get(self, lane: Int) -> SurfaceId[1]:
        return SurfaceId[1](self.value[lane])


struct Lambertian(Copyable, Writable):
    var albedo: Color
    var texture_index: UInt32

    def __init__(out self, albedo: Color, texture_index: UInt32 = NO_TEXTURE):
        self.albedo = albedo
        self.texture_index = texture_index

    def validate(self) raises:
        if not self.albedo.is_finite()[0]:
            raise Error("lambertian albedo must be finite")
        if (
            self.albedo.x[0] < 0.0
            or self.albedo.x[0] > 1.0
            or self.albedo.y[0] < 0.0
            or self.albedo.y[0] > 1.0
            or self.albedo.z[0] < 0.0
            or self.albedo.z[0] > 1.0
        ):
            raise Error("lambertian albedo must be within [0, 1]")


struct Metal(Copyable, Writable):
    var albedo: Color
    var fuzz: Float32
    var texture_index: UInt32

    def __init__(
        out self,
        albedo: Color,
        fuzz: Float32,
        texture_index: UInt32 = NO_TEXTURE,
    ):
        self.albedo = albedo
        self.fuzz = fuzz
        self.texture_index = texture_index

    def validate(self) raises:
        if not self.albedo.is_finite()[0]:
            raise Error("metal albedo must be finite")
        if (
            self.albedo.x[0] < 0.0
            or self.albedo.x[0] > 1.0
            or self.albedo.y[0] < 0.0
            or self.albedo.y[0] > 1.0
            or self.albedo.z[0] < 0.0
            or self.albedo.z[0] > 1.0
        ):
            raise Error("metal albedo must be within [0, 1]")
        if not isfinite(self.fuzz):
            raise Error("metal fuzz must be finite")
        if self.fuzz < 0.0 or self.fuzz > 1.0:
            raise Error("metal fuzz must be within [0, 1]")


@fieldwise_init
struct CoatedDiffuse(Copyable, Writable):
    """Diffuse substrate below a rough dielectric coating."""

    var albedo: Color
    var roughness: Float32
    var eta: Float32
    var texture_index: UInt32
    var displacement_texture_index: UInt32
    var displacement_scale: Float32
    var displacement_u_scale: Float32
    var displacement_v_scale: Float32
    var thickness: Float32
    var layer_albedo: Color
    var g: Float32
    var max_depth: Int
    var n_samples: Int

    def validate(self) raises:
        if not self.albedo.is_finite()[0]:
            raise Error("coated diffuse albedo must be finite")
        if (
            self.albedo.x[0] < 0.0
            or self.albedo.x[0] > 1.0
            or self.albedo.y[0] < 0.0
            or self.albedo.y[0] > 1.0
            or self.albedo.z[0] < 0.0
            or self.albedo.z[0] > 1.0
        ):
            raise Error("coated diffuse albedo must be within [0, 1]")
        if (
            not isfinite(self.roughness)
            or self.roughness < 0.0
            or self.roughness > 1.0
        ):
            raise Error("coated diffuse roughness must be within [0, 1]")
        if not isfinite(self.eta) or self.eta <= 0.0:
            raise Error("coated diffuse eta must be positive")
        if not isfinite(self.thickness) or self.thickness < 0.0:
            raise Error("coated diffuse thickness must be non-negative")
        if (
            not self.layer_albedo.is_finite()[0]
            or self.layer_albedo.x[0] < 0.0
            or self.layer_albedo.x[0] > 1.0
            or self.layer_albedo.y[0] < 0.0
            or self.layer_albedo.y[0] > 1.0
            or self.layer_albedo.z[0] < 0.0
            or self.layer_albedo.z[0] > 1.0
        ):
            raise Error("coated diffuse layer albedo must be within [0, 1]")
        if not isfinite(self.g) or self.g < -1.0 or self.g > 1.0:
            raise Error("coated diffuse g must be within [-1, 1]")
        if self.max_depth <= 0:
            raise Error("coated diffuse max depth must be positive")
        if self.n_samples <= 0:
            raise Error("coated diffuse sample count must be positive")
        if (
            not isfinite(self.displacement_scale)
            or not isfinite(self.displacement_u_scale)
            or not isfinite(self.displacement_v_scale)
        ):
            raise Error("coated diffuse displacement parameters must be finite")


@fieldwise_init
struct Dielectric(Copyable, Writable):
    var refraction_index: Float32

    def validate(self) raises:
        if not isfinite(self.refraction_index):
            raise Error("dielectric refraction index must be finite")
        if self.refraction_index <= 0.0:
            raise Error("dielectric refraction index must be positive")


@fieldwise_init
struct Emissive(Copyable, Writable):
    var radiance: Color

    def validate(self) raises:
        if not self.radiance.is_finite()[0]:
            raise Error("emissive radiance must be finite")
        if (
            self.radiance.x[0] < 0.0
            or self.radiance.y[0] < 0.0
            or self.radiance.z[0] < 0.0
        ):
            raise Error("emissive radiance must be non-negative")


struct SurfaceStore:
    var lambertians: List[Lambertian]
    var metals: List[Metal]
    var dielectrics: List[Dielectric]
    var emissives: List[Emissive]
    var coated_diffuses: List[CoatedDiffuse]
    var image_textures: List[ImageTexture]

    def __init__(out self):
        self.lambertians = List[Lambertian]()
        self.metals = List[Metal]()
        self.dielectrics = List[Dielectric]()
        self.emissives = List[Emissive]()
        self.coated_diffuses = List[CoatedDiffuse]()
        self.image_textures = List[ImageTexture]()

    def contains(self, surface: SurfaceId[1]) -> Bool:
        if surface.kind() == .LAMBERTIAN:
            return surface.index() < UInt32(len(self.lambertians))
        if surface.kind() == .METAL:
            return surface.index() < UInt32(len(self.metals))
        if surface.kind() == .DIELECTRIC:
            return surface.index() < UInt32(len(self.dielectrics))
        if surface.kind() == .EMISSIVE:
            return surface.index() < UInt32(len(self.emissives))
        if surface.kind() == .COATED_DIFFUSE:
            return surface.index() < UInt32(len(self.coated_diffuses))

        return False

    def emitted_radiance(
        self, surface: SurfaceId[1], front_face: Bool
    ) -> Color:
        if surface.kind() == .EMISSIVE and front_face:
            return self.emissives[Int(surface.index())].radiance
        return Color(0.0)

    def add_image_texture(mut self, var texture: ImageTexture) -> UInt32:
        var index = UInt32(len(self.image_textures))
        self.image_textures.append(texture^)
        return index

    def sample_albedo(
        self, surface: SurfaceId[1], u: Float32, v: Float32
    ) -> Color:
        if surface.kind() == .LAMBERTIAN:
            ref material = self.lambertians[Int(surface.index())]
            if material.texture_index != NO_TEXTURE:
                return material.albedo * self.image_textures[
                    Int(material.texture_index)
                ].sample(u, v)
            return material.albedo
        if surface.kind() == .METAL:
            ref material = self.metals[Int(surface.index())]
            if material.texture_index != NO_TEXTURE:
                return material.albedo * self.image_textures[
                    Int(material.texture_index)
                ].sample(u, v)
            return material.albedo
        if surface.kind() == .COATED_DIFFUSE:
            ref material = self.coated_diffuses[Int(surface.index())]
            if material.texture_index != NO_TEXTURE:
                return material.albedo * self.image_textures[
                    Int(material.texture_index)
                ].sample(u, v)
            return material.albedo
        return Color(0.0)

    def shading_normal(
        self,
        surface: SurfaceId[1],
        normal: Vec3f32[.WORLD],
        u: Float32,
        v: Float32,
    ) -> Vec3f32[.WORLD]:
        if surface.kind() != .COATED_DIFFUSE:
            return normal
        ref material = self.coated_diffuses[Int(surface.index())]
        if (
            material.displacement_texture_index == NO_TEXTURE
            or material.displacement_scale == 0.0
        ):
            return normal
        ref texture = self.image_textures[
            Int(material.displacement_texture_index)
        ]
        var texture_u = u * material.displacement_u_scale
        var texture_v = v * material.displacement_v_scale
        var du = Float32(0.5) / Float32(texture.width)
        var dv = Float32(0.5) / Float32(texture.height)
        var slope_u = (
            (
                texture.sample_scalar(texture_u + du, texture_v)
                - texture.sample_scalar(texture_u - du, texture_v)
            )
            * Float32(texture.width)
            * material.displacement_u_scale
            * material.displacement_scale
        )
        var slope_v = (
            (
                texture.sample_scalar(texture_u, texture_v + dv)
                - texture.sample_scalar(texture_u, texture_v - dv)
            )
            * Float32(texture.height)
            * material.displacement_v_scale
            * material.displacement_scale
        )
        var helper = Vec3f32[.WORLD](0.0, 1.0, 0.0)
        if abs(normal.y[0]) > 0.99:
            helper = Vec3f32[.WORLD](1.0, 0.0, 0.0)
        var tangent = normalize(cross(helper, normal))
        var bitangent = cross(normal, tangent)
        return normalize(normal - tangent * slope_u - bitangent * slope_v)

    def add_lambertian(
        mut self, albedo: Color, texture_index: UInt32 = NO_TEXTURE
    ) -> SurfaceId[1]:
        var index = UInt32(len(self.lambertians))
        self.lambertians.append(Lambertian(albedo, texture_index))
        return SurfaceId(.LAMBERTIAN, index)

    def add_metal(
        mut self,
        albedo: Color,
        fuzz: Float32,
        texture_index: UInt32 = NO_TEXTURE,
    ) -> SurfaceId[1]:
        var index = UInt32(len(self.metals))
        self.metals.append(Metal(albedo, fuzz, texture_index))
        return SurfaceId(.METAL, index)

    def add_dielectric(mut self, refraction_index: Float32) -> SurfaceId[1]:
        var index = UInt32(len(self.dielectrics))
        self.dielectrics.append(Dielectric(refraction_index))
        return SurfaceId(.DIELECTRIC, index)

    def add_coated_diffuse(
        mut self,
        albedo: Color,
        roughness: Float32,
        eta: Float32,
        texture_index: UInt32 = NO_TEXTURE,
        displacement_texture_index: UInt32 = NO_TEXTURE,
        displacement_scale: Float32 = 0.0,
        displacement_u_scale: Float32 = 1.0,
        displacement_v_scale: Float32 = 1.0,
        thickness: Float32 = 0.01,
        layer_albedo: Color = Color(0.0),
        g: Float32 = 0.0,
        max_depth: Int = 10,
        n_samples: Int = 1,
    ) -> SurfaceId[1]:
        var index = UInt32(len(self.coated_diffuses))
        self.coated_diffuses.append(
            CoatedDiffuse(
                albedo,
                roughness,
                eta,
                texture_index,
                displacement_texture_index,
                displacement_scale,
                displacement_u_scale,
                displacement_v_scale,
                thickness,
                layer_albedo,
                g,
                max_depth,
                n_samples,
            )
        )
        return SurfaceId(.COATED_DIFFUSE, index)

    def add_emissive(mut self, radiance: Color) -> SurfaceId[1]:
        var index = UInt32(len(self.emissives))
        self.emissives.append(Emissive(radiance))
        return SurfaceId(.EMISSIVE, index)
