"""Material, texture, environment, and surface storage types."""

from std.sys.info import size_of
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from bajo.rt.texture_math import (
    sample_image_texture,
    sample_environment_texture,
    texture_luminance,
    albedo_from_texture,
    roughness_from_texture,
    bump_slope,
    bump_normal,
)
from std.utils.numerics import isfinite

from bajo.core import Affine3f32, Vec3
from bajo.core.tensor_layout import record_tile, gather_record_vec3
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

    @always_inline
    def sample[
        length: SIMDLength = 1
    ](self, u: SIMD[.float32, length], v: SIMD[.float32, length]) -> Vec3[
        .float32, .WORLD, length
    ]:
        return sample_image_texture(
            record_tile[.float32, 3](
                self.pixels.unsafe_ptr(), self.width * self.height
            ),
            self.width,
            self.height,
            u,
            v,
        )

    @always_inline
    def sample_luminance[
        length: SIMDLength = 1
    ](self, u: SIMD[.float32, length], v: SIMD[.float32, length]) -> SIMD[
        .float32, length
    ]:
        return texture_luminance(self.sample(u, v))

    @always_inline
    def sample_environment[
        length: SIMDLength = 1
    ](self, u: SIMD[.float32, length], v: SIMD[.float32, length]) -> Vec3[
        .float32, .WORLD, length
    ]:
        """Nearest lookup in PBRT's top-to-bottom equal-area image domain."""
        return sample_environment_texture(
            record_tile[.float32, 3](
                self.pixels.unsafe_ptr(), self.width * self.height
            ),
            self.width,
            self.height,
            u,
            v,
        )


struct EnvironmentKind(
    DevicePassable, EnumLike, Equatable, TrivialRegisterPassable, Writable
):
    """Environment model with checked compile-time construction."""

    var _value: UInt32
    comptime BLACK = Self.__init__[0]()
    comptime PROCEDURAL = Self.__init__[1]()
    comptime UNIFORM = Self.__init__[2]()
    comptime IMAGE = Self.__init__[3]()
    comptime _enum_case_names = ParameterList.of[
        "BLACK".value,
        "PROCEDURAL".value,
        "UNIFORM".value,
        "IMAGE".value,
    ].values
    comptime _enum_case_types = TypeList.of[
        Trait=AnyType, NoneType, NoneType, NoneType, NoneType
    ].values

    def __init__[value: Int](out self):
        """Construct an environment kind from a compile-time model ID."""
        comptime assert 0 <= value <= 3
        self._value = UInt32(value)

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "EnvironmentKind"

    def _get_enum_discriminant(self) -> Int:
        return Int(self._value)

    def _unsafe_get_enum_payload[
        id: Int
    ](ref self) -> ref[self] TypeList[Trait=AnyType, Self._enum_case_types]()[
        id
    ]:
        while True:
            pass


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
    comptime MIX = Self(5)
    comptime _enum_case_names = ParameterList.of[
        "LAMBERTIAN".value,
        "METAL".value,
        "DIELECTRIC".value,
        "EMISSIVE".value,
        "COATED_DIFFUSE".value,
        "MIX".value,
    ].values
    comptime _enum_case_types = TypeList.of[
        Trait=AnyType,
        NoneType,
        NoneType,
        NoneType,
        NoneType,
        NoneType,
        NoneType,
    ].values
    comptime has_bsdf[kind: Self] = (
        kind.value == Self.LAMBERTIAN.value
        or kind.value == Self.METAL.value
        or kind.value == Self.DIELECTRIC.value
        or kind.value == Self.COATED_DIFFUSE.value
        or kind.value == Self.MIX.value
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


@always_inline
def sample_mix_branch(amount: Float32, mut random: Float32) -> Int:
    """Choose a child and remap its uniform variate for nested mixes."""
    if random < amount:
        random /= amount
        return 1
    random = (random - amount) / (1.0 - amount)
    return 0


struct Integrator(EnumLike, Equatable, TrivialRegisterPassable, Writable):
    """Render algorithm with checked compile-time construction."""

    var _value: UInt32
    comptime PATH = Self.__init__[0]()
    comptime NORMALS = Self.__init__[1]()
    comptime AO = Self.__init__[2]()
    comptime NEE = Self.__init__[3]()
    comptime MIS = Self.__init__[4]()
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
        integrator == Self.PATH
        or integrator == Self.NEE
        or integrator == Self.MIS
    )
    comptime uses_direct_lighting[integrator: Self] = (
        integrator == Self.NEE or integrator == Self.MIS
    )
    comptime uses_visibility[integrator: Self] = (
        integrator == Self.AO or Self.uses_direct_lighting[integrator]
    )

    def __init__[value: Int](out self):
        """Construct an integrator from a compile-time algorithm ID."""
        comptime assert 0 <= value <= 4
        self._value = UInt32(value)

    def _get_enum_discriminant(self) -> Int:
        return Int(self._value)

    def _unsafe_get_enum_payload[
        id: Int
    ](ref self) -> ref[self] TypeList[Trait=AnyType, Self._enum_case_types]()[
        id
    ]:
        while True:
            pass


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
            or (self.texture_index == NO_TEXTURE and self.albedo.x[0] > 1.0)
            or self.albedo.y[0] < 0.0
            or (self.texture_index == NO_TEXTURE and self.albedo.y[0] > 1.0)
            or self.albedo.z[0] < 0.0
            or (self.texture_index == NO_TEXTURE and self.albedo.z[0] > 1.0)
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
            or (self.texture_index == NO_TEXTURE and self.albedo.x[0] > 1.0)
            or self.albedo.y[0] < 0.0
            or (self.texture_index == NO_TEXTURE and self.albedo.y[0] > 1.0)
            or self.albedo.z[0] < 0.0
            or (self.texture_index == NO_TEXTURE and self.albedo.z[0] > 1.0)
        ):
            raise Error("metal albedo must be within [0, 1]")
        if not isfinite(self.fuzz):
            raise Error("metal fuzz must be finite")
        if self.fuzz < 0.0 or self.fuzz > 1.0:
            raise Error("metal fuzz must be within [0, 1]")


struct CoatedDiffuse(Copyable, Writable):
    """Shared layered material: dielectric coat over diffuse or conductor.

    Zero conductor eta and k select the diffuse substrate. The historical type and
    surface kind retain one CPU/GPU shading pipeline for both substrates.
    """

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
    var conductor_eta: Color
    var conductor_k: Color
    var conductor_roughness: Float32
    var roughness_texture_index: UInt32
    var roughness_scale: Float32
    var roughness_u_scale: Float32
    var roughness_v_scale: Float32
    var remap_roughness: Bool

    def __init__(
        out self,
        albedo: Color,
        roughness: Float32,
        eta: Float32,
        texture_index: UInt32,
        displacement_texture_index: UInt32,
        displacement_scale: Float32,
        displacement_u_scale: Float32,
        displacement_v_scale: Float32,
        thickness: Float32,
        layer_albedo: Color,
        g: Float32,
        max_depth: Int,
        n_samples: Int,
        conductor_eta: Color = Color(0.0),
        conductor_k: Color = Color(0.0),
        conductor_roughness: Float32 = 0.0,
    ):
        self.albedo = albedo
        self.roughness = roughness
        self.eta = eta
        self.texture_index = texture_index
        self.displacement_texture_index = displacement_texture_index
        self.displacement_scale = displacement_scale
        self.displacement_u_scale = displacement_u_scale
        self.displacement_v_scale = displacement_v_scale
        self.thickness = thickness
        self.layer_albedo = layer_albedo
        self.g = g
        self.max_depth = max_depth
        self.n_samples = n_samples
        self.conductor_eta = conductor_eta
        self.conductor_k = conductor_k
        self.conductor_roughness = conductor_roughness
        self.roughness_texture_index = NO_TEXTURE
        self.roughness_scale = 1.0
        self.roughness_u_scale = 1.0
        self.roughness_v_scale = 1.0
        self.remap_roughness = True

    def validate(self) raises:
        if (
            not self.conductor_eta.is_finite()[0]
            or not self.conductor_k.is_finite()[0]
        ):
            raise Error("conductor optical constants must be finite")
        for value in [
            self.conductor_eta.x[0],
            self.conductor_eta.y[0],
            self.conductor_eta.z[0],
            self.conductor_k.x[0],
            self.conductor_k.y[0],
            self.conductor_k.z[0],
        ]:
            if value < 0.0:
                raise Error("conductor optical constants must be non-negative")
        if (
            not isfinite(self.roughness_scale)
            or not isfinite(self.roughness_u_scale)
            or not isfinite(self.roughness_v_scale)
            or self.roughness_scale < 0.0
        ):
            raise Error("conductor roughness texture parameters are invalid")
        if (
            not isfinite(self.conductor_roughness)
            or self.conductor_roughness < 0.0
            or self.conductor_roughness > 1.0
        ):
            raise Error("conductor roughness must be within [0, 1]")
        if not self.albedo.is_finite()[0]:
            raise Error("coated diffuse albedo must be finite")
        if (
            self.albedo.x[0] < 0.0
            or (self.texture_index == NO_TEXTURE and self.albedo.x[0] > 1.0)
            or self.albedo.y[0] < 0.0
            or (self.texture_index == NO_TEXTURE and self.albedo.y[0] > 1.0)
            or self.albedo.z[0] < 0.0
            or (self.texture_index == NO_TEXTURE and self.albedo.z[0] > 1.0)
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


struct Dielectric(Copyable, Writable):
    var refraction_index: Float32
    var displacement_texture_index: UInt32
    var displacement_scale: Float32
    var displacement_u_scale: Float32
    var displacement_v_scale: Float32

    def __init__(
        out self,
        refraction_index: Float32,
        displacement_texture_index: UInt32 = NO_TEXTURE,
        displacement_scale: Float32 = 0.0,
        displacement_u_scale: Float32 = 1.0,
        displacement_v_scale: Float32 = 1.0,
    ):
        self.refraction_index = refraction_index
        self.displacement_texture_index = displacement_texture_index
        self.displacement_scale = displacement_scale
        self.displacement_u_scale = displacement_u_scale
        self.displacement_v_scale = displacement_v_scale

    def validate(self) raises:
        if (
            not isfinite(self.displacement_scale)
            or not isfinite(self.displacement_u_scale)
            or not isfinite(self.displacement_v_scale)
        ):
            raise Error("dielectric displacement parameters must be finite")
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
    # Mix nodes refer only to existing surfaces, forming an acyclic graph.
    var mix_surfaces: List[UInt32]
    var mix_amounts: List[Float32]
    var mix_texture_indices: List[UInt32]
    var mix_uv_scales: List[Float32]

    def __init__(out self):
        self.lambertians = List[Lambertian]()
        self.metals = List[Metal]()
        self.dielectrics = List[Dielectric]()
        self.emissives = List[Emissive]()
        self.coated_diffuses = List[CoatedDiffuse]()
        self.image_textures = List[ImageTexture]()
        self.mix_surfaces = List[UInt32]()
        self.mix_amounts = List[Float32]()
        self.mix_texture_indices = List[UInt32]()
        self.mix_uv_scales = List[Float32]()

    def contains(self, surface: SurfaceId[1]) -> Bool:
        match surface.kind():
            case .LAMBERTIAN:
                return surface.index() < UInt32(len(self.lambertians))
            case .METAL:
                return surface.index() < UInt32(len(self.metals))
            case .DIELECTRIC:
                return surface.index() < UInt32(len(self.dielectrics))
            case .EMISSIVE:
                return surface.index() < UInt32(len(self.emissives))
            case .COATED_DIFFUSE:
                return surface.index() < UInt32(len(self.coated_diffuses))
            case .MIX:
                return surface.index() < UInt32(len(self.mix_amounts))

    def add_mix(
        mut self,
        first: SurfaceId[1],
        second: SurfaceId[1],
        amount: Float32,
        texture_index: UInt32 = NO_TEXTURE,
        u_scale: Float32 = 1.0,
        v_scale: Float32 = 1.0,
    ) raises -> SurfaceId[1]:
        if not isfinite(amount) or amount < 0.0 or amount > 1.0:
            raise Error("mix amount must be within [0, 1]")
        if not self.contains(first) or not self.contains(second):
            raise Error("mix material reference is out of range")
        if first.kind() == .EMISSIVE or second.kind() == .EMISSIVE:
            raise Error("mix materials must reference BSDFs")
        var index = UInt32(len(self.mix_amounts))
        self.mix_surfaces.append(first.value[0])
        self.mix_surfaces.append(second.value[0])
        self.mix_amounts.append(amount)
        self.mix_texture_indices.append(texture_index)
        self.mix_uv_scales.append(u_scale)
        self.mix_uv_scales.append(v_scale)
        return SurfaceId(.MIX, index)

    @always_inline
    def _albedo_metadata(self, surface: SurfaceId[1]) -> Tuple[Color, UInt32]:
        match surface.kind():
            case .LAMBERTIAN:
                ref material = self.lambertians[Int(surface.index())]
                return material.albedo, material.texture_index
            case .METAL:
                ref material = self.metals[Int(surface.index())]
                return material.albedo, material.texture_index
            case .COATED_DIFFUSE:
                ref material = self.coated_diffuses[Int(surface.index())]
                return material.albedo, material.texture_index
            case _:
                return Color(0.0), NO_TEXTURE

    @always_inline
    def _displacement_metadata(
        self, surface: SurfaceId[1]
    ) -> Tuple[UInt32, Float32, Float32, Float32]:
        match surface.kind():
            case .COATED_DIFFUSE:
                ref material = self.coated_diffuses[Int(surface.index())]
                return (
                    material.displacement_texture_index,
                    material.displacement_scale,
                    material.displacement_u_scale,
                    material.displacement_v_scale,
                )
            case .DIELECTRIC:
                ref material = self.dielectrics[Int(surface.index())]
                return (
                    material.displacement_texture_index,
                    material.displacement_scale,
                    material.displacement_u_scale,
                    material.displacement_v_scale,
                )
            case _:
                return NO_TEXTURE, Float32(0.0), Float32(1.0), Float32(1.0)

    def sample_mix_amount(self, index: Int, u: Float32, v: Float32) -> Float32:
        var amount = self.mix_amounts[index]
        var texture_index = self.mix_texture_indices[index]
        if texture_index != NO_TEXTURE:
            amount *= self.image_textures[Int(texture_index)].sample_luminance(
                u * self.mix_uv_scales[2 * index],
                v * self.mix_uv_scales[2 * index + 1],
            )
        return amount.clamp(0.0, 1.0)

    @always_inline
    def resolve_mix(
        self,
        surface: SurfaceId[1],
        var random: Float32,
        u: Float32 = 0.0,
        v: Float32 = 0.0,
    ) -> SurfaceId[1]:
        var selected = surface.copy()
        random = random.clamp(0.0, 0.99999994)
        while selected.kind() == .MIX:
            var index = Int(selected.index())
            var amount = self.sample_mix_amount(index, u, v)
            var branch = sample_mix_branch(amount, random)
            selected = SurfaceId.from_raw(self.mix_surfaces[2 * index + branch])
        return selected^

    def sample_conductor_roughness[
        length: SIMDLength = 1
    ](
        self,
        surface: SurfaceId[length],
        u: SIMD[.float32, length],
        v: SIMD[.float32, length],
        active: SIMD[.bool, length] = SIMD[.bool, length](fill=True),
    ) -> SIMD[.float32, length]:
        var roughness = SIMD[.float32, length](0.0)
        var textures = SIMD[.uint32, length](NO_TEXTURE)
        var scale = SIMD[.float32, length](1.0)
        var us = scale
        var vs = scale
        var remap = SIMD[.bool, length](fill=False)
        comptime for lane in range(Int(length)):
            if active[lane]:
                ref material = self.coated_diffuses[
                    Int(surface.get(lane).index())
                ]
                roughness[lane] = material.conductor_roughness
                textures[lane] = material.roughness_texture_index
                scale[lane] = material.roughness_scale
                us[lane] = material.roughness_u_scale
                vs[lane] = material.roughness_v_scale
                remap[lane] = material.remap_roughness
        if not textures.ne(NO_TEXTURE).reduce_or():
            return roughness
        var mapped = roughness_from_texture(
            self.sample_image_textures(textures, u * us, v * vs), scale, remap
        )
        return textures.ne(NO_TEXTURE).select(mapped, roughness)

    @always_inline
    def emitted_radiance[
        length: SIMDLength = 1
    ](
        self,
        surface: SurfaceId[length],
        front_face: SIMD[.bool, length],
        active: SIMD[.bool, length] = SIMD[.bool, length](fill=True),
    ) -> Vec3[.float32, .WORLD, length]:
        var emitting = (
            active
            & front_face
            & (surface.value >> SURFACE_INDEX_BITS).eq(
                MaterialKind.EMISSIVE.value
            )
        )
        var zero = Vec3[.float32, .WORLD, length](0.0)
        if not emitting.reduce_or():
            return zero
        var ids = emitting.select(surface.value & SURFACE_INDEX_MASK, 0)
        debug_assert["safe", _use_compiler_assume=True](
            (~emitting | ids.lt(UInt32(len(self.emissives)))).reduce_and(),
            "emissive surface index is out of range",
        )
        comptime assert size_of[Emissive]() == 3 * size_of[Float32]()
        var records = record_tile[.float32, 3](
            self.emissives.unsafe_ptr().unsafe_bitcast[Float32](),
            len(self.emissives),
        )
        return Vec3.select(emitting, gather_record_vec3(records, ids), zero)

    def add_image_texture(mut self, var texture: ImageTexture) -> UInt32:
        var index = UInt32(len(self.image_textures))
        self.image_textures.append(texture^)
        return index

    @always_inline
    def sample_image_textures[
        length: SIMDLength
    ](
        self,
        indices: SIMD[.uint32, length],
        u: SIMD[.float32, length],
        v: SIMD[.float32, length],
    ) -> Vec3[.float32, .WORLD, length]:
        """Use native gathers for a shared map and lane-local reads otherwise.

        Both paths use ImageTexture.sample at the appropriate width. Work
        stays linear with distinct images; missing maps contribute identity.
        """
        var result = Vec3[.float32, .WORLD, length](1.0)
        var mapped = indices.ne(NO_TEXTURE)
        if not mapped.reduce_or():
            return result
        var first = 0
        while not mapped[first]:
            first += 1
        var safe_u = mapped.select(u, 0.0)
        var safe_v = mapped.select(v, 0.0)
        if (~mapped | indices.eq(indices[first])).reduce_and():
            var sampled = self.image_textures[Int(indices[first])].sample(
                safe_u, safe_v
            )
            return Vec3.select(mapped, sampled, result)
        comptime for lane in range(Int(length)):
            if mapped[lane]:
                var sampled = self.image_textures[Int(indices[lane])].sample(
                    safe_u[lane], safe_v[lane]
                )
                result.x[lane] = sampled.x
                result.y[lane] = sampled.y
                result.z[lane] = sampled.z
        return result

    def sample_albedo[
        length: SIMDLength = 1
    ](
        self,
        surface: SurfaceId[length],
        u: SIMD[.float32, length],
        v: SIMD[.float32, length],
        active: SIMD[.bool, length] = SIMD[.bool, length](fill=True),
    ) -> Vec3[.float32, .WORLD, length]:
        var albedo = Vec3[.float32, .WORLD, length](0.0)
        var textures = SIMD[.uint32, length](NO_TEXTURE)
        comptime for lane in range(Int(length)):
            if active[lane]:
                var value = SurfaceId.from_raw(surface.value[lane])
                var metadata = self._albedo_metadata(value)
                albedo.x[lane] = metadata[0].x
                albedo.y[lane] = metadata[0].y
                albedo.z[lane] = metadata[0].z
                textures[lane] = metadata[1]
        if not textures.ne(NO_TEXTURE).reduce_or():
            return albedo
        return albedo_from_texture(
            albedo,
            self.sample_image_textures(textures, u, v),
            textures.ne(NO_TEXTURE),
        )

    def shading_normal[
        length: SIMDLength = 1
    ](
        self,
        surface: SurfaceId[length],
        normal: Vec3[.float32, .WORLD, length],
        u: SIMD[.float32, length],
        v: SIMD[.float32, length],
        active: SIMD[.bool, length] = SIMD[.bool, length](fill=True),
    ) -> Vec3[.float32, .WORLD, length]:
        var textures = SIMD[.uint32, length](NO_TEXTURE)
        var scale = SIMD[.float32, length](0.0)
        var u_scale = SIMD[.float32, length](1.0)
        var v_scale = SIMD[.float32, length](1.0)
        var widths = SIMD[.float32, length](1.0)
        var heights = SIMD[.float32, length](1.0)
        comptime for lane in range(Int(length)):
            if active[lane]:
                var value = SurfaceId.from_raw(surface.value[lane])
                var metadata = self._displacement_metadata(value)
                if metadata[0] != NO_TEXTURE and metadata[1] != 0.0:
                    textures[lane] = metadata[0]
                    scale[lane] = metadata[1]
                    u_scale[lane] = metadata[2]
                    v_scale[lane] = metadata[3]
                    ref texture = self.image_textures[Int(textures[lane])]
                    widths[lane] = Float32(texture.width)
                    heights[lane] = Float32(texture.height)
        var mapped = textures.ne(NO_TEXTURE)
        if not mapped.reduce_or():
            return normal
        var tu = u * u_scale
        var tv = v * v_scale
        var du = Float32(0.5) / widths
        var dv = Float32(0.5) / heights
        var slope_u = bump_slope(
            self.sample_image_textures(textures, tu - du, tv),
            self.sample_image_textures(textures, tu + du, tv),
            widths,
            u_scale,
            scale,
        )
        var slope_v = bump_slope(
            self.sample_image_textures(textures, tu, tv - dv),
            self.sample_image_textures(textures, tu, tv + dv),
            heights,
            v_scale,
            scale,
        )
        var bumped = bump_normal(normal, slope_u, slope_v)
        return Vec3.select(mapped, bumped, normal)

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

    def add_coated_conductor(
        mut self,
        conductor_eta: Color,
        conductor_k: Color,
        conductor_roughness: Float32,
        roughness: Float32 = 0.0,
        eta: Float32 = 1.5,
        thickness: Float32 = 0.01,
        layer_albedo: Color = Color(0.0),
        g: Float32 = 0.0,
        max_depth: Int = 10,
        n_samples: Int = 1,
    ) -> SurfaceId[1]:
        var index = UInt32(len(self.coated_diffuses))
        self.coated_diffuses.append(
            CoatedDiffuse(
                Color(1.0),
                roughness,
                eta,
                NO_TEXTURE,
                NO_TEXTURE,
                0.0,
                1.0,
                1.0,
                thickness,
                layer_albedo,
                g,
                max_depth,
                n_samples,
                conductor_eta / eta,
                conductor_k / eta,
                conductor_roughness,
            )
        )
        return SurfaceId(.COATED_DIFFUSE, index)

    def add_conductor(
        mut self,
        eta: Color,
        k: Color,
        roughness: Float32,
    ) -> SurfaceId[1]:
        # A unit-IOR, zero-thickness interface is evaluated directly as a GGX
        # conductor, sharing the layered table and queue.
        return self.add_coated_conductor(eta, k, roughness, 0.0, 1.0, 0.0)

    def add_emissive(mut self, radiance: Color) -> SurfaceId[1]:
        var index = UInt32(len(self.emissives))
        self.emissives.append(Emissive(radiance))
        return SurfaceId(.EMISSIVE, index)
