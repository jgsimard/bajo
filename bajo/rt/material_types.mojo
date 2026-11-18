"""Material, texture, environment, and surface storage types."""

from layout import TileTensor, Coord, Idx, row_major
from bajo.core.image import ImageTexture
from bajo.core.geometry import normal_from_slopes as bump_normal

from std.sys.info import size_of
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from bajo.rt.texture_math import (
    mapped_normal,
    albedo_from_texture,
    roughness_from_texture,
    bump_slope,
)
from bajo.core.color import Color, luminance
from bajo.core.spectrum import PiecewiseLinearSpectrum
from bajo.rt.spectrum_sources import SpectrumSource, NO_SPECTRUM
from std.utils.numerics import isfinite

from bajo.core import Affine3f32, Vec3
from bajo.core.tensor_gather import gather_record_vec3
from bajo.bvh.constants import PrimitiveKind


comptime NO_TEXTURE = UInt32.MAX


__enum EnvironmentKind(DevicePassable, Equatable, ImplicitlyCopyable, Writable):
    """Environment model."""

    case BLACK
    case PROCEDURAL
    case UNIFORM
    case IMAGE

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "EnvironmentKind"


struct Environment(Copyable, Writable):
    """Scene-owned radiance for rays that leave the world."""

    var kind: EnvironmentKind
    var texture_index: UInt32
    var scale: Color
    var spectrum: UInt32
    var world_to_light: Affine3f32[.WORLD, .LOCAL]
    var light_to_world: Affine3f32[.LOCAL, .WORLD]

    def __init__(out self):
        self.kind = .PROCEDURAL
        self.texture_index = NO_TEXTURE
        self.scale = Color(1.0)
        self.spectrum = NO_SPECTRUM
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


__enum MaterialKind(Equatable, ImplicitlyCopyable, Writable):
    case LAMBERTIAN
    case METAL
    case DIELECTRIC
    case EMISSIVE
    case COATED_DIFFUSE
    case MIX

    comptime has_bsdf[kind: Self] = kind != Self.EMISSIVE

    def __init__(out self, value: UInt32):
        debug_assert(value <= 5)
        match Int(value):
            case 0:
                self = Self.LAMBERTIAN
            case 1:
                self = Self.METAL
            case 2:
                self = Self.DIELECTRIC
            case 3:
                self = Self.EMISSIVE
            case 4:
                self = Self.COATED_DIFFUSE
            case _:
                self = Self.MIX

    def value(self) -> UInt32:
        return UInt32(self._get_enum_discriminant())


comptime SURFACE_KIND_BITS = UInt32(4)
comptime SURFACE_INDEX_BITS = 32 - SURFACE_KIND_BITS
comptime SURFACE_INDEX_MASK = UInt32((1 << SURFACE_INDEX_BITS) - 1)


comptime PRIMITIVE_KIND_BITS = UInt32(4)
comptime PRIMITIVE_INDEX_BITS = 32 - PRIMITIVE_KIND_BITS
comptime PRIMITIVE_INDEX_MASK = UInt32((1 << PRIMITIVE_INDEX_BITS) - 1)


def sample_mix_branch(amount: Float32, mut random: Float32) -> Int:
    """Choose a child and remap its uniform variate for nested mixes."""
    if random < amount:
        random /= amount
        return 1
    random = (random - amount) / (1.0 - amount)
    return 0


__enum Integrator(Equatable, ImplicitlyCopyable, Writable):
    """Render algorithm."""

    case PATH
    case NORMALS
    case AO
    case NEE
    case MIS

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


@fieldwise_init
struct PrimitiveId(Copyable, Writable):
    var value: UInt32

    def __init__(out self, kind: PrimitiveKind, index: UInt32):
        debug_assert["safe", _use_compiler_assume=True](kind != .UNKNOWN)
        debug_assert["safe", _use_compiler_assume=True](
            index < (UInt32(1) << PRIMITIVE_INDEX_BITS)
        )
        self.value = (kind.value() << PRIMITIVE_INDEX_BITS) | index

    def kind(self) -> PrimitiveKind:
        return PrimitiveKind(self.value >> PRIMITIVE_INDEX_BITS)

    def index(self) -> UInt32:
        return self.value & PRIMITIVE_INDEX_MASK


@fieldwise_init
struct SurfaceId[length: SIMDLength = 1](Copyable, Writable):
    var value: SIMD[.uint32, Self.length]

    def __init__(out self, kind: MaterialKind, index: UInt32):
        debug_assert["safe", _use_compiler_assume=True](
            index < (UInt32(1) << SURFACE_INDEX_BITS)
        )
        self.value = Self.pack_raw(kind, index)

    @staticmethod
    def pack_raw(kind: MaterialKind, index: UInt32) -> UInt32:
        return (kind.value() << SURFACE_INDEX_BITS) | index

    def kind(self) -> MaterialKind:
        comptime assert Self.length == 1
        return Self.kind_from_raw(self.value[0])

    def index(self) -> UInt32:
        comptime assert Self.length == 1
        return Self.index_from_raw(self.value[0])

    @staticmethod
    def from_raw(value: UInt32) -> SurfaceId[1]:
        return SurfaceId[1](SIMD[.uint32, 1](value))

    @staticmethod
    def kind_from_raw(value: UInt32) -> MaterialKind:
        return MaterialKind(value >> SURFACE_INDEX_BITS)

    @staticmethod
    def index_from_raw(value: UInt32) -> UInt32:
        return value & SURFACE_INDEX_MASK

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
    var conductor_eta_spectrum: UInt32
    var conductor_k_spectrum: UInt32
    var conductor_reflectance: Bool

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
        self.conductor_eta_spectrum = NO_TEXTURE
        self.conductor_k_spectrum = NO_TEXTURE
        self.conductor_reflectance = False

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
    var roughness: Float32
    var eta_spectrum: UInt32
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
        roughness: Float32 = 0.0,
    ):
        self.refraction_index = refraction_index
        self.roughness = roughness
        self.eta_spectrum = NO_TEXTURE
        self.displacement_texture_index = displacement_texture_index
        self.displacement_scale = displacement_scale
        self.displacement_u_scale = displacement_u_scale
        self.displacement_v_scale = displacement_v_scale

    def validate(self) raises:
        if (
            not isfinite(self.roughness)
            or self.roughness < 0.0
            or self.roughness > 1.0
        ):
            raise Error("dielectric roughness must be finite and within [0, 1]")
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


@inline(.always)
def _gather_albedo_fields[
    length: SIMDLength
](
    albedo: ImmPointer[Float32, _],
    textures: ImmPointer[UInt32, _],
    stride: Int,
    indices: SIMD[.uint32, length],
) -> Tuple[Vec3[.float32, .WORLD, length], SIMD[.uint32, length]]:
    var offsets = indices.cast[.int64]() * Int64(stride)
    return (
        Vec3[.float32, .WORLD, length](
            albedo.unsafe_gather(offsets),
            albedo.unsafe_gather(offsets + 1),
            albedo.unsafe_gather(offsets + 2),
        ),
        textures.unsafe_gather(offsets),
    )


struct SurfaceStore(Defaultable):
    var lambertians: List[Lambertian]
    var metals: List[Metal]
    var dielectrics: List[Dielectric]
    var emissives: List[Emissive]
    var coated_diffuses: List[CoatedDiffuse]
    var image_textures: List[ImageTexture]
    var optical_spectra: List[PiecewiseLinearSpectrum]
    var spectrum_sources: List[SpectrumSource]
    var albedo_spectra: Dict[UInt32, UInt32]
    var dielectric_eta_sources: Dict[UInt32, UInt32]
    var layer_albedo_spectra: Dict[UInt32, UInt32]
    var normal_maps: Dict[UInt32, UInt32]
    # Mix nodes refer only to existing surfaces, forming an acyclic graph.
    var mix_surfaces: List[UInt32]
    var mix_amounts: List[Float32]
    var mix_texture_indices: List[UInt32]
    var mix_uv_scales: List[Float32]

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

    def _albedo_metadata[
        length: SIMDLength = 1
    ](
        self,
        surface: SurfaceId[length],
        active: SIMD[.bool, length] = SIMD[.bool, length](fill=True),
    ) -> Tuple[Vec3[.float32, .WORLD, length], SIMD[.uint32, length]]:
        comptime if length == 1:
            if not active[0]:
                return Vec3[.float32, .WORLD, length](0), SIMD[.uint32, length](
                    NO_TEXTURE
                )
            var scalar = surface.get(0)
            match scalar.kind():
                case .LAMBERTIAN:
                    ref material = self.lambertians[Int(scalar.index())]
                    return material.albedo.broadcast[length](), SIMD[
                        .uint32, length
                    ](material.texture_index)
                case .METAL:
                    ref material = self.metals[Int(scalar.index())]
                    return material.albedo.broadcast[length](), SIMD[
                        .uint32, length
                    ](material.texture_index)
                case .COATED_DIFFUSE:
                    ref material = self.coated_diffuses[Int(scalar.index())]
                    return material.albedo.broadcast[length](), SIMD[
                        .uint32, length
                    ](material.texture_index)
                case _:
                    return Vec3[.float32, .WORLD, length](0), SIMD[
                        .uint32, length
                    ](NO_TEXTURE)
        var color = Vec3[.float32, .WORLD, length](0)
        var textures = SIMD[.uint32, length](NO_TEXTURE)
        comptime for kind in ParameterList.of[
            MaterialKind.LAMBERTIAN,
            MaterialKind.METAL,
            MaterialKind.COATED_DIFFUSE,
        ]():
            var selected = active & (surface.value >> SURFACE_INDEX_BITS).eq(
                kind.value()
            )
            if not selected.reduce_or():
                continue
            var ids = selected.select(surface.value & SURFACE_INDEX_MASK, 0)
            var fields: Tuple[
                Vec3[.float32, .WORLD, length], SIMD[.uint32, length]
            ]
            comptime if kind == .LAMBERTIAN:
                fields = _gather_albedo_fields(
                    Pointer(to=self.lambertians[0].albedo.x),
                    Pointer(to=self.lambertians[0].texture_index),
                    size_of[Lambertian]() // size_of[Float32](),
                    ids,
                )
            elif kind == .METAL:
                fields = _gather_albedo_fields(
                    Pointer(to=self.metals[0].albedo.x),
                    Pointer(to=self.metals[0].texture_index),
                    size_of[Metal]() // size_of[Float32](),
                    ids,
                )
            else:
                fields = _gather_albedo_fields(
                    Pointer(to=self.coated_diffuses[0].albedo.x),
                    Pointer(to=self.coated_diffuses[0].texture_index),
                    size_of[CoatedDiffuse]() // size_of[Float32](),
                    ids,
                )
            color = Vec3.select(selected, fields[0], color)
            textures = selected.select(fields[1], textures)
        return color, textures

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

    def resolve_mix_packet[
        length: SIMDLength
    ](
        self,
        surface: SurfaceId[length],
        var random: SIMD[.float32, length],
        u: SIMD[.float32, length],
        v: SIMD[.float32, length],
        active: SIMD[.bool, length] = SIMD[.bool, length](fill=True),
    ) -> SurfaceId[length]:
        comptime if length == 1:
            if active[0]:
                var resolved = self.resolve_mix(
                    surface.get(0), random[0], u[0], v[0]
                )
                return SurfaceId[length](
                    SIMD[.uint32, length](resolved.value[0])
                )
            return surface.copy()
        var selected = surface.copy()
        random = random.clamp(0, 0.99999994)
        var pending = active & (selected.value >> SURFACE_INDEX_BITS).eq(
            MaterialKind.MIX.value()
        )
        while pending.reduce_or():
            var ids = pending.select(
                selected.value & SURFACE_INDEX_MASK, 0
            ).cast[.int64]()
            var amount = self.mix_amounts.unsafe_ptr().unsafe_gather(ids)
            var textures = pending.select(
                self.mix_texture_indices.unsafe_ptr().unsafe_gather(ids),
                NO_TEXTURE,
            )
            var us = self.mix_uv_scales.unsafe_ptr().unsafe_gather(2 * ids)
            var vs = self.mix_uv_scales.unsafe_ptr().unsafe_gather(2 * ids + 1)
            var texel = self.sample_image_textures(textures, u * us, v * vs)
            amount = (
                textures.ne(NO_TEXTURE)
                .select(amount * luminance(texel), amount)
                .clamp(0, 1)
            )
            var second = random.lt(amount)
            var remapped = second.select(
                random / second.select(amount, 1),
                (random - amount) / second.select(Float32(1), 1 - amount),
            )
            random = pending.select(remapped, random)
            var child = self.mix_surfaces.unsafe_ptr().unsafe_gather(
                2 * ids + second.cast[.int64]()
            )
            selected.value = pending.select(child, selected.value)
            pending = active & (selected.value >> SURFACE_INDEX_BITS).eq(
                MaterialKind.MIX.value()
            )
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
                MaterialKind.EMISSIVE.value()
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
        var records = TileTensor(
            ptr=self.emissives.unsafe_ptr().unsafe_bitcast[Float32](),
            layout=row_major(Coord(len(self.emissives), Idx[3])),
        )
        return Vec3.select(emitting, gather_record_vec3(records, ids), zero)

    def add_image_texture(mut self, var texture: ImageTexture) -> UInt32:
        var index = UInt32(len(self.image_textures))
        self.image_textures.append(texture^)
        return index

    @inline(.always)
    def sample_image_textures[
        length: SIMDLength, bilinear: Bool = False
    ](
        self,
        indices: SIMD[.uint32, length],
        u: SIMD[.float32, length],
        v: SIMD[.float32, length],
    ) -> Vec3[.float32, .WORLD, length]:
        """Share SIMD work within bilinear groups and gather shared maps.

        Bilinear samples share work within image groups. Missing maps
        contribute identity without reading their coordinates or storage.
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
            ref texture = self.image_textures[Int(indices[first])]
            var sampled = texture.sample_bilinear(
                safe_u, safe_v
            ) if bilinear else texture.sample(safe_u, safe_v)
            return Vec3.select(mapped, sampled, result)
        comptime if not bilinear:
            # Nearest sampling has too little arithmetic to amortize gathering
            # different image descriptors. Measurements favor direct reads.
            comptime for lane in range(Int(length)):
                if mapped[lane]:
                    var sampled = self.image_textures[
                        Int(indices[lane])
                    ].sample(safe_u[lane], safe_v[lane])
                    result.x[lane] = sampled.x
                    result.y[lane] = sampled.y
                    result.z[lane] = sampled.z
            return result
        return self._sample_mixed_image_textures[length, bilinear](
            indices, safe_u, safe_v
        )

    def _sample_mixed_image_textures[
        length: SIMDLength, bilinear: Bool
    ](
        self,
        indices: SIMD[.uint32, length],
        safe_u: SIMD[.float32, length],
        safe_v: SIMD[.float32, length],
    ) -> Vec3[.float32, .WORLD, length]:
        var mapped = indices.ne(NO_TEXTURE)
        var result = Vec3[.float32, .WORLD, length](1)
        var first = 0
        var pending = mapped
        while pending.reduce_or():
            while not pending[first]:
                first += 1
            var group = pending & indices.eq(indices[first])
            var peers = group
            peers[first] = False
            if peers.reduce_or():
                ref texture = self.image_textures[Int(indices[first])]
                var tu = group.select(safe_u, 0)
                var tv = group.select(safe_v, 0)
                var sampled = texture.sample_bilinear(
                    tu, tv
                ) if bilinear else texture.sample(tu, tv)
                result = Vec3.select(group, sampled, result)
            else:
                ref texture = self.image_textures[Int(indices[first])]
                var sampled = texture.sample_bilinear(
                    safe_u[first], safe_v[first]
                ) if bilinear else texture.sample(safe_u[first], safe_v[first])
                result.x[first] = sampled.x
                result.y[first] = sampled.y
                result.z[first] = sampled.z
            pending &= ~group
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
        var albedo, textures = self._albedo_metadata(surface, active)
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
        dpdu: Vec3[.float32, .WORLD, length] = Vec3[.float32, .WORLD, length](
            0.0
        ),
        front_face: SIMD[.bool, length] = SIMD[.bool, length](fill=True),
    ) -> Vec3[.float32, .WORLD, length]:
        var base_normal = normal
        var normal_maps = SIMD[.uint32, length](NO_TEXTURE)
        for lane in range(Int(length)):
            if active[lane]:
                if case .Some(index) = self.normal_maps.get(
                    surface.value[lane]
                ):
                    normal_maps[lane] = index
        var normal_mapped = normal_maps.ne(NO_TEXTURE)
        if normal_mapped.reduce_or():
            var outward = Vec3.select(front_face, normal, -normal)
            var color = self.sample_image_textures[length, True](
                normal_maps, u, v
            )
            var mapped = mapped_normal(outward, dpdu, color)
            mapped = Vec3.select(front_face, mapped, -mapped)
            base_normal = Vec3.select(normal_mapped, mapped, base_normal)
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
            return base_normal
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
        var bumped = bump_normal(base_normal, slope_u, slope_v)
        return Vec3.select(mapped, bumped, base_normal)

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

    def add_coated_conductor_reflectance(
        mut self,
        reflectance: Color,
        conductor_roughness: Float32,
        roughness: Float32 = 0.0,
        eta: Float32 = 1.5,
        thickness: Float32 = 0.01,
        layer_albedo: Color = Color(0.0),
        g: Float32 = 0.0,
        max_depth: Int = 10,
        n_samples: Int = 1,
        texture_index: UInt32 = NO_TEXTURE,
    ) -> SurfaceId[1]:
        var surface = self.add_coated_conductor(
            Color(1.0),
            Color(0.0),
            conductor_roughness,
            roughness,
            eta,
            thickness,
            layer_albedo,
            g,
            max_depth,
            n_samples,
        )
        ref material = self.coated_diffuses[Int(surface.index())]
        material.albedo = reflectance
        material.texture_index = texture_index
        material.conductor_reflectance = True
        return surface^

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
