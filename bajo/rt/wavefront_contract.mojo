"""Shared CPU/GPU ABI and orchestration contract for wavefront rendering."""

from std.memory import bitcast
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder

from bajo.core import Point3f32, Rayf32, Vec3f32
from bajo.core.color import Color
from bajo.rt.path_values import PathFloatField as WavePathFloatAbi, PathValues
from bajo.rt.path_store import store_path_values
from bajo.core.tensor_layout import field_tile
from bajo.rt.rays import RT_RAY_T_MIN
from bajo.bvh.constants import f32_max
from bajo.rt.material_types import SurfaceId
from bajo.rt.shading_types import SurfaceHit
from bajo.rt.wavefront_queue import (
    FRONT_FACE_BIT,
    PATH_INDEX_MASK,
    WavePath,
    WaveShade,
)


struct WAVE_COUNTER:
    comptime ACTIVE = 0
    comptime NEXT = 1
    comptime SHADE = 2
    comptime SHADOW = 3
    comptime STATUS = 4
    comptime COUNT = 5


struct WAVE_STATUS(Equatable, TrivialRegisterPassable, Writable):
    comptime OK = UInt32(0)
    comptime PATH_OVERFLOW = UInt32(1)
    comptime SHADE_OVERFLOW = UInt32(2)
    comptime SHADOW_OVERFLOW = UInt32(3)


@fieldwise_init
struct WAVE_RNG_DOMAIN(EnumLike, Equatable, TrivialRegisterPassable, Writable):
    var v: UInt32
    comptime SHIFT = UInt32(30)
    comptime BSDF = Self(0)
    comptime ROULETTE = Self(1)
    comptime LIGHT = Self(2)
    comptime MATERIAL = Self(3)
    comptime _enum_case_names = ParameterList.of[
        "BSDF".value, "ROULETTE".value, "LIGHT".value, "MATERIAL".value
    ].values
    comptime _enum_case_types = TypeList.of[
        Trait=AnyType, NoneType, NoneType, NoneType, NoneType
    ].values

    def _get_enum_discriminant(self) -> Int:
        return Int(self.v)

    def _unsafe_get_enum_payload[
        id: Int
    ](ref self) -> ref[self] TypeList[Trait=AnyType, Self._enum_case_types]()[
        id
    ]:
        while True:
            pass


# The path queue owns the full UInt32 ID word, unlike material path references.
# Pack the one-bit previous-event property into its otherwise unused high bit.
comptime WAVE_PATH_DELTA_BIT = UInt32(0x80000000)
comptime WAVE_PATH_ID_MASK = UInt32(0x7FFFFFFF)


struct WaveShadeFloatAbi:
    """Field-major float planes for a GPU material queue."""

    comptime NX = 0
    comptime NY = 1
    comptime NZ = 2
    comptime T = 3
    comptime PLANES = 4


struct WaveSampleFloatAbi:
    """Per-primary sample output, reduced to pixels in a final stage."""

    comptime R = 0
    comptime G = 1
    comptime B = 2
    comptime PLANES = 3


struct WaveShadowFloatAbi:
    """Field-major ray and deferred contribution planes for shadow work."""

    comptime OX = 0
    comptime OY = 1
    comptime OZ = 2
    comptime T_MIN = 3
    comptime DX = 4
    comptime DY = 5
    comptime DZ = 6
    comptime T_MAX = 7
    comptime R = 8
    comptime G = 9
    comptime B = 10
    comptime PLANES = 11


@always_inline
def wavefront_plane_index(plane: Int, capacity: Int, idx: Int) -> Int:
    return plane * capacity + idx


@always_inline
def wavefront_rng_subsequence[
    length: SIMDLength = 1
](path_id: SIMD[.uint32, length], rng_stage: UInt32) -> SIMD[.uint64, length]:
    """Schedule-independent Philox subsequence owned by one path and stage."""
    return (path_id.cast[.uint64]() << UInt64(32)) | UInt64(rng_stage)


@always_inline
def wavefront_rng_stage(bounce: UInt32) -> UInt32:
    """Map zero-based bounce to the existing shade RNG stage."""
    return (WAVE_RNG_DOMAIN.BSDF.v << WAVE_RNG_DOMAIN.SHIFT) | (
        bounce + UInt32(1)
    )


@always_inline
def wavefront_rng_roulette_stage(bounce: UInt32) -> UInt32:
    """Independent Philox domain for zero-based-bounce roulette decisions."""
    return (WAVE_RNG_DOMAIN.ROULETTE.v << WAVE_RNG_DOMAIN.SHIFT) | (
        bounce + UInt32(1)
    )


@always_inline
def wavefront_rng_light_stage(bounce: UInt32) -> UInt32:
    """Independent Philox domain for zero-based-bounce light sampling."""
    return (WAVE_RNG_DOMAIN.LIGHT.v << WAVE_RNG_DOMAIN.SHIFT) | (
        bounce + UInt32(1)
    )


@always_inline
def wavefront_rng_material_stage(bounce: UInt32) -> UInt32:
    """Independent material selection shared by CPU and GPU schedulers."""
    return (WAVE_RNG_DOMAIN.MATERIAL.v << WAVE_RNG_DOMAIN.SHIFT) | (
        bounce + UInt32(1)
    )


struct DeviceWavePath(
    DevicePassable, PathValues, TrivialRegisterPassable, Writable
):
    """Register representation reconstructed from coalesced device planes."""

    comptime WIDTH = SIMDLength(1)

    var path_id: UInt32
    var ox: Float32
    var oy: Float32
    var oz: Float32
    var t_min: Float32
    var dx: Float32
    var dy: Float32
    var dz: Float32
    var t_max: Float32
    var tx: Float32
    var ty: Float32
    var tz: Float32
    var bsdf_pdf: Float32
    var delta: Bool
    var medium_id: UInt32

    def __init__(
        out self,
        path_id: UInt32,
        ox: Float32,
        oy: Float32,
        oz: Float32,
        t_min: Float32,
        dx: Float32,
        dy: Float32,
        dz: Float32,
        t_max: Float32,
        tx: Float32,
        ty: Float32,
        tz: Float32,
        bsdf_pdf: Float32,
        delta: Bool,
        medium_id: UInt32 = 0,
    ):
        self.path_id = path_id
        self.ox = ox
        self.oy = oy
        self.oz = oz
        self.t_min = t_min
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.t_max = t_max
        self.tx = tx
        self.ty = ty
        self.tz = tz
        self.bsdf_pdf = bsdf_pdf
        self.delta = delta
        self.medium_id = medium_id

    @always_inline
    def ids(self) -> SIMD[.uint32, Self.WIDTH]:
        return self.path_id

    @always_inline
    def media(self) -> SIMD[.uint32, Self.WIDTH]:
        return self.medium_id

    @always_inline
    def delta_events(self) -> SIMD[.bool, Self.WIDTH]:
        return self.delta

    @always_inline
    def float_field[field: Int](self) -> SIMD[.float32, Self.WIDTH]:
        comptime match field:
            case WavePathFloatAbi.OX:
                return self.ox
            case WavePathFloatAbi.OY:
                return self.oy
            case WavePathFloatAbi.OZ:
                return self.oz
            case WavePathFloatAbi.T_MIN:
                return self.t_min
            case WavePathFloatAbi.DX:
                return self.dx
            case WavePathFloatAbi.DY:
                return self.dy
            case WavePathFloatAbi.DZ:
                return self.dz
            case WavePathFloatAbi.T_MAX:
                return self.t_max
            case WavePathFloatAbi.TX:
                return self.tx
            case WavePathFloatAbi.TY:
                return self.ty
            case WavePathFloatAbi.TZ:
                return self.tz
            case WavePathFloatAbi.BSDF_PDF:
                return self.bsdf_pdf
            case _:
                comptime assert False, "not a floating-point path value"

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "DeviceWavePath"

    @staticmethod
    def from_path(path: WavePath) -> Self:
        return Self(
            path.path_id,
            path.ray.o.x,
            path.ray.o.y,
            path.ray.o.z,
            path.ray.t_min,
            path.ray.d.x,
            path.ray.d.y,
            path.ray.d.z,
            path.ray.t_max,
            path.throughput.x,
            path.throughput.y,
            path.throughput.z,
            0.0,
            True,
            path.medium_id,
        )

    def to_path(self) -> WavePath:
        return WavePath(
            self.path_id,
            Rayf32[.WORLD](
                Point3f32[.WORLD](self.ox, self.oy, self.oz),
                Vec3f32[.WORLD](self.dx, self.dy, self.dz),
                self.t_min,
                self.t_max,
            ),
            Color(self.tx, self.ty, self.tz),
            self.medium_id,
        )


@fieldwise_init
struct DeviceWaveShade(DevicePassable, TrivialRegisterPassable, Writable):
    """Register representation reconstructed from a material queue."""

    var path_idx: UInt32
    var nx: Float32
    var ny: Float32
    var nz: Float32
    var surface_value: UInt32
    var t: Float32
    var front_face: Bool

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "DeviceWaveShade"

    @staticmethod
    def from_work(work: WaveShade) -> Self:
        return Self(
            work.path_idx,
            work.hit.normal.x,
            work.hit.normal.y,
            work.hit.normal.z,
            work.hit.surface.value,
            work.hit.t,
            work.hit.front_face,
        )

    def to_work(self) -> WaveShade:
        return WaveShade(
            self.path_idx,
            SurfaceHit(
                Vec3f32[.WORLD](self.nx, self.ny, self.nz),
                SurfaceId.from_raw(self.surface_value),
                self.t,
                self.front_face,
                True,
            ),
        )


@fieldwise_init
struct DeviceWaveShadow(DevicePassable, TrivialRegisterPassable, Writable):
    """Compact ray plus deferred radiance for one visibility query."""

    var path_id: UInt32
    var ox: Float32
    var oy: Float32
    var oz: Float32
    var t_min: Float32
    var dx: Float32
    var dy: Float32
    var dz: Float32
    var t_max: Float32
    var r: Float32
    var g: Float32
    var b: Float32

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "DeviceWaveShadow"


@always_inline
def load_device_wave_path[
    LOAD_BOUNDS: Bool = True,
    LOAD_THROUGHPUT: Bool = True,
    LOAD_PDF: Bool = True,
    LOAD_MEDIUM: Bool = True,
    LOAD_DELTA: Bool = True,
](
    path_ids: ImmPointer[UInt32, _],
    fields: ImmPointer[Float32, _],
    capacity: Int,
    idx: Int,
) -> DeviceWavePath:
    """Decode host/device planes, reconstructing fields absent from RT stages.
    """
    var tile = field_tile[.float32, WavePathFloatAbi.PLANES](fields, capacity)
    var packed_id = path_ids[unsafe_offset=idx]
    var t_min = RT_RAY_T_MIN
    var t_max = f32_max
    comptime if LOAD_BOUNDS:
        t_min = tile[(WavePathFloatAbi.T_MIN, idx)]
        t_max = tile[(WavePathFloatAbi.T_MAX, idx)]
    var tx = Float32(1.0)
    var ty = Float32(1.0)
    var tz = Float32(1.0)
    comptime if LOAD_THROUGHPUT:
        tx = tile[(WavePathFloatAbi.TX, idx)]
        ty = tile[(WavePathFloatAbi.TY, idx)]
        tz = tile[(WavePathFloatAbi.TZ, idx)]
    var pdf = Float32(0.0)
    comptime if LOAD_PDF:
        pdf = tile[(WavePathFloatAbi.BSDF_PDF, idx)]
    var medium = UInt32(0)
    comptime if LOAD_MEDIUM:
        medium = bitcast[.uint32](tile[(WavePathFloatAbi.MEDIUM, idx)])
    var delta = True
    comptime if LOAD_DELTA:
        delta = (packed_id & WAVE_PATH_DELTA_BIT) != 0
    return DeviceWavePath(
        packed_id & WAVE_PATH_ID_MASK,
        tile[(WavePathFloatAbi.OX, idx)],
        tile[(WavePathFloatAbi.OY, idx)],
        tile[(WavePathFloatAbi.OZ, idx)],
        t_min,
        tile[(WavePathFloatAbi.DX, idx)],
        tile[(WavePathFloatAbi.DY, idx)],
        tile[(WavePathFloatAbi.DZ, idx)],
        t_max,
        tx,
        ty,
        tz,
        pdf,
        delta,
        medium,
    )


@always_inline
def store_device_wave_path(
    path: DeviceWavePath,
    path_ids: MutPointer[UInt32, _],
    fields: MutPointer[Float32, _],
    capacity: Int,
    idx: Int,
):
    var tile = field_tile[.float32, WavePathFloatAbi.PLANES](fields, capacity)
    path_ids[unsafe_offset=idx] = (path.path_id & WAVE_PATH_ID_MASK) | (
        WAVE_PATH_DELTA_BIT if path.delta else UInt32(0)
    )
    store_path_values(tile, path, idx)
    tile[(WavePathFloatAbi.MEDIUM, idx)] = bitcast[.float32](path.medium_id)


@always_inline
def load_device_wave_shade(
    path_refs: ImmPointer[UInt32, _],
    surface_values: ImmPointer[UInt32, _],
    fields: ImmPointer[Float32, _],
    capacity: Int,
    idx: Int,
) -> DeviceWaveShade:
    var path_ref = path_refs[unsafe_offset=idx]
    var tile = field_tile[.float32, WaveShadeFloatAbi.PLANES](fields, capacity)
    return DeviceWaveShade(
        path_ref & PATH_INDEX_MASK,
        tile[WaveShadeFloatAbi.NX, idx],
        tile[WaveShadeFloatAbi.NY, idx],
        tile[WaveShadeFloatAbi.NZ, idx],
        surface_values[unsafe_offset=idx],
        tile[WaveShadeFloatAbi.T, idx],
        (path_ref & FRONT_FACE_BIT) != 0,
    )


@always_inline
def store_device_wave_shade(
    work: DeviceWaveShade,
    path_refs: MutPointer[UInt32, _],
    surface_values: MutPointer[UInt32, _],
    fields: MutPointer[Float32, _],
    capacity: Int,
    idx: Int,
):
    var path_ref = work.path_idx
    if work.front_face:
        path_ref |= FRONT_FACE_BIT
    path_refs[unsafe_offset=idx] = path_ref
    surface_values[unsafe_offset=idx] = work.surface_value
    var tile = field_tile[.float32, WaveShadeFloatAbi.PLANES](fields, capacity)
    tile[WaveShadeFloatAbi.NX, idx] = work.nx
    tile[WaveShadeFloatAbi.NY, idx] = work.ny
    tile[WaveShadeFloatAbi.NZ, idx] = work.nz
    tile[WaveShadeFloatAbi.T, idx] = work.t


struct PackedWavePathQueue(Sized):
    """Host mirror of the field-major device path queue."""

    var capacity: Int
    var count: Int
    var path_ids: List[UInt32]
    var fields: List[Float32]

    def __init__(out self, capacity: Int):
        debug_assert["safe", _use_compiler_assume=True](capacity >= 0)
        self.capacity = capacity
        self.count = 0
        self.path_ids = List[UInt32](length=capacity, fill=UInt32(0))
        self.fields = List[Float32](
            length=capacity * WavePathFloatAbi.PLANES, fill=0.0
        )

    def __len__(self) -> Int:
        return self.count

    def clear(mut self):
        self.count = 0

    def append(mut self, path: WavePath):
        debug_assert["safe", _use_compiler_assume=True](
            self.count < self.capacity, "packed path queue overflow"
        )
        self.store(self.count, DeviceWavePath.from_path(path))
        self.count += 1

    def store(mut self, idx: Int, path: DeviceWavePath):
        debug_assert["safe", _use_compiler_assume=True](
            idx >= 0 and idx < self.capacity
        )
        store_device_wave_path(
            path,
            self.path_ids.unsafe_ptr(),
            self.fields.unsafe_ptr(),
            self.capacity,
            idx,
        )

    def get(self, idx: Int) -> WavePath:
        debug_assert["safe", _use_compiler_assume=True](
            idx >= 0 and idx < self.count
        )
        return load_device_wave_path(
            self.path_ids.unsafe_ptr(),
            self.fields.unsafe_ptr(),
            self.capacity,
            idx,
        ).to_path()


struct PackedWaveShadeQueue(Sized):
    """Host mirror of one field-major device material queue."""

    var capacity: Int
    var count: Int
    var path_refs: List[UInt32]
    var surface_values: List[UInt32]
    var fields: List[Float32]

    def __init__(out self, capacity: Int):
        debug_assert["safe", _use_compiler_assume=True](capacity >= 0)
        self.capacity = capacity
        self.count = 0
        self.path_refs = List[UInt32](length=capacity, fill=UInt32(0))
        self.surface_values = List[UInt32](length=capacity, fill=UInt32(0))
        self.fields = List[Float32](
            length=capacity * WaveShadeFloatAbi.PLANES, fill=0.0
        )

    def __len__(self) -> Int:
        return self.count

    def clear(mut self):
        self.count = 0

    def append(mut self, work: WaveShade):
        debug_assert["safe", _use_compiler_assume=True](
            self.count < self.capacity, "packed shade queue overflow"
        )
        store_device_wave_shade(
            DeviceWaveShade.from_work(work),
            self.path_refs.unsafe_ptr(),
            self.surface_values.unsafe_ptr(),
            self.fields.unsafe_ptr(),
            self.capacity,
            self.count,
        )
        self.count += 1

    def get(self, idx: Int) -> WaveShade:
        debug_assert["safe", _use_compiler_assume=True](
            idx >= 0 and idx < self.count
        )
        return load_device_wave_shade(
            self.path_refs.unsafe_ptr(),
            self.surface_values.unsafe_ptr(),
            self.fields.unsafe_ptr(),
            self.capacity,
            idx,
        ).to_work()


def pack_wave_paths(
    paths: ImmSpan[WavePath, _], capacity: Int = -1
) -> PackedWavePathQueue:
    """Convert the CPU AoS queue once at a host/device boundary."""
    var packed_capacity = max(len(paths), capacity)
    var packed = PackedWavePathQueue(packed_capacity)
    for path in paths:
        packed.append(path)
    return packed^


def unpack_wave_paths(packed: PackedWavePathQueue) -> List[WavePath]:
    return [packed.get(i) for i in range(len(packed))]


def pack_wave_shades(
    works: ImmSpan[WaveShade, _], capacity: Int = -1
) -> PackedWaveShadeQueue:
    var packed_capacity = max(len(works), capacity)
    var packed = PackedWaveShadeQueue(packed_capacity)
    for work in works:
        packed.append(work)
    return packed^


def unpack_wave_shades(packed: PackedWaveShadeQueue) -> List[WaveShade]:
    return [packed.get(i) for i in range(len(packed))]
