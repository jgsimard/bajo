"""Shared CPU/GPU ABI and orchestration contract for wavefront rendering."""

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder

from bajo.core import Frame, Point3f32, Rayf32, Vec3f32
from bajo.rt.types import Color, SurfaceHit, SurfaceId, MaterialKind
from bajo.rt.wavefront_queue import (
    FRONT_FACE_BIT,
    PacketPathQueue,
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
struct WAVE_RNG_DOMAIN(Equatable, TrivialRegisterPassable, Writable):
    var v: UInt32
    comptime SHIFT = UInt32(30)
    comptime BSDF = Self(0)
    comptime ROULETTE = Self(1)
    comptime LIGHT = Self(2)


struct WavePathFloatAbi:
    """Field-major float planes for a GPU path queue."""

    comptime OX = 0
    comptime OY = 1
    comptime OZ = 2
    comptime T_MIN = 3
    comptime DX = 4
    comptime DY = 5
    comptime DZ = 6
    comptime T_MAX = 7
    comptime TX = 8
    comptime TY = 9
    comptime TZ = 10
    comptime BSDF_PDF = 11
    comptime PLANES = 12


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
def wavefront_rng_subsequence(path_id: UInt32, rng_stage: UInt32) -> UInt64:
    """Schedule-independent Philox subsequence owned by one path and stage."""
    return (UInt64(path_id) << UInt64(32)) | UInt64(rng_stage)


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


@fieldwise_init
struct DeviceWavePath(DevicePassable, TrivialRegisterPassable, Writable):
    """Register representation reconstructed from coalesced device planes."""

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
                SurfaceId(
                    MaterialKind(self.surface_value >> UInt32(28)),
                    self.surface_value & UInt32(0x0FFFFFFF),
                ),
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
def load_device_wave_path(
    path_ids: Pointer[UInt32, ImmutAnyOrigin],
    fields: Pointer[Float32, ImmutAnyOrigin],
    capacity: Int,
    idx: Int,
) -> DeviceWavePath:
    var packed_path_id = path_ids[unsafe_offset=idx]
    return DeviceWavePath(
        packed_path_id & WAVE_PATH_ID_MASK,
        fields[
            unsafe_offset=wavefront_plane_index(
                WavePathFloatAbi.OX, capacity, idx
            )
        ],
        fields[
            unsafe_offset=wavefront_plane_index(
                WavePathFloatAbi.OY, capacity, idx
            )
        ],
        fields[
            unsafe_offset=wavefront_plane_index(
                WavePathFloatAbi.OZ, capacity, idx
            )
        ],
        fields[
            unsafe_offset=wavefront_plane_index(
                WavePathFloatAbi.T_MIN, capacity, idx
            )
        ],
        fields[
            unsafe_offset=wavefront_plane_index(
                WavePathFloatAbi.DX, capacity, idx
            )
        ],
        fields[
            unsafe_offset=wavefront_plane_index(
                WavePathFloatAbi.DY, capacity, idx
            )
        ],
        fields[
            unsafe_offset=wavefront_plane_index(
                WavePathFloatAbi.DZ, capacity, idx
            )
        ],
        fields[
            unsafe_offset=wavefront_plane_index(
                WavePathFloatAbi.T_MAX, capacity, idx
            )
        ],
        fields[
            unsafe_offset=wavefront_plane_index(
                WavePathFloatAbi.TX, capacity, idx
            )
        ],
        fields[
            unsafe_offset=wavefront_plane_index(
                WavePathFloatAbi.TY, capacity, idx
            )
        ],
        fields[
            unsafe_offset=wavefront_plane_index(
                WavePathFloatAbi.TZ, capacity, idx
            )
        ],
        fields[
            unsafe_offset=wavefront_plane_index(
                WavePathFloatAbi.BSDF_PDF, capacity, idx
            )
        ],
        (packed_path_id & WAVE_PATH_DELTA_BIT) != 0,
    )


@always_inline
def store_device_wave_path(
    path: DeviceWavePath,
    path_ids: Pointer[UInt32, MutAnyOrigin],
    fields: Pointer[Float32, MutAnyOrigin],
    capacity: Int,
    idx: Int,
):
    path_ids[unsafe_offset=idx] = (path.path_id & WAVE_PATH_ID_MASK) | (
        WAVE_PATH_DELTA_BIT if path.delta else UInt32(0)
    )
    fields[
        unsafe_offset=wavefront_plane_index(WavePathFloatAbi.OX, capacity, idx)
    ] = path.ox
    fields[
        unsafe_offset=wavefront_plane_index(WavePathFloatAbi.OY, capacity, idx)
    ] = path.oy
    fields[
        unsafe_offset=wavefront_plane_index(WavePathFloatAbi.OZ, capacity, idx)
    ] = path.oz
    fields[
        unsafe_offset=wavefront_plane_index(
            WavePathFloatAbi.T_MIN, capacity, idx
        )
    ] = path.t_min
    fields[
        unsafe_offset=wavefront_plane_index(WavePathFloatAbi.DX, capacity, idx)
    ] = path.dx
    fields[
        unsafe_offset=wavefront_plane_index(WavePathFloatAbi.DY, capacity, idx)
    ] = path.dy
    fields[
        unsafe_offset=wavefront_plane_index(WavePathFloatAbi.DZ, capacity, idx)
    ] = path.dz
    fields[
        unsafe_offset=wavefront_plane_index(
            WavePathFloatAbi.T_MAX, capacity, idx
        )
    ] = path.t_max
    fields[
        unsafe_offset=wavefront_plane_index(WavePathFloatAbi.TX, capacity, idx)
    ] = path.tx
    fields[
        unsafe_offset=wavefront_plane_index(WavePathFloatAbi.TY, capacity, idx)
    ] = path.ty
    fields[
        unsafe_offset=wavefront_plane_index(WavePathFloatAbi.TZ, capacity, idx)
    ] = path.tz
    fields[
        unsafe_offset=wavefront_plane_index(
            WavePathFloatAbi.BSDF_PDF, capacity, idx
        )
    ] = path.bsdf_pdf


@always_inline
def load_device_wave_shade(
    path_refs: Pointer[UInt32, ImmutAnyOrigin],
    surface_values: Pointer[UInt32, ImmutAnyOrigin],
    fields: Pointer[Float32, ImmutAnyOrigin],
    capacity: Int,
    idx: Int,
) -> DeviceWaveShade:
    var path_ref = path_refs[unsafe_offset=idx]
    return DeviceWaveShade(
        path_ref & PATH_INDEX_MASK,
        fields[
            unsafe_offset=wavefront_plane_index(
                WaveShadeFloatAbi.NX, capacity, idx
            )
        ],
        fields[
            unsafe_offset=wavefront_plane_index(
                WaveShadeFloatAbi.NY, capacity, idx
            )
        ],
        fields[
            unsafe_offset=wavefront_plane_index(
                WaveShadeFloatAbi.NZ, capacity, idx
            )
        ],
        surface_values[unsafe_offset=idx],
        fields[
            unsafe_offset=wavefront_plane_index(
                WaveShadeFloatAbi.T, capacity, idx
            )
        ],
        (path_ref & FRONT_FACE_BIT) != 0,
    )


@always_inline
def store_device_wave_shade(
    work: DeviceWaveShade,
    path_refs: Pointer[UInt32, MutAnyOrigin],
    surface_values: Pointer[UInt32, MutAnyOrigin],
    fields: Pointer[Float32, MutAnyOrigin],
    capacity: Int,
    idx: Int,
):
    var path_ref = work.path_idx
    if work.front_face:
        path_ref |= FRONT_FACE_BIT
    path_refs[unsafe_offset=idx] = path_ref
    surface_values[unsafe_offset=idx] = work.surface_value
    fields[
        unsafe_offset=wavefront_plane_index(WaveShadeFloatAbi.NX, capacity, idx)
    ] = work.nx
    fields[
        unsafe_offset=wavefront_plane_index(WaveShadeFloatAbi.NY, capacity, idx)
    ] = work.ny
    fields[
        unsafe_offset=wavefront_plane_index(WaveShadeFloatAbi.NZ, capacity, idx)
    ] = work.nz
    fields[
        unsafe_offset=wavefront_plane_index(WaveShadeFloatAbi.T, capacity, idx)
    ] = work.t


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
        self.path_ids[idx] = (path.path_id & WAVE_PATH_ID_MASK) | (
            WAVE_PATH_DELTA_BIT if path.delta else UInt32(0)
        )
        self.fields[
            wavefront_plane_index(WavePathFloatAbi.OX, self.capacity, idx)
        ] = path.ox
        self.fields[
            wavefront_plane_index(WavePathFloatAbi.OY, self.capacity, idx)
        ] = path.oy
        self.fields[
            wavefront_plane_index(WavePathFloatAbi.OZ, self.capacity, idx)
        ] = path.oz
        self.fields[
            wavefront_plane_index(WavePathFloatAbi.T_MIN, self.capacity, idx)
        ] = path.t_min
        self.fields[
            wavefront_plane_index(WavePathFloatAbi.DX, self.capacity, idx)
        ] = path.dx
        self.fields[
            wavefront_plane_index(WavePathFloatAbi.DY, self.capacity, idx)
        ] = path.dy
        self.fields[
            wavefront_plane_index(WavePathFloatAbi.DZ, self.capacity, idx)
        ] = path.dz
        self.fields[
            wavefront_plane_index(WavePathFloatAbi.T_MAX, self.capacity, idx)
        ] = path.t_max
        self.fields[
            wavefront_plane_index(WavePathFloatAbi.TX, self.capacity, idx)
        ] = path.tx
        self.fields[
            wavefront_plane_index(WavePathFloatAbi.TY, self.capacity, idx)
        ] = path.ty
        self.fields[
            wavefront_plane_index(WavePathFloatAbi.TZ, self.capacity, idx)
        ] = path.tz
        self.fields[
            wavefront_plane_index(WavePathFloatAbi.BSDF_PDF, self.capacity, idx)
        ] = path.bsdf_pdf

    def get(self, idx: Int) -> WavePath:
        debug_assert["safe", _use_compiler_assume=True](
            idx >= 0 and idx < self.count
        )
        return DeviceWavePath(
            self.path_ids[idx] & WAVE_PATH_ID_MASK,
            self.fields[
                wavefront_plane_index(WavePathFloatAbi.OX, self.capacity, idx)
            ],
            self.fields[
                wavefront_plane_index(WavePathFloatAbi.OY, self.capacity, idx)
            ],
            self.fields[
                wavefront_plane_index(WavePathFloatAbi.OZ, self.capacity, idx)
            ],
            self.fields[
                wavefront_plane_index(
                    WavePathFloatAbi.T_MIN, self.capacity, idx
                )
            ],
            self.fields[
                wavefront_plane_index(WavePathFloatAbi.DX, self.capacity, idx)
            ],
            self.fields[
                wavefront_plane_index(WavePathFloatAbi.DY, self.capacity, idx)
            ],
            self.fields[
                wavefront_plane_index(WavePathFloatAbi.DZ, self.capacity, idx)
            ],
            self.fields[
                wavefront_plane_index(
                    WavePathFloatAbi.T_MAX, self.capacity, idx
                )
            ],
            self.fields[
                wavefront_plane_index(WavePathFloatAbi.TX, self.capacity, idx)
            ],
            self.fields[
                wavefront_plane_index(WavePathFloatAbi.TY, self.capacity, idx)
            ],
            self.fields[
                wavefront_plane_index(WavePathFloatAbi.TZ, self.capacity, idx)
            ],
            self.fields[
                wavefront_plane_index(
                    WavePathFloatAbi.BSDF_PDF, self.capacity, idx
                )
            ],
            (self.path_ids[idx] & WAVE_PATH_DELTA_BIT) != 0,
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
        var record = DeviceWaveShade.from_work(work)
        var path_ref = record.path_idx
        if record.front_face:
            path_ref |= FRONT_FACE_BIT
        self.path_refs[self.count] = path_ref
        self.surface_values[self.count] = record.surface_value
        self.fields[
            wavefront_plane_index(
                WaveShadeFloatAbi.NX, self.capacity, self.count
            )
        ] = record.nx
        self.fields[
            wavefront_plane_index(
                WaveShadeFloatAbi.NY, self.capacity, self.count
            )
        ] = record.ny
        self.fields[
            wavefront_plane_index(
                WaveShadeFloatAbi.NZ, self.capacity, self.count
            )
        ] = record.nz
        self.fields[
            wavefront_plane_index(
                WaveShadeFloatAbi.T, self.capacity, self.count
            )
        ] = record.t
        self.count += 1

    def get(self, idx: Int) -> WaveShade:
        debug_assert["safe", _use_compiler_assume=True](
            idx >= 0 and idx < self.count
        )
        var path_ref = self.path_refs[idx]
        return DeviceWaveShade(
            path_ref & PATH_INDEX_MASK,
            self.fields[
                wavefront_plane_index(WaveShadeFloatAbi.NX, self.capacity, idx)
            ],
            self.fields[
                wavefront_plane_index(WaveShadeFloatAbi.NY, self.capacity, idx)
            ],
            self.fields[
                wavefront_plane_index(WaveShadeFloatAbi.NZ, self.capacity, idx)
            ],
            self.surface_values[idx],
            self.fields[
                wavefront_plane_index(WaveShadeFloatAbi.T, self.capacity, idx)
            ],
            (path_ref & FRONT_FACE_BIT) != 0,
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
    var paths = List[WavePath](capacity=len(packed))
    for i in range(len(packed)):
        paths.append(packed.get(i))
    return paths^


def pack_wave_shades(
    works: ImmSpan[WaveShade, _], capacity: Int = -1
) -> PackedWaveShadeQueue:
    var packed_capacity = max(len(works), capacity)
    var packed = PackedWaveShadeQueue(packed_capacity)
    for work in works:
        packed.append(work)
    return packed^


def unpack_wave_shades(packed: PackedWaveShadeQueue) -> List[WaveShade]:
    var works = List[WaveShade](capacity=len(packed))
    for i in range(len(packed)):
        works.append(packed.get(i))
    return works^


struct WavefrontCounterBlock:
    """Host mirror of the device atomic counter block."""

    var values: List[UInt32]

    def __init__(out self):
        self.values = List[UInt32](length=WAVE_COUNTER.COUNT, fill=UInt32(0))

    def reset_outputs(mut self):
        for i in range(1, WAVE_COUNTER.COUNT):
            self.values[i] = UInt32(0)

    def begin(mut self, active_count: UInt32):
        self.values[WAVE_COUNTER.ACTIVE] = active_count
        self.reset_outputs()
        self.values[WAVE_COUNTER.STATUS] = WAVE_STATUS.OK

    def finish_bounce(mut self):
        self.values[WAVE_COUNTER.ACTIVE] = self.values[WAVE_COUNTER.NEXT]
        self.reset_outputs()


@fieldwise_init
struct WavefrontDispatchState(Copyable, Writable):
    """Host stage state; buffers ping-pong instead of moving path records."""

    var chunk_path_begin: UInt32
    var chunk_path_count: UInt32
    var bounce: UInt32
    var active_slot: UInt32

    def rng_stage(self) -> UInt32:
        return wavefront_rng_stage(self.bounce)

    def advance_bounce(mut self):
        self.bounce += UInt32(1)
        self.active_slot ^= UInt32(1)
