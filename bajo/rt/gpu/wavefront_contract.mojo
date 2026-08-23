"""GPU arena and executable probe for the shared wavefront ABI."""

from std.atomic import Atomic, Ordering
from std.gpu import global_idx
from std.math import ceildiv
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh.constants import f32_max
from bajo.rt.types import Integrator
from bajo.rt.wavefront_contract import (
    DeviceWavePath,
    DeviceWaveShade,
    DeviceWaveShadow,
    PackedWavePathQueue,
    WAVE_PATH_DELTA_BIT,
    WAVE_PATH_ID_MASK,
    WAVE_COUNTER,
    WAVE_STATUS,
    WavePathFloatAbi,
    WaveSampleFloatAbi,
    WaveShadeFloatAbi,
    WaveShadowFloatAbi,
    load_device_wave_path,
    store_device_wave_path,
    store_device_wave_shade,
    wavefront_plane_index,
    wavefront_rng_subsequence,
)


comptime WAVEFRONT_CONTRACT_BLOCK_SIZE = 128


@always_inline
def load_gpu_rt_path[
    ALGORITHM: Integrator,
](
    path_ids: Pointer[UInt32, ImmutAnyOrigin],
    fields: Pointer[Float32, ImmutAnyOrigin],
    capacity: Int,
    idx: Int,
) -> DeviceWavePath:
    """Load only the path planes consumed by one compile-time integrator."""
    comptime assert ALGORITHM.is_valid()
    var packed_path_id = path_ids[unsafe_offset=idx]
    var tx = Float32(1.0)
    var ty = Float32(1.0)
    var tz = Float32(1.0)
    comptime if ALGORITHM in (Integrator.PATH, Integrator.NEE, Integrator.MIS):
        tx = fields[
            unsafe_offset=wavefront_plane_index(
                WavePathFloatAbi.TX, capacity, idx
            )
        ]
        ty = fields[
            unsafe_offset=wavefront_plane_index(
                WavePathFloatAbi.TY, capacity, idx
            )
        ]
        tz = fields[
            unsafe_offset=wavefront_plane_index(
                WavePathFloatAbi.TZ, capacity, idx
            )
        ]
    var bsdf_pdf = Float32(0.0)
    comptime if ALGORITHM == .MIS:
        bsdf_pdf = fields[
            unsafe_offset=wavefront_plane_index(
                WavePathFloatAbi.BSDF_PDF, capacity, idx
            )
        ]
    var delta = True
    comptime if ALGORITHM in (Integrator.NEE, Integrator.MIS):
        delta = (packed_path_id & WAVE_PATH_DELTA_BIT) != 0
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
        0.001,
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
        f32_max,
        tx,
        ty,
        tz,
        bsdf_pdf,
        delta,
    )


@always_inline
def store_gpu_rt_path[
    ALGORITHM: Integrator,
](
    path: DeviceWavePath,
    path_ids: Pointer[UInt32, MutAnyOrigin],
    fields: Pointer[Float32, MutAnyOrigin],
    capacity: Int,
    idx: Int,
):
    """Store only the path planes consumed by one compile-time integrator."""
    comptime assert ALGORITHM.is_valid()
    var packed_path_id = path.path_id & WAVE_PATH_ID_MASK
    comptime if ALGORITHM in (Integrator.NEE, Integrator.MIS):
        if path.delta:
            packed_path_id |= WAVE_PATH_DELTA_BIT
    path_ids[unsafe_offset=idx] = packed_path_id
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
        unsafe_offset=wavefront_plane_index(WavePathFloatAbi.DX, capacity, idx)
    ] = path.dx
    fields[
        unsafe_offset=wavefront_plane_index(WavePathFloatAbi.DY, capacity, idx)
    ] = path.dy
    fields[
        unsafe_offset=wavefront_plane_index(WavePathFloatAbi.DZ, capacity, idx)
    ] = path.dz
    comptime if ALGORITHM in (Integrator.PATH, Integrator.NEE, Integrator.MIS):
        fields[
            unsafe_offset=wavefront_plane_index(
                WavePathFloatAbi.TX, capacity, idx
            )
        ] = path.tx
        fields[
            unsafe_offset=wavefront_plane_index(
                WavePathFloatAbi.TY, capacity, idx
            )
        ] = path.ty
        fields[
            unsafe_offset=wavefront_plane_index(
                WavePathFloatAbi.TZ, capacity, idx
            )
        ] = path.tz
    comptime if ALGORITHM == .MIS:
        fields[
            unsafe_offset=wavefront_plane_index(
                WavePathFloatAbi.BSDF_PDF, capacity, idx
            )
        ] = path.bsdf_pdf


@always_inline
def load_gpu_rt_shadow[
    LOAD_CONTRIBUTION: Bool,
](
    path_ids: Pointer[UInt32, ImmutAnyOrigin],
    fields: Pointer[Float32, ImmutAnyOrigin],
    capacity: Int,
    idx: Int,
) -> DeviceWaveShadow:
    """Load an RT shadow ray while reconstructing its invariant lower bound."""
    var r = Float32(0.0)
    var g = Float32(0.0)
    var b = Float32(0.0)
    comptime if LOAD_CONTRIBUTION:
        r = fields[
            unsafe_offset=wavefront_plane_index(
                WaveShadowFloatAbi.R, capacity, idx
            )
        ]
        g = fields[
            unsafe_offset=wavefront_plane_index(
                WaveShadowFloatAbi.G, capacity, idx
            )
        ]
        b = fields[
            unsafe_offset=wavefront_plane_index(
                WaveShadowFloatAbi.B, capacity, idx
            )
        ]
    return DeviceWaveShadow(
        path_ids[unsafe_offset=idx],
        fields[
            unsafe_offset=wavefront_plane_index(
                WaveShadowFloatAbi.OX, capacity, idx
            )
        ],
        fields[
            unsafe_offset=wavefront_plane_index(
                WaveShadowFloatAbi.OY, capacity, idx
            )
        ],
        fields[
            unsafe_offset=wavefront_plane_index(
                WaveShadowFloatAbi.OZ, capacity, idx
            )
        ],
        0.001,
        fields[
            unsafe_offset=wavefront_plane_index(
                WaveShadowFloatAbi.DX, capacity, idx
            )
        ],
        fields[
            unsafe_offset=wavefront_plane_index(
                WaveShadowFloatAbi.DY, capacity, idx
            )
        ],
        fields[
            unsafe_offset=wavefront_plane_index(
                WaveShadowFloatAbi.DZ, capacity, idx
            )
        ],
        fields[
            unsafe_offset=wavefront_plane_index(
                WaveShadowFloatAbi.T_MAX, capacity, idx
            )
        ],
        r,
        g,
        b,
    )


@always_inline
def store_gpu_rt_shadow[
    STORE_CONTRIBUTION: Bool,
](
    work: DeviceWaveShadow,
    path_ids: Pointer[UInt32, MutAnyOrigin],
    fields: Pointer[Float32, MutAnyOrigin],
    capacity: Int,
    idx: Int,
):
    """Store an RT shadow ray without writing its invariant lower bound."""
    path_ids[unsafe_offset=idx] = work.path_id
    fields[
        unsafe_offset=wavefront_plane_index(
            WaveShadowFloatAbi.OX, capacity, idx
        )
    ] = work.ox
    fields[
        unsafe_offset=wavefront_plane_index(
            WaveShadowFloatAbi.OY, capacity, idx
        )
    ] = work.oy
    fields[
        unsafe_offset=wavefront_plane_index(
            WaveShadowFloatAbi.OZ, capacity, idx
        )
    ] = work.oz
    fields[
        unsafe_offset=wavefront_plane_index(
            WaveShadowFloatAbi.DX, capacity, idx
        )
    ] = work.dx
    fields[
        unsafe_offset=wavefront_plane_index(
            WaveShadowFloatAbi.DY, capacity, idx
        )
    ] = work.dy
    fields[
        unsafe_offset=wavefront_plane_index(
            WaveShadowFloatAbi.DZ, capacity, idx
        )
    ] = work.dz
    fields[
        unsafe_offset=wavefront_plane_index(
            WaveShadowFloatAbi.T_MAX, capacity, idx
        )
    ] = work.t_max
    comptime if STORE_CONTRIBUTION:
        fields[
            unsafe_offset=wavefront_plane_index(
                WaveShadowFloatAbi.R, capacity, idx
            )
        ] = work.r
        fields[
            unsafe_offset=wavefront_plane_index(
                WaveShadowFloatAbi.G, capacity, idx
            )
        ] = work.g
        fields[
            unsafe_offset=wavefront_plane_index(
                WaveShadowFloatAbi.B, capacity, idx
            )
        ] = work.b


struct GpuWavePathQueue:
    var capacity: Int
    var path_ids: DeviceBuffer[.uint32]
    var fields: DeviceBuffer[.float32]

    def __init__(out self, mut ctx: DeviceContext, capacity: Int) raises:
        debug_assert["safe", _use_compiler_assume=True](capacity > 0)
        self.capacity = capacity
        self.path_ids = ctx.enqueue_create_buffer[.uint32](capacity)
        self.fields = ctx.enqueue_create_buffer[.float32](
            capacity * WavePathFloatAbi.PLANES
        )


struct GpuWaveShadeQueue:
    var capacity: Int
    var path_refs: DeviceBuffer[.uint32]
    var surface_values: DeviceBuffer[.uint32]
    var fields: DeviceBuffer[.float32]

    def __init__(out self, mut ctx: DeviceContext, capacity: Int) raises:
        debug_assert["safe", _use_compiler_assume=True](capacity > 0)
        self.capacity = capacity
        self.path_refs = ctx.enqueue_create_buffer[.uint32](capacity)
        self.surface_values = ctx.enqueue_create_buffer[.uint32](capacity)
        self.fields = ctx.enqueue_create_buffer[.float32](
            capacity * WaveShadeFloatAbi.PLANES
        )


struct GpuWaveShadowQueue:
    var capacity: Int
    var path_ids: DeviceBuffer[.uint32]
    var fields: DeviceBuffer[.float32]

    def __init__(out self, mut ctx: DeviceContext, capacity: Int) raises:
        debug_assert["safe", _use_compiler_assume=True](capacity > 0)
        self.capacity = capacity
        self.path_ids = ctx.enqueue_create_buffer[.uint32](capacity)
        self.fields = ctx.enqueue_create_buffer[.float32](
            capacity * WaveShadowFloatAbi.PLANES
        )


struct GpuWavefrontArena:
    """Reusable bounded storage for one serially submitted GPU path chunk."""

    var capacity: Int
    var sample_base: UInt32
    var path_a: GpuWavePathQueue
    var path_b: GpuWavePathQueue
    var shade: GpuWaveShadeQueue
    var shadow: GpuWaveShadowQueue
    var counters: DeviceBuffer[.uint32]
    var sample_radiance: DeviceBuffer[.float32]

    def __init__(out self, mut ctx: DeviceContext, capacity: Int) raises:
        debug_assert["safe", _use_compiler_assume=True](capacity > 0)
        self.capacity = capacity
        self.sample_base = UInt32(0)
        self.path_a = GpuWavePathQueue(ctx, capacity)
        self.path_b = GpuWavePathQueue(ctx, capacity)
        self.shade = GpuWaveShadeQueue(ctx, capacity)
        self.shadow = GpuWaveShadowQueue(ctx, capacity)
        self.counters = ctx.enqueue_create_buffer[.uint32](
            WAVE_COUNTER.COUNT
        )
        self.sample_radiance = ctx.enqueue_create_buffer[.float32](
            capacity * WaveSampleFloatAbi.PLANES
        )

    def upload_active(
        mut self, ctx: DeviceContext, packed: PackedWavePathQueue
    ) raises:
        """Diagnostic adapter: upload a host-packed queue for ABI probes."""
        debug_assert["safe", _use_compiler_assume=True](
            packed.capacity == self.capacity,
            "host/device path capacities must match",
        )
        with self.path_a.path_ids.map_to_host() as ids:
            for i in range(self.capacity):
                ids[i] = packed.path_ids[i]
        with self.path_a.fields.map_to_host() as fields:
            for i, field in enumerate(packed.fields):
                fields[i] = field
        ctx.synchronize()

    def download_next(
        self, ctx: DeviceContext, count: Int
    ) raises -> PackedWavePathQueue:
        """Diagnostic adapter: download a queue after an ABI probe."""
        debug_assert["safe", _use_compiler_assume=True](
            count >= 0 and count <= self.capacity
        )
        ctx.synchronize()
        var packed = PackedWavePathQueue(self.capacity)
        packed.count = count
        with self.path_b.path_ids.map_to_host() as ids:
            for i in range(self.capacity):
                packed.path_ids[i] = ids[i]
        with self.path_b.fields.map_to_host() as fields:
            for i in range(len(packed.fields)):
                packed.fields[i] = fields[i]
        return packed^


@always_inline
def _reset_wavefront_output_counters(
    counters: Pointer[UInt32, MutAnyOrigin],
):
    counters[unsafe_offset=WAVE_COUNTER.NEXT] = UInt32(0)
    counters[unsafe_offset=WAVE_COUNTER.SHADE] = UInt32(0)
    counters[unsafe_offset=WAVE_COUNTER.SHADOW] = UInt32(0)


def _init_wavefront_contract_counters_kernel(
    counters: Pointer[UInt32, MutAnyOrigin], active_count: UInt32
):
    counters[unsafe_offset=WAVE_COUNTER.ACTIVE] = active_count
    _reset_wavefront_output_counters(counters)
    counters[unsafe_offset=WAVE_COUNTER.STATUS] = WAVE_STATUS.OK


def _advance_wavefront_contract_counters_kernel(
    counters: Pointer[UInt32, MutAnyOrigin],
):
    """Promote the compacted queue without a device-to-host synchronization."""
    counters[unsafe_offset=WAVE_COUNTER.ACTIVE] = counters[
        unsafe_offset=WAVE_COUNTER.NEXT
    ]
    _reset_wavefront_output_counters(counters)


@always_inline
def _reserve_slot(
    counters: Pointer[UInt32, MutAnyOrigin], counter_idx: Int
) -> Int:
    return Int(
        Atomic.fetch_add[ordering=Ordering.RELAXED](
            counters.unsafe_offset(counter_idx), UInt32(1)
        )
    )


@always_inline
def _mark_status(counters: Pointer[UInt32, MutAnyOrigin], status: UInt32):
    Atomic.store[ordering=Ordering.RELAXED](
        counters.unsafe_offset(WAVE_COUNTER.STATUS), status
    )


def wavefront_contract_probe_kernel(
    src_path_ids: Pointer[UInt32, ImmutAnyOrigin],
    src_path_fields: Pointer[Float32, ImmutAnyOrigin],
    dst_path_ids: Pointer[UInt32, MutAnyOrigin],
    dst_path_fields: Pointer[Float32, MutAnyOrigin],
    shade_path_refs: Pointer[UInt32, MutAnyOrigin],
    shade_surfaces: Pointer[UInt32, MutAnyOrigin],
    shade_fields: Pointer[Float32, MutAnyOrigin],
    counters: Pointer[UInt32, MutAnyOrigin],
    subsequences: Pointer[UInt64, MutAnyOrigin],
    capacity_i32: Int32,
    active_count_i32: Int32,
    rng_stage: UInt32,
):
    """Diagnostic kernel covering atomic compaction and tagged shade ABI."""
    var idx = global_idx.x
    var active_count = Int(active_count_i32)
    if idx >= active_count:
        return
    var capacity = Int(capacity_i32)
    var path = load_device_wave_path(
        src_path_ids, src_path_fields, capacity, idx
    )

    # Queue order may change; path ID owns every stochastic stream.
    var next_slot = _reserve_slot(counters, WAVE_COUNTER.NEXT)
    if next_slot >= capacity:
        _mark_status(counters, WAVE_STATUS.PATH_OVERFLOW)
        return
    store_device_wave_path(
        path, dst_path_ids, dst_path_fields, capacity, next_slot
    )
    subsequences[unsafe_offset=next_slot] = wavefront_rng_subsequence(
        path.path_id, rng_stage
    )

    var kind = path.path_id % UInt32(3)
    var work = DeviceWaveShade(
        UInt32(idx),
        -path.dx,
        -path.dy,
        -path.dz,
        (kind << UInt32(28)) | (path.path_id & UInt32(0x0FFFFFFF)),
        path.t_min + Float32(idx),
        (path.path_id & UInt32(1)) == 0,
    )
    var shade_slot = _reserve_slot(counters, WAVE_COUNTER.SHADE)
    if shade_slot < capacity:
        store_device_wave_shade(
            work,
            shade_path_refs,
            shade_surfaces,
            shade_fields,
            capacity,
            shade_slot,
        )
    else:
        _mark_status(counters, WAVE_STATUS.SHADE_OVERFLOW)


def enqueue_wavefront_contract_probe(
    ctx: DeviceContext,
    mut arena: GpuWavefrontArena,
    subsequences: DeviceBuffer[.uint64],
    active_count: Int,
    rng_stage: UInt32,
) raises:
    """Enqueue the diagnostic wavefront ABI probe; not used by rendering."""
    debug_assert["safe", _use_compiler_assume=True](
        active_count >= 0 and active_count <= arena.capacity
    )
    debug_assert["safe", _use_compiler_assume=True](
        len(subsequences) >= arena.capacity
    )
    ctx.enqueue_function[_init_wavefront_contract_counters_kernel](
        arena.counters,
        UInt32(active_count),
        grid_dim=1,
        block_dim=1,
    )
    if active_count == 0:
        return
    ctx.enqueue_function[wavefront_contract_probe_kernel](
        arena.path_a.path_ids,
        arena.path_a.fields,
        arena.path_b.path_ids,
        arena.path_b.fields,
        arena.shade.path_refs,
        arena.shade.surface_values,
        arena.shade.fields,
        arena.counters,
        subsequences,
        Int32(arena.capacity),
        Int32(active_count),
        rng_stage,
        grid_dim=ceildiv(active_count, WAVEFRONT_CONTRACT_BLOCK_SIZE),
        block_dim=WAVEFRONT_CONTRACT_BLOCK_SIZE,
    )


def enqueue_wavefront_begin(
    ctx: DeviceContext, mut arena: GpuWavefrontArena, active_count: Int
) raises:
    """Reset device counters and publish the initial active path count."""
    debug_assert["safe", _use_compiler_assume=True](
        active_count >= 0 and active_count <= arena.capacity
    )
    ctx.enqueue_function[_init_wavefront_contract_counters_kernel](
        arena.counters,
        UInt32(active_count),
        grid_dim=1,
        block_dim=1,
    )


def enqueue_wavefront_advance(
    ctx: DeviceContext, mut arena: GpuWavefrontArena
) raises:
    """Advance queue counters entirely on-device after all shade stages."""
    ctx.enqueue_function[_advance_wavefront_contract_counters_kernel](
        arena.counters,
        grid_dim=1,
        block_dim=1,
    )
