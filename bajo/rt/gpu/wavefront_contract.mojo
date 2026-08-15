"""GPU arena and executable probe for the shared wavefront ABI."""

from std.atomic import Atomic, Ordering
from std.gpu import global_idx
from std.math import ceildiv
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.rt.wavefront_contract import (
    DeviceWaveShade,
    PackedWavePathQueue,
    WAVE_COUNTER,
    WAVE_STATUS,
    WavePathFloatAbi,
    WaveSampleFloatAbi,
    WaveShadeFloatAbi,
    load_device_wave_path,
    store_device_wave_path,
    store_device_wave_shade,
    wavefront_rng_subsequence,
)


comptime WAVEFRONT_CONTRACT_BLOCK_SIZE = 128


struct GpuWavePathQueue:
    var capacity: Int
    var path_ids: DeviceBuffer[DType.uint32]
    var fields: DeviceBuffer[DType.float32]

    def __init__(out self, mut ctx: DeviceContext, capacity: Int) raises:
        debug_assert["safe", _use_compiler_assume=True](capacity > 0)
        self.capacity = capacity
        self.path_ids = ctx.enqueue_create_buffer[DType.uint32](capacity)
        self.fields = ctx.enqueue_create_buffer[DType.float32](
            capacity * WavePathFloatAbi.PLANES
        )


struct GpuWaveShadeQueue:
    var capacity: Int
    var path_refs: DeviceBuffer[DType.uint32]
    var surface_values: DeviceBuffer[DType.uint32]
    var fields: DeviceBuffer[DType.float32]

    def __init__(out self, mut ctx: DeviceContext, capacity: Int) raises:
        debug_assert["safe", _use_compiler_assume=True](capacity > 0)
        self.capacity = capacity
        self.path_refs = ctx.enqueue_create_buffer[DType.uint32](capacity)
        self.surface_values = ctx.enqueue_create_buffer[DType.uint32](capacity)
        self.fields = ctx.enqueue_create_buffer[DType.float32](
            capacity * WaveShadeFloatAbi.PLANES
        )


struct GpuWavefrontArena:
    """Reusable bounded storage for one serially submitted GPU path chunk."""

    var capacity: Int
    var path_a: GpuWavePathQueue
    var path_b: GpuWavePathQueue
    var lambertian: GpuWaveShadeQueue
    var metal: GpuWaveShadeQueue
    var dielectric: GpuWaveShadeQueue
    var counters: DeviceBuffer[DType.uint32]
    var sample_radiance: DeviceBuffer[DType.float32]

    def __init__(out self, mut ctx: DeviceContext, capacity: Int) raises:
        debug_assert["safe", _use_compiler_assume=True](capacity > 0)
        self.capacity = capacity
        self.path_a = GpuWavePathQueue(ctx, capacity)
        self.path_b = GpuWavePathQueue(ctx, capacity)
        self.lambertian = GpuWaveShadeQueue(ctx, capacity)
        self.metal = GpuWaveShadeQueue(ctx, capacity)
        self.dielectric = GpuWaveShadeQueue(ctx, capacity)
        self.counters = ctx.enqueue_create_buffer[DType.uint32](
            WAVE_COUNTER.COUNT
        )
        self.sample_radiance = ctx.enqueue_create_buffer[DType.float32](
            capacity * WaveSampleFloatAbi.PLANES
        )

    def upload_active(
        mut self, ctx: DeviceContext, packed: PackedWavePathQueue
    ) raises:
        debug_assert["safe", _use_compiler_assume=True](
            packed.capacity == self.capacity,
            "host/device path capacities must match",
        )
        with self.path_a.path_ids.map_to_host() as ids:
            for i in range(self.capacity):
                ids[i] = packed.path_ids[i]
        with self.path_a.fields.map_to_host() as fields:
            for i in range(len(packed.fields)):
                fields[i] = packed.fields[i]
        ctx.synchronize()

    def download_next(
        self, ctx: DeviceContext, count: Int
    ) raises -> PackedWavePathQueue:
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


def _init_wavefront_contract_counters_kernel(
    counters: Pointer[UInt32, MutAnyOrigin], active_count: UInt32
):
    var idx = global_idx.x
    if idx < WAVE_COUNTER.COUNT:
        counters[unsafe_offset=idx] = UInt32(0)
    if idx == 0:
        counters[unsafe_offset=WAVE_COUNTER.ACTIVE] = active_count
        counters[unsafe_offset=WAVE_COUNTER.STATUS] = WAVE_STATUS.OK


def _advance_wavefront_contract_counters_kernel(
    counters: Pointer[UInt32, MutAnyOrigin],
):
    """Promote the compacted queue without a device-to-host synchronization."""
    var idx = global_idx.x
    if idx == WAVE_COUNTER.ACTIVE:
        counters[unsafe_offset=WAVE_COUNTER.ACTIVE] = counters[
            unsafe_offset=WAVE_COUNTER.NEXT
        ]
    elif idx == WAVE_COUNTER.NEXT:
        counters[unsafe_offset=WAVE_COUNTER.NEXT] = UInt32(0)
    elif idx == WAVE_COUNTER.LAMBERTIAN:
        counters[unsafe_offset=WAVE_COUNTER.LAMBERTIAN] = UInt32(0)
    elif idx == WAVE_COUNTER.METAL:
        counters[unsafe_offset=WAVE_COUNTER.METAL] = UInt32(0)
    elif idx == WAVE_COUNTER.DIELECTRIC:
        counters[unsafe_offset=WAVE_COUNTER.DIELECTRIC] = UInt32(0)
    elif idx == WAVE_COUNTER.ESCAPED:
        counters[unsafe_offset=WAVE_COUNTER.ESCAPED] = UInt32(0)
    elif idx == WAVE_COUNTER.ABSORBED:
        counters[unsafe_offset=WAVE_COUNTER.ABSORBED] = UInt32(0)


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
    lambert_path_refs: Pointer[UInt32, MutAnyOrigin],
    lambert_surfaces: Pointer[UInt32, MutAnyOrigin],
    lambert_fields: Pointer[Float32, MutAnyOrigin],
    metal_path_refs: Pointer[UInt32, MutAnyOrigin],
    metal_surfaces: Pointer[UInt32, MutAnyOrigin],
    metal_fields: Pointer[Float32, MutAnyOrigin],
    dielectric_path_refs: Pointer[UInt32, MutAnyOrigin],
    dielectric_surfaces: Pointer[UInt32, MutAnyOrigin],
    dielectric_fields: Pointer[Float32, MutAnyOrigin],
    counters: Pointer[UInt32, MutAnyOrigin],
    subsequences: Pointer[UInt64, MutAnyOrigin],
    capacity_i32: Int32,
    active_count_i32: Int32,
    rng_stage: UInt32,
):
    var idx = global_idx.x
    var active_count = Int(active_count_i32)
    if idx >= active_count:
        return
    var capacity = Int(capacity_i32)
    var path = load_device_wave_path(
        src_path_ids, src_path_fields, capacity, idx
    )

    # The atomic output slot is the exact contract used by a future scatter
    # kernel. Queue order may change; path ID owns every stochastic stream.
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
    if kind == UInt32(0):
        var slot = _reserve_slot(counters, WAVE_COUNTER.LAMBERTIAN)
        if slot < capacity:
            store_device_wave_shade(
                work,
                lambert_path_refs,
                lambert_surfaces,
                lambert_fields,
                capacity,
                slot,
            )
        else:
            _mark_status(counters, WAVE_STATUS.SHADE_OVERFLOW)
    elif kind == UInt32(1):
        var slot = _reserve_slot(counters, WAVE_COUNTER.METAL)
        if slot < capacity:
            store_device_wave_shade(
                work,
                metal_path_refs,
                metal_surfaces,
                metal_fields,
                capacity,
                slot,
            )
        else:
            _mark_status(counters, WAVE_STATUS.SHADE_OVERFLOW)
    else:
        var slot = _reserve_slot(counters, WAVE_COUNTER.DIELECTRIC)
        if slot < capacity:
            store_device_wave_shade(
                work,
                dielectric_path_refs,
                dielectric_surfaces,
                dielectric_fields,
                capacity,
                slot,
            )
        else:
            _mark_status(counters, WAVE_STATUS.SHADE_OVERFLOW)


def enqueue_wavefront_contract_probe(
    ctx: DeviceContext,
    mut arena: GpuWavefrontArena,
    subsequences: DeviceBuffer[DType.uint64],
    active_count: Int,
    rng_stage: UInt32,
) raises:
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
        block_dim=WAVE_COUNTER.COUNT,
    )
    if active_count == 0:
        return
    ctx.enqueue_function[wavefront_contract_probe_kernel](
        arena.path_a.path_ids,
        arena.path_a.fields,
        arena.path_b.path_ids,
        arena.path_b.fields,
        arena.lambertian.path_refs,
        arena.lambertian.surface_values,
        arena.lambertian.fields,
        arena.metal.path_refs,
        arena.metal.surface_values,
        arena.metal.fields,
        arena.dielectric.path_refs,
        arena.dielectric.surface_values,
        arena.dielectric.fields,
        arena.counters,
        subsequences,
        Int32(arena.capacity),
        Int32(active_count),
        rng_stage,
        grid_dim=ceildiv(active_count, WAVEFRONT_CONTRACT_BLOCK_SIZE),
        block_dim=WAVEFRONT_CONTRACT_BLOCK_SIZE,
    )


def enqueue_wavefront_advance(
    ctx: DeviceContext, mut arena: GpuWavefrontArena
) raises:
    """Advance queue counters entirely on-device after all shade stages."""
    ctx.enqueue_function[_advance_wavefront_contract_counters_kernel](
        arena.counters,
        grid_dim=1,
        block_dim=WAVE_COUNTER.COUNT,
    )
