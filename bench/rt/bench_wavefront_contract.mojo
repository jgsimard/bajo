"""Host packing cost and memory accounting for the GPU wavefront ABI."""

from std.math import round
from std.time import perf_counter_ns

from bajo.bvh.camera import Camera
from bajo.core import Point3W, Vec3W
from bajo.core.utils import ns_to_ms
from bajo.rt import RenderSettings
from bajo.rt.cpu.wavefront.primary import _initialize_path_packets_range
from bajo.rt.wavefront_queue import PacketPathQueue
from bajo.rt.wavefront_contract import (
    WAVE_COUNTER,
    WavePathFloatAbi,
    WaveSampleFloatAbi,
    WaveShadeFloatAbi,
    packet_path_queue_to_wave_paths,
    pack_wave_paths,
    unpack_wave_paths,
)
from bench.rt.bench_cpu_end_to_end import sort_timings


comptime CAPACITY = 8192
comptime REPEATS = 101


def main():
    var settings = RenderSettings(320, 180, 4, UInt64(2026))
    var camera = Camera.from_vfov(
        Point3W(13.0, 2.0, 3.0),
        Point3W(0.0, 0.0, 0.0),
        Vec3W(0.0, 1.0, 0.0),
        20.0,
        10.0,
        0.6,
    )
    var packet_paths = PacketPathQueue[8](CAPACITY)
    _initialize_path_packets_range[8](
        packet_paths, settings, camera, 0, CAPACITY
    )
    var paths = packet_path_queue_to_wave_paths(packet_paths)
    var warm_packed = pack_wave_paths(paths, CAPACITY)
    var warm_unpacked = unpack_wave_paths(warm_packed)
    var guard = UInt64(warm_unpacked[CAPACITY - 1].path_id)

    var pack_times = List[Int](capacity=REPEATS)
    var unpack_times = List[Int](capacity=REPEATS)
    for _ in range(REPEATS):
        var pack_t0 = perf_counter_ns()
        var packed = pack_wave_paths(paths, CAPACITY)
        var pack_t1 = perf_counter_ns()
        guard += UInt64(packed.path_ids[CAPACITY - 1])
        pack_times.append(Int(pack_t1 - pack_t0))

        var unpack_t0 = perf_counter_ns()
        var unpacked = unpack_wave_paths(packed)
        var unpack_t1 = perf_counter_ns()
        guard += UInt64(unpacked[CAPACITY - 1].path_id)
        unpack_times.append(Int(unpack_t1 - unpack_t0))

    sort_timings(pack_times)
    sort_timings(unpack_times)
    var middle = (REPEATS - 1) >> 1
    var path_queue_bytes = CAPACITY * (4 + 4 * WavePathFloatAbi.PLANES)
    var shade_queue_bytes = CAPACITY * (8 + 4 * WaveShadeFloatAbi.PLANES)
    var arena_bytes = (
        2 * path_queue_bytes
        + 3 * shade_queue_bytes
        + CAPACITY * 4 * WaveSampleFloatAbi.PLANES
        + WAVE_COUNTER.COUNT * 4
    )
    print("GPU-shaped wavefront contract")
    print(t"  capacity={CAPACITY}, guard={guard}")
    print(
        t"  host AoS -> field-major median="
        t"{round(ns_to_ms(pack_times[middle]), 4)} ms"
    )
    print(
        t"  field-major -> host AoS median="
        t"{round(ns_to_ms(unpack_times[middle]), 4)} ms"
    )
    print(t"  path queue={path_queue_bytes} bytes")
    print(t"  shade queue={shade_queue_bytes} bytes")
    print(t"  full reusable arena={arena_bytes} bytes")
