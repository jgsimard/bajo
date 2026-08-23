"""Diagnostic coverage for the GPU field-major wavefront ABI."""

from std.testing import TestSuite, assert_equal, assert_true
from max.gpu.host import DeviceContext

from bajo.core import Frame, Point3f32, Rayf32, Vec3f32
from bajo.rt.gpu.wavefront_contract import (
    GpuWaveShadeQueue,
    GpuWavefrontArena,
    enqueue_wavefront_advance,
    enqueue_wavefront_contract_probe,
)
from bajo.rt.types import Color
from bajo.rt.wavefront_contract import (
    PackedWavePathQueue,
    WAVE_COUNTER,
    WAVE_STATUS,
    WavePathFloatAbi,
    WaveSampleFloatAbi,
    WaveShadeFloatAbi,
    wavefront_plane_index,
    wavefront_rng_subsequence,
    WAVE_PATH_DELTA_BIT,
)
from bajo.rt.wavefront_queue import (
    FRONT_FACE_BIT,
    PATH_INDEX_MASK,
    WavePath,
)


comptime CAPACITY = 13
comptime ACTIVE_COUNT = 11
comptime PATH_ID_BASE = UInt32(100)
comptime RNG_STAGE = UInt32(7)


def _path(path_id: UInt32) -> WavePath:
    var x = Float32(path_id)
    return WavePath(
        path_id,
        Rayf32[.WORLD](
            Point3f32[.WORLD](x, x + 1.0, x + 2.0),
            Vec3f32[.WORLD](x + 3.0, -(x + 4.0), x + 5.0),
            0.01,
            x + 1000.0,
        ),
        Color(x * 0.01, x * 0.02, x * 0.03),
    )


def _assert_path_equal(actual: WavePath, expected: WavePath) raises:
    assert_equal(actual.path_id, expected.path_id)
    assert_equal(actual.ray.o.x, expected.ray.o.x)
    assert_equal(actual.ray.o.y, expected.ray.o.y)
    assert_equal(actual.ray.o.z, expected.ray.o.z)
    assert_equal(actual.ray.t_min, expected.ray.t_min)
    assert_equal(actual.ray.d.x, expected.ray.d.x)
    assert_equal(actual.ray.d.y, expected.ray.d.y)
    assert_equal(actual.ray.d.z, expected.ray.d.z)
    assert_equal(actual.ray.t_max, expected.ray.t_max)
    assert_equal(actual.throughput.x, expected.throughput.x)
    assert_equal(actual.throughput.y, expected.throughput.y)
    assert_equal(actual.throughput.z, expected.throughput.z)


def _assert_tagged_shade_queue(
    queue: GpuWaveShadeQueue,
    count: Int,
    source: PackedWavePathQueue,
) raises:
    with queue.path_refs.map_to_host() as path_refs, queue.surface_values.map_to_host() as surfaces, queue.fields.map_to_host() as fields:
        for slot in range(count):
            var path_ref = path_refs[slot]
            var path_idx = Int(path_ref & PATH_INDEX_MASK)
            assert_true(path_idx >= 0 and path_idx < len(source))
            var path = source.get(path_idx)
            var expected_kind = path.path_id % UInt32(3)
            assert_equal(surfaces[slot] >> UInt32(28), expected_kind)
            assert_equal(surfaces[slot] & UInt32(0x0FFFFFFF), path.path_id)
            assert_equal(
                fields[
                    wavefront_plane_index(
                        WaveShadeFloatAbi.NX, queue.capacity, slot
                    )
                ],
                -path.ray.d.x,
            )
            assert_equal(
                fields[
                    wavefront_plane_index(
                        WaveShadeFloatAbi.NY, queue.capacity, slot
                    )
                ],
                -path.ray.d.y,
            )
            assert_equal(
                fields[
                    wavefront_plane_index(
                        WaveShadeFloatAbi.NZ, queue.capacity, slot
                    )
                ],
                -path.ray.d.z,
            )
            assert_equal(
                fields[
                    wavefront_plane_index(
                        WaveShadeFloatAbi.T, queue.capacity, slot
                    )
                ],
                path.ray.t_min + Float32(path_idx),
            )
            assert_equal(
                (path_ref & FRONT_FACE_BIT) != 0,
                (path.path_id & UInt32(1)) == 0,
            )


def test_gpu_wavefront_contract_atomic_roundtrip() raises:
    var source = PackedWavePathQueue(CAPACITY)
    for i in range(ACTIVE_COUNT):
        source.append(_path(PATH_ID_BASE + UInt32(i)))

    with DeviceContext() as ctx:
        var arena = GpuWavefrontArena(ctx, CAPACITY)
        assert_equal(len(arena.path_a.path_ids), CAPACITY)
        assert_equal(
            len(arena.path_a.fields), CAPACITY * WavePathFloatAbi.PLANES
        )
        assert_equal(
            len(arena.shade.fields),
            CAPACITY * WaveShadeFloatAbi.PLANES,
        )
        assert_equal(
            len(arena.sample_radiance),
            CAPACITY * WaveSampleFloatAbi.PLANES,
        )

        arena.upload_active(ctx, source)
        var subsequences = ctx.enqueue_create_buffer[.uint64](CAPACITY)
        enqueue_wavefront_contract_probe(
            ctx, arena, subsequences, ACTIVE_COUNT, RNG_STAGE
        )
        ctx.synchronize()

        var shade_count: Int
        with arena.counters.map_to_host() as counters:
            assert_equal(counters[WAVE_COUNTER.ACTIVE], UInt32(ACTIVE_COUNT))
            assert_equal(counters[WAVE_COUNTER.NEXT], UInt32(ACTIVE_COUNT))
            assert_equal(counters[WAVE_COUNTER.STATUS], WAVE_STATUS.OK)
            shade_count = Int(counters[WAVE_COUNTER.SHADE])
            assert_equal(shade_count, ACTIVE_COUNT)

        var output = arena.download_next(ctx, ACTIVE_COUNT)
        with arena.path_b.fields.map_to_host() as output_fields, arena.path_b.path_ids.map_to_host() as output_ids:
            for slot in range(ACTIVE_COUNT):
                assert_equal(
                    output_fields[
                        wavefront_plane_index(
                            WavePathFloatAbi.BSDF_PDF, CAPACITY, slot
                        )
                    ],
                    Float32(0.0),
                )
                assert_true((output_ids[slot] & WAVE_PATH_DELTA_BIT) != 0)
        with subsequences.map_to_host() as host_subsequences:
            for slot in range(ACTIVE_COUNT):
                var path = output.get(slot)
                var source_idx = Int(path.path_id - PATH_ID_BASE)
                assert_true(source_idx >= 0 and source_idx < ACTIVE_COUNT)
                _assert_path_equal(path, source.get(source_idx))
                assert_equal(
                    host_subsequences[slot],
                    wavefront_rng_subsequence(path.path_id, RNG_STAGE),
                )

        _assert_tagged_shade_queue(arena.shade, shade_count, source)

        enqueue_wavefront_advance(ctx, arena)
        ctx.synchronize()
        with arena.counters.map_to_host() as counters:
            assert_equal(counters[WAVE_COUNTER.ACTIVE], UInt32(ACTIVE_COUNT))
            assert_equal(counters[WAVE_COUNTER.NEXT], UInt32(0))
            assert_equal(counters[WAVE_COUNTER.SHADE], UInt32(0))
            assert_equal(counters[WAVE_COUNTER.STATUS], WAVE_STATUS.OK)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
