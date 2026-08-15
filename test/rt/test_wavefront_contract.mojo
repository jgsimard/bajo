from std.testing import TestSuite, assert_equal, assert_false, assert_true
from std.sys import size_of

from bajo.core import Frame, Point3f32, Rayf32, Vec3f32
from bajo.rt.types import Color, MAT, SurfaceHit, SurfaceId
from bajo.rt.wavefront_contract import (
    DeviceWavePath,
    DeviceWaveShade,
    PackedWavePathQueue,
    PackedWaveShadeQueue,
    WAVE_COUNTER,
    WAVE_STAGE,
    WAVE_STATUS,
    WavePathFloatAbi,
    WaveShadeFloatAbi,
    WavefrontCounterBlock,
    WavefrontDispatchState,
    pack_wave_paths,
    pack_wave_shades,
    unpack_wave_paths,
    unpack_wave_shades,
    wavefront_rng_stage,
    wavefront_rng_light_stage,
    wavefront_rng_roulette_stage,
    wavefront_rng_subsequence,
)
from bajo.rt.wavefront_queue import FRONT_FACE_BIT, WavePath, WaveShade


def _path(path_id: UInt32) -> WavePath:
    return WavePath(
        path_id,
        Rayf32[Frame.WORLD](
            Point3f32[Frame.WORLD](1.0, 2.0, 3.0),
            Vec3f32[Frame.WORLD](-4.0, 5.0, -6.0),
            0.125,
            99.0,
        ),
        Color(0.25, 0.5, 0.75),
    )


def test_wavefront_contract_abi_and_philox_ownership() raises:
    assert_equal(WavePathFloatAbi.PLANES, 11)
    assert_equal(WaveShadeFloatAbi.PLANES, 4)
    assert_equal(WAVE_COUNTER.COUNT, 8)
    assert_equal(size_of[DeviceWavePath](), 48)
    assert_equal(size_of[DeviceWaveShade](), 28)

    assert_equal(WAVE_STAGE.PRIMARY, UInt32(0))
    assert_true(WAVE_STAGE.TRACE != WAVE_STAGE.LAMBERTIAN)
    assert_true(WAVE_STAGE.LAMBERTIAN != WAVE_STAGE.METAL)
    assert_true(WAVE_STAGE.METAL != WAVE_STAGE.DIELECTRIC)
    assert_true(WAVE_STAGE.DIELECTRIC != WAVE_STAGE.RESOLVE)

    var expected = (UInt64(7) << UInt64(32)) | UInt64(3)
    assert_equal(wavefront_rng_subsequence(UInt32(7), UInt32(3)), expected)
    assert_equal(wavefront_rng_stage(UInt32(0)), UInt32(1))
    assert_equal(wavefront_rng_stage(UInt32(6)), UInt32(7))
    assert_equal(wavefront_rng_roulette_stage(UInt32(0)), UInt32(0x40000001))
    assert_equal(wavefront_rng_light_stage(UInt32(0)), UInt32(0x80000001))
    assert_true(
        wavefront_rng_roulette_stage(UInt32(6))
        != wavefront_rng_stage(UInt32(6))
    )
    assert_true(
        wavefront_rng_light_stage(UInt32(6))
        != wavefront_rng_roulette_stage(UInt32(6))
    )


def test_wavefront_path_field_major_roundtrip() raises:
    var packed = PackedWavePathQueue(5)
    var source = _path(UInt32(17))
    packed.append(source)
    assert_equal(len(packed), 1)
    assert_equal(len(packed.path_ids), 5)
    assert_equal(len(packed.fields), 5 * WavePathFloatAbi.PLANES)

    var loaded = packed.get(0)
    assert_equal(loaded.path_id, source.path_id)
    assert_equal(loaded.ray.o.x, source.ray.o.x)
    assert_equal(loaded.ray.o.y, source.ray.o.y)
    assert_equal(loaded.ray.o.z, source.ray.o.z)
    assert_equal(loaded.ray.t_min, source.ray.t_min)
    assert_equal(loaded.ray.d.x, source.ray.d.x)
    assert_equal(loaded.ray.d.y, source.ray.d.y)
    assert_equal(loaded.ray.d.z, source.ray.d.z)
    assert_equal(loaded.ray.t_max, source.ray.t_max)
    assert_equal(loaded.throughput.x, source.throughput.x)
    assert_equal(loaded.throughput.y, source.throughput.y)
    assert_equal(loaded.throughput.z, source.throughput.z)

    packed.clear()
    assert_equal(len(packed), 0)
    packed.append(_path(UInt32(23)))
    assert_equal(packed.get(0).path_id, UInt32(23))


def test_wavefront_shade_field_major_roundtrip() raises:
    var hit = SurfaceHit(
        Vec3f32[Frame.WORLD](0.0, 1.0, 0.0),
        SurfaceId(MAT.METAL, UInt32(9)),
        3.5,
        True,
        True,
    )
    var source = WaveShade(UInt32(41), hit.copy())
    var packed = PackedWaveShadeQueue(4)
    packed.append(source)
    assert_equal(len(packed), 1)
    assert_true((packed.path_refs[0] & FRONT_FACE_BIT) != 0)
    assert_equal(len(packed.fields), 4 * WaveShadeFloatAbi.PLANES)

    var loaded = packed.get(0)
    assert_equal(loaded.path_idx, source.path_idx)
    assert_equal(loaded.hit.normal.x, source.hit.normal.x)
    assert_equal(loaded.hit.normal.y, source.hit.normal.y)
    assert_equal(loaded.hit.normal.z, source.hit.normal.z)
    assert_equal(loaded.hit.surface.value, source.hit.surface.value)
    assert_equal(loaded.hit.t, source.hit.t)
    assert_true(loaded.hit.front_face)
    assert_true(loaded.hit.hit)

    var back = WaveShade(
        UInt32(2),
        SurfaceHit(
            hit.normal,
            hit.surface.copy(),
            hit.t,
            False,
            True,
        ),
    )
    packed.append(back)
    assert_false((packed.path_refs[1] & FRONT_FACE_BIT) != 0)
    assert_false(packed.get(1).hit.front_face)


def test_wavefront_counter_and_dispatch_transitions() raises:
    var counters = WavefrontCounterBlock()
    counters.begin(UInt32(128))
    assert_equal(counters.values[WAVE_COUNTER.ACTIVE], UInt32(128))
    assert_equal(counters.values[WAVE_COUNTER.STATUS], WAVE_STATUS.OK)
    counters.values[WAVE_COUNTER.NEXT] = UInt32(73)
    counters.values[WAVE_COUNTER.LAMBERTIAN] = UInt32(40)
    counters.values[WAVE_COUNTER.METAL] = UInt32(20)
    counters.values[WAVE_COUNTER.DIELECTRIC] = UInt32(13)
    counters.finish_bounce()
    assert_equal(counters.values[WAVE_COUNTER.ACTIVE], UInt32(73))
    assert_equal(counters.values[WAVE_COUNTER.NEXT], UInt32(0))
    assert_equal(counters.values[WAVE_COUNTER.LAMBERTIAN], UInt32(0))
    assert_equal(counters.values[WAVE_COUNTER.METAL], UInt32(0))
    assert_equal(counters.values[WAVE_COUNTER.DIELECTRIC], UInt32(0))

    var dispatch = WavefrontDispatchState(
        UInt32(8192), UInt32(512), UInt32(0), UInt32(0)
    )
    assert_equal(dispatch.rng_stage(), UInt32(1))
    dispatch.advance_bounce()
    assert_equal(dispatch.bounce, UInt32(1))
    assert_equal(dispatch.active_slot, UInt32(1))
    assert_equal(dispatch.rng_stage(), UInt32(2))
    dispatch.advance_bounce()
    assert_equal(dispatch.active_slot, UInt32(0))


def test_cpu_aos_boundary_adapters() raises:
    var paths = List[WavePath]()
    paths.append(_path(UInt32(31)))
    paths.append(_path(UInt32(47)))
    var packed_paths = pack_wave_paths(paths, capacity=5)
    assert_equal(packed_paths.capacity, 5)
    var restored_paths = unpack_wave_paths(packed_paths)
    assert_equal(len(restored_paths), 2)
    assert_equal(restored_paths[0].path_id, UInt32(31))
    assert_equal(restored_paths[1].path_id, UInt32(47))

    var hit = SurfaceHit(
        Vec3f32[Frame.WORLD](1.0, 0.0, 0.0),
        SurfaceId(MAT.METAL, UInt32(4)),
        2.25,
        False,
        True,
    )
    var works = List[WaveShade]()
    works.append(WaveShade(UInt32(1), hit.copy()))
    var packed_works = pack_wave_shades(works, capacity=3)
    assert_equal(packed_works.capacity, 3)
    var restored_works = unpack_wave_shades(packed_works)
    assert_equal(len(restored_works), 1)
    assert_equal(restored_works[0].path_idx, UInt32(1))
    assert_equal(restored_works[0].hit.surface.value, hit.surface.value)
    assert_false(restored_works[0].hit.front_face)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
