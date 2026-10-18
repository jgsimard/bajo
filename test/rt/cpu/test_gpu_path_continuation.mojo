"""Host-executable checks of the device-safe GPU continuation operation."""

from std.testing import TestSuite, assert_equal

from bajo.bvh.constants import f32_max
from bajo.core import Vec3f32
from bajo.rt.common import russian_roulette
from bajo.rt.gpu.path_shading import _continue_gpu_path
from bajo.rt.gpu.wavefront_contract import load_gpu_rt_path
from bajo.rt.material_types import Integrator
from bajo.rt.rays import RT_RAY_T_MIN
from bajo.rt.render_types import Color, RenderSettings, SamplingConfig
from bajo.rt.shading_types import BsdfSample
from bajo.rt.wavefront_contract import (
    DeviceWavePath,
    WAVE_COUNTER,
    WAVE_STATUS,
    WavePathFloatAbi,
)


comptime CAPACITY = 2
comptime UNWRITTEN_ID = UInt32(12345)
comptime UNWRITTEN_FIELD = Float32(-999.0)


def _test_continuation[
    integrator: Integrator
](
    ok: Bool = True,
    zero_weight: Bool = False,
    full: Bool = False,
    bounce: UInt32 = 0,
    delta: Bool = True,
) raises:
    var sampling = SamplingConfig.from_settings(RenderSettings(8, 1, 1, 42, 8))
    var path = DeviceWavePath(
        UInt32(7),
        1.0,
        2.0,
        3.0,
        0.1,
        1.0,
        0.0,
        0.0,
        100.0,
        1.0,
        2.0,
        0.5,
        0.5,
        False,
    )
    var weight = Color(0.0) if zero_weight else Color(0.5, 0.25, 1.0)
    var sampled = BsdfSample[1](
        Vec3f32[.WORLD](0.0, 1.0, 0.0),
        weight,
        0.125,
        delta,
        ok,
    )
    var ids = List[UInt32](length=CAPACITY, fill=UNWRITTEN_ID)
    var fields = List[Float32](
        length=CAPACITY * WavePathFloatAbi.PLANES, fill=UNWRITTEN_FIELD
    )
    var counters = List[UInt32](length=WAVE_COUNTER.COUNT, fill=0)
    counters[WAVE_COUNTER.NEXT] = UInt32(CAPACITY if full else 0)

    _continue_gpu_path[integrator](
        path,
        2.0,
        sampled,
        ids.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        fields.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        counters.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        CAPACITY,
        sampling,
        bounce,
    )

    var roulette = russian_roulette(
        sampling,
        path.path_id,
        bounce + UInt32(1),
        Color(1.0, 2.0, 0.5) * weight,
    )
    var appended = ok and roulette.survived
    assert_equal(
        Int(counters[WAVE_COUNTER.NEXT]),
        (CAPACITY if full else 0) + Int(appended),
    )
    assert_equal(
        counters[WAVE_COUNTER.STATUS],
        WAVE_STATUS.PATH_OVERFLOW if appended and full else WAVE_STATUS.OK,
    )
    if not appended or full:
        for value in ids:
            assert_equal(value, UNWRITTEN_ID)
        for value in fields:
            assert_equal(value, UNWRITTEN_FIELD)
        return

    var result = load_gpu_rt_path[integrator](
        ids.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin](),
        fields.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin](),
        CAPACITY,
        0,
    )
    assert_equal(result.path_id, path.path_id)
    assert_equal(result.ox, Float32(3.0))
    assert_equal(result.oy, Float32(2.0))
    assert_equal(result.oz, Float32(3.0))
    assert_equal(result.dx, Float32(0.0))
    assert_equal(result.dy, Float32(1.0))
    assert_equal(result.dz, Float32(0.0))
    assert_equal(result.t_min, RT_RAY_T_MIN)
    assert_equal(result.t_max, f32_max)
    assert_equal(result.tx, roulette.throughput.x)
    assert_equal(result.ty, roulette.throughput.y)
    assert_equal(result.tz, roulette.throughput.z)
    assert_equal(
        result.delta,
        delta if Integrator.uses_direct_lighting[integrator] else True,
    )
    assert_equal(result.bsdf_pdf, Float32(0.125 if integrator == .MIS else 0.0))
    assert_equal(ids[1], UNWRITTEN_ID)
    for plane in range(WavePathFloatAbi.PLANES):
        assert_equal(fields[plane * CAPACITY + 1], UNWRITTEN_FIELD)


def test_continuation_preserves_path_and_integrator_metadata() raises:
    _test_continuation[.PATH]()
    _test_continuation[.NEE]()
    _test_continuation[.MIS]()
    _test_continuation[.MIS](delta=False)
    _test_continuation[.NEE](delta=False)
    _test_continuation[.PATH](delta=False)


def test_rejected_bsdf_does_not_reserve_or_write() raises:
    _test_continuation[.MIS](ok=False)


def test_roulette_termination_does_not_reserve_or_write() raises:
    _test_continuation[.MIS](zero_weight=True, bounce=4)


def test_queue_overflow_reports_status_without_writing() raises:
    _test_continuation[.MIS](full=True)


def test_continuation_uses_roulette_throughput_at_late_bounces() raises:
    for bounce in range(4, 16):
        _test_continuation[.MIS](bounce=UInt32(bounce))


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
