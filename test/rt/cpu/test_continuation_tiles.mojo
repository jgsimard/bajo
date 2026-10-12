"""Shared continuation values across CPU SIMD and GPU scalar layouts."""

from std.testing import TestSuite, assert_equal, assert_almost_equal
from bajo.core import Point3, Vec3
from bajo.rt.continuation import finish_continuation, continuation_medium
from bajo.rt.common import russian_roulette
from bajo.rt.path_values import PathContinuation, PathFloatField
from bajo.rt.path_store import store_path_values
from bajo.rt.tensor_layout import rt_field_tile
from bajo.rt.render_types import Color, RenderSettings, SamplingConfig
from bajo.rt.shading_types import BsdfSample
from bajo.rt.wavefront_queue import PacketPathQueue


def _values[length: SIMDLength](base: Int) -> PathContinuation[length]:
    var ids = SIMD[.uint32, length](0)
    var ox = SIMD[.float32, length](0.0)
    var medium = SIMD[.uint32, length](0)
    var delta = SIMD[.bool, length](fill=False)
    for lane in range(Int(length)):
        ids[lane] = UInt32(base + lane)
        ox[lane] = Float32(base + lane)
        medium[lane] = UInt32(0xF0000000) | UInt32(base + lane)
        delta[lane] = (base + lane) % 3 == 0
    return PathContinuation[length](
        ids,
        Point3[.float32, .WORLD, length](ox, ox + 1.0, ox + 2.0),
        Vec3[.float32, .WORLD, length](ox + 3.0, ox + 4.0, ox + 5.0),
        Vec3[.float32, .WORLD, length](ox + 6.0, ox + 7.0, ox + 8.0),
        ox + 9.0,
        delta,
        medium,
    )


def _append_cases[length: SIMDLength]() raises:
    var prefixes: List[Int] = [0, 1, Int(length) - 1]
    for prefix in prefixes:
        var queue = PacketPathQueue[length](0)
        var expected = List[Int]()
        for iteration in range(19):
            var source = _values[length](100 * iteration)
            var tail = prefix if iteration == 0 else Int(length)
            var mask = SIMD[.bool, length](fill=True)
            if iteration % 3 == 1:
                for lane in range(Int(length)):
                    mask[lane] = lane % 2 == 0
            queue.append_continuation(source, mask, tail)
            for lane in range(tail):
                if mask[lane]:
                    expected.append(100 * iteration + lane)
            assert_equal(len(queue), len(expected))
            for index in range(len(expected)):
                ref packet = queue.packets[index / length]
                var lane = index % length
                var id = expected[index]
                assert_equal(packet.path_ids[lane], UInt32(id))
                assert_equal(
                    packet.medium_ids[lane], UInt32(0xF0000000) | UInt32(id)
                )
                assert_equal(packet.deltas[lane], id % 3 == 0)
                var expected_values = _values[1](id)
                var tile = packet.tile()
                comptime for field in range(PathFloatField.VALUE_PLANES):
                    assert_equal(
                        tile[field, lane],
                        expected_values.float_field[field]()[0],
                    )


def test_continuation_append_preserves_order_across_partial_packets() raises:
    _append_cases[1]()
    _append_cases[4]()
    _append_cases[8]()
    _append_cases[16]()
    _append_cases[64]()
    _append_cases[128]()


def test_path_store_static_and_dynamic_extents_match() raises:
    var values = _values[16](20)
    var cpu = List[Float32](length=12 * 16, fill=-999.0)
    var gpu = List[Float32](length=12 * 16, fill=-999.0)
    var a = rt_field_tile[.float32, 12](cpu.unsafe_ptr(), Int(16))
    from layout import Idx

    var b = rt_field_tile[.float32, 12](gpu.unsafe_ptr(), Idx[16])
    store_path_values(a, values, 0)
    store_path_values(b, values, 0)
    for i in range(len(cpu)):
        assert_equal(cpu[i], gpu[i])


def test_shared_roulette_matches_scalar_streams() raises:
    var sampling = SamplingConfig.from_settings(
        RenderSettings(4, 4, 4, UInt64(2026), 8)
    )
    var input = _values[16](0)
    var incident = Vec3[.float32, .WORLD, 16](0.0, 0.0, -1.0)
    var weight = Vec3[.float32, .WORLD, 16](0.1, 0.2, 0.3)
    var sampled = BsdfSample[16](
        incident,
        weight,
        0.25,
        SIMD[.bool, 16](fill=False),
        SIMD[.bool, 16](fill=True),
    )
    var active = SIMD[.bool, 16](fill=True)
    active[15] = False
    for depth in range(1, 10):
        var result = finish_continuation(
            input.path_ids,
            input.origin,
            incident,
            Float32(2.0),
            Vec3[.float32, .WORLD, 16](1.0),
            input.medium_ids,
            sampled,
            active,
            sampling,
            UInt32(depth),
        )
        for lane in range(15):
            var reference = russian_roulette(
                sampling,
                input.path_ids[lane],
                UInt32(depth),
                Color(0.1, 0.2, 0.3),
            )
            assert_equal(result.active[lane], reference.survived)
            if reference.survived:
                assert_almost_equal(
                    result.values.throughput.x[lane], reference.throughput.x[0]
                )
                assert_almost_equal(
                    result.values.throughput.y[lane], reference.throughput.y[0]
                )
                assert_almost_equal(
                    result.values.throughput.z[lane], reference.throughput.z[0]
                )
        assert_equal(result.active[15], False)


def test_medium_transition_is_width_generic() raises:
    var direction = Vec3[.float32, .WORLD, 4](
        0.0, 0.0, SIMD[.float32, 4](-1.0, -1.0, 1.0, -1.0)
    )
    var result = continuation_medium(
        SIMD[.uint32, 4](7),
        SIMD[.uint32, 4](
            UInt32(2) | (UInt32(3) << 16),
            UInt32(2) | (UInt32(3) << 16),
            UInt32(2) | (UInt32(3) << 16),
            0,
        ),
        SIMD[.bool, 4](True, False, True, True),
        direction,
        Vec3[.float32, .WORLD, 4](0.0, 0.0, 1.0),
    )
    assert_equal(result, SIMD[.uint32, 4](2, 3, 7, 7))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
