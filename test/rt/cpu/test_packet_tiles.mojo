"""Packet tile aliasing, stable compaction, tails and queue growth."""

from std.testing import TestSuite, assert_equal, assert_almost_equal
from bajo.core import Vec3f32, Point3, Vec3
from bajo.rt.material_types import SurfaceId
from bajo.rt.shading_types import SurfaceHit, BsdfSample
from bajo.rt.continuation import finish_continuation, continuation_medium
from bajo.rt.common import russian_roulette
from bajo.rt.path_values import PathContinuation, PathFloatField
from bajo.rt.path_store import store_path_values
from bajo.rt.tensor_layout import rt_field_tile
from bajo.rt.render_types import Color, RenderSettings, SamplingConfig
from bajo.rt.wavefront_queue import (
    PathPacket,
    ShadePacket,
    PacketPathQueue,
    PacketShadeQueue,
)


def _packet[length: SIMDLength](base: Int) -> PathPacket[length]:
    var packet = PathPacket[length]()
    var tile = packet.tile()
    for lane in range(Int(length)):
        packet.path_ids[lane] = UInt32(0x80000000) | UInt32(base + lane)
        packet.medium_ids[lane] = UInt32(0xF0000000) | UInt32(base + lane)
        packet.deltas[lane] = (base + lane) % 3 == 0
        for field in range(PathPacket[length].FLOAT_FIELDS):
            tile[field, lane] = Float32(field * 10000 + base + lane)
    return packet^


def _check[
    length: SIMDLength
](queue: PacketPathQueue[length], expected: List[Int]) raises:
    assert_equal(len(queue), len(expected))
    assert_equal(len(queue.packets), (len(expected) + length - 1) / length)
    for index in range(len(expected)):
        ref packet = queue.packets[index / length]
        var lane = index % length
        var id = expected[index]
        assert_equal(packet.path_ids[lane], UInt32(0x80000000) | UInt32(id))
        assert_equal(packet.medium_ids[lane], UInt32(0xF0000000) | UInt32(id))
        assert_equal(packet.deltas[lane], id % 3 == 0)
        var tile = packet.tile()
        for field in range(PathPacket[length].FLOAT_FIELDS):
            assert_equal(tile[field, lane], Float32(field * 10000 + id))


def _masked_cases[length: SIMDLength]() raises:
    var prefixes: List[Int] = [0, 1, Int(length) - 1]
    var tails: List[Int] = [0, 1, Int(length) - 1, Int(length)]
    for bits in range(1 << Int(length)):
        for prefix in prefixes:
            for tail in tails:
                # Zero capacity deliberately forces growth during an append.
                var queue = PacketPathQueue[length](0)
                var expected = List[Int]()
                queue.append_packet(_packet[length](0), prefix)
                for lane in range(prefix):
                    expected.append(lane)
                var mask = SIMD[.bool, length](fill=False)
                for lane in range(Int(length)):
                    mask[lane] = (bits & (1 << lane)) != 0
                    if lane < tail and mask[lane]:
                        expected.append(100 + lane)
                queue.append_packet_masked(_packet[length](100), mask, tail)
                _check(queue, expected)
                # A following full packet can straddle the partially filled
                # destination; stable order must survive the extra append.
                queue.append_packet(_packet[length](200), Int(length))
                for lane in range(Int(length)):
                    expected.append(200 + lane)
                _check(queue, expected)
                queue.clear()
                assert_equal(len(queue), 0)
                queue.append_packet_masked(_packet[length](100), mask, tail)
                var reused = List[Int]()
                for lane in range(tail):
                    if mask[lane]:
                        reused.append(100 + lane)
                _check(queue, reused)


def test_packet_tile_aliases_named_fields_and_copies_own_storage() raises:
    var packet = _packet[4](20)
    var copy = packet.copy()
    var tile = packet.tile()
    tile[0, 2] = 3.5
    assert_equal(packet.ox[2], Float32(3.5))
    packet.bsdf_pdfs[3] = 7.5
    assert_equal(tile[11, 3], Float32(7.5))
    assert_equal(copy.ox[2], Float32(22.0))
    assert_equal(copy.bsdf_pdfs[3], Float32(110023.0))
    var shade = ShadePacket[4]()
    shade.uv_u[1] = 0.25
    var shade_tile = shade.tile()
    assert_equal(shade_tile[12, 1], Float32(0.25))
    shade_tile[17, 3] = -2.5
    assert_equal(shade.gnz[3], Float32(-2.5))


def test_masked_tile_append_all_four_lane_masks_and_tails() raises:
    _masked_cases[4]()


def test_masked_tile_append_all_eight_lane_masks_and_tails() raises:
    _masked_cases[8]()


def _wide_cases[length: SIMDLength]() raises:
    var queue = PacketPathQueue[length](0)
    var expected = List[Int]()
    for iteration in range(19):
        var mask = SIMD[.bool, length](fill=False)
        var tail = (iteration * 7) % (Int(length) + 1)
        var base = iteration * Int(length)
        for lane in range(Int(length)):
            mask[lane] = (lane + iteration) % 3 != 0
            if lane < tail and mask[lane]:
                expected.append(base + lane)
        queue.append_packet_masked(_packet[length](base), mask, tail)
        _check(queue, expected)


def test_wide_masked_tile_append_and_generic_fallback() raises:
    _wide_cases[1]()
    _wide_cases[16]()
    _wide_cases[32]()
    _wide_cases[64]()
    _wide_cases[128]()


def test_shade_tile_append_preserves_hits_and_path_metadata() raises:
    var queue = PacketShadeQueue[8](0)
    for index in range(19):
        var packet = _packet[8](index * 8)
        var hit = SurfaceHit(
            Vec3f32[.WORLD](1.0, 2.0, 3.0),
            SurfaceId(.METAL, UInt32(index)),
            Float32(index) + 0.5,
            index % 2 == 0,
            True,
        )
        hit.uv_u = 0.25
        hit.uv_v = 0.75
        hit.geometric_normal = Vec3f32[.WORLD](4.0, 5.0, 6.0)
        hit.medium_boundary = UInt32(index + 1)
        queue.append(packet, index % 8, hit)
    assert_equal(len(queue), 19)
    for index in range(19):
        ref shade = queue.packets[index / 8]
        var lane = index % 8
        var id = index * 8 + index % 8
        assert_equal(shade.path_ids[lane], UInt32(0x80000000) | UInt32(id))
        assert_equal(shade.medium_ids[lane], UInt32(0xF0000000) | UInt32(id))
        assert_equal(shade.ox[lane], Float32(id))
        assert_equal(shade.dx[lane], Float32(40000 + id))
        assert_equal(shade.tx[lane], Float32(80000 + id))
        assert_equal(shade.nz[lane], Float32(3.0))
        assert_equal(shade.gnz[lane], Float32(6.0))
        assert_equal(shade.hit_t[lane], Float32(index) + 0.5)
        assert_equal(shade.uv_u[lane], Float32(0.25))
        assert_equal(shade.uv_v[lane], Float32(0.75))
        assert_equal(shade.surface_indices[lane], UInt32(index))
        assert_equal(shade.medium_boundaries[lane], UInt32(index + 1))
        assert_equal(shade.front_faces[lane], index % 2 == 0)


def _shade_masked_cases[length: SIMDLength]() raises:
    var queue = PacketShadeQueue[length](0)
    var reference = PacketShadeQueue[length](0)
    for iteration in range(19):
        var paths = _packet[length](iteration * Int(length))
        var hits = SurfaceHit[length](1000.0)
        var selected = SIMD[.bool, length](fill=False)
        var tail = (iteration * 7) % (Int(length) + 1)
        for lane in range(Int(length)):
            var value = Float32(iteration * Int(length) + lane)
            hits.normal.x[lane] = value + 1.0
            hits.normal.y[lane] = value + 2.0
            hits.normal.z[lane] = value + 3.0
            hits.geometric_normal.x[lane] = value + 4.0
            hits.geometric_normal.y[lane] = value + 5.0
            hits.geometric_normal.z[lane] = value + 6.0
            hits.t[lane] = value + 7.0
            hits.uv_u[lane] = value + 8.0
            hits.uv_v[lane] = value + 9.0
            hits.surface.value[lane] = SurfaceId.pack_raw(
                .METAL, UInt32(lane + iteration)
            )
            hits.medium_boundary[lane] = UInt32(0xF0000000) | UInt32(
                lane + iteration
            )
            hits.front_face[lane] = lane % 2 == 0
            hits.hit[lane] = True
            selected[lane] = (lane + iteration) % 3 != 0
            if lane < tail and selected[lane]:
                reference.append(paths, lane, hits.get(lane))
        var batch = ShadePacket[length]()
        batch.set_paths(paths)
        batch.set_hits(hits)
        queue.append_packet_masked(batch, selected, tail)
        assert_equal(len(queue), len(reference))
        for index in range(len(queue)):
            ref actual = queue.packets[index / length]
            ref expected = reference.packets[index / length]
            var lane = index % length
            assert_equal(actual.path_ids[lane], expected.path_ids[lane])
            assert_equal(actual.medium_ids[lane], expected.medium_ids[lane])
            assert_equal(
                actual.medium_boundaries[lane], expected.medium_boundaries[lane]
            )
            assert_equal(
                actual.surface_indices[lane], expected.surface_indices[lane]
            )
            assert_equal(actual.front_faces[lane], expected.front_faces[lane])
            var a = actual.tile()
            var b = expected.tile()
            for field in range(ShadePacket[length].FLOAT_FIELDS):
                assert_equal(a[field, lane], b[field, lane])


def test_masked_shade_tiles_match_scalar_append() raises:
    _shade_masked_cases[1]()
    _shade_masked_cases[4]()
    _shade_masked_cases[8]()
    _shade_masked_cases[16]()
    _shade_masked_cases[64]()
    _shade_masked_cases[128]()


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
