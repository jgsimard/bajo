"""Packet tile aliasing, stable compaction, tails and queue growth."""

from std.testing import TestSuite, assert_equal
from bajo.core import Vec3f32
from bajo.rt.material_types import SurfaceId
from bajo.rt.shading_types import SurfaceHit
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
