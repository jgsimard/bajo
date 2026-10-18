"""Packet input gathers preserve active values and ignore invalid masked IDs."""
from std.memory import bitcast
from std.testing import TestSuite, assert_equal, assert_almost_equal
from bajo.core import Vec3
from bajo.rt.media import SceneMedia, medium_transmittance
from bajo.rt.render_types import Color
from bajo.rt.material_types import SurfaceStore, MaterialKind
from bajo.rt.cpu.wavefront.packet import _MaterialParameters


def _media[width: SIMDLength]() raises:
    var media = SceneMedia()
    _ = media.add(Color(0, 0.2, 0.5))
    _ = media.add(Color(0.7, 0, 0.01))
    var infinity = bitcast[.float32](UInt32(0x7F800000))
    var nan = bitcast[.float32](UInt32(0x7FC00000))
    for pattern in range(5):
        var ids = SIMD[.uint32, width](0)
        var distances = SIMD[.float32, width](0)
        var active = SIMD[.bool, width](fill=True)
        for lane in range(Int(width)):
            ids[lane] = UInt32((lane + pattern) % 3)
            distances[lane] = infinity if lane % 3 == 0 else Float32(lane) * 0.7
            active[lane] = pattern != 0 and (
                pattern == 1 or (lane + pattern) % 3 == 0
            )
            if not active[lane]:
                ids[lane] = UInt32.MAX
                distances[lane] = nan
        var actual = media.transmittance(ids, distances, active)
        for lane in range(Int(width)):
            var expected = Color(1)
            if active[lane]:
                expected = medium_transmittance(
                    media.absorption[Int(ids[lane])], distances[lane]
                )
            comptime for axis in range(3):
                assert_almost_equal(
                    actual[axis][lane], expected[axis][0], atol=2e-6
                )
    # Independent infinite-distance expectations for channels with zero/nonzero
    # absorption, rather than only comparing two instantiations of the helper.
    var infinite = medium_transmittance(
        Vec3[.float32, .WORLD, width](0, 0.2, 0),
        SIMD[.float32, width](infinity),
    )
    for lane in range(Int(width)):
        assert_equal(infinite.x[lane], Float32(1))
        assert_equal(infinite.y[lane], Float32(0))
        assert_equal(infinite.z[lane], Float32(1))


def test_medium_packets_mixed_vacuum_infinity_and_invalid_inactive_ids() raises:
    _media[1]()
    _media[4]()
    _media[8]()
    _media[16]()


def _parameters[kind: MaterialKind, width: SIMDLength]() raises:
    var surfaces = SurfaceStore()
    for i in range(7):
        _ = surfaces.add_metal(Color(0.5), Float32(i) / 7)
        _ = surfaces.add_dielectric(1 + Float32(i) / 7)
    for pattern in range(4):
        var ids = SIMD[.uint32, width](0)
        var active = SIMD[.bool, width](fill=False)
        for lane in range(Int(width)):
            active[lane] = pattern != 0 and (
                pattern == 1 or (lane + pattern) % 3 == 0
            )
            ids[lane] = UInt32((lane * 3 + pattern) % 7) if active[
                lane
            ] else UInt32.MAX
        var material = _MaterialParameters[width]()
        material.gather[kind](surfaces, ids, active)
        for lane in range(Int(width)):
            var expected = Float32(1)
            if active[lane]:
                comptime if kind == .METAL:
                    expected = Float32(Int(ids[lane])) / 7
                else:
                    expected = 1 + Float32(Int(ids[lane])) / 7
            assert_equal(material.parameter[lane], expected)
    # An entirely empty material table is valid when no lane requests it.
    var empty = SurfaceStore()
    var material = _MaterialParameters[width]()
    material.gather[kind](
        empty, SIMD[.uint32, width](UInt32.MAX), SIMD[.bool, width](fill=False)
    )


def test_material_parameter_packets_use_record_stride_and_masks() raises:
    comptime for width in [1, 4, 8, 16]:
        _parameters[.METAL, width]()
        _parameters[.DIELECTRIC, width]()


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
