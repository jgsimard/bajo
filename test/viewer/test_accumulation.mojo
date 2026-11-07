"""Progressive accumulation preserves scalar pixels at SIMD boundaries."""

from std.memory import bitcast
from std.testing import TestSuite, assert_equal, assert_false, assert_true

from bajo.core.color import Color, color_to_byte
from bajo.rt.render_types import RenderResult, RenderSettings, RenderTimings
from bajo.viewer.accumulation import FrameAccumulator
from bajo.viewer.worker import CompletedFrame
from bajo.rt.gpu.preview import GpuRtPreviewResult


def _frame(
    count: Int, samples: Int, generation: UInt64, phase: Int
) -> CompletedFrame:
    var pixels = List[Color](capacity=count)
    for index in range(count):
        pixels.append(
            Color(
                Float32((index + 17 * phase) % 251 - 13) / 97.0,
                Float32((index + 11 * phase) % 97) / 97.0,
                Float32((index + 7 * phase) % 43) / 41.0,
            )
        )
    if count:
        pixels[0] = Color(
            bitcast[.float32](UInt32(0x80000000)),
            bitcast[.float32](UInt32(1)),
            bitcast[.float32](UInt32(0x80000001)),
        )
    var settings = RenderSettings(max(1, count), 1, samples, UInt64(2026))
    return CompletedFrame(
        RenderResult(
            pixels^, RenderTimings(0, 0, 0, count, count * samples, 0)
        ),
        generation,
        settings,
    )


def _check_batch(
    mut actual: FrameAccumulator,
    mut expected: List[Color],
    mut samples: Int,
    frame: CompletedFrame,
) raises:
    if len(expected) != len(frame.pixels):
        expected = List[Color](length=len(frame.pixels), fill=Color(0))
    for index in range(len(expected)):
        expected[index] += frame.pixels[index] * Float32(frame.samples)
    samples += frame.samples
    assert_true(actual.add(frame))
    assert_equal(actual.samples, samples)
    assert_equal(len(actual.linear), len(expected))
    assert_equal(len(actual.rgba), 4 * len(expected))
    for index in range(len(expected)):
        var color = expected[index] / Float32(samples)
        comptime for channel in range(3):
            assert_equal(
                bitcast[.uint32](actual.linear[index][channel]),
                bitcast[.uint32](expected[index][channel]),
            )
            assert_equal(
                actual.rgba[4 * index + channel], color_to_byte(color[channel])
            )
        assert_equal(actual.rgba[4 * index + 3], UInt8(255))


def test_weighted_batches_match_scalar_for_small_images_and_simd_tails() raises:
    for count in range(41):
        var actual = FrameAccumulator()
        var expected = List[Color]()
        var samples = 0
        var phase = 0
        for batch in [1, 3, 2, 9, 4]:
            _check_batch(
                actual, expected, samples, _frame(count, batch, 0, phase)
            )
            phase += 1
            assert_equal(actual.batches, phase)


def test_accumulation_resizes_and_reuses_buffers_with_the_same_generation() raises:
    var actual = FrameAccumulator()
    var expected = List[Color]()
    var samples = 0
    for count in [17, 2, 37, 0, 6, 5, 9]:
        _check_batch(actual, expected, samples, _frame(count, 3, 0, count))


def test_reset_rejects_stale_frames_without_changing_accumulation() raises:
    var actual = FrameAccumulator()
    var old_frame = _frame(17, 2, 0, 1)
    assert_true(actual.add(old_frame))
    actual.reset()
    assert_equal(actual.generation, UInt64(1))
    assert_false(actual.add(old_frame))
    assert_equal(actual.samples, 0)
    assert_equal(actual.batches, 0)
    assert_equal(len(actual.linear), 0)
    assert_equal(len(actual.rgba), 0)
    var expected = List[Color]()
    var samples = 0
    _check_batch(actual, expected, samples, _frame(17, 3, 1, 2))
    var linear = actual.linear.copy()
    var rgba = actual.rgba.copy()
    assert_false(actual.add(old_frame))
    assert_equal(actual.samples, 3)
    assert_equal(actual.batches, 1)
    for index in range(len(rgba)):
        assert_equal(actual.rgba[index], rgba[index])
    for index in range(len(linear)):
        comptime for channel in range(3):
            assert_equal(
                bitcast[.uint32](actual.linear[index][channel]),
                bitcast[.uint32](linear[index][channel]),
            )


def test_gpu_preview_frames_publish_counts_and_ignore_stale_generations() raises:
    var actual = FrameAccumulator()
    var settings = RenderSettings(2, 1, 3, 2026)
    var bytes = List[UInt8](length=8, fill=137)
    var frame = CompletedFrame.from_preview(
        GpuRtPreviewResult(bytes.copy(), RenderTimings(0, 0, 0, 2, 6, 1), 7, 3),
        0,
        settings,
    )
    assert_true(actual.add(frame))
    assert_equal(actual.samples, 7)
    assert_equal(actual.batches, 3)
    assert_equal(len(actual.linear), 0)
    for i in range(8):
        assert_equal(actual.rgba[i], bytes[i])
    actual.reset()
    assert_false(actual.add(frame))
    assert_equal(actual.samples, 0)
    var next = CompletedFrame.from_preview(
        GpuRtPreviewResult(bytes.copy(), RenderTimings(0, 0, 0, 2, 6, 1), 3, 1),
        actual.generation,
        settings,
    )
    assert_true(actual.add(next))
    assert_equal(actual.samples, 3)
    assert_equal(actual.batches, 1)


def test_owned_preview_handoff_preserves_address_metadata_and_stale_frame() raises:
    var actual = FrameAccumulator()
    var settings = RenderSettings(17, 1, 3, 2026)
    var frame = CompletedFrame.from_preview(
        GpuRtPreviewResult(
            List[UInt8](length=68, fill=137),
            RenderTimings(0, 0, 0, 17, 51, 1),
            7,
            3,
        ),
        0,
        settings,
    )
    var pointer = Int(frame.rgba.unsafe_ptr())
    assert_true(actual.take_frame(frame))
    assert_equal(Int(actual.rgba.unsafe_ptr()), pointer)
    assert_equal(len(frame.rgba), 0)
    assert_equal(frame.width, 17)
    assert_equal(actual.samples, 7)
    assert_equal(actual.batches, 3)
    actual.reset()
    var stale = CompletedFrame.from_preview(
        GpuRtPreviewResult(
            List[UInt8](length=68, fill=42),
            RenderTimings(0, 0, 0, 17, 51, 1),
            3,
            1,
        ),
        0,
        settings,
    )
    pointer = Int(stale.rgba.unsafe_ptr())
    assert_false(actual.take_frame(stale))
    assert_equal(Int(stale.rgba.unsafe_ptr()), pointer)
    assert_equal(len(stale.rgba), 68)
    assert_equal(actual.samples, 0)
    var cpu_frame = _frame(17, 3, actual.generation, 2)
    var expected = FrameAccumulator()
    expected.generation = actual.generation
    assert_true(expected.add(cpu_frame))
    assert_true(actual.take_frame(cpu_frame))
    assert_equal(actual.samples, expected.samples)
    for i in range(len(actual.rgba)):
        assert_equal(actual.rgba[i], expected.rgba[i])


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


def main() raises:
    run_tests()
