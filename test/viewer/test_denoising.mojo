"""Display-only reconstruction, asynchronous refresh, and stale presentations."""
from std.testing import TestSuite, assert_equal, assert_true, assert_false
from std.memory import bitcast
from bajo.viewer.core import ViewerScene
from bajo.viewer.scenes import load_viewer_scene
from bajo.viewer.worker import CpuRenderWorker
from bajo.viewer.accumulation import FrameAccumulator
from bajo.viewer.native import denoise_button_rect
from test.viewer.test_native import _wait


def test_worker_refresh_preserves_raw_samples_and_restores_bytes() raises:
    var scene = ViewerScene(load_viewer_scene(1))
    var camera = scene.state.camera
    var settings = scene.state.settings.copy()
    settings.image_width = 9
    settings.image_height = 7
    settings.samples_per_pixel = 2
    settings.sample_sequence_length = 3
    settings.max_depth = 2
    var worker = CpuRenderWorker(scene^)
    worker.submit_preview(camera, settings, 4, 2, 0, True, False, False, 0)
    _wait(worker)
    assert_equal(worker.state[].error, String(""))
    var first = worker.state[].result.take()
    var raw = first.rgba.copy()
    assert_equal(first.preview_samples, 2)
    assert_true(first.preparation_ns > 0)
    assert_true(first.batch_ns > 0)
    assert_true(first.postprocess_ns > 0)
    assert_true(first.work_ns >= first.batch_ns + first.postprocess_ns)
    var sums = worker.state[].cpu_preview.value().sums.copy()
    assert_false(Bool(worker.state[].cpu_preview.value().guides))
    var nodes = Int(
        worker.state[].scene.cpu_scene().triangle_bvh.value().nodes.unsafe_ptr()
    )
    worker.acknowledge()
    for revision in range(1, 5):
        worker.refresh_preview(
            camera, settings, 4, 2, 0, revision % 2 == 1, UInt64(revision)
        )
        _wait(worker)
        assert_equal(worker.state[].error, String(""))
        var frame = worker.state[].result.take()
        assert_equal(frame.samples, 0)
        assert_equal(frame.batch_ns, 0)
        assert_true(frame.postprocess_ns > 0)
        assert_true(frame.work_ns >= frame.postprocess_ns)
        assert_equal(frame.preview_samples, 2)
        assert_equal(frame.preview_batches, 1)
        assert_equal(frame.presentation_revision, UInt64(revision))
        if revision % 2 == 0:
            for i in range(len(raw)):
                assert_equal(frame.rgba[i], raw[i])
        for i in range(len(sums)):
            comptime for c in range(3):
                assert_equal(
                    bitcast[.uint32](
                        worker.state[].cpu_preview.value().sums[i][c]
                    ),
                    bitcast[.uint32](sums[i][c]),
                )
        assert_equal(
            Int(
                worker.state[]
                .scene.cpu_scene()
                .triangle_bvh.value()
                .nodes.unsafe_ptr()
            ),
            nodes,
        )
        worker.acknowledge()
    worker.state[].cpu_preview.value().guides.value().width = 0
    worker.refresh_preview(camera, settings, 4, 2, 0, True, 5)
    _wait(worker)
    assert_true(worker.state[].error != "")
    var recovered = worker.state[].result.take()
    assert_equal(recovered.preview_samples, 2)
    for i in range(len(raw)):
        assert_equal(recovered.rgba[i], raw[i])
    worker.state[].cpu_preview.value().guides.value().width = 9
    worker.acknowledge()
    settings.sample_offset = 2
    settings.samples_per_pixel = 1
    worker.submit_preview(camera, settings, 4, 2, 0, True, False, True, 5)
    _wait(worker)
    assert_equal(worker.state[].error, String(""))
    var last = worker.state[].result.take()
    assert_equal(last.preview_samples, 3)
    assert_equal(last.preview_batches, 2)
    worker.acknowledge()
    settings.color_mode = .SPECTRAL
    settings.sample_offset = 0
    worker.submit_preview(camera, settings, 6, 2, 0, True, False, True, 5)
    _wait(worker)
    assert_equal(worker.state[].error, String(""))
    assert_equal(worker.state[].result.value().preview_samples, 1)
    worker.acknowledge()
    # NORMALS is presented verbatim even when the preference remains enabled.
    worker.submit_preview(camera, settings, 5, 3, 0, True, False, True, 5)
    _wait(worker)
    assert_equal(worker.state[].error, String(""))
    assert_false(Bool(worker.state[].cpu_preview.value().guides))
    worker.acknowledge()


def test_stale_presentation_advances_counts_without_replacing_display() raises:
    var scene = ViewerScene(load_viewer_scene(1))
    var camera = scene.state.camera
    var settings = scene.state.settings.copy()
    settings.image_width = 3
    settings.image_height = 2
    settings.samples_per_pixel = 1
    var worker = CpuRenderWorker(scene^)
    worker.submit_preview(camera, settings, 0, 0, 0, True, False, False, 0)
    _wait(worker)
    assert_equal(worker.state[].error, String(""))
    var frame = worker.state[].result.take()
    var accumulation = FrameAccumulator()
    assert_false(accumulation.take_frame(frame, 1))
    assert_equal(accumulation.samples, 1)
    assert_equal(accumulation.batches, 1)
    assert_equal(len(accumulation.rgba), 0)
    worker.acknowledge()
    worker.refresh_preview(camera, settings, 0, 0, 0, True, 1)
    _wait(worker)
    assert_equal(worker.state[].error, String(""))
    frame = worker.state[].result.take()
    assert_true(accumulation.take_frame(frame, 1))
    assert_equal(accumulation.samples, 1)
    assert_equal(accumulation.batches, 1)
    assert_equal(len(accumulation.rgba), 24)
    worker.acknowledge()


def test_button_layout_stays_below_other_controls() raises:
    var narrow = denoise_button_rect(320)
    assert_equal(narrow[0], 10)
    assert_true(narrow[1] >= 140)
    var medium = denoise_button_rect(640)
    assert_true(medium[0] >= 260)
    assert_true(medium[0] + medium[2] < 640)
    var wide = denoise_button_rect(960)
    assert_true(wide[0] + wide[2] < 960 - 148)


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
