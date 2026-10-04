"""Run with: pixi run python test/viewer/test_viewer_extensions.py"""
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from viewer_extensions import ExtensionLoader, ViewerRenderer


class ExtensionCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("bajo_viewer.mojo", "bajo_viewer_prepare.mojo"):
            (self.root / name).write_text("source")
        self.loader = ExtensionLoader(self.root, "sm_120")
        self.builds = []
        self.fail = False
        self.version = "Mojo test compiler"
        def run(command, **kwargs):
            if command[-1] == "--version":
                return SimpleNamespace(stdout=self.version)
            self.builds.append(command)
            output = Path(command[command.index("-o") + 1])
            output.write_bytes(b"partial" if self.fail else b"compiled")
            return SimpleNamespace(returncode=int(self.fail), stdout="", stderr="failed")
        def spec(name, path):
            loader = SimpleNamespace(exec_module=lambda module: None)
            return SimpleNamespace(loader=loader)
        for target, replacement in (
            ("viewer_extensions.shutil.which", lambda _: "/mojo"),
            ("viewer_extensions.subprocess.run", run),
            ("viewer_extensions.importlib.util.spec_from_file_location", spec),
            ("viewer_extensions.importlib.util.module_from_spec", lambda _: SimpleNamespace()),
        ):
            mocked = patch(target, replacement)
            mocked.start()
            self.addCleanup(mocked.stop)
        for name in ("bajo_viewer", "bajo_viewer_prepare"):
            old = sys.modules.get(name)
            self.addCleanup(self.restore_module, name, old)

    @staticmethod
    def restore_module(name, old):
        if old is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old

    def test_cpu_switches_reuse_the_independent_half(self):
        first = self.loader.load(0, 0, 0, 0)
        integrator = self.loader.load(0, 2, 0, 0)
        traversal = self.loader.load(0, 2, 2, 0)
        builder = self.loader.load(0, 2, 2, 1)
        self.assertIs(first.preparation, integrator.preparation)
        self.assertIs(integrator.preparation, traversal.preparation)
        self.assertIs(traversal.rendering, builder.rendering)
        self.assertIsNot(traversal.preparation, builder.preparation)
        self.assertEqual(len(self.builds), 5)
        self.loader.gpu_arch = "sm_90"
        self.assertEqual(self.loader.load(0, 2, 2, 1), builder)
        self.assertEqual(len(self.builds), 5)

    def test_gpu_ignores_cpu_policies(self):
        first = self.loader.load(1, 0, 0, 0)
        policy = self.loader.load(1, 0, 2, 3)
        integrator = self.loader.load(1, 1, 2, 3)
        self.assertEqual(first, policy)
        self.assertIs(first.preparation, integrator.preparation)
        self.assertEqual(len(self.builds), 3)
        self.loader.gpu_arch = "sm_90"
        other_arch = self.loader.load(1, 1, 2, 3)
        self.assertIsNot(other_arch.preparation, integrator.preparation)
        self.assertEqual(len(self.builds), 5)

    def test_compiler_and_source_changes_rebuild_both_halves(self):
        first = self.loader.load(0, 0, 0, 0)
        self.version += " updated"
        second = self.loader.load(0, 0, 0, 0)
        self.assertIsNot(first.preparation, second.preparation)
        (self.root / "bajo_viewer.mojo").write_text("changed")
        third = self.loader.load(0, 0, 0, 0)
        self.assertIsNot(second.preparation, third.preparation)
        self.assertEqual(len(self.builds), 6)

    def test_failed_build_is_never_cached(self):
        self.fail = True
        with self.assertRaisesRegex(RuntimeError, "failed"):
            self.loader.load(0, 0, 0, 0)
        self.assertEqual(list((self.root / "__mojocache__").iterdir()), [])
        self.fail = False
        self.loader.load(0, 0, 0, 0)
        self.assertEqual(len(self.builds), 3)


class SceneOwnershipTests(unittest.TestCase):
    def setUp(self):
        from viewer import Viewer
        self.viewer = Viewer.__new__(Viewer)
        self.viewer.cpu_state = None
        self.viewer.gpu_state = None
        self.work = SimpleNamespace(
            backend_index=0, scene_index=1, scene_path="", traversal_index=0
        )
        self.owner = Mock()
        self.owner.create_cpu_state.return_value = [123, 1.0, 2.0, "CPU | S1 T0 I0"]
        self.owner.create_gpu_state.return_value = [456, 1, 1.0, 2.0, "GPU"]
        self.render = Mock()
        self.pixels = np.zeros((2, 3, 3), dtype=np.float32)
        self.render.render_cpu_state.return_value = [1.0, 0.0, 1.0, self.pixels]
        self.render.render_gpu_state.return_value = [1.0, 0.0, 1.0, self.pixels]
        self.extension = ViewerRenderer(self.owner, self.render)

    def test_integrator_and_traversal_switches_keep_the_scene(self):
        self.viewer._render_batch(self.extension, self.work, {})
        self.work.traversal_index = 2
        next_render = Mock()
        next_render.render_cpu_state.return_value = [1.0, 0.0, 1.0, self.pixels]
        stats, pixels = self.viewer._render_batch(
            ViewerRenderer(self.owner, next_render), self.work, {}
        )
        self.owner.create_cpu_state.assert_called_once()
        self.owner.destroy_cpu_state.assert_not_called()
        next_render.render_cpu_state.assert_called_once_with(123, {})
        self.assertIn("adaptive-16-8-4-scalar", stats.bvh_stats)
        self.assertIs(pixels, self.pixels)
        self.viewer._destroy_cpu_state()
        self.viewer._destroy_cpu_state()
        self.owner.destroy_cpu_state.assert_called_once_with(123)

    def test_builder_and_backend_switches_free_with_the_original_owner(self):
        _, pixels = self.viewer._render_batch(self.extension, self.work, {})
        self.assertIs(pixels, self.pixels)
        next_owner = Mock()
        next_owner.create_cpu_state.return_value = [789, 1.0, 2.0, "CPU"]
        self.viewer._render_batch(
            ViewerRenderer(next_owner, self.render), self.work, {}
        )
        self.owner.destroy_cpu_state.assert_called_once_with(123)
        self.work.backend_index = 1
        self.viewer._render_batch(self.extension, self.work, {})
        next_owner.destroy_cpu_state.assert_called_once_with(789)
        self.viewer._render_batch(
            ViewerRenderer(self.owner, self.render), self.work, {}
        )
        self.owner.create_gpu_state.assert_called_once()
        self.viewer._destroy_gpu_state()
        self.viewer._destroy_gpu_state()
        self.owner.destroy_gpu_state.assert_called_once_with(456, 1)


class FrameDisplayTests(unittest.TestCase):
    def setUp(self):
        from viewer import Viewer
        from viewer_config import Camera, RenderOptions, RenderSnapshot, RenderStats

        self.viewer = Viewer.__new__(Viewer)
        self.viewer.closed = False
        self.viewer.rendering = True
        self.viewer.lock = threading.Lock()
        self.viewer.render_generation = 0
        self.viewer.accumulated_spp = 0
        self.viewer.completed_batches = 0
        self.viewer.linear_sum = None
        self.viewer.dragging = False
        self.viewer.pressed = set()
        self.viewer.renderer = Mock()
        self.viewer.renderer_config = (0, 0, 0, 0)
        self.viewer.root = SimpleNamespace(after=lambda delay, callback: callback())
        self.viewer.show_frame = Mock()
        self.options = RenderOptions(width=2, height=1, batches=2, max_samples=3)
        self.stats = RenderStats(1.0, 0.0, 0.0, 0.0, 1.0, "CPU")
        self.work = RenderSnapshot(
            camera=Camera(), options=self.options, generation=0,
            batch_spp=2, sample_offset=0, preview=False,
            integrator_index=0, backend_index=0, traversal_index=0,
            build_index=0, sampler_index=0, scene_index=1, scene_path="",
        )

    def test_progressive_frames_keep_sample_weighting(self):
        from dataclasses import replace

        self.viewer.snapshot = Mock(side_effect=[
            self.work, replace(self.work, batch_spp=1, sample_offset=2),
        ])
        self.viewer._render_batch = Mock(side_effect=[
            (self.stats, np.full((1, 2, 3), 0.25, dtype=np.float32)),
            (self.stats, np.full((1, 2, 3), 1.0, dtype=np.float32)),
        ])
        self.viewer.render_worker()
        self.assertEqual(self.viewer.accumulated_spp, 3)
        self.assertEqual(self.viewer.completed_batches, 2)
        self.assertFalse(self.viewer.rendering)
        self.assertEqual(self.viewer.show_frame.call_count, 2)
        np.testing.assert_array_equal(self.viewer.linear_sum, np.full((1, 2, 3), 1.5))
        image = self.viewer.show_frame.call_args.args[0]
        np.testing.assert_array_equal(np.asarray(image), np.full((1, 2, 3), 181))
        for call in self.viewer._render_batch.call_args_list:
            self.assertNotIn("output", call.args[2])

    def test_preview_displays_without_accumulation(self):
        from dataclasses import replace

        self.viewer.snapshot = Mock(return_value=replace(self.work, preview=True))
        self.viewer._render_batch = Mock(return_value=(
            self.stats, np.full((1, 2, 3), 0.25, dtype=np.float32),
        ))
        self.viewer.render_worker()
        self.assertEqual(self.viewer.accumulated_spp, 0)
        self.assertIsNone(self.viewer.linear_sum)
        self.assertFalse(self.viewer.rendering)
        image = self.viewer.show_frame.call_args.args[0]
        np.testing.assert_array_equal(np.asarray(image), np.full((1, 2, 3), 128))


if __name__ == "__main__":
    unittest.main()
