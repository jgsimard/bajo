"""Opt-in real OpenGL checks for native viewer input and scene loading."""

import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


@unittest.skipUnless(
    os.environ.get("BAJO_TEST_VIEWER_GRAPHICS") == "1",
    "set BAJO_TEST_VIEWER_GRAPHICS=1 with a working display/OpenGL context",
)
class ViewerGraphicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        cls.raylib = cls.root / ".pixi/raylib"
        cls.temp = tempfile.TemporaryDirectory(prefix="bajo scene tests ")
        cls.addClassCleanup(cls.temp.cleanup)
        directory = Path(cls.temp.name)
        cls.shim = directory / "scene_input.so"
        subprocess.run(
            [
                "cc", "-shared", "-fPIC",
                str(Path(__file__).with_name("raylib_scene_input.c")),
                f"-I{cls.raylib / 'include'}", f"-L{cls.raylib / 'lib'}",
                "-Wl,--no-as-needed", "-lraylib",
                f"-Wl,-rpath,{cls.raylib / 'lib'}", "-ldl", "-o", str(cls.shim),
            ],
            check=True, capture_output=True, text=True,
        )
        cls.scene = directory / "valid" / "scene with spaces.pbrt-v4"
        cls.scene.parent.mkdir()
        cls.scene.write_text(
            'LookAt 0 0 4  0 0 0  0 1 0\n'
            'Camera "perspective" "float fov" [45]\n'
            'Film "rgb" "integer xresolution" [24] "integer yresolution" [16]\n'
            'Sampler "halton" "integer pixelsamples" [3]\n'
            'WorldBegin\nShape "sphere"\nWorldEnd\n'
        )
        cls.invalid = directory / "invalid.pbrt"
        cls.invalid.write_text('Include "missing-scene-asset.pbrt"\n')

    def run_viewer(self, *args, input_mode="cycle", path="", check=True, timeout=240):
        return subprocess.run(
            [
                "pixi", "run", "viewer", "--scene", "cornell", "--hidden",
                "--raylib-library", str(self.shim), *args,
            ],
            cwd=self.root,
            env=dict(
                os.environ,
                BAJO_TEST_RAYLIB=str(self.raylib / "lib/libraylib.so"),
                BAJO_TEST_SCENE_INPUT=input_mode,
                BAJO_TEST_SCENE_PATH=str(path),
            ),
            check=check, capture_output=True, text=True, timeout=timeout,
        )

    def launch(self, mode, path="", trace=False):
        render_args = (
            ["--mode", "trace", "--width", "32", "--height", "24", "--max-spp", "2"]
            if trace else []
        )
        result = self.run_viewer(*render_args, input_mode=mode, path=path)
        summary = re.search(
            r"SCENE_INPUT_TEST frames=(\d+) scenes=(\d+)", result.stdout
        )
        details = re.search(
            r"FILE_INPUT_TEST browser=(\d+) custom=(\d+) error=(\d+) drop_unloaded=(\d+)",
            result.stdout,
        )
        self.assertIsNotNone(summary, result.stdout + result.stderr)
        self.assertIsNotNone(details, result.stdout + result.stderr)
        self.assertEqual(result.stdout.count("Viewer closed:"), 1, result.stdout)
        if trace:
            self.assertIn("trace=1", result.stdout)
        return tuple(map(int, summary.groups())), tuple(map(int, details.groups()))

    def launch_batches(self, samples_per_batch, mode="batch"):
        screenshot = Path(self.temp.name) / f"batch-{mode}-{samples_per_batch}.png"
        result = self.run_viewer(
            "--mode", "trace", "--integrator", "normals", "--sampler", "halton",
            "--width", "32", "--height", "24", "--max-spp", "5",
            "--samples-per-batch", str(samples_per_batch),
            "--screenshot", str(screenshot), input_mode=mode,
        )
        match = re.search(
            r"BATCH_INPUT_TEST samples=(\d+) total=(\d+) batches=(\d+) "
            r"resets=(\d+) control=(\d+) setting_four=(\d+)",
            result.stdout,
        )
        self.assertIsNotNone(match, result.stdout + result.stderr)
        return tuple(map(int, match.groups())), screenshot

    def test_all_samples_in_one_batch_matches_partial_batches(self):
        from PIL import Image, ImageChops

        all_stats, all_image = self.launch_batches(-1)
        partial_stats, partial_image = self.launch_batches(2)
        self.assertEqual(all_stats[:4], (5, 5, 1, 0))
        self.assertEqual(partial_stats[:4], (5, 5, 3, 0))
        with Image.open(all_image) as a, Image.open(partial_image) as b:
            # Exclude the toolbar; only sample grouping should differ.
            bounds = (0, 140, a.width, a.height)
            difference = ImageChops.difference(
                a.convert("RGB").crop(bounds), b.convert("RGB").crop(bounds)
            )
            self.assertTrue(all(high <= 1 for _, high in difference.getextrema()))

    def test_batch_size_larger_than_remaining_budget_is_clamped(self):
        stats, _ = self.launch_batches(8)
        self.assertEqual(stats[:4], (5, 5, 1, 0))

    def test_keyboard_changes_batch_size_without_resetting_samples(self):
        stats, _ = self.launch_batches(2, "batch-key")
        self.assertEqual(stats[:2], (5, 5))
        self.assertLessEqual(stats[2], 3)
        self.assertEqual(stats[3:], (0, 1, 1))

    def test_plus_button_changes_batch_size_without_resetting_samples(self):
        stats, _ = self.launch_batches(2, "batch-click")
        self.assertEqual(stats[:2], (5, 5))
        self.assertLessEqual(stats[2], 3)
        self.assertEqual(stats[3:], (0, 1, 1))

    def test_invalid_samples_per_batch_are_rejected(self):
        for value in (0, -2):
            with self.subTest(value=value):
                result = self.run_viewer(
                    "--samples-per-batch", str(value), check=False, timeout=20,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("invalid viewer", result.stderr)

    def test_one_c_press_switches_one_scene_and_finishes_frames(self):
        summary, _ = self.launch("cycle")
        self.assertEqual(summary, (6, 2))

    def test_keyboard_browser_navigates_and_loads_scene_with_spaces(self):
        summary, (browser, custom, error, _) = self.launch("open", self.scene.parent)
        self.assertEqual(summary, (8, 2))
        self.assertGreater(browser, 0)
        self.assertEqual((custom, error), (1, 0))

    def test_open_button_and_browser_open_button_load_scene(self):
        summary, (browser, custom, error, _) = self.launch("button", self.scene.parent)
        self.assertEqual(summary, (8, 2))
        self.assertGreater(browser, 0)
        self.assertEqual((custom, error), (1, 0))

    def test_double_click_loads_scene(self):
        summary, (browser, custom, error, _) = self.launch("double", self.scene.parent)
        self.assertEqual(summary, (8, 2))
        self.assertGreater(browser, 0)
        self.assertEqual((custom, error), (1, 0))

    def test_replacement_scene_can_ray_trace(self):
        (_, scenes), (_, custom, error, unloaded) = self.launch(
            "drop-trace", self.scene, trace=True
        )
        self.assertEqual(scenes, 2)
        self.assertEqual((custom, error, unloaded), (1, 0, 1))

    def test_escape_cancels_browser_without_closing_viewer(self):
        summary, (browser, custom, _, _) = self.launch("cancel")
        self.assertEqual(summary, (8, 1))
        self.assertGreater(browser, 0)
        self.assertEqual(custom, 0)

    def test_invalid_scene_keeps_old_scene_and_reports_error(self):
        summary, (browser, custom, error, _) = self.launch("invalid", self.invalid)
        self.assertEqual(summary, (8, 1))
        self.assertGreater(browser, 0)
        self.assertEqual((custom, error), (0, 1))

    def test_dropped_scene_loads_and_releases_file_list(self):
        summary, (_, custom, error, unloaded) = self.launch("drop", self.scene)
        self.assertEqual(summary, (8, 2))
        self.assertEqual((custom, error, unloaded), (1, 0, 1))

    def test_invalid_drop_keeps_old_scene(self):
        summary, (_, custom, error, unloaded) = self.launch("bad-drop", self.invalid)
        self.assertEqual(summary, (8, 1))
        self.assertEqual((custom, error, unloaded), (0, 1, 1))


if __name__ == "__main__":
    unittest.main()
