"""Opt-in real OpenGL regression checks for the native viewer's frame loop."""

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
    def test_one_c_press_switches_one_scene_and_finishes_frames(self):
        root = Path(__file__).resolve().parents[2]
        raylib = root / ".pixi/raylib"
        with tempfile.TemporaryDirectory() as directory:
            shim = Path(directory) / "scene_input.so"
            subprocess.run(
                [
                    "cc", "-shared", "-fPIC",
                    str(Path(__file__).with_name("raylib_scene_input.c")),
                    f"-I{raylib / 'include'}", f"-L{raylib / 'lib'}",
                    "-Wl,--no-as-needed", "-lraylib",
                    f"-Wl,-rpath,{raylib / 'lib'}", "-ldl", "-o", str(shim),
                ],
                check=True, capture_output=True, text=True,
            )
            result = subprocess.run(
                [
                    "pixi", "run", "viewer", "--scene", "cornell", "--hidden",
                    "--raylib-library", str(shim),
                ],
                cwd=root,
                env=dict(os.environ, BAJO_TEST_RAYLIB=str(raylib / "lib/libraylib.so")),
                check=True, capture_output=True, text=True, timeout=240,
            )
        match = re.search(
            r"SCENE_INPUT_TEST frames=(\d+) scenes=(\d+)", result.stdout
        )
        self.assertIsNotNone(match, result.stdout + result.stderr)
        self.assertEqual(tuple(map(int, match.groups())), (6, 2), result.stdout)
        self.assertEqual(result.stdout.count("Viewer closed:"), 1, result.stdout)


if __name__ == "__main__":
    unittest.main()
