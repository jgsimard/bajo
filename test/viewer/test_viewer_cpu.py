"""Headless integration: allocate, borrow across rendering modules, then free."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from viewer_extensions import ExtensionLoader


class CpuExtensionTests(unittest.TestCase):
    def test_shared_scene_across_integrators_and_traversal(self):
        loader = ExtensionLoader(ROOT, "sm_120")
        with tempfile.TemporaryDirectory() as tmp:
            config = dict(
                output=str(Path(tmp) / "frame.rgb32"),
                width=24, height=16, samples=1, max_depth=3,
                sampler=0, sample_offset=0, sample_sequence_length=1,
                scene=1, scene_path="", x=0.0, y=1.0, z=3.2,
                yaw=0.0, pitch=0.0, vfov=28.0,
            )
            normals = loader.load(0, 3, 0, 0)
            created = normals.preparation.create_cpu_state(config)
            handle = int(created[0])
            try:
                def render(extension):
                    stats = extension.rendering.render_cpu_state(handle, config)
                    self.assertGreaterEqual(float(stats[0]), 0.0)
                    pixels = np.fromfile(config["output"], dtype="<f4")
                    self.assertEqual(pixels.size, 24 * 16 * 3)
                    self.assertTrue(np.isfinite(pixels).all())
                    self.assertTrue((pixels > 0).any())
                    return pixels
                before = render(normals)
                path = loader.load(0, 0, 0, 0)
                self.assertIs(path.preparation, normals.preparation)
                path_pixels = render(path)
                self.assertFalse(np.array_equal(before, path_pixels))
                fixed = loader.load(0, 0, 1, 0)
                self.assertIs(fixed.preparation, normals.preparation)
                np.testing.assert_allclose(render(fixed), path_pixels, atol=1e-5, rtol=1e-5)
                # Reloaded modules must safely borrow the same allocation.
                np.testing.assert_array_equal(render(loader.load(0, 3, 0, 0)), before)
            finally:
                normals.preparation.destroy_cpu_state(handle)
            lbvh = loader.load(0, 0, 0, 1)
            self.assertIs(lbvh.rendering, path.rendering)
            self.assertIsNot(lbvh.preparation, path.preparation)
            built = lbvh.preparation.create_cpu_state(config)
            try:
                handle = int(built[0])
                render(lbvh)
                self.assertIn("lbvh", str(built[3]))
            finally:
                lbvh.preparation.destroy_cpu_state(int(built[0]))


if __name__ == "__main__":
    unittest.main()
