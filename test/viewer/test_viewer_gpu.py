"""Run with BAJO_TEST_GPU=1 on a machine with a GPU."""
import os
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from viewer_extensions import ExtensionLoader


@unittest.skipUnless(os.environ.get("BAJO_TEST_GPU") == "1", "requires BAJO_TEST_GPU=1")
class GpuFrameTests(unittest.TestCase):
    def test_lbvh_meshes_scene_renders(self):
        renderer = ExtensionLoader(ROOT, "sm_120").load(1, 3, 0, 0)
        config = dict(
            width=24, height=16, samples=1, max_depth=1,
            sampler=0, sample_offset=0, sample_sequence_length=1,
            scene=3, scene_path="", x=0.0, y=6.0, z=-28.0,
            yaw=180.0, pitch=-8.0, vfov=35.0,
        )
        created = renderer.preparation.create_gpu_state(config)
        handle, tag = int(created[0]), int(created[1])
        try:
            pixels = renderer.rendering.render_gpu_state(handle, tag, config)[3]
            self.assertEqual(pixels.shape, (16, 24, 3))
            self.assertTrue(np.isfinite(pixels).all())
            self.assertTrue((pixels > 0).any())
        finally:
            renderer.preparation.destroy_gpu_state(handle, tag)

    def test_owned_frame_survives_render_resize_and_scene_destruction(self):
        renderer = ExtensionLoader(ROOT, "sm_120").load(1, 3, 0, 0)
        config = dict(
            width=24, height=16, samples=1, max_depth=3,
            sampler=0, sample_offset=0, sample_sequence_length=1,
            scene=1, scene_path="", x=0.0, y=1.0, z=3.2,
            yaw=0.0, pitch=0.0, vfov=28.0,
        )
        created = renderer.preparation.create_gpu_state(config)
        handle, tag = int(created[0]), int(created[1])
        try:
            result = renderer.rendering.render_gpu_state(handle, tag, config)
            pixels = result[3]
            self.assertEqual(pixels.shape, (16, 24, 3))
            self.assertEqual(pixels.dtype, np.float32)
            self.assertTrue(pixels.flags.c_contiguous)
            self.assertTrue(pixels.flags.owndata)
            self.assertTrue(np.isfinite(pixels).all())
            self.assertTrue((pixels > 0).any())
            saved = pixels.copy()
            config.update(width=13, height=9)
            resized = renderer.rendering.render_gpu_state(handle, tag, config)[3]
            self.assertEqual(resized.shape, (9, 13, 3))
            self.assertTrue(np.isfinite(resized).all())
            np.testing.assert_array_equal(pixels, saved)
        finally:
            renderer.preparation.destroy_gpu_state(handle, tag)
        np.testing.assert_array_equal(pixels, saved)
        self.assertTrue(np.isfinite(resized).all())


if __name__ == "__main__":
    unittest.main()
