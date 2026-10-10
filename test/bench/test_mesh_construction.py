"""Independent construction checks must detect corrupt published geometry/trees."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'bench/warp'))
spec = importlib.util.spec_from_file_location('mesh_construction', ROOT / 'bench/warp/compare_mesh_construction.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class ConstructionValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.prefix = Path(self.directory.name) / 'fixture'
        corners = np.array([[0, 0, 0], [.14637, 0, 0], [0, .14637, 0], [0, 0, .14637]], dtype=np.float32)
        points = np.concatenate([corners + np.array([i*.43113, 0, 0], dtype=np.float32) for i in range(4)])
        indices = np.concatenate([np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]) + i*4 for i in range(4)]).astype(np.uint32)
        triangles = points[indices]
        boxes = np.concatenate([triangles.min(axis=1), triangles.max(axis=1)], axis=1)
        nodes, bounds = [], []
        def tree(faces):
            node = len(nodes)
            nodes.append([0, 0xffffffff])
            bounds.append(None)
            if len(faces) == 1:
                nodes[node][1] = int(faces[0])
                bounds[node] = boxes[faces[0]]
            else:
                left = tree(faces[:len(faces)//2])
                right = tree(faces[len(faces)//2:])
                bounds[node] = np.concatenate([np.minimum(bounds[left][:3], bounds[right][:3]),
                                               np.maximum(bounds[left][3:], bounds[right][3:])])
            nodes[node][0] = len(nodes)
            return node
        tree(np.arange(16))
        roots = [0]*4 + [tree(np.arange(group*4, group*4+4)) for group in range(4)]
        queries = np.tile(np.array([-1, -1, -1, 1000, 1000, 1000], dtype=np.float32), (8, 1))
        candidates = np.ones((8, 16), dtype=np.float32)
        for q in range(4, 8):
            candidates[q] = (np.arange(16)//4 == q%4)
        self.arrays = {'points': points, 'published-points': points.copy(),
                       'indices': indices.astype(np.float32), 'published-indices': indices,
                       'velocities': np.zeros_like(points), 'published-velocities': np.zeros_like(points),
                       'nodes': np.array(nodes, dtype=np.uint32), 'bounds': np.array(bounds, dtype=np.float32),
                       'roots': np.array(roots, dtype=np.uint32), 'boxes': queries, 'candidates': candidates}
        self.write()

    def write(self):
        for suffix, array in self.arrays.items():
            array.tofile(str(self.prefix) + '.' + suffix)

    def test_complete_independent_geometry_and_forest_pass(self):
        result = benchmark.check_bajo(self.prefix, 16, False)
        self.assertEqual(result[-1], 59)

    def test_geometry_bounds_and_duplicate_candidates_fail(self):
        for suffix in ('published-points', 'published-indices', 'published-velocities', 'bounds', 'candidates'):
            with self.subTest(suffix=suffix):
                original = self.arrays[suffix].copy()
                self.arrays[suffix].flat[0] += 1
                self.write()
                with self.assertRaises((AssertionError, RuntimeError)):
                    benchmark.check_bajo(self.prefix, 16, False)
                self.arrays[suffix] = original

    def test_wrong_group_root_and_invalid_escape_fail(self):
        self.arrays['roots'][5] = 0
        self.write()
        with self.assertRaises(AssertionError):
            benchmark.check_bajo(self.prefix, 16, False)
        self.arrays['roots'][5] = 38
        self.arrays['nodes'][0, 0] = 1
        self.write()
        with self.assertRaises(RuntimeError):
            benchmark.check_bajo(self.prefix, 16, False)


if __name__ == '__main__':
    unittest.main()
