"""Changed-input graph evidence must cover every bound and grouped primitive."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'bench/warp'))
spec = importlib.util.spec_from_file_location('compare_spatial', Path(__file__).resolve().parents[2] / 'bench/warp/compare_spatial.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class SpatialGraphEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.prefix = Path(self.directory.name) / 'graph'
        self.groups = np.arange(runner.ITEMS) % 4
        lo = np.arange(runner.ITEMS * 3, dtype=np.float32).reshape(-1, 3) * .125
        self.boxes = np.concatenate((lo, lo + .25), axis=1)
        self.nodes, self.bounds, self.roots = [], [], []
        def emit(ids):
            node = len(self.nodes)
            self.nodes.append([0, 4294967296])
            self.bounds.append(np.concatenate((self.boxes[ids, :3].min(axis=0), self.boxes[ids, 3:].max(axis=0))))
            if len(ids) == 1:
                self.nodes[node][1] = int(ids[0])
            else:
                middle = len(ids) // 2
                emit(ids[:middle])
                emit(ids[middle:])
            self.nodes[node][0] = len(self.nodes)
        emit(np.arange(runner.ITEMS))
        for group in range(4):
            self.roots.append(len(self.nodes))
            emit(np.flatnonzero(self.groups == group))
        self.nodes = np.array(self.nodes, dtype='<f4')
        self.bounds = np.array(self.bounds, dtype='<f4')
        self.roots = np.array(self.roots, dtype='<f4')

    def check(self):
        for suffix, data in (('nodes', self.nodes), ('bounds', self.bounds), ('roots', self.roots)):
            data.tofile(str(self.prefix) + '.' + suffix)
        return runner.validate_graph_bounds(self.prefix, self.boxes, self.groups)

    def test_complete_evidence_passes(self):
        self.assertEqual(self.check(), 1019)

    def test_stale_parent_or_leaf_bounds_fail(self):
        for node in (0, 8, 511, 1018):
            with self.subTest(node=node):
                self.bounds[node, 0] += .125
                with self.assertRaises(RuntimeError):
                    self.check()
                self.bounds[node, 0] -= .125

    def test_wrong_group_or_missing_nodes_fail(self):
        self.roots[[0, 1]] = self.roots[[1, 0]]
        with self.assertRaises(RuntimeError):
            self.check()
        self.roots[[0, 1]] = self.roots[[1, 0]]
        self.nodes = self.nodes[:-1]
        with self.assertRaises(RuntimeError):
            self.check()

    def test_invalid_escape_or_duplicate_primitive_fails(self):
        self.nodes[0, 0] = len(self.nodes) + 1
        with self.assertRaises(RuntimeError):
            self.check()
        self.nodes[0, 0] = 511
        leaves = np.flatnonzero(self.nodes[:511, 1] < runner.ITEMS)
        self.nodes[leaves[1], 1] = self.nodes[leaves[0], 1]
        with self.assertRaises(RuntimeError):
            self.check()


if __name__ == '__main__':
    unittest.main()
