"""Host checks for timing boundaries and the shared reference identity."""
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'bench/warp'))
import benchmark_common as common


class BenchmarkCommonTests(unittest.TestCase):
    def test_timing_requires_both_warmups_and_keeps_owner_until_sync(self):
        # An elapsed warmup period alone cannot bypass the two warmup calls.
        # Conversely, two calls cannot bypass the elapsed warmup period.
        for starts, calls in (([50, 60, 70, 80], 4),
                              ([0, 20, 40, 50, 70], 5)):
            with self.subTest(starts=starts):
                clock = iter([0, *(v for start in starts for v in (start, start + 3))])
                objects = []
                syncs = []
                destroyed = []

                class Owner:
                    def __del__(self):
                        destroyed.append(True)

                def operation():
                    owner = Owner()
                    objects.append(weakref.ref(owner))
                    return owner

                def synchronize(device):
                    self.assertEqual(device, 'fake-device')
                    if len(syncs) % 2:
                        self.assertIsNotNone(objects[-1]())
                    elif objects:
                        self.assertIsNone(objects[-1]())
                    syncs.append(device)

                wp = SimpleNamespace(synchronize_device=synchronize)
                with patch.object(common, 'WARMUP_NS', 50), \
                     patch.object(common.time, 'perf_counter_ns', side_effect=lambda: next(clock)):
                    self.assertEqual(common.timed(wp, 'fake-device', operation, repeats=2), [3, 3])
                self.assertEqual(len(objects), calls)
                self.assertEqual(len(syncs), calls * 2)
                self.assertEqual(len(destroyed), calls)

    def test_command_failure_preserves_output_in_log(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'command.txt'
            with self.assertRaisesRegex(RuntimeError, 'before failure'):
                common.run([sys.executable, '-c',
                            "print('before failure'); raise SystemExit(7)"], log=log)
            self.assertEqual(log.read_text(), 'before failure\n')

    def test_reference_identifies_source_and_native_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory)
            (checkout / '.git').mkdir()
            package = checkout / 'warp'
            files = {'__init__.py': b'', 'bin/warp.so': b'native library',
                     'native/bvh.cu': b'CUDA source', 'native/mesh.h': b'header'}
            for name, contents in files.items():
                path = package / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(contents)
            wp = SimpleNamespace(__file__=str(package / '__init__.py'), __version__='test')
            with patch.object(common, 'run', return_value='exact-commit\n'):
                identity = common.warp_reference(wp)
            self.assertEqual(identity['commit'], 'exact-commit')
            self.assertEqual(identity['version'], 'test')
            self.assertEqual(identity['native_hashes'], {
                'bin/warp.so': hashlib.sha256(files['bin/warp.so']).hexdigest()})
            self.assertEqual(identity['source_hashes'], {
                name: hashlib.sha256(contents).hexdigest()
                for name, contents in files.items() if name.startswith('native/')})


if __name__ == '__main__':
    unittest.main()
