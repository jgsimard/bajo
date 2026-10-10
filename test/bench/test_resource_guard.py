"""Memory safeguards must stop only their own diagnostic process group."""

import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[2]
GUARD = ROOT / 'bench/diagnostic/resource_guard.py'
SPEC = importlib.util.spec_from_file_location('resource_guard', GUARD)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ResourceGuardTests(unittest.TestCase):
    def run_guard(self, code, *options):
        return subprocess.run(
            [sys.executable, str(GUARD), '--min-available-mib', '0',
             *options, '--', sys.executable, '-c', code],
            capture_output=True, text=True, timeout=15,
        )

    def test_preserves_child_exit_code(self):
        result = self.run_guard('raise SystemExit(7)')
        self.assertEqual(result.returncode, 7, result.stderr)

    def test_reserve_refuses_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'started'
            result = self.run_guard(
                f'from pathlib import Path; Path({str(marker)!r}).touch()',
                '--min-available-mib', '1000000000',
            )
            self.assertEqual(result.returncode, 125, result.stderr)
            self.assertFalse(marker.exists())

    def test_budget_stops_allocating_job(self):
        result = self.run_guard(
            'import time; allocation = bytearray(32 * 1024**2); time.sleep(10)',
            '--max-rss-mib', '16',
        )
        self.assertEqual(result.returncode, 125, result.stderr)
        self.assertIn('stopping own job', result.stdout)

    def test_cleans_child_that_outlives_parent_and_ignores_term(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'child-group'
            child = (
                'import os, signal, time; from pathlib import Path; '
                'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                f'Path({str(marker)!r}).write_text(str(os.getpgrp())); '
                'time.sleep(10)'
            )
            parent = (
                'import subprocess, sys, time; from pathlib import Path; '
                f'subprocess.Popen([sys.executable, "-c", {child!r}]); '
                f'marker = Path({str(marker)!r}); '
                '\nwhile not marker.exists(): time.sleep(0.01)'
            )
            result = self.run_guard(parent)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(marker.exists())
            self.assertEqual(MODULE.group_rss(int(marker.read_text())), 0)

    def test_terminating_guard_cleans_its_isolated_job(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'child-group'
            child = (
                'import os, signal, time; from pathlib import Path; '
                'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                f'Path({str(marker)!r}).write_text(str(os.getpgrp())); '
                'time.sleep(20)'
            )
            guard = subprocess.Popen(
                [sys.executable, str(GUARD), '--min-available-mib', '0',
                 '--', sys.executable, '-c', child],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            try:
                deadline = time.monotonic() + 5
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(marker.exists())
                group = int(marker.read_text())
                guard.terminate()
                stdout, stderr = guard.communicate(timeout=10)
                self.assertEqual(guard.returncode, 143, stderr)
                self.assertIn('peak observed RSS', stdout)
                self.assertEqual(MODULE.group_rss(group), 0)
            finally:
                if guard.poll() is None:
                    guard.kill()
                    guard.communicate(timeout=5)
                if marker.exists():
                    try:
                        os.killpg(int(marker.read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass


if __name__ == '__main__':
    unittest.main()
