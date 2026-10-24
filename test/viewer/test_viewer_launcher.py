"""The native viewer must run a compiled binary, including after cache hits."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


class ViewerLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for directory in ("scripts", "bajo", "examples", "bin"):
            (self.root / directory).mkdir()
        script = Path(__file__).resolve().parents[2] / "scripts/run_viewer.sh"
        shutil.copy(script, self.root / "scripts/run_viewer.sh")
        (self.root / "examples/viewer_raylib.mojo").write_text("# viewer\n")
        self.log = self.root / "compiler.jsonl"
        compiler = self.root / "bin/mojo"
        executable = (
            f"#!{sys.executable}\nimport json, sys\n"
            "print(json.dumps(sys.argv[1:]))\n"
        )
        compiler.write_text(
            f"#!{sys.executable}\n" + textwrap.dedent(f"""\
                import json, os, pathlib, sys
                args = sys.argv[1:]
                with open(os.environ['COMPILER_LOG'], 'a') as log:
                    log.write(json.dumps(args) + '\\n')
                if args == ['--version']:
                    print('test Mojo compiler')
                elif args[0] == 'build':
                    output = pathlib.Path(args[args.index('-o') + 1])
                    output.write_text({executable!r})
                    output.chmod(0o755)
                else:
                    raise SystemExit('viewer must not use mojo run')
                """)
        )
        compiler.chmod(0o755)
        self.env = dict(
            os.environ,
            PATH=f"{self.root / 'bin'}:{os.environ['PATH']}",
            COMPILER_LOG=str(self.log),
        )

    def launch(self, *args):
        result = subprocess.run(
            ["bash", str(self.root / "scripts/run_viewer.sh"), *args],
            env=self.env, text=True, capture_output=True, check=True,
        )
        return json.loads(result.stdout.splitlines()[-1])

    def builds(self):
        calls = map(json.loads, self.log.read_text().splitlines())
        return [args for args in calls if args[0] == "build"]

    def test_executes_compiled_viewer_and_forwards_arguments(self):
        args = ["--scene", "cornell", "--hidden"]
        self.assertEqual(self.launch(*args), args)
        self.assertEqual(len(self.builds()), 1)
        self.assertNotIn("--target-accelerator", self.builds()[0])

    def test_reuses_cache_and_rebuilds_for_new_source(self):
        self.launch("--help")
        self.launch("--help")
        self.assertEqual(len(self.builds()), 1)
        (self.root / "bajo/new_module.mojo").write_text("# untracked module\n")
        self.launch("--help")
        self.assertEqual(len(self.builds()), 2)
        (self.root / "bajo/new_module.mojo").write_text("# changed module\n")
        self.launch("--help")
        self.assertEqual(len(self.builds()), 3)

    def test_launcher_changes_invalidate_cached_build(self):
        self.launch("--help")
        script = self.root / "scripts/run_viewer.sh"
        script.write_text(script.read_text() + "\n# changed build configuration\n")
        self.launch("--help")
        self.assertEqual(len(self.builds()), 2)

    def test_gpu_build_uses_requested_architecture(self):
        self.launch("--help")
        self.env["BAJO_GPU_ARCH"] = "sm_89"
        self.assertEqual(
            self.launch("--gpu", "--backend", "gpu"), ["--backend", "gpu"]
        )
        self.assertEqual(len(self.builds()), 2)
        args = self.builds()[-1]
        self.assertEqual(args[args.index("--target-accelerator") + 1], "sm_89")
        self.launch("--gpu", "--help")
        self.assertEqual(len(self.builds()), 2)


if __name__ == "__main__":
    unittest.main()
