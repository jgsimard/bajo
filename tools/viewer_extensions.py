"""Load preparation owners independently of specialized rendering borrowers."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


@dataclass(frozen=True)
class ViewerRenderer:
    preparation: object
    rendering: object


class ExtensionLoader:
    def __init__(self, root: Path, gpu_arch: str) -> None:
        self.root = root
        self.gpu_arch = gpu_arch
        self._modules: dict[Path, object] = {}

    def load(
        self, backend: int, integrator: int, traversal: int, builder: int
    ) -> ViewerRenderer:
        mojo = shutil.which("mojo")
        if mojo is None:
            raise RuntimeError(
                "could not find mojo; run the viewer with `pixi run viewer`"
            )
        # Handles cross module boundaries as pointers to Mojo structs. Give all
        # modules the same compiler/source fingerprint so their layouts agree.
        version = subprocess.run(
            [mojo, "--version"], check=True, capture_output=True, text=True
        ).stdout
        fingerprint = hashlib.sha256(version.encode())
        sources = sorted([
            self.root / "bajo_viewer.mojo",
            self.root / "bajo_viewer_prepare.mojo",
            *(self.root / "bajo").rglob("*.mojo"),
            *(self.root / "examples").rglob("*.mojo"),
        ])
        for source in sources:
            fingerprint.update(str(source.relative_to(self.root)).encode())
            fingerprint.update(source.read_bytes())
        revision = fingerprint.hexdigest()[:20]
        common = {"VIEWER_BACKEND": backend}
        preparation = self._load_extension(
            mojo, "bajo_viewer_prepare", revision,
            {**common, **({"VIEWER_BUILD": builder} if backend == 0 else {})},
        )
        rendering = self._load_extension(
            mojo, "bajo_viewer", revision,
            {**common, "VIEWER_INTEGRATOR": integrator,
             **({"VIEWER_TRAVERSAL": traversal} if backend == 0 else {})},
        )
        return ViewerRenderer(preparation, rendering)

    def _load_extension(
        self, mojo: str, name: str, revision: str, defines: dict[str, int]
    ) -> object:
        cache_dir = self.root / "__mojocache__"
        cache_dir.mkdir(exist_ok=True)
        backend = defines["VIEWER_BACKEND"]
        config = "-".join(f"{key}={value}" for key, value in defines.items())
        arch = f"-{self.gpu_arch}" if backend == 1 else ""
        cache_path = cache_dir / f"{name}-{revision}{arch}-{config}.so"
        if cache_path in self._modules:
            return self._modules[cache_path]
        if not cache_path.exists():
            # Publish only complete builds, and never overwrite a loaded module.
            with tempfile.TemporaryDirectory(prefix="build-", dir=cache_dir) as tmp:
                output = Path(tmp) / f"{name}.so"
                command = [mojo, "build", "-I", str(self.root)]
                for key, value in defines.items():
                    command.extend(["-D", f"{key}={value}"])
                if backend == 1:
                    command.extend(["--target-accelerator", self.gpu_arch])
                command.extend([
                    "--emit", "shared-lib", "-o", str(output),
                    str(self.root / f"{name}.mojo"),
                ])
                result = subprocess.run(
                    command, cwd=self.root, capture_output=True, text=True
                )
                if result.returncode != 0:
                    details = (result.stderr or result.stdout).strip()
                    raise RuntimeError(f"could not build {name} ({config}): {details}")
                output.replace(cache_path)
        spec = importlib.util.spec_from_file_location(name, cache_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load Mojo viewer extension: {cache_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        self._modules[cache_path] = module
        return module
