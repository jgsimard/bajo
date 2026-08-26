from __future__ import annotations

import argparse
from collections import deque
import importlib.util
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk


ROOT = Path(__file__).resolve().parents[1]
BUILTIN_PBRT_PATH = ROOT / "examples" / "scenes" / "pbrt_showcase.pbrt"
SETTLE_DELAY_SECONDS = 0.25
PREVIEW_INTERVAL_SECONDS = 0.08
INTEGRATORS = ("PATH", "NEE", "MIS", "NORMALS", "AO")
BACKENDS = ("CPU", "GPU")
TRAVERSALS = (
    "AUTO COHERENT",
    "FIXED PACKET",
    "ADAPTIVE 16/8/4/SCALAR",
)
BUILDERS = ("SAH", "LBVH", "H-PLOC", "MEDIAN")
TRAVERSAL_CLI = ("auto", "fixed", "adaptive")
BUILDER_CLI = ("sah", "lbvh", "hploc", "median")
SAMPLER_CLI = ("independent", "halton", "r2", "sobol", "sz", "stbn")
SAMPLERS = (
    "INDEPENDENT",
    "HALTON",
    "R2",
    "OWEN SOBOL",
    "SZ",
    "STBN",
)


@dataclass
class Camera:
    x: float = 13.0
    y: float = 2.0
    z: float = 3.0
    # Looking from (13, 2, 3) towards the RTIAW scene origin.
    yaw: float = -77.0
    pitch: float = -8.5
    vfov: float = 28.0

    def copy(self) -> "Camera":
        return Camera(self.x, self.y, self.z, self.yaw, self.pitch, self.vfov)


@dataclass(frozen=True)
class SceneSpec:
    cli_name: str | None
    label: str
    camera: Camera


SCENE_SPECS = (
    SceneSpec("rtiaw", "RTIAW", Camera()),
    SceneSpec(
        "cornell",
        "CORNELL",
        Camera(x=0.0, y=1.0, z=3.2, yaw=0.0, pitch=0.0, vfov=28.0),
    ),
    SceneSpec(
        "veach",
        "VEACH",
        Camera(x=0.0, y=3.0, z=6.2, yaw=0.0, pitch=-12.0, vfov=31.0),
    ),
    SceneSpec(
        "lbvh",
        "LBVH MESHES",
        Camera(x=0.0, y=6.0, z=-28.0, yaw=180.0, pitch=-8.0, vfov=35.0),
    ),
    SceneSpec(
        "emissive-instance",
        "EMISSIVE INSTANCE",
        Camera(x=0.0, y=1.6, z=5.8, yaw=0.0, pitch=-7.0, vfov=42.0),
    ),
    SceneSpec(
        "many-lights",
        "MANY LIGHTS",
        Camera(x=0.0, y=4.3, z=11.0, yaw=0.0, pitch=-12.0, vfov=52.0),
    ),
    SceneSpec(
        "indirect-hall",
        "INDIRECT HALL",
        Camera(x=0.0, y=2.2, z=8.0, yaw=0.0, pitch=-2.0, vfov=55.0),
    ),
    SceneSpec(
        "specular-transport",
        "SPECULAR TRANSPORT",
        Camera(x=0.0, y=2.7, z=9.5, yaw=0.0, pitch=-8.0, vfov=48.0),
    ),
    SceneSpec("pbrt", "PBRT MESHES", Camera()),
    SceneSpec(None, "LOAD PBRT…", Camera()),
)
SCENES = tuple(spec.label for spec in SCENE_SPECS)
SCENE_INDEX_BY_CLI = {
    spec.cli_name: index
    for index, spec in enumerate(SCENE_SPECS)
    if spec.cli_name is not None
}
BUILTIN_PBRT_SCENE = SCENE_INDEX_BY_CLI["pbrt"]
CUSTOM_PBRT_SCENE = len(SCENE_SPECS) - 1


@dataclass
class RenderOptions:
    width: int = 320
    height: int = 214
    batches: int = 4
    max_samples: int = 32
    max_depth: int = 8


@dataclass
class GpuState:
    renderer: object
    handle: int
    tag: int
    key: tuple[object, ...]
    bvh_stats: str


@dataclass(frozen=True)
class RenderSnapshot:
    camera: Camera
    options: RenderOptions
    generation: int
    batch_spp: int
    sample_offset: int
    preview: bool
    integrator_index: int
    backend_index: int
    traversal_index: int
    build_index: int
    sampler_index: int
    scene_index: int
    scene_path: str


@dataclass(frozen=True)
class RenderStats:
    render_ms: float
    build_ms: float
    mrays: float
    bvh_stats: str


def default_camera(scene_index: int) -> Camera:
    return SCENE_SPECS[scene_index].camera.copy()


def camera_from_pbrt(values) -> Camera:
    origin = (float(values[0]), float(values[1]), float(values[2]))
    forward = (float(values[3]), float(values[4]), float(values[5]))
    yaw = math.degrees(math.atan2(forward[0], -forward[2]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, forward[1]))))
    vfov = math.degrees(2.0 * math.atan(float(values[6])))
    return Camera(origin[0], origin[1], origin[2], yaw, pitch, vfov)


class Viewer:
    def __init__(
        self,
        options: RenderOptions,
        scene_index: int = 0,
        pbrt_path: str | None = None,
        backend_index: int = 0,
        traversal_index: int = 0,
        build_index: int = 0,
        sampler_index: int = 0,
        gpu_arch: str = "sm_120",
    ) -> None:
        self.options = options
        self.scene_index = scene_index
        self.pbrt_path = pbrt_path
        if self.scene_index == BUILTIN_PBRT_SCENE and self.pbrt_path is None:
            self.pbrt_path = str(BUILTIN_PBRT_PATH)
        self.gpu_arch = gpu_arch
        self.camera = default_camera(scene_index)
        self.initial_camera = self.camera.copy()
        self.pressed: set[str] = set()
        self.drag_anchor: tuple[int, int] | None = None
        self.dragging = False
        self.backend_index = backend_index
        self.integrator_index = 0
        self.traversal_index = traversal_index
        self.build_index = build_index
        self.sampler_index = sampler_index
        self.last_tick = time.monotonic()
        self.last_camera_change = self.last_tick
        self.render_generation = 0
        self.accumulated_spp = 0
        self.completed_batches = 0
        self.linear_sum: np.ndarray | None = None
        self.display_times: deque[float] = deque()
        self.rendering = False
        self.closed = False
        self.lock = threading.Lock()
        self.image: Image.Image | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.gpu_state: GpuState | None = None
        self.temp_dir = tempfile.TemporaryDirectory(prefix="bajo-viewer-")
        self.output_path = Path(self.temp_dir.name) / "frame.rgb32"
        self.renderer = self._load_renderer(
            self.backend_index,
            self.integrator_index,
            self.traversal_index,
            self.build_index,
        )
        self.renderer_config = (
            self.backend_index,
            self.integrator_index,
            self.traversal_index,
            self.build_index,
        )
        if (
            self.scene_index in (BUILTIN_PBRT_SCENE, CUSTOM_PBRT_SCENE)
            and self.pbrt_path
        ):
            self.camera = camera_from_pbrt(
                self.renderer.pbrt_camera(self.pbrt_path)
            )
            self.initial_camera = self.camera.copy()

        self.root = tk.Tk()
        self.root.title("Bajo renderer viewer")
        self.root.geometry("960x640")
        self.root.configure(bg="#111111")
        self.root.minsize(640, 420)

        toolbar = tk.Frame(self.root, bg="#202020")
        toolbar.pack(fill=tk.X)
        tk.Label(
            toolbar,
            text="Backend:",
            padx=10,
            pady=4,
            fg="#dddddd",
            bg="#202020",
        ).pack(side=tk.LEFT)
        self.backend_var = tk.StringVar(value=BACKENDS[self.backend_index])
        tk.OptionMenu(
            toolbar,
            self.backend_var,
            *BACKENDS,
            command=self.on_backend_changed,
        ).pack(side=tk.LEFT, padx=(0, 8), pady=2)
        tk.Label(
            toolbar,
            text="Integrator:",
            padx=10,
            pady=4,
            fg="#dddddd",
            bg="#202020",
        ).pack(side=tk.LEFT)
        self.integrator_var = tk.StringVar(
            value=INTEGRATORS[self.integrator_index]
        )
        tk.OptionMenu(
            toolbar,
            self.integrator_var,
            *INTEGRATORS,
            command=self.on_integrator_changed,
        ).pack(side=tk.LEFT, padx=(0, 8), pady=2)
        tk.Label(
            toolbar,
            text="Scene:",
            padx=6,
            pady=4,
            fg="#dddddd",
            bg="#202020",
        ).pack(side=tk.LEFT)
        self.scene_var = tk.StringVar(value=SCENES[self.scene_index])
        tk.OptionMenu(
            toolbar,
            self.scene_var,
            *SCENES,
            command=self.on_scene_changed,
        ).pack(side=tk.LEFT, padx=(0, 8), pady=2)
        tk.Label(
            toolbar,
            text="Batches:",
            padx=6,
            pady=4,
            fg="#dddddd",
            bg="#202020",
        ).pack(side=tk.LEFT)
        self.batches_var = tk.StringVar(value=str(self.options.batches))
        self.batches_spinbox = tk.Spinbox(
            toolbar,
            from_=1,
            to=1_000_000,
            width=7,
            textvariable=self.batches_var,
            command=self.on_batches_changed,
        )
        self.batches_spinbox.pack(side=tk.LEFT, padx=(0, 8), pady=2)
        self.batches_spinbox.bind("<Return>", self.on_batches_changed)
        self.batches_spinbox.bind("<FocusOut>", self.on_batches_changed)
        tk.Label(
            toolbar,
            text="Max spp:",
            padx=6,
            pady=4,
            fg="#dddddd",
            bg="#202020",
        ).pack(side=tk.LEFT)
        self.max_spp_var = tk.StringVar(value=str(self.options.max_samples))
        self.max_spp_spinbox = tk.Spinbox(
            toolbar,
            from_=1,
            to=1_000_000,
            width=7,
            textvariable=self.max_spp_var,
            command=self.on_max_spp_changed,
        )
        self.max_spp_spinbox.pack(side=tk.LEFT, padx=(0, 8), pady=2)
        self.max_spp_spinbox.bind("<Return>", self.on_max_spp_changed)
        self.max_spp_spinbox.bind("<FocusOut>", self.on_max_spp_changed)
        tk.Label(
            toolbar,
            text="Max depth:",
            padx=6,
            pady=4,
            fg="#dddddd",
            bg="#202020",
        ).pack(side=tk.LEFT)
        self.max_depth_var = tk.StringVar(value=str(self.options.max_depth))
        self.max_depth_spinbox = tk.Spinbox(
            toolbar,
            from_=1,
            to=16,
            width=5,
            textvariable=self.max_depth_var,
            command=self.on_max_depth_changed,
        )
        self.max_depth_spinbox.pack(side=tk.LEFT, padx=(0, 8), pady=2)
        self.max_depth_spinbox.bind("<Return>", self.on_max_depth_changed)
        self.max_depth_spinbox.bind("<FocusOut>", self.on_max_depth_changed)

        policybar = tk.Frame(self.root, bg="#181818")
        policybar.pack(fill=tk.X)
        tk.Label(
            policybar,
            text="CPU traversal:",
            padx=10,
            pady=4,
            fg="#dddddd",
            bg="#181818",
        ).pack(side=tk.LEFT)
        self.traversal_var = tk.StringVar(
            value=TRAVERSALS[self.traversal_index]
        )
        self.traversal_menu = tk.OptionMenu(
            policybar,
            self.traversal_var,
            *TRAVERSALS,
            command=self.on_traversal_changed,
        )
        self.traversal_menu.pack(side=tk.LEFT, padx=(0, 12), pady=2)
        tk.Label(
            policybar,
            text="CPU build:",
            padx=6,
            pady=4,
            fg="#dddddd",
            bg="#181818",
        ).pack(side=tk.LEFT)
        self.build_var = tk.StringVar(value=BUILDERS[self.build_index])
        self.build_menu = tk.OptionMenu(
            policybar,
            self.build_var,
            *BUILDERS,
            command=self.on_build_changed,
        )
        self.build_menu.pack(side=tk.LEFT, padx=(0, 12), pady=2)
        tk.Label(
            policybar,
            text="Sampler:",
            padx=6,
            pady=4,
            fg="#dddddd",
            bg="#181818",
        ).pack(side=tk.LEFT)
        self.sampler_var = tk.StringVar(value=SAMPLERS[self.sampler_index])
        self.sampler_menu = tk.OptionMenu(
            policybar,
            self.sampler_var,
            *SAMPLERS,
            command=self.on_sampler_changed,
        )
        self.sampler_menu.pack(side=tk.LEFT, padx=(0, 12), pady=2)
        self._update_policy_controls()

        self.image_canvas = tk.Canvas(
            self.root,
            bg="#111111",
            highlightthickness=0,
            cursor="crosshair",
        )
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        self.status = tk.Label(
            self.root,
            text="Starting renderer…",
            width=1,
            anchor="w",
            padx=10,
            pady=6,
            fg="#dddddd",
            bg="#202020",
        )
        self.status.pack(fill=tk.X)
        self.stats = tk.Label(
            self.root,
            text="BVH/RT stats unavailable",
            width=1,
            anchor="w",
            padx=10,
            pady=3,
            fg="#aaaaaa",
            bg="#181818",
        )
        self.stats.pack(fill=tk.X, before=self.status)

        self.root.bind("<KeyPress>", self.on_key_press)
        self.root.bind("<KeyRelease>", self.on_key_release)
        self.root.bind("<Escape>", lambda _event: self.close())
        self.image_canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.image_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.image_canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.image_canvas.bind(
            "<Configure>", lambda _event: self.refresh_image()
        )
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.request_render()
        self.root.after(40, self.tick)

    def _load_renderer(
        self,
        backend_index: int,
        integrator_index: int,
        traversal_index: int,
        build_index: int,
    ):
        """Build and load the Mojo renderer as a Python extension module."""
        mojo = shutil.which("mojo")
        if mojo is None:
            raise RuntimeError(
                "could not find mojo; run the viewer with `pixi run viewer`"
            )

        cache_dir = ROOT / "__mojocache__"
        cache_dir.mkdir(exist_ok=True)
        cache_path = cache_dir / (
            f"bajo_viewer-{self.gpu_arch}-b{backend_index}"
            f"-i{integrator_index}-t{traversal_index}-u{build_index}.so"
        )
        sources = [ROOT / "bajo_viewer.mojo"]
        sources.extend((ROOT / "bajo").rglob("*.mojo"))
        sources.extend((ROOT / "examples").rglob("*.mojo"))
        needs_build = not cache_path.exists()
        if not needs_build:
            cache_mtime = cache_path.stat().st_mtime_ns
            needs_build = any(
                source.stat().st_mtime_ns > cache_mtime for source in sources
            )
        if needs_build:
            command = [
                mojo,
                "build",
                "-I",
                str(ROOT),
                "-D",
                f"VIEWER_BACKEND={backend_index}",
                "-D",
                f"VIEWER_INTEGRATOR={integrator_index}",
                "-D",
                f"VIEWER_TRAVERSAL={traversal_index}",
                "-D",
                f"VIEWER_BUILD={build_index}",
                "--emit",
                "shared-lib",
                "-o",
                str(cache_path),
                str(ROOT / "bajo_viewer.mojo"),
            ]
            if backend_index == 1:
                command[command.index("--emit") : command.index("--emit")] = [
                    "--target-accelerator",
                    self.gpu_arch,
                ]
            result = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                details = (result.stderr or result.stdout).strip()
                raise RuntimeError(
                    "could not build viewer for "
                    f"backend {backend_index}, integrator {integrator_index}, "
                    f"traversal {traversal_index}, build {build_index}: {details}"
                )

        spec = importlib.util.spec_from_file_location("bajo_viewer", cache_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"could not load Mojo viewer extension: {cache_path}"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules["bajo_viewer"] = module
        spec.loader.exec_module(module)
        return module

    def snapshot(self) -> RenderSnapshot:
        with self.lock:
            options = RenderOptions(
                self.options.width,
                self.options.height,
                self.options.batches,
                self.options.max_samples,
                self.options.max_depth,
            )
            preview = (
                self.dragging
                or bool(self.pressed)
                or time.monotonic() - self.last_camera_change
                < SETTLE_DELAY_SECONDS
            )
            if preview:
                batch_spp = 1
            else:
                batch_count = min(options.batches, options.max_samples)
                next_batch = min(self.completed_batches + 1, batch_count)
                next_target = (
                    next_batch * options.max_samples + batch_count - 1
                ) // batch_count
                batch_spp = next_target - self.accumulated_spp
            return RenderSnapshot(
                camera=self.camera.copy(),
                options=options,
                generation=self.render_generation,
                batch_spp=batch_spp,
                sample_offset=0 if preview else self.accumulated_spp,
                preview=preview,
                integrator_index=self.integrator_index,
                backend_index=self.backend_index,
                traversal_index=self.traversal_index,
                build_index=self.build_index,
                sampler_index=self.sampler_index,
                scene_index=self.scene_index,
                scene_path=self.pbrt_path or "",
            )

    def mark_camera_changed(self) -> None:
        with self.lock:
            self.last_camera_change = time.monotonic()
        self.request_render()

    def request_render(self, reset_accumulation: bool = True) -> None:
        with self.lock:
            if reset_accumulation:
                self.accumulated_spp = 0
                self.completed_batches = 0
                self.linear_sum = None
            self.render_generation += 1
            should_start = not self.rendering
            if should_start:
                self.rendering = True
        if should_start:
            threading.Thread(target=self.render_worker, daemon=True).start()
        self.update_status("Rendering ....")

    def _render_config(self, work: RenderSnapshot) -> dict[str, object]:
        return {
            "output": str(self.output_path),
            "width": work.options.width,
            "height": work.options.height,
            "samples": work.batch_spp,
            "sample_offset": work.sample_offset,
            "sample_sequence_length": (
                work.options.max_samples if not work.preview else 1
            ),
            "sampler": work.sampler_index,
            "max_depth": work.options.max_depth,
            "scene": work.scene_index,
            "scene_path": work.scene_path,
            "x": work.camera.x,
            "y": work.camera.y,
            "z": work.camera.z,
            "yaw": work.camera.yaw,
            "pitch": work.camera.pitch,
            "vfov": work.camera.vfov,
        }

    def _render_batch(
        self,
        renderer,
        work: RenderSnapshot,
        renderer_config: tuple[int, int, int, int],
        render_config: dict[str, object],
    ) -> RenderStats:
        if work.backend_index == 1:
            state_key = (renderer_config, work.scene_index, work.scene_path)
            state_build_ms = 0.0
            if self.gpu_state is None or self.gpu_state.key != state_key:
                self._destroy_gpu_state()
                created = renderer.create_gpu_state(render_config)
                self.gpu_state = GpuState(
                    renderer=renderer,
                    handle=int(created[0]),
                    tag=int(created[1]),
                    key=state_key,
                    bvh_stats=str(created[3]),
                )
                state_build_ms = float(created[2])
            state = self.gpu_state
            assert state is not None
            raw_stats = renderer.render_gpu_state(
                state.handle, state.tag, render_config
            )
            return RenderStats(
                render_ms=float(raw_stats[0]),
                build_ms=state_build_ms + float(raw_stats[1]),
                mrays=float(raw_stats[2]),
                bvh_stats=state.bvh_stats,
            )

        self._destroy_gpu_state()
        raw_stats = renderer.render_frame(render_config)
        return RenderStats(
            render_ms=float(raw_stats[0]),
            build_ms=float(raw_stats[1]),
            mrays=float(raw_stats[2]),
            bvh_stats=str(raw_stats[3]),
        )

    def _read_frame(self, options: RenderOptions) -> np.ndarray:
        linear = np.fromfile(self.output_path, dtype="<f4")
        expected = options.width * options.height * 3
        if linear.size != expected:
            raise ValueError(
                f"expected {expected} linear values, got {linear.size}"
            )
        return linear.reshape((options.height, options.width, 3))

    @staticmethod
    def _display_image(linear: np.ndarray) -> Image.Image:
        gamma = np.sqrt(
            np.maximum(
                np.nan_to_num(
                    linear,
                    nan=0.0,
                    posinf=0.999,
                    neginf=0.0,
                ),
                0.0,
            )
        )
        rgb = (np.minimum(gamma, 0.999) * 256.0).astype(np.uint8)
        return Image.fromarray(rgb)

    def render_worker(self) -> None:
        while not self.closed:
            work = self.snapshot()
            started = time.monotonic()
            try:
                renderer = self.renderer
                requested_config = (
                    work.backend_index,
                    work.integrator_index,
                    work.traversal_index,
                    work.build_index,
                )
                if self.renderer_config != requested_config:
                    self._destroy_gpu_state()
                    self.root.after(
                        0,
                        lambda backend=work.backend_index, integrator=work.integrator_index, traversal=work.traversal_index, build=work.build_index: self.update_status(
                            f"Compiling {BACKENDS[backend]} / "
                            f"{INTEGRATORS[integrator]} / "
                            f"{BUILDERS[build]} / {TRAVERSALS[traversal]} ..... "
                        ),
                    )
                    renderer = self._load_renderer(
                        work.backend_index,
                        work.integrator_index,
                        work.traversal_index,
                        work.build_index,
                    )
                    with self.lock:
                        if work.generation == self.render_generation:
                            self.renderer = renderer
                            self.renderer_config = requested_config
                        else:
                            continue
                render_config = self._render_config(work)
                stats = self._render_batch(
                    renderer, work, requested_config, render_config
                )
                error = None
            except Exception as exc:  # surface Mojo/Python errors in the UI
                stats = None
                error = str(exc).strip().splitlines()[-1]
            elapsed_ms = (time.monotonic() - started) * 1000.0

            with self.lock:
                latest = work.generation == self.render_generation

            if error is not None:
                if latest:
                    self.root.after(
                        0,
                        lambda error=error: self.update_status(
                            f"Error: {error}"
                        ),
                    )
                    with self.lock:
                        self.rendering = False
                    return
            elif latest:
                try:
                    linear = self._read_frame(work.options)
                except (OSError, ValueError) as exc:
                    self.root.after(
                        0,
                        lambda exc=exc: self.update_status(
                            f"Could not display frame: {exc}"
                        ),
                    )
                    with self.lock:
                        self.rendering = False
                    return
                with self.lock:
                    latest = work.generation == self.render_generation
                    if not latest:
                        continue
                    if work.preview:
                        display_linear = linear
                        displayed_spp = 0
                    else:
                        if (
                            self.linear_sum is None
                            or self.linear_sum.shape != linear.shape
                        ):
                            self.linear_sum = np.zeros(
                                linear.shape, dtype=np.float64
                            )
                        self.linear_sum += linear * float(work.batch_spp)
                        self.accumulated_spp += work.batch_spp
                        self.completed_batches += 1
                        displayed_spp = self.accumulated_spp
                        display_linear = self.linear_sum / float(displayed_spp)
                frame = self._display_image(display_linear)
                assert stats is not None
                self.root.after(
                    0,
                    lambda frame=frame, elapsed_ms=elapsed_ms, stats=stats, displayed_spp=displayed_spp, work=work: self.show_frame(
                        frame,
                        elapsed_ms,
                        stats,
                        displayed_spp,
                        work,
                    ),
                )

                if work.preview:
                    with self.lock:
                        pending = work.generation != self.render_generation
                        active_motion = self.dragging or bool(self.pressed)
                        if not pending and not active_motion:
                            self.rendering = False
                            return
                    time.sleep(PREVIEW_INTERVAL_SECONDS)
                elif displayed_spp >= work.options.max_samples:
                    with self.lock:
                        # The current pose has reached its accumulation cap.
                        # A later camera change will set this back to zero.
                        if work.generation == self.render_generation:
                            self.rendering = False
                            return

            # The camera is unchanged: immediately render the next sample
            # batch. A newer camera generation will be picked up here and has
            # already reset accumulated_spp to zero.

    def _destroy_gpu_state(self) -> None:
        state = self.gpu_state
        if state is None:
            return
        self.gpu_state = None
        state.renderer.destroy_gpu_state(state.handle, state.tag)

    def show_frame(
        self,
        frame: Image.Image,
        elapsed_ms: float,
        stats: RenderStats,
        accumulated_spp: int,
        work: RenderSnapshot,
    ) -> None:
        if self.closed:
            return
        self.image = frame
        self.display_times.append(time.monotonic())
        cutoff = self.display_times[-1] - 1.0
        while self.display_times and self.display_times[0] < cutoff:
            self.display_times.popleft()
        if len(self.display_times) > 1:
            duration = self.display_times[-1] - self.display_times[0]
            fps = (
                len(self.display_times) - 1
            ) / duration if duration > 0 else 0.0
        else:
            fps = 0.0
        spp_text = (
            "preview" if work.preview else f"accumulated {accumulated_spp}/{work.options.max_samples} spp"
        )
        integrator_name = INTEGRATORS[work.integrator_index]
        backend_name = BACKENDS[work.backend_index]
        sampler_name = SAMPLERS[work.sampler_index]
        scene_name = (
            f"PBRT:{Path(work.scene_path).name}" if work.scene_index
            == CUSTOM_PBRT_SCENE
            and work.scene_path else SCENES[work.scene_index]
        )
        self.refresh_image()
        self.stats.configure(text=f"BVH/RT  |  {stats.bvh_stats}")
        self.update_status(
            f"{self.image.width}×{self.image.height}  |  "
            f"FPS {fps:.1f}  |  "
            f"{backend_name}  |  "
            f"{integrator_name}  |  "
            f"{sampler_name}  |  "
            f"{scene_name}  |  "
            f"Depth {work.options.max_depth}  |  "
            f"Build {stats.build_ms:.1f} ms  |  "
            f"Render {stats.render_ms:.2f} ms  |  "
            f"{stats.mrays:.2f} MRays/s  |  "
            f"Wall {elapsed_ms:.2f}ms  |  "
            f"{spp_text}"
        )

    def refresh_image(self) -> None:
        if self.image is None or self.closed:
            return
        available_w = max(1, self.image_canvas.winfo_width())
        available_h = max(1, self.image_canvas.winfo_height())
        scale = min(
            available_w / self.image.width, available_h / self.image.height
        )
        size = (
            max(1, round(self.image.width * scale)),
            max(1, round(self.image.height * scale)),
        )
        display = self.image.resize(size, Image.Resampling.BILINEAR)
        self.photo = ImageTk.PhotoImage(display)
        self.image_canvas.delete("frame")
        self.image_canvas.create_image(
            available_w // 2,
            available_h // 2,
            image=self.photo,
            tags="frame",
        )

    def update_status(self, text: str) -> None:
        if not self.closed:
            self.status.configure(text=text)

    def key_move(self, key: str, amount: float) -> None:
        yaw = math.radians(self.camera.yaw)
        pitch = math.radians(self.camera.pitch)
        cp = math.cos(pitch)
        forward = (math.sin(yaw) * cp, math.sin(pitch), -math.cos(yaw) * cp)
        right = (math.cos(yaw), 0.0, math.sin(yaw))
        if key == "w":
            direction = forward
        elif key == "s":
            direction = tuple(-v for v in forward)
        elif key == "a":
            direction = tuple(-v for v in right)
        elif key == "d":
            direction = right
        elif key == "q":
            direction = (0.0, -1.0, 0.0)
        else:
            direction = (0.0, 1.0, 0.0)
        self.camera.x += direction[0] * amount
        self.camera.y += direction[1] * amount
        self.camera.z += direction[2] * amount

    def on_key_press(self, event: tk.Event) -> None:
        key = event.keysym.lower()
        if key in {"w", "a", "s", "d", "q", "e"}:
            self.pressed.add(key)
        elif key == "r":
            self.camera = self.initial_camera.copy()
            self.mark_camera_changed()
        elif key in {"1", "2", "3", "4", "5"}:
            self.set_integrator(int(key) - 1)
        elif key == "b":
            self.set_backend(1 - self.backend_index)
        elif key in {"plus", "equal"}:
            self.options.batches = min(64, self.options.batches + 1)
            self.batches_var.set(str(self.options.batches))
            self.request_render()
        elif key in {"minus", "underscore"}:
            self.options.batches = max(1, self.options.batches - 1)
            self.batches_var.set(str(self.options.batches))
            self.request_render()

    def on_key_release(self, event: tk.Event) -> None:
        self.pressed.discard(event.keysym.lower())

    def set_integrator(self, index: int) -> None:
        self._set_choice(
            index, INTEGRATORS, "integrator_index", self.integrator_var
        )

    def on_integrator_changed(self, value: str) -> None:
        if value in INTEGRATORS:
            self.set_integrator(INTEGRATORS.index(value))

    def set_backend(self, index: int) -> None:
        if index < 0 or index >= len(BACKENDS):
            return
        if index == self.backend_index:
            return
        self.backend_index = index
        self.backend_var.set(BACKENDS[index])
        self._update_policy_controls()
        self.request_render()

    def on_backend_changed(self, value: str) -> None:
        if value in BACKENDS:
            self.set_backend(BACKENDS.index(value))

    def _update_policy_controls(self) -> None:
        state = tk.NORMAL if self.backend_index == 0 else tk.DISABLED
        self.traversal_menu.configure(state=state)
        self.build_menu.configure(state=state)

    def _set_choice(
        self,
        index: int,
        choices: tuple[str, ...],
        index_attribute: str,
        variable: tk.StringVar,
    ) -> None:
        if index < 0 or index >= len(choices):
            return
        if index == getattr(self, index_attribute):
            return
        setattr(self, index_attribute, index)
        variable.set(choices[index])
        self.request_render()

    def set_traversal(self, index: int) -> None:
        self._set_choice(
            index, TRAVERSALS, "traversal_index", self.traversal_var
        )

    def on_traversal_changed(self, value: str) -> None:
        if value in TRAVERSALS:
            self.set_traversal(TRAVERSALS.index(value))

    def set_build(self, index: int) -> None:
        self._set_choice(index, BUILDERS, "build_index", self.build_var)

    def on_build_changed(self, value: str) -> None:
        if value in BUILDERS:
            self.set_build(BUILDERS.index(value))

    def set_sampler(self, index: int) -> None:
        self._set_choice(index, SAMPLERS, "sampler_index", self.sampler_var)

    def on_sampler_changed(self, value: str) -> None:
        if value in SAMPLERS:
            self.set_sampler(SAMPLERS.index(value))

    def set_scene(self, index: int) -> None:
        if index < 0 or index >= len(SCENES):
            return
        if index == CUSTOM_PBRT_SCENE:
            self.choose_pbrt()
            return
        if index == self.scene_index:
            return
        self.scene_index = index
        if index == BUILTIN_PBRT_SCENE:
            self.pbrt_path = str(BUILTIN_PBRT_PATH)
            try:
                self.camera = camera_from_pbrt(
                    self.renderer.pbrt_camera(self.pbrt_path)
                )
            except Exception as exc:
                self.pbrt_path = None
                self.scene_index = 0
                self.scene_var.set(SCENES[0])
                self.update_status(f"PBRT scene error: {exc}")
                return
        else:
            self.pbrt_path = None
            self.camera = default_camera(index)
        self.initial_camera = self.camera.copy()
        self.scene_var.set(SCENES[index])
        self.mark_camera_changed()

    def on_scene_changed(self, value: str) -> None:
        if value in SCENES:
            self.set_scene(SCENES.index(value))

    def on_max_spp_changed(self, _event=None) -> None:
        try:
            value = int(self.max_spp_var.get())
        except ValueError:
            self.max_spp_var.set(str(self.options.max_samples))
            return
        if value <= 0:
            self.max_spp_var.set(str(self.options.max_samples))
            return
        if value == self.options.max_samples:
            return
        self.options.max_samples = value
        self.max_spp_var.set(str(value))
        self.request_render()

    def on_max_depth_changed(self, _event=None) -> None:
        try:
            value = int(self.max_depth_var.get())
        except ValueError:
            self.max_depth_var.set(str(self.options.max_depth))
            return
        if value < 1 or value > 16:
            self.max_depth_var.set(str(self.options.max_depth))
            return
        if value == self.options.max_depth:
            return
        self.options.max_depth = value
        self.max_depth_var.set(str(value))
        self.request_render()

    def on_batches_changed(self, _event=None) -> None:
        try:
            value = int(self.batches_var.get())
        except ValueError:
            self.batches_var.set(str(self.options.batches))
            return
        if value <= 0:
            self.batches_var.set(str(self.options.batches))
            return
        if value == self.options.batches:
            return
        self.options.batches = value
        self.batches_var.set(str(value))
        self.request_render()

    def choose_pbrt(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title="Open PBRT scene",
            filetypes=[("PBRT scenes", "*.pbrt *.pbrt-v4"), ("All files", "*")],
        )
        if not path:
            current = (
                f"PBRT:{Path(self.pbrt_path).name}" if self.scene_index
                == CUSTOM_PBRT_SCENE
                and self.pbrt_path else SCENES[self.scene_index]
            )
            self.scene_var.set(current)
            return
        try:
            camera = camera_from_pbrt(self.renderer.pbrt_camera(path))
        except Exception as exc:
            self.update_status(f"PBRT load error: {exc}")
            self.scene_var.set(SCENES[self.scene_index])
            return
        self.pbrt_path = path
        self.scene_index = CUSTOM_PBRT_SCENE
        self.camera = camera
        self.initial_camera = camera.copy()
        self.scene_var.set(SCENES[self.scene_index])
        self.mark_camera_changed()

    def tick(self) -> None:
        if self.closed:
            return
        now = time.monotonic()
        dt = min(0.1, now - self.last_tick)
        self.last_tick = now
        if self.pressed:
            changed = False
            for key in self.pressed:
                self.key_move(key, 4.0 * dt)
                changed = True
            if changed:
                self.mark_camera_changed()
        else:
            with self.lock:
                ready_to_accumulate = (
                    not self.dragging
                    and now - self.last_camera_change >= SETTLE_DELAY_SECONDS
                    and not self.rendering
                    and self.accumulated_spp < self.options.max_samples
                )
            if ready_to_accumulate:
                self.request_render(reset_accumulation=False)
        self.root.after(40, self.tick)

    def on_mouse_down(self, event: tk.Event) -> None:
        self.drag_anchor = (event.x, event.y)
        self.dragging = True
        self.image_canvas.focus_set()

    def on_mouse_drag(self, event: tk.Event) -> None:
        if self.drag_anchor is None:
            return
        old_x, old_y = self.drag_anchor
        dx, dy = event.x - old_x, event.y - old_y
        self.drag_anchor = (event.x, event.y)
        self.camera.yaw += dx * 0.35
        self.camera.pitch = max(-89.0, min(89.0, self.camera.pitch - dy * 0.35))
        self.mark_camera_changed()

    def on_mouse_up(self, _event: tk.Event) -> None:
        self.drag_anchor = None
        self.dragging = False

    def close(self) -> None:
        self.closed = True
        if not self.rendering:
            self._destroy_gpu_state()
        self.temp_dir.cleanup()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=214)
    parser.add_argument(
        "--batches",
        type=int,
        default=4,
        help="number of progressive batches (default: 4)",
    )
    parser.add_argument(
        "--max-spp",
        type=int,
        default=32,
        help="total samples accumulated at one camera pose (default: 32)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=8,
        help="maximum path depth (runtime setting, default: 8)",
    )
    parser.add_argument(
        "--scene",
        choices=tuple(SCENE_INDEX_BY_CLI),
        default="rtiaw",
        help="initial scene (default: rtiaw)",
    )
    parser.add_argument(
        "--backend",
        choices=("cpu", "gpu"),
        default="cpu",
        help="rendering backend (default: cpu)",
    )
    parser.add_argument(
        "--traversal",
        choices=TRAVERSAL_CLI,
        default="auto",
        help="initial CPU traversal policy (default: auto)",
    )
    parser.add_argument(
        "--build",
        choices=BUILDER_CLI,
        default="sah",
        help="initial CPU BVH builder (default: sah)",
    )
    parser.add_argument(
        "--sampler",
        choices=SAMPLER_CLI,
        default="independent",
        help="sample sequence (default: independent)",
    )
    parser.add_argument(
        "--gpu-arch",
        default=os.environ.get("BAJO_GPU_ARCH", "sm_120"),
        help="Mojo GPU target architecture (default: sm_120; or BAJO_GPU_ARCH)",
    )
    parser.add_argument(
        "--pbrt",
        type=Path,
        help="open a PBRT scene at startup",
    )
    args = parser.parse_args()
    if (
        args.width <= 0
        or args.height <= 0
        or args.batches <= 0
        or args.max_spp <= 0
        or args.max_depth < 1
        or args.max_depth > 16
    ):
        parser.error(
            "width, height, batches, max-spp, and max-depth must be valid"
        )
    scene_index = SCENE_INDEX_BY_CLI[args.scene]
    if args.pbrt is not None:
        scene_index = CUSTOM_PBRT_SCENE
    Viewer(
        RenderOptions(
            args.width, args.height, args.batches, args.max_spp, args.max_depth
        ),
        scene_index,
        str(args.pbrt) if args.pbrt is not None else None,
        BACKENDS.index(args.backend.upper()),
        TRAVERSAL_CLI.index(args.traversal),
        BUILDER_CLI.index(args.build),
        SAMPLER_CLI.index(args.sampler),
        args.gpu_arch,
    ).run()


if __name__ == "__main__":
    main()
