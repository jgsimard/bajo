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
from PIL import Image, ImageTk


ROOT = Path(__file__).resolve().parents[1]
BUILTIN_PBRT_PATH = ROOT / "examples" / "scenes" / "pbrt_showcase.pbrt"
LBVH_SCENE = 3
BUILTIN_PBRT_SCENE = 4
CUSTOM_PBRT_SCENE = 5
SETTLE_DELAY_SECONDS = 0.25
PREVIEW_INTERVAL_SECONDS = 0.08
ALGORITHMS = ("PATH", "NEE", "MIS", "NORMALS", "AO")
BACKENDS = ("CPU", "GPU")
SCENES = (
    "RTIAW",
    "CORNELL",
    "VEACH",
    "LBVH MESHES",
    "PBRT MESHES",
    "LOAD PBRT…",
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


@dataclass
class RenderOptions:
    width: int = 320
    height: int = 214
    samples: int = 4
    max_samples: int = 32
    max_depth: int = 8


def default_camera(scene_index: int) -> Camera:
    if scene_index == 1:
        return Camera(x=0.0, y=1.0, z=3.2, yaw=0.0, pitch=0.0, vfov=28.0)
    if scene_index == 2:
        return Camera(x=0.0, y=3.0, z=6.2, yaw=0.0, pitch=-12.0, vfov=31.0)
    if scene_index == LBVH_SCENE:
        return Camera(x=0.0, y=6.0, z=-28.0, yaw=180.0, pitch=-8.0, vfov=35.0)
    return Camera()


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
        self.algorithm_index = 0
        self.last_tick = time.monotonic()
        self.last_camera_change = self.last_tick
        self.render_generation = 0
        self.accumulated_spp = 0
        self.display_times: deque[float] = deque()
        self.rendering = False
        self.closed = False
        self.lock = threading.Lock()
        self.image: Image.Image | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.temp_dir = tempfile.TemporaryDirectory(prefix="bajo-viewer-")
        self.output_path = Path(self.temp_dir.name) / "frame.ppm"
        self.renderer = self._load_renderer(
            self.backend_index,
            self.algorithm_index,
        )
        self.renderer_config = (self.backend_index, self.algorithm_index)
        if self.scene_index in (BUILTIN_PBRT_SCENE, CUSTOM_PBRT_SCENE) and self.pbrt_path:
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
            text="Algorithm:",
            padx=10,
            pady=4,
            fg="#dddddd",
            bg="#202020",
        ).pack(side=tk.LEFT)
        self.algorithm_var = tk.StringVar(value=ALGORITHMS[self.algorithm_index])
        tk.OptionMenu(
            toolbar,
            self.algorithm_var,
            *ALGORITHMS,
            command=self.on_algorithm_changed,
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
            text="SPP:",
            padx=6,
            pady=4,
            fg="#dddddd",
            bg="#202020",
        ).pack(side=tk.LEFT)
        self.spp_var = tk.StringVar(value=str(self.options.samples))
        self.spp_spinbox = tk.Spinbox(
            toolbar,
            from_=1,
            to=1_000_000,
            width=7,
            textvariable=self.spp_var,
            command=self.on_spp_changed,
        )
        self.spp_spinbox.pack(side=tk.LEFT, padx=(0, 8), pady=2)
        self.spp_spinbox.bind("<Return>", self.on_spp_changed)
        self.spp_spinbox.bind("<FocusOut>", self.on_spp_changed)
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

        self.root.bind("<KeyPress>", self.on_key_press)
        self.root.bind("<KeyRelease>", self.on_key_release)
        self.root.bind("<Escape>", lambda _event: self.close())
        self.image_canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.image_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.image_canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.image_canvas.bind("<Configure>", lambda _event: self.refresh_image())
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.request_render()
        self.root.after(40, self.tick)

    def _load_renderer(self, backend_index: int, algorithm_index: int):
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
            f"-a{algorithm_index}.so"
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
                f"VIEWER_ALGORITHM={algorithm_index}",
                "--emit",
                "shared-lib",
                "-o",
                str(cache_path),
                str(ROOT / "bajo_viewer.mojo"),
            ]
            if backend_index == 1:
                command[command.index("--emit"):command.index("--emit")] = [
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
                    f"could not build viewer for backend {backend_index}, algorithm {algorithm_index}: {details}"
                )

        spec = importlib.util.spec_from_file_location("bajo_viewer", cache_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load Mojo viewer extension: {cache_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["bajo_viewer"] = module
        spec.loader.exec_module(module)
        return module

    def snapshot(self) -> tuple[Camera, RenderOptions, int, int, bool, int, int]:
        with self.lock:
            options = RenderOptions(
                self.options.width,
                self.options.height,
                self.options.samples,
                self.options.max_samples,
                self.options.max_depth,
            )
            preview = (
                self.dragging
                or bool(self.pressed)
                or time.monotonic() - self.last_camera_change
                < SETTLE_DELAY_SECONDS
            )
            target_spp = 1 if preview else min(
                self.accumulated_spp + options.samples,
                options.max_samples,
            )
            return (
                self.camera.copy(),
                options,
                self.render_generation,
                target_spp,
                preview,
                self.algorithm_index,
                self.backend_index,
            )

    def mark_camera_changed(self) -> None:
        with self.lock:
            self.last_camera_change = time.monotonic()
        self.request_render()

    def request_render(self, reset_accumulation: bool = True) -> None:
        with self.lock:
            if reset_accumulation:
                self.accumulated_spp = 0
            self.render_generation += 1
            should_start = not self.rendering
            if should_start:
                self.rendering = True
        if should_start:
            threading.Thread(target=self.render_worker, daemon=True).start()
        self.update_status("Rendering ....")

    def render_worker(self) -> None:
        while not self.closed:
            (
                camera,
                options,
                generation,
                target_spp,
                preview,
                algorithm_index,
                backend_index,
            ) = self.snapshot()
            started = time.monotonic()
            try:
                renderer = self.renderer
                requested_config = (backend_index, algorithm_index)
                if self.renderer_config != requested_config:
                    self.root.after(
                        0,
                        lambda backend=backend_index, algorithm=algorithm_index: self.update_status(
                            f"Compiling {BACKENDS[backend]} / "
                            f"{ALGORITHMS[algorithm]} ..... "
                        ),
                    )
                    renderer = self._load_renderer(
                        backend_index,
                        algorithm_index,
                    )
                    with self.lock:
                        if generation == self.render_generation:
                            self.renderer = renderer
                            self.renderer_config = requested_config
                        else:
                            continue
                render_stats = renderer.render_frame(
                    {
                        "output": str(self.output_path),
                        "width": options.width,
                        "height": options.height,
                        "samples": target_spp,
                        "max_depth": options.max_depth,
                        "scene": self.scene_index,
                        "scene_path": self.pbrt_path or "",
                        "x": camera.x,
                        "y": camera.y,
                        "z": camera.z,
                        "yaw": camera.yaw,
                        "pitch": camera.pitch,
                        "vfov": camera.vfov,
                    }
                )
                render_ms = float(render_stats[0])
                build_ms = float(render_stats[1])
                mrays = float(render_stats[2])
                error = None
            except Exception as exc:  # surface Mojo/Python errors in the UI
                render_ms = None
                build_ms = None
                mrays = None
                error = str(exc).strip().splitlines()[-1]
            elapsed_ms = (time.monotonic() - started) * 1000.0

            with self.lock:
                latest = generation == self.render_generation
                if latest:
                    self.accumulated_spp = target_spp

            if error is not None:
                if latest:
                    self.root.after(
                        0, lambda error=error: self.update_status(f"Error: {error}")
                    )
                    with self.lock:
                        self.rendering = False
                    return
            elif latest:
                try:
                    # Read the completed frame before starting the next Mojo
                    # pass, since all progressive passes share this path.
                    frame = Image.open(self.output_path).convert("RGB")
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
                self.root.after(
                    0,
                    lambda frame=frame, elapsed_ms=elapsed_ms, render_ms=render_ms, build_ms=build_ms, mrays=mrays, target_spp=target_spp, max_depth=options.max_depth: self.show_frame(
                        frame,
                        elapsed_ms,
                        render_ms,
                        build_ms,
                        mrays,
                        target_spp,
                        options.max_samples,
                        max_depth,
                        preview,
                        algorithm_index,
                        backend_index,
                    ),
                )

                if preview:
                    with self.lock:
                        pending = generation != self.render_generation
                        active_motion = self.dragging or bool(self.pressed)
                        if not pending and not active_motion:
                            self.rendering = False
                            return
                    time.sleep(PREVIEW_INTERVAL_SECONDS)
                elif target_spp >= options.max_samples:
                    with self.lock:
                        # The current pose has reached its accumulation cap.
                        # A later camera change will set this back to zero.
                        if generation == self.render_generation:
                            self.rendering = False
                            return

            # The camera is unchanged: immediately render the next sample
            # batch. A newer camera generation will be picked up here and has
            # already reset accumulated_spp to zero.

    def show_frame(
        self,
        frame: Image.Image,
        elapsed_ms: float,
        render_ms: float,
        build_ms: float,
        mrays: float,
        accumulated_spp: int,
        max_spp: int,
        max_depth: int,
        preview: bool,
        algorithm_index: int,
        backend_index: int,
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
            fps = (len(self.display_times) - 1) / duration if duration > 0 else 0.0
        else:
            fps = 0.0
        spp_text = (
            "preview"
            if preview
            else f"accumulated {accumulated_spp}/{max_spp} spp"
        )
        algorithm_name = ALGORITHMS[algorithm_index]
        backend_name = BACKENDS[backend_index]
        scene_name = (
            f"PBRT:{Path(self.pbrt_path).name}"
            if self.scene_index == CUSTOM_PBRT_SCENE and self.pbrt_path
            else SCENES[self.scene_index]
        )
        self.refresh_image()
        self.update_status(
            f"{self.image.width}×{self.image.height}  |  "
            f"FPS {fps:.1f}  |  "
            f"{backend_name}  |  "
            f"{algorithm_name}  |  "
            f"{scene_name}  |  "
            f"Depth {max_depth}  |  "
            f"Build {float(build_ms):.1f} ms  |  "
            f"Render {float(render_ms):.2f} ms  |  "
            f"{float(mrays):.2f} MRays/s  |  "
            f"Wall {elapsed_ms:.2f}ms  |  "
            f"{spp_text}"
        )

    def refresh_image(self) -> None:
        if self.image is None or self.closed:
            return
        available_w = max(1, self.image_canvas.winfo_width())
        available_h = max(1, self.image_canvas.winfo_height())
        scale = min(available_w / self.image.width, available_h / self.image.height)
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
        if key == "w": direction = forward
        elif key == "s": direction = tuple(-v for v in forward)
        elif key == "a": direction = tuple(-v for v in right)
        elif key == "d": direction = right
        elif key == "q": direction = (0.0, -1.0, 0.0)
        else: direction = (0.0, 1.0, 0.0)
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
            self.set_algorithm(int(key) - 1)
        elif key == "b":
            self.set_backend(1 - self.backend_index)
        elif key in {"plus", "equal"}:
            self.options.samples = min(64, self.options.samples + 1)
            self.spp_var.set(str(self.options.samples))
            self.request_render()
        elif key in {"minus", "underscore"}:
            self.options.samples = max(1, self.options.samples - 1)
            self.spp_var.set(str(self.options.samples))
            self.request_render()

    def on_key_release(self, event: tk.Event) -> None:
        self.pressed.discard(event.keysym.lower())

    def set_algorithm(self, index: int) -> None:
        if index < 0 or index >= len(ALGORITHMS):
            return
        if index == self.algorithm_index:
            return
        self.algorithm_index = index
        self.algorithm_var.set(ALGORITHMS[index])
        self.request_render()

    def on_algorithm_changed(self, value: str) -> None:
        if value in ALGORITHMS:
            self.set_algorithm(ALGORITHMS.index(value))

    def set_backend(self, index: int) -> None:
        if index < 0 or index >= len(BACKENDS):
            return
        if index == self.backend_index:
            return
        self.backend_index = index
        self.backend_var.set(BACKENDS[index])
        self.request_render()

    def on_backend_changed(self, value: str) -> None:
        if value in BACKENDS:
            self.set_backend(BACKENDS.index(value))

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

    def on_spp_changed(self, _event=None) -> None:
        try:
            value = int(self.spp_var.get())
        except ValueError:
            self.spp_var.set(str(self.options.samples))
            return
        if value <= 0:
            self.spp_var.set(str(self.options.samples))
            return
        if value == self.options.samples:
            return
        self.options.samples = value
        self.spp_var.set(str(value))
        self.request_render()

    def choose_pbrt(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title="Open PBRT scene",
            filetypes=[("PBRT scenes", "*.pbrt *.pbrt-v4"), ("All files", "*")],
        )
        if not path:
            current = (
                f"PBRT:{Path(self.pbrt_path).name}"
                if self.scene_index == CUSTOM_PBRT_SCENE and self.pbrt_path
                else SCENES[self.scene_index]
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
        self.temp_dir.cleanup()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=214)
    parser.add_argument("--spp", type=int, default=4)
    parser.add_argument(
        "--max-spp",
        type=int,
        default=32,
        help="maximum total samples accumulated at one camera pose (default: 256)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=8,
        help="maximum path depth (runtime setting, default: 8)",
    )
    parser.add_argument(
        "--scene",
        choices=("rtiaw", "cornell", "veach", "lbvh", "pbrt"),
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
        or args.spp <= 0
        or args.max_spp <= 0
        or args.max_depth < 1
        or args.max_depth > 16
    ):
        parser.error("width, height, spp, max-spp, and max-depth must be valid")
    scene_index = {
        "rtiaw": 0,
        "cornell": 1,
        "veach": 2,
        "lbvh": LBVH_SCENE,
        "pbrt": BUILTIN_PBRT_SCENE,
    }[args.scene]
    if args.pbrt is not None:
        scene_index = CUSTOM_PBRT_SCENE
    Viewer(
        RenderOptions(
            args.width, args.height, args.spp, args.max_spp, args.max_depth
        ),
        scene_index,
        str(args.pbrt) if args.pbrt is not None else None,
        BACKENDS.index(args.backend.upper()),
        args.gpu_arch,
    ).run()


if __name__ == "__main__":
    main()
