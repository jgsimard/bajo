"""Configuration and state model for the Bajo interactive viewer."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path


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
    moving_samples: int = 1


@dataclass
class GpuState:
    renderer: object
    handle: int
    tag: int
    key: tuple[object, ...]
    bvh_stats: str
    parse_ms: float
    bvh_ms: float


@dataclass
class CpuState:
    renderer: object
    handle: int
    key: tuple[object, ...]
    bvh_stats: str
    parse_ms: float
    bvh_ms: float


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
    parse_ms: float
    bvh_ms: float
    init_ms: float
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
