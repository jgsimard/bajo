"""Scene selection and loading for native viewer frontends."""

from bajo.core.camera import Camera
from bajo.bvh.cpu import CpuBvhBuildMethod, CpuTraversalMode
from bajo.core import Point3W
from bajo.rt.material_types import Integrator
from bajo.rt.render_types import RenderSettings
from bajo.rt.scene_data import SceneData
from bajo.rt.scene_description import SceneDescription
from .core import viewer_camera
from examples.rtiaw import make_weekend_scene
from examples.cornell_box import make_cornell_scene
from examples.mis_showcase import make_mis_showcase_scene
from examples.lbvh_scene import make_lbvh_scene
from examples.emissive_instances import make_emissive_instance_scene
from examples.stress_scenes import (
    make_indirect_hall_scene,
    make_many_lights_scene,
    make_specular_transport_scene,
)
from bajo.parser.pbrt import read_pbrt
from bajo.parser.asset_loader import AssetLoader, PathAssetLoader


# PBRT entries follow the procedural scenes.
comptime _SCENE_NAMES: List[String] = [
    "rtiaw",
    "cornell",
    "veach",
    "lbvh",
    "emissive-instance",
    "many-lights",
    "indirect-hall",
    "specular-transport",
    "pbrt",
    "custom PBRT",
]
comptime CUSTOM_SCENE = len(_SCENE_NAMES) - 1
comptime BUILTIN_SCENE_COUNT = CUSTOM_SCENE
comptime BUILTIN_PBRT_SCENE = BUILTIN_SCENE_COUNT - 1
comptime BUILTIN_PBRT_PATH = "examples/scenes/pbrt_showcase.pbrt"


def viewer_scene_name(scene: Int) raises -> String:
    var names = materialize[_SCENE_NAMES]()
    if scene >= 0 and scene < len(names):
        return names[scene]
    raise Error("unknown viewer scene")


def viewer_scene_index(name: String) raises -> Int:
    var names = materialize[_SCENE_NAMES]()
    for index in range(BUILTIN_SCENE_COUNT):
        if names[index] == name:
            return index
    raise Error("unknown viewer scene: " + name)


def default_viewer_camera(scene: Int) raises -> Camera:
    """Return the existing built-in presets; PBRT supplies its own camera."""
    match scene:
        case 0:
            return viewer_camera(Point3W(13, 2, 3), -77, -8.5, 28)
        case 1:
            return viewer_camera(Point3W(0, 1, 3.2), 0, 0, 28)
        case 2:
            return viewer_camera(Point3W(0, 3, 6.2), 0, -12, 31)
        case 3:
            return viewer_camera(Point3W(0, 6, -28), 180, -8, 35)
        case 4:
            return viewer_camera(Point3W(0, 1.6, 5.8), 0, -7, 42)
        case 5:
            return viewer_camera(Point3W(0, 4.3, 11), 0, -12, 52)
        case 6:
            return viewer_camera(Point3W(0, 2.2, 8), 0, -2, 55)
        case 7:
            return viewer_camera(Point3W(0, 2.7, 9.5), 0, -8, 48)
        case _:
            raise Error("scene has no built-in camera preset")


def load_viewer_scene(
    scene: Int, scene_path: String = ""
) raises -> SceneDescription:
    var loader = PathAssetLoader()
    return load_viewer_scene(scene, scene_path, loader)


def load_viewer_scene[
    Loader: AssetLoader
](scene: Int, scene_path: String, loader: Loader) raises -> SceneDescription:
    """Load once, retaining authored PBRT camera, film, and sampling settings.
    """
    if scene == BUILTIN_PBRT_SCENE or scene == CUSTOM_SCENE:
        var path = scene_path
        if path == "":
            if scene == CUSTOM_SCENE:
                raise Error("custom PBRT scene requires a path")
            path = BUILTIN_PBRT_PATH
        return read_pbrt(path, loader)
    var data: SceneData
    match scene:
        case 0:
            data = make_weekend_scene()
        case 1:
            data = make_cornell_scene()
        case 2:
            data = make_mis_showcase_scene()
        case 3:
            data = make_lbvh_scene()
        case 4:
            data = make_emissive_instance_scene()
        case 5:
            data = make_many_lights_scene()
        case 6:
            data = make_indirect_hall_scene()
        case 7:
            data = make_specular_transport_scene()
        case _:
            raise Error("unsupported viewer scene")
    return SceneDescription(
        data^,
        default_viewer_camera(scene),
        RenderSettings(320, 214, 32, UInt64(1234), 8),
        Integrator.PATH,
    )


def load_viewer_scene_data(
    scene: Int, scene_path: String = ""
) raises -> SceneData:
    """Compatibility entry point for frontends that supply their own settings.
    """
    var description = load_viewer_scene(scene, scene_path)
    return description^.take_data()


def viewer_bvh_stats[
    BACKEND: Int,
    world_bvh_width: SIMDLength = 16,
    instance_bvh_width: SIMDLength = 16,
    build_method: CpuBvhBuildMethod = .SAH,
    traversal_mode: CpuTraversalMode = .AUTO_COHERENT,
](data: SceneData, include_traversal: Bool = True) -> String:
    var result: String
    if BACKEND == 0:
        result = "CPU W" + String(Int(world_bvh_width))
        result += "/I" + String(Int(instance_bvh_width)) + " | "
        if len(data.spheres()) > 0:
            result += (
                "sphere"
                + String(Int(world_bvh_width))
                + "/"
                + build_method.name()
            )
        if len(data.triangle_vertices()) > 0:
            if len(data.spheres()) > 0:
                result += " "
            result += (
                "tri" + String(Int(world_bvh_width)) + "/" + build_method.name()
            )
        if len(data.triangle_instances()) > 0:
            if len(data.spheres()) > 0 or len(data.triangle_vertices()) > 0:
                result += " "
            result += (
                "BLAS"
                + String(Int(instance_bvh_width))
                + "/"
                + build_method.name()
                + " TLAS"
                + String(Int(instance_bvh_width))
                + "/1 LBVH"
            )
        if include_traversal:
            comptime if traversal_mode == .ADAPTIVE:
                result += " | traversal adaptive-16-8-4-scalar"
            else:
                result += " | traversal " + traversal_mode.name()
    else:
        result = "GPU | "
        if len(data.spheres()) > 0:
            result += "sphere4/4 LBVH"
        if len(data.triangle_vertices()) > 0:
            if len(data.spheres()) > 0:
                result += " "
            result += "tri8/4 H-PLOC CWBVH8"
        if len(data.triangle_instances()) > 0:
            if len(data.spheres()) > 0 or len(data.triangle_vertices()) > 0:
                result += " "
            result += "BLAS8/4 H-PLOC CWBVH8"
            result += " TLAS2/1 LBVH"
    result += " | S" + String(len(data.spheres()))
    result += " T" + String(len(data.triangle_vertices()) // 3)
    result += " I" + String(len(data.triangle_instances()))
    return result
