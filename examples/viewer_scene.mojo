"""Inspect the native viewer core without opening a window or using Python."""

from std.sys.arg import argv
from bajo.viewer import ViewerScene, load_viewer_scene, viewer_scene_index


def main() raises:
    var args = argv()
    var scene_index = viewer_scene_index("cornell")
    var path = String("")
    var prepare_cpu = False
    var index = 1
    while index < len(args):
        var option = String(args[index])
        match option:
            case "--scene" | "--pbrt":
                if index + 1 >= len(args):
                    raise Error("missing value for " + option)
                index += 1
                if option == "--scene":
                    scene_index = viewer_scene_index(String(args[index]))
                    path = ""
                else:
                    scene_index = 9
                    path = String(args[index])
            case "--prepare-cpu":
                prepare_cpu = True
            case "--help":
                print(
                    "usage: viewer_scene [--scene NAME | --pbrt PATH]"
                    " [--prepare-cpu]"
                )
                return
            case _:
                raise Error("unknown option: " + option)
        index += 1
    var scene = ViewerScene(load_viewer_scene(scene_index, path))
    if prepare_cpu:
        scene.prepare_cpu()
    print("Spheres:", len(scene.scene_data().spheres()))
    print("Triangles:", len(scene.scene_data().triangle_vertices()) // 3)
    print("Meshes:", len(scene.scene_data().triangle_meshes()))
    print("Instances:", len(scene.scene_data().triangle_instances()))
    print("Camera:", scene.state.camera.origin)
    print("Render settings:", scene.state.settings)
    print("CPU prepared:", scene.has_cpu_scene())
