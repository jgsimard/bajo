"""The native Mojo/raylib viewer: raster presentation and Bajo tracing."""

from std.math import sin, cos, sqrt, min, max
from std.sys.arg import argv
from std.ffi import c_int
from bajo.bvh import Camera
from bajo.core import Vec3W, cross, dot, normalize
from bajo.core.random import Sampler
from bajo.rt.render_types import Color as LinearColor
from bajo.rt.image_output import color_to_byte
from .core import ViewerScene
from .scenes import load_viewer_scene, viewer_scene_index
from .raylib import Window, Color, BoundingBox, Vector3, Frame
from .raster import RasterScene, DisplayTexture, raster_camera
from .worker import RenderWorker
from .debug import DebugBox, object_boxes


struct ViewerOptions(Copyable):
    var scene: Int
    var path: String
    var library: String
    var width: Int
    var height: Int
    var max_samples: Int
    var moving_samples: Int
    var max_depth: Int
    var integrator: Int
    var backend: Int
    var sampler: Int
    var build_method: Int
    var traversal: Int
    var traced: Bool
    var hidden: Bool
    var frame_limit: Int
    var screenshot: String
    var bounds: Int
    var bvh_depth: Int
    var help: Bool

    def __init__(out self):
        self.scene = 0
        self.path = ""
        self.library = ".pixi/raylib/lib/libraylib.so"
        self.width = 0
        self.height = 0
        self.max_samples = 0
        self.moving_samples = 1
        self.max_depth = -1
        self.integrator = 0
        self.backend = 0
        self.sampler = -1
        self.build_method = 0
        self.traversal = 0
        self.traced = False
        self.hidden = False
        self.frame_limit = 0
        self.screenshot = ""
        self.bounds = 0
        self.bvh_depth = 2
        self.help = False


def _choice(value: String, choices: List[String]) raises -> Int:
    for i in range(len(choices)):
        if value == choices[i]:
            return i
    raise Error("unsupported value: " + value)


def parse_options() raises -> ViewerOptions:
    var options = ViewerOptions()
    var args = argv()
    var index = 1
    while index < len(args):
        var option = String(args[index])
        if option == "--help":
            options.help = True
            return options^
        if option == "--hidden":
            options.hidden = True
            index += 1
            continue
        if index + 1 >= len(args):
            raise Error("missing value for " + option)
        index += 1
        var value = String(args[index])
        match option:
            case "--scene":
                options.scene = viewer_scene_index(value)
            case "--pbrt":
                options.scene = 9
                options.path = value
            case "--raylib-library":
                options.library = value
            case "--width":
                options.width = Int(value)
            case "--height":
                options.height = Int(value)
            case "--max-spp":
                options.max_samples = Int(value)
            case "--moving-spp":
                options.moving_samples = Int(value)
            case "--max-depth":
                options.max_depth = Int(value)
            case "--frames":
                options.frame_limit = Int(value)
            case "--screenshot":
                options.screenshot = value
            case "--mode":
                options.traced = Bool(_choice(value, ["raster", "trace"]))
            case "--backend":
                options.backend = _choice(value, ["cpu", "gpu"])
            case "--integrator":
                options.integrator = _choice(
                    value, ["path", "nee", "mis", "normals", "ao"]
                )
            case "--sampler":
                options.sampler = _choice(
                    value,
                    ["independent", "halton", "r2", "sobol", "sz", "stbn"],
                )
            case "--build":
                options.build_method = _choice(
                    value, ["sah", "lbvh", "hploc", "median"]
                )
            case "--traversal":
                options.traversal = _choice(
                    value, ["auto", "fixed", "adaptive"]
                )
            case "--bounds":
                options.bounds = _choice(
                    value, ["off", "objects", "bvh", "both"]
                )
            case "--bvh-depth":
                options.bvh_depth = Int(value)
            case _:
                raise Error("unknown option: " + option)
        index += 1
    if (
        options.width < 0
        or options.height < 0
        or options.max_samples < 0
        or options.moving_samples < 1
        or options.max_depth < -1
        or options.max_depth > 32
        or options.frame_limit < 0
        or options.bvh_depth < 0
    ):
        raise Error("invalid viewer dimensions, samples, depth, or frame count")
    return options^


def print_help():
    print("Bajo native Mojo/raylib viewer")
    print(
        "--scene"
        " rtiaw|cornell|veach|lbvh|emissive-instance|many-lights|indirect-hall|specular-transport|pbrt"
    )
    print("--pbrt PATH  --mode raster|trace  --backend cpu|gpu")
    print("--width N --height N --max-spp N --moving-spp N --max-depth N")
    print(
        "--integrator path|nee|mis|normals|ao --sampler"
        " independent|halton|r2|sobol|sz|stbn"
    )
    print("--build sah|lbvh|hploc|median --traversal auto|fixed|adaptive")
    print("--bounds off|objects|bvh|both --bvh-depth N")
    print("--raylib-library PATH --hidden --frames N --screenshot PATH")
    print(
        "WASD/QE move; mouse drag looks; Shift accelerates; R resets; F"
        " fullscreen"
    )
    print(
        "Tab raster/trace; G CPU/GPU; 1-5 integrator; P sampler; C next scene"
    )
    print(
        "B bounds; [/] BVH depth; +/- maximum samples; ,/. path depth; Esc"
        " closes"
    )


def rotate_camera(mut camera: Camera, yaw: Float32, pitch: Float32):
    """Rotate the shared basis; preserve camera roll, lens size and FOV."""
    var angle = yaw * 0.006
    var c = cos(angle)
    var s = sin(angle)
    var forward = Vec3W(
        c * camera.forward.x + s * camera.forward.z,
        camera.forward.y,
        -s * camera.forward.x + c * camera.forward.z,
    )
    var right = Vec3W(
        c * camera.right.x + s * camera.right.z,
        camera.right.y,
        -s * camera.right.x + c * camera.right.z,
    )
    angle = pitch * 0.006
    var candidate = forward * cos(angle) + cross(right, forward) * sin(angle)
    if abs(candidate.y) < 0.9998:
        forward = candidate
    var lens_u = sqrt(dot(camera.defocus_disk_u, camera.defocus_disk_u))
    var lens_v = sqrt(dot(camera.defocus_disk_v, camera.defocus_disk_v))
    camera.forward = normalize(forward)
    camera.right = normalize(right)
    camera.up = normalize(cross(camera.right, camera.forward))
    camera.defocus_disk_u = camera.right * lens_u
    camera.defocus_disk_v = camera.up * lens_v


def _draw_boxes(
    frame: Frame, boxes: List[DebugBox], color: Color, depth: Int = 100000
) raises:
    for box in boxes:
        if box.depth <= depth:
            frame.draw_box(
                BoundingBox(
                    Vector3(box.minimum.x, box.minimum.y, box.minimum.z),
                    Vector3(box.maximum.x, box.maximum.y, box.maximum.z),
                ),
                color,
            )


def _session(mut window: Window, options: ViewerOptions) raises -> Bool:
    var scene = ViewerScene(load_viewer_scene(options.scene, options.path))
    var camera = scene.state.camera
    var initial_camera = camera
    var settings = scene.state.settings.copy()
    if options.width:
        settings.image_width = options.width
    if options.height:
        settings.image_height = options.height
    var max_samples = (
        options.max_samples if options.max_samples else settings.sample_sequence_length
    )
    if options.max_depth >= 0:
        settings.max_depth = options.max_depth
    if options.sampler >= 0:
        settings.sampler = Sampler(options.sampler)
    var max_depth = settings.max_depth
    settings.samples_per_pixel = 1
    settings.sample_offset = 0
    settings.sample_sequence_length = max_samples
    settings.validate()
    var window_width = 960
    var window_height = max(
        240, window_width * settings.image_height // settings.image_width
    )
    window.library.get_function[NoneType]("SetWindowSize")(
        c_int(window_width), c_int(window_height)
    )
    var raster = RasterScene(options.library, window, scene.scene_data())
    var objects = object_boxes(scene.scene_data())
    var worker = RenderWorker(scene^, options.build_method, options.traversal)
    var traced = options.traced
    var backend = options.backend
    var integrator = options.integrator
    var bounds = options.bounds
    var depth = options.bvh_depth
    var boxes = List[DebugBox]()
    var have_boxes = False
    var box_request_in_flight = False
    var generation = UInt64(0)
    var samples = 0
    var linear = List[LinearColor]()
    var rgba = List[UInt8]()
    var texture = Optional[DisplayTexture]()
    var render_ms = Float64(0)
    var moving_time = Float32(0)
    var display_frames = 0
    var error_text = String("")
    var next_scene = False
    while not window.should_close():
        var changed = False
        var dt = min(window.frame_time(), Float32(0.1))
        if window.mouse_down():
            var delta = window.mouse_delta()
            if delta.x != 0 or delta.y != 0:
                rotate_camera(camera, -delta.x, -delta.y)
                changed = True
        var movement = Vec3W(0)
        if window.is_key_down(87):
            movement += camera.forward
        if window.is_key_down(83):
            movement = movement - camera.forward
        if window.is_key_down(68):
            movement += camera.right
        if window.is_key_down(65):
            movement = movement - camera.right
        if window.is_key_down(69):
            movement.y += 1
        if window.is_key_down(81):
            movement.y -= 1
        if dot(movement, movement) > 0:
            var speed = Float32(8 if window.is_key_down(340) else 2)
            camera.origin = camera.origin + normalize(movement) * speed * dt
            changed = True
        if window.is_key_pressed(82):
            camera = initial_camera
            changed = True
        if window.is_key_pressed(70):
            window.toggle_fullscreen()
        if window.is_key_pressed(258):
            traced = not traced
        if window.is_key_pressed(71):
            backend = 1 - backend
            changed = True
        if window.is_key_pressed(80):
            settings.sampler = Sampler((Int(settings.sampler) + 1) % 6)
            changed = True
        for index in range(5):
            if window.is_key_pressed(49 + index):
                integrator = index
                changed = True
        if window.is_key_pressed(66):
            bounds = (bounds + 1) % 4
        if window.is_key_pressed(91):
            depth = max(0, depth - 1)
        if window.is_key_pressed(93):
            depth = min(64, depth + 1)
        if window.is_key_pressed(61):
            max_samples *= 2
            changed = True
        if window.is_key_pressed(45):
            max_samples = max(1, max_samples // 2)
            changed = True
        if window.is_key_pressed(44):
            max_depth = max(0, max_depth - 1)
            changed = True
        if window.is_key_pressed(46):
            max_depth = min(32, max_depth + 1)
            changed = True
        if window.is_key_pressed(67):
            next_scene = True
            break
        var width = max(1, window.width())
        var height = max(1, window.height())
        # Keep the render and raster projection at the same current aspect.
        var render_height = max(1, settings.image_width * height // width)
        if render_height != settings.image_height:
            settings.image_height = render_height
            changed = True
        if changed:
            generation += 1
            samples = 0
            linear.clear()
            moving_time = 0.25
            error_text = ""
        else:
            moving_time = max(Float32(0), moving_time - dt)
        if worker.ready():
            if worker.state[].error != "":
                error_text = worker.state[].error
            if box_request_in_flight:
                boxes = worker.state[].take_boxes()
                have_boxes = True
                box_request_in_flight = False
            if worker.state[].result:
                var completed = worker.state[].result.take()
                if completed.generation == generation:
                    if len(linear) != len(completed.pixels):
                        linear = List[LinearColor](
                            length=len(completed.pixels), fill=LinearColor(0)
                        )
                    for index in range(len(linear)):
                        linear[index] += completed.pixels[index] * Float32(
                            completed.samples
                        )
                    samples += completed.samples
                    render_ms = completed.render_ms
                    rgba.resize(len(linear) * 4, 0)
                    for index in range(len(linear)):
                        var color = linear[index] / Float32(samples)
                        rgba[4 * index] = color_to_byte(color.x)
                        rgba[4 * index + 1] = color_to_byte(color.y)
                        rgba[4 * index + 2] = color_to_byte(color.z)
                        rgba[4 * index + 3] = 255
                    if texture:
                        if (
                            Int(texture.value().texture.width)
                            != completed.width
                            or Int(texture.value().texture.height)
                            != completed.height
                        ):
                            _ = texture.take()
                    if not texture:
                        texture = DisplayTexture(
                            options.library,
                            window,
                            completed.width,
                            completed.height,
                            rgba,
                        )
                    else:
                        window.update_texture(texture.value().texture, rgba)
            worker.acknowledge()
        var needs_boxes = (
            bounds >= 2 and not have_boxes and not box_request_in_flight
        )
        if (
            worker.idle()
            and error_text == ""
            and (needs_boxes or (traced and samples < max_samples))
        ):
            var batch = settings.copy()
            batch.max_depth = max_depth
            batch.samples_per_pixel = min(
                options.moving_samples if moving_time > 0 else 1,
                max_samples - samples,
            ) if traced else 1
            batch.sample_offset = samples
            batch.sample_sequence_length = max_samples
            worker.submit(
                camera,
                batch,
                generation,
                integrator,
                backend,
                traced,
                needs_boxes,
            )
            box_request_in_flight = needs_boxes
        var frame = window^.begin_drawing()
        try:
            frame.clear_background(Color(24, 27, 33, 255))
            if traced and texture:
                frame.draw_texture(texture.value().texture, width, height)
            if not traced:
                frame.begin_3d(raster_camera(camera))
                raster.draw(frame)
                if bounds == 1 or bounds == 3:
                    _draw_boxes(frame, objects, Color(255, 196, 60, 255))
                if bounds >= 2 and have_boxes:
                    _draw_boxes(frame, boxes, Color(40, 210, 255, 255), depth)
                frame.end_3d()
            frame.draw_rectangle(
                0,
                0,
                width,
                74 if error_text == "" else 96,
                Color(10, 13, 18, 225),
            )
            var mode = String("TRACE") if traced else String("RASTER")
            var backend_name = String("CPU") if backend == 0 else String("GPU")
            var trace_stats = String(
                t"{samples}/{max_samples} spp | {render_ms} ms"
            )
            frame.draw_text(
                String(
                    t"{mode} | Bajo {backend_name} | integrator"
                    t" {integrator + 1} | {trace_stats}"
                ),
                10,
                8,
                18,
                Color(235, 240, 250, 255),
            )
            frame.draw_text(
                String(
                    "WASD/QE move | drag look | Tab raster/trace | G CPU/GPU |"
                    " 1-5 integrator | C scene"
                ),
                10,
                31,
                14,
                Color(180, 195, 210, 255),
            )
            frame.draw_text(
                String(
                    t"B bounds {bounds} | [/] CPU BVH depth {depth} | capped:"
                    t" {len(objects)} objects, {len(boxes)} BVH | R reset | F"
                    t" fullscreen"
                ),
                10,
                51,
                14,
                Color(180, 195, 210, 255),
            )
            if error_text != "":
                frame.draw_text(
                    error_text, 10, 75, 14, Color(255, 100, 100, 255)
                )
        except error:
            window = frame^.end()
            raise error
        window = frame^.end()
        if not traced or samples > 0:
            display_frames += 1
        if options.frame_limit and display_frames >= options.frame_limit:
            break
        if options.frame_limit and error_text != "":
            raise Error(error_text)
    if options.screenshot != "":
        window.screenshot(options.screenshot)
    print(
        "Viewer closed:",
        samples,
        "samples; raster mesh count",
        len(raster.meshes),
        "draw count",
        len(raster.draws),
    )
    return next_scene


def run_viewer(var options: ViewerOptions) raises:
    if options.help:
        print_help()
        return
    var window = Window(
        options.library,
        960,
        642,
        String("Bajo - raylib raster / ray tracing"),
        60,
        UInt32(36 + (128 if options.hidden else 0)),
    )
    try:
        while _session(window, options):
            options.scene = options.scene + 1 if options.scene < 8 else 0
            options.path = ""
    except error:
        window^.close()
        raise error
    window^.close()
