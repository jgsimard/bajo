"""The native Mojo/raylib viewer: raster presentation and Bajo tracing."""

from std.math import sin, cos, min, max
from std.sys.arg import argv
from std.ffi import c_int
from bajo.bvh import Camera
from bajo.core import Vec3W, cross, dot, length, normalize
from bajo.core.random import Sampler
from bajo.rt.render_types import RenderSettings
from .core import ViewerScene
from .scenes import (
    BUILTIN_SCENE_COUNT,
    CUSTOM_SCENE,
    load_viewer_scene,
    viewer_scene_index,
    viewer_scene_name,
)
from .raylib import Window, Color, BoundingBox, Vector3, Frame, TextFont
from .raster import (
    RasterScene,
    DisplayTexture,
    raster_camera,
    raster_view_matrix,
)
from .worker import RenderWorker
from .debug import DebugBox, object_boxes
from .file_picker import FilePicker, contains, is_scene_file
from .batching import samples_for_batch, smaller_batch, larger_batch
from .accumulation import FrameAccumulator


struct ViewerOptions(Copyable):
    var scene: Int
    var path: String
    var library: String
    var width: Int
    var height: Int
    var max_samples: Int
    var moving_samples: Int
    var samples_per_batch: Int
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
        self.samples_per_batch = 1
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
                options.scene = CUSTOM_SCENE
                options.path = value
            case "--raylib-library":
                options.library = value
            case "--width":
                options.width = Int(value)
            case "--height":
                options.height = Int(value)
            case "--max-spp":
                options.max_samples = Int(value)
            case "--samples-per-batch" | "--batch-spp":
                options.samples_per_batch = Int(value)
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
        or (options.samples_per_batch != -1 and options.samples_per_batch < 1)
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
        "--samples-per-batch N (or --batch-spp): positive count, -1 for all"
        " remaining samples"
    )
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
        "Tab raster/trace; G CPU/GPU; 1-5 integrator; P sampler; C next scene;"
        " O open PBRT"
    )
    print(
        "B bounds; [/] BVH depth; +/- maximum samples; ,/. path depth; Esc"
        " closes; J/K smaller/larger batch (-1 = all)"
    )


def rotate_camera(mut camera: Camera, yaw: Float32, pitch: Float32):
    """Rotate the shared basis; preserve camera roll, lens size and FOV."""
    var handedness: Float32 = (
        -1 if dot(cross(camera.right, camera.forward), camera.up) < 0 else 1
    )
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
    angle = pitch * 0.006 * handedness
    var candidate = forward * cos(angle) + cross(right, forward) * sin(angle)
    if abs(candidate.y) < 0.9998:
        forward = candidate
    var lens_u = length(camera.defocus_disk_u)
    var lens_v = length(camera.defocus_disk_v)
    camera.forward = normalize(forward)
    camera.right = normalize(right)
    camera.up = normalize(cross(camera.right, camera.forward)) * handedness
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


struct PreparedScene:
    """Construct a replacement fully before releasing the displayed scene."""

    var camera: Camera
    var settings: RenderSettings
    var raster: RasterScene
    var objects: List[DebugBox]
    var worker: RenderWorker
    var name: String

    def __init__(out self, window: Window, options: ViewerOptions) raises:
        var scene = ViewerScene(load_viewer_scene(options.scene, options.path))
        var camera = scene.state.camera
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
        settings.samples_per_pixel = 1
        settings.sample_offset = 0
        settings.sample_sequence_length = max_samples
        settings.validate()
        var raster = RasterScene(options.library, window, scene.scene_data())
        var objects = object_boxes(scene.scene_data())
        var worker = RenderWorker(
            scene^, options.build_method, options.traversal
        )
        var name = viewer_scene_name(options.scene)
        if options.scene == CUSTOM_SCENE:
            name = "PBRT: " + window.file_name(options.path)
        self.camera = camera
        self.settings = settings^
        self.raster = raster^
        self.objects = objects^
        self.worker = worker^
        self.name = name^


def _session(mut window: Window, mut options: ViewerOptions) raises -> Bool:
    var view = PreparedScene(window, options)
    var scene_name = view.name
    var camera = view.camera
    var initial_camera = camera
    var settings = view.settings.copy()
    var max_samples = settings.sample_sequence_length
    var max_depth = settings.max_depth
    var window_width = 960
    var window_height = max(
        240, window_width * settings.image_height // settings.image_width
    )
    window.library.get_function[NoneType]("SetWindowSize")(
        c_int(window_width), c_int(window_height)
    )
    var font = TextFont(
        options.library, String("resources/fonts/DejaVuSans.ttf")
    )
    var traced = options.traced
    var backend = options.backend
    var integrator = options.integrator
    var algorithm_names: List[String] = ["PATH", "NEE", "MIS", "NORMALS", "AO"]
    var sampler_names: List[String] = [
        "Independent",
        "Halton",
        "R2",
        "Owen-Sobol",
        "SZ",
        "STBN",
    ]
    var bounds = options.bounds
    var depth = options.bvh_depth
    var boxes = List[DebugBox]()
    var have_boxes = False
    var box_request_in_flight = False
    var accumulation = FrameAccumulator()
    var texture = Optional[DisplayTexture]()
    var render_ms = Float64(0)
    var moving_time = Float32(0)
    var display_frames = 0
    var error_text = String("")
    var next_scene = False
    var picker = FilePicker()
    var load_error = String("")
    while not window.should_close():
        var changed = False
        var camera_moved = False
        var width = max(1, window.width())
        var batch_x = 10 if width < 840 else width - 260
        var batch_y = 112 if width < 840 else 32
        var dt = min(window.frame_time(), Float32(0.1))
        if not picker.active:
            if window.is_key_pressed(256):
                break
            if window.is_key_pressed(79) or (
                window.mouse_pressed()
                and contains(
                    window.mouse_position(), window.width() - 148, 58, 138, 28
                )
            ):
                try:
                    picker.show(window, options.path)
                    load_error = ""
                except error:
                    load_error = String(error)
        if not picker.active:
            var batch_click = window.mouse_pressed()
            var mouse = window.mouse_position()
            if window.is_key_pressed(74) or (
                batch_click and contains(mouse, batch_x, batch_y, 28, 28)
            ):
                options.samples_per_batch = smaller_batch(
                    options.samples_per_batch, max_samples
                )
            if window.is_key_pressed(75) or (
                batch_click and contains(mouse, batch_x + 222, batch_y, 28, 28)
            ):
                options.samples_per_batch = larger_batch(
                    options.samples_per_batch, max_samples
                )
            if window.mouse_down() and not contains(
                mouse, batch_x, batch_y, 250, 28
            ):
                var delta = window.mouse_delta()
                if delta.x != 0 or delta.y != 0:
                    rotate_camera(camera, -delta.x, -delta.y)
                    camera_moved = True
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
                camera_moved = True
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
        var chosen = Optional[String]()
        if picker.active:
            try:
                chosen = picker.update(window)
            except error:
                picker.error = String(error)
        var dropped = window.dropped_files()
        if len(dropped):
            chosen = dropped[0]
        if chosen:
            try:
                var path = chosen.take()
                if not is_scene_file(window, path):
                    raise Error("Select a .pbrt or .pbrt-v4 scene.")
                var candidate_options = options.copy()
                candidate_options.scene = CUSTOM_SCENE
                candidate_options.path = path
                var candidate = PreparedScene(window, candidate_options)
                # Commit only after parsing and raster upload have succeeded.
                view = candidate^
                options.scene = CUSTOM_SCENE
                options.path = path
                scene_name = view.name
                camera = view.camera
                initial_camera = camera
                settings = view.settings.copy()
                max_samples = settings.sample_sequence_length
                max_depth = settings.max_depth
                if texture:
                    _ = texture.take()
                boxes.clear()
                have_boxes = False
                box_request_in_flight = False
                render_ms = 0
                error_text = ""
                load_error = ""
                picker.active = False
                next_scene = False
                changed = True
            except error:
                if picker.active:
                    picker.error = String(error)
                else:
                    load_error = "Scene load failed: " + String(error)
        width = max(1, window.width())
        batch_x = 10 if width < 840 else width - 260
        batch_y = 112 if width < 840 else 32
        var height = max(1, window.height())
        # Keep the render and raster projection at the same current aspect.
        var render_height = max(1, settings.image_width * height // width)
        if render_height != settings.image_height:
            settings.image_height = render_height
            changed = True
        if changed:
            accumulation.reset()
            moving_time = 0.25 if camera_moved else 0
            error_text = ""
        else:
            moving_time = max(Float32(0), moving_time - dt)
        if view.worker.ready():
            if view.worker.state[].error != "":
                error_text = view.worker.state[].error
            if box_request_in_flight:
                boxes = view.worker.state[].take_boxes()
                have_boxes = True
                box_request_in_flight = False
            if view.worker.state[].result:
                var completed = view.worker.state[].result.take()
                if accumulation.add(completed):
                    render_ms = completed.render_ms
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
                            accumulation.rgba,
                        )
                    else:
                        window.update_texture(
                            texture.value().texture, accumulation.rgba
                        )
            view.worker.acknowledge()
        var needs_boxes = (
            bounds >= 2 and not have_boxes and not box_request_in_flight
        )
        if (
            view.worker.idle()
            and not next_scene
            and error_text == ""
            and (needs_boxes or (traced and accumulation.samples < max_samples))
        ):
            var batch = settings.copy()
            batch.max_depth = max_depth
            batch.samples_per_pixel = samples_for_batch(
                options.samples_per_batch,
                accumulation.samples,
                max_samples,
                moving_time > 0,
                options.moving_samples,
            ) if traced else 1
            batch.sample_offset = accumulation.samples
            batch.sample_sequence_length = max_samples
            view.worker.submit(
                camera,
                batch,
                accumulation.generation,
                integrator,
                backend,
                traced,
                needs_boxes,
            )
            box_request_in_flight = needs_boxes
        var message = load_error if load_error != "" else error_text
        var toolbar_height = 138 if width < 840 else 110
        var frame = window^.begin_drawing()
        try:
            frame.clear_background(Color(24, 27, 33, 255))
            if traced and texture:
                frame.draw_texture(texture.value().texture, width, height)
            # Keep a useful view during backend preparation and the first batch.
            if not traced or not texture:
                frame.begin_3d(
                    raster_camera(camera), raster_view_matrix(camera)
                )
                view.raster.draw(frame)
                if bounds == 1 or bounds == 3:
                    _draw_boxes(frame, view.objects, Color(255, 196, 60, 255))
                if bounds >= 2 and have_boxes:
                    _draw_boxes(frame, boxes, Color(40, 210, 255, 255), depth)
                frame.end_3d()
            frame.draw_rectangle(
                0,
                0,
                width,
                toolbar_height + (26 if message != "" else 0),
                Color(10, 13, 18, 225),
            )
            var mode = String("TRACE") if traced else String("RASTER")
            if traced and not texture:
                mode = (
                    "TRACE (loading; raster preview)" if error_text
                    == "" else "TRACE ERROR (raster preview)"
                )
            var backend_name = String("CPU") if backend == 0 else String("GPU")
            var batch_count = String(
                "1 batch"
            ) if accumulation.batches == 1 else String(
                t"{accumulation.batches} batches"
            )
            var trace_stats = String(
                t"{accumulation.samples}/{max_samples} spp | {batch_count} |"
                t" {render_ms} ms"
            )
            frame.draw_text_ex(
                font.font,
                String(
                    t"{scene_name} | {mode} | Bajo {backend_name} |"
                    t" {trace_stats}"
                ),
                10,
                8,
                20,
                Color(235, 240, 250, 255),
            )
            frame.draw_text_ex(
                font.font,
                String(
                    t"Algorithm: {algorithm_names[integrator]} (1-5) | Sampler:"
                    t" {sampler_names[Int(settings.sampler)]} (P)"
                ),
                10,
                35,
                18,
                Color(235, 240, 250, 255),
            )
            frame.draw_text_ex(
                font.font,
                String(
                    "WASD/QE move | drag look | Tab raster/trace | G CPU/GPU |"
                    " C scene"
                ),
                10,
                62,
                16,
                Color(180, 195, 210, 255),
            )
            frame.draw_text_ex(
                font.font,
                String(
                    t"B bounds {bounds} | [/] CPU BVH depth {depth} | capped:"
                    t" {len(view.objects)} objects, {len(boxes)} BVH | R reset"
                    t" | F fullscreen"
                ),
                10,
                86,
                16,
                Color(180, 195, 210, 255),
            )
            var batch_label = String(
                "all"
            ) if options.samples_per_batch == -1 else String(
                t"{options.samples_per_batch} spp"
            )
            frame.draw_rectangle(
                batch_x, batch_y, 28, 28, Color(45, 75, 108, 255)
            )
            frame.draw_text_ex(
                font.font,
                String("-"),
                batch_x + 9,
                batch_y + 3,
                20,
                Color(235, 240, 250, 255),
            )
            frame.begin_scissor(batch_x + 32, batch_y, 186, 28)
            frame.draw_text_ex(
                font.font,
                String(t"Batch: {batch_label} (J/K)"),
                batch_x + 36,
                batch_y + 5,
                16,
                Color(235, 240, 250, 255),
            )
            frame.end_scissor()
            frame.draw_rectangle(
                batch_x + 222, batch_y, 28, 28, Color(45, 75, 108, 255)
            )
            frame.draw_text_ex(
                font.font,
                String("+"),
                batch_x + 229,
                batch_y + 3,
                20,
                Color(235, 240, 250, 255),
            )
            frame.draw_rectangle(
                width - 148, 58, 138, 28, Color(45, 75, 108, 255)
            )
            frame.draw_text_ex(
                font.font,
                String("Open scene (O)"),
                width - 138,
                62,
                16,
                Color(235, 240, 250, 255),
            )
            if message != "":
                frame.draw_text_ex(
                    font.font,
                    message,
                    10,
                    toolbar_height + 2,
                    16,
                    Color(255, 100, 100, 255),
                )
            if picker.active:
                picker.draw(frame, font.font, width, height)
        except error:
            window = frame^.end()
            raise error
        window = frame^.end()
        # EndDrawing polls input. Without it the C press remains latched and
        # every new session immediately switches again without rendering.
        if next_scene:
            break
        if not traced or accumulation.samples > 0:
            display_frames += 1
        if options.frame_limit and display_frames >= options.frame_limit:
            break
        if options.frame_limit and error_text != "":
            raise Error(error_text)
    if not next_scene:
        if options.screenshot != "":
            window.screenshot(options.screenshot)
        print(
            "Viewer closed:",
            accumulation.samples,
            "samples; raster mesh count",
            len(view.raster.meshes),
            "draw count",
            len(view.raster.draws),
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
        window.set_exit_key(0)
        while _session(window, options):
            options.scene = (
                options.scene + 1 if options.scene + 1
                < BUILTIN_SCENE_COUNT else 0
            )
            options.path = ""
    except error:
        window^.close()
        raise error
    window^.close()
