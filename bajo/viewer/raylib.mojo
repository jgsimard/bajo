"""Reusable raylib C ABI types, window lifetime, input, and drawing calls."""

from std.ffi import OwnedDLHandle, c_int, c_uint, external_call
from std.os import abort
from std.memory import Pointer


@fieldwise_init
struct Color(ImplicitlyCopyable, RegisterPassable):
    """Raylib Color: four unsigned bytes, passed by value."""

    var r: UInt8
    var g: UInt8
    var b: UInt8
    var a: UInt8


@fieldwise_init
struct Vector2(ImplicitlyCopyable, RegisterPassable):
    """Raylib Vector2: two C floats, passed by value."""

    var x: Float32
    var y: Float32


@fieldwise_init
struct Vector3(ImplicitlyCopyable, RegisterPassable):
    var x: Float32
    var y: Float32
    var z: Float32


@fieldwise_init
struct Camera3D(ImplicitlyCopyable, RegisterPassable):
    var position: Vector3
    var target: Vector3
    var up: Vector3
    var fovy: Float32
    var projection: c_int


@fieldwise_init
struct Matrix(ImplicitlyCopyable, RegisterPassable):
    # raylib stores its column-numbered fields in row order in the C struct.
    var m0: Float32
    var m4: Float32
    var m8: Float32
    var m12: Float32
    var m1: Float32
    var m5: Float32
    var m9: Float32
    var m13: Float32
    var m2: Float32
    var m6: Float32
    var m10: Float32
    var m14: Float32
    var m3: Float32
    var m7: Float32
    var m11: Float32
    var m15: Float32


@fieldwise_init
struct Rectangle(ImplicitlyCopyable, RegisterPassable):
    var x: Float32
    var y: Float32
    var width: Float32
    var height: Float32


@fieldwise_init
struct FilePathList(ImplicitlyCopyable, RegisterPassable):
    var count: c_uint
    var paths: UInt


def _copy_c_string(address: UInt) -> String:
    if not address:
        return ""
    var pointer = Pointer[UInt8, MutUntrackedOrigin](
        unsafe_from_address=Int(address)
    )
    var length = Int(external_call["strlen", UInt](pointer))
    return String(
        StringSpan(unsafe_from_utf8=ImmSpan(unsafe_ptr=pointer, length=length))
    )


@fieldwise_init
struct BoundingBox(ImplicitlyCopyable, RegisterPassable):
    var minimum: Vector3
    var maximum: Vector3


@fieldwise_init
struct Image(ImplicitlyCopyable, RegisterPassable):
    var data: UInt
    var width: c_int
    var height: c_int
    var mipmaps: c_int
    var format: c_int


@fieldwise_init
struct Texture(ImplicitlyCopyable, RegisterPassable):
    var id: c_uint
    var width: c_int
    var height: c_int
    var mipmaps: c_int
    var format: c_int


@fieldwise_init
struct Font(ImplicitlyCopyable, RegisterPassable):
    """Raylib Font ABI; glyph data remains owned by TextFont."""

    var base_size: c_int
    var glyph_count: c_int
    var glyph_padding: c_int
    var texture: Texture
    var rectangles: UInt
    var glyphs: UInt


struct TextFont:
    """Load once per session and release before closing the window."""

    var library: OwnedDLHandle
    var font: Font

    def __init__(out self, library_path: String, var path: String) raises:
        var library = OwnedDLHandle(library_path)
        var font = library.get_function[Font]("LoadFontEx")(
            path.as_c_string_span(), c_int(32), UInt(0), c_int(0)
        )
        # raylib returns its small default font when loading fails.
        if (
            not library.get_function[Bool]("IsFontValid")(font)
            or font.base_size != 32
        ):
            raise Error("could not load viewer font: " + path)
        library.get_function[NoneType]("SetTextureFilter")(
            font.texture, c_int(1)
        )
        self.font = font
        self.library = library^

    def __deinit__(deinit self):
        try:
            self.library.get_function[NoneType]("UnloadFont")(self.font)
        except error:
            abort(String(t"raylib font cleanup failed: {error}"))


@fieldwise_init
struct Shader(ImplicitlyCopyable, RegisterPassable):
    var id: c_uint
    var locs: UInt


@fieldwise_init
struct MaterialMap(ImplicitlyCopyable, RegisterPassable):
    var texture: Texture
    var color: Color
    var value: Float32


@fieldwise_init
struct Material(ImplicitlyCopyable, RegisterPassable):
    var shader: Shader
    var maps: UInt
    var p0: Float32
    var p1: Float32
    var p2: Float32
    var p3: Float32


@fieldwise_init
struct Mesh(ImplicitlyCopyable, RegisterPassable):
    """Raylib 6.2 mesh ABI; native-size integers hold opaque C addresses."""

    var vertex_count: c_int
    var triangle_count: c_int
    var vertices: UInt
    var texcoords: UInt
    var texcoords2: UInt
    var normals: UInt
    var tangents: UInt
    var colors: UInt
    var indices: UInt
    var bone_count: c_int
    var bone_indices: UInt
    var bone_weights: UInt
    var anim_vertices: UInt
    var anim_normals: UInt
    var vao_id: c_uint
    var vbo_id: UInt

    def __init__(out self):
        self.vertex_count = 0
        self.triangle_count = 0
        self.vertices = 0
        self.texcoords = 0
        self.texcoords2 = 0
        self.normals = 0
        self.tangents = 0
        self.colors = 0
        self.indices = 0
        self.bone_count = 0
        self.bone_indices = 0
        self.bone_weights = 0
        self.anim_vertices = 0
        self.anim_normals = 0
        self.vao_id = 0
        self.vbo_id = 0


struct Window(
    Deinitable where (False, "Call window^.close() to close the native window"),
    Movable,
    Writable,
):
    var library: OwnedDLHandle

    def __init__(
        out self,
        path: String,
        width: Int,
        height: Int,
        var title: String,
        fps: Int,
        flags: UInt32 = 0x20,
    ) raises:
        var library = OwnedDLHandle(path)
        # Check all symbols before creating a window.
        var symbols: List[String] = [
            "SetConfigFlags",
            "InitWindow",
            "IsWindowReady",
            "SetTargetFPS",
            "WindowShouldClose",
            "IsKeyDown",
            "IsKeyPressed",
            "SetExitKey",
            "GetMousePosition",
            "IsMouseButtonPressed",
            "GetMouseWheelMove",
            "GetTime",
            "GetClipboardText",
            "DirectoryExists",
            "FileExists",
            "GetDirectoryPath",
            "GetPrevDirectoryPath",
            "GetFileName",
            "GetFileExtension",
            "LoadDirectoryFiles",
            "UnloadDirectoryFiles",
            "IsFileDropped",
            "LoadDroppedFiles",
            "UnloadDroppedFiles",
            "BeginScissorMode",
            "EndScissorMode",
            "ToggleFullscreen",
            "BeginDrawing",
            "ClearBackground",
            "DrawRectangle",
            "LoadFontEx",
            "IsFontValid",
            "SetTextureFilter",
            "UnloadFont",
            "DrawTextEx",
            "EndDrawing",
            "CloseWindow",
            "BeginMode3D",
            "EndMode3D",
            "DrawBoundingBox",
            "DrawMesh",
            "UploadMesh",
            "UnloadMesh",
            "GenMeshSphere",
            "MemAlloc",
            "LoadMaterialDefault",
            "UnloadMaterial",
            "LoadShaderFromMemory",
            "IsShaderValid",
            "GetShaderLocation",
            "LoadTextureFromImage",
            "UpdateTexture",
            "UnloadTexture",
            "DrawTexturePro",
            "GetMouseDelta",
            "IsMouseButtonDown",
            "GetFrameTime",
            "GetScreenWidth",
            "GetScreenHeight",
            "SetWindowSize",
            "TakeScreenshot",
            "SetTraceLogLevel",
            "rlDisableBackfaceCulling",
            "rlEnableBackfaceCulling",
        ]
        for symbol in symbols:
            if not library.check_symbol(symbol):
                raise Error("raylib is missing symbol: " + symbol)
        library.get_function[NoneType]("SetTraceLogLevel")(c_int(4))
        library.get_function[NoneType]("SetConfigFlags")(c_uint(flags))
        library.get_function[NoneType]("InitWindow")(
            c_int(width),
            c_int(height),
            title.as_c_string_span(),
        )
        if not library.get_function[Bool]("IsWindowReady")():
            library.get_function[NoneType]("CloseWindow")()
            raise Error("raylib could not initialize the window")
        library.get_function[NoneType]("SetTargetFPS")(c_int(fps))

        self.library = library^

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Window(raylib FFI)")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def should_close(self) raises -> Bool:
        return self.library.get_function[Bool]("WindowShouldClose")()

    def is_key_down(self, key: Int) raises -> Bool:
        return self.library.get_function[Bool]("IsKeyDown")(c_int(key))

    def is_key_pressed(self, key: Int) raises -> Bool:
        return self.library.get_function[Bool]("IsKeyPressed")(c_int(key))

    def set_exit_key(self, key: Int) raises:
        self.library.get_function[NoneType]("SetExitKey")(c_int(key))

    def mouse_position(self) raises -> Vector2:
        return self.library.get_function[Vector2]("GetMousePosition")()

    def mouse_pressed(self) raises -> Bool:
        return self.library.get_function[Bool]("IsMouseButtonPressed")(c_int(0))

    def mouse_wheel(self) raises -> Float32:
        return self.library.get_function[Float32]("GetMouseWheelMove")()

    def time(self) raises -> Float64:
        return self.library.get_function[Float64]("GetTime")()

    def clipboard_text(self) raises -> String:
        return _copy_c_string(
            self.library.get_function[UInt]("GetClipboardText")()
        )

    def directory_exists(self, var path: String) raises -> Bool:
        return self.library.get_function[Bool]("DirectoryExists")(
            path.as_c_string_span()
        )

    def canonical_path(self, var path: String) raises -> String:
        # The supported Linux viewer uses libc to resolve relative paths/symlinks.
        var address = external_call["realpath", UInt](
            path.as_c_string_span(), UInt(0)
        )
        if not address:
            raise Error("could not resolve path: " + path)
        var result = _copy_c_string(address)
        external_call["free", NoneType](address)
        return result^

    def file_exists(self, var path: String) raises -> Bool:
        return self.library.get_function[Bool]("FileExists")(
            path.as_c_string_span()
        )

    def directory_path(self, var path: String) raises -> String:
        return _copy_c_string(
            self.library.get_function[UInt]("GetDirectoryPath")(
                path.as_c_string_span()
            )
        )

    def parent_directory(self, var path: String) raises -> String:
        return _copy_c_string(
            self.library.get_function[UInt]("GetPrevDirectoryPath")(
                path.as_c_string_span()
            )
        )

    def file_name(self, var path: String) raises -> String:
        return _copy_c_string(
            self.library.get_function[UInt]("GetFileName")(
                path.as_c_string_span()
            )
        )

    def file_extension(self, var path: String) raises -> String:
        return _copy_c_string(
            self.library.get_function[UInt]("GetFileExtension")(
                path.as_c_string_span()
            )
        )

    def directory_files(self, var path: String) raises -> List[String]:
        var files = self.library.get_function[FilePathList](
            "LoadDirectoryFiles"
        )(path.as_c_string_span())
        var result = List[String](capacity=Int(files.count))
        var paths = Pointer[UInt, MutUntrackedOrigin](
            unsafe_from_address=Int(files.paths)
        )
        for index in range(Int(files.count)):
            result.append(_copy_c_string(paths[unsafe_offset=index]))
        self.library.get_function[NoneType]("UnloadDirectoryFiles")(files)
        return result^

    def dropped_files(self) raises -> List[String]:
        var result = List[String]()
        if self.library.get_function[Bool]("IsFileDropped")():
            var files = self.library.get_function[FilePathList](
                "LoadDroppedFiles"
            )()
            var paths = Pointer[UInt, MutUntrackedOrigin](
                unsafe_from_address=Int(files.paths)
            )
            for index in range(Int(files.count)):
                result.append(_copy_c_string(paths[unsafe_offset=index]))
            self.library.get_function[NoneType]("UnloadDroppedFiles")(files)
        return result^

    def toggle_fullscreen(self) raises:
        self.library.get_function[NoneType]("ToggleFullscreen")()

    def frame_time(self) raises -> Float32:
        return self.library.get_function[Float32]("GetFrameTime")()

    def mouse_delta(self) raises -> Vector2:
        return self.library.get_function[Vector2]("GetMouseDelta")()

    def mouse_down(self, button: Int = 0) raises -> Bool:
        return self.library.get_function[Bool]("IsMouseButtonDown")(
            c_int(button)
        )

    def width(self) raises -> Int:
        return Int(self.library.get_function[c_int]("GetScreenWidth")())

    def height(self) raises -> Int:
        return Int(self.library.get_function[c_int]("GetScreenHeight")())

    def screenshot(self, var path: String) raises:
        self.library.get_function[NoneType]("TakeScreenshot")(
            path.as_c_string_span()
        )

    def allocate(self, size: Int) raises -> UInt:
        if size <= 0 or UInt64(size) > UInt64(UInt32.MAX):
            raise Error("raylib allocation exceeds its 32-bit size limit")
        var address = self.library.get_function[UInt]("MemAlloc")(c_uint(size))
        if address == 0:
            raise Error("raylib memory allocation failed")
        return address

    def create_mesh(
        self,
        vertices: ImmSpan[Float32, _],
        normals: ImmSpan[Float32, _],
        colors: ImmSpan[UInt8, _],
    ) raises -> Mesh:
        if (
            len(vertices) == 0
            or len(vertices) % 9 != 0
            or len(normals) != len(vertices)
            or len(colors) != len(vertices) // 3 * 4
        ):
            raise Error("invalid raster mesh attributes")
        var mesh = Mesh()
        mesh.vertex_count = c_int(len(vertices) // 3)
        mesh.triangle_count = c_int(len(vertices) // 9)
        try:
            mesh.vertices = self.allocate(len(vertices) * 4)
            mesh.normals = self.allocate(len(normals) * 4)
            mesh.colors = self.allocate(len(colors))
            MutSpan(
                unsafe_ptr=Pointer[Float32, MutUntrackedOrigin](
                    unsafe_from_address=Int(mesh.vertices)
                ),
                length=len(vertices),
            ).copy_from(vertices)
            MutSpan(
                unsafe_ptr=Pointer[Float32, MutUntrackedOrigin](
                    unsafe_from_address=Int(mesh.normals)
                ),
                length=len(normals),
            ).copy_from(normals)
            MutSpan(
                unsafe_ptr=Pointer[UInt8, MutUntrackedOrigin](
                    unsafe_from_address=Int(mesh.colors)
                ),
                length=len(colors),
            ).copy_from(colors)
            self.library.get_function[NoneType]("UploadMesh")(
                Pointer(to=mesh), False
            )
            if mesh.vao_id == 0:
                raise Error("could not upload raster mesh")
        except error:
            self.library.get_function[NoneType]("UnloadMesh")(mesh)
            raise error
        return mesh

    def sphere_mesh(self) raises -> Mesh:
        return self.library.get_function[Mesh]("GenMeshSphere")(
            Float32(1), c_int(20), c_int(24)
        )

    def create_texture(
        self, width: Int, height: Int, pixels: ImmSpan[UInt8, _]
    ) raises -> Texture:
        if width <= 0 or height <= 0 or len(pixels) != width * height * 4:
            raise Error("invalid raster texture dimensions")
        var image = Image(
            UInt(Int(pixels.unsafe_ptr())),
            c_int(width),
            c_int(height),
            c_int(1),
            c_int(7),
        )
        var texture = self.library.get_function[Texture](
            "LoadTextureFromImage"
        )(image)
        if texture.id == 0:
            raise Error("could not upload display texture")
        return texture

    def update_texture(
        self, texture: Texture, pixels: ImmSpan[UInt8, _]
    ) raises:
        if len(pixels) != Int(texture.width) * Int(texture.height) * 4:
            raise Error("texture update dimensions do not match")
        self.library.get_function[NoneType]("UpdateTexture")(
            texture, pixels.unsafe_ptr()
        )

    def begin_drawing(deinit self) -> Frame:
        """Transfer this window into a unique drawing session."""
        return Frame(self^)

    def close(deinit self):
        """Consume the window owner; its handle cannot be reused afterward."""
        try:
            self.library.get_function[NoneType]("CloseWindow")()
        except error:
            abort(String(t"raylib CloseWindow failed: {error}"))


struct Frame(
    Deinitable where (
        False,
        "Call frame^.end() to complete the drawing session",
    ),
    Movable,
):
    """A drawing session that owns its window until end() returns it."""

    var window: Window

    def __init__(out self, var window: Window):
        try:
            window.library.get_function[NoneType]("BeginDrawing")()
        except error:
            abort(String(t"raylib BeginDrawing failed: {error}"))
        self.window = window^

    def end(deinit self) -> Window:
        """End drawing exactly once and return ownership of the window."""
        try:
            self.window.library.get_function[NoneType]("EndDrawing")()
        except error:
            abort(String(t"raylib EndDrawing failed: {error}"))
        return self.window^

    def clear_background(self, color: Color) raises:
        self.window.library.get_function[NoneType]("ClearBackground")(color)

    def draw_rectangle(
        self, x: Int, y: Int, width: Int, height: Int, color: Color
    ) raises:
        self.window.library.get_function[NoneType]("DrawRectangle")(
            c_int(x), c_int(y), c_int(width), c_int(height), color
        )

    def begin_scissor(self, x: Int, y: Int, width: Int, height: Int) raises:
        self.window.library.get_function[NoneType]("BeginScissorMode")(
            c_int(x), c_int(y), c_int(width), c_int(height)
        )

    def end_scissor(self) raises:
        self.window.library.get_function[NoneType]("EndScissorMode")()

    def draw_text_ex(
        self,
        font: Font,
        var label: String,
        x: Int,
        y: Int,
        size: Int,
        color: Color,
    ) raises:
        self.window.library.get_function[NoneType]("DrawTextEx")(
            font,
            label.as_c_string_span(),
            Vector2(Float32(x), Float32(y)),
            Float32(size),
            Float32(0),
            color,
        )

    def begin_3d(self, camera: Camera3D) raises:
        self.window.library.get_function[NoneType]("BeginMode3D")(camera)
        self.window.library.get_function[NoneType]("rlDisableBackfaceCulling")()

    def end_3d(self) raises:
        self.window.library.get_function[NoneType]("rlEnableBackfaceCulling")()
        self.window.library.get_function[NoneType]("EndMode3D")()

    def draw_mesh(
        self, mesh: Mesh, material: Material, transform: Matrix
    ) raises:
        self.window.library.get_function[NoneType]("DrawMesh")(
            mesh, material, transform
        )

    def draw_box(self, box: BoundingBox, color: Color) raises:
        self.window.library.get_function[NoneType]("DrawBoundingBox")(
            box, color
        )

    def draw_texture(self, texture: Texture, width: Int, height: Int) raises:
        self.window.library.get_function[NoneType]("DrawTexturePro")(
            texture,
            Rectangle(0, 0, Float32(texture.width), Float32(texture.height)),
            Rectangle(0, 0, Float32(width), Float32(height)),
            Vector2(0, 0),
            Float32(0),
            Color(255, 255, 255, 255),
        )
