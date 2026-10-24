"""Reusable raylib C ABI types, window lifetime, input, and drawing calls."""

from std.ffi import OwnedDLHandle, c_int, c_uint
from std.os import abort
from std.memory import Pointer


@fieldwise_init
struct Color(ImplicitlyCopyable, RegisterPassable):
    """raylib Color: four unsigned bytes, passed by value."""

    var r: UInt8
    var g: UInt8
    var b: UInt8
    var a: UInt8


@fieldwise_init
struct Vector2(ImplicitlyCopyable, RegisterPassable):
    """raylib Vector2: two C floats, passed by value."""

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
    """raylib 6.2 mesh ABI; native-size integers hold opaque C addresses."""

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


comptime KEY_LEFT_SHIFT = 340
comptime KEY_RIGHT = 262
comptime KEY_D = 68
comptime KEY_TAB = 258


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
            "ToggleFullscreen",
            "BeginDrawing",
            "ClearBackground",
            "DrawLine",
            "DrawRectangle",
            "DrawLineEx",
            "DrawText",
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
            "UnloadShader",
            "LoadTextureFromImage",
            "UpdateTexture",
            "UnloadTexture",
            "DrawTexturePro",
            "GetMouseDelta",
            "IsMouseButtonDown",
            "GetFrameTime",
            "GetScreenWidth",
            "GetScreenHeight",
            "GetFPS",
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

    def unload_mesh(self, mesh: Mesh) raises:
        self.library.get_function[NoneType]("UnloadMesh")(mesh)

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

    def unload_texture(self, texture: Texture) raises:
        self.library.get_function[NoneType]("UnloadTexture")(texture)

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

    def draw_line(
        self, x1: Int, y1: Int, x2: Int, y2: Int, color: Color
    ) raises:
        self.window.library.get_function[NoneType]("DrawLine")(
            c_int(x1), c_int(y1), c_int(x2), c_int(y2), color
        )

    def draw_rectangle(
        self, x: Int, y: Int, width: Int, height: Int, color: Color
    ) raises:
        self.window.library.get_function[NoneType]("DrawRectangle")(
            c_int(x), c_int(y), c_int(width), c_int(height), color
        )

    def draw_line_ex(
        self, start: Vector2, end: Vector2, thickness: Float32, color: Color
    ) raises:
        self.window.library.get_function[NoneType]("DrawLineEx")(
            start, end, thickness, color
        )

    def draw_text(
        self, var label: String, x: Int, y: Int, size: Int, color: Color
    ) raises:
        self.window.library.get_function[NoneType]("DrawText")(
            label.as_c_string_span(), c_int(x), c_int(y), c_int(size), color
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
