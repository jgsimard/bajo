import std.os.path
from max.algorithm import parallelize
from std.math import abs, cos, pi, sin, sqrt
from std.sys import num_logical_cores

from bajo.bvh import Camera
from bajo.core import Affine3f32, Point3f32, Vec3f32
from bajo.parser.number import parse_f32_fast_at, parse_int_f32
from bajo.parser.ply import PlyMesh
from bajo.parser.ply.pack import _pack_ply_mesh
from bajo.rt.render_types import (
    Color,
    RenderSettings,
)
from bajo.rt.material_types import (
    Environment,
    ImageTexture,
    Integrator,
    NO_TEXTURE,
    SurfaceId,
)
from bajo.rt.scene_data import SceneBuilder
from bajo.rt.scene_description import SceneDescription

from bajo.parser.asset_loader import AssetLoader


comptime _Transform = Affine3f32[.LOCAL, .WORLD]
comptime _PointL = Point3f32[.LOCAL]
comptime _PointW = Point3f32[.WORLD]
comptime _VecL = Vec3f32[.LOCAL]
comptime _VecW = Vec3f32[.WORLD]


@fieldwise_init
struct _Token[origin: ImmOrigin]:
    var value: StringSpan[Self.origin]
    var quoted: Bool
    var line: Int


struct _Lexer[origin: ImmOrigin]:
    var bytes: ImmSpan[UInt8, Self.origin]
    var pos: Int
    var line: Int

    def __init__(out self, bytes: ImmSpan[UInt8, Self.origin]):
        self.bytes = bytes
        self.pos = 0
        self.line = 1

    def _skip_space(mut self):
        while self.pos < len(self.bytes):
            var c = self.bytes.unsafe_get(self.pos)
            if c == UInt8(35):  # # comment
                while self.pos < len(self.bytes):
                    c = self.bytes.unsafe_get(self.pos)
                    self.pos += 1
                    if c == UInt8(10):
                        self.line += 1
                        break
            elif c == UInt8(10):
                self.pos += 1
                self.line += 1
            elif c == UInt8(9) or c == UInt8(13) or c == UInt8(32):
                self.pos += 1
            else:
                break

    def has_next(mut self) -> Bool:
        self._skip_space()
        return self.pos < len(self.bytes)

    def next_is_quoted(mut self) -> Bool:
        self._skip_space()
        return self.pos < len(self.bytes) and self.bytes.unsafe_get(
            self.pos
        ) == UInt8(34)

    def next(mut self) raises -> _Token[Self.origin]:
        self._skip_space()
        if self.pos >= len(self.bytes):
            raise Error("unexpected end of PBRT input")

        var token_line = self.line
        var quoted = self.bytes.unsafe_get(self.pos) == UInt8(34)
        var first_char = self.bytes.unsafe_get(self.pos)
        if first_char == UInt8(91) or first_char == UInt8(93):
            var start = self.pos
            self.pos += 1
            return _Token(
                StringSpan[Self.origin](
                    unsafe_from_utf8=self.bytes[start : self.pos]
                ),
                False,
                token_line,
            )
        if quoted:
            self.pos += 1
            var start = self.pos
            while self.pos < len(self.bytes):
                var c = self.bytes.unsafe_get(self.pos)
                if c == UInt8(34):
                    var value = StringSpan[Self.origin](
                        unsafe_from_utf8=self.bytes[start : self.pos]
                    )
                    self.pos += 1
                    return _Token(value, True, token_line)
                if c == UInt8(10):
                    self.line += 1
                self.pos += 1
            raise Error(t"unterminated PBRT string at line {token_line}")

        var start = self.pos
        while self.pos < len(self.bytes):
            var c = self.bytes.unsafe_get(self.pos)
            if (
                c == UInt8(9)
                or c == UInt8(10)
                or c == UInt8(13)
                or c == UInt8(32)
                or c == UInt8(35)
                or c == UInt8(91)
                or c == UInt8(93)
            ):
                break
            self.pos += 1
        return _Token(
            StringSpan[Self.origin](
                unsafe_from_utf8=self.bytes[start : self.pos]
            ),
            False,
            token_line,
        )


struct _Parameter[origin: ImmOrigin](Copyable):
    var declaration: StringSpan[Self.origin]
    var values: List[StringSpan[Self.origin]]

    def __init__(
        out self,
        declaration: StringSpan[Self.origin],
        var values: List[StringSpan[Self.origin]],
    ):
        self.declaration = declaration
        self.values = values^


struct _Parameters[origin: ImmOrigin]:
    # Entries own the token lists; token contents borrow the current source text.
    var entries: List[_Parameter[Self.origin]]

    def __init__(out self):
        self.entries = List[_Parameter[Self.origin]]()

    def has(self, declaration: ImmStringSpan) -> Bool:
        for entry in self.entries:
            if entry.declaration == declaration:
                return True
        return False

    def string(
        self, declaration: ImmStringSpan
    ) -> Optional[StringSpan[Self.origin]]:
        for entry in self.entries:
            if entry.declaration == declaration and len(entry.values) > 0:
                return entry.values[0]
        return Optional[StringSpan[Self.origin]]()

    def string_or_empty(
        self, declaration: ImmStringSpan
    ) -> StringSpan[Self.origin]:
        var value = self.string(declaration)
        if value:
            return value.value()
        return StringSpan[Self.origin]()

    def f32(
        self, declaration: ImmStringSpan, default: Float32
    ) raises -> Float32:
        for entry in self.entries:
            if entry.declaration == declaration and len(entry.values) > 0:
                return _parse_f32(entry.values[0])
        return default

    def integer(self, declaration: ImmStringSpan, default: Int) raises -> Int:
        for entry in self.entries:
            if entry.declaration == declaration and len(entry.values) > 0:
                return parse_int_f32(entry.values[0], "PBRT")
        return default

    def color(self, name: ImmStringSpan, default: Color) raises -> Color:
        var rgb_name = "rgb " + name
        var color_name = "color " + name
        for entry in self.entries:
            if entry.declaration == rgb_name or entry.declaration == color_name:
                if len(entry.values) != 3:
                    raise Error("PBRT color parameter requires three values")
                return Color(
                    _parse_f32(entry.values[0]),
                    _parse_f32(entry.values[1]),
                    _parse_f32(entry.values[2]),
                )
        return default

    def values(
        self, declaration: ImmStringSpan
    ) -> ImmSpan[StringSpan[Self.origin], origin_of(self)]:
        for entry in self.entries:
            if entry.declaration == declaration:
                # The enclosing parameter set owns each entry's storage.
                return Span(
                    unsafe_ptr=entry.values.unsafe_ptr().unsafe_origin_cast[
                        origin_of(self)
                    ](),
                    length=len(entry.values),
                )
        return ImmSpan[StringSpan[Self.origin], origin_of(self)]()


struct _CameraOptions(Copyable):
    """Camera options shared by full-scene and camera-only parsing."""

    var origin: _PointW
    var target: _PointW
    var up: _VecW
    var fov: Float32

    def __init__(out self):
        self.origin = _PointW(0.0, 0.0, 5.0)
        self.target = _PointW(0.0, 0.0, 0.0)
        self.up = _VecW(0.0, 1.0, 0.0)
        self.fov = 45.0

    def parse_look_at(mut self, mut lexer: _Lexer) raises:
        var values = _fixed_f32(lexer, 9)
        self.origin = _PointW(values[0], values[1], values[2])
        self.target = _PointW(values[3], values[4], values[5])
        self.up = _VecW(values[6], values[7], values[8])

    def parse_camera(mut self, mut lexer: _Lexer) raises:
        var kind = lexer.next().value
        if not kind == "perspective":
            raise Error("only PBRT perspective cameras are supported")
        var params = _parse_params(lexer)
        self.fov = params.f32("float fov", 45.0)

    def to_camera(self) -> Camera:
        return Camera.from_vfov(self.origin, self.target, self.up, self.fov)


@fieldwise_init
struct _GraphicsState(Copyable):
    var transform: _Transform
    var surface: SurfaceId[1]
    var area_light: Bool
    var emission: Color
    var reverse_orientation: Bool


@fieldwise_init
struct _ColorTexture(Copyable):
    var scale: Color
    var image_index: UInt32


@fieldwise_init
struct _FloatTexture(Copyable):
    var scale: Float32
    var image_index: UInt32
    var u_scale: Float32
    var v_scale: Float32


struct _Builder(
    Deinitable where (False, "call finish() or abort() to consume the parser")
):
    var scene: SceneBuilder
    var named_materials: Dict[String, SurfaceId[1]]
    var color_textures: Dict[String, _ColorTexture]
    var scalar_textures: Dict[String, _FloatTexture]
    var image_paths: List[String]
    var state: _GraphicsState
    var attribute_stack: List[_GraphicsState]
    var transform_stack: List[_Transform]
    var camera_options: _CameraOptions
    var image_width: Int
    var image_height: Int
    var samples_per_pixel: Int
    var max_depth: Int
    var integrator: Integrator
    var environment: Environment
    var has_environment: Bool

    def __init__(out self):
        self.scene = SceneBuilder()
        self.named_materials = Dict[String, SurfaceId[1]]()
        self.color_textures = Dict[String, _ColorTexture]()
        self.scalar_textures = Dict[String, _FloatTexture]()
        self.image_paths = List[String]()
        var default_surface = self.scene.surfaces.add_lambertian(Color(0.5))
        self.state = _GraphicsState(
            _Transform.identity(),
            default_surface.copy(),
            False,
            Color(0.0),
            False,
        )
        self.attribute_stack = List[_GraphicsState]()
        self.transform_stack = List[_Transform]()
        self.camera_options = _CameraOptions()
        self.image_width = 640
        self.image_height = 480
        self.samples_per_pixel = 16
        self.max_depth = 8
        self.integrator = .PATH
        self.environment = Environment.black()
        self.has_environment = False

    def add_ply_mesh(
        mut self,
        mesh: PlyMesh,
        surface: SurfaceId[1],
    ) raises:
        var packed = _pack_ply_mesh(
            mesh, self.state.reverse_orientation, min(8, num_logical_cores())
        )
        var vertices = List[_PointL]()
        var normals = List[Float32]()
        var texcoords = List[Float32]()
        var bounds = packed^.transfer_to(vertices, normals, texcoords)
        _ = self.scene.add_owned_triangle_mesh_instance(
            vertices^,
            normals^,
            texcoords^,
            self.state.transform,
            bounds,
            surface,
        )

    def _prepare_finish[
        Loader: AssetLoader
    ](mut self, loader: Loader) raises -> Tuple[Camera, RenderSettings]:
        if len(self.attribute_stack) != 0 or len(self.transform_stack) != 0:
            raise Error("unclosed PBRT attribute or transform scope")
        if (
            len(self.scene.spheres) == 0
            and len(self.scene.triangle_vertices) == 0
            and len(self.scene.triangle_instances) == 0
        ):
            raise Error("PBRT scene contains no supported shapes")
        var loaded_images = List[Optional[ImageTexture]](
            length=len(self.image_paths),
            fill_with=lambda (index: Int) -> Optional[ImageTexture]: Optional[
                ImageTexture
            ](),
        )
        var image_errors = List[String](length=len(self.image_paths), fill="")

        def load_image(
            image_idx: Int,
        ) {imm, mut loaded_images, mut image_errors}:
            try:
                var image = loader.read_image_texture(
                    self.image_paths[image_idx]
                )
                loaded_images[image_idx] = image^
            except error:
                image_errors[image_idx] = String(error)

        if len(self.image_paths) > 1:
            parallelize(
                load_image,
                len(self.image_paths),
                min(num_logical_cores(), len(self.image_paths)),
            )
        elif len(self.image_paths) == 1:
            load_image(0)

        for image_idx in range(len(self.image_paths)):
            if image_errors[image_idx].byte_length() != 0:
                raise Error(
                    "failed to load PBRT image '"
                    + self.image_paths[image_idx]
                    + "': "
                    + image_errors[image_idx]
                )
            if not loaded_images[image_idx]:
                raise Error("PBRT image loader returned no image")
            var stored_idx = self.scene.surfaces.add_image_texture(
                loaded_images[image_idx].take()
            )
            debug_assert(stored_idx == UInt32(image_idx))

        var camera = self.camera_options.to_camera()
        var settings = RenderSettings(
            self.image_width,
            self.image_height,
            self.samples_per_pixel,
            UInt64(2026),
            self.max_depth,
        )
        self.scene.set_environment(self.environment)
        return (camera, settings^)

    def finish[
        Loader: AssetLoader
    ](deinit self, loader: Loader) raises -> SceneDescription:
        var prepared: Tuple[Camera, RenderSettings]
        try:
            prepared = self._prepare_finish(loader)
        except error:
            self.scene^.abort()
            raise error
        var data = self.scene^.finish()
        return SceneDescription(
            data^,
            prepared[0],
            prepared[1].copy(),
            self.integrator,
        )

    def abort(deinit self):
        """Consume and release a partially parsed scene after an error."""
        self.scene^.abort()


def _parse_f32(text: StringSpan) raises -> Float32:
    var parsed = parse_f32_fast_at(text.as_bytes(), 0)
    if parsed.pos != text.byte_length():
        raise Error("invalid PBRT number: " + String(text))
    return parsed.value


def _parse_params(mut lexer: _Lexer) raises -> _Parameters[lexer.origin]:
    var params = _Parameters[lexer.origin]()
    while lexer.next_is_quoted():
        var declaration = lexer.next().value
        if not lexer.has_next():
            raise Error(
                "missing value for PBRT parameter " + String(declaration)
            )
        var values = List[StringSpan[lexer.origin]]()
        var first = lexer.next()
        if first.value == "[":
            while True:
                if not lexer.has_next():
                    raise Error("unterminated PBRT parameter array")
                var item = lexer.next()
                if item.value == "]":
                    break
                values.append(item.value)
        else:
            values.append(first.value)
        params.entries.append(_Parameter(declaration, values^))
    return params^


def _compose(a: _Transform, b: _Transform) -> _Transform:
    """Return a transform that applies b, then a."""
    return _Transform(
        a.m00 * b.m00 + a.m01 * b.m10 + a.m02 * b.m20,
        a.m00 * b.m01 + a.m01 * b.m11 + a.m02 * b.m21,
        a.m00 * b.m02 + a.m01 * b.m12 + a.m02 * b.m22,
        a.m00 * b.tx + a.m01 * b.ty + a.m02 * b.tz + a.tx,
        a.m10 * b.m00 + a.m11 * b.m10 + a.m12 * b.m20,
        a.m10 * b.m01 + a.m11 * b.m11 + a.m12 * b.m21,
        a.m10 * b.m02 + a.m11 * b.m12 + a.m12 * b.m22,
        a.m10 * b.tx + a.m11 * b.ty + a.m12 * b.tz + a.ty,
        a.m20 * b.m00 + a.m21 * b.m10 + a.m22 * b.m20,
        a.m20 * b.m01 + a.m21 * b.m11 + a.m22 * b.m21,
        a.m20 * b.m02 + a.m21 * b.m12 + a.m22 * b.m22,
        a.m20 * b.tx + a.m21 * b.ty + a.m22 * b.tz + a.tz,
    )


def _translation(x: Float32, y: Float32, z: Float32) -> _Transform:
    return _Transform.from_translation(_VecW(x, y, z))


def _scale(x: Float32, y: Float32, z: Float32) -> _Transform:
    return _Transform.from_scale(_VecL(x, y, z))


def _rotation(
    angle: Float32, x: Float32, y: Float32, z: Float32
) raises -> _Transform:
    var axis_length = sqrt(x * x + y * y + z * z)
    if axis_length == 0.0:
        raise Error("PBRT Rotate axis must be non-zero")
    var nx = x / axis_length
    var ny = y / axis_length
    var nz = z / axis_length
    var radians = angle * Float32(pi / 180.0)
    var c = cos(radians)
    var s = sin(radians)
    var t = 1.0 - c
    return _Transform(
        t * nx * nx + c,
        t * nx * ny - s * nz,
        t * nx * nz + s * ny,
        0.0,
        t * nx * ny + s * nz,
        t * ny * ny + c,
        t * ny * nz - s * nx,
        0.0,
        t * nx * nz - s * ny,
        t * ny * nz + s * nx,
        t * nz * nz + c,
        0.0,
    )


def _matrix[
    origin: ImmOrigin
](values: ImmSpan[StringSpan[origin], _]) raises -> _Transform:
    if len(values) != 16:
        raise Error("PBRT transform matrix requires 16 values")
    var m = List[Float32](capacity=16)
    for value in values:
        m.append(_parse_f32(value))
    if m[12] != 0.0 or m[13] != 0.0 or m[14] != 0.0 or m[15] != 1.0:
        raise Error("projective PBRT transforms are not supported")
    return _Transform(
        m[0],
        m[1],
        m[2],
        m[3],
        m[4],
        m[5],
        m[6],
        m[7],
        m[8],
        m[9],
        m[10],
        m[11],
    )


def _color_texture_parameter(
    builder: _Builder,
    params: _Parameters,
    name: ImmStringSpan,
    default: Color,
) raises -> _ColorTexture:
    var token = params.string_or_empty("texture " + name)
    if token.byte_length() != 0:
        var texture_name = String(token)
        if texture_name not in builder.color_textures:
            raise Error("unknown PBRT spectrum texture: " + texture_name)
        return builder.color_textures[texture_name].copy()
    return _ColorTexture(params.color(name, default), NO_TEXTURE)


def _scalar_parameter(
    builder: _Builder,
    params: _Parameters,
    name: ImmStringSpan,
    default: Float32,
) raises -> _FloatTexture:
    var token = params.string_or_empty("texture " + name)
    if token.byte_length() != 0:
        var texture_name = String(token)
        if texture_name not in builder.scalar_textures:
            raise Error("unknown PBRT float texture: " + texture_name)
        return builder.scalar_textures[texture_name].copy()
    return _FloatTexture(
        params.f32("float " + name, default), NO_TEXTURE, 1.0, 1.0
    )


def _roughness_to_alpha(roughness: Float32) -> Float32:
    """PBRT v4's Trowbridge-Reitz perceptual roughness mapping."""
    return sqrt(max(roughness, 0.0))


def _texture(
    mut builder: _Builder,
    name: String,
    value_type: ImmStringSpan,
    implementation: ImmStringSpan,
    params: _Parameters,
    source_path: String,
) raises:
    if value_type == "spectrum" or value_type == "color":
        var value: _ColorTexture
        __match implementation:
            case "constant":
                value = _ColorTexture(
                    params.color("value", Color(1.0)), NO_TEXTURE
                )
            case "scale":
                var tex = _color_texture_parameter(
                    builder, params, "tex", Color(1.0)
                )
                var scale = _color_texture_parameter(
                    builder, params, "scale", Color(1.0)
                )
                if (
                    tex.image_index != NO_TEXTURE
                    and scale.image_index != NO_TEXTURE
                ):
                    raise Error(
                        "PBRT scale of two image textures is not supported"
                    )
                value = _ColorTexture(tex.scale * scale.scale, tex.image_index)
                if value.image_index == NO_TEXTURE:
                    value.image_index = scale.image_index
            case "imagemap":
                var filename = params.string_or_empty("string filename")
                if filename.byte_length() == 0:
                    raise Error("PBRT imagemap texture requires a filename")
                var image_path = std.os.path.join(
                    std.os.path.dirname(source_path), String(filename)
                )
                var image_index = UInt32(len(builder.image_paths))
                builder.image_paths.append(image_path)
                value = _ColorTexture(Color(1.0), image_index)
            case _:
                raise Error(
                    "unsupported PBRT spectrum texture: " + implementation
                )
        builder.color_textures[name] = value.copy()
        return

    if value_type == "float":
        var value: _FloatTexture
        __match implementation:
            case "constant":
                value = _FloatTexture(
                    params.f32("float value", 1.0), NO_TEXTURE, 1.0, 1.0
                )
            case "scale":
                var tex = _scalar_parameter(builder, params, "tex", 1.0)
                var scale = _scalar_parameter(builder, params, "scale", 1.0)
                if (
                    tex.image_index != NO_TEXTURE
                    and scale.image_index != NO_TEXTURE
                ):
                    raise Error(
                        "PBRT scale of two image textures is not supported"
                    )
                value = _FloatTexture(
                    tex.scale * scale.scale,
                    tex.image_index,
                    tex.u_scale,
                    tex.v_scale,
                )
                if value.image_index == NO_TEXTURE:
                    value.image_index = scale.image_index
                    value.u_scale = scale.u_scale
                    value.v_scale = scale.v_scale
            case "imagemap":
                var filename = params.string_or_empty("string filename")
                if filename.byte_length() == 0:
                    raise Error("PBRT imagemap texture requires a filename")
                var image_path = std.os.path.join(
                    std.os.path.dirname(source_path), String(filename)
                )
                var image_index = UInt32(len(builder.image_paths))
                builder.image_paths.append(image_path)
                value = _FloatTexture(
                    params.f32("float scale", 1.0),
                    image_index,
                    params.f32("float uscale", 1.0),
                    params.f32("float vscale", 1.0),
                )
            case _:
                raise Error("unsupported PBRT float texture: " + implementation)
        builder.scalar_textures[name] = value.copy()
        return

    raise Error("unsupported PBRT texture value type: " + value_type)


def _surface(
    mut builder: _Builder, model: ImmStringSpan, params: _Parameters
) raises -> SurfaceId[1]:
    __match model:
        case "coateddiffuse":
            var reflectance = _color_texture_parameter(
                builder, params, "reflectance", Color(0.5)
            )
            var displacement = _scalar_parameter(
                builder, params, "displacement", 0.0
            )
            var roughness = params.f32("float roughness", 0.0)
            var u_roughness = params.f32("float uroughness", roughness)
            var v_roughness = params.f32("float vroughness", roughness)
            roughness = sqrt(
                max(u_roughness, 0.0) * max(v_roughness, 0.0)
            ).clamp(0.0, 1.0)
            var remap = params.string("bool remaproughness")
            if not remap or remap.value() != ImmStringSpan("false"):
                roughness = _roughness_to_alpha(roughness)
            return builder.scene.surfaces.add_coated_diffuse(
                reflectance.scale,
                roughness,
                params.f32("float eta", 1.5),
                reflectance.image_index,
                displacement.image_index,
                displacement.scale,
                displacement.u_scale,
                displacement.v_scale,
                params.f32("float thickness", 0.01),
                params.color("albedo", Color(0.0)),
                params.f32("float g", 0.0),
                params.integer("integer maxdepth", 10),
                params.integer("integer nsamples", 1),
            )
        case "diffuse" | "matte":
            var reflectance = _color_texture_parameter(
                builder, params, "reflectance", Color(0.5)
            )
            return builder.scene.surfaces.add_lambertian(
                reflectance.scale, reflectance.image_index
            )
        case "conductor" | "metal":
            var roughness = params.f32("float roughness", 0.05).clamp(0.0, 1.0)
            var reflectance = _color_texture_parameter(
                builder, params, "reflectance", Color(0.9)
            )
            return builder.scene.surfaces.add_metal(
                reflectance.scale, roughness, reflectance.image_index
            )
        case "dielectric" | "glass":
            return builder.scene.surfaces.add_dielectric(
                params.f32("float eta", 1.5)
            )
        case _:
            raise Error("unsupported PBRT material: " + model)


def _shape[
    Loader: AssetLoader
](
    mut builder: _Builder,
    kind: ImmStringSpan,
    params: _Parameters,
    source_path: String,
    loader: Loader,
) raises:
    var surface = builder.state.surface.copy()
    if builder.state.area_light:
        surface = builder.scene.surfaces.add_emissive(builder.state.emission)

    __match kind:
        case "sphere":
            var center = builder.state.transform.point(_PointL(0.0))
            var x_axis = builder.state.transform.vector(_VecL(1.0, 0.0, 0.0))
            var y_axis = builder.state.transform.vector(_VecL(0.0, 1.0, 0.0))
            var z_axis = builder.state.transform.vector(_VecL(0.0, 0.0, 1.0))
            var sx = sqrt(
                x_axis.x * x_axis.x + x_axis.y * x_axis.y + x_axis.z * x_axis.z
            )
            var sy = sqrt(
                y_axis.x * y_axis.x + y_axis.y * y_axis.y + y_axis.z * y_axis.z
            )
            var sz = sqrt(
                z_axis.x * z_axis.x + z_axis.y * z_axis.y + z_axis.z * z_axis.z
            )
            if abs(sx - sy) > 1e-5 or abs(sx - sz) > 1e-5:
                raise Error(
                    "non-uniformly transformed PBRT spheres are not supported"
                )
            builder.scene.add_sphere(
                center,
                params.f32("float radius", 1.0) * sx,
                surface,
            )
            return

        case "plymesh":
            var filename = params.string_or_empty("string filename")
            if filename.byte_length() == 0:
                raise Error("PBRT plymesh requires a string filename")
            var mesh_path = std.os.path.join(
                std.os.path.dirname(source_path), String(filename)
            )
            var mesh = loader.read_ply_mesh(mesh_path)
            builder.add_ply_mesh(mesh, surface)
            return

        case "trianglemesh" | "loopsubdiv":
            # The control cage is already an indexed triangle mesh. Full Loop
            # refinement can be added later without changing scene ingestion.
            var points = params.values("point3 P")
            if len(points) == 0:
                points = params.values("point P")
            if len(points) % 3 != 0:
                raise Error("PBRT trianglemesh P must contain xyz triples")
            var indices = params.values("integer indices")
            var indexed = len(indices) != 0
            if indexed and len(indices) % 3 != 0:
                raise Error("PBRT trianglemesh indices must contain triples")
            if not indexed and len(points) % 9 != 0:
                raise Error(
                    "unindexed PBRT trianglemesh must contain triangles"
                )
            var point_count = len(points) / 3
            # Leave unused point strings unevaluated and retain the original
            # index-before-coordinate validation order.
            var vertices = List[Optional[_PointW]](
                length=point_count if indexed else 0,
                fill=Optional[_PointW](),
            )
            var transform = builder.state.transform.copy()

            @always_inline
            def decode_vertex(index: Int) raises {imm} -> _PointW:
                var base = 3 * index
                return transform.point(
                    _PointL(
                        _parse_f32(points[base]),
                        _parse_f32(points[base + 1]),
                        _parse_f32(points[base + 2]),
                    )
                )

            @always_inline
            def vertex(index: Int) raises {imm, mut vertices} -> _PointW:
                if not indexed:
                    return decode_vertex(index)
                if not vertices[index]:
                    vertices[index] = decode_vertex(index)
                return vertices[index].value()

            var element_count = len(indices) if indexed else point_count
            for base in range(0, element_count, 3):
                var i0 = parse_int_f32(
                    indices[base], "PBRT"
                ) if indexed else base
                var i1 = (
                    parse_int_f32(
                        indices[base + 1], "PBRT"
                    ) if indexed else base
                    + 1
                )
                var i2 = (
                    parse_int_f32(
                        indices[base + 2], "PBRT"
                    ) if indexed else base
                    + 2
                )
                if (
                    i0 < 0
                    or i0 >= point_count
                    or i1 < 0
                    or i1 >= point_count
                    or i2 < 0
                    or i2 >= point_count
                ):
                    raise Error("PBRT trianglemesh index is out of range")
                var p0 = vertex(i0)
                var p1 = vertex(i1)
                var p2 = vertex(i2)
                if builder.state.reverse_orientation:
                    builder.scene.add_triangle(p0, p2, p1, surface)
                else:
                    builder.scene.add_triangle(p0, p1, p2, surface)
            return
    raise Error("unsupported PBRT shape: " + kind)


def _fixed_f32(mut lexer: _Lexer, count: Int) raises -> List[Float32]:
    var values = List[Float32](capacity=count)
    for _ in range(count):
        values.append(_parse_f32(lexer.next().value))
    return values^


def _bracket_values(mut lexer: _Lexer) raises -> List[StringSpan[lexer.origin]]:
    if not lexer.next().value == "[":
        raise Error("expected '[' in PBRT transform")
    var values = List[StringSpan[lexer.origin]]()
    while True:
        var token = lexer.next()
        if token.value == "]":
            return values^
        values.append(token.value)


def _named_params(mut lexer: _Lexer) raises -> _Parameters[lexer.origin]:
    _ = lexer.next()  # Implementation name precedes its parameter declarations.
    return _parse_params(lexer)


def _include_path(mut lexer: _Lexer, path: String) raises -> String:
    return std.os.path.join(
        std.os.path.dirname(path), String(lexer.next().value)
    )


def _parse_text[
    Loader: AssetLoader
](
    mut builder: _Builder,
    text: String,
    path: String,
    loader: Loader,
    depth: Int,
) raises:
    if depth > 32:
        raise Error("PBRT Include nesting exceeds 32 files")
    var span = StringSpan(text)
    var lexer = _Lexer(span.as_bytes())
    while lexer.has_next():
        var command_token = lexer.next()
        if command_token.quoted:
            raise Error(t"expected PBRT directive at line {command_token.line}")

        __match command_token.value:
            case "LookAt":
                builder.camera_options.parse_look_at(lexer)

            case "Camera":
                builder.camera_options.parse_camera(lexer)

            case "Film":
                var params = _named_params(lexer)
                builder.image_width = params.integer("integer xresolution", 640)
                builder.image_height = params.integer(
                    "integer yresolution", 480
                )

            case "Sampler":
                var params = _named_params(lexer)
                builder.samples_per_pixel = params.integer(
                    "integer pixelsamples", 16
                )

            case "Integrator":
                var integrator_name = lexer.next().value
                if not (
                    integrator_name == "path" or integrator_name == "volpath"
                ):
                    raise Error(
                        "only the PBRT path/volpath integrators are supported"
                    )
                builder.integrator = .PATH
                var params = _parse_params(lexer)
                builder.max_depth = params.integer("integer maxdepth", 8)

            case "PixelFilter" | "Accelerator":
                # Bajo supplies these implementation details itself, but consuming
                # their declarations keeps ordinary PBRT scene headers portable.
                _ = _named_params(lexer)

            case "ColorSpace":
                var color_space = lexer.next().value
                if not color_space == "srgb":
                    raise Error("only the PBRT sRGB color space is supported")

            case "Option":
                # Options affect pbrt's runtime rather than the scene description.
                _ = _parse_params(lexer)

            case "WorldBegin":
                builder.state.transform = _Transform.identity()

            case "AttributeBegin":
                builder.attribute_stack.append(builder.state.copy())

            case "AttributeEnd":
                if len(builder.attribute_stack) == 0:
                    raise Error("PBRT AttributeEnd without AttributeBegin")
                builder.state = builder.attribute_stack.pop()

            case "TransformBegin":
                builder.transform_stack.append(builder.state.transform.copy())

            case "TransformEnd":
                if len(builder.transform_stack) == 0:
                    raise Error("PBRT TransformEnd without TransformBegin")
                builder.state.transform = builder.transform_stack.pop()

            case "Identity":
                builder.state.transform = _Transform.identity()

            case "Translate":
                var v = _fixed_f32(lexer, 3)
                builder.state.transform = _compose(
                    builder.state.transform, _translation(v[0], v[1], v[2])
                )

            case "Scale":
                var v = _fixed_f32(lexer, 3)
                builder.state.transform = _compose(
                    builder.state.transform, _scale(v[0], v[1], v[2])
                )

            case "Rotate":
                var v = _fixed_f32(lexer, 4)
                builder.state.transform = _compose(
                    builder.state.transform, _rotation(v[0], v[1], v[2], v[3])
                )

            case "Transform":
                builder.state.transform = _matrix(_bracket_values(lexer))

            case "ConcatTransform":
                builder.state.transform = _compose(
                    builder.state.transform, _matrix(_bracket_values(lexer))
                )

            case "ReverseOrientation":
                builder.state.reverse_orientation = (
                    not builder.state.reverse_orientation
                )

            case "Texture":
                _texture(
                    builder,
                    String(lexer.next().value),
                    lexer.next().value,
                    lexer.next().value,
                    _parse_params(lexer),
                    path,
                )

            case "Material":
                var model = lexer.next().value
                builder.state.surface = _surface(
                    builder, model, _parse_params(lexer)
                )

            case "MakeNamedMaterial":
                var name = String(lexer.next().value)
                var params = _parse_params(lexer)
                var model = params.string("string type")
                if model:
                    builder.named_materials[name] = _surface(
                        builder, model.value(), params
                    )
                else:
                    builder.named_materials[name] = _surface(
                        builder, ImmStringSpan("diffuse"), params
                    )

            case "NamedMaterial":
                var name = String(lexer.next().value)
                if name not in builder.named_materials:
                    raise Error("unknown PBRT named material: " + name)
                builder.state.surface = builder.named_materials[name].copy()

            case "LightSource":
                var model = lexer.next().value
                var params = _parse_params(lexer)
                if not model == "infinite":
                    raise Error(
                        "only infinite PBRT non-area lights are supported"
                    )
                if builder.has_environment:
                    raise Error(
                        "multiple PBRT infinite lights are not supported"
                    )
                var multiplier = params.color("L", Color(1.0)) * params.f32(
                    "float scale", 1.0
                )
                var filename = params.string_or_empty("string filename")
                if filename.byte_length() == 0:
                    builder.environment = Environment.uniform(multiplier)
                else:
                    var image_path = std.os.path.join(
                        std.os.path.dirname(path), String(filename)
                    )
                    var image_index = UInt32(len(builder.image_paths))
                    builder.image_paths.append(image_path)
                    var inverse = builder.state.transform.inverse()
                    if not inverse.mask[0]:
                        raise Error("PBRT infinite light transform is singular")
                    builder.environment = Environment.image(
                        image_index,
                        multiplier,
                        inverse.inv,
                        builder.state.transform,
                    )
                builder.has_environment = True

            case "AreaLightSource":
                var model = lexer.next().value
                if not model == "diffuse":
                    raise Error("only diffuse PBRT area lights are supported")
                var params = _parse_params(lexer)
                builder.state.area_light = True
                builder.state.emission = params.color(
                    "L", Color(1.0)
                ) * params.f32("float scale", 1.0)

            case "Shape":
                _shape(
                    builder,
                    lexer.next().value,
                    _parse_params(lexer),
                    path,
                    loader,
                )

            case "Include":
                var include_path = _include_path(lexer, path)
                _parse_text(
                    builder,
                    loader.read_text(include_path),
                    include_path,
                    loader,
                    depth + 1,
                )

            case "WorldEnd":
                pass

            case _:
                raise Error(
                    t"unsupported PBRT directive '{command_token.value}' at"
                    t" line {command_token.line}"
                )


def _parse_pbrt[
    Loader: AssetLoader
](text: String, path: String, loader: Loader) raises -> SceneDescription:
    var builder = _Builder()
    try:
        _parse_text(builder, text, path, loader, 0)
    except error:
        builder^.abort()
        raise error
    return builder^.finish(loader)


def _parse_camera_text[
    Loader: AssetLoader
](
    mut options: _CameraOptions,
    text: String,
    path: String,
    loader: Loader,
    depth: Int,
) raises -> Bool:
    """Parse only the PBRT options block, stopping before scene assets."""
    if depth > 32:
        raise Error("PBRT Include nesting exceeds 32 files")
    var span = StringSpan(text)
    var lexer = _Lexer(span.as_bytes())
    while lexer.has_next():
        var command_token = lexer.next()
        if command_token.quoted:
            raise Error(t"expected PBRT directive at line {command_token.line}")

        __match command_token.value:
            case "WorldBegin":
                return True

            case "LookAt":
                options.parse_look_at(lexer)

            case "Camera":
                options.parse_camera(lexer)

            case "Film" | "Sampler" | "Integrator" | "PixelFilter" | "Accelerator":
                _ = _named_params(lexer)

            case "ColorSpace":
                _ = lexer.next()

            case "Option":
                _ = _parse_params(lexer)

            case "Identity":
                pass

            case "Translate" | "Scale":
                _ = _fixed_f32(lexer, 3)

            case "Rotate":
                _ = _fixed_f32(lexer, 4)

            case "Transform" | "ConcatTransform":
                _ = _bracket_values(lexer)

            case "Include":
                var include_path = _include_path(lexer, path)
                if _parse_camera_text(
                    options,
                    loader.read_text(include_path),
                    include_path,
                    loader,
                    depth + 1,
                ):
                    return True
            case _:
                raise Error(
                    t"unsupported PBRT options directive"
                    t" '{command_token.value}' at line {command_token.line}"
                )
    return False


def _parse_pbrt_camera[
    Loader: AssetLoader
](text: String, path: String, loader: Loader) raises -> Camera:
    var options = _CameraOptions()
    _ = _parse_camera_text(options, text, path, loader, 0)
    return options.to_camera()
