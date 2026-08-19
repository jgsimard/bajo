import std.os.path
from std.math import abs, cos, pi, sin, sqrt

from bajo.bvh.camera import Camera
from bajo.bvh.types import Instance, Sphere
from bajo.core import Affine3f32, Frame, Point3f32, Vec3f32
from bajo.parser.obj.f32 import parse_f32_at
from bajo.rt.types import (
    Color,
    RenderSettings,
    SurfaceId,
    SurfaceStore,
    World,
    add_sphere,
    add_triangle,
)

from .loaders import PbrtTextLoader
from .types import PbrtScene


comptime _LOCAL = Frame.LOCAL
comptime _WORLD = Frame.WORLD
comptime _Transform = Affine3f32[_LOCAL, _WORLD]
comptime _PointL = Point3f32[_LOCAL]
comptime _PointW = Point3f32[_WORLD]
comptime _VecL = Vec3f32[_LOCAL]
comptime _VecW = Vec3f32[_WORLD]


@fieldwise_init
struct _Token:
    var value: String
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

    def next(mut self) raises -> _Token:
        self._skip_space()
        if self.pos >= len(self.bytes):
            raise Error("unexpected end of PBRT input")

        var token_line = self.line
        var quoted = self.bytes.unsafe_get(self.pos) == UInt8(34)
        var first_char = self.bytes.unsafe_get(self.pos)
        if first_char == UInt8(91) or first_char == UInt8(93):
            self.pos += 1
            if first_char == UInt8(91):
                return _Token("[", False, token_line)
            return _Token("]", False, token_line)
        if quoted:
            self.pos += 1
            var start = self.pos
            while self.pos < len(self.bytes):
                var c = self.bytes.unsafe_get(self.pos)
                if c == UInt8(34):
                    var value = String(
                        StringSpan[Self.origin](
                            unsafe_from_utf8=self.bytes[start : self.pos]
                        )
                    )
                    self.pos += 1
                    return _Token(value^, True, token_line)
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
            String(
                StringSpan[Self.origin](
                    unsafe_from_utf8=self.bytes[start : self.pos]
                )
            ),
            False,
            token_line,
        )


struct _Parameter(Copyable):
    var declaration: String
    var values: List[String]

    def __init__(out self, declaration: String, var values: List[String]):
        self.declaration = declaration
        self.values = values^


struct _Parameters:
    var entries: List[_Parameter]

    def __init__(out self):
        self.entries = List[_Parameter]()

    def has(self, declaration: String) -> Bool:
        for entry in self.entries:
            if entry.declaration == declaration:
                return True
        return False

    def string(self, declaration: String, default: String) -> String:
        for entry in self.entries:
            if entry.declaration == declaration and len(entry.values) > 0:
                return entry.values[0]
        return default

    def f32(self, declaration: String, default: Float32) raises -> Float32:
        for entry in self.entries:
            if entry.declaration == declaration and len(entry.values) > 0:
                return _parse_f32(entry.values[0])
        return default

    def integer(self, declaration: String, default: Int) raises -> Int:
        for entry in self.entries:
            if entry.declaration == declaration and len(entry.values) > 0:
                return _parse_int(entry.values[0])
        return default

    def color(self, name: String, default: Color) raises -> Color:
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

    def values(self, declaration: String) -> List[String]:
        for entry in self.entries:
            if entry.declaration == declaration:
                return entry.values.copy()
        return List[String]()


@fieldwise_init
struct _GraphicsState(Copyable):
    var transform: _Transform
    var surface: SurfaceId[1]
    var area_light: Bool
    var emission: Color
    var reverse_orientation: Bool


struct _Builder:
    var spheres: List[Sphere[_WORLD]]
    var sphere_surfaces: List[SurfaceId[1]]
    var triangle_vertices: List[_PointW]
    var triangle_surfaces: List[SurfaceId[1]]
    var surfaces: SurfaceStore
    var named_materials: Dict[String, SurfaceId[1]]
    var state: _GraphicsState
    var attribute_stack: List[_GraphicsState]
    var transform_stack: List[_Transform]
    var camera_origin: _PointW
    var camera_target: _PointW
    var camera_up: _VecW
    var camera_fov: Float32
    var image_width: Int
    var image_height: Int
    var samples_per_pixel: Int
    var max_depth: Int
    var integrator: String

    def __init__(out self):
        self.spheres = List[Sphere[_WORLD]]()
        self.sphere_surfaces = List[SurfaceId[1]]()
        self.triangle_vertices = List[_PointW]()
        self.triangle_surfaces = List[SurfaceId[1]]()
        self.surfaces = SurfaceStore()
        self.named_materials = Dict[String, SurfaceId[1]]()
        var default_surface = self.surfaces.add_lambertian(Color(0.5))
        self.state = _GraphicsState(
            _Transform.identity(),
            default_surface.copy(),
            False,
            Color(0.0),
            False,
        )
        self.attribute_stack = List[_GraphicsState]()
        self.transform_stack = List[_Transform]()
        self.camera_origin = _PointW(0.0, 0.0, 5.0)
        self.camera_target = _PointW(0.0, 0.0, 0.0)
        self.camera_up = _VecW(0.0, 1.0, 0.0)
        self.camera_fov = 45.0
        self.image_width = 640
        self.image_height = 480
        self.samples_per_pixel = 16
        self.max_depth = 8
        self.integrator = "path"

    def finish(mut self) raises -> PbrtScene:
        if len(self.attribute_stack) != 0 or len(self.transform_stack) != 0:
            raise Error("unclosed PBRT attribute or transform scope")
        if len(self.spheres) == 0 and len(self.triangle_vertices) == 0:
            raise Error("PBRT scene contains no supported shapes")
        var camera = Camera.from_vfov(
            self.camera_origin,
            self.camera_target,
            self.camera_up,
            self.camera_fov,
        )
        var settings = RenderSettings(
            self.image_width,
            self.image_height,
            self.samples_per_pixel,
            UInt64(2026),
            self.max_depth,
        )
        var surfaces = SurfaceStore()
        surfaces.lambertians = self.surfaces.lambertians.copy()
        surfaces.metals = self.surfaces.metals.copy()
        surfaces.dielectrics = self.surfaces.dielectrics.copy()
        surfaces.emissives = self.surfaces.emissives.copy()
        var meshes = List[List[Point3f32[_LOCAL]]]()
        var instances = List[Instance]()
        var instance_surfaces = List[SurfaceId[1]]()
        var world = World[](
            self.spheres.copy(),
            self.sphere_surfaces.copy(),
            self.triangle_vertices.copy(),
            self.triangle_surfaces.copy(),
            meshes^,
            instances^,
            instance_surfaces^,
            surfaces^,
        )
        return PbrtScene(
            world^,
            camera,
            settings,
            self.max_depth,
            self.integrator,
        )


def _parse_f32(text: String) raises -> Float32:
    var span = StringSpan(text)
    var parsed = parse_f32_at(span.as_bytes(), 0)
    if parsed.pos != span.byte_length():
        raise Error("invalid PBRT number: " + text)
    return parsed.value


def _parse_int(text: String) raises -> Int:
    var value = _parse_f32(text)
    var integer = Int(value)
    if Float32(integer) != value:
        raise Error("invalid PBRT integer: " + text)
    return integer


def _parse_params(mut lexer: _Lexer) raises -> _Parameters:
    var params = _Parameters()
    while lexer.next_is_quoted():
        var declaration = lexer.next().value
        if not lexer.has_next():
            raise Error("missing value for PBRT parameter " + declaration)
        var values = List[String]()
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
        params.entries.append(_Parameter(declaration^, values^))
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


def _matrix(values: ImmSpan[String, _]) raises -> _Transform:
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


def _surface(
    mut builder: _Builder, model: String, params: _Parameters
) raises -> SurfaceId[1]:
    if model == "diffuse" or model == "matte" or model == "coateddiffuse":
        # Bajo does not have a layered dielectric coating yet. Preserve the
        # diffuse substrate so official scenes remain useful in the meantime.
        return builder.surfaces.add_lambertian(
            params.color("reflectance", Color(0.5))
        )
    if model == "conductor" or model == "metal":
        var roughness = params.f32("float roughness", 0.05).clamp(0.0, 1.0)
        return builder.surfaces.add_metal(
            params.color("reflectance", Color(0.9)), roughness
        )
    if model == "dielectric" or model == "glass":
        return builder.surfaces.add_dielectric(params.f32("float eta", 1.5))
    raise Error("unsupported PBRT material: " + model)


def _shape(mut builder: _Builder, kind: String, params: _Parameters) raises:
    var surface = builder.state.surface.copy()
    if builder.state.area_light:
        surface = builder.surfaces.add_emissive(builder.state.emission)

    if kind == "sphere":
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
        add_sphere(
            builder.spheres,
            builder.sphere_surfaces,
            center,
            params.f32("float radius", 1.0) * sx,
            surface,
        )
        return

    if kind == "trianglemesh" or kind == "loopsubdiv":
        # The control cage is already an indexed triangle mesh. Full Loop
        # refinement can be added later without changing scene ingestion.
        var points = params.values("point3 P")
        if len(points) == 0:
            points = params.values("point P")
        if len(points) % 3 != 0:
            raise Error("PBRT trianglemesh P must contain xyz triples")
        var indices = params.values("integer indices")
        if len(indices) == 0:
            if len(points) % 9 != 0:
                raise Error(
                    "unindexed PBRT trianglemesh must contain triangles"
                )
            for base in range(0, len(points), 9):
                var p0 = builder.state.transform.point(
                    _PointL(
                        _parse_f32(points[base]),
                        _parse_f32(points[base + 1]),
                        _parse_f32(points[base + 2]),
                    )
                )
                var p1 = builder.state.transform.point(
                    _PointL(
                        _parse_f32(points[base + 3]),
                        _parse_f32(points[base + 4]),
                        _parse_f32(points[base + 5]),
                    )
                )
                var p2 = builder.state.transform.point(
                    _PointL(
                        _parse_f32(points[base + 6]),
                        _parse_f32(points[base + 7]),
                        _parse_f32(points[base + 8]),
                    )
                )
                if builder.state.reverse_orientation:
                    add_triangle(
                        builder.triangle_vertices,
                        builder.triangle_surfaces,
                        p0,
                        p2,
                        p1,
                        surface,
                    )
                else:
                    add_triangle(
                        builder.triangle_vertices,
                        builder.triangle_surfaces,
                        p0,
                        p1,
                        p2,
                        surface,
                    )
            return
        if len(indices) % 3 != 0:
            raise Error("PBRT trianglemesh indices must contain triples")
        var point_count = len(points) / 3
        for base in range(0, len(indices), 3):
            var i0 = _parse_int(indices[base])
            var i1 = _parse_int(indices[base + 1])
            var i2 = _parse_int(indices[base + 2])
            if (
                i0 < 0
                or i0 >= point_count
                or i1 < 0
                or i1 >= point_count
                or i2 < 0
                or i2 >= point_count
            ):
                raise Error("PBRT trianglemesh index is out of range")
            var p0 = builder.state.transform.point(
                _PointL(
                    _parse_f32(points[3 * i0]),
                    _parse_f32(points[3 * i0 + 1]),
                    _parse_f32(points[3 * i0 + 2]),
                )
            )
            var p1 = builder.state.transform.point(
                _PointL(
                    _parse_f32(points[3 * i1]),
                    _parse_f32(points[3 * i1 + 1]),
                    _parse_f32(points[3 * i1 + 2]),
                )
            )
            var p2 = builder.state.transform.point(
                _PointL(
                    _parse_f32(points[3 * i2]),
                    _parse_f32(points[3 * i2 + 1]),
                    _parse_f32(points[3 * i2 + 2]),
                )
            )
            if builder.state.reverse_orientation:
                add_triangle(
                    builder.triangle_vertices,
                    builder.triangle_surfaces,
                    p0,
                    p2,
                    p1,
                    surface,
                )
            else:
                add_triangle(
                    builder.triangle_vertices,
                    builder.triangle_surfaces,
                    p0,
                    p1,
                    p2,
                    surface,
                )
        return
    raise Error("unsupported PBRT shape: " + kind)


def _fixed_f32(mut lexer: _Lexer, count: Int) raises -> List[Float32]:
    var values = List[Float32](capacity=count)
    for _ in range(count):
        values.append(_parse_f32(lexer.next().value))
    return values^


def _bracket_values(mut lexer: _Lexer) raises -> List[String]:
    if lexer.next().value != "[":
        raise Error("expected '[' in PBRT transform")
    var values = List[String]()
    while True:
        var token = lexer.next()
        if token.value == "]":
            return values^
        values.append(token.value)


def _parse_text[
    Loader: PbrtTextLoader
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
        var command = command_token.value

        if command == "LookAt":
            var v = _fixed_f32(lexer, 9)
            builder.camera_origin = _PointW(v[0], v[1], v[2])
            builder.camera_target = _PointW(v[3], v[4], v[5])
            builder.camera_up = _VecW(v[6], v[7], v[8])
        elif command == "Camera":
            var kind = lexer.next().value
            if kind != "perspective":
                raise Error("only PBRT perspective cameras are supported")
            var params = _parse_params(lexer)
            builder.camera_fov = params.f32("float fov", 45.0)
        elif command == "Film":
            _ = (
                lexer.next()
            )  # Film implementation; rgb/image are equivalent here.
            var params = _parse_params(lexer)
            builder.image_width = params.integer("integer xresolution", 640)
            builder.image_height = params.integer("integer yresolution", 480)
        elif command == "Sampler":
            _ = lexer.next()
            var params = _parse_params(lexer)
            builder.samples_per_pixel = params.integer(
                "integer pixelsamples", 16
            )
        elif command == "Integrator":
            builder.integrator = lexer.next().value
            var params = _parse_params(lexer)
            builder.max_depth = params.integer("integer maxdepth", 8)
        elif command == "PixelFilter" or command == "Accelerator":
            # Bajo supplies these implementation details itself, but consuming
            # their declarations keeps ordinary PBRT scene headers portable.
            _ = lexer.next()
            _ = _parse_params(lexer)
        elif command == "ColorSpace":
            var color_space = lexer.next().value
            if color_space != "srgb":
                raise Error("only the PBRT sRGB color space is supported")
        elif command == "Option":
            # Options affect pbrt's runtime rather than the scene description.
            _ = _parse_params(lexer)
        elif command == "WorldBegin":
            builder.state.transform = _Transform.identity()
        elif command == "AttributeBegin":
            builder.attribute_stack.append(builder.state.copy())
        elif command == "AttributeEnd":
            if len(builder.attribute_stack) == 0:
                raise Error("PBRT AttributeEnd without AttributeBegin")
            builder.state = builder.attribute_stack.pop()
        elif command == "TransformBegin":
            builder.transform_stack.append(builder.state.transform.copy())
        elif command == "TransformEnd":
            if len(builder.transform_stack) == 0:
                raise Error("PBRT TransformEnd without TransformBegin")
            builder.state.transform = builder.transform_stack.pop()
        elif command == "Identity":
            builder.state.transform = _Transform.identity()
        elif command == "Translate":
            var v = _fixed_f32(lexer, 3)
            builder.state.transform = _compose(
                builder.state.transform, _translation(v[0], v[1], v[2])
            )
        elif command == "Scale":
            var v = _fixed_f32(lexer, 3)
            builder.state.transform = _compose(
                builder.state.transform, _scale(v[0], v[1], v[2])
            )
        elif command == "Rotate":
            var v = _fixed_f32(lexer, 4)
            builder.state.transform = _compose(
                builder.state.transform, _rotation(v[0], v[1], v[2], v[3])
            )
        elif command == "Transform":
            builder.state.transform = _matrix(_bracket_values(lexer))
        elif command == "ConcatTransform":
            builder.state.transform = _compose(
                builder.state.transform, _matrix(_bracket_values(lexer))
            )
        elif command == "ReverseOrientation":
            builder.state.reverse_orientation = (
                not builder.state.reverse_orientation
            )
        elif command == "Material":
            var model = lexer.next().value
            builder.state.surface = _surface(
                builder, model, _parse_params(lexer)
            )
        elif command == "MakeNamedMaterial":
            var name = lexer.next().value
            var params = _parse_params(lexer)
            var model = params.string("string type", "diffuse")
            builder.named_materials[name] = _surface(builder, model, params)
        elif command == "NamedMaterial":
            var name = lexer.next().value
            if name not in builder.named_materials:
                raise Error("unknown PBRT named material: " + name)
            builder.state.surface = builder.named_materials[name].copy()
        elif command == "AreaLightSource":
            var model = lexer.next().value
            if model != "diffuse":
                raise Error("only diffuse PBRT area lights are supported")
            var params = _parse_params(lexer)
            builder.state.area_light = True
            builder.state.emission = params.color("L", Color(1.0)) * params.f32(
                "float scale", 1.0
            )
        elif command == "Shape":
            _shape(builder, lexer.next().value, _parse_params(lexer))
        elif command == "Include":
            var include_name = lexer.next().value
            var include_path = std.os.path.join(
                std.os.path.dirname(path), include_name
            )
            _parse_text(
                builder,
                loader.read_text(include_path),
                include_path,
                loader,
                depth + 1,
            )
        elif command == "WorldEnd":
            pass
        else:
            raise Error(
                t"unsupported PBRT directive '{command}' at line"
                t" {command_token.line}"
            )


def _parse_pbrt[
    Loader: PbrtTextLoader
](text: String, path: String, loader: Loader) raises -> PbrtScene:
    var builder = _Builder()
    _parse_text(builder, text, path, loader, 0)
    return builder.finish()
