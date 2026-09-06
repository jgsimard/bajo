import std.os.path
from max.algorithm import parallelize
from std.math import abs, cos, pi, sin, sqrt
from std.sys import num_logical_cores

from bajo.bvh import Camera, Instance, Sphere
from bajo.bvh.host_utils import compute_bounds
from bajo.core import Affine3f32, Point3f32, Vec3f32
from bajo.parser.number import parse_f32_at
from bajo.parser.ply import PlyMesh
from bajo.rt.types import (
    Color,
    ImageTexture,
    Integrator,
    NO_TEXTURE,
    RenderSettings,
    SceneBuilder,
    SurfaceId,
    SurfaceStore,
)
from bajo.rt.scene_description import SceneDescription

from bajo.parser.text_loader import TextLoader


comptime _Transform = Affine3f32[.LOCAL, .WORLD]
comptime _PointL = Point3f32[.LOCAL]
comptime _PointW = Point3f32[.WORLD]
comptime _VecL = Vec3f32[.LOCAL]
comptime _VecW = Vec3f32[.WORLD]


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
    var spheres: List[Sphere[.WORLD]]
    var sphere_surfaces: List[SurfaceId[1]]
    var triangle_vertices: List[_PointW]
    var triangle_surfaces: List[SurfaceId[1]]
    var triangle_meshes: List[List[_PointL]]
    var triangle_mesh_normals: List[List[Float32]]
    var triangle_mesh_texcoords: List[List[Float32]]
    var triangle_instances: List[Instance]
    var triangle_instance_surfaces: List[SurfaceId[1]]
    var surfaces: SurfaceStore
    var named_materials: Dict[String, SurfaceId[1]]
    var color_textures: Dict[String, _ColorTexture]
    var scalar_textures: Dict[String, _FloatTexture]
    var image_paths: List[String]
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
    var integrator: Integrator

    def __init__(out self):
        self.spheres = List[Sphere[.WORLD]]()
        self.sphere_surfaces = List[SurfaceId[1]]()
        self.triangle_vertices = List[_PointW]()
        self.triangle_surfaces = List[SurfaceId[1]]()
        self.triangle_meshes = List[List[_PointL]]()
        self.triangle_mesh_normals = List[List[Float32]]()
        self.triangle_mesh_texcoords = List[List[Float32]]()
        self.triangle_instances = List[Instance]()
        self.triangle_instance_surfaces = List[SurfaceId[1]]()
        self.surfaces = SurfaceStore()
        self.named_materials = Dict[String, SurfaceId[1]]()
        self.color_textures = Dict[String, _ColorTexture]()
        self.scalar_textures = Dict[String, _FloatTexture]()
        self.image_paths = List[String]()
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
        self.integrator = .PATH

    def add_sphere(
        mut self,
        center: _PointW,
        radius: Float32,
        surface: SurfaceId[1],
    ):
        self.spheres.append(Sphere[.WORLD](center, radius))
        self.sphere_surfaces.append(surface.copy())

    def add_triangle(
        mut self,
        v0: _PointW,
        v1: _PointW,
        v2: _PointW,
        surface: SurfaceId[1],
    ):
        self.triangle_vertices.append(v0)
        self.triangle_vertices.append(v1)
        self.triangle_vertices.append(v2)
        self.triangle_surfaces.append(surface.copy())

    def add_ply_mesh(
        mut self,
        mesh: PlyMesh,
        surface: SurfaceId[1],
    ) raises:
        if len(mesh.indices) == 0:
            raise Error("PBRT plymesh must contain at least one triangle")

        # Bajo's current triangle BLAS consumes triangle soup. Expand the PLY
        # index stream in local space and keep the PBRT transform on an instance.
        var vertices = List[_PointL](capacity=len(mesh.indices))
        var normals = List[Float32]()
        var texcoords = List[Float32]()
        if mesh.has_normals():
            normals = List[Float32](capacity=3 * len(mesh.indices))
        if mesh.has_texcoords():
            texcoords = List[Float32](capacity=2 * len(mesh.indices))
        for base in range(0, len(mesh.indices), 3):
            var i0 = Int(mesh.indices[base])
            var i1 = Int(mesh.indices[base + 1])
            var i2 = Int(mesh.indices[base + 2])
            if self.state.reverse_orientation:
                var tmp = i1
                i1 = i2
                i2 = tmp
            for lane in range(3):
                var index = i0
                if lane == 1:
                    index = i1
                elif lane == 2:
                    index = i2
                var position_base = 3 * index
                vertices.append(
                    _PointL(
                        mesh.positions[position_base],
                        mesh.positions[position_base + 1],
                        mesh.positions[position_base + 2],
                    )
                )
                if mesh.has_normals():
                    var normal_base = 3 * index
                    normals.append(mesh.normals[normal_base])
                    normals.append(mesh.normals[normal_base + 1])
                    normals.append(mesh.normals[normal_base + 2])
                if mesh.has_texcoords():
                    var texcoord_base = 2 * index
                    texcoords.append(mesh.texcoords[texcoord_base])
                    texcoords.append(mesh.texcoords[texcoord_base + 1])

        var mesh_idx = UInt32(len(self.triangle_meshes))
        var bounds = compute_bounds(vertices)
        self.triangle_meshes.append(vertices^)
        self.triangle_mesh_normals.append(normals^)
        self.triangle_mesh_texcoords.append(texcoords^)

        var instance = Instance()
        instance.transform = self.state.transform.copy()
        instance.bounds = bounds.apply_transform(self.state.transform)
        instance.blas_idx = mesh_idx
        instance.kind = .TRIANGLE
        self.triangle_instances.append(instance^)
        self.triangle_instance_surfaces.append(surface.copy())

    def finish[
        Loader: TextLoader
    ](deinit self, loader: Loader) raises -> SceneDescription:
        if len(self.attribute_stack) != 0 or len(self.transform_stack) != 0:
            raise Error("unclosed PBRT attribute or transform scope")
        if (
            len(self.spheres) == 0
            and len(self.triangle_vertices) == 0
            and len(self.triangle_instances) == 0
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
            var stored_idx = self.surfaces.add_image_texture(
                loaded_images[image_idx].take()
            )
            debug_assert(stored_idx == UInt32(image_idx))

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
        # PBRT owns these authoring buffers exclusively. Transfer them through
        # SceneBuilder so finalization validates in place without cloning data.
        var scene_builder = SceneBuilder(
            self.spheres^,
            self.sphere_surfaces^,
            self.triangle_vertices^,
            self.triangle_surfaces^,
            self.triangle_meshes^,
            self.triangle_mesh_normals^,
            self.triangle_mesh_texcoords^,
            self.triangle_instances^,
            self.triangle_instance_surfaces^,
            self.surfaces^,
        )
        var data = scene_builder^.finish()
        return SceneDescription(
            data^,
            camera,
            settings,
            self.integrator,
        )

    def abort(deinit self):
        """Consume and release a partially parsed scene after an error."""
        pass


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


def _color_texture_parameter(
    builder: _Builder,
    params: _Parameters,
    name: String,
    default: Color,
) raises -> _ColorTexture:
    var texture_name = params.string("texture " + name, "")
    if texture_name.byte_length() != 0:
        if texture_name not in builder.color_textures:
            raise Error("unknown PBRT spectrum texture: " + texture_name)
        return builder.color_textures[texture_name].copy()
    return _ColorTexture(params.color(name, default), NO_TEXTURE)


def _scalar_parameter(
    builder: _Builder,
    params: _Parameters,
    name: String,
    default: Float32,
) raises -> _FloatTexture:
    var texture_name = params.string("texture " + name, "")
    if texture_name.byte_length() != 0:
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
    value_type: String,
    implementation: String,
    params: _Parameters,
    source_path: String,
) raises:
    if value_type == "spectrum" or value_type == "color":
        var value: _ColorTexture
        if implementation == "constant":
            value = _ColorTexture(params.color("value", Color(1.0)), NO_TEXTURE)
        elif implementation == "scale":
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
                raise Error("PBRT scale of two image textures is not supported")
            value = _ColorTexture(tex.scale * scale.scale, tex.image_index)
            if value.image_index == NO_TEXTURE:
                value.image_index = scale.image_index
        elif implementation == "imagemap":
            var filename = params.string("string filename", "")
            if filename.byte_length() == 0:
                raise Error("PBRT imagemap texture requires a filename")
            var image_path = std.os.path.join(
                std.os.path.dirname(source_path), filename
            )
            var image_index = UInt32(len(builder.image_paths))
            builder.image_paths.append(image_path)
            value = _ColorTexture(Color(1.0), image_index)
        else:
            raise Error("unsupported PBRT spectrum texture: " + implementation)
        builder.color_textures[name] = value.copy()
        return

    if value_type == "float":
        var value: _FloatTexture
        if implementation == "constant":
            value = _FloatTexture(
                params.f32("float value", 1.0), NO_TEXTURE, 1.0, 1.0
            )
        elif implementation == "scale":
            var tex = _scalar_parameter(builder, params, "tex", 1.0)
            var scale = _scalar_parameter(builder, params, "scale", 1.0)
            if (
                tex.image_index != NO_TEXTURE
                and scale.image_index != NO_TEXTURE
            ):
                raise Error("PBRT scale of two image textures is not supported")
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
        elif implementation == "imagemap":
            var filename = params.string("string filename", "")
            if filename.byte_length() == 0:
                raise Error("PBRT imagemap texture requires a filename")
            var image_path = std.os.path.join(
                std.os.path.dirname(source_path), filename
            )
            var image_index = UInt32(len(builder.image_paths))
            builder.image_paths.append(image_path)
            value = _FloatTexture(
                params.f32("float scale", 1.0),
                image_index,
                params.f32("float uscale", 1.0),
                params.f32("float vscale", 1.0),
            )
        else:
            raise Error("unsupported PBRT float texture: " + implementation)
        builder.scalar_textures[name] = value.copy()
        return

    raise Error("unsupported PBRT texture value type: " + value_type)


def _surface(
    mut builder: _Builder, model: String, params: _Parameters
) raises -> SurfaceId[1]:
    if model == "coateddiffuse":
        var reflectance = _color_texture_parameter(
            builder, params, "reflectance", Color(0.5)
        )
        var displacement = _scalar_parameter(
            builder, params, "displacement", 0.0
        )
        var roughness = params.f32("float roughness", 0.0)
        var u_roughness = params.f32("float uroughness", roughness)
        var v_roughness = params.f32("float vroughness", roughness)
        roughness = sqrt(max(u_roughness, 0.0) * max(v_roughness, 0.0)).clamp(
            0.0, 1.0
        )
        if params.string("bool remaproughness", "true") != "false":
            roughness = _roughness_to_alpha(roughness)
        return builder.surfaces.add_coated_diffuse(
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
    if model == "diffuse" or model == "matte":
        var reflectance = _color_texture_parameter(
            builder, params, "reflectance", Color(0.5)
        )
        return builder.surfaces.add_lambertian(
            reflectance.scale, reflectance.image_index
        )
    if model == "conductor" or model == "metal":
        var roughness = params.f32("float roughness", 0.05).clamp(0.0, 1.0)
        var reflectance = _color_texture_parameter(
            builder, params, "reflectance", Color(0.9)
        )
        return builder.surfaces.add_metal(
            reflectance.scale, roughness, reflectance.image_index
        )
    if model == "dielectric" or model == "glass":
        return builder.surfaces.add_dielectric(params.f32("float eta", 1.5))
    raise Error("unsupported PBRT material: " + model)


def _shape[
    Loader: TextLoader
](
    mut builder: _Builder,
    kind: String,
    params: _Parameters,
    source_path: String,
    loader: Loader,
) raises:
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
        builder.add_sphere(
            center,
            params.f32("float radius", 1.0) * sx,
            surface,
        )
        return

    if kind == "plymesh":
        var filename = params.string("string filename", "")
        if filename.byte_length() == 0:
            raise Error("PBRT plymesh requires a string filename")
        var mesh_path = std.os.path.join(
            std.os.path.dirname(source_path), filename
        )
        var mesh = loader.read_ply_mesh(mesh_path)
        builder.add_ply_mesh(mesh, surface)
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
                    builder.add_triangle(
                        p0,
                        p2,
                        p1,
                        surface,
                    )
                else:
                    builder.add_triangle(
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
                builder.add_triangle(
                    p0,
                    p2,
                    p1,
                    surface,
                )
            else:
                builder.add_triangle(
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
    Loader: TextLoader
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
            var integrator_name = lexer.next().value
            if integrator_name != "path":
                raise Error("only the PBRT path integrator is supported")
            builder.integrator = .PATH
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
        elif command == "Texture":
            _texture(
                builder,
                lexer.next().value,
                lexer.next().value,
                lexer.next().value,
                _parse_params(lexer),
                path,
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
            _shape(
                builder,
                lexer.next().value,
                _parse_params(lexer),
                path,
                loader,
            )
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
    Loader: TextLoader
](text: String, path: String, loader: Loader) raises -> SceneDescription:
    var builder = _Builder()
    try:
        _parse_text(builder, text, path, loader, 0)
    except error:
        builder^.abort()
        raise error
    return builder^.finish(loader)


def _parse_camera_text[
    Loader: TextLoader
](
    mut builder: _Builder,
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
        var command = command_token.value
        if command == "WorldBegin":
            return True
        if command == "LookAt":
            var values = _fixed_f32(lexer, 9)
            builder.camera_origin = _PointW(values[0], values[1], values[2])
            builder.camera_target = _PointW(values[3], values[4], values[5])
            builder.camera_up = _VecW(values[6], values[7], values[8])
        elif command == "Camera":
            var kind = lexer.next().value
            if kind != "perspective":
                raise Error("only PBRT perspective cameras are supported")
            var params = _parse_params(lexer)
            builder.camera_fov = params.f32("float fov", 45.0)
        elif command == "Film":
            _ = lexer.next()
            _ = _parse_params(lexer)
        elif command == "Sampler":
            _ = lexer.next()
            _ = _parse_params(lexer)
        elif command == "Integrator":
            _ = lexer.next()
            _ = _parse_params(lexer)
        elif command == "PixelFilter" or command == "Accelerator":
            _ = lexer.next()
            _ = _parse_params(lexer)
        elif command == "ColorSpace":
            _ = lexer.next()
        elif command == "Option":
            _ = _parse_params(lexer)
        elif command == "Identity":
            pass
        elif command == "Translate" or command == "Scale":
            _ = _fixed_f32(lexer, 3)
        elif command == "Rotate":
            _ = _fixed_f32(lexer, 4)
        elif command == "Transform" or command == "ConcatTransform":
            _ = _bracket_values(lexer)
        elif command == "Include":
            var include_name = lexer.next().value
            var include_path = std.os.path.join(
                std.os.path.dirname(path), include_name
            )
            if _parse_camera_text(
                builder,
                loader.read_text(include_path),
                include_path,
                loader,
                depth + 1,
            ):
                return True
        else:
            raise Error(
                t"unsupported PBRT options directive '{command}' at line"
                t" {command_token.line}"
            )
    return False


def _parse_pbrt_camera[
    Loader: TextLoader
](text: String, path: String, loader: Loader) raises -> Camera:
    var builder = _Builder()
    try:
        _ = _parse_camera_text(builder, text, path, loader, 0)
        var camera = Camera.from_vfov(
            builder.camera_origin,
            builder.camera_target,
            builder.camera_up,
            builder.camera_fov,
        )
        builder^.abort()
        return camera
    except error:
        builder^.abort()
        raise error
