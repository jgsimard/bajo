# fast_obj.mojo
#
# Full practical Mojo port of fast_obj v1.3's public data model and parser behavior.
#
# What this ports:
# - OBJ: v, vt, vn, f, l, g, o, usemtl, mtllib
# - MTL: newmtl, Ka, Kd, Ks, Ke, Kt, Ns, Ni, Tf, d, Tr, illum,
#        map_Ka, map_Kd, map_Ks, map_Ke, map_Kt, map_Ns, map_Ni,
#        map_d, map_bump, bump
# - 1-based positions / texcoords / normals / textures with dummy slot 0
# - 0-based materials, matching fast_obj's face_materials convention
# - polygon faces are preserved; triangulation is a separate helper
# - object/group ranges and line-element flags are preserved
# - vertex colors follow fast_obj behavior: if any color appears, missing prior
#   colors are filled with white; if no colors appear, color_count is zero.
#
# Design note:
# This is a source-level Mojo port focused on behavior and API shape. It uses
# String.split() for clarity. For benchmark parity with the C library, replace
# the line/token parsing layer with a byte-slice scanner over Path.read_bytes().

from std.pathlib import Path


@fieldwise_init
struct ObjTexture(Copyable):
    var name: String
    var path: String


struct ObjMaterial(Copyable):
    var name: String

    var Ka0: Float32
    var Ka1: Float32
    var Ka2: Float32
    var Kd0: Float32
    var Kd1: Float32
    var Kd2: Float32
    var Ks0: Float32
    var Ks1: Float32
    var Ks2: Float32
    var Ke0: Float32
    var Ke1: Float32
    var Ke2: Float32
    var Kt0: Float32
    var Kt1: Float32
    var Kt2: Float32
    var Ns: Float32
    var Ni: Float32
    var Tf0: Float32
    var Tf1: Float32
    var Tf2: Float32
    var d: Float32
    var illum: Int
    var fallback: Bool

    var map_Ka: Int
    var map_Kd: Int
    var map_Ks: Int
    var map_Ke: Int
    var map_Kt: Int
    var map_Ns: Int
    var map_Ni: Int
    var map_d: Int
    var map_bump: Int

    def __init__(out self, name: String = "", fallback: Bool = False):
        self.name = name

        self.Ka0 = 0.0
        self.Ka1 = 0.0
        self.Ka2 = 0.0
        self.Kd0 = 1.0
        self.Kd1 = 1.0
        self.Kd2 = 1.0
        self.Ks0 = 0.0
        self.Ks1 = 0.0
        self.Ks2 = 0.0
        self.Ke0 = 0.0
        self.Ke1 = 0.0
        self.Ke2 = 0.0
        self.Kt0 = 0.0
        self.Kt1 = 0.0
        self.Kt2 = 0.0
        self.Ns = 1.0
        self.Ni = 1.0
        self.Tf0 = 1.0
        self.Tf1 = 1.0
        self.Tf2 = 1.0
        self.d = 1.0
        self.illum = 1
        self.fallback = fallback

        self.map_Ka = 0
        self.map_Kd = 0
        self.map_Ks = 0
        self.map_Ke = 0
        self.map_Kt = 0
        self.map_Ns = 0
        self.map_Ni = 0
        self.map_d = 0
        self.map_bump = 0

    def set_Ka(mut self, a: Float32, b: Float32, c: Float32):
        self.Ka0 = a
        self.Ka1 = b
        self.Ka2 = c

    def set_Kd(mut self, a: Float32, b: Float32, c: Float32):
        self.Kd0 = a
        self.Kd1 = b
        self.Kd2 = c

    def set_Ks(mut self, a: Float32, b: Float32, c: Float32):
        self.Ks0 = a
        self.Ks1 = b
        self.Ks2 = c

    def set_Ke(mut self, a: Float32, b: Float32, c: Float32):
        self.Ke0 = a
        self.Ke1 = b
        self.Ke2 = c

    def set_Kt(mut self, a: Float32, b: Float32, c: Float32):
        self.Kt0 = a
        self.Kt1 = b
        self.Kt2 = c

    def set_Tf(mut self, a: Float32, b: Float32, c: Float32):
        self.Tf0 = a
        self.Tf1 = b
        self.Tf2 = c


@fieldwise_init
struct ObjIndex(Copyable, TrivialRegisterPassable):
    var p: Int
    var t: Int
    var n: Int


@fieldwise_init
struct ObjGroup(Copyable):
    var name: String
    var face_count: Int
    var face_offset: Int
    var index_offset: Int


struct ObjMesh(Movable):
    var positions: List[Float32]
    var texcoords: List[Float32]
    var normals: List[Float32]
    var colors: List[Float32]

    var face_vertices: List[Int]
    var face_materials: List[Int]
    var face_lines: List[UInt8]
    var indices: List[ObjIndex]

    var materials: List[ObjMaterial]
    var material_names: Dict[String, Int]
    var textures: List[ObjTexture]
    var texture_names: Dict[String, Int]
    var objects: List[ObjGroup]
    var groups: List[ObjGroup]

    var current_material: Int
    var current_object: ObjGroup
    var current_group: ObjGroup

    def __init__(out self):
        self.positions = List[Float32]()
        self.texcoords = List[Float32]()
        self.normals = List[Float32]()
        self.colors = List[Float32]()
        self.face_vertices = List[Int]()
        self.face_materials = List[Int]()
        self.face_lines = List[UInt8]()
        self.indices = List[ObjIndex]()
        self.materials = List[ObjMaterial]()
        self.material_names = Dict[String, Int]()
        self.textures = List[ObjTexture]()
        self.texture_names = Dict[String, Int]()
        self.objects = List[ObjGroup]()
        self.groups = List[ObjGroup]()
        self.current_material = 0
        self.current_object = ObjGroup("", 0, 0, 0)
        self.current_group = ObjGroup("", 0, 0, 0)

        self.positions.append(0.0)
        self.positions.append(0.0)
        self.positions.append(0.0)
        self.texcoords.append(0.0)
        self.texcoords.append(0.0)
        self.normals.append(0.0)
        self.normals.append(0.0)
        self.normals.append(1.0)
        self.textures.append(ObjTexture("", ""))

    def position_count(self) -> Int:
        return len(self.positions) / 3

    def texcoord_count(self) -> Int:
        return len(self.texcoords) / 2

    def normal_count(self) -> Int:
        return len(self.normals) / 3

    def color_count(self) -> Int:
        return len(self.colors) / 3

    def face_count(self) -> Int:
        return len(self.face_vertices)

    def index_count(self) -> Int:
        return len(self.indices)

    def material_count(self) -> Int:
        return len(self.materials)

    def texture_count(self) -> Int:
        return len(self.textures)

    def object_count(self) -> Int:
        return len(self.objects)

    def group_count(self) -> Int:
        return len(self.groups)

    def actual_position_count(self) -> Int:
        return self.position_count() - 1

    def actual_texcoord_count(self) -> Int:
        return self.texcoord_count() - 1

    def actual_normal_count(self) -> Int:
        return self.normal_count() - 1

    def actual_texture_count(self) -> Int:
        return self.texture_count() - 1

    def _flush_object(mut self):
        if self.current_object.face_count > 0:
            self.objects.append(self.current_object.copy())
        self.current_object = ObjGroup(
            "", 0, len(self.face_vertices), len(self.indices)
        )

    def _flush_group(mut self):
        if self.current_group.face_count > 0:
            self.groups.append(self.current_group.copy())
        self.current_group = ObjGroup(
            "", 0, len(self.face_vertices), len(self.indices)
        )

    def begin_object(mut self, name: String):
        self._flush_object()
        self.current_object.name = name

    def begin_group(mut self, name: String):
        self._flush_group()
        self.current_group.name = name

    def finish(mut self):
        self._flush_group()
        self._flush_object()

    def ensure_material(
        mut self, name: String, fallback: Bool = True
    ) raises -> Int:
        if name in self.material_names:
            return self.material_names[name]
        var idx = len(self.materials)
        self.materials.append(ObjMaterial(name, fallback=fallback))
        self.material_names[name] = idx
        return idx

    def upsert_material(mut self, material: ObjMaterial) raises -> Int:
        if material.name in self.material_names:
            var idx = self.material_names[material.name]
            if self.materials[idx].fallback:
                self.materials[idx] = material.copy()
                return idx
        var idx = len(self.materials)
        self.materials.append(material.copy())
        if not (material.name in self.material_names):
            self.material_names[material.name] = idx
        return idx

    def add_texture(mut self, name: String, base: String) raises -> Int:
        var path = _fix_separators(_join_path(base, name))
        if path in self.texture_names:
            return self.texture_names[path]
        var idx = len(self.textures)
        self.textures.append(ObjTexture(name, path))
        self.texture_names[path] = idx
        return idx

    def push_position(mut self, x: Float32, y: Float32, z: Float32):
        self.positions.append(x)
        self.positions.append(y)
        self.positions.append(z)

    def push_color(mut self, r: Float32, g: Float32, b: Float32):
        var target_before_this_color = len(self.positions) - 3
        while len(self.colors) < target_before_this_color:
            self.colors.append(1.0)
        self.colors.append(r)
        self.colors.append(g)
        self.colors.append(b)

    def push_texcoord(mut self, u: Float32, v: Float32):
        self.texcoords.append(u)
        self.texcoords.append(v)

    def push_normal(mut self, x: Float32, y: Float32, z: Float32):
        self.normals.append(x)
        self.normals.append(y)
        self.normals.append(z)

    def push_element(mut self, verts: List[ObjIndex], is_line: Bool = False):
        var n = len(verts)
        if n == 0:
            return
        if not is_line and n < 3:
            return
        if is_line and n < 2:
            return

        self.face_vertices.append(n)
        self.face_materials.append(self.current_material)

        if is_line or len(self.face_lines) > 0:
            while len(self.face_lines) < len(self.face_vertices) - 1:
                self.face_lines.append(0)
            if is_line:
                self.face_lines.append(1)
            else:
                self.face_lines.append(0)

        for i in range(n):
            self.indices.append(verts[i])

        self.current_group.face_count += 1
        self.current_object.face_count += 1

    def print_summary(self):
        print("ObjMesh summary")
        print(
            t" - positions: {self.position_count()} including dummy; actual:"
            t" {self.actual_position_count()}"
        )
        print(
            t" - texcoords: {self.texcoord_count()} including dummy; actual:"
            t" {self.actual_texcoord_count()}"
        )
        print(
            t" - normals: {self.normal_count()} including dummy; actual:"
            t" {self.actual_normal_count()}"
        )
        print(t" - colors: {self.color_count()}")
        print(t" - faces/lines: {self.face_count()}")
        print(t" - indices: {self.index_count()}")
        print(t" - materials: {self.material_count()}")
        print(
            t" - textures: {self.texture_count()} including dummy; actual:"
            t" {self.actual_texture_count()}"
        )
        print(t" - objects: {self.object_count()}")
        print(t" - groups:  {self.group_count()}")


trait ObjTextLoader:
    def read_text(self, path: String) raises -> String:
        ...


@fieldwise_init
struct PathObjTextLoader(Copyable, ObjTextLoader):
    def read_text(self, path: String) raises -> String:
        return Path(path).read_text()


struct MemoryObjTextLoader(Movable, ObjTextLoader):
    var files: Dict[String, String]

    def __init__(out self):
        self.files = Dict[String, String]()

    def add_file(mut self, path: String, text: String):
        self.files[_fix_separators(path)] = text

    def read_text(self, path: String) raises -> String:
        var key = _fix_separators(path)
        if key in self.files:
            return self.files[key]
        raise Error("MemoryObjTextLoader: file not found: " + key)


def _parse_i32[o: Origin](token: StringSlice[o]) raises -> Int:
    return atol(token)


def _parse_f32[o: Origin](token: StringSlice[o]) raises -> Float32:
    return Float32(atof(token))


def _strip_comment[o: Origin](line: StringSlice[o]) -> StringSlice[o]:
    var end = line.byte_length()

    for i in range(line.byte_length()):
        if line[byte=i] == "#":
            end = i
            break

    return line[byte=:end].strip()


def _join_tokens[o: Origin](tokens: List[StringSlice[o]], start: Int) -> String:
    var out = ""
    for i in range(start, len(tokens)):
        if i > start:
            out += " "
        out += String(tokens[i])
    return String(out.strip())


def _is_abs_path(path: String) -> Bool:
    if path.byte_length() == 0:
        return False
    if path[byte=0] == "/" or path[byte=0] == "\\":
        return True
    if (
        path.byte_length() >= 3
        and path[byte=1] == ":"
        and (path[byte=2] == "/" or path[byte=2] == "\\")
    ):
        return True
    return False


def _base_dir_with_sep(path: String) -> String:
    var last = -1
    for i in range(path.byte_length()):
        if path[byte=i] == "/" or path[byte=i] == "\\":
            last = i
    if last < 0:
        return ""
    return String(path[byte = : last + 1])


def _fix_separators(path: String) -> String:
    var out = ""
    for i in range(path.byte_length()):
        if path[byte=i] == "\\":
            out += "/"
        else:
            out += String(path[byte=i])
    return out


def _join_path(base: String, child: String) -> String:
    if _is_abs_path(child):
        return _fix_separators(child)
    return _fix_separators(base + child)


def _fix_index(raw: Int, count_with_dummy: Int) -> Int:
    if raw > 0:
        return raw
    if raw < 0:
        return count_with_dummy + raw
    return 0


def _parse_index[
    o: Origin
](token: StringSlice[o], mesh: ObjMesh) raises -> ObjIndex:
    var parts = token.split("/")
    var p_raw = 0
    var t_raw = 0
    var n_raw = 0

    if len(parts) > 0 and parts[0].byte_length() > 0:
        p_raw = _parse_i32(parts[0])
    if len(parts) > 1 and parts[1].byte_length() > 0:
        t_raw = _parse_i32(parts[1])
    if len(parts) > 2 and parts[2].byte_length() > 0:
        n_raw = _parse_i32(parts[2])

    return ObjIndex(
        _fix_index(p_raw, mesh.position_count()),
        _fix_index(t_raw, mesh.texcoord_count()),
        _fix_index(n_raw, mesh.normal_count()),
    )


def _tail_after_tag[o: Origin](line: StringSlice[o], tag: String) -> String:
    var i = tag.byte_length()
    while i < line.byte_length() and (
        line[byte=i] == " " or line[byte=i] == "\t"
    ):
        i += 1

    if i >= line.byte_length():
        return ""

    return String(line[byte=i:].strip())


def _map_name_from_tail[
    o: Origin
](tokens: List[StringSlice[o]], start: Int) -> String:
    # fast_obj itself does not support full texture-map options. This port keeps
    # functional support for common option-bearing maps by choosing the last
    # non-option-looking token as the filename.
    var candidate = ""
    for i in range(start, len(tokens)):
        var tok = tokens[i]
        if tok.byte_length() == 0:
            continue
        if tok[byte=0] == "-":
            continue
        candidate = String(tok)
    return candidate


def _read_triple[
    o: Origin
](tokens: List[StringSlice[o]], start: Int) raises -> Tuple[
    Float32, Float32, Float32
]:
    return (
        _parse_f32(tokens[start]),
        _parse_f32(tokens[start + 1]),
        _parse_f32(tokens[start + 2]),
    )


def _read_mtl_text(mut mesh: ObjMesh, base: String, text: String) raises:
    var current = ObjMaterial()
    var have_current = False
    var found_d = False

    for raw_line in text.split("\n"):
        var line = _strip_comment(raw_line)
        if line.byte_length() == 0:
            continue

        var tokens = line.split()
        if len(tokens) == 0:
            continue

        var tag = tokens[0]

        if tag == "newmtl":
            if have_current:
                _ = mesh.upsert_material(current)
            var name = _join_tokens(tokens, 1)
            current = ObjMaterial(name, fallback=False)
            have_current = True
            found_d = False
            continue

        if not have_current:
            continue

        if tag == "Ka" and len(tokens) >= 4:
            var (a, b, c) = _read_triple(tokens, 1)
            current.set_Ka(a, b, c)
        elif tag == "Kd" and len(tokens) >= 4:
            var a, b, c = _read_triple(tokens, 1)
            current.set_Kd(a, b, c)
        elif tag == "Ks" and len(tokens) >= 4:
            var a, b, c = _read_triple(tokens, 1)
            current.set_Ks(a, b, c)
        elif tag == "Ke" and len(tokens) >= 4:
            var a, b, c = _read_triple(tokens, 1)
            current.set_Ke(a, b, c)
        elif tag == "Kt" and len(tokens) >= 4:
            var a, b, c = _read_triple(tokens, 1)
            current.set_Kt(a, b, c)
        elif tag == "Tf" and len(tokens) >= 4:
            var a, b, c = _read_triple(tokens, 1)
            current.set_Tf(a, b, c)
        elif tag == "Ns" and len(tokens) >= 2:
            current.Ns = _parse_f32(tokens[1])
        elif tag == "Ni" and len(tokens) >= 2:
            current.Ni = _parse_f32(tokens[1])
        elif tag == "illum" and len(tokens) >= 2:
            current.illum = _parse_i32(tokens[1])
        elif tag == "d" and len(tokens) >= 2:
            current.d = _parse_f32(tokens[1])
            found_d = True
        elif tag == "Tr" and len(tokens) >= 2:
            if not found_d:
                current.d = 1.0 - _parse_f32(tokens[1])
        elif tag == "map_Ka" and len(tokens) >= 2:
            var name = _map_name_from_tail(tokens, 1)
            if name.byte_length() > 0:
                current.map_Ka = mesh.add_texture(name, base)
        elif tag == "map_Kd" and len(tokens) >= 2:
            var name = _map_name_from_tail(tokens, 1)
            if name.byte_length() > 0:
                current.map_Kd = mesh.add_texture(name, base)
        elif tag == "map_Ks" and len(tokens) >= 2:
            var name = _map_name_from_tail(tokens, 1)
            if name.byte_length() > 0:
                current.map_Ks = mesh.add_texture(name, base)
        elif tag == "map_Ke" and len(tokens) >= 2:
            var name = _map_name_from_tail(tokens, 1)
            if name.byte_length() > 0:
                current.map_Ke = mesh.add_texture(name, base)
        elif tag == "map_Kt" and len(tokens) >= 2:
            var name = _map_name_from_tail(tokens, 1)
            if name.byte_length() > 0:
                current.map_Kt = mesh.add_texture(name, base)
        elif tag == "map_Ns" and len(tokens) >= 2:
            var name = _map_name_from_tail(tokens, 1)
            if name.byte_length() > 0:
                current.map_Ns = mesh.add_texture(name, base)
        elif tag == "map_Ni" and len(tokens) >= 2:
            var name = _map_name_from_tail(tokens, 1)
            if name.byte_length() > 0:
                current.map_Ni = mesh.add_texture(name, base)
        elif tag == "map_d" and len(tokens) >= 2:
            var name = _map_name_from_tail(tokens, 1)
            if name.byte_length() > 0:
                current.map_d = mesh.add_texture(name, base)
        elif (tag == "map_bump" or tag == "bump") and len(tokens) >= 2:
            var name = _map_name_from_tail(tokens, 1)
            if name.byte_length() > 0:
                current.map_bump = mesh.add_texture(name, base)
        else:
            pass

    if have_current:
        _ = mesh.upsert_material(current)


def _read_mtl_file[
    Loader: ObjTextLoader
](mut mesh: ObjMesh, obj_path: String, mtl_name: String, loader: Loader) raises:
    var base = _base_dir_with_sep(obj_path)
    var mtl_path = _join_path(base, mtl_name)
    var text = loader.read_text(mtl_path)
    _read_mtl_text(mesh, _base_dir_with_sep(mtl_path), text)


def parse_obj_text_with_loader[
    Loader: ObjTextLoader
](path: String, text: String, loader: Loader) raises -> ObjMesh:
    var mesh = ObjMesh()

    for raw_line in text.split("\n"):
        var line = _strip_comment(raw_line)
        if line.byte_length() == 0:
            continue

        var tokens = line.split()
        if len(tokens) == 0:
            continue

        var tag = tokens[0]

        if tag == "v":
            if len(tokens) < 4:
                continue
            mesh.push_position(
                _parse_f32(tokens[1]),
                _parse_f32(tokens[2]),
                _parse_f32(tokens[3]),
            )
            if len(tokens) >= 7:
                mesh.push_color(
                    _parse_f32(tokens[4]),
                    _parse_f32(tokens[5]),
                    _parse_f32(tokens[6]),
                )

        elif tag == "vt":
            if len(tokens) < 3:
                continue
            mesh.push_texcoord(
                _parse_f32(tokens[1]),
                _parse_f32(tokens[2]),
            )

        elif tag == "vn":
            if len(tokens) < 4:
                continue
            mesh.push_normal(
                _parse_f32(tokens[1]),
                _parse_f32(tokens[2]),
                _parse_f32(tokens[3]),
            )

        elif tag == "f" or tag == "l":
            var verts = List[ObjIndex](capacity=len(tokens) - 1)
            var valid = True
            for i in range(1, len(tokens)):
                var idx = _parse_index(tokens[i], mesh)
                if idx.p == 0:
                    valid = False
                    break
                verts.append(idx)
            if valid:
                mesh.push_element(verts, is_line=(tag == "l"))

        elif tag == "usemtl":
            if len(tokens) >= 2:
                var name = _join_tokens(tokens, 1)
                mesh.current_material = mesh.ensure_material(
                    name, fallback=True
                )

        elif tag == "g":
            var name = ""
            if len(tokens) > 1:
                name = _join_tokens(tokens, 1)
            mesh.begin_group(name)

        elif tag == "o":
            var name = ""
            if len(tokens) > 1:
                name = _join_tokens(tokens, 1)
            mesh.begin_object(name)

        elif tag == "mtllib":
            var mtl_name = _tail_after_tag(line, "mtllib")
            if mtl_name.byte_length() > 0:
                try:
                    _read_mtl_file(mesh, path, mtl_name, loader)
                except:
                    # Missing or invalid MTL files should not make the geometry unreadable.
                    pass

        else:
            pass

    if len(mesh.colors) > 0:
        while len(mesh.colors) < len(mesh.positions):
            mesh.colors.append(1.0)

    mesh.finish()
    return mesh^


def parse_obj_text(path: String, text: String) raises -> ObjMesh:
    return parse_obj_text_with_loader(path, text, PathObjTextLoader())


def read_obj_with_loader[
    Loader: ObjTextLoader
](path: String, loader: Loader) raises -> ObjMesh:
    var text = loader.read_text(path)
    return parse_obj_text_with_loader(path, text, loader)


def read_obj(path: String) raises -> ObjMesh:
    return read_obj_with_loader(path, PathObjTextLoader())


def read_obj_from_string(text: String) raises -> ObjMesh:
    return parse_obj_text("", text)


def read_obj_from_memory(
    path: String, mut loader: MemoryObjTextLoader
) raises -> ObjMesh:
    return read_obj_with_loader(path, loader)


def triangulated_indices(mesh: ObjMesh) -> List[ObjIndex]:
    var out = List[ObjIndex]()
    var offset = 0
    for f in range(len(mesh.face_vertices)):
        var n = mesh.face_vertices[f]
        var is_line = False
        if len(mesh.face_lines) > 0:
            is_line = mesh.face_lines[f] != 0
        if not is_line and n >= 3:
            var first = mesh.indices[offset]
            for i in range(1, n - 1):
                out.append(first)
                out.append(mesh.indices[offset + i])
                out.append(mesh.indices[offset + i + 1])
        offset += n
    return out^
