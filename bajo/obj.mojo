from std.pathlib import Path
from std.os import path as os_path

comptime TAB = UInt8(ord("\t"))
comptime SPACE = UInt8(ord(" "))
comptime CR = UInt8(ord("\r"))
comptime ZERO = UInt8(ord("0"))
comptime NINE = UInt8(ord("9"))
comptime PLUS = UInt8(ord("+"))
comptime MINUS = UInt8(ord("-"))
comptime SLASH = UInt8(ord("/"))
comptime DOT = UInt8(ord("."))
comptime HASH = UInt8(ord("#"))
comptime CHAR_e = UInt8(ord("e"))
comptime CHAR_E = UInt8(ord("E"))


@fieldwise_init
struct ObjTexture(Copyable):
    var name: String
    var path: String


struct ObjMaterial(Copyable):
    var name: String

    var Ka: Tuple[Float32, Float32, Float32]
    var Kd: Tuple[Float32, Float32, Float32]
    var Ks: Tuple[Float32, Float32, Float32]
    var Ke: Tuple[Float32, Float32, Float32]
    var Kt: Tuple[Float32, Float32, Float32]
    var Ns: Float32
    var Ni: Float32
    var Tf: Tuple[Float32, Float32, Float32]
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

        self.Ka = (0.0, 0.0, 0.0)
        self.Kd = (1.0, 1.0, 1.0)
        self.Ks = (0.0, 0.0, 0.0)
        self.Ke = (0.0, 0.0, 0.0)
        self.Kt = (0.0, 0.0, 0.0)
        self.Ns = 1.0
        self.Ni = 1.0
        self.Tf = (1.0, 1.0, 1.0)
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
        self.positions = [0.0, 0.0, 0.0]
        self.texcoords = [0.0, 0.0]
        self.normals = [0.0, 0.0, 1.0]
        self.colors = List[Float32]()
        self.face_vertices = List[Int]()
        self.face_materials = List[Int]()
        self.face_lines = List[UInt8]()
        self.indices = List[ObjIndex]()
        self.materials = List[ObjMaterial]()
        self.material_names = Dict[String, Int]()
        self.textures = [ObjTexture("", "")]
        self.texture_names = Dict[String, Int]()
        self.objects = List[ObjGroup]()
        self.groups = List[ObjGroup]()
        self.current_material = 0
        self.current_object = ObjGroup("", 0, 0, 0)
        self.current_group = ObjGroup("", 0, 0, 0)

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
        var path = os_path.join(base, name)

        if path in self.texture_names:
            return self.texture_names[path]

        var idx = len(self.textures)
        self.textures.append(ObjTexture(name, path))
        self.texture_names[path] = idx
        return idx

    def push_color(mut self, r: Float32, g: Float32, b: Float32):
        var target_before_this_color = len(self.positions) - 3
        while len(self.colors) < target_before_this_color:
            self.colors.append(1.0)
        self.colors.extend([r, g, b])

    def push_element(
        mut self, var verts: List[ObjIndex], is_line: Bool = False
    ):
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

        self.indices.extend(verts^)

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
        self.files[path] = text

    def read_text(self, path: String) raises -> String:
        if path in self.files:
            return self.files[path]
        raise Error("MemoryObjTextLoader: file not found: " + path)


@always_inline
def _parse_i32[o: Origin](token: StringSlice[o]) -> Int:
    var length = token.byte_length()
    if length == 0:
        return 0

    var ptr = token.unsafe_ptr()
    var p = 0
    var sign = 1
    if ptr.load(p) == MINUS:
        sign = -1
        p += 1
    elif ptr.load(p) == PLUS:
        p += 1

    var num = 0
    while p < length:
        var b = ptr.load(p)
        if b >= ZERO and b <= NINE:
            num = num * 10 + Int(b - ZERO)
            p += 1
        else:
            break

    return sign * num


@always_inline
def _parse_f32[o: Origin](token: StringSlice[o]) -> Float32:
    var bytes = token.as_bytes()
    var length = token.byte_length()
    if length == 0:
        return 0.0

    var p = 0
    var sign: Float64 = 1.0

    if bytes[p] == MINUS:
        sign = -1.0
        p += 1
    elif bytes[p] == PLUS:
        p += 1

    var num: Float64 = 0.0
    while p < length:
        var b = bytes[p]
        if b >= ZERO and b <= NINE:
            num = num * 10.0 + Float64(Int(b - ZERO))
            p += 1
        else:
            break

    if p < length and bytes[p] == DOT:
        p += 1
        var fra: Float64 = 0.0
        var div: Float64 = 1.0
        while p < length:
            var b = bytes[p]
            if b >= ZERO and b <= NINE:
                fra = fra * 10.0 + Float64(Int(b - ZERO))
                div *= 10.0
                p += 1
            else:
                break
        num += fra / div

    if p < length and (bytes[p] == CHAR_e or bytes[p] == CHAR_E):
        p += 1
        var exp_sign = 1
        if p < length and bytes[p] == MINUS:
            exp_sign = -1
            p += 1
        elif p < length and bytes[p] == PLUS:
            p += 1

        var eval = 0
        while p < length:
            var b = bytes[p]
            if b >= ZERO and b <= NINE:
                eval = eval * 10 + Int(b - ZERO)
                p += 1
            else:
                break

        if eval > 0:
            var power: Float64 = 1.0
            for _ in range(eval):
                power *= 10.0
            if exp_sign == 1:
                num *= power
            else:
                num /= power

    return Float32(sign * num)


def _fix_index(raw: Int, count_with_dummy: Int) -> Int:
    if raw > 0:
        return raw
    if raw < 0:
        return count_with_dummy + raw
    return 0


@always_inline
def _parse_index[o: Origin](token: StringSlice[o], mesh: ObjMesh) -> ObjIndex:
    var length = token.byte_length()
    if length == 0:
        return ObjIndex(0, 0, 0)

    var ptr = token.unsafe_ptr()
    var p = 0

    var p_raw = 0
    var t_raw = 0
    var n_raw = 0

    # position
    var sign = 1
    if p < length and ptr.load(p) == MINUS:
        sign = -1
        p += 1
    while p < length:
        var b = ptr.load(p)
        if b == SLASH:
            break
        p_raw = p_raw * 10 + Int(b - ZERO)
        p += 1
    p_raw *= sign

    # texcoord
    if p < length and ptr.load(p) == SLASH:
        p += 1
        sign = 1
        if p < length and ptr.load(p) == MINUS:
            sign = -1
            p += 1
        while p < length:
            var b = ptr.load(p)
            if b == SLASH:
                break
            t_raw = t_raw * 10 + Int(b - ZERO)
            p += 1
        t_raw *= sign

        # normal
        if p < length and ptr.load(p) == SLASH:
            p += 1
            sign = 1
            if p < length and ptr.load(p) == MINUS:
                sign = -1
                p += 1
            while p < length:
                var b = ptr.load(p)
                n_raw = n_raw * 10 + Int(b - ZERO)
                p += 1
            n_raw *= sign

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


def _map_name_from_tail[o: Origin](mut tokens: TokenIterator[o]) -> String:
    var candidate = ""
    while tokens.has_next():
        var tok = tokens.next()
        if tok.byte_length() == 0:
            continue
        if tok.as_bytes()[0] == MINUS:
            continue
        candidate = String(tok)
    return candidate


def _read_triple[
    o: Origin
](mut tokens: TokenIterator[o]) -> Tuple[Float32, Float32, Float32]:
    return (
        _parse_f32(tokens.next()),
        _parse_f32(tokens.next()),
        _parse_f32(tokens.next()),
    )


def _read_mtl_text(mut mesh: ObjMesh, base: String, text: String) raises:
    var current = ObjMaterial()
    var have_current = False
    var found_d = False

    var text_slice = StringSlice(text)
    var text_len = text_slice.byte_length()
    var line_start = 0

    while line_start < text_len:
        var line_end = text_slice.find("\n", line_start)
        if line_end == -1:
            line_end = text_len

        var raw_line = text_slice[byte=line_start:line_end]
        line_start = line_end + 1

        var tokens = TokenIterator(raw_line)
        if not tokens.has_next():
            continue

        var tag = tokens.next()

        if tag == "newmtl":
            if have_current:
                _ = mesh.upsert_material(current)
            var name = tokens.joined_rest_of_line()
            current = ObjMaterial(name, fallback=False)
            have_current = True
            found_d = False
            continue

        if not have_current:
            continue

        if tag == "Ka":
            current.Ka = _read_triple(tokens)
        elif tag == "Kd":
            current.Kd = _read_triple(tokens)
        elif tag == "Ks":
            current.Ks = _read_triple(tokens)
        elif tag == "Ke":
            current.Ke = _read_triple(tokens)
        elif tag == "Kt":
            current.Kt = _read_triple(tokens)
        elif tag == "Tf":
            current.Tf = _read_triple(tokens)
        elif tag == "Ns":
            current.Ns = _parse_f32(tokens.next())
        elif tag == "Ni":
            current.Ni = _parse_f32(tokens.next())
        elif tag == "illum":
            current.illum = _parse_i32(tokens.next())
        elif tag == "d":
            current.d = _parse_f32(tokens.next())
            found_d = True
        elif tag == "Tr":
            if not found_d:
                current.d = 1.0 - _parse_f32(tokens.next())
        elif tag == "map_Ka":
            var name = _map_name_from_tail(tokens)
            if name.byte_length() > 0:
                current.map_Ka = mesh.add_texture(name, base)
        elif tag == "map_Kd":
            var name = _map_name_from_tail(tokens)
            if name.byte_length() > 0:
                current.map_Kd = mesh.add_texture(name, base)
        elif tag == "map_Ks":
            var name = _map_name_from_tail(tokens)
            if name.byte_length() > 0:
                current.map_Ks = mesh.add_texture(name, base)
        elif tag == "map_Ke":
            var name = _map_name_from_tail(tokens)
            if name.byte_length() > 0:
                current.map_Ke = mesh.add_texture(name, base)
        elif tag == "map_Kt":
            var name = _map_name_from_tail(tokens)
            if name.byte_length() > 0:
                current.map_Kt = mesh.add_texture(name, base)
        elif tag == "map_Ns":
            var name = _map_name_from_tail(tokens)
            if name.byte_length() > 0:
                current.map_Ns = mesh.add_texture(name, base)
        elif tag == "map_Ni":
            var name = _map_name_from_tail(tokens)
            if name.byte_length() > 0:
                current.map_Ni = mesh.add_texture(name, base)
        elif tag == "map_d":
            var name = _map_name_from_tail(tokens)
            if name.byte_length() > 0:
                current.map_d = mesh.add_texture(name, base)
        elif tag == "map_bump" or tag == "bump":
            var name = _map_name_from_tail(tokens)
            if name.byte_length() > 0:
                current.map_bump = mesh.add_texture(name, base)

    if have_current:
        _ = mesh.upsert_material(current)


def _read_mtl_file[
    Loader: ObjTextLoader
](mut mesh: ObjMesh, obj_path: String, mtl_name: String, loader: Loader) raises:
    var base = os_path.dirname(obj_path)
    var mtl_path = os_path.join(base, mtl_name)
    var text = loader.read_text(mtl_path)
    _read_mtl_text(mesh, os_path.dirname(mtl_path), text)


struct TokenIterator[o: Origin]:
    var line: StringSlice[Self.o]
    var ptr: UnsafePointer[UInt8, Self.o]
    var pos: Int
    var length: Int

    def __init__(out self, line: StringSlice[Self.o]):
        self.line = line
        self.ptr = line.unsafe_ptr()
        self.pos = 0
        self.length = line.byte_length()

        # skip '#' lines
        for i in range(self.length):
            if self.ptr.load(i) == HASH:
                self.length = i
                break

    @always_inline
    def has_next(self) -> Bool:
        var p = self.pos
        while p < self.length:
            var b = self.ptr.load(p)
            if b != SPACE and b != TAB and b != CR:
                return True
            p += 1
        return False

    @always_inline
    def next(mut self) -> StringSlice[Self.o]:
        # skip leading whitespace
        while self.pos < self.length:
            var b = self.ptr.load(self.pos)
            if b != SPACE and b != TAB and b != CR:
                break
            self.pos += 1

        var start = self.pos

        # until next whitespace
        while self.pos < self.length:
            var b = self.ptr.load(self.pos)
            if b == SPACE or b == TAB or b == CR:
                break
            self.pos += 1

        return self.line[byte = start : self.pos]

    def joined_rest_of_line(mut self) -> String:
        var out = ""
        var first = True
        while self.has_next():
            if not first:
                out += " "
            out += String(self.next())
            first = False
        return out


def parse_obj_text_with_loader[
    Loader: ObjTextLoader
](path: String, text: String, loader: Loader) raises -> ObjMesh:
    var mesh = ObjMesh()

    # Pre-allocate memory to avoid vector resizing
    var est_elements = text.byte_length() // 50
    mesh.positions.reserve(est_elements * 3)
    mesh.indices.reserve(est_elements * 6)

    var text_slice = StringSlice(text)
    var text_len = text_slice.byte_length()
    var line_start = 0

    while line_start < text_len:
        var line_end = text_slice.find("\n", line_start)
        if line_end == -1:
            line_end = text_len

        var raw_line = text_slice[byte=line_start:line_end]
        line_start = line_end + 1

        var tokens = TokenIterator(raw_line)
        if not tokens.has_next():
            continue

        var tag = tokens.next()

        if tag == "v":
            var v1 = tokens.next()
            var v2 = tokens.next()
            var v3 = tokens.next()

            if (
                v1.byte_length() > 0
                and v2.byte_length() > 0
                and v3.byte_length() > 0
            ):
                mesh.positions.append(_parse_f32(v1))
                mesh.positions.append(_parse_f32(v2))
                mesh.positions.append(_parse_f32(v3))

            var v4 = tokens.next()
            var v5 = tokens.next()
            var v6 = tokens.next()
            if (
                v4.byte_length() > 0
                and v5.byte_length() > 0
                and v6.byte_length() > 0
            ):
                mesh.push_color(
                    _parse_f32(v4),
                    _parse_f32(v5),
                    _parse_f32(v6),
                )

        elif tag == "vt":
            var v1 = tokens.next()
            var v2 = tokens.next()
            if v1.byte_length() > 0 and v2.byte_length() > 0:
                mesh.texcoords.append(_parse_f32(v1))
                mesh.texcoords.append(_parse_f32(v2))

        elif tag == "vn":
            var v1 = tokens.next()
            var v2 = tokens.next()
            var v3 = tokens.next()
            if (
                v1.byte_length() > 0
                and v2.byte_length() > 0
                and v3.byte_length() > 0
            ):
                mesh.normals.append(_parse_f32(v1))
                mesh.normals.append(_parse_f32(v2))
                mesh.normals.append(_parse_f32(v3))

        elif tag == "f" or tag == "l":
            var verts = List[ObjIndex](capacity=4)
            var valid = True
            while tokens.has_next():
                var tok = tokens.next()
                var idx = _parse_index(tok, mesh)
                if idx.p == 0:
                    valid = False
                    break
                verts.append(idx)

            if valid and len(verts) > 0:
                mesh.push_element(verts^, is_line=(tag == "l"))

        elif tag == "usemtl":
            var name = tokens.joined_rest_of_line()
            if name.byte_length() > 0:
                mesh.current_material = mesh.ensure_material(
                    name, fallback=True
                )

        elif tag == "g":
            mesh.begin_group(tokens.joined_rest_of_line())

        elif tag == "o":
            mesh.begin_object(tokens.joined_rest_of_line())

        elif tag == "mtllib":
            var mtl_name = tokens.joined_rest_of_line()
            if mtl_name.byte_length() > 0:
                try:
                    _read_mtl_file(mesh, path, mtl_name, loader)
                except:
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
    var total_indices = 0
    var face_count = len(mesh.face_vertices)
    for i in range(face_count):
        var n = mesh.face_vertices[i]
        if n >= 3:
            if len(mesh.face_lines) > 0:
                if mesh.face_lines[i] == 0:
                    total_indices += (n - 2) * 3
            else:
                total_indices += (n - 2) * 3

    var out = List[ObjIndex](length=total_indices, fill=ObjIndex(0, 0, 0))

    var dest_ptr = out.unsafe_ptr()
    var src_ptr = mesh.indices.unsafe_ptr()
    var write_idx = 0
    var offset = 0

    for f in range(face_count):
        var n = mesh.face_vertices[f]
        var is_line = False
        if len(mesh.face_lines) > 0:
            is_line = mesh.face_lines[f] != 0

        if not is_line and n >= 3:
            var first = src_ptr[offset]
            for i in range(1, n - 1):
                dest_ptr[write_idx] = first
                dest_ptr[write_idx + 1] = src_ptr[offset + i]
                dest_ptr[write_idx + 2] = src_ptr[offset + i + 1]
                write_idx += 3
        offset += n

    return out^
