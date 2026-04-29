from std.pathlib import Path
from std.os import path as os_path


comptime TAB = UInt8(ord("\t"))
comptime SPACE = UInt8(ord(" "))
comptime CR = UInt8(ord("\r"))
comptime LF = UInt8(ord("\n"))

comptime ZERO = UInt8(ord("0"))
comptime NINE = UInt8(ord("9"))
comptime PLUS = UInt8(ord("+"))
comptime MINUS = UInt8(ord("-"))
comptime SLASH = UInt8(ord("/"))
comptime DOT = UInt8(ord("."))
comptime HASH = UInt8(ord("#"))

comptime CHAR_e = UInt8(ord("e"))
comptime CHAR_E = UInt8(ord("E"))
comptime CHAR_v = UInt8(ord("v"))
comptime CHAR_t = UInt8(ord("t"))
comptime CHAR_n = UInt8(ord("n"))
comptime CHAR_f = UInt8(ord("f"))
comptime CHAR_l = UInt8(ord("l"))
comptime CHAR_g = UInt8(ord("g"))
comptime CHAR_o = UInt8(ord("o"))


# Data model
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
    # arrays use OBJ-style dummy element at index 0 for p/t/n/texture
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

    # parser state
    var _current_material: Int
    var _current_object: ObjGroup
    var _current_group: ObjGroup

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

        self._current_material = 0
        self._current_object = ObjGroup("", 0, 0, 0)
        self._current_group = ObjGroup("", 0, 0, 0)

    def position_count(self, include_dummy: Bool = True) -> Int:
        var n = len(self.positions) / 3
        if include_dummy:
            return n
        return n - 1

    def texcoord_count(self, include_dummy: Bool = True) -> Int:
        var n = len(self.texcoords) / 2
        if include_dummy:
            return n
        return n - 1

    def normal_count(self, include_dummy: Bool = True) -> Int:
        var n = len(self.normals) / 3
        if include_dummy:
            return n
        return n - 1

    def color_count(self) -> Int:
        return len(self.colors) / 3

    def face_count(self) -> Int:
        return len(self.face_vertices)

    def index_count(self) -> Int:
        return len(self.indices)

    def material_count(self) -> Int:
        return len(self.materials)

    def texture_count(self, include_dummy: Bool = True) -> Int:
        var n = len(self.textures)
        if include_dummy:
            return n
        return n - 1

    def object_count(self) -> Int:
        return len(self.objects)

    def group_count(self) -> Int:
        return len(self.groups)

    def print_summary(self):
        print("ObjMesh summary")
        print(
            t" - positions: {self.position_count()} including dummy; actual:"
            t" {self.position_count(include_dummy=False)}"
        )
        print(
            t" - texcoords: {self.texcoord_count()} including dummy; actual:"
            t" {self.texcoord_count(include_dummy=False)}"
        )
        print(
            t" - normals: {self.normal_count()} including dummy; actual:"
            t" {self.normal_count(include_dummy=False)}"
        )
        print(t" - colors: {self.color_count()}")
        print(t" - faces/lines: {self.face_count()}")
        print(t" - indices: {self.index_count()}")
        print(t" - materials: {self.material_count()}")
        print(
            t" - textures: {self.texture_count()} including dummy; actual:"
            t" {self.texture_count(include_dummy=False)}"
        )
        print(t" - objects: {self.object_count()}")
        print(t" - groups:  {self.group_count()}")

    def _flush_object(mut self):
        if self._current_object.face_count > 0:
            self.objects.append(self._current_object.copy())
        self._current_object = ObjGroup(
            "", 0, len(self.face_vertices), len(self.indices)
        )

    def _flush_group(mut self):
        if self._current_group.face_count > 0:
            self.groups.append(self._current_group.copy())
        self._current_group = ObjGroup(
            "", 0, len(self.face_vertices), len(self.indices)
        )

    def _begin_object(mut self, name: String):
        self._flush_object()
        self._current_object.name = name

    def _begin_group(mut self, name: String):
        self._flush_group()
        self._current_group.name = name

    def _finish(mut self):
        self._flush_group()
        self._flush_object()

    def _ensure_material(
        mut self, name: String, fallback: Bool = True
    ) raises -> Int:
        if name in self.material_names:
            return self.material_names[name]

        var idx = len(self.materials)
        self.materials.append(ObjMaterial(name, fallback=fallback))
        self.material_names[name] = idx
        return idx

    def _upsert_material(mut self, material: ObjMaterial) raises -> Int:
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

    def _add_texture(mut self, name: String, base: String) raises -> Int:
        var path = os_path.join(base, name)

        if path in self.texture_names:
            return self.texture_names[path]

        var idx = len(self.textures)
        self.textures.append(ObjTexture(name, path))
        self.texture_names[path] = idx
        return idx

    def _push_color(mut self, r: Float32, g: Float32, b: Float32):
        var target_before_this_color = len(self.positions) - 3
        while len(self.colors) < target_before_this_color:
            self.colors.append(1.0)
        self.colors.extend([r, g, b])

    @always_inline
    def _push_element_meta(mut self, n: Int, is_line: Bool = False):
        if n == 0:
            return
        if not is_line and n < 3:
            return
        if is_line and n < 2:
            return

        self.face_vertices.append(n)
        self.face_materials.append(self._current_material)

        if is_line or len(self.face_lines) > 0:
            while len(self.face_lines) < len(self.face_vertices) - 1:
                self.face_lines.append(UInt8(0))

            if is_line:
                self.face_lines.append(UInt8(1))
            else:
                self.face_lines.append(UInt8(0))

        self._current_group.face_count += 1
        self._current_object.face_count += 1


# text loading
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


# public API
# File path, normal filesystem.
def read_obj(path: String) raises -> ObjMesh:
    var loader = PathObjTextLoader()
    return read_obj(path, loader)


# File path, custom loader.
def read_obj[
    Loader: ObjTextLoader
](path: String, loader: Loader) raises -> ObjMesh:
    var text = loader.read_text(path)
    return parse_obj(text, path, loader)


# Raw OBJ text. MTL files are intentionally ignored because no loader is provided.
def parse_obj(text: String, path: String = "") raises -> ObjMesh:
    var loader = MemoryObjTextLoader()
    return parse_obj(text, path, loader)


# Raw OBJ text plus loader for mtllib resolution.
def parse_obj[
    Loader: ObjTextLoader
](text: String, path: String, loader: Loader) raises -> ObjMesh:
    return _parse_obj_text(path, text, loader)


# parsing primitive
@always_inline
def _is_ws(b: UInt8) -> Bool:
    return b == SPACE or b == TAB or b == CR


@always_inline
def _is_digit(b: UInt8) -> Bool:
    return b >= ZERO and b <= NINE


@always_inline
def _fix_index(raw: Int, count_with_dummy: Int) -> Int:
    if raw > 0:
        return raw
    if raw < 0:
        return count_with_dummy + raw
    return 0


# MTL parsing
def _map_name_from_tail[o: Origin](mut cur: ObjLineCursor[o]) -> String:
    var candidate = ""
    while cur.has_next():
        var word = cur.next_word()
        if word.byte_length() == 0:
            break
        # Skip flags like '-clamp' or '-s'
        if word.unsafe_ptr().load(0) == MINUS:
            continue
        candidate = String(word)
    return candidate


def _read_mtl_text(mut mesh: ObjMesh, base: String, text: String) raises:
    var current = ObjMaterial()
    var have_current = False
    var found_d = False

    var text_slice = StringSlice(text)
    var ptr = text_slice.unsafe_ptr()
    var text_len = text_slice.byte_length()
    var line_start = 0

    while line_start < text_len:
        # Fast newline scan
        var line_end = text_slice.find("\n", line_start)
        if line_end == -1:
            line_end = text_len

        # Strip CR and comments (#)
        var end = line_end
        var s = line_start
        while s < end:
            var b = ptr.load(s)
            if b == CR or b == HASH:
                end = s
                break
            s += 1

        var cur = ObjLineCursor(text_slice, line_start, end)
        var tag = cur.next_word()

        if tag.byte_length() == 0:
            line_start = line_end + 1
            continue

        if tag == "newmtl":
            if have_current:
                _ = mesh._upsert_material(current)
            var name = cur.joined_rest_of_line()
            current = ObjMaterial(name, fallback=False)
            have_current = True
            found_d = False

        elif have_current:
            if tag == "Ka":
                current.Ka = (cur.next_f32(), cur.next_f32(), cur.next_f32())
            elif tag == "Kd":
                current.Kd = (cur.next_f32(), cur.next_f32(), cur.next_f32())
            elif tag == "Ks":
                current.Ks = (cur.next_f32(), cur.next_f32(), cur.next_f32())
            elif tag == "Ke":
                current.Ke = (cur.next_f32(), cur.next_f32(), cur.next_f32())
            elif tag == "Kt":
                current.Kt = (cur.next_f32(), cur.next_f32(), cur.next_f32())
            elif tag == "Tf":
                current.Tf = (cur.next_f32(), cur.next_f32(), cur.next_f32())
            elif tag == "Ns":
                current.Ns = cur.next_f32()
            elif tag == "Ni":
                current.Ni = cur.next_f32()
            elif tag == "illum":
                current.illum = Int(
                    cur.next_f32()
                )  # Can just cast the f32 parser for illum ints
            elif tag == "d":
                current.d = cur.next_f32()
                found_d = True
            elif tag == "Tr":
                if not found_d:
                    current.d = 1.0 - cur.next_f32()
            elif tag[byte=:4] == "map_" or tag == "bump":
                var name = _map_name_from_tail(cur)
                if name.byte_length() > 0:
                    var val = mesh._add_texture(name, base)
                    if tag == "map_Ka":
                        current.map_Ka = val
                    elif tag == "map_Kd":
                        current.map_Kd = val
                    elif tag == "map_Ks":
                        current.map_Ks = val
                    elif tag == "map_Ke":
                        current.map_Ke = val
                    elif tag == "map_Kt":
                        current.map_Kt = val
                    elif tag == "map_Ns":
                        current.map_Ns = val
                    elif tag == "map_Ni":
                        current.map_Ni = val
                    elif tag == "map_d":
                        current.map_d = val
                    elif tag == "map_bump" or tag == "bump":
                        current.map_bump = val

        line_start = line_end + 1

    if have_current:
        _ = mesh._upsert_material(current)


def _read_mtl_file[
    Loader: ObjTextLoader
](mut mesh: ObjMesh, obj_path: String, mtl_name: String, loader: Loader) raises:
    var base = os_path.dirname(obj_path)
    var mtl_path = os_path.join(base, mtl_name)
    var text = loader.read_text(mtl_path)
    _read_mtl_text(mesh, os_path.dirname(mtl_path), text)


# OBJ parsing
@always_inline
def _word_ends_here[
    o: Origin
](ptr: UnsafePointer[UInt8, o], p: Int, end: Int) -> Bool:
    return p >= end or _is_ws(ptr.load(p))


struct ObjLineCursor[o: Origin]:
    var text: StringSlice[Self.o]
    var ptr: UnsafePointer[UInt8, Self.o]
    var pos: Int
    var end: Int

    def __init__(out self, text: StringSlice[Self.o], start: Int, end: Int):
        self.text = text
        self.ptr = text.unsafe_ptr()
        self.pos = start
        self.end = end

    @always_inline
    def skip_ws(mut self):
        while self.pos < self.end:
            var b = self.ptr.load(self.pos)
            if not _is_ws(b):
                break
            self.pos += 1

    @always_inline
    def has_next(mut self) -> Bool:
        self.skip_ws()
        return self.pos < self.end

    @always_inline
    def next_f32(mut self) -> Float32:
        self.skip_ws()
        if self.pos >= self.end:
            return 0.0

        var p = self.pos
        var sign: Float64 = 1.0

        if self.ptr.load(p) == MINUS:
            sign = -1.0
            p += 1
        elif self.ptr.load(p) == PLUS:
            p += 1

        var num: Float64 = 0.0
        while p < self.end:
            var b = self.ptr.load(p)
            if _is_digit(b):
                num = num * 10.0 + Float64(Int(b - ZERO))
                p += 1
            else:
                break

        if p < self.end and self.ptr.load(p) == DOT:
            p += 1
            var fra: Float64 = 0.0
            var div: Float64 = 1.0

            while p < self.end:
                var b = self.ptr.load(p)
                if _is_digit(b):
                    fra = fra * 10.0 + Float64(Int(b - ZERO))
                    div *= 10.0
                    p += 1
                else:
                    break

            num += fra / div

        if p < self.end:
            var b = self.ptr.load(p)
            if b == CHAR_e or b == CHAR_E:
                p += 1
                var exp_sign = 1

                if p < self.end and self.ptr.load(p) == MINUS:
                    exp_sign = -1
                    p += 1
                elif p < self.end and self.ptr.load(p) == PLUS:
                    p += 1

                var eval = 0
                while p < self.end:
                    var eb = self.ptr.load(p)
                    if _is_digit(eb):
                        eval = eval * 10 + Int(eb - ZERO)
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

        self.pos = p
        return Float32(sign * num)

    @always_inline
    def _parse_index_int(mut self, slash_terminates: Bool) -> Int:
        var sign = 1
        var output = 0

        if self.pos < self.end and self.ptr.load(self.pos) == MINUS:
            sign = -1
            self.pos += 1
        elif self.pos < self.end and self.ptr.load(self.pos) == PLUS:
            self.pos += 1

        while self.pos < self.end:
            var b = self.ptr.load(self.pos)
            if slash_terminates and b == SLASH:
                break
            if _is_ws(b):
                break
            if not _is_digit(b):
                break
            output = output * 10 + Int(b - ZERO)
            self.pos += 1

        return sign * output

    @always_inline
    def _parse_positive_index_int(mut self, slash_terminates: Bool) -> Int:
        # Fast path for the common OBJ case: positive decimal indices.
        # Falls back to signed parsing only when a sign is actually present.
        if self.pos < self.end:
            var first = self.ptr.load(self.pos)
            if first == MINUS or first == PLUS:
                return self._parse_index_int(slash_terminates)

        var output = 0
        while self.pos < self.end:
            var b = self.ptr.load(self.pos)
            if slash_terminates and b == SLASH:
                break
            if _is_ws(b):
                break
            if not _is_digit(b):
                break

            output = output * 10 + Int(b - ZERO)
            self.pos += 1

        return output

    @always_inline
    def _finish_index_token(mut self):
        while self.pos < self.end and not _is_ws(self.ptr.load(self.pos)):
            self.pos += 1

    @always_inline
    def _peek_face_shape(mut self) -> Int:
        # Returns:
        #   0 = generic / weird
        #   1 = p
        #   2 = p/t
        #   3 = p//n
        #   4 = p/t/n
        self.skip_ws()

        var p = self.pos
        var slash_count = 0
        var first_slash = -1

        while p < self.end:
            var b = self.ptr.load(p)
            if _is_ws(b):
                break
            if b == SLASH:
                if slash_count == 0:
                    first_slash = p
                slash_count += 1
            p += 1

        if slash_count == 0:
            return 1
        if slash_count == 1:
            return 2
        if slash_count == 2:
            if (
                first_slash + 1 < self.end
                and self.ptr.load(first_slash + 1) == SLASH
            ):
                return 3
            return 4

        return 0

    @always_inline
    def next_index_p_only_at_token(mut self, position_count: Int) -> ObjIndex:
        var p_raw = self._parse_positive_index_int(slash_terminates=False)
        self._finish_index_token()
        return ObjIndex(_fix_index(p_raw, position_count), 0, 0)

    @always_inline
    def next_index_p_t_at_token(
        mut self, position_count: Int, texcoord_count: Int
    ) -> ObjIndex:
        var p_raw = self._parse_positive_index_int(slash_terminates=True)
        var t_raw = 0

        if self.pos < self.end and self.ptr.load(self.pos) == SLASH:
            self.pos += 1
            t_raw = self._parse_positive_index_int(slash_terminates=False)

        self._finish_index_token()
        return ObjIndex(
            _fix_index(p_raw, position_count),
            _fix_index(t_raw, texcoord_count),
            0,
        )

    @always_inline
    def next_index_p_n_at_token(
        mut self, position_count: Int, normal_count: Int
    ) -> ObjIndex:
        var p_raw = self._parse_positive_index_int(slash_terminates=True)
        var n_raw = 0

        if self.pos < self.end and self.ptr.load(self.pos) == SLASH:
            self.pos += 1
            if self.pos < self.end and self.ptr.load(self.pos) == SLASH:
                self.pos += 1
                n_raw = self._parse_positive_index_int(slash_terminates=False)

        self._finish_index_token()
        return ObjIndex(
            _fix_index(p_raw, position_count),
            0,
            _fix_index(n_raw, normal_count),
        )

    @always_inline
    def next_index_p_t_n_at_token(
        mut self,
        position_count: Int,
        texcoord_count: Int,
        normal_count: Int,
    ) -> ObjIndex:
        var p_raw = self._parse_positive_index_int(slash_terminates=True)
        var t_raw = 0
        var n_raw = 0

        if self.pos < self.end and self.ptr.load(self.pos) == SLASH:
            self.pos += 1
            t_raw = self._parse_positive_index_int(slash_terminates=True)

            if self.pos < self.end and self.ptr.load(self.pos) == SLASH:
                self.pos += 1
                n_raw = self._parse_positive_index_int(slash_terminates=False)

        self._finish_index_token()
        return ObjIndex(
            _fix_index(p_raw, position_count),
            _fix_index(t_raw, texcoord_count),
            _fix_index(n_raw, normal_count),
        )

    @always_inline
    def next_index_generic_at_token(
        mut self,
        position_count: Int,
        texcoord_count: Int,
        normal_count: Int,
    ) -> ObjIndex:
        var p_raw = self._parse_index_int(slash_terminates=True)
        var t_raw = 0
        var n_raw = 0

        if self.pos < self.end and self.ptr.load(self.pos) == SLASH:
            self.pos += 1
            t_raw = self._parse_index_int(slash_terminates=True)

            if self.pos < self.end and self.ptr.load(self.pos) == SLASH:
                self.pos += 1
                n_raw = self._parse_index_int(slash_terminates=False)

        self._finish_index_token()

        return ObjIndex(
            _fix_index(p_raw, position_count),
            _fix_index(t_raw, texcoord_count),
            _fix_index(n_raw, normal_count),
        )

    def joined_rest_of_line(mut self) -> String:
        self.skip_ws()
        if self.pos >= self.end:
            return ""

        var start = self.pos
        var end_pos = self.end - 1

        # Trim trailing whitespace
        while end_pos > start:
            var b = self.ptr.load(end_pos)
            if not _is_ws(b):
                break
            end_pos -= 1

        self.pos = self.end
        return String(self.text[byte = start : end_pos + 1])

    @always_inline
    def next_word(mut self) -> StringSlice[Self.o]:
        self.skip_ws()
        if self.pos >= self.end:
            return StringSlice[Self.o]()

        var start = self.pos
        while self.pos < self.end:
            var b = self.ptr.load(self.pos)
            if _is_ws(b):
                break
            self.pos += 1

        return self.text[byte = start : self.pos]


@always_inline
def _parse_v_cursor[o: Origin](mut mesh: ObjMesh, mut cur: ObjLineCursor[o]):
    if not cur.has_next():
        return
    var x = cur.next_f32()

    if not cur.has_next():
        return
    var y = cur.next_f32()

    if not cur.has_next():
        return
    var z = cur.next_f32()

    mesh.positions.append(x)
    mesh.positions.append(y)
    mesh.positions.append(z)

    # Optional vertex color: v x y z r g b.
    if cur.has_next():
        var r = cur.next_f32()
        if cur.has_next():
            var g = cur.next_f32()
            if cur.has_next():
                var b = cur.next_f32()
                mesh._push_color(r, g, b)


@always_inline
def _parse_vt_cursor[o: Origin](mut mesh: ObjMesh, mut cur: ObjLineCursor[o]):
    if not cur.has_next():
        return
    var u = cur.next_f32()

    if not cur.has_next():
        return
    var v = cur.next_f32()

    mesh.texcoords.append(u)
    mesh.texcoords.append(v)


@always_inline
def _parse_vn_cursor[o: Origin](mut mesh: ObjMesh, mut cur: ObjLineCursor[o]):
    if not cur.has_next():
        return
    var x = cur.next_f32()

    if not cur.has_next():
        return
    var y = cur.next_f32()

    if not cur.has_next():
        return
    var z = cur.next_f32()

    mesh.normals.append(x)
    mesh.normals.append(y)
    mesh.normals.append(z)


@always_inline
def _finish_face_parse(
    mut mesh: ObjMesh,
    index_start: Int,
    count: Int,
    valid: Bool,
    is_line: Bool,
):
    if valid:
        if not is_line:
            if count >= 3:
                mesh._push_element_meta(count, is_line=False)
            else:
                mesh.indices.shrink(index_start)
        else:
            if count >= 2:
                mesh._push_element_meta(count, is_line=True)
            else:
                mesh.indices.shrink(index_start)
    else:
        mesh.indices.shrink(index_start)


@always_inline
def _parse_face_cursor[
    o: Origin
](mut mesh: ObjMesh, mut cur: ObjLineCursor[o], is_line: Bool = False):
    var index_start = len(mesh.indices)
    var count = 0
    var valid = True

    # Counts are constant for this face. Do not recompute per vertex token.
    var position_count = mesh.position_count()
    var texcoord_count = mesh.texcoord_count()
    var normal_count = mesh.normal_count()

    # Detect p / p/t / p//n / p/t/n once from the first token of the face.
    # Real OBJ faces almost always use one shape consistently for the whole line.
    var shape = cur._peek_face_shape()

    if shape == 1:
        # f p p p ...
        while True:
            cur.skip_ws()
            if cur.pos >= cur.end:
                break

            var idx = cur.next_index_p_only_at_token(position_count)
            if idx.p == 0:
                valid = False
                break

            mesh.indices.append(idx)
            count += 1

    elif shape == 2:
        # f p/t p/t p/t ...
        while True:
            cur.skip_ws()
            if cur.pos >= cur.end:
                break

            var idx = cur.next_index_p_t_at_token(
                position_count, texcoord_count
            )
            if idx.p == 0:
                valid = False
                break

            mesh.indices.append(idx)
            count += 1

    elif shape == 3:
        # f p//n p//n p//n ...
        while True:
            cur.skip_ws()
            if cur.pos >= cur.end:
                break

            var idx = cur.next_index_p_n_at_token(position_count, normal_count)
            if idx.p == 0:
                valid = False
                break

            mesh.indices.append(idx)
            count += 1

    elif shape == 4:
        # f p/t/n p/t/n p/t/n ...
        while True:
            cur.skip_ws()
            if cur.pos >= cur.end:
                break

            var idx = cur.next_index_p_t_n_at_token(
                position_count, texcoord_count, normal_count
            )
            if idx.p == 0:
                valid = False
                break

            mesh.indices.append(idx)
            count += 1

    else:
        # Weird or malformed first token: preserve the old generic behavior.
        while True:
            cur.skip_ws()
            if cur.pos >= cur.end:
                break

            var idx = cur.next_index_generic_at_token(
                position_count, texcoord_count, normal_count
            )
            if idx.p == 0:
                valid = False
                break

            mesh.indices.append(idx)
            count += 1

    _finish_face_parse(mesh, index_start, count, valid, is_line)


def _parse_obj_text[
    Loader: ObjTextLoader
](path: String, text: String, loader: Loader) raises -> ObjMesh:
    var mesh = ObjMesh()

    # Same rough heuristic as before, with reserves for the other hot arrays too.
    var est_elements = text.byte_length() // 15
    mesh.positions.reserve(est_elements * 3)
    mesh.texcoords.reserve(est_elements * 2)
    mesh.normals.reserve(est_elements * 3)
    mesh.indices.reserve(est_elements * 6)
    mesh.face_vertices.reserve(est_elements)
    mesh.face_materials.reserve(est_elements)

    var text_slice = StringSlice(text)
    var ptr = text_slice.unsafe_ptr()
    var text_len = text_slice.byte_length()
    var line_start = 0

    while line_start < text_len:
        var line_end = text_slice.find("\n", line_start)
        if line_end == -1:
            line_end = text_len

        # Logical line end strips CR and inline comments.
        var end = line_end
        var s = line_start
        while s < end:
            var b = ptr.load(s)
            if b == CR or b == HASH:
                end = s
                break
            s += 1

        var cur = ObjLineCursor(text_slice, line_start, end)
        cur.skip_ws()

        if cur.pos < end:
            var p = cur.pos
            var c0 = ptr.load(p)

            # Hot one/two-byte tag dispatch.
            if c0 == CHAR_v:
                if _word_ends_here(ptr, p + 1, end):
                    cur.pos = p + 1
                    _parse_v_cursor(mesh, cur)
                elif p + 1 < end and ptr.load(p + 1) == CHAR_t:
                    if _word_ends_here(ptr, p + 2, end):
                        cur.pos = p + 2
                        _parse_vt_cursor(mesh, cur)
                elif p + 1 < end and ptr.load(p + 1) == CHAR_n:
                    if _word_ends_here(ptr, p + 2, end):
                        cur.pos = p + 2
                        _parse_vn_cursor(mesh, cur)

            elif c0 == CHAR_f and _word_ends_here(ptr, p + 1, end):
                cur.pos = p + 1
                _parse_face_cursor(mesh, cur, is_line=False)

            elif c0 == CHAR_l and _word_ends_here(ptr, p + 1, end):
                cur.pos = p + 1
                _parse_face_cursor(mesh, cur, is_line=True)

            elif c0 == CHAR_g and _word_ends_here(ptr, p + 1, end):
                cur.pos = p + 1
                mesh._begin_group(cur.joined_rest_of_line())

            elif c0 == CHAR_o and _word_ends_here(ptr, p + 1, end):
                cur.pos = p + 1
                mesh._begin_object(cur.joined_rest_of_line())

            else:
                # Rare multi-char tags
                var tag_start = p
                var tag_end = p
                while tag_end < end and not _is_ws(ptr.load(tag_end)):
                    tag_end += 1

                var tag = text_slice[byte=tag_start:tag_end]
                cur.pos = tag_end

                if tag == "usemtl":
                    var name = cur.joined_rest_of_line()
                    if name.byte_length() > 0:
                        mesh._current_material = mesh._ensure_material(
                            name, fallback=True
                        )

                elif tag == "mtllib":
                    var mtl_name = cur.joined_rest_of_line()
                    if mtl_name.byte_length() > 0:
                        try:
                            _read_mtl_file(mesh, path, mtl_name, loader)
                        except:
                            pass

        line_start = line_end + 1

    if len(mesh.colors) > 0:
        while len(mesh.colors) < len(mesh.positions):
            mesh.colors.append(1.0)

    mesh._finish()
    return mesh^


# mesh utilities
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
