import std.os.path


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
struct ObjIndex(TrivialRegisterPassable):
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
            t" {self.position_count(include_dummy=False)}\n"
            t" - texcoords: {self.texcoord_count()} including dummy; actual:"
            t" {self.texcoord_count(include_dummy=False)}\n"
            t" - normals: {self.normal_count()} including dummy; actual:"
            t" {self.normal_count(include_dummy=False)}\n"
            t" - colors: {self.color_count()}\n"
            t" - faces/lines: {self.face_count()}\n"
            t" - indices: {self.index_count()}\n"
            t" - materials: {self.material_count()}\n"
            t" - textures: {self.texture_count()} including dummy; actual:"
            t" {self.texture_count(include_dummy=False)}\n"
            t" - objects: {self.object_count()}\n"
            t" - groups:  {self.group_count()}"
        )

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
        var path = std.os.path.join(base, name)

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
