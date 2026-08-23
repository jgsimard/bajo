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


struct ObjMesh:
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
