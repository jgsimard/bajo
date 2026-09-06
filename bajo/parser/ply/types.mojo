"""Backend-neutral indexed mesh data loaded from a PLY file."""


struct PlyMesh:
    """Indexed polygon mesh with optional per-vertex normals and UVs.

    Positions and normals are stored as xyz triples, texture coordinates as uv
    pairs, and `indices` as an already-triangulated index stream.
    """

    var positions: List[Float32]
    var normals: List[Float32]
    var texcoords: List[Float32]
    var indices: List[UInt32]
    var _source_face_count: Int

    def __init__(out self):
        self.positions = List[Float32]()
        self.normals = List[Float32]()
        self.texcoords = List[Float32]()
        self.indices = List[UInt32]()
        self._source_face_count = 0

    def vertex_count(self) -> Int:
        return len(self.positions) / 3

    def face_count(self) -> Int:
        """Return the number of polygon faces in the source PLY file."""
        return self._source_face_count

    def triangle_count(self) -> Int:
        return len(self.indices) / 3

    def has_normals(self) -> Bool:
        return len(self.normals) != 0

    def has_texcoords(self) -> Bool:
        return len(self.texcoords) != 0
