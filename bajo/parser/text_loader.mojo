from std.pathlib import Path

from bajo.parser.ply import PlyMesh, parse_ply, read_ply
from bajo.parser.png import parse_png, read_png
from bajo.rt.types import ImageTexture


trait TextLoader:
    def read_text(self, path: String) raises -> String:
        ...

    def read_ply_mesh(self, path: String) raises -> PlyMesh:
        ...

    def read_image_texture(self, path: String) raises -> ImageTexture:
        ...


@fieldwise_init
struct PathTextLoader(Copyable, TextLoader):
    def read_text(self, path: String) raises -> String:
        return Path(path).read_text()

    def read_ply_mesh(self, path: String) raises -> PlyMesh:
        return read_ply(path)

    def read_image_texture(self, path: String) raises -> ImageTexture:
        return read_png(path)


struct MemoryTextLoader(TextLoader):
    var files: Dict[String, String]
    var ply_files: Dict[String, List[UInt8]]
    var image_files: Dict[String, List[UInt8]]

    def __init__(out self):
        self.files = Dict[String, String]()
        self.ply_files = Dict[String, List[UInt8]]()
        self.image_files = Dict[String, List[UInt8]]()

    def add_file(mut self, path: String, text: String):
        self.files[path] = text

    def add_ply_file(mut self, path: String, var bytes: List[UInt8]):
        self.ply_files[path] = bytes^

    def add_image_file(mut self, path: String, var bytes: List[UInt8]):
        self.image_files[path] = bytes^

    def read_text(self, path: String) raises -> String:
        if path in self.files:
            return self.files[path]
        raise Error("MemoryTextLoader: file not found: " + path)

    def read_ply_mesh(self, path: String) raises -> PlyMesh:
        if path in self.ply_files:
            return parse_ply(self.ply_files[path])
        raise Error("MemoryTextLoader: PLY file not found: " + path)

    def read_image_texture(self, path: String) raises -> ImageTexture:
        if path in self.image_files:
            return parse_png(self.image_files[path])
        raise Error("MemoryTextLoader: image file not found: " + path)
