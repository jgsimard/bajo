from bajo.parser.obj.types import ObjMesh, ObjIndex
from bajo.parser.text_loader import (
    PathTextLoader,
    TextLoader,
    MemoryTextLoader,
)
from bajo.parser.obj.obj import _parse_obj
from bajo.parser.obj.mmap import MMap


def read_obj(path: String) raises -> ObjMesh:
    var loader = PathTextLoader()
    var mapped = MMap[ImmutAnyOrigin](path)
    return parse_obj(mapped.as_string_span(), path, loader)


def read_obj[
    Loader: TextLoader
](path: String, loader: Loader) raises -> ObjMesh:
    var text = loader.read_text(path)
    return parse_obj(text, path, loader)


def parse_obj(text: String, path: String = "") raises -> ObjMesh:
    """Raw OBJ text. MTL files are ignored because no loader is provided."""
    var loader = MemoryTextLoader()
    return parse_obj(text, path, loader)


def parse_obj(text: ImmStringSpan, path: String = "") raises -> ObjMesh:
    """Raw OBJ StringSpan."""
    var loader = MemoryTextLoader()
    return parse_obj(text, path, loader)


def parse_obj[
    Loader: TextLoader
](text: String, path: String, loader: Loader) raises -> ObjMesh:
    """Raw OBJ text plus loader for mtllib resolution."""
    return _parse_obj(path, text, loader)


def parse_obj[
    Loader: TextLoader
](text: ImmStringSpan, path: String, loader: Loader) raises -> ObjMesh:
    """Raw OBJ StringSpan plus loader for mtllib resolution."""
    return _parse_obj(path, text, loader)


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

    var write_idx = 0
    var offset = 0

    for f in range(face_count):
        var n = mesh.face_vertices[f]
        var is_line = False
        if len(mesh.face_lines) > 0:
            is_line = mesh.face_lines[f] != 0

        if not is_line and n >= 3:
            var first = mesh.indices[offset]
            for i in range(1, n - 1):
                out[write_idx] = first
                out[write_idx + 1] = mesh.indices[offset + i]
                out[write_idx + 2] = mesh.indices[offset + i + 1]
                write_idx += 3
        offset += n

    return out^
