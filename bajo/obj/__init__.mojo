from bajo.obj.types import ObjMesh, ObjIndex
from bajo.obj.loaders import (
    PathObjTextLoader,
    ObjTextLoader,
    MemoryObjTextLoader,
)
from bajo.obj.obj import _parse_obj
from bajo.obj.mmap import MMap


def read_obj(path: String) raises -> ObjMesh:
    var loader = PathObjTextLoader()
    var mapped = MMap[ImmutAnyOrigin](path)
    return parse_obj(mapped.as_string_slice(), path, loader)


def read_obj[
    Loader: ObjTextLoader
](path: String, loader: Loader) raises -> ObjMesh:
    var text = loader.read_text(path)
    return parse_obj(text, path, loader)


def parse_obj(text: String, path: String = "") raises -> ObjMesh:
    """Raw OBJ text. MTL files are ignored because no loader is provided."""
    var loader = MemoryObjTextLoader()
    return parse_obj(text, path, loader)


def parse_obj(
    text: StringSlice[mut=False, _], path: String = ""
) raises -> ObjMesh:
    """Raw OBJ StringSlice."""
    var loader = MemoryObjTextLoader()
    return parse_obj(text, path, loader)


def parse_obj[
    Loader: ObjTextLoader
](text: String, path: String, loader: Loader) raises -> ObjMesh:
    """Raw OBJ text plus loader for mtllib resolution."""
    return _parse_obj(path, text, loader)


def parse_obj[
    Loader: ObjTextLoader
](
    text: StringSlice[mut=False, _], path: String, loader: Loader
) raises -> ObjMesh:
    """Raw OBJ StringSlice plus loader for mtllib resolution."""
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
