from bajo.parser.obj.types import ObjMesh, ObjIndex
from bajo.parser.asset_loader import (
    PathAssetLoader,
    AssetLoader,
    MemoryAssetLoader,
)
from bajo.parser.obj.obj import _parse_obj
from bajo.parser.obj.triangulate import triangulated_indices
from bajo.parser.mmap import MMap


def read_obj(path: String) raises -> ObjMesh:
    var loader = PathAssetLoader()
    var mapped = MMap(path)
    return parse_obj(mapped.as_string_span(), path, loader)


def read_obj[
    Loader: AssetLoader
](path: String, loader: Loader) raises -> ObjMesh:
    var text = loader.read_text(path)
    return parse_obj(text, path, loader)


def parse_obj(text: String, path: String = "") raises -> ObjMesh:
    """Raw OBJ text. MTL files are ignored because no loader is provided."""
    var loader = MemoryAssetLoader()
    return parse_obj(text, path, loader)


def parse_obj(text: ImmStringSpan, path: String = "") raises -> ObjMesh:
    """Raw OBJ StringSpan."""
    var loader = MemoryAssetLoader()
    return parse_obj(text, path, loader)


def parse_obj[
    Loader: AssetLoader
](text: String, path: String, loader: Loader) raises -> ObjMesh:
    """Raw OBJ text plus loader for mtllib resolution."""
    return _parse_obj(path, StringSpan(text), loader)


def parse_obj[
    Loader: AssetLoader
](text: ImmStringSpan, path: String, loader: Loader) raises -> ObjMesh:
    """Raw OBJ StringSpan plus loader for mtllib resolution."""
    return _parse_obj(path, text, loader)
