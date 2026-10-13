"""OBJ entry point: scalar for small input, counted decoding for large input."""

from bajo.parser.asset_loader import AssetLoader
from std.sys import num_logical_cores
from .types import ObjMesh
from .serial import _parse_obj_serial
from .parallel import _parse_obj_bulk


def _parse_obj[
    Loader: AssetLoader
](path: String, text: ImmStringSpan, loader: Loader) raises -> ObjMesh:
    if text.byte_length() < 256 * 1024:
        return _parse_obj_serial(path, text, loader)
    return _parse_obj_bulk(path, text, loader, min(8, num_logical_cores()))
