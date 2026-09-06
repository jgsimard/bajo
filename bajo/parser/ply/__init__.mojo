"""Binary little-endian PLY mesh loading."""

from bajo.parser.obj.mmap import MMap
from bajo.parser.ply.parser import _parse_ply
from bajo.parser.ply.types import PlyMesh


def read_ply(path: String) raises -> PlyMesh:
    var mapped = MMap(path)
    return _parse_ply(mapped.as_bytes_span())


def parse_ply[
    origin: ImmOrigin
](bytes: ImmSpan[UInt8, origin]) raises -> PlyMesh:
    return _parse_ply(bytes)
