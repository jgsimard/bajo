"""Binary little-endian PLY mesh loading."""

from bajo.parser.mmap import MMap
from bajo.parser.ply.parser import parse_ply
from bajo.parser.ply.types import PlyMesh


def read_ply(path: String) raises -> PlyMesh:
    var mapped = MMap(path)
    return parse_ply(mapped.as_bytes_span())
