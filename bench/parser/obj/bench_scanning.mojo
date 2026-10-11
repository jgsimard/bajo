"""Measure OBJ structural scanning, bulk parsing and the public entry point.

Build with `mojo build -O3 -I . bench/parser/obj/bench_scanning.mojo`.
Run with `<mesh.obj> [workers]`; input is mapped outside the timed regions.
"""

from std.benchmark import Unit, keep, run
from std.sys import num_logical_cores, simd_width_of
from std.sys.arg import argv

from bajo.parser.mmap import MMap
from bajo.parser.obj import PathAssetLoader, parse_obj
from bajo.parser.obj.parallel import _count_chunk, _parse_obj_bulk


def main() raises:
    var args = argv()
    if len(args) < 2 or len(args) > 3:
        raise Error("usage: bench_scanning <mesh.obj> [workers]")
    var path = String(args[1])
    var workers = Int(String(args[2])) if len(args) == 3 else min(
        8, num_logical_cores()
    )
    var mapped = MMap(path)
    var text = mapped.as_string_span()
    var counts = _count_chunk(text, 0, text.byte_length())
    print(
        "bytes",
        text.byte_length(),
        "positions",
        counts.positions,
        "faces",
        counts.faces,
        "indices",
        counts.indices,
        "workers",
        workers,
        "byte SIMD width",
        simd_width_of[UInt8](),
    )

    def scan() {text}:
        var result = _count_chunk(text, 0, text.byte_length())
        keep(result.positions)
        keep(result.faces)
        keep(result.indices)
        keep(result.events.unsafe_ptr())

    def parse() raises {path, text, workers}:
        var mesh = _parse_obj_bulk(path, text, PathAssetLoader(), workers)
        keep(mesh.positions.unsafe_ptr())
        keep(mesh.texcoords.unsafe_ptr())
        keep(mesh.normals.unsafe_ptr())
        keep(mesh.colors.unsafe_ptr())
        keep(mesh.indices.unsafe_ptr())
        keep(mesh.face_vertices.unsafe_ptr())
        keep(mesh.face_materials.unsafe_ptr())
        keep(mesh.index_count())

    def parse_public() raises {path, text}:
        var mesh = parse_obj(text, path, PathAssetLoader())
        keep(mesh.positions.unsafe_ptr())
        keep(mesh.texcoords.unsafe_ptr())
        keep(mesh.normals.unsafe_ptr())
        keep(mesh.colors.unsafe_ptr())
        keep(mesh.indices.unsafe_ptr())
        keep(mesh.face_vertices.unsafe_ptr())
        keep(mesh.face_materials.unsafe_ptr())
        keep(mesh.index_count())

    print("OBJ_SIMD_METRIC,scan,", run(scan, max_iters=5).mean(Unit.us))
    print("OBJ_SIMD_METRIC,parse,", run(parse, max_iters=5).mean(Unit.us))
    print(
        "OBJ_SIMD_METRIC,public_parse,",
        run(parse_public, max_iters=5).mean(Unit.us),
    )
