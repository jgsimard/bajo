"""Time PLY decoding stages and PBRT token scanning on mapped input.

Build with `mojo build -O3 -I . bench/parser/bench_simd_stages.mojo`.
Run with `<ply|pbrt> <asset-path>`. Input mapping is outside the timings;
planning and packing include their allocations, decoding reuses output buffers.
"""

from std.benchmark import Unit, keep, run
from std.sys.arg import argv
from bajo.parser.mmap import MMap
from bajo.parser.pbrt.parser import _Lexer
from bajo.parser.ply import parse_ply
from bajo.parser.ply.parser import (
    _parse_header,
    _face_chunks,
    _decode_vertices,
    _emit_faces,
)
from bajo.parser.ply.pack import _pack_ply_mesh


def main() raises:
    var args = argv()
    if len(args) != 3:
        raise Error("usage: bench_simd_stages <ply|pbrt> <asset-path>")
    var kind = String(args[1])
    if kind != "ply" and kind != "pbrt":
        raise Error("expected parser kind ply or pbrt")
    var mapped = MMap(String(args[2]))
    var bytes = mapped.as_bytes_span()
    if kind == "pbrt":
        var count = 0
        var lexer = _Lexer(bytes)
        while lexer.has_next():
            _ = lexer.next()
            count += 1
        print("tokens", count)

        def lex() raises {bytes}:
            var lexer = _Lexer(bytes)
            var count = 0
            while lexer.has_next():
                var token = lexer.next()
                keep(token.value.unsafe_ptr())
                count += 1
            keep(count)
            keep(lexer.line)

        print("lex_us", run(lex, max_iters=10).mean(Unit.us))
        return
    var header = _parse_header(bytes)
    var chunks = _face_chunks(bytes, header)
    var mesh = parse_ply(bytes)
    print(
        "vertices",
        mesh.vertex_count(),
        "faces",
        mesh.face_count(),
        "triangles",
        mesh.triangle_count(),
    )

    def plan() raises {bytes, header}:
        var output = _face_chunks(bytes, header)
        keep(output.unsafe_ptr())
        keep(len(output))

    def vertices() raises {imm, mut mesh}:
        _decode_vertices(bytes, header, mesh, 8)
        keep(mesh.positions.unsafe_ptr())
        keep(mesh.normals.unsafe_ptr())
        keep(mesh.texcoords.unsafe_ptr())

    def emit() raises {imm, mut mesh}:
        for chunk in chunks:
            _emit_faces(
                bytes,
                header,
                chunk,
                Span(mesh.indices)[
                    chunk.output_offset : chunk.output_offset
                    + chunk.output_count
                ],
            )
        keep(mesh.indices.unsafe_ptr())

    def pack() raises {mesh}:
        var output = _pack_ply_mesh(mesh, False, 8)
        keep(output.vertices.unsafe_ptr())
        keep(output.normals.unsafe_ptr())
        keep(output.texcoords.unsafe_ptr())
        keep(output.bounds)

    print("plan_serial_us", run(plan, max_iters=10).mean(Unit.us))
    print("vertices_8workers_us", run(vertices, max_iters=10).mean(Unit.us))
    print("emit_serial_us", run(emit, max_iters=10).mean(Unit.us))
    print("pack_8workers_us", run(pack, max_iters=10).mean(Unit.us))
