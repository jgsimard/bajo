from std.benchmark import run, Unit, keep
from std.sys.arg import argv
from std.sys import size_of, num_logical_cores, simd_width_of

from bajo.parser.obj import (
    ObjMesh,
    ObjIndex,
    PathAssetLoader,
    read_obj,
    parse_obj,
    triangulated_indices,
)
from bajo.parser.obj.pack import pack_obj_triangles
from bajo.parser.mmap import MMap
from bajo.parser.obj.parallel import _count_chunk, _parse_obj_bulk


def bench_read_obj(path: String) raises:
    var mesh = read_obj(path)

    keep(mesh.index_count())
    keep(mesh.face_count())
    keep(mesh.position_count())


def bench_parse_obj(path: String, text: StringSpan) raises:
    var mesh = parse_obj(text, path, PathAssetLoader())

    keep(mesh.index_count())
    keep(mesh.face_count())
    keep(mesh.position_count())


def bench_triangulated_indices(mesh: ObjMesh) raises:
    var tris = triangulated_indices(mesh)

    keep(len(tris))
    if len(tris) > 0:
        keep(tris[0].p)


def bench_pack_mesh(mesh: ObjMesh):
    var vertices = pack_obj_triangles(mesh)
    keep(len(vertices))
    if len(vertices) > 0:
        keep(vertices[0].x)


def bench_pack_obj(path: String) raises:
    var vertices = pack_obj_triangles(path)
    keep(len(vertices))
    if len(vertices) > 0:
        keep(vertices[0].x)


def main() raises:
    comptime OBJ_PATH = "./assets/bunny/bunny.obj"
    comptime PARSE_ITERS = 5
    comptime TRI_ITERS = 20
    var args = argv()
    if len(args) > 3:
        raise Error("usage: pixi run bench_obj [mesh.obj [workers]]")
    var path = String(args[1]) if len(args) >= 2 else String(OBJ_PATH)
    var workers = Int(String(args[2])) if len(args) == 3 else min(
        8, num_logical_cores()
    )
    if workers < 1:
        raise Error("workers must be positive")

    print("Benchmarking OBJ loader")
    print("Path:", path)

    # Map once for parse-only and structural-stage benchmarks.
    var mapped = MMap(path)
    var text = mapped.as_string_span()

    # Parse once for summary and triangulation benchmark input.
    var mesh = parse_obj(text, path, PathAssetLoader())
    mesh.print_summary()
    print("OBJ index bytes:", size_of[ObjIndex]())
    print("Bulk workers:", workers, "byte SIMD width:", simd_width_of[UInt8]())

    def run_read_obj() raises {path}:
        bench_read_obj(path)

    def run_parse_obj() raises {path, text}:
        bench_parse_obj(path, text)

    def run_triangulated_indices() raises {mesh}:
        bench_triangulated_indices(mesh)

    def run_pack_mesh() {mesh}:
        bench_pack_mesh(mesh)

    def run_pack_obj() raises {path}:
        bench_pack_obj(path)

    def run_scan() {text}:
        var counts = _count_chunk(text, 0, text.byte_length())
        keep(counts.positions)
        keep(counts.faces)
        keep(counts.indices)
        keep(counts.events.unsafe_ptr())

    def run_bulk_parse() raises {path, text, workers}:
        var output = _parse_obj_bulk(path, text, PathAssetLoader(), workers)
        keep(output.positions.unsafe_ptr())
        keep(output.texcoords.unsafe_ptr())
        keep(output.normals.unsafe_ptr())
        keep(output.colors.unsafe_ptr())
        keep(output.indices.unsafe_ptr())
        keep(output.face_vertices.unsafe_ptr())
        keep(output.face_materials.unsafe_ptr())
        keep(output.index_count())

    var scan_us = run(run_scan, max_iters=PARSE_ITERS).mean(Unit.us)
    var bulk_us = run(run_bulk_parse, max_iters=PARSE_ITERS).mean(Unit.us)

    var report_read = run(run_read_obj, max_iters=PARSE_ITERS)
    var read_us = report_read.mean(Unit.us)

    var report_parse = run(run_parse_obj, max_iters=PARSE_ITERS)
    var parse_us = report_parse.mean(Unit.us)

    var report_tris = run(run_triangulated_indices, max_iters=TRI_ITERS)
    var tris_us = report_tris.mean(Unit.us)
    var pack_report = run(run_pack_mesh, max_iters=TRI_ITERS)
    var pack_us = pack_report.mean(Unit.us)
    var packed_read_report = run(run_pack_obj, max_iters=PARSE_ITERS)
    var packed_read_us = packed_read_report.mean(Unit.us)

    var faces = mesh.face_count()
    var indices = mesh.index_count()
    var tris = triangulated_indices(mesh)
    var tri_indices = len(tris)

    print("\nResults")
    print("-------")
    print(
        t"read_obj              | Avg: {round(read_us, 2)} us"
        t" | faces/s: {round(Float64(faces) / read_us, 2)} M"
        t" | indices/s: {round(Float64(indices) / read_us, 2)} M"
    )
    print(
        t"parse_obj             | Avg: {round(parse_us, 2)} us"
        t" | faces/s: {round(Float64(faces) / parse_us, 2)} M"
        t" | indices/s: {round(Float64(indices) / parse_us, 2)} M"
    )
    print(
        t"triangulated_indices  | Avg: {round(tris_us, 2)} us"
        t" | out_indices/s: {round(Float64(tri_indices) / tris_us, 2)} M"
    )
    print(t"pack_mesh             | Avg: {round(pack_us, 2)} us")
    print(t"pack_obj              | Avg: {round(packed_read_us, 2)} us")
    print(t"OBJ_METRIC,read_obj,{read_us}")
    print(t"OBJ_METRIC,parse_obj,{parse_us}")
    print(t"OBJ_METRIC,triangulated_indices,{tris_us}")
    print(t"OBJ_METRIC,pack_mesh,{pack_us}")
    print(t"OBJ_METRIC,pack_obj,{packed_read_us}")
    print(t"OBJ_SIMD_METRIC,scan,{scan_us}")
    print(t"OBJ_SIMD_METRIC,parse,{bulk_us}")
    print(t"OBJ_SIMD_METRIC,public_parse,{parse_us}")
