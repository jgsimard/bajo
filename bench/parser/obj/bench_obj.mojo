from std.benchmark import run, Unit, keep
from std.pathlib import Path
from std.sys.arg import argv
from std.sys import size_of

from bajo.parser.obj import (
    ObjMesh,
    ObjIndex,
    PathAssetLoader,
    read_obj,
    parse_obj,
    triangulated_indices,
)
from bajo.parser.obj.pack import pack_obj_triangles


def bench_read_obj(path: String) raises:
    var mesh = read_obj(path)

    keep(mesh.index_count())
    keep(mesh.face_count())
    keep(mesh.position_count())


def bench_parse_obj(path: String, text: String) raises:
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
    # comptime OBJ_PATH = "./assets/rungholt/rungholt.obj"
    # comptime OBJ_PATH = "./assets/powerplant/powerplant.obj"
    comptime PARSE_ITERS = 5
    comptime TRI_ITERS = 20
    var args = argv()
    if len(args) > 2:
        raise Error("usage: pixi run bench_obj [-- <mesh.obj>]")
    var path = String(args[1]) if len(args) == 2 else String(OBJ_PATH)

    print("Benchmarking OBJ loader")
    print("Path:", path)

    # Preload text for parse-only benchmark.
    var text = Path(path).read_text()

    # Parse once for summary and triangulation benchmark input.
    var mesh = parse_obj(text, path, PathAssetLoader())
    mesh.print_summary()
    print("OBJ index bytes:", size_of[ObjIndex]())

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
