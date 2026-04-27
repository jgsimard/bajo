from std.benchmark import run, Unit, keep
from std.pathlib import Path
from std.reflection import get_function_name

from bajo.obj import (
    ObjMesh,
    read_obj,
    parse_obj_text,
    triangulated_indices,
)


# Change this if your asset path differs.
comptime OBJ_PATH = "./assets/bunny/bunny.obj"

# For big OBJ files, 200 iterations can be too much while iterating.
# Start low, then increase.
comptime PARSE_ITERS = 5
comptime TRI_ITERS = 200


def bench_read_obj(path: String) raises:
    # Includes:
    # - file read
    # - OBJ parse
    # - MTL file read/parse if mtllib is present
    var mesh = read_obj(path)

    # Prevent dead-code elimination and also sanity-check the result.
    keep(mesh.index_count())
    keep(mesh.face_count())
    keep(mesh.position_count())


def bench_parse_obj_text(path: String, text: String) raises:
    # Includes:
    # - OBJ parse
    # - MTL file read/parse if mtllib is present, because parse_obj_text uses PathObjTextLoader
    #
    # Excludes:
    # - reading the OBJ file itself
    var mesh = parse_obj_text(path, text)

    keep(mesh.index_count())
    keep(mesh.face_count())
    keep(mesh.position_count())


def bench_triangulated_indices(mesh: ObjMesh) raises:
    # Includes:
    # - fan triangulation allocation + index writes
    #
    # Excludes:
    # - OBJ parsing
    var tris = triangulated_indices(mesh)

    keep(len(tris))
    if len(tris) > 0:
        keep(tris[0].p)


def main() raises:
    var path = String(OBJ_PATH)

    print("Benchmarking OBJ loader")
    print("Path:", path)

    # Preload text for parse-only benchmark.
    var text = Path(path).read_text()

    # Parse once for summary and triangulation benchmark input.
    var mesh = parse_obj_text(path, text)
    mesh.print_summary()

    print("")

    def run_read_obj() raises capturing:
        bench_read_obj(path)

    def run_parse_obj_text() raises capturing:
        bench_parse_obj_text(path, text)

    def run_triangulated_indices() raises capturing:
        bench_triangulated_indices(mesh)

    var report_read = run[run_read_obj](max_iters=PARSE_ITERS)
    var read_us = report_read.mean(Unit.us)

    var report_parse = run[run_parse_obj_text](max_iters=PARSE_ITERS)
    var parse_us = report_parse.mean(Unit.us)

    var report_tris = run[run_triangulated_indices](max_iters=TRI_ITERS)
    var tris_us = report_tris.mean(Unit.us)

    var faces = mesh.face_count()
    var indices = mesh.index_count()
    var tris = triangulated_indices(mesh)
    var tri_indices = len(tris)

    print("Results")
    print("-------")
    print(
        t"read_obj              | Avg: {round(read_us, 2)} us"
        t" | faces/s: {round(Float64(faces) / read_us, 2)} M"
        t" | indices/s: {round(Float64(indices) / read_us, 2)} M"
    )
    print(
        t"parse_obj_text        | Avg: {round(parse_us, 2)} us"
        t" | faces/s: {round(Float64(faces) / parse_us, 2)} M"
        t" | indices/s: {round(Float64(indices) / parse_us, 2)} M"
    )
    print(
        t"triangulated_indices  | Avg: {round(tris_us, 2)} us"
        t" | out_indices/s: {round(Float64(tri_indices) / tris_us, 2)} M"
    )
