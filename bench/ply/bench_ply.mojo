"""Microbenchmark for binary little-endian PLY loading."""

from std.benchmark import run, Unit, keep
from std.sys.arg import argv

from bajo.parser.obj.mmap import MMap
from bajo.parser.ply import parse_ply, read_ply


comptime BENCH_ITERS = 10
comptime DEFAULT_PLY_PATH = "./assets/crown/geometry/mesh_00009.ply"


def _bench_read(path: String) raises:
    var mesh = read_ply(path)
    keep(mesh.vertex_count())
    keep(mesh.triangle_count())


def _bench_parse[origin: ImmOrigin](bytes: ImmSpan[UInt8, origin]) raises:
    var mesh = parse_ply(bytes)
    keep(mesh.vertex_count())
    keep(mesh.triangle_count())


def _print_result(
    name: String,
    mean_us: Float64,
    byte_count: Int,
    vertex_count: Int,
    triangle_count: Int,
):
    var mib = Float64(byte_count) / (1024.0 * 1024.0)
    var mib_per_s = mib * 1.0e6 / mean_us
    print(
        t"{name} | Avg: {round(mean_us, 2)} us"
        t" | {round(mib_per_s, 2)} MiB/s"
        t" | {round(Float64(vertex_count) / mean_us, 2)} M vertices/s"
        t" | {round(Float64(triangle_count) / mean_us, 2)} M triangles/s"
    )


def main() raises:
    var args = argv()
    if len(args) > 2:
        raise Error("usage: pixi run bench_ply [-- <mesh.ply>]")

    var path = String(DEFAULT_PLY_PATH)
    if len(args) == 2:
        path = String(args[1])
    var mapped = MMap[ImmutAnyOrigin](path)
    var bytes = mapped.as_bytes_span()
    var mesh = parse_ply(bytes)

    print("Binary little-endian PLY loader benchmark")
    print("Path:", path)
    print(
        t"Input: {mapped.byte_length()} bytes, {mesh.vertex_count()} vertices, "
        t"{mesh.face_count()} faces, {mesh.triangle_count()} triangles"
    )
    print(t"Mean of up to {BENCH_ITERS} iterations after harness warmup")

    def run_read() raises {path}:
        _bench_read(path)

    def run_parse() raises {bytes}:
        _bench_parse(bytes)

    var read_report = run(run_read, max_iters=BENCH_ITERS)
    var parse_report = run(run_parse, max_iters=BENCH_ITERS)

    print("\nResults")
    print("-------")
    _print_result(
        "read_ply ",
        read_report.mean(Unit.us),
        mapped.byte_length(),
        mesh.vertex_count(),
        mesh.triangle_count(),
    )
    _print_result(
        "parse_ply",
        parse_report.mean(Unit.us),
        mapped.byte_length(),
        mesh.vertex_count(),
        mesh.triangle_count(),
    )
