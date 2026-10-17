"""Compare OBJ worker counts with mapped input and output allocation timed.

Usage: mojo -I . bench/parser/bench_obj.mojo path.obj
Material-library loading is included; mapping the OBJ file is outside timings.
Use bench_loaders.mojo for public read_obj timings and full mesh digests.
"""

from std.benchmark import Unit, keep, run
from std.sys import num_logical_cores
from std.sys.arg import argv

from bajo.parser.asset_loader import PathAssetLoader
from bajo.parser.mmap import MMap
from bajo.parser.obj.parallel import _parse_obj_bulk


def main() raises:
    var args = argv()
    if len(args) != 2:
        raise Error("usage: bench_obj path.obj")
    var path = String(args[1])
    var mapped = MMap(path)
    var text = mapped.as_string_span()
    var loader = PathAssetLoader()
    var previous = 0
    for requested in [1, 2, 4, 8, 16]:
        var workers = min(requested, num_logical_cores())
        if workers == previous:
            continue
        previous = workers

        def decode() raises {text, path, loader, workers}:
            var mesh = _parse_obj_bulk(path, text, loader, workers)
            keep(mesh.positions.unsafe_ptr())
            keep(mesh.texcoords.unsafe_ptr())
            keep(mesh.normals.unsafe_ptr())
            keep(mesh.colors.unsafe_ptr())
            keep(mesh.indices.unsafe_ptr())
            keep(mesh.face_vertices.unsafe_ptr())
            keep(mesh.face_materials.unsafe_ptr())
            keep(mesh.face_lines.unsafe_ptr())
            keep(mesh.materials.unsafe_ptr())
            keep(mesh.textures.unsafe_ptr())
            keep(mesh.material_count())

        print(
            "workers",
            workers,
            "mean_us",
            run(decode, max_iters=10).mean(Unit.us),
        )
