"""Parser timings and full geometry/pixel digests for before/after comparisons.

Build with `mojo build -O3 -I . bench/parser/bench_loaders.mojo -o /tmp/loaders`
and run `/tmp/loaders <ply|png|exr|pbrt|mtl> <path>`.
"""

from std.benchmark import Unit, keep, run
from std.memory import bitcast
from std.sys.arg import argv

from bajo.parser.mmap import MMap
from bajo.parser.ply import parse_ply
from bajo.parser.png import parse_png
from bajo.parser.exr import parse_exr
from bajo.parser.pbrt import read_pbrt
from bajo.parser.obj import read_obj


@always_inline
def _hash(mut digest: UInt64, word: UInt32):
    digest = (digest ^ UInt64(word)) * UInt64(1099511628211)


def _floats(mut digest: UInt64, values: ImmSpan[Float32, _]):
    for value in values:
        _hash(digest, bitcast[.uint32](value))


def main() raises:
    var args = argv()
    if len(args) != 3:
        raise Error("usage: bench_loaders <ply|png|exr|pbrt|mtl> <path>")
    var kind = String(args[1])
    var path = String(args[2])
    var mapped = MMap(path)
    var bytes = mapped.as_bytes_span()
    var digest = UInt64(14695981039346656037)
    var mean_us: Float64
    __match kind:
        case "ply":
            var mesh = parse_ply(bytes)
            _floats(digest, mesh.positions)
            _floats(digest, mesh.normals)
            _floats(digest, mesh.texcoords)
            for index in mesh.indices:
                _hash(digest, index)
            print(
                "vertices",
                mesh.vertex_count(),
                "triangles",
                mesh.triangle_count(),
            )

            def bench_ply() raises {bytes}:
                var output = parse_ply(bytes)
                keep(output.vertex_count())
                keep(output.triangle_count())
                keep(output.positions.unsafe_ptr())
                keep(output.normals.unsafe_ptr())
                keep(output.texcoords.unsafe_ptr())
                keep(output.indices.unsafe_ptr())

            mean_us = run(bench_ply, max_iters=10).mean(Unit.us)
        case "png":
            var image = parse_png(bytes)
            _floats(digest, image.pixels)
            print("size", image.width, image.height)

            def bench_png() raises {bytes}:
                var output = parse_png(bytes)
                keep(output.width)
                keep(output.pixels.unsafe_ptr())

            mean_us = run(bench_png, max_iters=10).mean(Unit.us)
        case "exr":
            var image = parse_exr(bytes)
            _floats(digest, image.pixels)
            print("size", image.width, image.height)

            def bench_exr() raises {bytes}:
                var output = parse_exr(bytes)
                keep(output.width)
                keep(output.pixels.unsafe_ptr())

            mean_us = run(bench_exr, max_iters=10).mean(Unit.us)
        case "pbrt":
            var scene = read_pbrt(path)
            for point in scene.data.triangle_vertices():
                _hash(digest, bitcast[.uint32](point.x))
                _hash(digest, bitcast[.uint32](point.y))
                _hash(digest, bitcast[.uint32](point.z))
            for mesh in scene.data.triangle_meshes():
                for point in mesh:
                    _hash(digest, bitcast[.uint32](point.x))
                    _hash(digest, bitcast[.uint32](point.y))
                    _hash(digest, bitcast[.uint32](point.z))
            for values in scene.data.triangle_mesh_normals():
                _floats(digest, values)
            for values in scene.data.triangle_mesh_texcoords():
                _floats(digest, values)
            print("meshes", len(scene.data.triangle_meshes()))

            def bench_pbrt() raises {path}:
                var output = read_pbrt(path)
                keep(len(output.data.triangle_meshes()))
                keep(len(output.data.triangle_vertices()))
                keep(output.data.triangle_meshes().unsafe_ptr())
                keep(output.data.triangle_vertices().unsafe_ptr())

            mean_us = run(bench_pbrt, max_iters=3).mean(Unit.us)
        case "mtl":
            var mesh = read_obj(path)
            for material in mesh.materials:
                for byte in StringSpan(material.name).as_bytes():
                    _hash(digest, UInt32(byte))
                _hash(digest, bitcast[.uint32](material.Kd[0]))
                _hash(digest, bitcast[.uint32](material.Kd[1]))
                _hash(digest, bitcast[.uint32](material.Kd[2]))
                _hash(digest, UInt32(material.map_Kd))
            for texture in mesh.textures:
                for byte in StringSpan(texture.path).as_bytes():
                    _hash(digest, UInt32(byte))
            print(
                "materials",
                mesh.material_count(),
                "textures",
                len(mesh.textures),
            )

            def bench_mtl() raises {path}:
                var output = read_obj(path)
                keep(output.material_count())
                keep(len(output.textures))
                keep(output.materials.unsafe_ptr())
                keep(output.textures.unsafe_ptr())

            mean_us = run(bench_mtl, max_iters=10).mean(Unit.us)
        case _:
            raise Error("unknown parser: " + kind)
    print("digest", digest)
    print("mean_us", mean_us)
