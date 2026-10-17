"""Parser timings and full geometry/pixel digests for before/after comparisons.

Build with `mojo build -O3 -I . bench/parser/bench_loaders.mojo -o /tmp/loaders`
and run `/tmp/loaders <obj|ply|png|png-libpng|exr|pbrt> <path>`.
Use `png-libpng` as the reference for the SIMD PNG loader.
PLY reports file-read + parse time as `read_mean_us` and mapped-input parse
time as `mean_us`; both timings exclude the digest calculation.
"""

from std.benchmark import Unit, keep, run
from std.memory import bitcast
from std.sys.arg import argv

from bajo.parser.mmap import MMap
from bajo.parser.ply import parse_ply, read_ply
from bajo.parser.png import parse_png, _parse_png_libpng
from bajo.parser.exr import parse_exr
from bajo.parser.pbrt import read_pbrt
from bajo.parser.obj import read_obj
from bajo.rt.material_types import ImageTexture, NO_TEXTURE


@always_inline
def _hash(mut digest: UInt64, word: UInt32):
    digest = (digest ^ UInt64(word)) * UInt64(1099511628211)


def _floats(mut digest: UInt64, values: ImmSpan[Float32, _]):
    for value in values:
        _hash(digest, bitcast[.uint32](value))


def _image(mut digest: UInt64, images: ImmSpan[ImageTexture, _], index: UInt32):
    if index != NO_TEXTURE:
        ref image = images[Int(index)]
        _hash(digest, UInt32(image.width))
        _hash(digest, UInt32(image.height))
        _floats(digest, image.pixels)


def main() raises:
    var args = argv()
    if len(args) != 3:
        raise Error(
            "usage: bench_loaders <obj|ply|png|png-libpng|exr|pbrt> <path>"
        )
    var kind = String(args[1])
    var path = String(args[2])
    var mapped = MMap(path)
    var bytes = mapped.as_bytes_span()
    var digest = UInt64(14695981039346656037)
    var mean_us: Float64
    match kind:
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

            def bench_read_ply() raises {path}:
                var output = read_ply(path)
                keep(output.vertex_count())
                keep(output.triangle_count())
                keep(output.positions.unsafe_ptr())
                keep(output.normals.unsafe_ptr())
                keep(output.texcoords.unsafe_ptr())
                keep(output.indices.unsafe_ptr())

            print(
                "read_mean_us", run(bench_read_ply, max_iters=10).mean(Unit.us)
            )
            mean_us = run(bench_ply, max_iters=10).mean(Unit.us)
        case "png" | "png-libpng":
            var reference = kind == "png-libpng"
            var image = _parse_png_libpng(bytes) if reference else parse_png(
                bytes
            )
            _floats(digest, image.pixels)
            print("size", image.width, image.height)

            def bench_png() raises {bytes, reference}:
                var output = _parse_png_libpng(
                    bytes
                ) if reference else parse_png(bytes)
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
            # Hash geometry per instance and images per material reference so
            # storage deduplication does not change the semantic digest.
            for instance in scene.data.triangle_instances():
                var mesh_index = Int(instance.blas_idx)
                for local in scene.data.triangle_meshes()[mesh_index]:
                    var point = instance.transform.point(local)
                    _hash(digest, bitcast[.uint32](point.x))
                    _hash(digest, bitcast[.uint32](point.y))
                    _hash(digest, bitcast[.uint32](point.z))
                _floats(digest, scene.data.triangle_mesh_normals()[mesh_index])
                _floats(
                    digest, scene.data.triangle_mesh_texcoords()[mesh_index]
                )
            ref surfaces = scene.data.surfaces()
            for material in surfaces.lambertians:
                _image(digest, surfaces.image_textures, material.texture_index)
            for material in surfaces.metals:
                _image(digest, surfaces.image_textures, material.texture_index)
            for material in surfaces.coated_diffuses:
                _image(digest, surfaces.image_textures, material.texture_index)
                _image(
                    digest,
                    surfaces.image_textures,
                    material.displacement_texture_index,
                )
            _image(
                digest,
                surfaces.image_textures,
                scene.data.environment().texture_index,
            )
            print(
                "meshes",
                len(scene.data.triangle_meshes()),
                "instances",
                len(scene.data.triangle_instances()),
                "images",
                len(surfaces.image_textures),
            )

            def bench_pbrt() raises {path}:
                var output = read_pbrt(path)
                keep(len(output.data.triangle_meshes()))
                keep(len(output.data.triangle_vertices()))
                keep(output.data.triangle_meshes().unsafe_ptr())
                keep(output.data.triangle_vertices().unsafe_ptr())
                keep(output.data.triangle_instances().unsafe_ptr())
                keep(output.data.surfaces().image_textures.unsafe_ptr())

            mean_us = run(bench_pbrt, max_iters=3).mean(Unit.us)
        case "obj":
            var mesh = read_obj(path)
            _floats(digest, mesh.positions)
            _floats(digest, mesh.normals)
            _floats(digest, mesh.texcoords)
            _floats(digest, mesh.colors)
            for index in mesh.indices:
                _hash(digest, UInt32(index.p))
                _hash(digest, UInt32(index.t))
                _hash(digest, UInt32(index.n))
            for count in mesh.face_vertices:
                _hash(digest, UInt32(count))
            for material in mesh.face_materials:
                _hash(digest, UInt32(material))
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
                "vertices",
                mesh.position_count(include_dummy=False),
                "faces",
                mesh.face_count(),
                "materials",
                mesh.material_count(),
                "textures",
                len(mesh.textures),
            )

            def bench_obj() raises {path}:
                var output = read_obj(path)
                keep(output.positions.unsafe_ptr())
                keep(output.indices.unsafe_ptr())
                keep(output.material_count())
                keep(len(output.textures))
                keep(output.materials.unsafe_ptr())
                keep(output.textures.unsafe_ptr())

            mean_us = run(bench_obj, max_iters=10).mean(Unit.us)
        case _:
            raise Error("unknown parser: " + kind)
    print("digest", digest)
    print("mean_us", mean_us)
