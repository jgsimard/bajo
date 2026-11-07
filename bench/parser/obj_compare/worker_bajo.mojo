"""Same public-file timing boundary and canonical output as the C/C++ workers."""

from std.benchmark import keep
from std.io import FileHandle
from std.memory import bitcast
from std.sys import size_of
from std.sys.arg import argv
from std.time import perf_counter_ns

from bajo.parser.asset_loader import PathAssetLoader
from bajo.parser.mmap import MMap
from bajo.parser.obj import ObjMesh, read_obj
from bajo.parser.obj.parallel import _parse_obj_bulk


def load(path: String, workers: Int, variant: String) raises -> ObjMesh:
    if variant == "public":
        return read_obj(path)
    var mapped = MMap(path)
    var loader = PathAssetLoader()
    var chunk_bytes = 1024 * 1024 if variant == "bulk1024" else 256 * 1024
    return _parse_obj_bulk(
        path, mapped.as_string_span(), loader, workers, chunk_bytes
    )


def write_values[T: Copyable](path: String, values: ImmSpan[T, _]) raises:
    with open(path, "w") as file:
        file.write_bytes(
            ImmSpan(
                unsafe_ptr=values.unsafe_ptr().unsafe_bitcast[UInt8](),
                length=len(values) * size_of[T](),
            )
        )


def write_word(mut file: FileHandle, value: UInt32):
    var values = [value]
    file.write_bytes(
        ImmSpan(
            unsafe_ptr=values.unsafe_ptr().unsafe_bitcast[UInt8](),
            length=4,
        )
    )


def write_string(mut file: FileHandle, value: String):
    write_word(file, UInt32(value.byte_length()))
    file.write_bytes(StringSpan(value).as_bytes())


def dump(mesh: ObjMesh, directory: String) raises:
    write_values(directory + "/positions.f32", Span(mesh.positions)[3:])
    write_values(directory + "/texcoords.f32", Span(mesh.texcoords)[2:])
    write_values(directory + "/normals.f32", Span(mesh.normals)[3:])
    write_values(directory + "/colors.f32", Span(mesh.colors))
    var indices = List[Int32](capacity=3 * len(mesh.indices))
    for index in mesh.indices:
        indices.append(index.p - 1)
        indices.append(index.t - 1)
        indices.append(index.n - 1)
    write_values(directory + "/indices.i32", Span(indices))
    var faces = List[UInt32](capacity=len(mesh.face_vertices))
    var materials = List[Int32](capacity=len(mesh.face_vertices))
    for face in range(len(mesh.face_vertices)):
        if len(mesh.face_lines) and mesh.face_lines[face]:
            raise Error("line fixture unsupported by comparison")
        faces.append(UInt32(mesh.face_vertices[face]))
        materials.append(
            Int32(mesh.face_materials[face]) if len(mesh.materials) else Int32(
                -1
            )
        )
    write_values(directory + "/faces.u32", Span(faces))
    write_values(directory + "/face_materials.i32", Span(materials))
    with open(directory + "/materials.bin", "w") as file:
        for material in mesh.materials:
            write_string(file, material.name)
            comptime for axis in range(3):
                write_word(file, bitcast[.uint32](material.Kd[axis]))
            write_string(file, mesh.textures[material.map_Kd].name)


def main() raises:
    var args = argv()
    if len(args) != 8:
        raise Error(
            "usage: bajo path repeats warmup threads variant cache"
            " dump-dir-or-dash"
        )
    var path = String(args[1])
    var repeats = Int(args[2])
    var warmup = Int(args[3])
    var workers = Int(args[4])
    var variant = String(args[5])
    var directory = String(args[7])
    if repeats < 1 or warmup < 0:
        raise Error("invalid repetitions")
    var samples = List[Int]()
    var vertices = 0
    var faces = 0
    var corners = 0
    var materials = 0
    for iteration in range(warmup + repeats):
        var start = perf_counter_ns()
        var mesh = load(path, workers, variant)
        var elapsed = perf_counter_ns() - start
        keep(mesh.positions.unsafe_ptr())
        keep(mesh.indices.unsafe_ptr())
        vertices = mesh.position_count(include_dummy=False)
        faces = len(mesh.face_vertices)
        corners = len(mesh.indices)
        materials = len(mesh.materials)
        if iteration >= warmup:
            samples.append(elapsed)
        if iteration + 1 == warmup + repeats and directory != "-":
            dump(mesh, directory)
    print(
        '{"samples_ns":',
        samples,
        ',"vertices":',
        vertices,
        ',"faces":',
        faces,
        ',"corners":',
        corners,
        ',"materials":',
        materials,
        "}",
    )
