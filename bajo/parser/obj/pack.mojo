from bajo.core import Point3f32, Frame
from bajo.parser.obj import ObjMesh, read_obj
from max.algorithm import parallelize
from std.sys import num_logical_cores
from .triangulate import (
    _PARALLEL_TRIANGLE_MIN_FACES,
    _triangle_counts,
    _triangle_chunks,
)


def pack_obj_triangles[
    frame: Frame = .LOCAL
](path: String) raises -> List[Point3f32[frame]]:
    var mesh = read_obj(path)
    return pack_obj_triangles[frame](mesh)


def pack_obj_triangles[
    frame: Frame = .LOCAL
](mesh: ObjMesh) -> List[Point3f32[frame]]:
    """Expand triangle fans in face order, skipping OBJ line elements."""
    if len(mesh.face_vertices) < _PARALLEL_TRIANGLE_MIN_FACES:
        return _pack_obj_triangles_serial[frame](mesh)
    var workers = min(8, num_logical_cores())
    if workers <= 1:
        return _pack_obj_triangles_serial[frame](mesh)
    return _pack_obj_triangles_parallel[frame](mesh, workers)


def _pack_obj_triangles_serial[
    frame: Frame = .LOCAL
](mesh: ObjMesh) -> List[Point3f32[frame]]:
    var counts = _triangle_counts(mesh, 0, len(mesh.face_vertices))
    var out = List[Point3f32[frame]](unsafe_uninit_length=counts[1])
    _pack_range[frame](mesh, Span(out), 0, len(mesh.face_vertices), 0)
    return out^


def _pack_range[
    frame: Frame
](
    mesh: ObjMesh,
    output: MutSpan[Point3f32[frame], _],
    begin: Int,
    end: Int,
    index_offset: Int,
):
    var write = 0
    var offset = index_offset

    @always_inline
    def point(index: Int) {mesh} -> Point3f32[frame]:
        var base = 3 * Int(mesh.indices[index].p)
        return Point3f32[frame].load(mesh.positions, base)

    for face in range(begin, end):
        var count = mesh.face_vertices[face]
        var is_line = len(mesh.face_lines) > 0 and mesh.face_lines[face] != 0
        if not is_line and count >= 3:
            var first = point(offset)
            var previous = point(offset + 1)
            for corner in range(2, count):
                var next = point(offset + corner)
                output[write] = first
                output[write + 1] = previous
                output[write + 2] = next
                previous = next
                write += 3
        offset += count


def _pack_obj_triangles_parallel[
    frame: Frame = .LOCAL
](mesh: ObjMesh, workers: Int, faces_per_task: Int = 16 * 1024) -> List[
    Point3f32[frame]
]:
    if workers <= 1:
        return _pack_obj_triangles_serial[frame](mesh)
    var chunks = _triangle_chunks(mesh, workers, faces_per_task)
    ref last = chunks[len(chunks) - 1]
    var output = List[Point3f32[frame]](
        unsafe_uninit_length=last.output_offset + last.output_count
    )
    if len(output) == 0:
        return output^

    def pack_worker(task: Int) {mesh, chunks, mut output}:
        ref chunk = chunks[task]
        # Each worker owns a contiguous output slice; triangle order follows
        # face order even when workers finish in a different order.
        var vertices = Span(output)[
            chunk.output_offset : chunk.output_offset + chunk.output_count
        ]
        _pack_range[frame](
            mesh, vertices, chunk.begin, chunk.end, chunk.input_offset
        )

    if len(chunks) > 1:
        parallelize(pack_worker, len(chunks), min(workers, len(chunks)))
    else:
        pack_worker(0)
    return output^
