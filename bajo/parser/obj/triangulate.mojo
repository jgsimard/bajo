"""Ordered triangle-fan expansion and shared chunk planning for OBJ output."""

from max.algorithm import parallelize
from std.math import ceildiv
from std.sys import num_logical_cores
from .types import ObjMesh, ObjIndex


comptime _PARALLEL_TRIANGLE_MIN_FACES = 64 * 1024


@fieldwise_init
struct _TriangleChunk(Copyable):
    var begin: Int
    var end: Int
    var input_count: Int
    var output_count: Int
    var input_offset: Int
    var output_offset: Int


@always_inline
def _triangle_counts(mesh: ObjMesh, begin: Int, end: Int) -> Tuple[Int, Int]:
    var input_count = 0
    var output_count = 0
    for face in range(begin, end):
        var count = mesh.face_vertices[face]
        input_count += count
        if count >= 3 and (
            len(mesh.face_lines) == 0 or mesh.face_lines[face] == 0
        ):
            output_count += 3 * (count - 2)
    return input_count, output_count


def _triangle_chunks(
    mesh: ObjMesh, workers: Int, faces_per_task: Int = 16 * 1024
) -> List[_TriangleChunk]:
    """Count disjoint face ranges and prefix their input/output corner offsets.
    """
    var face_count = len(mesh.face_vertices)
    var task_count = min(
        max(1, workers) * 4,
        max(1, ceildiv(face_count, max(1, faces_per_task))),
    )
    var chunks = List[_TriangleChunk](
        length=task_count, fill=_TriangleChunk(0, 0, 0, 0, 0, 0)
    )

    def count_worker(task: Int) {mesh, face_count, task_count, mut chunks}:
        var begin = face_count * task // task_count
        var end = face_count * (task + 1) // task_count
        var counts = _triangle_counts(mesh, begin, end)
        chunks[task] = _TriangleChunk(begin, end, counts[0], counts[1], 0, 0)

    if task_count > 1 and workers > 1:
        parallelize(count_worker, task_count, min(workers, task_count))
    else:
        for task in range(task_count):
            count_worker(task)
    var input_offset = 0
    var output_offset = 0
    for task in range(task_count):
        ref chunk = chunks[task]
        chunk.input_offset = input_offset
        chunk.output_offset = output_offset
        input_offset += chunk.input_count
        output_offset += chunk.output_count
    return chunks^


def triangulated_indices(mesh: ObjMesh) -> List[ObjIndex]:
    """Expand triangle fans in face order, retaining position/UV/normal indices.
    """
    if len(mesh.face_vertices) < _PARALLEL_TRIANGLE_MIN_FACES:
        return _triangulated_indices_serial(mesh)
    var workers = min(8, num_logical_cores())
    if workers <= 1:
        return _triangulated_indices_serial(mesh)
    return _triangulated_indices_parallel(mesh, workers)


def _triangulate_range(
    mesh: ObjMesh,
    output: MutSpan[ObjIndex, _],
    begin: Int,
    end: Int,
    input_offset: Int,
):
    var write = 0
    var offset = input_offset
    for face in range(begin, end):
        var count = mesh.face_vertices[face]
        var is_line = len(mesh.face_lines) > 0 and mesh.face_lines[face] != 0
        if not is_line and count >= 3:
            var first = mesh.indices[offset]
            for corner in range(1, count - 1):
                output[write] = first
                output[write + 1] = mesh.indices[offset + corner]
                output[write + 2] = mesh.indices[offset + corner + 1]
                write += 3
        offset += count


def _triangulated_indices_serial(mesh: ObjMesh) -> List[ObjIndex]:
    var counts = _triangle_counts(mesh, 0, len(mesh.face_vertices))
    var output = List[ObjIndex](unsafe_uninit_length=counts[1])
    _triangulate_range(mesh, Span(output), 0, len(mesh.face_vertices), 0)
    return output^


def _triangulated_indices_parallel(
    mesh: ObjMesh, workers: Int, faces_per_task: Int = 16 * 1024
) -> List[ObjIndex]:
    if workers <= 1:
        return _triangulated_indices_serial(mesh)
    var chunks = _triangle_chunks(mesh, workers, faces_per_task)
    ref last = chunks[len(chunks) - 1]
    var output = List[ObjIndex](
        unsafe_uninit_length=last.output_offset + last.output_count
    )
    if len(output) == 0:
        return output^

    def emit_worker(task: Int) {mesh, chunks, mut output}:
        ref chunk = chunks[task]
        var indices = Span(output)[
            chunk.output_offset : chunk.output_offset + chunk.output_count
        ]
        _triangulate_range(
            mesh, indices, chunk.begin, chunk.end, chunk.input_offset
        )

    if len(chunks) > 1:
        parallelize(emit_worker, len(chunks), min(workers, len(chunks)))
    else:
        emit_worker(0)
    return output^
