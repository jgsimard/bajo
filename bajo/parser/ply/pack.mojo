"""Pack indexed PLY attributes into triangle order with a bounds reduction."""

from max.algorithm import parallelize
from std.math import ceildiv

from bajo.core import AABB, Point3f32, cross, length2
from std.utils.numerics import isfinite
from bajo.parser.ply.types import PlyMesh


def _remove_degenerate_triangles(mut mesh: PlyMesh) raises:
    """Compact imported indices, preserving attributes and valid face order.

    Zero-area faces cannot intersect a ray. Invalid indices and non-finite
    geometry remain errors; this does not relax finalized-scene validation.
    """
    if len(mesh.indices) % 3:
        raise Error("PLY triangles require complete index triples")

    @always_inline
    def point(index: UInt32) raises {mesh} -> Point3f32[.LOCAL]:
        if index >= UInt32(mesh.vertex_count()):
            raise Error("PLY face index is out of range")
        var source = 3 * Int(index)
        var result = Point3f32[.LOCAL](
            mesh.positions[source],
            mesh.positions[source + 1],
            mesh.positions[source + 2],
        )
        if not result.is_finite()[0]:
            raise Error("PLY triangle vertices must be finite")
        return result

    var write = 0
    for base in range(0, len(mesh.indices), 3):
        var p0 = point(mesh.indices[base])
        var p1 = point(mesh.indices[base + 1])
        var p2 = point(mesh.indices[base + 2])
        var area = length2(cross(p1 - p0, p2 - p0))[0]
        if not isfinite(area):
            raise Error("PLY triangle area must be finite")
        if area == 0.0:
            continue
        if write != base:
            for lane in range(3):
                mesh.indices[write + lane] = mesh.indices[base + lane]
        write += 3
    mesh.indices.resize(write, fill=UInt32(0))


@fieldwise_init
struct _PackedPlyMesh:
    var vertices: List[Point3f32[.LOCAL]]
    var normals: List[Float32]
    var texcoords: List[Float32]
    var bounds: AABB[.LOCAL]

    def transfer_to(
        deinit self,
        mut vertices: List[Point3f32[.LOCAL]],
        mut normals: List[Float32],
        mut texcoords: List[Float32],
    ) -> AABB[.LOCAL]:
        vertices = self.vertices^
        normals = self.normals^
        texcoords = self.texcoords^
        return self.bounds


def _pack_range(
    mesh: PlyMesh,
    vertices: MutSpan[Point3f32[.LOCAL], _],
    normals: MutSpan[Float32, _],
    texcoords: MutSpan[Float32, _],
    begin: Int,
    reverse_orientation: Bool,
) raises -> AABB[.LOCAL]:
    var bounds = AABB[.LOCAL].invalid()
    for base in range(0, len(vertices), 3):
        for lane in range(3):
            var source_lane = (
                3 - lane if reverse_orientation and lane > 0 else lane
            )
            var index = Int(mesh.indices[begin + base + source_lane])
            if index >= mesh.vertex_count():
                raise Error("PLY face index is out of range")
            var source = 3 * index
            var point = Point3f32[.LOCAL](
                mesh.positions[source],
                mesh.positions[source + 1],
                mesh.positions[source + 2],
            )
            var target = base + lane
            vertices[target] = point
            bounds.grow(point)
            if len(normals):
                normals[3 * target] = mesh.normals[source]
                normals[3 * target + 1] = mesh.normals[source + 1]
                normals[3 * target + 2] = mesh.normals[source + 2]
            if len(texcoords):
                texcoords[2 * target] = mesh.texcoords[2 * index]
                texcoords[2 * target + 1] = mesh.texcoords[2 * index + 1]
    return bounds


def _pack_ply_mesh(
    mesh: PlyMesh,
    reverse_orientation: Bool,
    workers: Int = 1,
    triangles_per_task: Int = 16 * 1024,
) raises -> _PackedPlyMesh:
    if len(mesh.indices) == 0:
        raise Error("PBRT plymesh must contain at least one triangle")
    if len(mesh.indices) % 3:
        raise Error("PBRT plymesh requires complete index triples")
    var vertices = List[Point3f32[.LOCAL]]()
    var normals = List[Float32]()
    var texcoords = List[Float32]()
    vertices.resize(unsafe_uninit_length=len(mesh.indices))
    if mesh.has_normals():
        normals.resize(unsafe_uninit_length=3 * len(mesh.indices))
    if mesh.has_texcoords():
        texcoords.resize(unsafe_uninit_length=2 * len(mesh.indices))
    var tasks = min(
        max(1, workers) * 4,
        max(1, ceildiv(mesh.triangle_count(), max(1, triangles_per_task))),
    )
    if mesh.triangle_count() < 64 * 1024:
        tasks = 1
    var bounds = List[AABB[.LOCAL]](length=tasks, fill=AABB[.LOCAL].invalid())
    var errors = List[String](length=tasks, fill="")

    def pack(
        task: Int,
    ) {imm, mut vertices, mut normals, mut texcoords, mut bounds, mut errors}:
        var begin = 3 * (mesh.triangle_count() * task // tasks)
        var end = 3 * (mesh.triangle_count() * (task + 1) // tasks)
        try:
            var normal_output = Span(normals)
            var uv_output = Span(texcoords)
            if len(normals):
                normal_output = normal_output[3 * begin : 3 * end]
            if len(texcoords):
                uv_output = uv_output[2 * begin : 2 * end]
            bounds[task] = _pack_range(
                mesh,
                Span(vertices)[begin:end],
                normal_output,
                uv_output,
                begin,
                reverse_orientation,
            )
        except error:
            errors[task] = String(error)

    if workers > 1 and tasks > 1:
        parallelize(pack, tasks, min(workers, tasks))
    else:
        for task in range(tasks):
            pack(task)
    var combined = AABB[.LOCAL].invalid()
    for task in range(tasks):
        if errors[task].byte_length():
            raise Error(errors[task])
        combined.grow(bounds[task])
    return _PackedPlyMesh(vertices^, normals^, texcoords^, combined)
