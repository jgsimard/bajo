from std.memory import bitcast
from std.testing import (
    assert_almost_equal,
    assert_raises,
    assert_true,
    TestSuite,
    assert_equal,
)

from bajo.parser.ply import parse_ply
from bajo.parser.ply.parser import _parse_ply_bulk, _parse_ply_serial
from bajo.parser.ply.parser import (
    _decode_common_vertices,
    _emit_triangles,
    _parse_header,
)
from std.sys import simd_width_of
from bajo.parser.ply.types import PlyMesh
from bajo.parser.ply.pack import _pack_ply_mesh


def _append_text(mut bytes: List[UInt8], text: String):
    var span = StringSpan(text)
    for c in span.as_bytes():
        bytes.append(c)


def _append_u32(mut bytes: List[UInt8], value: UInt32):
    bytes.append(UInt8(value & UInt32(0xFF)))
    bytes.append(UInt8((value >> UInt32(8)) & UInt32(0xFF)))
    bytes.append(UInt8((value >> UInt32(16)) & UInt32(0xFF)))
    bytes.append(UInt8(value >> UInt32(24)))


def _append_f32(mut bytes: List[UInt8], value: Float32):
    _append_u32(bytes, bitcast[.uint32](value))


def _fixture(with_uv: Bool = True, quad: Bool = False) -> List[UInt8]:
    var bytes = List[UInt8]()
    var vertex_count = 4 if quad else 3
    var header = "ply\nformat binary_little_endian 1.0\n"
    header += "comment generated in test\n"
    header += "element vertex " + String(vertex_count) + "\n"
    header += "property float x\nproperty float y\nproperty float z\n"
    header += "property float nx\nproperty float ny\nproperty float nz\n"
    if with_uv:
        header += "property float u\nproperty float v\n"
    header += "element face 1\n"
    header += "property list uchar int vertex_indices\nend_header\n"
    _append_text(bytes, header)

    for i in range(vertex_count):
        var x = Float32(1.0) if i == 1 or i == 2 else Float32(0.0)
        var y = Float32(1.0) if i >= 2 else Float32(0.0)
        _append_f32(bytes, x)
        _append_f32(bytes, y)
        _append_f32(bytes, 0.0)
        _append_f32(bytes, 0.0)
        _append_f32(bytes, 0.0)
        _append_f32(bytes, 1.0)
        if with_uv:
            _append_f32(bytes, x)
            _append_f32(bytes, y)

    bytes.append(UInt8(vertex_count))
    for i in range(vertex_count):
        _append_u32(bytes, UInt32(i))
    return bytes^


def test_binary_triangle_with_normals_and_uvs() raises:
    var bytes = _fixture()
    var mesh = parse_ply(bytes)
    assert_equal(mesh.vertex_count(), 3)
    assert_equal(mesh.face_count(), 1)
    assert_equal(mesh.triangle_count(), 1)
    assert_true(mesh.has_normals())
    assert_true(mesh.has_texcoords())
    assert_almost_equal(mesh.positions[3], 1.0)
    assert_almost_equal(mesh.positions[7], 1.0)
    assert_almost_equal(mesh.normals[8], 1.0)
    assert_almost_equal(mesh.texcoords[2], 1.0)
    assert_equal(mesh.indices, [0, 1, 2])


def test_quad_is_fan_triangulated_without_uvs() raises:
    var bytes = _fixture(with_uv=False, quad=True)
    var mesh = parse_ply(bytes)
    assert_equal(mesh.vertex_count(), 4)
    assert_equal(mesh.face_count(), 1)
    assert_equal(mesh.triangle_count(), 2)
    assert_true(mesh.has_normals())
    assert_true(not mesh.has_texcoords())
    assert_equal(len(mesh.indices), 6)
    assert_equal(mesh.indices, [0, 1, 2, 0, 2, 3])


def test_rejects_unsupported_format() raises:
    var bytes = List[UInt8]()
    _append_text(bytes, "ply\nformat ascii 1.0\nend_header\n")
    with assert_raises():
        _ = parse_ply(bytes)


def test_rejects_out_of_range_face_index() raises:
    var bytes = _fixture()
    var n = len(bytes)
    bytes[n - 4] = UInt8(9)
    with assert_raises():
        _ = parse_ply(bytes)


def test_rejects_truncated_payload() raises:
    var bytes = _fixture()
    _ = bytes.pop()
    with assert_raises():
        _ = parse_ply(bytes)


def _append_u16(mut bytes: List[UInt8], value: UInt16):
    bytes.append(UInt8(value))
    bytes.append(UInt8(value >> UInt16(8)))


def _mixed_fixture(vertices: Int, faces: Int) -> List[UInt8]:
    var bytes = List[UInt8]()
    _append_text(
        bytes,
        "ply\nformat binary_little_endian 1.0\nelement vertex "
        + String(vertices)
        + "\nproperty float z\nproperty uchar ignored\nproperty double"
        " x\nproperty float y\nproperty char nx\nproperty char ny\nproperty"
        " char nz\nproperty ushort u\nproperty short v\nelement face "
        + String(faces)
        + "\nproperty float ignored\nproperty list uchar ushort other\nproperty"
        " list uchar int vertex_indices\nproperty uchar tail\nend_header\n",
    )
    for vertex in range(vertices):
        _append_f32(bytes, -0.0 if vertex % 2 == 0 else Float32(vertex))
        bytes.append(UInt8(255))
        var bits = bitcast[.uint64](Float64(vertex) + 0.25)
        for shift in range(8):
            bytes.append(UInt8(bits >> UInt64(shift * 8)))
        _append_f32(bytes, Float32(vertex) / 3.0)
        bytes.extend([UInt8(255), UInt8(0), UInt8(1)])
        _append_u16(bytes, UInt16(vertex % 65536))
        _append_u16(bytes, UInt16(65535))
    for face in range(faces):
        _append_f32(bytes, bitcast[.float32](UInt32(0x7FC00000)))
        bytes.append(UInt8(2))
        _append_u16(bytes, UInt16(5))
        _append_u16(bytes, UInt16(7))
        var corners = 3 + face % 3
        bytes.append(UInt8(corners))
        for corner in range(corners):
            _append_u32(bytes, UInt32((face + corner) % vertices))
        bytes.append(UInt8(17))
    return bytes^


def _same_mesh(actual: PlyMesh, expected: PlyMesh) raises:
    assert_equal(actual.face_count(), expected.face_count())
    assert_equal(actual.indices, expected.indices)
    assert_equal(len(actual.positions), len(expected.positions))
    assert_equal(len(actual.normals), len(expected.normals))
    assert_equal(len(actual.texcoords), len(expected.texcoords))
    for i in range(len(actual.positions)):
        assert_equal(
            bitcast[.uint32](actual.positions[i]),
            bitcast[.uint32](expected.positions[i]),
        )
    for i in range(len(actual.normals)):
        assert_equal(
            bitcast[.uint32](actual.normals[i]),
            bitcast[.uint32](expected.normals[i]),
        )
    for i in range(len(actual.texcoords)):
        assert_equal(
            bitcast[.uint32](actual.texcoords[i]),
            bitcast[.uint32](expected.texcoords[i]),
        )


def test_bulk_matches_serial_for_mixed_types_polygons_and_chunk_boundaries() raises:
    # Unaligned float64, reordered properties, signed integers, ignored scalars
    # and lists, mixed triangle/quad/ngon records, and multiple worker chunks.
    var bytes = _mixed_fixture(32771, 8201)
    var expected = _parse_ply_serial(bytes)
    for workers in [1, 2, 8]:
        _same_mesh(_parse_ply_bulk(bytes, workers), expected)
    _same_mesh(parse_ply(bytes), expected)
    assert_equal(expected.normals[0], Float32(-1))
    assert_equal(expected.texcoords[1], Float32(-1))


def test_bulk_handles_empty_face_stream() raises:
    var bytes = _mixed_fixture(3, 0)
    _same_mesh(_parse_ply_bulk(bytes, 8), _parse_ply_serial(bytes))


def test_bulk_rejects_invalid_late_chunk_and_truncation() raises:
    var bytes = _mixed_fixture(32771, 8201)
    var good = bytes.copy()
    # Final index lies just before the ignored one-byte tail.
    for i in range(4):
        bytes[len(bytes) - 5 + i] = UInt8(255)
    with assert_raises():
        _ = _parse_ply_bulk(bytes, 8)
    with assert_raises():
        _ = parse_ply(bytes)
    _ = good.pop()
    with assert_raises():
        _ = parse_ply(good)


def test_parallel_packing_preserves_attributes_orientation_and_bounds() raises:
    var bytes = _mixed_fixture(17, 32771)
    var mesh = _parse_ply_bulk(bytes, 8)
    for reverse in [False, True]:
        var serial = _pack_ply_mesh(mesh, reverse, 1)
        var parallel = _pack_ply_mesh(mesh, reverse, 8)
        assert_equal(len(parallel.vertices), len(mesh.indices))
        assert_equal(parallel.normals, serial.normals)
        assert_equal(parallel.texcoords, serial.texcoords)
        for i in range(len(mesh.indices)):
            var lane = i % 3
            var source = Int(
                mesh.indices[
                    i - lane + (3 - lane if reverse and lane else lane)
                ]
            )
            assert_equal(
                bitcast[.uint32](parallel.vertices[i].x),
                bitcast[.uint32](mesh.positions[3 * source]),
            )
            assert_equal(
                bitcast[.uint32](parallel.vertices[i].y),
                bitcast[.uint32](mesh.positions[3 * source + 1]),
            )
            assert_equal(
                bitcast[.uint32](parallel.vertices[i].z),
                bitcast[.uint32](mesh.positions[3 * source + 2]),
            )
            assert_equal(parallel.normals[3 * i], mesh.normals[3 * source])
            assert_equal(parallel.texcoords[2 * i], mesh.texcoords[2 * source])
        assert_equal(
            bitcast[.uint32](parallel.bounds._min.x),
            bitcast[.uint32](serial.bounds._min.x),
        )
        assert_equal(
            bitcast[.uint32](parallel.bounds._min.y),
            bitcast[.uint32](serial.bounds._min.y),
        )
        assert_equal(
            bitcast[.uint32](parallel.bounds._min.z),
            bitcast[.uint32](serial.bounds._min.z),
        )
        assert_equal(
            bitcast[.uint32](parallel.bounds._max.x),
            bitcast[.uint32](serial.bounds._max.x),
        )
        assert_equal(
            bitcast[.uint32](parallel.bounds._max.y),
            bitcast[.uint32](serial.bounds._max.y),
        )
        assert_equal(
            bitcast[.uint32](parallel.bounds._max.z),
            bitcast[.uint32](serial.bounds._max.z),
        )


def test_packing_handles_absent_optional_attributes_and_bad_indices() raises:
    var bytes = _fixture(with_uv=False)
    var mesh = parse_ply(bytes)
    mesh.normals.clear()
    var packed = _pack_ply_mesh(mesh, True, 8)
    assert_equal(len(packed.normals), 0)
    assert_equal(len(packed.texcoords), 0)
    assert_equal(packed.vertices[1].y, mesh.positions[7])
    mesh.indices[2] = UInt32(mesh.vertex_count())
    with assert_raises():
        _ = _pack_ply_mesh(mesh, False, 8)


def _common_fixture(
    vertices: Int,
    faces: Int,
    with_uv: Bool,
    padding: Int = 0,
    quad_at: Int = -1,
) -> List[UInt8]:
    var header = String("ply\nformat binary_little_endian 1.0\ncomment ")
    for _ in range(padding):
        header += "x"
    header += String(t"\nelement vertex {vertices}\n")
    header += "property float x\nproperty float y\nproperty float z\n"
    header += "property float nx\nproperty float ny\nproperty float nz\n"
    if with_uv:
        header += "property float u\nproperty float v\n"
    header += String(t"element face {faces}\n")
    header += "property list uchar int vertex_indices\nend_header\n"
    var bytes = List[UInt8]()
    _append_text(bytes, header)
    # Include signed zero, subnormals, infinities and NaN payloads. Decoding
    # rearranges stored values without doing floating-point arithmetic.
    var patterns = [
        UInt32(0),
        0x80000000,
        0x3F800000,
        0x7F800000,
        0xFF800000,
        0x7FC12345,
        0x7FA00001,
        0x80000001,
        0x7F7FFFFF,
        1,
        0xFFFFFFFF,
    ]
    for vertex in range(vertices):
        for lane in range(8 if with_uv else 6):
            _append_u32(bytes, patterns[(8 * vertex + lane) % len(patterns)])
    for face in range(faces):
        var count = 4 if face == quad_at else 3
        bytes.append(UInt8(count))
        for corner in range(count):
            _append_u32(bytes, UInt32((3 * face + corner) % vertices))
    return bytes^


def test_common_simd_layouts_preserve_bits_alignment_and_tails() raises:
    comptime width = simd_width_of[Float32]()
    for with_uv in [False, True]:
        for padding in range(8):
            for vertices in range(2 * width + 5):
                var bytes = _common_fixture(
                    vertices, vertices if vertices >= 3 else 0, with_uv, padding
                )
                var expected = _parse_ply_serial(bytes)
                _same_mesh(parse_ply(bytes), expected)
                _same_mesh(_parse_ply_bulk(bytes, 8), expected)


def test_common_simd_layouts_preserve_worker_and_polygon_boundaries() raises:
    for with_uv in [False, True]:
        for quad_at in [-1, 4095, 4096, 8200]:
            var bytes = _common_fixture(32771, 8201, with_uv, 1, quad_at)
            var expected = _parse_ply_serial(bytes)
            for workers in [1, 2, 8]:
                _same_mesh(_parse_ply_bulk(bytes, workers), expected)
            _same_mesh(parse_ply(bytes), expected)


def test_triangle_simd_stores_stay_inside_the_output_span() raises:
    comptime width = min(64, simd_width_of[UInt8]())
    for count in range(2 * width + 5):
        var bytes = List[UInt8](length=7, fill=UInt8(255))
        for face in range(count):
            bytes.append(UInt8(3))
            for corner in range(3):
                _append_u32(bytes, UInt32((3 * face + corner) % 17))
        var output = List[UInt32](length=3 * count + 6, fill=UInt32(0xDEADBEEF))
        _emit_triangles(Span(bytes)[7:], Span(output)[3 : 3 + 3 * count], 17)
        for index in range(3 * count):
            assert_equal(output[3 + index], UInt32(index % 17))
        for guard in range(3):
            assert_equal(output[guard], UInt32(0xDEADBEEF))
            assert_equal(output[3 + 3 * count + guard], UInt32(0xDEADBEEF))


def test_vertex_simd_stores_stay_inside_the_worker_range() raises:
    comptime width = simd_width_of[Float32]()
    comptime for record_width in [6, 8]:
        for padding in range(8):
            for count in range(2 * width + 5):
                var bytes = _common_fixture(
                    count + 6, 0, record_width == 8, padding
                )
                var header = _parse_header(bytes)
                var expected = _parse_ply_serial(bytes)
                var guard = bitcast[.float32](UInt32(0xDEADBEEF))
                var positions = List[Float32](
                    length=3 * (count + 6), fill=guard
                )
                var normals = List[Float32](length=3 * (count + 6), fill=guard)
                var texcoords = List[Float32](
                    length=2 * (count + 6), fill=guard
                )
                _decode_common_vertices[record_width](
                    bytes,
                    header.data_offset,
                    positions,
                    normals,
                    texcoords,
                    3,
                    3 + count,
                )
                for lane in range(len(positions)):
                    var active = lane >= 9 and lane < 3 * (3 + count)
                    assert_equal(
                        bitcast[.uint32](positions[lane]),
                        bitcast[.uint32](
                            expected.positions[lane]
                        ) if active else UInt32(0xDEADBEEF),
                    )
                    assert_equal(
                        bitcast[.uint32](normals[lane]),
                        bitcast[.uint32](
                            expected.normals[lane]
                        ) if active else UInt32(0xDEADBEEF),
                    )
                for lane in range(len(texcoords)):
                    var active = (
                        record_width == 8
                        and lane >= 6
                        and lane < 2 * (3 + count)
                    )
                    if active:
                        assert_equal(
                            bitcast[.uint32](texcoords[lane]),
                            bitcast[.uint32](expected.texcoords[lane]),
                        )
                    else:
                        assert_equal(
                            bitcast[.uint32](texcoords[lane]),
                            UInt32(0xDEADBEEF),
                        )


def _same_parse_failure(bytes: ImmSpan[UInt8, _]) raises:
    var expected = String()
    try:
        _ = _parse_ply_serial(bytes)
    except error:
        expected = String(error)
    assert_true(expected.byte_length() > 0)
    var actual = String()
    try:
        _ = parse_ply(bytes)
    except error:
        actual = String(error)
    assert_equal(actual, expected)


def test_common_simd_preserves_invalid_input_diagnostics() raises:
    comptime faces = 3 * (min(64, simd_width_of[UInt8]()) // 16) + 5
    for with_uv in [False, True]:
        var good = _common_fixture(17, faces, with_uv, 1)
        var header = _parse_header(good)
        var face_begin = len(good) - 13 * faces
        for length in range(header.data_offset, len(good)):
            _same_parse_failure(Span(good)[:length])
        for face in range(faces):
            for count in [UInt8(0), 1, 2, 18, 255]:
                var bytes = good.copy()
                bytes[face_begin + 13 * face] = count
                _same_parse_failure(bytes)
            for corner in range(3):
                for bad in [UInt32(17), 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF]:
                    var bytes = good.copy()
                    var offset = face_begin + 13 * face + 1 + 4 * corner
                    for lane in range(4):
                        bytes[offset + lane] = UInt8(
                            (bad >> UInt32(8 * lane)) & 255
                        )
                    _same_parse_failure(bytes)
    # Exercise the parallel path and both sides of its 4096-face boundaries.
    var large = _common_fixture(32771, 8201, True, 1)
    var face_begin = len(large) - 13 * 8201
    for face in [0, 4095, 4096, 8200]:
        var bytes = large.copy()
        bytes[face_begin + 13 * face] = UInt8(2)
        _same_parse_failure(bytes)
        bytes = large.copy()
        for lane in range(4):
            bytes[face_begin + 13 * face + 1 + lane] = UInt8(255)
        _same_parse_failure(bytes)


def test_triangle_simd_rejects_bad_indices_in_every_vector_and_tail_lane() raises:
    comptime faces = 3 * (min(64, simd_width_of[UInt8]()) // 16) + 5
    var good = List[UInt8]()
    for _ in range(faces):
        good.append(UInt8(3))
        for corner in range(3):
            _append_u32(good, UInt32(corner))
    for index in range(3 * faces):
        for bad in [UInt32(17), 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF]:
            var bytes = good.copy()
            var offset = 13 * (index // 3) + 1 + 4 * (index % 3)
            for lane in range(4):
                bytes[offset + lane] = UInt8((bad >> UInt32(8 * lane)) & 255)
            var output = List[UInt32](length=3 * faces, fill=0)
            with assert_raises():
                _emit_triangles(bytes, output, 17)


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
