from std.memory import bitcast
from std.testing import assert_almost_equal, assert_raises, assert_true

from bajo.parser.ply import parse_ply


def _append_text(mut bytes: List[UInt8], text: String):
    var span = StringSpan(text)
    for c in span.as_bytes():
        bytes.append(c)


def _append_u32(mut bytes: List[UInt8], value: UInt32):
    bytes.append(UInt8(value & UInt32(0xFF)))
    bytes.append(UInt8((value >> UInt32(8)) & UInt32(0xFF)))
    bytes.append(UInt8((value >> UInt32(16)) & UInt32(0xFF)))
    bytes.append(UInt8(value >> UInt32(24)))


def _append_i32(mut bytes: List[UInt8], value: Int32):
    _append_u32(bytes, bitcast[.uint32](value))


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
        _append_i32(bytes, Int32(i))
    return bytes^


def test_binary_triangle_with_normals_and_uvs() raises:
    var bytes = _fixture()
    var mesh = parse_ply(bytes)
    assert_true(mesh.vertex_count() == 3)
    assert_true(mesh.face_count() == 1)
    assert_true(mesh.triangle_count() == 1)
    assert_true(mesh.has_normals())
    assert_true(mesh.has_texcoords())
    assert_almost_equal(mesh.positions[3], 1.0)
    assert_almost_equal(mesh.positions[7], 1.0)
    assert_almost_equal(mesh.normals[8], 1.0)
    assert_almost_equal(mesh.texcoords[2], 1.0)
    assert_true(mesh.indices[0] == UInt32(0))
    assert_true(mesh.indices[1] == UInt32(1))
    assert_true(mesh.indices[2] == UInt32(2))


def test_quad_is_fan_triangulated_without_uvs() raises:
    var bytes = _fixture(with_uv=False, quad=True)
    var mesh = parse_ply(bytes)
    assert_true(mesh.vertex_count() == 4)
    assert_true(mesh.face_count() == 1)
    assert_true(mesh.triangle_count() == 2)
    assert_true(mesh.has_normals())
    assert_true(not mesh.has_texcoords())
    assert_true(len(mesh.indices) == 6)
    assert_true(mesh.indices[0] == UInt32(0))
    assert_true(mesh.indices[1] == UInt32(1))
    assert_true(mesh.indices[2] == UInt32(2))
    assert_true(mesh.indices[3] == UInt32(0))
    assert_true(mesh.indices[4] == UInt32(2))
    assert_true(mesh.indices[5] == UInt32(3))


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


def main() raises:
    test_binary_triangle_with_normals_and_uvs()
    test_quad_is_fan_triangulated_without_uvs()
    test_rejects_unsupported_format()
    test_rejects_out_of_range_face_index()
    test_rejects_truncated_payload()
