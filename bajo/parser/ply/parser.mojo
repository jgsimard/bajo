"""Parser for binary little-endian PLY polygon meshes."""

from std.memory import bitcast

from bajo.parser.ply.types import PlyMesh


@fieldwise_init
struct _ScalarType(Equatable, TrivialRegisterPassable):
    var value: UInt8

    comptime I8 = Self(0)
    comptime U8 = Self(1)
    comptime I16 = Self(2)
    comptime U16 = Self(3)
    comptime I32 = Self(4)
    comptime U32 = Self(5)
    comptime F32 = Self(6)
    comptime F64 = Self(7)

    def byte_width(self) -> Int:
        if self == .I8 or self == .U8:
            return 1
        if self == .I16 or self == .U16:
            return 2
        if self == .I32 or self == .U32 or self == .F32:
            return 4
        return 8

    def is_integer(self) -> Bool:
        return self != .F32 and self != .F64


@fieldwise_init
struct _Property(TrivialRegisterPassable):
    var scalar_type: _ScalarType
    var list_item_type: _ScalarType
    var semantic: UInt8
    var is_list: Bool


comptime _IGNORE = UInt8(0)
comptime _X = UInt8(1)
comptime _Y = UInt8(2)
comptime _Z = UInt8(3)
comptime _NX = UInt8(4)
comptime _NY = UInt8(5)
comptime _NZ = UInt8(6)
comptime _U = UInt8(7)
comptime _V = UInt8(8)
comptime _VERTEX_INDICES = UInt8(9)


struct _Header:
    var vertex_count: Int
    var face_count: Int
    var vertex_properties: List[_Property]
    var face_properties: List[_Property]
    var data_offset: Int
    var has_normals: Bool
    var has_texcoords: Bool

    def __init__(out self):
        self.vertex_count = -1
        self.face_count = -1
        self.vertex_properties = List[_Property]()
        self.face_properties = List[_Property]()
        self.data_offset = 0
        self.has_normals = False
        self.has_texcoords = False


struct _HeaderCursor[origin: ImmOrigin]:
    var bytes: ImmSpan[UInt8, Self.origin]
    var pos: Int
    var line: Int

    def __init__(out self, bytes: ImmSpan[UInt8, Self.origin]):
        self.bytes = bytes
        self.pos = 0
        self.line = 1

    @staticmethod
    def _is_space(c: UInt8) -> Bool:
        return (
            c == UInt8(9) or c == UInt8(10) or c == UInt8(13) or c == UInt8(32)
        )

    def _skip_space(mut self):
        while self.pos < len(self.bytes) and Self._is_space(
            self.bytes.unsafe_get(self.pos)
        ):
            if self.bytes.unsafe_get(self.pos) == UInt8(10):
                self.line += 1
            self.pos += 1

    def next(mut self) raises -> StringSpan[Self.origin]:
        self._skip_space()
        if self.pos >= len(self.bytes):
            raise Error("unexpected end of PLY header")
        var start = self.pos
        while self.pos < len(self.bytes) and not Self._is_space(
            self.bytes.unsafe_get(self.pos)
        ):
            self.pos += 1
        return StringSpan[Self.origin](
            unsafe_from_utf8=self.bytes[start : self.pos]
        )

    def skip_line(mut self):
        while self.pos < len(self.bytes):
            var c = self.bytes.unsafe_get(self.pos)
            self.pos += 1
            if c == UInt8(10):
                self.line += 1
                return

    def finish_header_line(mut self) raises -> Int:
        while self.pos < len(self.bytes):
            var c = self.bytes.unsafe_get(self.pos)
            self.pos += 1
            if c == UInt8(10):
                self.line += 1
                return self.pos
            if c != UInt8(9) and c != UInt8(13) and c != UInt8(32):
                raise Error("unexpected token after PLY end_header")
        raise Error("PLY end_header must end with a newline")


struct _BinaryCursor[origin: ImmOrigin]:
    var bytes: ImmSpan[UInt8, Self.origin]
    var pos: Int

    def __init__(
        out self, bytes: ImmSpan[UInt8, Self.origin], data_offset: Int
    ):
        self.bytes = bytes
        self.pos = data_offset

    def _require(self, count: Int) raises:
        if count < 0 or self.pos > len(self.bytes) - count:
            raise Error("truncated PLY binary payload")

    def read_u8(mut self) raises -> UInt8:
        self._require(1)
        var value = self.bytes.unsafe_get(self.pos)
        self.pos += 1
        return value

    def read_u16(mut self) raises -> UInt16:
        self._require(2)
        var value = UInt16(self.bytes.unsafe_get(self.pos)) | (
            UInt16(self.bytes.unsafe_get(self.pos + 1)) << UInt16(8)
        )
        self.pos += 2
        return value

    def read_u32(mut self) raises -> UInt32:
        self._require(4)
        var value = UInt32(self.bytes.unsafe_get(self.pos))
        value |= UInt32(self.bytes.unsafe_get(self.pos + 1)) << UInt32(8)
        value |= UInt32(self.bytes.unsafe_get(self.pos + 2)) << UInt32(16)
        value |= UInt32(self.bytes.unsafe_get(self.pos + 3)) << UInt32(24)
        self.pos += 4
        return value

    def read_u64(mut self) raises -> UInt64:
        self._require(8)
        var value = UInt64(0)
        for i in range(8):
            value |= UInt64(self.bytes.unsafe_get(self.pos + i)) << UInt64(
                8 * i
            )
        self.pos += 8
        return value

    def read_integer(mut self, kind: _ScalarType) raises -> Int:
        if kind == .I8:
            return Int(bitcast[.int8](self.read_u8()))
        if kind == .U8:
            return Int(self.read_u8())
        if kind == .I16:
            return Int(bitcast[.int16](self.read_u16()))
        if kind == .U16:
            return Int(self.read_u16())
        if kind == .I32:
            return Int(bitcast[.int32](self.read_u32()))
        if kind == .U32:
            return Int(self.read_u32())
        raise Error("PLY list sizes and indices must use integer types")

    def read_number(mut self, kind: _ScalarType) raises -> Float32:
        if kind == .F32:
            return bitcast[.float32](self.read_u32())
        if kind == .F64:
            return Float32(bitcast[.float64](self.read_u64()))
        return Float32(self.read_integer(kind))


def _parse_nonnegative_int(text: ImmStringSpan) raises -> Int:
    var bytes = text.as_bytes()
    if len(bytes) == 0 or len(bytes) > 18:
        raise Error("invalid PLY element count: " + String(text))
    var value = Int(0)
    for c in bytes:
        if c < UInt8(48) or c > UInt8(57):
            raise Error("invalid PLY element count: " + String(text))
        value = value * 10 + Int(c - UInt8(48))
    return value


def _parse_scalar_type(name: ImmStringSpan) raises -> _ScalarType:
    if name == "char" or name == "int8":
        return .I8
    if name == "uchar" or name == "uint8":
        return .U8
    if name == "short" or name == "int16":
        return .I16
    if name == "ushort" or name == "uint16":
        return .U16
    if name == "int" or name == "int32":
        return .I32
    if name == "uint" or name == "uint32":
        return .U32
    if name == "float" or name == "float32":
        return .F32
    if name == "double" or name == "float64":
        return .F64
    raise Error("unsupported PLY scalar type: " + String(name))


def _vertex_semantic(name: ImmStringSpan) -> UInt8:
    if name == "x":
        return _X
    if name == "y":
        return _Y
    if name == "z":
        return _Z
    if name == "nx":
        return _NX
    if name == "ny":
        return _NY
    if name == "nz":
        return _NZ
    if name == "u" or name == "s" or name == "texture_u":
        return _U
    if name == "v" or name == "t" or name == "texture_v":
        return _V
    return _IGNORE


def _parse_property(
    mut cursor: _HeaderCursor, element: UInt8
) raises -> _Property:
    var first = cursor.next()
    var is_list = first == "list"
    var scalar_type: _ScalarType
    var list_item_type = _ScalarType.U8
    var name: StringSpan[cursor.origin]
    if is_list:
        scalar_type = _parse_scalar_type(cursor.next())
        list_item_type = _parse_scalar_type(cursor.next())
        name = cursor.next()
    else:
        scalar_type = _parse_scalar_type(first)
        name = cursor.next()

    var semantic = _IGNORE
    if element == UInt8(1):
        if is_list:
            raise Error("PLY vertex list properties are not supported")
        semantic = _vertex_semantic(name)
    elif name == "vertex_indices" or name == "vertex_index":
        if not is_list:
            raise Error("PLY vertex_indices must be a list property")
        semantic = _VERTEX_INDICES
    return _Property(scalar_type, list_item_type, semantic, is_list)


def _validate_header(mut header: _Header, payload_bytes: Int) raises:
    if header.vertex_count < 0:
        raise Error("PLY header is missing the vertex element")
    if header.face_count < 0:
        raise Error("PLY header is missing the face element")

    var seen = SIMD[.bool, 16](fill=False)
    var vertex_stride = 0
    for prop in header.vertex_properties:
        vertex_stride += prop.scalar_type.byte_width()
        if prop.semantic != _IGNORE:
            if seen[Int(prop.semantic)]:
                raise Error("duplicate PLY vertex property")
            seen[Int(prop.semantic)] = True
    if not seen[Int(_X)] or not seen[Int(_Y)] or not seen[Int(_Z)]:
        raise Error("PLY vertices require x, y, and z properties")
    var any_normal = seen[Int(_NX)] or seen[Int(_NY)] or seen[Int(_NZ)]
    header.has_normals = seen[Int(_NX)] and seen[Int(_NY)] and seen[Int(_NZ)]
    if any_normal and not header.has_normals:
        raise Error("PLY vertex normals require nx, ny, and nz")
    var any_uv = seen[Int(_U)] or seen[Int(_V)]
    header.has_texcoords = seen[Int(_U)] and seen[Int(_V)]
    if any_uv and not header.has_texcoords:
        raise Error("PLY texture coordinates require both u and v")

    var has_indices = False
    for prop in header.face_properties:
        if prop.semantic == _VERTEX_INDICES:
            if has_indices:
                raise Error("duplicate PLY vertex_indices property")
            if (
                not prop.scalar_type.is_integer()
                or not prop.list_item_type.is_integer()
            ):
                raise Error("PLY vertex_indices must use integer types")
            has_indices = True
    if not has_indices and header.face_count > 0:
        raise Error("PLY faces require a vertex_indices list property")

    if vertex_stride == 0 or (
        header.vertex_count > 0
        and header.vertex_count > payload_bytes / vertex_stride
    ):
        raise Error("PLY vertex payload is smaller than its header declares")


def _parse_header(bytes: ImmSpan[UInt8, _]) raises -> _Header:
    var cursor = _HeaderCursor(bytes)
    if cursor.next() != StringSpan("ply"):
        raise Error("PLY file must begin with 'ply'")
    if cursor.next() != StringSpan("format"):
        raise Error("PLY header is missing its format declaration")
    var format = cursor.next()
    if format != StringSpan("binary_little_endian"):
        raise Error("only binary_little_endian PLY files are supported")
    if cursor.next() != StringSpan("1.0"):
        raise Error("only PLY format version 1.0 is supported")

    var header = _Header()
    var element = UInt8(0)
    var saw_face = False
    while True:
        var command = cursor.next()
        if command == "comment" or command == "obj_info":
            cursor.skip_line()
        elif command == "element":
            var name = cursor.next()
            var count = _parse_nonnegative_int(cursor.next())
            if name == "vertex":
                if header.vertex_count >= 0 or saw_face:
                    raise Error(
                        "PLY vertex element must appear once before faces"
                    )
                header.vertex_count = count
                element = UInt8(1)
            elif name == "face":
                if header.face_count >= 0:
                    raise Error("PLY face element appears more than once")
                header.face_count = count
                element = UInt8(2)
                saw_face = True
            else:
                raise Error("unsupported PLY element: " + String(name))
        elif command == "property":
            if element == UInt8(0):
                raise Error("PLY property appears before an element")
            var prop = _parse_property(cursor, element)
            if element == UInt8(1):
                header.vertex_properties.append(prop)
            else:
                header.face_properties.append(prop)
        elif command == "end_header":
            header.data_offset = cursor.finish_header_line()
            _validate_header(header, len(bytes) - header.data_offset)
            return header^
        else:
            raise Error("unsupported PLY header directive: " + String(command))


def _store_vertex_value(
    mut mesh: PlyMesh,
    vertex: Int,
    semantic: UInt8,
    value: Float32,
):
    if semantic >= _X and semantic <= _Z:
        mesh.positions[3 * vertex + Int(semantic - _X)] = value
    elif semantic >= _NX and semantic <= _NZ:
        mesh.normals[3 * vertex + Int(semantic - _NX)] = value
    elif semantic == _U or semantic == _V:
        mesh.texcoords[2 * vertex + Int(semantic - _U)] = value


def _parse_ply(bytes: ImmSpan[UInt8, _]) raises -> PlyMesh:
    var header = _parse_header(bytes)
    var cursor = _BinaryCursor(bytes, header.data_offset)
    var mesh = PlyMesh()
    mesh.positions = List[Float32](length=header.vertex_count * 3, fill=0.0)
    if header.has_normals:
        mesh.normals = List[Float32](length=header.vertex_count * 3, fill=0.0)
    if header.has_texcoords:
        mesh.texcoords = List[Float32](length=header.vertex_count * 2, fill=0.0)

    for vertex in range(header.vertex_count):
        for prop in header.vertex_properties:
            var value = cursor.read_number(prop.scalar_type)
            if prop.semantic != _IGNORE:
                _store_vertex_value(mesh, vertex, prop.semantic, value)

    var polygon = List[UInt32]()
    for _ in range(header.face_count):
        polygon.clear()
        for prop in header.face_properties:
            if prop.is_list:
                var count = cursor.read_integer(prop.scalar_type)
                if count < 0 or count > header.vertex_count:
                    raise Error("invalid PLY face list size")
                for _ in range(count):
                    var value = cursor.read_integer(prop.list_item_type)
                    if prop.semantic == _VERTEX_INDICES:
                        if value < 0 or value >= header.vertex_count:
                            raise Error("PLY face index is out of range")
                        polygon.append(UInt32(value))
            else:
                _ = cursor.read_number(prop.scalar_type)
        if len(polygon) < 3:
            raise Error("PLY faces require at least three vertices")
        for i in range(1, len(polygon) - 1):
            mesh.indices.append(polygon[0])
            mesh.indices.append(polygon[i])
            mesh.indices.append(polygon[i + 1])

    mesh._source_face_count = header.face_count
    return mesh^
