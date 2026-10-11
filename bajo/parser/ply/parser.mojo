"""Parser for binary little-endian PLY polygon meshes."""

from max.algorithm import parallelize
from std.math import ceildiv
from std.memory import bitcast
from std.sys import num_logical_cores

from bajo.parser.ply.types import PlyMesh


@fieldwise_init
struct _ScalarType(EnumLike, Equatable, TrivialRegisterPassable):
    var value: UInt8

    comptime I8 = Self(0)
    comptime U8 = Self(1)
    comptime I16 = Self(2)
    comptime U16 = Self(3)
    comptime I32 = Self(4)
    comptime U32 = Self(5)
    comptime F32 = Self(6)
    comptime F64 = Self(7)
    comptime _enum_case_names = ParameterList.of[
        "I8".value,
        "U8".value,
        "I16".value,
        "U16".value,
        "I32".value,
        "U32".value,
        "F32".value,
        "F64".value,
    ].values
    comptime _enum_case_types = TypeList.of[
        Trait=AnyType,
        NoneType,
        NoneType,
        NoneType,
        NoneType,
        NoneType,
        NoneType,
        NoneType,
        NoneType,
    ].values

    def _get_enum_discriminant(self) -> Int:
        return Int(self.value)

    def _unsafe_get_enum_payload[
        id: Int
    ](ref self) -> ref[self] TypeList[Trait=AnyType, Self._enum_case_types]()[
        id
    ]:
        while True:
            pass

    def byte_width(self) -> Int:
        __match self:
            case .I8 | .U8:
                return 1
            case .I16 | .U16:
                return 2
            case .I32 | .U32 | .F32:
                return 4
            case .F64:
                return 8

    def is_integer(self) -> Bool:
        __match self:
            case .I8 | .U8 | .I16 | .U16 | .I32 | .U32:
                return True
            case .F32 | .F64:
                return False


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

    def next(mut self) raises -> ImmStringSpan[Self.origin]:
        self._skip_space()
        if self.pos >= len(self.bytes):
            raise Error("unexpected end of PLY header")
        var start = self.pos
        while self.pos < len(self.bytes) and not Self._is_space(
            self.bytes.unsafe_get(self.pos)
        ):
            self.pos += 1
        return {unsafe_from_utf8 = self.bytes[start : self.pos]}

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
        __match kind:
            case .I8:
                return Int(bitcast[.int8](self.read_u8()))
            case .U8:
                return Int(self.read_u8())
            case .I16:
                return Int(bitcast[.int16](self.read_u16()))
            case .U16:
                return Int(self.read_u16())
            case .I32:
                return Int(bitcast[.int32](self.read_u32()))
            case .U32:
                return Int(self.read_u32())
            case .F32 | .F64:
                raise Error("PLY list sizes and indices must use integer types")

    def read_number(mut self, kind: _ScalarType) raises -> Float32:
        __match kind:
            case .F32:
                return bitcast[.float32](self.read_u32())
            case .F64:
                return Float32(bitcast[.float64](self.read_u64()))
            case .I8 | .U8 | .I16 | .U16 | .I32 | .U32:
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
    __match name:
        case "char" | "int8":
            return .I8

        case "uchar" | "uint8":
            return .U8

        case "short" | "int16":
            return .I16

        case "ushort" | "uint16":
            return .U16

        case "int" | "int32":
            return .I32

        case "uint" | "uint32":
            return .U32

        case "float" | "float32":
            return .F32

        case "double" | "float64":
            return .F64

    raise Error("unsupported PLY scalar type: " + String(name))


def _vertex_semantic(name: ImmStringSpan) -> UInt8:
    __match name:
        case "x":
            return _X

        case "y":
            return _Y

        case "z":
            return _Z

        case "nx":
            return _NX

        case "ny":
            return _NY

        case "nz":
            return _NZ

        case "u" | "s" | "texture_u":
            return _U

        case "v" | "t" | "texture_v":
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
        __match command:
            case "comment" | "obj_info":
                cursor.skip_line()
            case "element":
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
            case "property":
                if element == UInt8(0):
                    raise Error("PLY property appears before an element")
                var prop = _parse_property(cursor, element)
                if element == UInt8(1):
                    header.vertex_properties.append(prop)
                else:
                    header.face_properties.append(prop)
            case "end_header":
                header.data_offset = cursor.finish_header_line()
                _validate_header(header, len(bytes) - header.data_offset)
                return header^
            case _:
                raise Error(
                    "unsupported PLY header directive: " + String(command)
                )


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


def _parse_ply_serial(bytes: ImmSpan[UInt8, _]) raises -> PlyMesh:
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


def _decode_vertex_property(
    bytes: ImmSpan[UInt8, _],
    output: MutSpan[Float32, _],
    kind: _ScalarType,
    input_offset: Int,
    input_stride: Int,
    output_stride: Int,
    lane: Int,
    begin: Int,
    end: Int,
) raises:
    # The complete fixed-stride vertex payload was checked by the header.
    # Dispatch once per property, outside the vertex loop. Loads may be
    # unaligned because a preceding property can be a single byte.
    if kind == .F32:
        for vertex in range(begin, end):
            var ptr = bytes.unsafe_ptr().unsafe_offset(
                input_offset + vertex * input_stride
            )
            output[vertex * output_stride + lane] = bitcast[.float32](
                ptr.unsafe_bitcast[UInt32]().unsafe_load[alignment=1]()
            )
    elif kind == .F64:
        for vertex in range(begin, end):
            var ptr = bytes.unsafe_ptr().unsafe_offset(
                input_offset + vertex * input_stride
            )
            output[vertex * output_stride + lane] = Float32(
                bitcast[.float64](
                    ptr.unsafe_bitcast[UInt64]().unsafe_load[alignment=1]()
                )
            )
    else:
        var cursor = _BinaryCursor(bytes, input_offset)
        for vertex in range(begin, end):
            cursor.pos = input_offset + vertex * input_stride
            output[vertex * output_stride + lane] = cursor.read_number(kind)


def _decode_vertices(
    bytes: ImmSpan[UInt8, _], header: _Header, mut mesh: PlyMesh, workers: Int
) raises:
    var stride = 0
    for prop in header.vertex_properties:
        stride += prop.scalar_type.byte_width()
    var tasks = min(
        max(1, workers) * 4, max(1, ceildiv(header.vertex_count, 16 * 1024))
    )
    var failed = List[UInt8](length=tasks, fill=0)

    def decode(task: Int) {imm, mut mesh, mut failed}:
        var begin = header.vertex_count * task // tasks
        var end = header.vertex_count * (task + 1) // tasks
        var offset = header.data_offset
        try:
            for prop in header.vertex_properties:
                if prop.semantic >= _X and prop.semantic <= _Z:
                    _decode_vertex_property(
                        bytes,
                        Span(mesh.positions),
                        prop.scalar_type,
                        offset,
                        stride,
                        3,
                        Int(prop.semantic - _X),
                        begin,
                        end,
                    )
                elif prop.semantic >= _NX and prop.semantic <= _NZ:
                    _decode_vertex_property(
                        bytes,
                        Span(mesh.normals),
                        prop.scalar_type,
                        offset,
                        stride,
                        3,
                        Int(prop.semantic - _NX),
                        begin,
                        end,
                    )
                elif prop.semantic == _U or prop.semantic == _V:
                    _decode_vertex_property(
                        bytes,
                        Span(mesh.texcoords),
                        prop.scalar_type,
                        offset,
                        stride,
                        2,
                        Int(prop.semantic - _U),
                        begin,
                        end,
                    )
                offset += prop.scalar_type.byte_width()
        except:
            failed[task] = 1

    if workers > 1 and tasks > 1:
        parallelize(decode, tasks, min(workers, tasks))
    else:
        for task in range(tasks):
            decode(task)
    for failure in failed:
        if failure:
            raise Error("failed to decode PLY vertices")


@fieldwise_init
struct _FaceChunk(TrivialRegisterPassable):
    var begin: Int
    var end: Int
    var face_count: Int
    var output_offset: Int
    var output_count: Int


def _face_chunks(
    bytes: ImmSpan[UInt8, _], header: _Header
) raises -> List[_FaceChunk]:
    var stride = 0
    for prop in header.vertex_properties:
        stride += prop.scalar_type.byte_width()
    var cursor = _BinaryCursor(
        bytes, header.data_offset + header.vertex_count * stride
    )
    var min_face_bytes = 0
    for prop in header.face_properties:
        min_face_bytes += prop.scalar_type.byte_width()
        if prop.semantic == _VERTEX_INDICES:
            min_face_bytes += 3 * prop.list_item_type.byte_width()
    if header.face_count > 0 and (
        min_face_bytes == 0
        or header.face_count > (len(bytes) - cursor.pos) // min_face_bytes
    ):
        raise Error("PLY face payload is smaller than its header declares")
    var chunks = List[_FaceChunk](capacity=ceildiv(header.face_count, 4096))
    var total = 0
    for begin in range(0, header.face_count, 4096):
        var start = cursor.pos
        var output_count = 0
        var end = min(begin + 4096, header.face_count)
        for _ in range(begin, end):
            var polygon_count = 0
            for prop in header.face_properties:
                if prop.is_list:
                    var count = cursor.read_integer(prop.scalar_type)
                    if count < 0 or count > header.vertex_count:
                        raise Error("invalid PLY face list size")
                    if count > 0 and not prop.list_item_type.is_integer():
                        raise Error(
                            "PLY list sizes and indices must use integer types"
                        )
                    var size = count * prop.list_item_type.byte_width()
                    cursor._require(size)
                    cursor.pos += size
                    if prop.semantic == _VERTEX_INDICES:
                        polygon_count = count
                else:
                    var size = prop.scalar_type.byte_width()
                    cursor._require(size)
                    cursor.pos += size
            if polygon_count < 3:
                raise Error("PLY faces require at least three vertices")
            output_count += 3 * (polygon_count - 2)
        chunks.append(
            _FaceChunk(start, cursor.pos, end - begin, total, output_count)
        )
        total += output_count
    return chunks^


def _emit_faces(
    bytes: ImmSpan[UInt8, _],
    header: _Header,
    chunk: _FaceChunk,
    output: MutSpan[UInt32, _],
) raises:
    # Limit each cursor and output span to its chunk, including malformed input.
    var cursor = _BinaryCursor(bytes[chunk.begin : chunk.end], 0)
    var write = 0
    for _ in range(chunk.face_count):
        for prop in header.face_properties:
            if not prop.is_list:
                cursor.pos += prop.scalar_type.byte_width()
                continue
            var count = cursor.read_integer(prop.scalar_type)
            if prop.semantic != _VERTEX_INDICES:
                cursor.pos += count * prop.list_item_type.byte_width()
                continue
            var first = UInt32(0)
            var previous = UInt32(0)
            for corner in range(count):
                var value = cursor.read_integer(prop.list_item_type)
                if value < 0 or value >= header.vertex_count:
                    raise Error("PLY face index is out of range")
                var index = UInt32(value)
                if corner == 0:
                    first = index
                elif corner == 1:
                    previous = index
                else:
                    output[write] = first
                    output[write + 1] = previous
                    output[write + 2] = index
                    previous = index
                    write += 3


def _parse_ply_bulk(bytes: ImmSpan[UInt8, _], workers: Int) raises -> PlyMesh:
    var header = _parse_header(bytes)
    var chunks = _face_chunks(bytes, header)
    var mesh = PlyMesh()
    mesh.positions.resize(unsafe_uninit_length=header.vertex_count * 3)
    if header.has_normals:
        mesh.normals.resize(unsafe_uninit_length=header.vertex_count * 3)
    if header.has_texcoords:
        mesh.texcoords.resize(unsafe_uninit_length=header.vertex_count * 2)
    _decode_vertices(bytes, header, mesh, workers)
    if len(chunks):
        ref last = chunks[len(chunks) - 1]
        mesh.indices.resize(
            unsafe_uninit_length=last.output_offset + last.output_count
        )
    var failed = List[UInt8](length=len(chunks), fill=0)

    def emit(task: Int) {imm, mut mesh, mut failed}:
        ref chunk = chunks[task]
        try:
            _emit_faces(
                bytes,
                header,
                chunk,
                Span(mesh.indices)[
                    chunk.output_offset : chunk.output_offset
                    + chunk.output_count
                ],
            )
        except:
            failed[task] = 1

    if workers > 1 and len(chunks) > 1:
        parallelize(emit, len(chunks), min(workers, len(chunks)))
    else:
        for task in range(len(chunks)):
            emit(task)
    for failure in failed:
        if failure:
            raise Error("failed to decode PLY faces")
    mesh._source_face_count = header.face_count
    return mesh^


def _parse_ply(bytes: ImmSpan[UInt8, _]) raises -> PlyMesh:
    if len(bytes) < 256 * 1024:
        return _parse_ply_serial(bytes)
    try:
        return _parse_ply_bulk(bytes, min(8, num_logical_cores()))
    except:
        # Preserve the original diagnostics and validation order on invalid input.
        return _parse_ply_serial(bytes)
