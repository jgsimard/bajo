"""Packet queues and host/device wavefront boundary records."""

from std.collections import Array
from std.math import ceildiv
from std.sys.intrinsics import llvm_intrinsic
from layout import Coord
from bajo.rt.path_values import PathContinuation, PathFloatField
from bajo.rt.path_store import store_path_values

from bajo.core.packet_layout import (
    PacketFloatTile,
    packet_float_tile,
    compact_packet_floats,
    packet_field_index,
    packet_float_field,
    copy_packet_float_lane,
    TilePacket,
    append_masked_tile,
    append_masked_lanes,
)
from bajo.core import Rayf32
from bajo.core.color import Color
from bajo.rt.shading_types import SurfaceHit
from bajo.rt.material_types import SURFACE_INDEX_MASK


comptime FRONT_FACE_BIT = UInt32(0x80000000)
comptime PATH_INDEX_MASK = UInt32(0x7FFFFFFF)


struct WavePath(Copyable, Writable):
    var path_id: UInt32
    var ray: Rayf32[.WORLD]
    var throughput: Color
    var medium_id: UInt32

    def __init__(
        out self,
        path_id: UInt32,
        ray: Rayf32[.WORLD],
        throughput: Color,
        medium_id: UInt32 = 0,
    ):
        self.path_id = path_id
        self.ray = ray
        self.throughput = throughput
        self.medium_id = medium_id


@fieldwise_init
struct WaveShade(Copyable, Writable):
    var path_idx: UInt32
    var hit: SurfaceHit[1]


struct PathPacket[length: SIMDLength](TilePacket):
    """A field/lane float tile with separately typed path metadata."""

    comptime WIDTH = Self.length
    comptime FLOAT_NAMES = ParameterList.of[
        StaticString("ox"),
        StaticString("oy"),
        StaticString("oz"),
        StaticString("t_min"),
        StaticString("dx"),
        StaticString("dy"),
        StaticString("dz"),
        StaticString("t_max"),
        StaticString("tx"),
        StaticString("ty"),
        StaticString("tz"),
        StaticString("bsdf_pdfs"),
    ]()
    comptime FLOAT_FIELDS = len(Self.FLOAT_NAMES)
    var floats: Array[SIMD[.float32, Self.length], Self.FLOAT_FIELDS]
    var path_ids: SIMD[.uint32, Self.length]
    var deltas: SIMD[.bool, Self.length]
    var medium_ids: SIMD[.uint32, Self.length]

    def __init__(out self):
        self.floats = Array[SIMD[.float32, Self.length], Self.FLOAT_FIELDS](
            fill=0.0
        )
        self.path_ids = 0
        self.deltas = SIMD[.bool, Self.length](fill=True)
        self.medium_ids = 0

    @always_inline
    def __init__(out self, values: PathContinuation[Self.length]):
        self.floats = Array[SIMD[.float32, Self.length], Self.FLOAT_FIELDS](
            uninitialized=True
        )
        self.path_ids = values.path_ids
        self.deltas = values.delta
        self.medium_ids = values.medium_ids
        store_path_values(self.tile(), values, 0)

    @always_inline
    def tile(
        ref self,
    ) -> PacketFloatTile[
        Self.FLOAT_FIELDS, Self.length, origin_of(self.floats)
    ]:
        return packet_float_tile[Self.FLOAT_FIELDS, Self.length](
            self.floats.unsafe_ptr()
        )

    @staticmethod
    def field_index[name: StaticString]() -> Int:
        return packet_field_index[name, *Self.FLOAT_NAMES]()

    @always_inline
    def __getattr__[
        name: __mlir_type.`!kgen.string`
    ](ref self, attr: StringLiteral[name]) -> ref[self.floats] SIMD[
        .float32, Self.length
    ]:
        return packet_float_field[
            Self.length,
            Self.FLOAT_FIELDS,
            StaticString(StringLiteral[name]()),
            *Self.FLOAT_NAMES,
        ](self.floats)

    @always_inline
    def copy_metadata_lane(
        mut self, source: Self, source_lane: Int, destination_lane: Int
    ):
        self.path_ids[destination_lane] = source.path_ids[source_lane]
        self.medium_ids[destination_lane] = source.medium_ids[source_lane]
        self.deltas[destination_lane] = source.deltas[source_lane]

    @always_inline
    def copy_lane(
        mut self, source: Self, source_lane: Int, destination_lane: Int
    ):
        self.copy_metadata_lane(source, source_lane, destination_lane)
        copy_packet_float_lane[Self.FLOAT_FIELDS, Self.length](
            source.tile(), self.tile(), source_lane, destination_lane
        )

    @always_inline
    def copy_float_lanes(
        mut self,
        source: Self,
        selected: SIMD[.bool, Self.WIDTH],
        destination_lane: Int,
    ):
        compact_packet_floats[Self.FLOAT_FIELDS, Self.length](
            source.tile(), self.tile(), selected, destination_lane
        )


struct PacketPathQueue[length: SIMDLength](Sized):
    var packets: List[PathPacket[Self.length]]
    var count: Int

    def __init__(out self, capacity: Int):
        self.packets = List[PathPacket[Self.length]](
            capacity=ceildiv(capacity, Int(Self.length))
        )
        self.count = 0

    @always_inline
    def __len__(self) -> Int:
        return self.count

    @always_inline
    def clear(mut self):
        """Reset the queue while retaining its packet storage capacity."""
        self.packets.clear()
        self.count = 0

    @always_inline
    def _append_packet_lane(
        mut self,
        packet: PathPacket[Self.length],
        source_lane: Int,
    ):
        var destination_lane = self.count % Self.length
        if destination_lane == 0:
            self.packets.append(PathPacket[Self.length]())
        ref destination = self.packets[self.count / Self.length]
        destination.copy_lane(packet, source_lane, destination_lane)
        self.count += 1

    @always_inline
    def append_packet(
        mut self, var packet: PathPacket[Self.length], lane_count: Int
    ):
        debug_assert["safe", _use_compiler_assume=True](
            lane_count >= 0 and lane_count <= Self.length,
            "invalid path packet lane count",
        )
        if lane_count == Self.length and self.count % Self.length == 0:
            self.packets.append(packet^)
            self.count += Self.length
            return

        for lane in range(lane_count):
            self._append_packet_lane(packet, lane)

    @always_inline
    def append_packet_masked(
        mut self,
        var packet: PathPacket[Self.length],
        mask: SIMD[.bool, Self.length],
        lane_count: Int,
    ):
        append_masked_tile(self.packets, self.count, packet, mask, lane_count)

    @always_inline
    def append_continuation(
        mut self,
        values: PathContinuation[Self.length],
        mask: SIMD[.bool, Self.length],
        lane_count: Int,
    ):
        @always_inline
        def make_packet() {values} -> PathPacket[Self.length]:
            return PathPacket[Self.length](values)

        @always_inline
        def copy_metadata(
            mut destination: PathPacket[Self.length],
            source_lane: Int,
            target_lane: Int,
        ) {values}:
            destination.path_ids[target_lane] = values.path_ids[source_lane]
            destination.medium_ids[target_lane] = values.medium_ids[source_lane]
            destination.deltas[target_lane] = values.delta[source_lane]

        @always_inline
        def copy_lane(
            mut destination: PathPacket[Self.length],
            source_lane: Int,
            target_lane: Int,
        ) {values, copy_metadata}:
            copy_metadata(destination, source_lane, target_lane)
            var tile = destination.tile()
            comptime for field in range(PathFloatField.VALUE_PLANES):
                tile[field, target_lane] = values.float_field[field]()[
                    source_lane
                ]

        @always_inline
        def copy_floats(
            mut destination: PathPacket[Self.length],
            selected: SIMD[.bool, Self.length],
            target_lane: Int,
        ) {values}:
            var tile = destination.tile()
            comptime for field in range(PathFloatField.VALUE_PLANES):
                llvm_intrinsic["llvm.masked.compressstore", NoneType](
                    values.float_field[field](),
                    tile.ptr.unsafe_offset(
                        Int(tile.layout(Coord(field, target_lane)))
                    ),
                    selected,
                )

        append_masked_lanes(
            self.packets,
            self.count,
            mask,
            lane_count,
            make_packet,
            copy_lane,
            copy_metadata,
            copy_floats,
        )


struct ShadePacket[length: SIMDLength](TilePacket):
    """A field/lane float tile with separately typed path metadata."""

    comptime WIDTH = Self.length
    comptime FLOAT_NAMES = ParameterList.of[
        StaticString("ox"),
        StaticString("oy"),
        StaticString("oz"),
        StaticString("dx"),
        StaticString("dy"),
        StaticString("dz"),
        StaticString("tx"),
        StaticString("ty"),
        StaticString("tz"),
        StaticString("nx"),
        StaticString("ny"),
        StaticString("nz"),
        StaticString("uv_u"),
        StaticString("uv_v"),
        StaticString("hit_t"),
        StaticString("gnx"),
        StaticString("gny"),
        StaticString("gnz"),
    ]()
    comptime FLOAT_FIELDS = len(Self.FLOAT_NAMES)
    var floats: Array[SIMD[.float32, Self.length], Self.FLOAT_FIELDS]
    var path_ids: SIMD[.uint32, Self.length]
    var surface_indices: SIMD[.uint32, Self.length]
    var front_faces: SIMD[.bool, Self.length]
    var medium_ids: SIMD[.uint32, Self.length]
    var medium_boundaries: SIMD[.uint32, Self.length]

    def __init__(out self):
        self.floats = Array[SIMD[.float32, Self.length], Self.FLOAT_FIELDS](
            fill=0.0
        )
        self.path_ids = 0
        self.surface_indices = 0
        self.front_faces = SIMD[.bool, Self.length](fill=False)
        self.medium_ids = 0
        self.medium_boundaries = 0

    @always_inline
    def tile(
        ref self,
    ) -> PacketFloatTile[
        Self.FLOAT_FIELDS, Self.length, origin_of(self.floats)
    ]:
        return packet_float_tile[Self.FLOAT_FIELDS, Self.length](
            self.floats.unsafe_ptr()
        )

    @staticmethod
    def field_index[name: StaticString]() -> Int:
        return packet_field_index[name, *Self.FLOAT_NAMES]()

    @always_inline
    def __getattr__[
        name: __mlir_type.`!kgen.string`
    ](ref self, attr: StringLiteral[name]) -> ref[self.floats] SIMD[
        .float32, Self.length
    ]:
        return packet_float_field[
            Self.length,
            Self.FLOAT_FIELDS,
            StaticString(StringLiteral[name]()),
            *Self.FLOAT_NAMES,
        ](self.floats)

    @always_inline
    def set_paths(mut self, paths: PathPacket[Self.length]):
        """Copy shared ray/throughput rows with SIMD loads and stores."""
        self.path_ids = paths.path_ids
        self.medium_ids = paths.medium_ids
        var source = paths.tile()
        var destination = self.tile()
        comptime for name in [
            "ox",
            "oy",
            "oz",
            "dx",
            "dy",
            "dz",
            "tx",
            "ty",
            "tz",
        ]:
            comptime source_field = PathPacket[Self.length].field_index[
                StaticString(name)
            ]()
            comptime destination_field = Self.field_index[StaticString(name)]()
            destination.store[width=Self.length](
                (destination_field, 0),
                source.load[width=Self.length]((source_field, 0)),
            )

    @always_inline
    def set_hit(mut self, lane: Int, hit: SurfaceHit[1]):
        self.medium_boundaries[lane] = hit.medium_boundary
        self.gnx[lane] = hit.geometric_normal.x
        self.gny[lane] = hit.geometric_normal.y
        self.gnz[lane] = hit.geometric_normal.z
        self.nx[lane] = hit.normal.x
        self.ny[lane] = hit.normal.y
        self.nz[lane] = hit.normal.z
        self.uv_u[lane] = hit.uv_u
        self.uv_v[lane] = hit.uv_v
        self.hit_t[lane] = hit.t
        self.surface_indices[lane] = hit.surface.index()
        self.front_faces[lane] = hit.front_face

    @always_inline
    def set_hits(mut self, hits: SurfaceHit[Self.length]):
        self.medium_boundaries = hits.medium_boundary
        self.gnx = hits.geometric_normal.x
        self.gny = hits.geometric_normal.y
        self.gnz = hits.geometric_normal.z
        self.nx = hits.normal.x
        self.ny = hits.normal.y
        self.nz = hits.normal.z
        self.uv_u = hits.uv_u
        self.uv_v = hits.uv_v
        self.hit_t = hits.t
        self.surface_indices = hits.surface.value & SURFACE_INDEX_MASK
        self.front_faces = hits.front_face

    @always_inline
    def copy_metadata_lane(
        mut self, source: Self, source_lane: Int, destination_lane: Int
    ):
        self.path_ids[destination_lane] = source.path_ids[source_lane]
        self.medium_ids[destination_lane] = source.medium_ids[source_lane]
        self.medium_boundaries[destination_lane] = source.medium_boundaries[
            source_lane
        ]
        self.surface_indices[destination_lane] = source.surface_indices[
            source_lane
        ]
        self.front_faces[destination_lane] = source.front_faces[source_lane]

    @always_inline
    def copy_lane(
        mut self, source: Self, source_lane: Int, destination_lane: Int
    ):
        self.copy_metadata_lane(source, source_lane, destination_lane)
        copy_packet_float_lane[Self.FLOAT_FIELDS, Self.length](
            source.tile(), self.tile(), source_lane, destination_lane
        )

    @always_inline
    def copy_float_lanes(
        mut self,
        source: Self,
        selected: SIMD[.bool, Self.WIDTH],
        destination_lane: Int,
    ):
        compact_packet_floats[Self.FLOAT_FIELDS, Self.length](
            source.tile(), self.tile(), selected, destination_lane
        )


struct PacketShadeQueue[length: SIMDLength](Sized):
    var packets: List[ShadePacket[Self.length]]
    var count: Int

    def __init__(out self, capacity: Int):
        self.packets = List[ShadePacket[Self.length]](
            capacity=ceildiv(capacity, Int(Self.length))
        )
        self.count = 0

    @always_inline
    def __len__(self) -> Int:
        return self.count

    @always_inline
    def clear(mut self):
        self.packets.clear()
        self.count = 0

    @always_inline
    def append(
        mut self,
        path_packet: PathPacket[Self.length],
        path_lane: Int,
        hit: SurfaceHit[1],
    ):
        var shade_lane = self.count % Self.length
        if shade_lane == 0:
            self.packets.append(ShadePacket[Self.length]())
        ref shade = self.packets[self.count / Self.length]
        shade.path_ids[shade_lane] = path_packet.path_ids[path_lane]
        shade.medium_ids[shade_lane] = path_packet.medium_ids[path_lane]
        var source = path_packet.tile()
        var destination = shade.tile()
        comptime for name in [
            "ox",
            "oy",
            "oz",
            "dx",
            "dy",
            "dz",
            "tx",
            "ty",
            "tz",
        ]:
            comptime source_field = PathPacket[Self.length].field_index[
                StaticString(name)
            ]()
            comptime destination_field = ShadePacket[Self.length].field_index[
                StaticString(name)
            ]()
            destination[destination_field, shade_lane] = source[
                source_field, path_lane
            ]
        shade.set_hit(shade_lane, hit)
        self.count += 1

    @always_inline
    def append_packet_masked(
        mut self,
        packet: ShadePacket[Self.length],
        mask: SIMD[.bool, Self.length],
        lane_count: Int,
    ):
        append_masked_tile(self.packets, self.count, packet, mask, lane_count)
