"""Borrowed field/lane TileTensor views of homogeneous CPU SIMD packets."""

from std.sys.intrinsics import llvm_intrinsic
from std.bit import count_trailing_zeros
from std.memory import pack_bits
from layout import Coord, Idx
from bajo.core.tensor_layout import FieldLayout, FieldTile, field_tile


comptime PacketFloatLayout[fields: Int, length: SIMDLength] = FieldLayout[
    fields, type_of(Idx[Int(length)])
]
comptime PacketFloatTile[
    fields: Int, length: SIMDLength, origin: Origin
] = FieldTile[.float32, fields, type_of(Idx[Int(length)]), origin]


@always_inline
def packet_float_tile[
    mut: Bool, origin: Origin[mut=mut], //, fields: Int, length: SIMDLength
](storage: Pointer[SIMD[.float32, length], origin]) -> PacketFloatTile[
    fields, length, origin
]:
    return field_tile[.float32, fields](
        storage.unsafe_bitcast[Float32](), Idx[Int(length)]
    )


@always_inline
def packet_field_index[name: StaticString, *names: StaticString]() -> Int:
    """Resolve a named float row at compile time for either packet layout."""
    comptime for field in range(len(names)):
        comptime if name == names[field]:
            return field
    return -1


@always_inline
def packet_float_field[
    length: SIMDLength, fields: Int, name: StaticString, *names: StaticString
](ref rows: Array[SIMD[.float32, length], fields]) -> ref[rows] SIMD[
    .float32, length
]:
    comptime field = packet_field_index[name, *names]()
    comptime assert field >= 0, "unknown float packet field"
    return rows[field]


@always_inline
def copy_packet_float_lane[
    source_mut: Bool,
    source_origin: Origin[mut=source_mut],
    destination_origin: Origin[mut=True],
    //,
    fields: Int,
    length: SIMDLength,
](
    source: PacketFloatTile[fields, length, source_origin],
    destination: PacketFloatTile[fields, length, destination_origin],
    source_lane: Int,
    destination_lane: Int,
):
    comptime for field in range(fields):
        destination[field, destination_lane] = source[field, source_lane]


@always_inline
def compact_packet_floats[
    source_mut: Bool,
    source_origin: Origin[mut=source_mut],
    destination_origin: Origin[mut=True],
    //,
    fields: Int,
    length: SIMDLength,
](
    source: PacketFloatTile[fields, length, source_origin],
    destination: PacketFloatTile[fields, length, destination_origin],
    selected: SIMD[.bool, length],
    destination_lane: Int,
):
    """Copy selected lanes in order into a contiguous destination row suffix.

    The caller limits the selected lane count to the space left in this packet.
    LLVM supplies a portable lowering; AVX-512 targets use compressed stores.
    """
    comptime assert length > 1
    comptime for field in range(fields):
        var values = source.load[width=length]((field, 0))
        llvm_intrinsic["llvm.masked.compressstore", NoneType](
            values,
            destination.ptr.unsafe_offset(
                Int(destination.layout(Coord(field, destination_lane)))
            ),
            selected,
        )


trait TilePacket(Copyable):
    """Typed metadata operations needed by the shared packet compactor."""

    comptime WIDTH: SIMDLength

    def __init__(out self):
        ...

    def copy_metadata_lane(
        mut self, source: Self, source_lane: Int, destination_lane: Int
    ):
        ...

    def copy_lane(
        mut self, source: Self, source_lane: Int, destination_lane: Int
    ):
        ...

    def copy_float_lanes(
        mut self,
        source: Self,
        selected: SIMD[.bool, Self.WIDTH],
        destination_lane: Int,
    ):
        ...


@always_inline
def select_packet_lanes[
    length: SIMDLength, CopyLaneFn: def(Int, Int)
](
    mut remaining: UInt64, destination_lane: Int, ref copy_lane: CopyLaneFn
) -> Tuple[SIMD[.bool, length], Int]:
    """Consume the next stable mask fragment that fits in a destination packet.
    """
    var selected = SIMD[.bool, length](fill=False)
    var appended = 0
    while remaining != 0 and destination_lane + appended < length:
        var source_lane = Int(count_trailing_zeros(remaining))
        remaining &= remaining - 1
        selected[source_lane] = True
        copy_lane(source_lane, destination_lane + appended)
        appended += 1
    return (selected, appended)


@always_inline
def append_masked_tile[
    Packet: TilePacket
](
    mut packets: List[Packet],
    mut count: Int,
    source: Packet,
    mask: SIMD[.bool, Packet.WIDTH],
    lane_count: Int,
):
    """Stable compaction shared by path and shade queues, including tails."""
    debug_assert["safe", _use_compiler_assume=True](
        lane_count >= 0 and lane_count <= Packet.WIDTH,
        "invalid masked packet lane count",
    )
    comptime if Packet.WIDTH == 1 or Packet.WIDTH > 64:
        for lane in range(lane_count):
            if mask[lane]:
                var destination_lane = count % Packet.WIDTH
                if destination_lane == 0:
                    packets.append(Packet())
                packets[count / Packet.WIDTH].copy_lane(
                    source, lane, destination_lane
                )
                count += 1
    else:
        var remaining = UInt64(pack_bits(mask))
        if lane_count < 64:
            remaining &= (UInt64(1) << UInt64(lane_count)) - 1
        if remaining == 0:
            return
        if (
            lane_count == Packet.WIDTH
            and count % Packet.WIDTH == 0
            and mask.reduce_and()
        ):
            packets.append(source.copy())
            count += Packet.WIDTH
            return
        while remaining != 0:
            var destination_lane = count % Packet.WIDTH
            if destination_lane == 0:
                packets.append(Packet())
            ref destination = packets[count / Packet.WIDTH]

            @always_inline
            def copy_lane(
                source_lane: Int, target_lane: Int
            ) {source, mut destination}:
                destination.copy_metadata_lane(source, source_lane, target_lane)

            var selection = select_packet_lanes[Packet.WIDTH](
                remaining, destination_lane, copy_lane
            )
            destination.copy_float_lanes(source, selection[0], destination_lane)
            count += selection[1]
