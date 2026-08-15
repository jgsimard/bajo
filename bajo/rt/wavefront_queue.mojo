"""Packet queues and host/device wavefront boundary records."""

from bajo.core import Frame, Rayf32
from bajo.rt.types import Color, SurfaceHit


comptime FRONT_FACE_BIT = UInt32(0x80000000)
comptime PATH_INDEX_MASK = UInt32(0x7FFFFFFF)


@fieldwise_init
struct WavePath(Copyable, Writable):
    var path_id: UInt32
    var ray: Rayf32[Frame.WORLD]
    var throughput: Color


@fieldwise_init
struct WaveShade(Copyable, Writable):
    var path_idx: UInt32
    var hit: SurfaceHit[1]


struct PathPacket[length: SIMDLength](Copyable):
    var path_ids: SIMD[DType.uint32, Self.length]
    var ox: SIMD[DType.float32, Self.length]
    var oy: SIMD[DType.float32, Self.length]
    var oz: SIMD[DType.float32, Self.length]
    var t_min: SIMD[DType.float32, Self.length]
    var dx: SIMD[DType.float32, Self.length]
    var dy: SIMD[DType.float32, Self.length]
    var dz: SIMD[DType.float32, Self.length]
    var t_max: SIMD[DType.float32, Self.length]
    var tx: SIMD[DType.float32, Self.length]
    var ty: SIMD[DType.float32, Self.length]
    var tz: SIMD[DType.float32, Self.length]
    var bsdf_pdfs: SIMD[DType.float32, Self.length]
    var deltas: SIMD[DType.bool, Self.length]

    def __init__(out self):
        self.path_ids = 0
        self.ox = 0.0
        self.oy = 0.0
        self.oz = 0.0
        self.t_min = 0.0
        self.dx = 0.0
        self.dy = 0.0
        self.dz = 0.0
        self.t_max = 0.0
        self.tx = 0.0
        self.ty = 0.0
        self.tz = 0.0
        self.bsdf_pdfs = 0.0
        self.deltas = SIMD[DType.bool, Self.length](fill=True)


struct PacketPathQueue[length: SIMDLength](Sized):
    var packets: List[PathPacket[Self.length]]
    var count: Int

    def __init__(out self, capacity: Int):
        self.packets = List[PathPacket[Self.length]](
            capacity=(capacity + Self.length - 1) / Self.length
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
        destination.path_ids[destination_lane] = packet.path_ids[source_lane]
        destination.ox[destination_lane] = packet.ox[source_lane]
        destination.oy[destination_lane] = packet.oy[source_lane]
        destination.oz[destination_lane] = packet.oz[source_lane]
        destination.t_min[destination_lane] = packet.t_min[source_lane]
        destination.dx[destination_lane] = packet.dx[source_lane]
        destination.dy[destination_lane] = packet.dy[source_lane]
        destination.dz[destination_lane] = packet.dz[source_lane]
        destination.t_max[destination_lane] = packet.t_max[source_lane]
        destination.tx[destination_lane] = packet.tx[source_lane]
        destination.ty[destination_lane] = packet.ty[source_lane]
        destination.tz[destination_lane] = packet.tz[source_lane]
        destination.bsdf_pdfs[destination_lane] = packet.bsdf_pdfs[source_lane]
        destination.deltas[destination_lane] = packet.deltas[source_lane]
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
        mask: SIMD[DType.bool, Self.length],
        lane_count: Int,
    ):
        debug_assert["safe", _use_compiler_assume=True](
            lane_count >= 0 and lane_count <= Self.length,
            "invalid masked path packet lane count",
        )
        for lane in range(lane_count):
            if mask[lane]:
                self._append_packet_lane(packet, lane)


struct ShadePacket[length: SIMDLength](Copyable):
    var path_ids: SIMD[DType.uint32, Self.length]
    var ox: SIMD[DType.float32, Self.length]
    var oy: SIMD[DType.float32, Self.length]
    var oz: SIMD[DType.float32, Self.length]
    var dx: SIMD[DType.float32, Self.length]
    var dy: SIMD[DType.float32, Self.length]
    var dz: SIMD[DType.float32, Self.length]
    var tx: SIMD[DType.float32, Self.length]
    var ty: SIMD[DType.float32, Self.length]
    var tz: SIMD[DType.float32, Self.length]
    var nx: SIMD[DType.float32, Self.length]
    var ny: SIMD[DType.float32, Self.length]
    var nz: SIMD[DType.float32, Self.length]
    var hit_t: SIMD[DType.float32, Self.length]
    var surface_indices: SIMD[DType.uint32, Self.length]
    var front_faces: SIMD[DType.bool, Self.length]

    def __init__(out self):
        self.path_ids = 0
        self.ox = 0.0
        self.oy = 0.0
        self.oz = 0.0
        self.dx = 0.0
        self.dy = 0.0
        self.dz = 0.0
        self.tx = 0.0
        self.ty = 0.0
        self.tz = 0.0
        self.nx = 0.0
        self.ny = 0.0
        self.nz = 0.0
        self.hit_t = 0.0
        self.surface_indices = 0
        self.front_faces = SIMD[DType.bool, Self.length](fill=False)


struct PacketShadeQueue[length: SIMDLength](Sized):
    var packets: List[ShadePacket[Self.length]]
    var count: Int

    def __init__(out self, capacity: Int):
        self.packets = List[ShadePacket[Self.length]](
            capacity=(capacity + Self.length - 1) / Self.length
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
        shade.ox[shade_lane] = path_packet.ox[path_lane]
        shade.oy[shade_lane] = path_packet.oy[path_lane]
        shade.oz[shade_lane] = path_packet.oz[path_lane]
        shade.dx[shade_lane] = path_packet.dx[path_lane]
        shade.dy[shade_lane] = path_packet.dy[path_lane]
        shade.dz[shade_lane] = path_packet.dz[path_lane]
        shade.tx[shade_lane] = path_packet.tx[path_lane]
        shade.ty[shade_lane] = path_packet.ty[path_lane]
        shade.tz[shade_lane] = path_packet.tz[path_lane]
        shade.nx[shade_lane] = hit.normal.x
        shade.ny[shade_lane] = hit.normal.y
        shade.nz[shade_lane] = hit.normal.z
        shade.hit_t[shade_lane] = hit.t
        shade.surface_indices[shade_lane] = hit.surface.index()
        shade.front_faces[shade_lane] = hit.front_face
        self.count += 1
