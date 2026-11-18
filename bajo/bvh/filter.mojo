"""Candidate acceptance shared by filtered triangle traversal backends."""


trait TriangleHitFilter(Copyable, Deinitable):
    def accept(self, primitive: UInt32, u: Float32, v: Float32) -> Bool:
        ...

    def accept_packet[
        length: SIMDLength
    ](
        self,
        primitive: UInt32,
        u: SIMD[.float32, length],
        v: SIMD[.float32, length],
        active: SIMD[.bool, length],
    ) -> SIMD[.bool, length]:
        var accepted = active
        for lane in range(Int(length)):
            if active[lane]:
                accepted[lane] = self.accept(primitive, u[lane], v[lane])
        return accepted


@fieldwise_init
struct AcceptAllTriangles(TriangleHitFilter, TrivialRegisterPassable):
    def accept(self, primitive: UInt32, u: Float32, v: Float32) -> Bool:
        return True
