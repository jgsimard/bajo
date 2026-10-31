"""Candidate acceptance shared by filtered triangle traversal backends."""


trait TriangleHitFilter(Copyable, Deinitable):
    def accept(self, primitive: UInt32, u: Float32, v: Float32) -> Bool:
        ...


@fieldwise_init
struct AcceptAllTriangles(TriangleHitFilter, TrivialRegisterPassable):
    @always_inline
    def accept(self, primitive: UInt32, u: Float32, v: Float32) -> Bool:
        return True
