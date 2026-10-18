"""Ordered preorder traversal with statically specialized query callbacks."""
from std.memory import stack_allocation
from bajo.bvh.constants import EMPTY_LANE


@always_inline
def traverse_ordered[
    dtype: DType,
    furthest: Bool,
    AcceptFn: def(Scalar[dtype], Scalar[dtype]) -> Bool,
    ChildrenFn: def(UInt32, UInt32) -> Tuple[Scalar[dtype], Scalar[dtype]],
    VisitFn: def(UInt32) -> Scalar[dtype],
](
    nodes: ImmSpan[UInt32, _],
    root: UInt32,
    initial_bound: Scalar[dtype],
    initial_cutoff: Scalar[dtype],
    ref accept: AcceptFn,
    ref children: ChildrenFn,
    ref visit: VisitFn,
):
    # Spatial construction and device rebuild retain balanced skeletons.
    # UInt32 node indices limit depth to 32; one pending sibling per level.
    var pending = stack_allocation[64, UInt32]()
    var distances = stack_allocation[64, Scalar[dtype]]()
    var count = 0
    var node = root
    var bound = initial_bound
    var cutoff = initial_cutoff
    while True:
        if accept(bound, cutoff):
            var face = nodes.unsafe_ptr().unsafe_load(Int(node) * 2 + 1)
            if face == EMPTY_LANE:
                var left = node + 1
                var right = nodes.unsafe_ptr().unsafe_load(Int(left) * 2)
                var ld, rd = children(left, right)
                if rd > ld if furthest else rd < ld:
                    swap(left, right)
                    swap(ld, rd)
                if accept(rd, cutoff):
                    pending[unsafe_offset=count] = right
                    distances[unsafe_offset=count] = rd
                    count += 1
                node = left
                bound = ld
                continue
            cutoff = visit(face)
        if count == 0:
            break
        count -= 1
        node = pending[unsafe_offset=count]
        bound = distances[unsafe_offset=count]
