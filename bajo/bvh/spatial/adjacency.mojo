"""Retained vertex incidence and leaf locations for local normal-sign queries.

Packed storage contains vertex/face counts, CSR offsets and face IDs, then one
leaf location per face in the global tree and one in its optional group tree.
Leaf locations change on rebuild; incidence is independent of vertex positions.
"""
from max.gpu import global_idx
from bajo.bvh.constants import EMPTY_LANE
from .query import BoundsQuery
from bajo.core import Frame


@always_inline
def update_adjacency_leaf(
    nodes: ImmSpan[UInt32, _], data: MutSpan[UInt32, _], node: Int
):
    var face = nodes[node * 2 + 1]
    if face == EMPTY_LANE:
        return
    var vertices = Int(data[0])
    var faces = Int(data[1])
    var base = 3 + vertices + Int(data[2 + vertices])
    if node >= Int(nodes[0]):
        base += faces
    data[base + Int(face)] = UInt32(node)


def refresh_adjacency(nodes: ImmSpan[UInt32, _], data: MutSpan[UInt32, _]):
    if len(data) > 0:
        for node in range(len(nodes) // 2):
            update_adjacency_leaf(nodes, data, node)


def refresh_adjacency_kernel(
    nodes: ImmSpan[UInt32, ImmutAnyOrigin],
    data: MutSpan[UInt32, MutAnyOrigin],
    status: ImmSpan[UInt32, ImmutAnyOrigin],
):
    var node = Int(global_idx.x)
    if status[0] == 0 and node < len(nodes) // 2:
        update_adjacency_leaf(nodes, data, node)


def build_adjacency(
    indices: ImmSpan[UInt32, _], vertices: Int, nodes: ImmSpan[UInt32, _]
) -> List[UInt32]:
    var faces = len(indices) // 3
    var data = List[UInt32](length=3 + vertices, fill=0)
    data[0] = UInt32(vertices)
    data[1] = UInt32(faces)
    for face in range(faces):
        for corner in range(3):
            var vertex = indices[face * 3 + corner]
            var first = True
            for previous in range(corner):
                first &= vertex != indices[face * 3 + previous]
            if first:
                data[3 + Int(vertex)] += 1
    for vertex in range(vertices):
        data[3 + vertex] += data[2 + vertex]
    var incidence = Int(data[2 + vertices])
    data.resize(3 + vertices + incidence + 2 * faces, fill=EMPTY_LANE)
    var cursors = List[UInt32](capacity=vertices)
    for vertex in range(vertices):
        cursors.append(data[2 + vertex])
    for face in range(faces):
        for corner in range(3):
            var vertex = indices[face * 3 + corner]
            var first = True
            for previous in range(corner):
                first &= vertex != indices[face * 3 + previous]
            if first:
                var slot = Int(vertex)
                data[3 + vertices + Int(cursors[slot])] = UInt32(face)
                cursors[slot] += 1
    refresh_adjacency(nodes, data)
    return data^


struct NormalSignFaces[origin: ImmOrigin, frame: Frame]:
    var data: ImmSpan[UInt32, Self.origin]
    var fallback: BoundsQuery[Self.origin, Self.frame]
    var begin: Int
    var end: Int
    var locations: Int
    var root: UInt32
    var escape: UInt32
    var previous: UInt32
    var indexed: Bool

    def __init__(
        out self,
        data: ImmSpan[UInt32, Self.origin],
        vertex: UInt32,
        root: UInt32,
        escape: UInt32,
        var fallback: BoundsQuery[Self.origin, Self.frame],
    ):
        self.data = data
        self.fallback = fallback^
        self.begin = 0
        self.end = 0
        self.locations = 0
        self.root = root
        self.escape = escape
        self.previous = EMPTY_LANE
        self.indexed = False
        if len(data) > 0:
            var vertices = Int(data[0])
            self.begin = 3 + vertices + Int(data[2 + Int(vertex)])
            self.end = 3 + vertices + Int(data[3 + Int(vertex)])
            self.locations = 3 + vertices + Int(data[2 + vertices])
            # Match the original preorder accumulation exactly. Local valence
            # is normally small; a high-valence fan retains linear traversal
            # instead of paying quadratic selection costs.
            self.indexed = self.end - self.begin <= 32

    def next(mut self, mut face: UInt32) -> Bool:
        if not self.indexed:
            return self.fallback.next(face)
        var selected = EMPTY_LANE
        var best = self.escape
        var faces = Int(self.data[1])
        for slot in range(self.begin, self.end):
            var candidate = self.data[slot]
            var node = self.data[self.locations + Int(candidate)]
            if node < self.root or node >= self.escape:
                node = self.data[self.locations + faces + Int(candidate)]
            if (
                node >= self.root
                and node < best
                and (self.previous == EMPTY_LANE or node > self.previous)
            ):
                best = node
                selected = candidate
        if selected == EMPTY_LANE:
            return False
        self.previous = best
        face = selected
        return True
