"""Topology-certified closed components for exact winding pruning.

A closed oriented triangle chain has zero winding outside its bounds. Verify
opposite edge pairs once; retain unverified faces in an unconditional range.
Position updates cannot change this topological certificate.
"""

from std.math import min, max
from bajo.core import AABB, Frame
from bajo.bvh.constants import EMPTY_LANE
from .bounds import SpatialBoundsBvh
from .query import BoundsQueryView


@fieldwise_init
struct EdgeIncidence(Copyable):
    var face: UInt32
    var count: UInt32
    var balance: Int


def component_root(parents: MutSpan[UInt32, _], face: UInt32) -> UInt32:
    var root = face
    while parents[Int(root)] != root:
        root = parents[Int(root)]
    var current = face
    while parents[Int(current)] != root:
        var next = parents[Int(current)]
        parents[Int(current)] = root
        current = next
    return root


@always_inline
def _layout_component_root(parents: MutSpan[UInt32, _], face: UInt32) -> UInt32:
    # Private layout links start at in-range face IDs and union only roots.
    # Preserve the original two-phase full path compression and union order.
    var links = parents.unsafe_ptr()
    var root = face
    var parent = links.unsafe_load(Int(root))
    while parent != root:
        root = parent
        parent = links.unsafe_load(Int(root))
    var current = face
    var next = links.unsafe_load(Int(current))
    while next != root:
        links.unsafe_store(Int(current), root)
        current = next
        next = links.unsafe_load(Int(current))
    return root


struct WindingView[origin: ImmOrigin, frame: Frame](Copyable):
    var bvh: BoundsQueryView[Self.origin, Self.frame]
    var offsets: ImmSpan[Int, Self.origin]

    def __init__(out self):
        self.bvh = BoundsQueryView[Self.origin, Self.frame](
            ImmSpan[Float32, Self.origin](), ImmSpan[UInt32, Self.origin]()
        )
        self.offsets = ImmSpan[Int, Self.origin]()

    def __init__(
        out self,
        var bvh: BoundsQueryView[Self.origin, Self.frame],
        offsets: ImmSpan[Int, Self.origin],
    ):
        self.bvh = bvh^
        self.offsets = offsets


struct DeviceWindingView[origin: ImmOrigin, frame: Frame](Copyable):
    var bvh: BoundsQueryView[Self.origin, Self.frame]
    var offsets: ImmSpan[UInt32, Self.origin]
    var faces: ImmSpan[UInt32, Self.origin]

    def __init__(out self):
        self.bvh = BoundsQueryView[Self.origin, Self.frame](
            ImmSpan[Float32, Self.origin](), ImmSpan[UInt32, Self.origin]()
        )
        self.offsets = ImmSpan[UInt32, Self.origin]()
        self.faces = ImmSpan[UInt32, Self.origin]()

    def __init__(
        out self,
        var bvh: BoundsQueryView[Self.origin, Self.frame],
        offsets: ImmSpan[UInt32, Self.origin],
        faces: ImmSpan[UInt32, Self.origin],
    ):
        self.bvh = bvh^
        self.offsets = offsets
        self.faces = faces


struct WindingLayout[frame: Frame, width: SIMDLength](Copyable):
    var bvh: SpatialBoundsBvh[Self.frame]
    var offsets: List[Int]
    var face_slots: List[Int]
    var face_components: List[UInt32]

    def __init__(
        out self,
        indices: ImmSpan[UInt32, _],
        boxes: ImmSpan[AABB[Self.frame], _],
    ) raises:
        var count = len(boxes)
        var parents = List[UInt32](length=count, fill=0)
        var sizes = List[UInt32](length=count, fill=1)
        for face in range(count):
            parents[face] = UInt32(face)
        # At most three unique keys per face. A power-of-two table with
        # four slots per face stays at most 75% full, including open meshes.
        var capacity = 1
        while capacity < count * 4:
            capacity *= 2
        var mask = UInt64(capacity - 1)
        var keys = List[UInt64](capacity=capacity)
        keys.resize(unsafe_uninit_length=capacity)
        var edge_faces = List[UInt32](capacity=capacity)
        edge_faces.resize(unsafe_uninit_length=capacity)
        var balances = List[Int](capacity=capacity)
        balances.resize(unsafe_uninit_length=capacity)
        var counts = List[UInt32](length=capacity, fill=0)
        var occupied = List[Int](capacity=count * 3 // 2)
        for face in range(count):
            comptime for corner in range(3):
                var a = indices[face * 3 + corner]
                var b = indices[face * 3 + (corner + 1) % 3]
                var key = (UInt64(min(a, b)) << 32) | UInt64(max(a, b))
                var balance = 1 if a < b else -1 if a > b else 0
                # Fold the product so both vertex IDs influence bucket bits.
                # Exact key comparison resolves collisions during probing.
                var mixed = key * UInt64(0x9E3779B97F4A7C15)
                mixed ^= mixed >> 32
                var slot = Int(mixed & mask)
                while (
                    counts.unsafe_ptr().unsafe_load(slot) != 0
                    and keys.unsafe_ptr().unsafe_load(slot) != key
                ):
                    slot = Int((UInt64(slot) + 1) & mask)
                var incidence = counts.unsafe_ptr().unsafe_load(slot)
                if incidence != 0:
                    var left = _layout_component_root(parents, UInt32(face))
                    var right = _layout_component_root(
                        parents, edge_faces.unsafe_ptr().unsafe_load(slot)
                    )
                    if left != right:
                        if sizes.unsafe_get(Int(left)) < sizes.unsafe_get(
                            Int(right)
                        ):
                            var tmp = left
                            left = right
                            right = tmp
                        parents.unsafe_get(Int(right)) = left
                        sizes.unsafe_get(Int(left)) += sizes.unsafe_get(
                            Int(right)
                        )
                    counts.unsafe_ptr().unsafe_store(slot, incidence + 1)
                    balances.unsafe_ptr().unsafe_store(
                        slot, balances.unsafe_ptr().unsafe_load(slot) + balance
                    )
                else:
                    keys.unsafe_ptr().unsafe_store(slot, key)
                    edge_faces.unsafe_ptr().unsafe_store(slot, UInt32(face))
                    counts.unsafe_ptr().unsafe_store(slot, UInt32(1))
                    balances.unsafe_ptr().unsafe_store(slot, balance)
                    occupied.append(slot)
        var closed = List[Bool](length=count, fill=True)
        for slot in occupied:
            var key = keys.unsafe_ptr().unsafe_load(slot)
            if (
                counts.unsafe_ptr().unsafe_load(slot) != 2
                or balances.unsafe_ptr().unsafe_load(slot) != 0
                or (key >> 32) == (key & UInt64(0xFFFFFFFF))
            ):
                var root = _layout_component_root(
                    parents, edge_faces.unsafe_ptr().unsafe_load(slot)
                )
                closed[Int(root)] = False
        var root_ids = List[UInt32](length=count, fill=EMPTY_LANE)
        var component_boxes = List[AABB[Self.frame]]()
        var component_counts = List[UInt32]()
        self.face_components = List[UInt32](length=count, fill=EMPTY_LANE)
        var open_count = 0
        for face in range(count):
            var root = _layout_component_root(parents, UInt32(face))
            if closed[Int(root)]:
                var component = root_ids[Int(root)]
                if component == EMPTY_LANE:
                    component = UInt32(len(component_boxes))
                    root_ids[Int(root)] = component
                    component_boxes.append(AABB[Self.frame].invalid())
                    component_counts.append(0)
                self.face_components[face] = component
                component_counts[Int(component)] += 1
                component_boxes[Int(component)] = AABB[Self.frame].merge(
                    component_boxes[Int(component)], boxes[face]
                )
            else:
                open_count += 1
        self.offsets = List[Int]()
        self.offsets.append(0)
        for n in component_counts:
            self.offsets.append(
                self.offsets[len(self.offsets) - 1]
                + (Int(n) + Self.width - 1) // Self.width
            )
        self.offsets.append(
            self.offsets[len(self.offsets) - 1]
            + (open_count + Self.width - 1) // Self.width
        )
        var cursors = List[Int]()
        for component in range(len(component_boxes) + 1):
            cursors.append(self.offsets[component] * Self.width)
        self.face_slots = List[Int](length=count, fill=0)
        for face in range(count):
            var component = self.face_components[face]
            var bucket = len(
                component_boxes
            ) if component == EMPTY_LANE else Int(component)
            self.face_slots[face] = cursors[bucket]
            cursors[bucket] += 1
        self.bvh = SpatialBoundsBvh(component_boxes)

    def _view[origin: ImmOrigin](self) -> WindingView[origin, Self.frame]:
        # The containing mesh supplies its owner origin, as for its other spans.
        return WindingView[origin, Self.frame](
            BoundsQueryView[origin, Self.frame](
                ImmSpan(
                    unsafe_ptr=self.bvh.bounds.unsafe_ptr().unsafe_origin_cast[
                        origin
                    ](),
                    length=len(self.bvh.bounds),
                ),
                ImmSpan(
                    unsafe_ptr=self.bvh.nodes.unsafe_ptr().unsafe_origin_cast[
                        origin
                    ](),
                    length=len(self.bvh.nodes),
                ),
            ),
            ImmSpan(
                unsafe_ptr=self.offsets.unsafe_ptr().unsafe_origin_cast[
                    origin
                ](),
                length=len(self.offsets),
            ),
        )
