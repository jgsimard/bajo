from bajo.core.utils import min_argmin
from bajo.core import (
    GeoKind,
    Vec3,
    Vec3f32,
    Normal3f32,
    AABB,
    Point3,
    Point3f32,
    Frame,
    cross,
    normalize,
    Rayf32,
)
from bajo.bvh.constants import EMPTY_LANE, TRACE, f32_max
from bajo.bvh.cpu.bounds_bvh import (
    BoundsBvh,
    BoundsItem,
    BoundsBvhBuilder,
    encode_leaf_ref,
    is_leaf_ref,
    decode_ref_index,
)
from bajo.bvh.types import Hit, TriangleLeafBlock, TypedBvh
from bajo.core.intersect import intersect_ray_tri_edges
from bajo.bvh.cpu.trace import trace_bounds_bvh


struct TriangleBvh[
    frame: Frame,
    bounds_width: SIMDLength,
    leaf_width: SIMDLength = bounds_width,
](Copyable, TypedBvh):
    comptime bvh_frame: Frame = Self.frame

    """Triangle BVH with independent bounds and triangle packet widths.

    During BoundsBvh construction, a tagged leaf reference points into
    tree.leaf_ranges. After construction, _pack_leaves replaces that payload
    with a tagged TriangleLeafBlock index:

        node.data[lane] == EMPTY_LANE -> unused lane
        is_leaf_ref(node.data[lane])  -> TriangleLeafBlock index
        otherwise                     -> internal node index
    """

    var tree: BoundsBvh[Self.frame, Self.bounds_width]
    var leaf_blocks: List[TriangleLeafBlock[Self.frame, Self.leaf_width]]
    var tri_count: Int

    def __init__[
        split_method: String = "median"
    ](out self, var vertices: List[Point3f32[Self.frame]]):
        self.tri_count = len(vertices) / 3
        self.leaf_blocks = List[
            TriangleLeafBlock[Self.frame, Self.leaf_width]
        ]()

        var items = List[BoundsItem[Self.frame]](capacity=self.tri_count)

        for i in range(self.tri_count):
            ref v0 = vertices[i * 3 + 0]
            ref v1 = vertices[i * 3 + 1]
            ref v2 = vertices[i * 3 + 2]

            var bounds = AABB[Self.frame].invalid()
            bounds.grow(v0, v1, v2)

            items.append(BoundsItem(bounds, UInt32(i)))

        var builder = BoundsBvhBuilder[Self.frame, Self.leaf_width](items)
        builder.build[split_method]()

        self.tree = BoundsBvh[Self.frame, Self.bounds_width](builder)

        self._pack_leaves(vertices^)

    def bounds(self) -> AABB[Self.frame]:
        return self.tree.root_bounds()

    def _pack_leaves(mut self, var vertices: List[Point3f32[Self.frame]]):
        self.leaf_blocks = List[TriangleLeafBlock[Self.frame, Self.leaf_width]](
            capacity=(self.tri_count + Int(Self.leaf_width) - 1)
            // Int(Self.leaf_width)
        )
        debug_assert["safe", _use_compiler_assume=True](
            len(self.tree.item_indices) == len(self.tree.item_payloads),
            "triangle BVH item arrays have inconsistent lengths",
        )
        debug_assert["safe", _use_compiler_assume=True](
            len(vertices) == self.tri_count * 3,
            "triangle vertex count changed while packing leaves",
        )

        for ref node in self.tree.nodes:
            comptime for lane in range(Self.bounds_width):
                var child_ref = node.data[lane]

                if child_ref != EMPTY_LANE and is_leaf_ref(child_ref):
                    var leaf_range_idx = decode_ref_index(child_ref)
                    ref leaf_range = self.tree.leaf_ranges.unsafe_get(
                        Int(leaf_range_idx)
                    )

                    var first_item = leaf_range.first_item
                    var item_count = leaf_range.item_count
                    var first = Int(first_item)
                    var count = Int(item_count)

                    debug_assert["safe", _use_compiler_assume=True](
                        count <= Int(Self.leaf_width),
                        "triangle BVH leaf exceeds leaf SIMD width",
                    )
                    debug_assert["safe", _use_compiler_assume=True](
                        first <= len(self.tree.item_indices)
                        and count <= len(self.tree.item_indices) - first,
                        "triangle BVH leaf range is outside item indices",
                    )

                    var leaf_indices = Span(self.tree.item_indices)[
                        first : first + count
                    ]

                    var block = TriangleLeafBlock[Self.frame, Self.leaf_width]()

                    for k, item_idx_u32 in enumerate(leaf_indices):
                        var item_ref = Int(item_idx_u32)
                        var prim_idx = self.tree.item_payloads[item_ref]
                        var base = Int(prim_idx) * 3

                        ref p0 = vertices[base + 0]
                        ref p1 = vertices[base + 1]
                        ref p2 = vertices[base + 2]

                        block.v0.x[k] = p0.x
                        block.v0.y[k] = p0.y
                        block.v0.z[k] = p0.z

                        block.e1.x[k] = p1.x - p0.x
                        block.e1.y[k] = p1.y - p0.y
                        block.e1.z[k] = p1.z - p0.z

                        block.e2.x[k] = p2.x - p0.x
                        block.e2.y[k] = p2.y - p0.y
                        block.e2.z[k] = p2.z - p0.z

                        block.prim_indices[k] = prim_idx

                    var block_idx = UInt32(len(self.leaf_blocks))
                    self.leaf_blocks.append(block^)

                    # keep the leaf tag
                    node.data[lane] = encode_leaf_ref(block_idx)

    def trace[
        mode: TRACE
    ](self, ray: Rayf32[Self.bvh_frame]) -> Hit[Self.bvh_frame]:
        def leaf_fn(
            ray: Rayf32[Self.bvh_frame],
            O: Point3[DType.float32, Self.bvh_frame, Self.leaf_width],
            D: Vec3[DType.float32, Self.bvh_frame, Self.leaf_width],
            _ray_a: SIMD[DType.float32, Self.leaf_width],
            _ray_inv_a: SIMD[DType.float32, Self.leaf_width],
            leaf_block_idx: UInt32,
            mut hit: Hit[Self.bvh_frame],
        ) capturing -> Bool:
            ref block = self.leaf_blocks.unsafe_get(Int(leaf_block_idx))
            var tri_hit = intersect_ray_tri_edges(
                O,
                D,
                block.v0,
                block.e1,
                block.e2,
                hit.t,
                ray.t_min,
            )
            var valid_lane = block.prim_indices.ne(EMPTY_LANE)
            var hit_mask = tri_hit.mask & valid_lane

            if not hit_mask.reduce_or():
                return False

            comptime if mode == TRACE.CLOSEST_HIT:
                var _t = hit_mask.select(tri_hit.t, f32_max)
                var min_t, lane = min_argmin(_t)

                hit.t = min_t
                hit.u = tri_hit.u[lane]
                hit.v = tri_hit.v[lane]
                hit.prim = block.prim_indices[lane]
                hit.inst = EMPTY_LANE

                var e1 = Vec3f32[Self.bvh_frame](
                    block.e1.x[lane],
                    block.e1.y[lane],
                    block.e1.z[lane],
                )
                var e2 = Vec3f32[Self.bvh_frame](
                    block.e2.x[lane],
                    block.e2.y[lane],
                    block.e2.z[lane],
                )

                var normal = normalize(cross(e1, e2))

                hit.normal = Normal3f32[Self.bvh_frame](
                    normal.x,
                    normal.y,
                    normal.z,
                )

            return True

        return trace_bounds_bvh[
            Self.frame,
            Self.bounds_width,
            Self.leaf_width,
            mode,
            leaf_fn,
        ](self.tree, ray)
