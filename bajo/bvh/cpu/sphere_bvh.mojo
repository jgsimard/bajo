from bajo.core import (
    AABB,
    Vec3,
    Point3,
    Point3f32,
    Frame,
    GeoKind,
    normalize,
    Rayf32,
)
from bajo.core.intersect import intersect_ray_sphere_coefficients
from bajo.core.utils import min_argmin
from bajo.bvh.constants import EMPTY_LANE, TRACE, f32_max
from bajo.bvh.cpu.bounds_bvh import (
    BoundsBvh,
    BoundsItem,
    BoundsBvhBuilder,
    decode_ref_index,
    encode_leaf_ref,
    is_leaf_ref,
)
from bajo.bvh.cpu.trace import trace_bounds_bvh
from bajo.bvh.types import Hit, Sphere, SphereLeafBlock, TypedBvh


struct SphereBvh[frame: Frame, width: SIMDLength](Copyable, TypedBvh):
    comptime bvh_frame: Frame = Self.frame

    """Sphere-specific wrapper around BoundsBvh[width].

    The generic tree is built from BoundsItem ranges. During BoundsBvh
    construction, a tagged leaf reference points into tree.leaf_ranges.

    After construction, each leaf range is packed into a SphereLeafBlock and
    the tagged leaf payload is rewritten to the SphereLeafBlock index.
    """

    var tree: BoundsBvh[Self.frame, Self.width]
    var spheres: List[Sphere[Self.frame]]
    var leaf_blocks: List[SphereLeafBlock[Self.frame, Self.width]]
    var sphere_count: Int

    def __init__[
        split_method: String = "median"
    ](out self, var spheres: List[Sphere[Self.frame]]):
        self.spheres = spheres^
        self.sphere_count = len(self.spheres)
        self.leaf_blocks = []

        var items = [
            BoundsItem(s.bounds(), UInt32(i))
            for i, s in enumerate(self.spheres)
        ]

        var builder = BoundsBvhBuilder[Self.frame, Self.width](items)
        builder.build[split_method]()

        self.tree = BoundsBvh[Self.frame, Self.width](builder)

        self._pack_leaves()

    def bounds(self) -> AABB[Self.frame]:
        return self.tree.root_bounds()

    def _pack_leaves(mut self):
        self.leaf_blocks = List[SphereLeafBlock[Self.frame, Self.width]](
            capacity=(self.sphere_count + Int(Self.width) - 1)
            // Int(Self.width)
        )
        debug_assert["safe", _use_compiler_assume=True](
            len(self.tree.item_indices) == len(self.tree.item_payloads),
            "sphere BVH item arrays have inconsistent lengths",
        )
        debug_assert["safe", _use_compiler_assume=True](
            len(self.spheres) == self.sphere_count,
            "sphere count changed while packing leaves",
        )

        for ref node in self.tree.nodes:
            comptime for lane in range(Self.width):
                var child_ref = node.data[lane]

                if child_ref != EMPTY_LANE and is_leaf_ref(child_ref):
                    var leaf_range_idx = decode_ref_index(child_ref)
                    ref leaf_range = self.tree.leaf_ranges.unsafe_get(
                        Int(leaf_range_idx)
                    )

                    var first = Int(leaf_range.first_item)
                    var count = Int(leaf_range.item_count)

                    debug_assert["safe", _use_compiler_assume=True](
                        count <= Int(Self.width),
                        "sphere BVH leaf exceeds SIMD width",
                    )
                    debug_assert["safe", _use_compiler_assume=True](
                        first <= len(self.tree.item_indices)
                        and count <= len(self.tree.item_indices) - first,
                        "sphere BVH leaf range is outside item indices",
                    )

                    var leaf_indices = Span(self.tree.item_indices)[
                        first : first + count
                    ]

                    var block = SphereLeafBlock[Self.frame, Self.width]()

                    for k, item_idx_u32 in enumerate(leaf_indices):
                        var item_ref = Int(item_idx_u32)
                        var sphere_idx = self.tree.item_payloads[item_ref]
                        ref sphere = self.spheres.unsafe_get(Int(sphere_idx))

                        block.center.x[k] = sphere.center.x
                        block.center.y[k] = sphere.center.y
                        block.center.z[k] = sphere.center.z
                        block.radius[k] = sphere.radius

                        block.prim_indices[k] = sphere_idx

                    var block_idx = UInt32(len(self.leaf_blocks))
                    self.leaf_blocks.append(block^)

                    # Preserve the leaf tag, but replace the construction-time
                    # leaf-range payload with the packed leaf-block index.
                    node.data[lane] = encode_leaf_ref(block_idx)

    def trace[
        mode: TRACE
    ](self, ray: Rayf32[Self.bvh_frame]) -> Hit[Self.bvh_frame]:
        def leaf_fn(
            ray: Rayf32[Self.bvh_frame],
            O: Point3[DType.float32, Self.bvh_frame, Self.width],
            D: Vec3[DType.float32, Self.bvh_frame, Self.width],
            ray_a: SIMD[DType.float32, Self.width],
            ray_inv_a: SIMD[DType.float32, Self.width],
            leaf_block_idx: UInt32,
            mut hit: Hit[Self.bvh_frame],
        ) capturing -> Bool:
            # Unsafe access avoids bounds checks in the traversal hot path.
            ref block = self.leaf_blocks.unsafe_get(Int(leaf_block_idx))

            var sphere_hit = intersect_ray_sphere_coefficients(
                O,
                D,
                block.center,
                block.radius,
                ray_a,
                ray_inv_a,
                hit.t,
                ray.t_min,
            )
            var valid_lane = block.prim_indices.ne(EMPTY_LANE)
            var hit_mask = sphere_hit.mask & valid_lane

            if not hit_mask.reduce_or():
                return False

            comptime if mode == TRACE.CLOSEST_HIT:
                var candidate_t = hit_mask.select(sphere_hit.t, f32_max)
                var min_t, lane = min_argmin(candidate_t)

                hit.t = min_t
                hit.u = 0.0
                hit.v = 0.0
                hit.inst = EMPTY_LANE
                hit.prim = block.prim_indices[lane]
                var center = Point3f32[Self.bvh_frame](
                    block.center.x[lane],
                    block.center.y[lane],
                    block.center.z[lane],
                )
                var p = ray.o + min_t * ray.d
                hit.normal = normalize(p - center).unsafe_convert[
                    new_kind=GeoKind.NORMAL
                ]()

            return True

        return trace_bounds_bvh[
            Self.frame,
            Self.width,
            mode,
            leaf_fn,
            True,
        ](
            self.tree,
            ray,
        )
