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
)
from bajo.bvh.cpu.trace import trace_bounds_bvh
from bajo.bvh.types import Hit, Sphere, SphereLeafBlock, TypedBvh


struct SphereBvh[frame: Frame, width: SIMDLength](Copyable, TypedBvh):
    comptime bvh_frame: Frame = Self.frame

    """Sphere-specific wrapper around BoundsBvh[width].

    Binary-to-wide collapse packs each SphereLeafBlock in the same pass, so a
    tagged leaf reference points directly at its typed block.
    """

    var tree: BoundsBvh[Self.frame, Self.width]
    var spheres: List[Sphere[Self.frame]]
    var leaf_blocks: List[SphereLeafBlock[Self.frame, Self.width]]
    var sphere_count: Int

    def __init__[
        split_method: String = "median"
    ](out self, var spheres: List[Sphere[Self.frame]]):
        self.sphere_count = len(spheres)
        self.leaf_blocks = []

        var items = [
            BoundsItem(s.bounds(), UInt32(i)) for i, s in enumerate(spheres)
        ]

        var builder = BoundsBvhBuilder[Self.frame, Self.width](items^)
        builder.build[split_method]()

        var leaf_blocks = List[SphereLeafBlock[Self.frame, Self.width]](
            capacity=(Int(builder.nodes_used) + 1) // 2
        )

        @always_inline
        def pack_leaf(
            first_item: UInt32, item_count: UInt32
        ) capturing -> UInt32:
            var first = Int(first_item)
            var count = Int(item_count)

            debug_assert["safe", _use_compiler_assume=True](
                count <= Int(Self.width),
                "sphere BVH leaf exceeds SIMD width",
            )
            debug_assert["safe", _use_compiler_assume=True](
                first <= len(builder.item_indices)
                and count <= len(builder.item_indices) - first,
                "sphere BVH leaf range is outside item indices",
            )

            var block = SphereLeafBlock[Self.frame, Self.width]()

            for k in range(count):
                var item_ref = Int(builder.item_indices.unsafe_get(first + k))
                # Typed items are created in primitive order.
                var sphere_idx = UInt32(item_ref)
                ref sphere = spheres.unsafe_get(Int(sphere_idx))

                block.center.x[k] = sphere.center.x
                block.center.y[k] = sphere.center.y
                block.center.z[k] = sphere.center.z
                block.radius[k] = sphere.radius
                block.prim_indices[k] = sphere_idx

            var block_idx = UInt32(len(leaf_blocks))
            leaf_blocks.append(block^)
            return block_idx

        self.tree = BoundsBvh[Self.frame, Self.width].__init__[pack_leaf](
            builder
        )
        self.leaf_blocks = leaf_blocks^
        self.spheres = spheres^

    def bounds(self) -> AABB[Self.frame]:
        return self.tree.root_bounds()

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
            Self.width,
            mode,
            leaf_fn,
            True,
        ](
            self.tree,
            ray,
        )
