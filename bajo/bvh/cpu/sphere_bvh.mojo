from bajo.core import (
    AABB,
    Vec3,
    Point3,
    Point3f32,
    Frame,
    GeoKind,
    dot,
    normalize,
    Rayf32,
    Ray,
)
from bajo.core.intersect import intersect_ray_sphere_coefficients
from bajo.core.utils import min_argmin
from bajo.bvh.constants import EMPTY_LANE, TRACE, f32_max
from bajo.bvh.cpu.bounds_bvh import (
    BoundsBvh,
    BoundsItem,
    BinaryBoundsBvh,
    _checked_typed_leaf_range,
)
from bajo.bvh.cpu.trace import trace_sphere_bounds_bvh
from bajo.bvh.cpu.packet import (
    trace_packet_stack_bounds_bvh,
)
from bajo.bvh.types import Hit, Sphere, SphereLeafBlock, TypedBvh


@always_inline
def _trace_sphere_leaf_block[
    frame: Frame,
    width: SIMDLength,
    mode: TRACE,
](
    ray: Rayf32[frame],
    O: Point3[DType.float32, frame, width],
    D: Vec3[DType.float32, frame, width],
    ray_a: SIMD[DType.float32, width],
    ray_inv_a: SIMD[DType.float32, width],
    block: SphereLeafBlock[frame, width],
    mut hit: Hit[frame],
) -> Bool:
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
        var center = Point3f32[frame](
            block.center.x[lane],
            block.center.y[lane],
            block.center.z[lane],
        )
        var p = ray.o + min_t * ray.d
        hit.normal = normalize(p - center).unsafe_convert[
            new_kind=GeoKind.NORMAL
        ]()

    return True


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

        var builder = BinaryBoundsBvh[Self.frame, Self.width, split_method](
            items^
        )

        var leaf_blocks = List[SphereLeafBlock[Self.frame, Self.width]](
            capacity=(Int(builder.nodes_used) + 1) // 2
        )

        @always_inline
        def pack_leaf(
            first_item: UInt32, item_count: UInt32
        ) {imm, mut leaf_blocks} -> UInt32:
            var first, count = _checked_typed_leaf_range[Self.width](
                first_item, item_count, len(builder.item_indices)
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

        self.tree = BoundsBvh[Self.frame, Self.width](builder, pack_leaf)
        self.leaf_blocks = leaf_blocks^
        self.spheres = spheres^

    def bounds(self) -> AABB[Self.frame]:
        return self.tree.root_bounds()

    @always_inline
    def trace[
        mode: TRACE, length: SIMDLength
    ](
        self,
        rays: Ray[DType.float32, Self.bvh_frame, length],
        valid: SIMD[DType.bool, length] = SIMD[DType.bool, length](fill=True),
    ) -> Hit[Self.bvh_frame, length]:
        comptime if length == 1:
            if not valid[0]:
                return Hit[Self.bvh_frame, length].miss(rays.t_max)
            var ray = Rayf32[Self.bvh_frame](
                [rays.o.x[0], rays.o.y[0], rays.o.z[0]],
                [rays.d.x[0], rays.d.y[0], rays.d.z[0]],
                rays.t_min[0],
                rays.t_max[0],
            )
            var scalar_hit = self._trace_ordered[mode](ray)
            var result = Hit[Self.bvh_frame, length].miss(rays.t_max)
            result.u[0] = scalar_hit.u[0]
            result.v[0] = scalar_hit.v[0]
            result.prim[0] = scalar_hit.prim[0]
            result.inst[0] = scalar_hit.inst[0]
            result.normal.x[0] = scalar_hit.normal.x[0]
            result.normal.y[0] = scalar_hit.normal.y[0]
            result.normal.z[0] = scalar_hit.normal.z[0]
            result.t[0] = scalar_hit.t[0]
            return result
        else:
            comptime assert mode == TRACE.CLOSEST_HIT
            return self._trace_shared_stack(rays, valid)

    @always_inline
    def _trace_ordered[
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
        ) {imm} -> Bool:
            # Unsafe access avoids bounds checks in the traversal hot path.
            ref block = self.leaf_blocks.unsafe_get(Int(leaf_block_idx))
            return _trace_sphere_leaf_block[Self.bvh_frame, Self.width, mode](
                ray, O, D, ray_a, ray_inv_a, block, hit
            )

        return trace_sphere_bounds_bvh[
            frame=Self.frame,
            bounds_width=Self.width,
            leaf_width=Self.width,
            mode=mode,
        ](
            self.tree,
            ray,
            leaf_fn,
        )

    @always_inline
    def _trace_shared_stack[
        length: SIMDLength
    ](
        self,
        rays: Ray[DType.float32, Self.bvh_frame, length],
        valid: SIMD[DType.bool, length],
    ) -> Hit[Self.bvh_frame, length]:
        """Trace a coherent SIMD ray packet with one shared hierarchy stack."""
        var hit = Hit[Self.bvh_frame, length].miss(rays.t_max)
        var ray_a = dot(rays.d, rays.d)
        var ray_inv_a = Float32(1.0) / ray_a

        def leaf_fn(
            active: SIMD[DType.bool, length],
            leaf_block_idx: UInt32,
            mut packet_hit: Hit[Self.bvh_frame, length],
        ) {imm}:
            ref block = self.leaf_blocks.unsafe_get(Int(leaf_block_idx))
            comptime for prim_lane in range(Self.width):
                var prim_idx = block.prim_indices[prim_lane]
                if prim_idx != EMPTY_LANE:
                    var center = Point3[DType.float32, Self.bvh_frame, length](
                        block.center.x[prim_lane],
                        block.center.y[prim_lane],
                        block.center.z[prim_lane],
                    )
                    var radius = SIMD[DType.float32, length](
                        block.radius[prim_lane]
                    )
                    var candidate = intersect_ray_sphere_coefficients(
                        rays.o,
                        rays.d,
                        center,
                        radius,
                        ray_a,
                        ray_inv_a,
                        packet_hit.t,
                        rays.t_min,
                    )
                    var closer = active & candidate.mask
                    if closer.reduce_or():
                        packet_hit.t = closer.select(candidate.t, packet_hit.t)
                        packet_hit.u = closer.select(
                            SIMD[DType.float32, length](0.0), packet_hit.u
                        )
                        packet_hit.v = closer.select(
                            SIMD[DType.float32, length](0.0), packet_hit.v
                        )
                        packet_hit.prim = closer.select(
                            SIMD[DType.uint32, length](prim_idx),
                            packet_hit.prim,
                        )
                        packet_hit.inst = closer.select(
                            SIMD[DType.uint32, length](EMPTY_LANE),
                            packet_hit.inst,
                        )
                        var p = rays.o + candidate.t * rays.d
                        var normal = normalize(p - center)
                        packet_hit.normal.x = closer.select(
                            normal.x, packet_hit.normal.x
                        )
                        packet_hit.normal.y = closer.select(
                            normal.y, packet_hit.normal.y
                        )
                        packet_hit.normal.z = closer.select(
                            normal.z, packet_hit.normal.z
                        )

        trace_packet_stack_bounds_bvh[
            frame=Self.frame,
            bounds_width=Self.width,
            length=length,
        ](
            self.tree,
            rays,
            valid,
            hit,
            leaf_fn,
            lambda (
                active: SIMD[DType.bool, length],
                _child_ref: UInt32,
                mut _packet_hit: Hit[Self.bvh_frame, length],
            ): None,
            lambda (_child_ref: UInt32): None,
        )
        return hit
