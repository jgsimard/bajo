from bajo.core import (
    Vec3,
    Point3,
    Point3f32,
    Frame,
    GeoKind,
    normalize,
    Rayf32,
    Ray,
)
from bajo.core.intersect import intersect_ray_sphere_coefficients
from bajo.core.utils import min_argmin
from bajo.bvh.constants import EMPTY_LANE, TRACE, f32_max
from bajo.bvh.cpu.build_method import CpuBvhBuildMethod
from bajo.bvh.cpu.bounds_bvh import (
    BoundsBvh,
    BoundsItem,
    BinaryBoundsBvh,
    _checked_typed_leaf_range,
)
from bajo.bvh.types import Hit, Sphere, SphereLeafBlock


@always_inline
def _trace_sphere_leaf_block[
    frame: Frame,
    width: SIMDLength,
    mode: TRACE,
](
    ray: Rayf32[frame],
    O: Point3[DType.float32, frame, width],
    D: Vec3[.float32, frame, width],
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

    comptime if mode == .CLOSEST_HIT:
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


@always_inline
def _trace_sphere_packet_primitive[
    frame: Frame, length: SIMDLength
](
    rays: Ray[.float32, frame, length],
    active: SIMD[DType.bool, length],
    ray_a: SIMD[DType.float32, length],
    ray_inv_a: SIMD[DType.float32, length],
    prim_idx: UInt32,
    center_scalar: Point3f32[frame],
    radius_scalar: Float32,
    mut packet_hit: Hit[frame, length],
):
    var center = Point3[DType.float32, frame, length](
        center_scalar.x, center_scalar.y, center_scalar.z
    )
    var radius = SIMD[DType.float32, length](radius_scalar)
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
            SIMD[DType.uint32, length](prim_idx), packet_hit.prim
        )
        packet_hit.inst = closer.select(
            SIMD[DType.uint32, length](EMPTY_LANE), packet_hit.inst
        )
        var p = rays.o + candidate.t * rays.d
        var normal = normalize(p - center)
        packet_hit.normal.x = closer.select(normal.x, packet_hit.normal.x)
        packet_hit.normal.y = closer.select(normal.y, packet_hit.normal.y)
        packet_hit.normal.z = closer.select(normal.z, packet_hit.normal.z)


struct _SphereBuild[frame: Frame, width: SIMDLength](Copyable):
    """Private typed build result consumed immediately by `CpuBlasSet` packing.

    Binary-to-wide collapse packs each SphereLeafBlock in the same pass, so a
    tagged leaf reference points directly at its typed block.
    """

    var tree: BoundsBvh[Self.frame, Self.width]
    var leaf_blocks: List[SphereLeafBlock[Self.frame, Self.width]]
    var sphere_count: Int

    def __init__[
        method: CpuBvhBuildMethod = .MEDIAN
    ](out self, spheres: ImmSpan[Sphere[Self.frame], _]):
        self.sphere_count = len(spheres)
        self.leaf_blocks = []

        var sphere_ptr = spheres.unsafe_ptr()
        var items = List[BoundsItem[Self.frame]](capacity=len(spheres))
        for sphere_idx in range(len(spheres)):
            items.append(
                BoundsItem(
                    sphere_ptr[unsafe_offset=sphere_idx].bounds(),
                    UInt32(sphere_idx),
                )
            )

        var builder = BinaryBoundsBvh[Self.frame, Self.width, method](items^)

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
                ref sphere = sphere_ptr[unsafe_offset=Int(sphere_idx)]

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
