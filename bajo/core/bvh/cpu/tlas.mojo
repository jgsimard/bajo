from bajo.core.aabb import AABB
from bajo.core.vec import Vec3, Vec3f32, vmin, vmax, longest_axis, dot
from bajo.core.bvh.types import Ray
from bajo.core.mat import Mat44f32, transform_point, transform_vector
from bajo.core.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.core.bvh.cpu.sphere_bvh import SphereBvh
from bajo.core.bvh.cpu.bounds_bvh import BoundsBvh, BoundsBvhBuilder, BoundsItem
from bajo.core.bvh.cpu.traverse import traverse_wide_ray_bvh


@fieldwise_init
struct BvhInstance(Copyable):
    """Instance of a BLAS in world space.

    - `transform` maps BLAS-local points/vectors to world space.
    - `inv_transform` maps world-space rays to BLAS-local space.
    - `bounds` is the transformed world-space root AABB.
    - `blas_idx` indexes the BLAS array passed to traversal.
    """

    var transform: Mat44f32
    var inv_transform: Mat44f32
    var bounds: AABB
    var blas_idx: UInt32

    @always_inline
    def __init__(out self):
        self.transform = Mat44f32.identity()
        self.inv_transform = Mat44f32.identity()
        self.bounds = AABB.invalid()
        self.blas_idx = 0

    @always_inline
    def __init__(
        out self,
        transform: Mat44f32,
        inv_transform: Mat44f32,
        blas_idx: UInt32,
        blas_bounds: AABB,
    ):
        self.transform = transform.copy()
        self.inv_transform = inv_transform.copy()
        self.blas_idx = blas_idx
        self.bounds = _transform_bounds(transform, blas_bounds)

    @staticmethod
    def from_triangle_blas[
        blas_width: Int
    ](
        transform: Mat44f32,
        inv_transform: Mat44f32,
        blas_idx: UInt32,
        blas: TriangleBvh[blas_width],
    ) -> BvhInstance:
        return BvhInstance(
            transform,
            inv_transform,
            blas_idx,
            blas.bounds(),
        )

    @staticmethod
    def from_sphere_blas[
        blas_width: Int
    ](
        transform: Mat44f32,
        inv_transform: Mat44f32,
        blas_idx: UInt32,
        blas: SphereBvh[blas_width],
    ) -> BvhInstance:
        return BvhInstance(
            transform,
            inv_transform,
            blas_idx,
            blas.bounds(),
        )


@always_inline
def _transform_bounds(transform: Mat44f32, bounds: AABB) -> AABB:
    var corners = InlineArray[Vec3f32, 8](fill=Vec3f32(0.0))

    corners[0] = Vec3f32(bounds._min.x, bounds._min.y, bounds._min.z)
    corners[1] = Vec3f32(bounds._max.x, bounds._min.y, bounds._min.z)
    corners[2] = Vec3f32(bounds._min.x, bounds._max.y, bounds._min.z)
    corners[3] = Vec3f32(bounds._max.x, bounds._max.y, bounds._min.z)
    corners[4] = Vec3f32(bounds._min.x, bounds._min.y, bounds._max.z)
    corners[5] = Vec3f32(bounds._max.x, bounds._min.y, bounds._max.z)
    corners[6] = Vec3f32(bounds._min.x, bounds._max.y, bounds._max.z)
    corners[7] = Vec3f32(bounds._max.x, bounds._max.y, bounds._max.z)

    var out = AABB.invalid()

    comptime for i in range(8):
        var p = transform_point(transform, corners[i])
        out.grow(p)

    return out


struct Tlas[width: Int](Copyable):
    """Wide TLAS over BvhInstance records."""

    var tree: BoundsBvh[Self.width]
    var instances: List[BvhInstance]
    var inst_count: UInt32

    def __init__(out self):
        self.tree = BoundsBvh[Self.width]()
        self.instances = List[BvhInstance]()
        self.inst_count = 0

    def __init__(out self, instances: List[BvhInstance]):
        self.tree = BoundsBvh[Self.width]()
        self.instances = instances.copy()
        self.inst_count = UInt32(len(self.instances))
        self.build["median"]()

    def add_instance(mut self, instance: BvhInstance):
        self.instances.append(instance.copy())
        self.inst_count = UInt32(len(self.instances))

    def build[split_method: String](mut self):
        self.inst_count = UInt32(len(self.instances))

        var items = List[BoundsItem](capacity=Int(self.inst_count))

        for i in range(Int(self.inst_count)):
            ref inst = self.instances[i]
            items.append(BoundsItem(inst.bounds, UInt32(i)))

        var builder = BoundsBvhBuilder[Self.width](items)
        builder.build()

        self.tree = BoundsBvh[Self.width](builder)

    def bounds(self) -> AABB:
        return self.tree.root_bounds()

    @always_inline
    def _traverse_triangle_leaf[
        is_occlusion: Bool,
        blas_width: Int,
    ](
        self,
        mut ray: Ray,
        first_item: UInt32,
        item_count: UInt32,
        blases: UnsafePointer[TriangleBvh[blas_width], MutAnyOrigin],
    ) -> Bool:
        for i in range(Int(item_count)):
            var item_ref = Int(self.tree.item_indices[Int(first_item) + i])
            var inst_idx = self.tree.item_payloads[item_ref]
            ref inst = self.instances[Int(inst_idx)]

            var local_origin = transform_point(
                inst.inv_transform,
                ray.O,
            )
            var local_dir = transform_vector(
                inst.inv_transform,
                ray.D,
            )

            var local_ray = Ray(
                local_origin,
                local_dir,
                ray.hit.t,
            )

            comptime if is_occlusion:
                if blases[Int(inst.blas_idx)].is_occluded(local_ray):
                    return True
            else:
                blases[Int(inst.blas_idx)].traverse(local_ray)

                if local_ray.hit.t < ray.hit.t:
                    ray.hit.t = local_ray.hit.t
                    ray.hit.u = local_ray.hit.u
                    ray.hit.v = local_ray.hit.v
                    ray.hit.prim = local_ray.hit.prim
                    ray.hit.inst = inst_idx

        return False

    @always_inline
    def _traverse_sphere_leaf[
        is_occlusion: Bool,
        blas_width: Int,
    ](
        self,
        mut ray: Ray,
        first_item: UInt32,
        item_count: UInt32,
        blases: UnsafePointer[SphereBvh[blas_width], MutAnyOrigin],
    ) -> Bool:
        for i in range(Int(item_count)):
            var item_ref = Int(self.tree.item_indices[Int(first_item) + i])
            var inst_idx = self.tree.item_payloads[item_ref]
            ref inst = self.instances[Int(inst_idx)]

            var local_origin = transform_point(
                inst.inv_transform,
                ray.O,
            )
            var local_dir = transform_vector(
                inst.inv_transform,
                ray.D,
            )

            var local_ray = Ray(
                local_origin,
                local_dir,
                ray.hit.t,
            )

            comptime if is_occlusion:
                if blases[Int(inst.blas_idx)].is_occluded(local_ray):
                    return True
            else:
                blases[Int(inst.blas_idx)].traverse(local_ray)

                if local_ray.hit.t < ray.hit.t:
                    ray.hit.t = local_ray.hit.t
                    ray.hit.u = local_ray.hit.u
                    ray.hit.v = local_ray.hit.v
                    ray.hit.prim = local_ray.hit.prim
                    ray.hit.inst = inst_idx

        return False

    @always_inline
    def _traverse_triangles_generic[
        is_occlusion: Bool,
        blas_width: Int,
    ](
        self,
        mut ray: Ray,
        blases: UnsafePointer[TriangleBvh[blas_width], MutAnyOrigin],
    ) -> Bool:
        @always_inline
        def leaf_fn(
            mut ray: Ray,
            first_item: UInt32,
            item_count: UInt32,
        ) capturing -> Bool:
            return self._traverse_triangle_leaf[
                is_occlusion,
                blas_width,
            ](
                ray,
                first_item,
                item_count,
                blases,
            )

        return traverse_wide_ray_bvh[
            Self.width,
            is_occlusion,
            leaf_fn,
        ](
            self.tree,
            ray,
        )

    @always_inline
    def _traverse_spheres_generic[
        is_occlusion: Bool,
        blas_width: Int,
    ](
        self,
        mut ray: Ray,
        blases: UnsafePointer[SphereBvh[blas_width], MutAnyOrigin],
    ) -> Bool:
        @always_inline
        def leaf_fn(
            mut ray: Ray,
            first_item: UInt32,
            item_count: UInt32,
        ) capturing -> Bool:
            return self._traverse_sphere_leaf[
                is_occlusion,
                blas_width,
            ](
                ray,
                first_item,
                item_count,
                blases,
            )

        return traverse_wide_ray_bvh[
            Self.width,
            is_occlusion,
            leaf_fn,
        ](
            self.tree,
            ray,
        )

    def traverse_triangles[
        blas_width: Int
    ](
        self,
        mut ray: Ray,
        blases: UnsafePointer[TriangleBvh[blas_width], MutAnyOrigin],
    ):
        _ = self._traverse_triangles_generic[False, blas_width](ray, blases)

    def is_occluded_triangles[
        blas_width: Int
    ](
        self,
        mut ray: Ray,
        blases: UnsafePointer[TriangleBvh[blas_width], MutAnyOrigin],
    ) -> Bool:
        return self._traverse_triangles_generic[True, blas_width](ray, blases)

    def traverse_spheres[
        blas_width: Int
    ](
        self,
        mut ray: Ray,
        blases: UnsafePointer[SphereBvh[blas_width], MutAnyOrigin],
    ):
        _ = self._traverse_spheres_generic[False, blas_width](ray, blases)

    def is_occluded_spheres[
        blas_width: Int
    ](
        self,
        mut ray: Ray,
        blases: UnsafePointer[SphereBvh[blas_width], MutAnyOrigin],
    ) -> Bool:
        return self._traverse_spheres_generic[True, blas_width](ray, blases)
