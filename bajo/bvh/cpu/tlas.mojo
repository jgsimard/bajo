from bajo.core.aabb import AABB
from bajo.core.vec import Vec3, Vec3f32, vmin, vmax, longest_axis, dot
from bajo.bvh.types import Ray, Hit, Instance, transform_bounds
from bajo.bvh.constants import EMPTY_LANE, TRACE_CLOSEST_HIT, TRACE_ANY_HIT
from bajo.core.mat import Mat44f32, transform_point, transform_vector
from bajo.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.bvh.cpu.sphere_bvh import SphereBvh
from bajo.bvh.cpu.bounds_bvh import BoundsBvh, BoundsBvhBuilder, BoundsItem
from bajo.bvh.cpu.traverse import traverse_wide_ray_bvh


struct Tlas[width: Int](Copyable):
    """Wide TLAS over Instance records."""

    var tree: BoundsBvh[Self.width]
    var instances: List[Instance]
    var inst_count: UInt32

    def __init__(out self):
        self.tree = BoundsBvh[Self.width]()
        self.instances = List[Instance]()
        self.inst_count = 0

    def __init__(out self, instances: List[Instance]):
        self.tree = BoundsBvh[Self.width]()
        self.instances = instances.copy()
        self.inst_count = UInt32(len(self.instances))
        self.build["median"]()

    def add_instance(mut self, instance: Instance):
        self.instances.append(instance.copy())
        self.inst_count = UInt32(len(self.instances))

    def build[split_method: String](mut self):
        self.inst_count = UInt32(len(self.instances))

        var items = List[BoundsItem](capacity=Int(self.inst_count))

        for i in range(Int(self.inst_count)):
            ref inst = self.instances[i]
            items.append(BoundsItem(inst.bounds, UInt32(i)))

        var builder = BoundsBvhBuilder[Self.width](items)
        builder.build[split_method]()

        self.tree = BoundsBvh[Self.width](builder)

    def bounds(self) -> AABB:
        return self.tree.root_bounds()

    @always_inline
    def _traverse_triangle_leaf[
        mode: String,
        blas_width: Int,
    ](
        self,
        ray: Ray,
        first_item: UInt32,
        item_count: UInt32,
        blases: UnsafePointer[TriangleBvh[blas_width], MutAnyOrigin],
        mut hit: Hit,
    ) -> Bool:
        for i in range(Int(item_count)):
            var item_ref = Int(self.tree.item_indices[Int(first_item) + i])
            var inst_idx = self.tree.item_payloads[item_ref]
            ref inst = self.instances[Int(inst_idx)]

            var local_origin = transform_point(
                inst.inv_transform,
                ray.o,
            )
            var local_dir = transform_vector(
                inst.inv_transform,
                ray.d,
            )

            var local_ray = Ray(
                local_origin,
                local_dir,
                ray.t_min,
                hit.t,
                ray.mask,
            )

            comptime if mode == TRACE_ANY_HIT:
                if blases[Int(inst.blas_idx)].is_occluded(local_ray):
                    return True
            else:
                var local_hit = blases[Int(inst.blas_idx)].traverse(local_ray)

                if local_hit.is_hit() and local_hit.t < hit.t:
                    hit.t = local_hit.t
                    hit.u = local_hit.u
                    hit.v = local_hit.v
                    hit.prim = local_hit.prim
                    hit.inst = inst_idx
                    hit.occluded = UInt32(0)

        return False

    @always_inline
    def _traverse_sphere_leaf[
        mode: String,
        blas_width: Int,
    ](
        self,
        ray: Ray,
        first_item: UInt32,
        item_count: UInt32,
        blases: UnsafePointer[SphereBvh[blas_width], MutAnyOrigin],
        mut hit: Hit,
    ) -> Bool:
        for i in range(Int(item_count)):
            var item_ref = Int(self.tree.item_indices[Int(first_item) + i])
            var inst_idx = self.tree.item_payloads[item_ref]
            ref inst = self.instances[Int(inst_idx)]

            var local_origin = transform_point(
                inst.inv_transform,
                ray.o,
            )
            var local_dir = transform_vector(
                inst.inv_transform,
                ray.d,
            )

            var local_ray = Ray(
                local_origin,
                local_dir,
                ray.t_min,
                hit.t,
                ray.mask,
            )

            comptime if mode == TRACE_ANY_HIT:
                if blases[Int(inst.blas_idx)].is_occluded(local_ray):
                    return True
            else:
                var local_hit = blases[Int(inst.blas_idx)].traverse(local_ray)

                if local_hit.is_hit() and local_hit.t < hit.t:
                    hit.t = local_hit.t
                    hit.u = local_hit.u
                    hit.v = local_hit.v
                    hit.prim = local_hit.prim
                    hit.inst = inst_idx
                    hit.occluded = UInt32(0)

        return False

    @always_inline
    def _traverse_triangles_generic[
        mode: String,
        blas_width: Int,
    ](
        self,
        ray: Ray,
        blases: UnsafePointer[TriangleBvh[blas_width], MutAnyOrigin],
    ) -> Hit:
        @always_inline
        def leaf_fn(
            ray: Ray,
            first_item: UInt32,
            item_count: UInt32,
            mut hit: Hit,
        ) capturing -> Bool:
            return self._traverse_triangle_leaf[
                mode,
                blas_width,
            ](
                ray,
                first_item,
                item_count,
                blases,
                hit,
            )

        return traverse_wide_ray_bvh[
            Self.width,
            mode,
            leaf_fn,
        ](
            self.tree,
            ray,
        )

    @always_inline
    def _traverse_spheres_generic[
        mode: String,
        blas_width: Int,
    ](
        self,
        ray: Ray,
        blases: UnsafePointer[SphereBvh[blas_width], MutAnyOrigin],
    ) -> Hit:
        @always_inline
        def leaf_fn(
            ray: Ray,
            first_item: UInt32,
            item_count: UInt32,
            mut hit: Hit,
        ) capturing -> Bool:
            return self._traverse_sphere_leaf[
                mode,
                blas_width,
            ](
                ray,
                first_item,
                item_count,
                blases,
                hit,
            )

        return traverse_wide_ray_bvh[
            Self.width,
            mode,
            leaf_fn,
        ](
            self.tree,
            ray,
        )

    def traverse_triangles[
        blas_width: Int
    ](
        self,
        ray: Ray,
        blases: UnsafePointer[TriangleBvh[blas_width], MutAnyOrigin],
    ) -> Hit:
        return self._traverse_triangles_generic[TRACE_CLOSEST_HIT, blas_width](
            ray,
            blases,
        )

    def is_occluded_triangles[
        blas_width: Int
    ](
        self,
        ray: Ray,
        blases: UnsafePointer[TriangleBvh[blas_width], MutAnyOrigin],
    ) -> Bool:
        var hit = self._traverse_triangles_generic[TRACE_ANY_HIT, blas_width](
            ray,
            blases,
        )
        return hit.is_occluded()

    def traverse_spheres[
        blas_width: Int
    ](
        self,
        ray: Ray,
        blases: UnsafePointer[SphereBvh[blas_width], MutAnyOrigin],
    ) -> Hit:
        return self._traverse_spheres_generic[TRACE_CLOSEST_HIT, blas_width](
            ray,
            blases,
        )

    def is_occluded_spheres[
        blas_width: Int
    ](
        self,
        ray: Ray,
        blases: UnsafePointer[SphereBvh[blas_width], MutAnyOrigin],
    ) -> Bool:
        var hit = self._traverse_spheres_generic[TRACE_ANY_HIT, blas_width](
            ray,
            blases,
        )
        return hit.is_occluded()
