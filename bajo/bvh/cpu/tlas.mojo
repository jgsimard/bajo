from bajo.core.aabb import AABB
from bajo.core.vec import Vec3
from bajo.bvh.types import Ray, Hit, Instance, TypedBvh
from bajo.bvh.constants import TRACE
from bajo.bvh.cpu.triangle_bvh import TriangleBvh
from bajo.bvh.cpu.sphere_bvh import SphereBvh
from bajo.bvh.cpu.bounds_bvh import BoundsBvh, BoundsBvhBuilder, BoundsItem
from bajo.bvh.cpu.trace import trace_bounds_bvh


def _tree[
    width: Int, split_method: String
](instances: List[Instance]) -> BoundsBvh[width]:
    var builder = BoundsBvhBuilder[width](
        [BoundsItem(inst.bounds, UInt32(i)) for i, inst in enumerate(instances)]
    )
    builder.build[split_method]()
    return BoundsBvh[width](builder)


struct Tlas[width: Int](Copyable):
    """Wide TLAS over Instance records."""

    var tree: BoundsBvh[Self.width]
    var instances: List[Instance]
    var inst_count: Int

    def __init__(out self, instances: List[Instance]):
        self.instances = instances.copy()
        self.inst_count = len(self.instances)
        self.tree = _tree[self.width, "lbvh"](instances)

    def add_instance(mut self, instance: Instance):
        self.instances.append(instance.copy())
        self.inst_count += 1

    def build[split_method: String](mut self):
        self.tree = _tree[self.width, "lbvh"](self.instances)

    def bounds(self) -> AABB:
        return self.tree.root_bounds()

    def trace[
        typed_bvh: TypedBvh,
        mode: TRACE,
    ](self, ray: Ray, blases: UnsafePointer[typed_bvh, MutAnyOrigin]) -> Hit:
        def leaf_fn(
            ray: Ray,
            O: Vec3[DType.float32, Self.width],
            D: Vec3[DType.float32, Self.width],
            first_item: UInt32,
            item_count: UInt32,
            mut hit: Hit,
        ) capturing -> Bool:
            for i in range(Int(item_count)):
                var item_ref = Int(self.tree.item_indices[Int(first_item) + i])
                var inst_idx = self.tree.item_payloads[item_ref]
                ref inst = self.instances[Int(inst_idx)]
                ref transform = inst.inv_transform

                var local_origin = transform.point(ray.o)
                var local_dir = transform.vector(ray.d)

                var local_ray = Ray(local_origin, local_dir, ray.t_min, hit.t)

                var local_hit = blases[Int(inst.blas_idx)].trace[mode](
                    local_ray
                )

                comptime if mode == TRACE.ANY_HIT:
                    if local_hit.is_occluded():
                        return True
                else:
                    if local_hit.is_hit() and local_hit.t < hit.t:
                        hit.t = local_hit.t
                        hit.u = local_hit.u
                        hit.v = local_hit.v
                        hit.prim = local_hit.prim
                        hit.inst = inst_idx

            return False

        return trace_bounds_bvh[
            Self.width,
            mode,
            leaf_fn,
        ](
            self.tree,
            ray,
        )
