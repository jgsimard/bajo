"""Geometry-owned alpha acceptance inside CPU BLAS and TLAS traversal."""

from bajo.core import Point3, Rayf32, Vec3, dot
from std.math import sqrt
from bajo.rt.opacity import (
    opacity_accepts,
    opacity_ray_sample,
    opacity_sphere_uv,
)
from bajo.bvh.constants import EMPTY_LANE, TraceMode
from bajo.bvh.filter import TriangleHitFilter
from bajo.bvh.types import Hit
from bajo.bvh.cpu.blas_storage import CpuBlasSet
from bajo.bvh.cpu.filtered import trace_filtered_triangle_blas
from bajo.bvh.cpu.tlas import CpuTlas
from bajo.bvh.cpu.trace import trace_bounds_bvh
from bajo.bvh.tlas_common import (
    promote_tlas_local_hit,
    finalize_tlas_hit_normal,
)
from bajo.rt.scene_data import SceneData
from bajo.rt.material_types import NO_TEXTURE, PrimitiveId


def trace_opacity_spheres[
    mode: TraceMode
](data: SceneData, ray: Rayf32[.WORLD]) -> Hit[.WORLD]:
    var hit = Hit[.WORLD].miss(ray.t_max)
    var sample = opacity_ray_sample(ray)
    for index in range(len(data.spheres())):
        ref sphere = data.spheres()[index]
        var offset = ray.o - sphere.center
        var a = dot(ray.d, ray.d)
        var b = dot(offset, ray.d)
        var c = dot(offset, offset) - sphere.radius * sphere.radius
        var discriminant = b * b - a * c
        if discriminant < 0.0:
            continue
        var root = sqrt(discriminant)
        for numerator in [-b - root, -b + root]:
            var t = numerator / a
            if t <= ray.t_min or t >= hit.t:
                continue
            var normal = (ray.at(t) - sphere.center) / sphere.radius
            var key = PrimitiveId(.SPHERE, UInt32(index)).value
            if case .Some(opacity) = data.opacity().records.get(key):
                var uv = opacity_sphere_uv(normal)
                if not opacity_accepts(
                    opacity.evaluate(data.surfaces(), uv[0], uv[1]), sample
                ):
                    continue
            comptime if mode == .ANY_HIT:
                return Hit[.WORLD].shadow_hit()
            hit.t = t
            hit.prim = UInt32(index)
            hit.normal = normal.unsafe_convert[new_kind=.NORMAL]()
    return hit


@fieldwise_init
struct _CpuOpacityFilter[origin: ImmOrigin](Copyable, TriangleHitFilter):
    var data: ImmPointer[SceneData, Self.origin]
    var shape_key: UInt32
    var mesh: UInt32
    var sample: Float32

    @always_inline
    def accept(self, primitive: UInt32, u: Float32, v: Float32) -> Bool:
        ref data = self.data[unsafe_offset=0]
        var key = self.shape_key
        if key == UInt32.MAX:
            key = PrimitiveId(.TRIANGLE, primitive).value
        if case .Some(opacity) = data.opacity().records.get(key):
            var tu = u + v
            var tv = v
            if self.mesh != NO_TEXTURE:
                ref uv = data.triangle_mesh_texcoords()[Int(self.mesh)]
                if len(uv):
                    var base = 6 * Int(primitive)
                    var w = 1.0 - u - v
                    tu = uv[base] * w + uv[base + 2] * u + uv[base + 4] * v
                    tv = uv[base + 1] * w + uv[base + 3] * u + uv[base + 5] * v
            return opacity_accepts(
                opacity.evaluate(data.surfaces(), tu, tv), self.sample
            )
        return True


def trace_opacity_triangles[
    node_width: SIMDLength, leaf_width: SIMDLength, mode: TraceMode
](
    data: SceneData,
    blases: CpuBlasSet[.TRIANGLE, node_width, leaf_width],
    ray: Rayf32[.WORLD],
) -> Hit[.WORLD]:
    var filter = _CpuOpacityFilter(
        Pointer(to=data), UInt32.MAX, NO_TEXTURE, opacity_ray_sample(ray)
    )
    return trace_filtered_triangle_blas[.WORLD, node_width, leaf_width, mode](
        blases, 0, ray, filter
    )


def trace_opacity_instances[
    node_width: SIMDLength, blas_width: SIMDLength, mode: TraceMode
](
    data: SceneData,
    tlas: CpuTlas[node_width, 1],
    blases: CpuBlasSet[.TRIANGLE, blas_width, blas_width],
    ray: Rayf32[.WORLD],
) -> Hit[.WORLD]:
    var sample = opacity_ray_sample(ray)

    @always_inline
    def leaf_fn(
        world_ray: Rayf32[.WORLD],
        O: Point3[.float32, .WORLD, 1],
        D: Vec3[.float32, .WORLD, 1],
        a: Float32,
        inv_a: Float32,
        block_idx: UInt32,
        mut hit: Hit[.WORLD],
    ) {imm} -> Bool:
        var instance_idx = tlas.leaf_blocks[Int(block_idx)].inst_indices[0]
        if instance_idx == EMPTY_LANE:
            return False
        ref instance = data.triangle_instances()[Int(instance_idx)]
        var local_ray = instance.inv_transform.ray(world_ray, hit.t)
        var filter = _CpuOpacityFilter(
            Pointer(to=data),
            PrimitiveId(.TRIANGLE_INSTANCE, instance_idx).value,
            instance.blas_idx,
            sample,
        )
        var local = trace_filtered_triangle_blas[
            .LOCAL, blas_width, blas_width, mode
        ](blases, instance.blas_idx, local_ray, filter)
        comptime if mode == .ANY_HIT:
            return local.is_occluded()
        else:
            return promote_tlas_local_hit(local, instance_idx, hit)

    var hit = trace_bounds_bvh[.WORLD, node_width, 1, mode](
        tlas.tree, ray, leaf_fn
    )
    comptime if mode == .CLOSEST_HIT:
        if hit.is_hit():
            finalize_tlas_hit_normal(
                hit, data.triangle_instances()[Int(hit.inst)].inv_transform
            )
    return hit
