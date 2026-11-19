"""Geometry-owned alpha acceptance inside CPU BLAS and TLAS traversal."""

from layout import Coord, Idx, TileTensor, row_major
from bajo.core import Normal3, Affine3, Point3, Ray, Rayf32, Vec3, dot
from std.math import sqrt
from bajo.rt.opacity import (
    opacity_accepts,
    opacity_ray_sample,
    opacity_sphere_uv,
)
from bajo.bvh.constants import EMPTY_LANE, TraceMode
from bajo.bvh.filter import TriangleHitFilter
from bajo.bvh.types import BlasDesc, SphereLeafBlock, Hit
from bajo.bvh.cpu.blas_storage import CpuBlasSet
from bajo.bvh.cpu.filtered import (
    trace_filtered_triangle_blas,
    trace_filtered_triangle_blas_packet,
)
from bajo.bvh.cpu.packet import trace_packet_stack_bounds_bvh
from bajo.bvh.cpu.tlas import _transform_world_ray_packet
from bajo.bvh.cpu.tlas import CpuTlas
from bajo.bvh.cpu.trace import trace_bounds_bvh
from bajo.bvh.tlas_common import (
    promote_tlas_local_hit,
    finalize_tlas_hit_normal,
)
from bajo.rt.scene_data import SceneData
from bajo.rt.material_types import NO_TEXTURE, PrimitiveId


@fieldwise_init
struct _CpuOpacityFilter[origin: ImmOrigin, length: SIMDLength = 1](
    Copyable, TriangleHitFilter
):
    var data: ImmPointer[SceneData, Self.origin]
    var shape_key: UInt32
    var mesh: UInt32
    var sample: SIMD[.float32, Self.length]

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
                    var texcoords = TileTensor(
                        ImmSpan(uv),
                        row_major(Coord(len(uv) // 6, Idx[3], Idx[2])),
                    )[Int(primitive), :, :]
                    var w = 1.0 - u - v
                    tu = (
                        texcoords[0, 0] * w
                        + texcoords[1, 0] * u
                        + texcoords[2, 0] * v
                    )
                    tv = (
                        texcoords[0, 1] * w
                        + texcoords[1, 1] * u
                        + texcoords[2, 1] * v
                    )
            return opacity_accepts(
                opacity.evaluate(data.surfaces(), tu, tv), self.sample[0]
            )
        return True

    def accept_packet[
        width: SIMDLength
    ](
        self,
        primitive: UInt32,
        u: SIMD[.float32, width],
        v: SIMD[.float32, width],
        active: SIMD[.bool, width],
    ) -> SIMD[.bool, width]:
        comptime assert width == Self.length
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
                    var texcoords = TileTensor(
                        ImmSpan(uv),
                        row_major(Coord(len(uv) // 6, Idx[3], Idx[2])),
                    )[Int(primitive), :, :]
                    var w = 1 - u - v
                    tu = (
                        texcoords[0, 0] * w
                        + texcoords[1, 0] * u
                        + texcoords[2, 0] * v
                    )
                    tv = (
                        texcoords[0, 1] * w
                        + texcoords[1, 1] * u
                        + texcoords[2, 1] * v
                    )
            return active & opacity_accepts(
                opacity.evaluate(
                    data.surfaces(), active.select(tu, 0), active.select(tv, 0)
                ),
                rebind[SIMD[.float32, width]](self.sample),
            )
        return active


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


def trace_opacity_spheres[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    mode: TraceMode,
    length: SIMDLength = 1,
](
    data: SceneData,
    blases: CpuBlasSet[.SPHERE, node_width, leaf_width],
    rays: Ray[.float32, .WORLD, length],
    valid: SIMD[.bool, length] = SIMD[.bool, length](fill=True),
) -> Hit[.WORLD, length]:
    """Accelerated alpha spheres, testing both roots before hit reduction."""
    var hit = Hit[.WORLD, length].miss(rays.t_max)
    if not valid.reduce_or():
        return hit
    var desc = BlasDesc.load(blases.descs.unsafe_ptr(), UInt32(0))
    var view = blases._view[.WORLD](desc)
    var sample = opacity_ray_sample(rays)
    var a = dot(rays.d, rays.d)

    def leaf_fn(
        active: SIMD[.bool, length],
        block_idx: UInt32,
        mut result: Hit[.WORLD, length],
    ) {imm}:
        var block = SphereLeafBlock[.WORLD, leaf_width].load(
            view.leaves, block_idx
        )
        for lane in range(Int(leaf_width)):
            var primitive = block.prim_indices[lane]
            if primitive == EMPTY_LANE:
                continue
            var center = Point3[.float32, .WORLD, length](
                block.center.x[lane], block.center.y[lane], block.center.z[lane]
            )
            var radius = block.radius[lane]
            var offset = rays.o - center
            var b = dot(offset, rays.d)
            var c = dot(offset, offset) - radius * radius
            var discriminant = b * b - a * c
            var root = sqrt(max(discriminant, SIMD[.float32, length](0)))
            for side in range(2):
                var t = (-b + (root if side else -root)) / a
                var accepted = (
                    active
                    & discriminant.ge(0)
                    & t.gt(rays.t_min)
                    & t.lt(result.t)
                )
                comptime if mode == .ANY_HIT:
                    accepted &= result.t.ne(0)
                if not accepted.reduce_or():
                    continue
                var normal = (rays.at(t) - center) / radius
                var key = PrimitiveId(.SPHERE, primitive).value
                if case .Some(opacity) = data.opacity().records.get(key):
                    var uv = opacity_sphere_uv(normal)
                    accepted &= opacity_accepts(
                        opacity.evaluate(
                            data.surfaces(),
                            accepted.select(uv[0], 0),
                            accepted.select(uv[1], 0),
                        ),
                        sample,
                    )
                comptime if mode == .ANY_HIT:
                    result.t = accepted.select(Float32(0), result.t)
                else:
                    result.t = accepted.select(t, result.t)
                    result.prim = accepted.select(primitive, result.prim)
                    result.normal = Normal3.select(
                        accepted,
                        normal.unsafe_convert[new_kind=.NORMAL](),
                        result.normal,
                    )

    trace_packet_stack_bounds_bvh[
        frame=.WORLD,
        bounds_width=node_width,
        length=length,
        packed_meta=True,
        any_hit=mode == .ANY_HIT,
    ](
        view.nodes,
        rays,
        rays.reciprocal_direction(),
        valid,
        hit,
        leaf_fn,
        lambda (
            _active: SIMD[.bool, length],
            _child: UInt32,
            mut _hit: Hit[.WORLD, length],
        ): None,
        lambda (_child: UInt32): None,
    )
    return hit


def trace_opacity_triangles_packet[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    mode: TraceMode,
    length: SIMDLength,
](
    data: SceneData,
    blases: CpuBlasSet[.TRIANGLE, node_width, leaf_width],
    rays: Ray[.float32, .WORLD, length],
    valid: SIMD[.bool, length],
) -> Hit[.WORLD, length]:
    var filter = _CpuOpacityFilter(
        Pointer(to=data), UInt32.MAX, NO_TEXTURE, opacity_ray_sample(rays)
    )
    return trace_filtered_triangle_blas_packet[
        .WORLD, node_width, leaf_width, mode, length
    ](blases, 0, rays, filter, valid)


def trace_opacity_instances_packet[
    node_width: SIMDLength,
    blas_width: SIMDLength,
    mode: TraceMode,
    length: SIMDLength,
](
    data: SceneData,
    tlas: CpuTlas[node_width, 1],
    blases: CpuBlasSet[.TRIANGLE, blas_width, blas_width],
    rays: Ray[.float32, .WORLD, length],
    valid: SIMD[.bool, length],
) -> Hit[.WORLD, length]:
    var hit = Hit[.WORLD, length].miss(rays.t_max)
    var sample = opacity_ray_sample(rays)

    def leaf_fn(
        active: SIMD[.bool, length],
        block_idx: UInt32,
        mut result: Hit[.WORLD, length],
    ) {imm}:
        var index = tlas.leaf_blocks[Int(block_idx)].inst_indices[0]
        if index == EMPTY_LANE:
            return
        ref instance = data.triangle_instances()[Int(index)]
        var local_rays = _transform_world_ray_packet(
            instance.inv_transform, rays, result.t
        )
        var filter = _CpuOpacityFilter(
            Pointer(to=data),
            PrimitiveId(.TRIANGLE_INSTANCE, index).value,
            instance.blas_idx,
            sample,
        )
        var local = trace_filtered_triangle_blas_packet[
            .LOCAL, blas_width, blas_width, mode, length
        ](blases, instance.blas_idx, local_rays, filter, active)
        comptime if mode == .ANY_HIT:
            result.t = (active & local.t.eq(0)).select(Float32(0), result.t)
        else:
            var accepted = active & local.is_hit() & local.t.lt(result.t)
            if not accepted.reduce_or():
                return
            ref m = instance.inv_transform
            var inverse = Affine3[.float32, .WORLD, .LOCAL, length](
                m.m00,
                m.m01,
                m.m02,
                0,
                m.m10,
                m.m11,
                m.m12,
                0,
                m.m20,
                m.m21,
                m.m22,
                0,
            )
            var normal = Affine3[
                .float32, .LOCAL, .WORLD, length
            ].normal_from_inverse(local.normal, inverse)
            result.t = accepted.select(local.t, result.t)
            result.u = accepted.select(local.u, result.u)
            result.v = accepted.select(local.v, result.v)
            result.prim = accepted.select(local.prim, result.prim)
            result.inst = accepted.select(index, result.inst)
            result.normal = Normal3.select(accepted, normal, result.normal)

    trace_packet_stack_bounds_bvh[
        frame=.WORLD,
        bounds_width=node_width,
        length=length,
        any_hit=mode == .ANY_HIT,
    ](
        ImmSpan(tlas.tree.nodes),
        rays,
        rays.reciprocal_direction(),
        valid,
        hit,
        leaf_fn,
        lambda (
            _active: SIMD[.bool, length],
            _child: UInt32,
            mut _hit: Hit[.WORLD, length],
        ): None,
        lambda (_child: UInt32): None,
    )
    return hit
