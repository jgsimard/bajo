"""GPU opacity evaluation and filtered primitive/TLAS traversal."""

from std.math import acos, atan2, min, pi, sqrt
from bajo.core import Affine3f32, GeoKind, Point3f32, Rayf32, Vec3f32, dot
from bajo.core.tensor_layout import record_tile
from bajo.bvh.constants import EMPTY_LANE, SPHERE_LEAF_PACKED_STRIDE, TraceMode
from bajo.bvh.filter import TriangleHitFilter
from bajo.bvh.types import BlasDesc, Hit
from bajo.bvh.gpu.filtered import trace_gpu_blas_filtered
from bajo.bvh.gpu.trace import trace_bounds_bvh_state
from bajo.bvh.tlas_common import (
    promote_tlas_local_hit,
    finalize_tlas_hit_normal,
)
from bajo.rt.gpu.config import GpuRtBvhFormat
from bajo.rt.gpu.views import (
    GpuOpacityView,
    GpuRtInstanceView,
    GpuRtSceneView,
    GpuRtSphereView,
)
from bajo.rt.material_types import NO_TEXTURE, PrimitiveId
from bajo.rt.opacity import opacity_accepts, opacity_ray_sample
from bajo.rt.texture_math import sample_scalar_texture_bilinear


@always_inline
def gpu_opacity_value(
    view: GpuOpacityView, key: UInt32, u: Float32, v: Float32
) -> Float32:
    var begin = 0
    var end = Int(view.count)
    while begin < end:
        var middle = (begin + end) // 2
        if view.keys[unsafe_offset=middle] < key:
            begin = middle + 1
        else:
            end = middle
    if begin == Int(view.count) or view.keys[unsafe_offset=begin] != key:
        return 1.0
    var base = 3 * begin
    var alpha = view.parameters[unsafe_offset=base]
    var offset = view.descriptors[unsafe_offset=base]
    if offset != NO_TEXTURE:
        var width = Int(view.descriptors[unsafe_offset=base + 1])
        var height = Int(view.descriptors[unsafe_offset=base + 2])
        alpha *= sample_scalar_texture_bilinear(
            record_tile[.float32, 1](
                view.pixels.unsafe_offset(Int(offset)), width * height
            ),
            width,
            height,
            u * view.parameters[unsafe_offset=base + 1],
            v * view.parameters[unsafe_offset=base + 2],
        )
    return alpha.clamp(0.0, 1.0)


@fieldwise_init
struct _GpuOpacityFilter(TriangleHitFilter, TrivialRegisterPassable):
    var view: GpuOpacityView
    var shape_key: UInt32
    var texcoord_offset: UInt32
    var texcoords: ImmPointer[Float32, ImmUntrackedOrigin]
    var sample: Float32

    @always_inline
    def accept(self, primitive: UInt32, u: Float32, v: Float32) -> Bool:
        var key = self.shape_key
        if key == UInt32.MAX:
            key = PrimitiveId(.TRIANGLE, primitive).value
        var tu = u + v
        var tv = v
        if self.texcoord_offset != NO_TEXTURE:
            var base = Int(self.texcoord_offset) + 6 * Int(primitive)
            var w = 1.0 - u - v
            tu = (
                self.texcoords[unsafe_offset=base] * w
                + self.texcoords[unsafe_offset=base + 2] * u
                + self.texcoords[unsafe_offset=base + 4] * v
            )
            tv = (
                self.texcoords[unsafe_offset=base + 1] * w
                + self.texcoords[unsafe_offset=base + 3] * u
                + self.texcoords[unsafe_offset=base + 5] * v
            )
        return opacity_accepts(
            gpu_opacity_value(self.view, key, tu, tv), self.sample
        )


@always_inline
def trace_gpu_opacity_triangles[
    format: GpuRtBvhFormat, mode: TraceMode
](scene: GpuRtSceneView, ray: Rayf32[.WORLD],) -> Hit[.WORLD]:
    var triangles = scene.triangles.unsafe_value()
    var filter = _GpuOpacityFilter(
        scene.opacity,
        UInt32.MAX,
        NO_TEXTURE,
        scene.opacity.pixels,
        opacity_ray_sample(ray),
    )
    return trace_gpu_blas_filtered[
        .WORLD, format.node_width, format.leaf_width, mode, format.layout
    ](triangles.nodes, triangles.leaves, triangles.root, ray, filter)


@fieldwise_init
struct _OpacityTlasState(TrivialRegisterPassable):
    var instances: GpuRtInstanceView
    var opacity: GpuOpacityView
    var sample: Float32


@always_inline
def _opacity_instance_leaf[
    tlas_width: SIMDLength, blas_format: GpuRtBvhFormat, mode: TraceMode
](
    state: _OpacityTlasState,
    block: UInt32,
    count: UInt32,
    ray: Rayf32[.WORLD],
    mut hit: Hit[.WORLD],
) -> Bool:
    var found = False
    var transforms = ImmSpan(
        unsafe_ptr=state.instances.inv_transforms,
        length=Int(state.instances.count) * Affine3f32.STRIDE,
    )
    for lane in range(min(Int(tlas_width), Int(count))):
        var index = block
        comptime if tlas_width != 1:
            index = state.instances.tlas_leaf_instances[
                unsafe_offset=Int(block) * tlas_width + lane
            ]
        if index == EMPTY_LANE:
            continue
        var mesh = state.instances.blas_indices[unsafe_offset=Int(index)]
        var desc = BlasDesc.load(state.instances.blas_descs, mesh)
        if desc.prim_count == 0:
            continue
        var inverse = Affine3f32[.WORLD, .LOCAL].load(
            transforms, Int(index) * Affine3f32.STRIDE
        )
        var filter = _GpuOpacityFilter(
            state.opacity,
            PrimitiveId(.TRIANGLE_INSTANCE, index).value,
            state.instances.texcoord_offsets[unsafe_offset=Int(mesh)],
            state.instances.texcoords,
            state.sample,
        )
        var local = trace_gpu_blas_filtered[
            .LOCAL,
            blas_format.node_width,
            blas_format.leaf_width,
            mode,
            blas_format.layout,
        ](
            state.instances.blas_nodes.unsafe_offset(Int(desc.node_f32_base)),
            state.instances.blas_leaves.unsafe_offset(Int(desc.leaf_f32_base)),
            desc.root_idx,
            inverse.ray(ray, hit.t),
            filter,
        )
        comptime if mode == .ANY_HIT:
            if local.is_occluded():
                hit = Hit[.WORLD].shadow_hit()
                return True
        else:
            found |= promote_tlas_local_hit(local, index, hit)
    return found


@always_inline
def trace_gpu_opacity_instances[
    tlas_format: GpuRtBvhFormat, blas_format: GpuRtBvhFormat, mode: TraceMode
](scene: GpuRtSceneView, ray: Rayf32[.WORLD],) -> Hit[.WORLD]:
    var instances = scene.instances.unsafe_value()
    var state = _OpacityTlasState(
        instances, scene.opacity, opacity_ray_sample(ray)
    )
    var hit = trace_bounds_bvh_state[
        .WORLD,
        tlas_format.node_width,
        mode,
        _OpacityTlasState,
        _opacity_instance_leaf[tlas_format.leaf_width, blas_format, mode],
        tlas_format.node_width == 2,
    ](instances.tlas_nodes, state, instances.tlas_root, ray)
    comptime if mode == .CLOSEST_HIT:
        if hit.is_hit():
            var transforms = ImmSpan(
                unsafe_ptr=instances.inv_transforms,
                length=Int(instances.count) * Affine3f32.STRIDE,
            )
            var inverse = Affine3f32[.WORLD, .LOCAL].load(
                transforms, Int(hit.inst) * Affine3f32.STRIDE
            )
            finalize_tlas_hit_normal(hit, inverse)
    return hit


@fieldwise_init
struct _OpacitySphereState(TrivialRegisterPassable):
    var spheres: GpuRtSphereView
    var opacity: GpuOpacityView
    var sample: Float32


@always_inline
def _opacity_sphere_leaf[
    width: SIMDLength, mode: TraceMode
](
    state: _OpacitySphereState,
    block: UInt32,
    count: UInt32,
    ray: Rayf32[.WORLD],
    mut hit: Hit[.WORLD],
) -> Bool:
    var found = False
    var base = Int(block) * SPHERE_LEAF_PACKED_STRIDE * width
    var words = state.spheres.leaves.unsafe_bitcast[UInt32]()
    for lane in range(min(Int(width), Int(count))):
        var primitive = words[unsafe_offset=base + 4 * width + lane]
        if primitive == EMPTY_LANE:
            continue
        var center = Point3f32[.WORLD](
            state.spheres.leaves[unsafe_offset=base + lane],
            state.spheres.leaves[unsafe_offset=base + width + lane],
            state.spheres.leaves[unsafe_offset=base + 2 * width + lane],
        )
        var radius = state.spheres.leaves[unsafe_offset=base + 3 * width + lane]
        var offset = ray.o - center
        var a = dot(ray.d, ray.d)
        var b = dot(offset, ray.d)
        var discriminant = b * b - a * (dot(offset, offset) - radius * radius)
        if discriminant < 0.0:
            continue
        var root = sqrt(discriminant)
        for numerator in [-b - root, -b + root]:
            var t = numerator / a
            if t <= ray.t_min or t >= hit.t:
                continue
            var normal = (ray.at(t) - center) / radius
            var signed_normal = (
                normal if state.spheres.signed_radii[
                    unsafe_offset=Int(primitive)
                ]
                > 0.0 else -normal
            )
            var phi = atan2(signed_normal.y, signed_normal.x)
            if phi < 0.0:
                phi += Float32(2.0 * pi)
            var u = phi / Float32(2.0 * pi)
            var v = acos(signed_normal.z.clamp(-1.0, 1.0)) / Float32(pi)
            if not opacity_accepts(
                gpu_opacity_value(
                    state.opacity, PrimitiveId(.SPHERE, primitive).value, u, v
                ),
                state.sample,
            ):
                continue
            comptime if mode == .ANY_HIT:
                hit = Hit[.WORLD].shadow_hit()
                return True
            hit.t = t
            hit.prim = primitive
            hit.normal = normal.unsafe_convert[new_kind=GeoKind.NORMAL]()
            found = True
    return found


@always_inline
def trace_gpu_opacity_spheres[
    format: GpuRtBvhFormat, mode: TraceMode
](scene: GpuRtSceneView, ray: Rayf32[.WORLD]) -> Hit[.WORLD]:
    var spheres = scene.spheres.unsafe_value()
    var state = _OpacitySphereState(
        spheres, scene.opacity, opacity_ray_sample(ray)
    )
    return trace_bounds_bvh_state[
        .WORLD,
        format.node_width,
        mode,
        _OpacitySphereState,
        _opacity_sphere_leaf[format.leaf_width, mode],
        format.node_width == 2,
    ](spheres.nodes, state, spheres.root, ray)
