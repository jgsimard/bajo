"""Compile-time-specialized traversal kernel for all GPU RT scene shapes."""

from max.gpu import block_dim, global_idx, grid_dim
from std.math import ceildiv, pi
from max.gpu.host import DeviceContext

from bajo.bvh.constants import PrimitiveKind
from bajo.bvh.types import Hit
from bajo.rt.gpu.opacity import (
    gpu_opacity_value,
    trace_gpu_opacity_triangles,
    trace_gpu_opacity_instances,
    trace_gpu_opacity_spheres,
)
from bajo.bvh.gpu.sphere_bvh import _intersect_sphere_leaf
from bajo.bvh.gpu.tlas import _trace_tlas_ray
from bajo.bvh.gpu.trace import trace_bounds_bvh
from bajo.bvh.gpu.triangle_trace import _intersect_triangle_leaf
from bajo.bvh.gpu.blas_trace import trace_gpu_blas
from bajo.core import (
    Affine3f32,
    GeoKind,
    Point3f32,
    Rayf32,
    Vec3f32,
    dot,
    length,
)
from bajo.core.mappings import equal_area_sphere_to_square
from bajo.rt.common import path_stage_rng, sky_color
from bajo.core.geometry import (
    orient_surface_normal,
    interpolate_triangle_attribute,
)
from bajo.core.tensor_layout import record_tile
from bajo.rt.lighting import (
    _emissive_hit_weight_from_pdf,
    _emissive_hit_light_pdf,
)
from bajo.rt.gpu.emission import gpu_emitter
from bajo.rt.rays import make_ao_ray
from bajo.core.color import Color, luminance
from bajo.rt.render_types import SamplingConfig
from bajo.rt.gpu.sampling import GpuRtSamplingConfig
from bajo.core.optics import medium_transmittance
from bajo.rt.material_types import (
    PrimitiveId,
    NO_TEXTURE,
    Integrator,
    SurfaceId,
    sample_mix_branch,
)
from bajo.rt.lighting_types import _light_importance
from bajo.rt.opacity import (
    opacity_accepts,
    opacity_point_sample,
    opacity_sphere_uv,
)

from bajo.rt.gpu.config import GpuRtBvhFormat, GpuRtSceneKind
from bajo.rt.wavefront_contract import (
    DeviceWavePath,
    DeviceWaveShadow,
    WAVE_COUNTER,
    wavefront_rng_material_stage,
)
from bajo.rt.gpu.wavefront_contract import (
    load_gpu_rt_path,
    load_gpu_rt_shadow,
    store_gpu_rt_path_values,
)
from bajo.rt.gpu.common_kernels import (
    GPU_RT_MAX_BLOCKS,
    clear_gpu_rt_sample,
    make_gpu_rt_primary_path,
    gpu_rt_bounce_block_limit,
)
from bajo.rt.gpu.lighting import _sample_direct_light_candidate
from bajo.rt.gpu.textures import (
    _GpuImageTextureView,
    _gpu_normal_map,
    _gpu_coated_shading_normal,
    _gpu_dielectric_shading_normal,
    _sample_gpu_image_texture,
)
from bajo.rt.gpu.path_shading import (
    _accumulate_sample,
    _append_shadow,
    _append_direct_light_shadow,
    _route_surface_hit,
)
from bajo.rt.gpu.queue_order import (
    GpuRtQueueOrderScratch,
    gpu_rt_queue_order_enabled,
)
from bajo.rt.gpu.views import GpuRtSceneView, GpuRtTraceQueueView
from bajo.rt.gpu.spatial_order import gpu_rt_spatial_order_enabled
from bajo.rt.gpu.light_order import enqueue_gpu_rt_light_order


from bajo.rt.shading_types import SurfaceHit
from bajo.rt.gpu.scene_trace import GPU_RT_LARGE_QUEUE_MIN_CAPACITY


@always_inline
def spectral_trace_surface[
    integrator: Integrator,
    scene_kind: GpuRtSceneKind,
    sphere_format: GpuRtBvhFormat,
    triangle_format: GpuRtBvhFormat,
    tlas_format: GpuRtBvhFormat,
    blas_format: GpuRtBvhFormat,
](
    scene: GpuRtSceneView,
    ray: Rayf32[.WORLD],
    sampling: SamplingConfig,
    path_id: UInt32,
    bounce: UInt32,
    capacity: Int,
) -> Tuple[SurfaceHit[1], UInt32]:
    var texture_descs = scene.texture_descs.unsafe_origin_cast[ImmutAnyOrigin]()
    var texture_pixels = scene.texture_pixels.unsafe_origin_cast[
        ImmutAnyOrigin
    ]()
    var coated_diffuses = scene.coated_diffuses.unsafe_origin_cast[
        ImmutAnyOrigin
    ]()
    var coated_diffuse_texture_indices = (
        scene.coated_diffuse_texture_indices.unsafe_origin_cast[
            ImmutAnyOrigin
        ]()
    )
    var medium_boundary = UInt32(0)
    var geometric_outward = Vec3f32[.WORLD](0.0)
    var found = False
    var dpdu = Vec3f32[.WORLD](0.0)
    var closest_t = ray.t_max
    var outward = Vec3f32[.WORLD](0.0)
    var surface_value = UInt32(0)
    var primitive_value = UInt32.MAX
    var uv_u = Float32(0.0)
    var uv_v = Float32(0.0)

    comptime if scene_kind.has_spheres():
        debug_assert["safe", _use_compiler_assume=True](Bool(scene.spheres))
        var spheres = scene.spheres.unsafe_value()
        var sphere_nodes = spheres.nodes.unsafe_origin_cast[ImmutAnyOrigin]()
        var leaf_spheres = spheres.leaves.unsafe_origin_cast[ImmutAnyOrigin]()
        var sphere_root = spheres.root
        var sphere_surfaces = spheres.surfaces.unsafe_origin_cast[
            ImmutAnyOrigin
        ]()
        var signed_radii = spheres.signed_radii.unsafe_origin_cast[
            ImmutAnyOrigin
        ]()
        var sphere_hit: Hit[.WORLD]
        if scene.opacity.count > 0:
            sphere_hit = trace_gpu_opacity_spheres[sphere_format, .CLOSEST_HIT](
                scene, ray
            )
        else:
            sphere_hit = trace_bounds_bvh[
                .WORLD,
                sphere_format.node_width,
                .CLOSEST_HIT,
                _intersect_sphere_leaf[
                    .WORLD, sphere_format.leaf_width, .CLOSEST_HIT
                ],
                sphere_format.node_width == 2,
            ](sphere_nodes, leaf_spheres, sphere_root, ray)
        if sphere_hit.is_hit():
            found = True
            closest_t = sphere_hit.t
            outward = sphere_hit.normal.unsafe_convert[
                new_kind=GeoKind.VECTOR
            ]()
            var sphere_idx = Int(sphere_hit.prim)
            if signed_radii[unsafe_offset=sphere_idx] < 0.0:
                outward = -outward
            geometric_outward = outward
            primitive_value = PrimitiveId(.SPHERE, sphere_hit.prim).value
            var sphere_uv = opacity_sphere_uv(outward)
            uv_u = sphere_uv[0]
            uv_v = sphere_uv[1]
            medium_boundary = scene.sphere_media[
                unsafe_offset=sphere_idx
            ] if sphere_idx < Int(scene.sphere_media_count) else UInt32(0)
            surface_value = sphere_surfaces[unsafe_offset=sphere_idx]

    comptime if scene_kind.has_triangles():
        debug_assert["safe", _use_compiler_assume=True](Bool(scene.triangles))
        var triangles = scene.triangles.unsafe_value()
        var triangle_nodes = triangles.nodes.unsafe_origin_cast[
            ImmutAnyOrigin
        ]()
        var leaf_vertices = triangles.leaves.unsafe_origin_cast[
            ImmutAnyOrigin
        ]()
        var triangle_root = triangles.root
        var triangle_surfaces = triangles.surfaces.unsafe_origin_cast[
            ImmutAnyOrigin
        ]()
        var triangle_ray = Rayf32[.WORLD](ray.o, ray.d, ray.t_min, closest_t)
        # A separate GPU function avoids slow fused traversal/shading code
        # for large static MIS queues, without additional dispatch variants.
        var triangle_hit: Hit[.WORLD]
        if scene.opacity.count > 0:
            triangle_hit = trace_gpu_opacity_triangles[
                triangle_format, .CLOSEST_HIT
            ](scene, triangle_ray)
        else:
            triangle_hit = trace_gpu_blas[
                .WORLD,
                triangle_format.node_width,
                .CLOSEST_HIT,
                _intersect_triangle_leaf[
                    .WORLD,
                    triangle_format.leaf_width,
                    .CLOSEST_HIT,
                    triangle_format.leaf_width > triangle_format.node_width
                    or triangle_format.leaf_width == 8,
                ],
                triangle_format.layout,
                triangle_format.node_width == 4,
            ](
                triangle_nodes,
                leaf_vertices,
                triangle_root,
                triangle_ray,
                scene_kind == .TRIANGLES
                and integrator == .MIS
                and capacity >= GPU_RT_LARGE_QUEUE_MIN_CAPACITY,
            )
        if triangle_hit.is_hit() and triangle_hit.t < closest_t:
            found = True
            closest_t = triangle_hit.t
            outward = triangle_hit.normal.unsafe_convert[
                new_kind=GeoKind.VECTOR
            ]()
            geometric_outward = outward
            var triangle_idx = Int(triangle_hit.prim)
            primitive_value = PrimitiveId(.TRIANGLE, triangle_hit.prim).value
            uv_u = triangle_hit.u + triangle_hit.v
            uv_v = triangle_hit.v
            medium_boundary = scene.triangle_media[
                unsafe_offset=triangle_idx
            ] if triangle_idx < Int(scene.triangle_media_count) else UInt32(0)
            surface_value = triangle_surfaces[
                unsafe_offset=Int(triangle_hit.prim)
            ]

    comptime if scene_kind.has_instances():
        debug_assert["safe", _use_compiler_assume=True](Bool(scene.instances))
        var instances = scene.instances.unsafe_value()
        var tlas_nodes = instances.tlas_nodes.unsafe_origin_cast[
            ImmutAnyOrigin
        ]()
        var tlas_leaf_instances = (
            instances.tlas_leaf_instances.unsafe_origin_cast[ImmutAnyOrigin]()
        )
        var inst_inv_transform = instances.inv_transforms.unsafe_origin_cast[
            ImmutAnyOrigin
        ]()
        var inst_blas_indices = instances.blas_indices.unsafe_origin_cast[
            ImmutAnyOrigin
        ]()
        var blas_descs = instances.blas_descs.unsafe_origin_cast[
            ImmutAnyOrigin
        ]()
        var blas_nodes = instances.blas_nodes.unsafe_origin_cast[
            ImmutAnyOrigin
        ]()
        var blas_leaves = instances.blas_leaves.unsafe_origin_cast[
            ImmutAnyOrigin
        ]()
        var normal_offsets = instances.normal_offsets.unsafe_origin_cast[
            ImmutAnyOrigin
        ]()
        var vertex_normals = instances.normals.unsafe_origin_cast[
            ImmutAnyOrigin
        ]()
        var texcoord_offsets = instances.texcoord_offsets.unsafe_origin_cast[
            ImmutAnyOrigin
        ]()
        var vertex_texcoords = instances.texcoords.unsafe_origin_cast[
            ImmutAnyOrigin
        ]()
        var tlas_root = instances.tlas_root
        var instance_count_i32 = instances.count
        var instance_surfaces = instances.surfaces.unsafe_origin_cast[
            ImmutAnyOrigin
        ]()
        var instance_ray = Rayf32[.WORLD](ray.o, ray.d, ray.t_min, closest_t)
        var instance_hit: Hit[.WORLD]
        if scene.opacity.count > 0:
            instance_hit = trace_gpu_opacity_instances[
                tlas_format, blas_format, .CLOSEST_HIT
            ](scene, instance_ray)
        else:
            instance_hit = _trace_tlas_ray[
                tlas_format.node_width,
                tlas_format.leaf_width,
                blas_format.node_width,
                blas_format.leaf_width,
                .CLOSEST_HIT,
                _intersect_triangle_leaf[
                    .LOCAL,
                    blas_format.leaf_width,
                    .CLOSEST_HIT,
                    blas_format.leaf_width > blas_format.node_width
                    or blas_format.leaf_width == 8,
                ],
                blas_format.layout,
            ](
                tlas_nodes,
                tlas_leaf_instances,
                inst_inv_transform,
                inst_blas_indices,
                blas_descs,
                blas_nodes,
                blas_leaves,
                Int(instance_count_i32),
                Int(instances.blas_count),
                tlas_root,
                instance_ray,
            )
        if instance_hit.is_hit() and instance_hit.t < closest_t:
            found = True
            closest_t = instance_hit.t
            outward = instance_hit.normal.unsafe_convert[
                new_kind=GeoKind.VECTOR
            ]()
            geometric_outward = outward
            var instance_idx = Int(instance_hit.inst)
            primitive_value = PrimitiveId(
                .TRIANGLE_INSTANCE, instance_hit.inst
            ).value
            uv_u = instance_hit.u + instance_hit.v
            uv_v = instance_hit.v
            medium_boundary = scene.instance_media[
                unsafe_offset=instance_idx
            ] if instance_idx < Int(scene.instance_media_count) else UInt32(0)
            var mesh_idx = Int(inst_blas_indices[unsafe_offset=instance_idx])
            var triangle_idx = Int(instance_hit.prim)
            var inverse_span = Span(
                unsafe_ptr=inst_inv_transform,
                length=Int(instance_count_i32) * Affine3f32.STRIDE,
            )
            var inverse = Affine3f32[.WORLD, .LOCAL].load(
                inverse_span, instance_idx * Affine3f32.STRIDE
            )
            var texcoord_offset = texcoord_offsets[unsafe_offset=mesh_idx]
            if texcoord_offset != UInt32.MAX:
                # Tangents use three floats per triangle; UVs use six.
                var tangent_offset = (
                    Int(texcoord_offset) // 2 + 3 * triangle_idx
                )
                var tangent_ptr = instances.tangents.unsafe_offset(
                    tangent_offset
                )
                var tangent = Vec3f32[.LOCAL](
                    tangent_ptr[unsafe_offset=0],
                    tangent_ptr[unsafe_offset=1],
                    tangent_ptr[unsafe_offset=2],
                )
                var forward = inverse.inverse()
                dpdu = forward.inv.vector(tangent)
                var records = record_tile[.float32, 6](
                    vertex_texcoords.unsafe_offset(Int(texcoord_offset)),
                    triangle_idx + 1,
                )
                uv_u = interpolate_triangle_attribute[2, 0](
                    records,
                    UInt32(triangle_idx),
                    instance_hit.u,
                    instance_hit.v,
                )
                uv_v = interpolate_triangle_attribute[2, 1](
                    records,
                    UInt32(triangle_idx),
                    instance_hit.u,
                    instance_hit.v,
                )
            var normal_offset = normal_offsets[unsafe_offset=mesh_idx]
            if normal_offset != UInt32.MAX:
                var records = record_tile[.float32, 9](
                    vertex_normals.unsafe_offset(Int(normal_offset)),
                    triangle_idx + 1,
                )
                var local_normal = Vec3f32[.LOCAL](
                    interpolate_triangle_attribute[3, 0](
                        records,
                        UInt32(triangle_idx),
                        instance_hit.u,
                        instance_hit.v,
                    ),
                    interpolate_triangle_attribute[3, 1](
                        records,
                        UInt32(triangle_idx),
                        instance_hit.u,
                        instance_hit.v,
                    ),
                    interpolate_triangle_attribute[3, 2](
                        records,
                        UInt32(triangle_idx),
                        instance_hit.u,
                        instance_hit.v,
                    ),
                ).unsafe_convert[new_kind=GeoKind.NORMAL]()
                var shading_normal = (
                    Affine3f32[.LOCAL, .WORLD]
                    .normal_from_inverse(local_normal, inverse)
                    .unsafe_convert[new_kind=GeoKind.VECTOR]()
                )
                if dot(shading_normal, outward) < 0.0:
                    shading_normal = -shading_normal
                outward = shading_normal
            surface_value = instance_surfaces[unsafe_offset=instance_idx]

    if not found:
        return SurfaceHit[1](ray.t_max), UInt32.MAX
    if SurfaceId.kind_from_raw(surface_value) == .MIX:
        var material_rng = path_stage_rng(
            sampling, path_id, wavefront_rng_material_stage(UInt32(bounce))
        )
        var random = material_rng.f32()
        while SurfaceId.kind_from_raw(surface_value) == .MIX:
            var index = Int(SurfaceId.index_from_raw(surface_value))
            var amount = scene.mix_amounts[unsafe_offset=index]
            var texture_index = scene.mix_texture_indices[unsafe_offset=index]
            if texture_index != NO_TEXTURE:
                var color = _sample_gpu_image_texture(
                    texture_index,
                    uv_u * scene.mix_uv_scales[unsafe_offset=2 * index],
                    uv_v * scene.mix_uv_scales[unsafe_offset=2 * index + 1],
                    texture_descs,
                    texture_pixels,
                )
                amount *= luminance(color)
            var branch = sample_mix_branch(amount.clamp(0.0, 1.0), random)
            surface_value = scene.mix_surfaces[unsafe_offset=2 * index + branch]
    var geometric = orient_surface_normal(ray.d, geometric_outward)
    var oriented = orient_surface_normal(ray.d, outward)
    var normal = oriented.normal
    normal = _gpu_normal_map(
        surface_value,
        normal,
        dpdu,
        uv_u,
        uv_v,
        geometric.front_face,
        scene.normal_maps.unsafe_origin_cast[ImmutAnyOrigin](),
        Int(scene.normal_map_count),
        texture_descs,
        texture_pixels,
    )
    normal = _gpu_coated_shading_normal(
        surface_value,
        normal,
        uv_u,
        uv_v,
        coated_diffuses,
        coated_diffuse_texture_indices,
        texture_descs,
        texture_pixels,
    )
    if SurfaceId.kind_from_raw(surface_value) == .DIELECTRIC:
        normal = _gpu_dielectric_shading_normal(
            surface_value,
            normal,
            uv_u,
            uv_v,
            scene.dielectrics.unsafe_origin_cast[ImmutAnyOrigin](),
            texture_descs,
            texture_pixels,
        )
    return (
        SurfaceHit[1](
            normal,
            SurfaceId.from_raw(surface_value),
            uv_u,
            uv_v,
            closest_t,
            geometric.front_face,
            True,
            medium_boundary,
            geometric.normal,
            dpdu,
        ),
        primitive_value,
    )
