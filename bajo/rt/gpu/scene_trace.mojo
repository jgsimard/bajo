"""Compile-time-specialized traversal kernel for all GPU RT scene shapes."""

from max.gpu import block_dim, global_idx, grid_dim
from std.math import sqrt, ceildiv, pi
from max.gpu.host import DeviceContext

from bajo.bvh.constants import PrimitiveKind
from bajo.bvh.gpu.sphere_bvh import _intersect_sphere_leaf
from bajo.bvh.gpu.tlas import _trace_tlas_ray
from bajo.bvh.gpu.trace import trace_bounds_bvh
from bajo.bvh.gpu.triangle_trace import (
    _intersect_triangle_leaf,
)
from bajo.bvh.gpu.blas_trace import trace_gpu_blas
from bajo.core import (
    Affine3f32,
    GeoKind,
    Point3f32,
    Rayf32,
    Vec3f32,
    dot,
)
from bajo.rt.common import (
    equal_area_sphere_to_square,
    path_stage_rng,
    sky_color,
)
from bajo.rt.geometry import (
    orient_surface_normal,
    interpolate_triangle_attribute,
)
from bajo.core.tensor_layout import record_tile
from bajo.rt.lighting import _emissive_hit_weight_from_pdf
from bajo.rt.rays import make_ao_ray
from bajo.rt.render_types import (
    Color,
    SamplingConfig,
)
from bajo.rt.media import medium_transmittance
from bajo.rt.material_types import (
    NO_TEXTURE,
    EnvironmentKind,
    Integrator,
    SurfaceId,
)
from bajo.rt.lighting_types import _light_importance
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
from bajo.rt.gpu.light_order import (
    enqueue_gpu_rt_light_order,
)


comptime GPU_RT_SHADOW_BLOCK_SIZE = 128
# Below this size, retain the launch policy tuned for the 256K working set.
comptime GPU_RT_LARGE_QUEUE_MIN_CAPACITY = 1 << 19
comptime GPU_RT_DIRECT_SHADOW_MAX_BLOCKS = 2048
comptime GPU_RT_DIRECT_WIDE_SHADOW_MAX_BLOCKS = 8192


@always_inline
def _gpu_environment_radiance(
    scene: GpuRtSceneView,
    texture_descs: Pointer[UInt32, ImmutAnyOrigin],
    texture_pixels: Pointer[Float32, ImmutAnyOrigin],
    direction: Vec3f32[.WORLD],
) -> Color:
    var scale = Color(
        scene.environment_scale_x,
        scene.environment_scale_y,
        scene.environment_scale_z,
    )
    match EnvironmentKind(scene.environment_kind):
        case .BLACK:
            return Color(0.0)
        case .PROCEDURAL:
            return sky_color(direction)
        case .UNIFORM:
            return scale
        case .IMAGE:
            pass

    var light_direction = scene.environment_world_to_light.vector(direction)
    var uv = equal_area_sphere_to_square(light_direction)
    var desc_base = 3 * Int(scene.environment_texture_index)
    var pixel_offset = Int(texture_descs[unsafe_offset=desc_base])
    var width = Int(texture_descs[unsafe_offset=desc_base + 1])
    var height = Int(texture_descs[unsafe_offset=desc_base + 2])
    var x = min(max(Int(uv.u * Float32(width)), 0), width - 1)
    var y = min(max(Int(uv.v * Float32(height)), 0), height - 1)
    var base = pixel_offset + 3 * (y * width + x)
    return scale * Color(
        max(texture_pixels[unsafe_offset=base], 0.0),
        max(texture_pixels[unsafe_offset=base + 1], 0.0),
        max(texture_pixels[unsafe_offset=base + 2], 0.0),
    )


@always_inline
def _gpu_rt_scene_trace_path[
    integrator: Integrator,
    scene_kind: GpuRtSceneKind,
    sphere_format: GpuRtBvhFormat,
    triangle_format: GpuRtBvhFormat,
    tlas_format: GpuRtBvhFormat,
    blas_format: GpuRtBvhFormat,
    light_kind: PrimitiveKind = .UNKNOWN,
](
    idx: Int,
    var path: DeviceWavePath,
    scene: GpuRtSceneView,
    queues: GpuRtTraceQueueView,
    sampling: SamplingConfig,
    bounce: UInt32,
):
    comptime assert integrator.is_valid()
    comptime assert (
        scene_kind.has_spheres()
        or scene_kind.has_triangles()
        or scene_kind.has_instances()
    )
    var emissives = scene.emissives.unsafe_origin_cast[ImmutAnyOrigin]()
    var lambertians = scene.lambertians.unsafe_origin_cast[ImmutAnyOrigin]()
    var metals = scene.metals.unsafe_origin_cast[ImmutAnyOrigin]()
    var coated_diffuses = scene.coated_diffuses.unsafe_origin_cast[
        ImmutAnyOrigin
    ]()
    var lambertian_texture_indices = (
        scene.lambertian_texture_indices.unsafe_origin_cast[ImmutAnyOrigin]()
    )
    var coated_diffuse_texture_indices = (
        scene.coated_diffuse_texture_indices.unsafe_origin_cast[
            ImmutAnyOrigin
        ]()
    )
    var texture_descs = scene.texture_descs.unsafe_origin_cast[ImmutAnyOrigin]()
    var texture_pixels = scene.texture_pixels.unsafe_origin_cast[
        ImmutAnyOrigin
    ]()
    var total_light_weight = scene.total_light_weight
    var dst_path_ids = queues.dst_path_ids.unsafe_origin_cast[MutAnyOrigin]()
    var dst_path_fields = queues.dst_path_fields.unsafe_origin_cast[
        MutAnyOrigin
    ]()
    var shade_path_refs = queues.shade_path_refs.unsafe_origin_cast[
        MutAnyOrigin
    ]()
    var shade_surfaces = queues.shade_surfaces.unsafe_origin_cast[
        MutAnyOrigin
    ]()
    var shade_fields = queues.shade_fields.unsafe_origin_cast[MutAnyOrigin]()
    var shadow_path_ids = queues.shadow_path_ids.unsafe_origin_cast[
        MutAnyOrigin
    ]()
    var shadow_fields = queues.shadow_fields.unsafe_origin_cast[MutAnyOrigin]()
    var counters = queues.counters.unsafe_origin_cast[MutAnyOrigin]()
    var sample_radiance = queues.sample_radiance.unsafe_origin_cast[
        MutAnyOrigin
    ]()
    var capacity = Int(queues.capacity)
    var sample_base = queues.sample_base
    var ray = Rayf32[.WORLD](
        Point3f32[.WORLD](path.ox, path.oy, path.oz),
        Vec3f32[.WORLD](path.dx, path.dy, path.dz),
        path.t_min,
        path.t_max,
    )
    var medium_boundary = UInt32(0)
    var geometric_outward = Vec3f32[.WORLD](0.0)
    var found = False
    var closest_t = ray.t_max
    var outward = Vec3f32[.WORLD](0.0)
    var surface_value = UInt32(0)
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
        var sphere_hit = trace_bounds_bvh[
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
        var triangle_hit = trace_gpu_blas[
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
        var instance_hit = _trace_tlas_ray[
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
            # Measured TLAS2/leaf1 + BLAS4/leaf4 policies share one callback.
            # Pure-instance shading benefits from a separate GPU function.
            (2 if scene_kind == .INSTANCES else 1) if tlas_format.node_width
            == 2
            and tlas_format.leaf_width == 1
            and not blas_format.layout.compressed
            and blas_format.node_width == 4
            and blas_format.leaf_width == 4 else 0,
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
            medium_boundary = scene.instance_media[
                unsafe_offset=instance_idx
            ] if instance_idx < Int(scene.instance_media_count) else UInt32(0)
            var mesh_idx = Int(inst_blas_indices[unsafe_offset=instance_idx])
            var triangle_idx = Int(instance_hit.prim)
            var texcoord_offset = texcoord_offsets[unsafe_offset=mesh_idx]
            if texcoord_offset != UInt32.MAX:
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
                var inverse_span = Span(
                    unsafe_ptr=inst_inv_transform,
                    length=Int(instance_count_i32) * Affine3f32.STRIDE,
                )
                var inverse = Affine3f32[.WORLD, .LOCAL].load(
                    inverse_span, instance_idx * Affine3f32.STRIDE
                )
                var shading_normal = (
                    Affine3f32[.LOCAL, .WORLD]
                    .normal_from_inverse(local_normal, inverse)
                    .unsafe_convert[new_kind=GeoKind.VECTOR]()
                )
                if dot(shading_normal, outward) < 0.0:
                    shading_normal = -shading_normal
                outward = shading_normal
            surface_value = instance_surfaces[unsafe_offset=instance_idx]

    if path.medium_id != 0:
        var base = 3 * Int(path.medium_id)
        var sigma_a = Color(scene.medium_absorption.unsafe_offset(base))
        var attenuation = medium_transmittance(
            sigma_a,
            (closest_t if found else ray.t_max) * sqrt(dot(ray.d, ray.d)),
        )
        path.tx *= attenuation.x
        path.ty *= attenuation.y
        path.tz *= attenuation.z
    if not found:
        comptime if Integrator.is_path_tracing[integrator]:
            var environment_emission = _gpu_environment_radiance(
                scene, texture_descs, texture_pixels, ray.d
            )
            var environment_light_pdf = Float32(0.0)
            comptime if integrator == .MIS:
                if bounce > 0 and not path.delta:
                    match EnvironmentKind(scene.environment_kind):
                        case .IMAGE:
                            environment_light_pdf = (
                                _light_importance(environment_emission)
                                / scene.total_light_weight if scene.total_light_weight
                                > 0.0 else 0.0
                            )
                        case .BLACK | .PROCEDURAL | .UNIFORM:
                            if scene.total_light_weight > 0.0:
                                environment_light_pdf = (
                                    scene.environment_weight
                                    / (4.0 * pi * scene.total_light_weight)
                                )
            var environment_mis_weight = _emissive_hit_weight_from_pdf[
                integrator
            ](
                bounce,
                path.delta,
                path.bsdf_pdf,
                environment_light_pdf,
            )
            _accumulate_sample(
                sample_radiance,
                capacity,
                sample_base,
                path.path_id,
                Color(path.tx, path.ty, path.tz)
                * environment_emission
                * environment_mis_weight,
            )
        return

    if SurfaceId.kind_from_raw(surface_value) == .MIX:
        var material_rng = path_stage_rng(
            sampling, path.path_id, wavefront_rng_material_stage(UInt32(bounce))
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
                amount *= 0.2126 * color.x + 0.7152 * color.y + 0.0722 * color.z
            amount = amount.clamp(0.0, 1.0)
            if random < amount:
                surface_value = scene.mix_surfaces[unsafe_offset=2 * index + 1]
                random /= amount
            else:
                surface_value = scene.mix_surfaces[unsafe_offset=2 * index]
                random = (random - amount) / (1.0 - amount)
    var geometric = orient_surface_normal(ray.d, geometric_outward)
    var oriented = orient_surface_normal(ray.d, outward)
    var normal = oriented.normal
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
    comptime if integrator == .AO:
        var ao_rng = path_stage_rng(sampling, path.path_id, UInt32(1))
        var ao_ray = make_ao_ray(ray.at(closest_t), normal, ao_rng)
        _ = _append_shadow[False](
            DeviceWaveShadow(
                path.path_id,
                ao_ray.o.x,
                ao_ray.o.y,
                ao_ray.o.z,
                ao_ray.t_min,
                ao_ray.d.x,
                ao_ray.d.y,
                ao_ray.d.z,
                ao_ray.t_max,
                0.0,
                0.0,
                0.0,
            ),
            shadow_path_ids,
            shadow_fields,
            counters,
            capacity,
        )
        return

    comptime if Integrator.uses_direct_lighting[integrator]:
        var direct = _sample_direct_light_candidate[integrator, light_kind](
            path,
            ray,
            closest_t,
            normal,
            uv_u,
            uv_v,
            surface_value,
            scene,
            sampling,
            bounce,
        )
        _append_direct_light_shadow(
            direct, path, ray.at(closest_t), scene, queues
        )

    _route_surface_hit[integrator](
        idx,
        path,
        ray.d,
        normal,
        geometric.front_face,
        uv_u,
        uv_v,
        closest_t,
        surface_value,
        bounce,
        total_light_weight,
        emissives,
        lambertians,
        lambertian_texture_indices,
        texture_descs,
        texture_pixels,
        coated_diffuses,
        coated_diffuse_texture_indices,
        dst_path_ids,
        dst_path_fields,
        shade_path_refs,
        shade_surfaces,
        shade_fields,
        counters,
        sample_radiance,
        capacity,
        sample_base,
        sampling,
        medium_boundary,
        geometric.normal,
    )


@always_inline
def _gpu_rt_scene_trace_one[
    integrator: Integrator,
    scene_kind: GpuRtSceneKind,
    sphere_format: GpuRtBvhFormat,
    triangle_format: GpuRtBvhFormat,
    tlas_format: GpuRtBvhFormat,
    blas_format: GpuRtBvhFormat,
    light_kind: PrimitiveKind = .UNKNOWN,
](
    idx: Int,
    scene: GpuRtSceneView,
    queues: GpuRtTraceQueueView,
    sampling: SamplingConfig,
    bounce: UInt32,
):
    var src_path_ids = queues.src_path_ids.unsafe_origin_cast[ImmutAnyOrigin]()
    var src_path_fields = queues.src_path_fields.unsafe_origin_cast[
        ImmutAnyOrigin
    ]()
    var path = load_gpu_rt_path[integrator](
        src_path_ids, src_path_fields, Int(queues.capacity), idx
    )
    _gpu_rt_scene_trace_path[
        integrator,
        scene_kind,
        sphere_format,
        triangle_format,
        tlas_format,
        blas_format,
        light_kind,
    ](idx, path, scene, queues, sampling, bounce)


def gpu_rt_primary_scene_trace_kernel[
    integrator: Integrator,
    scene_kind: GpuRtSceneKind,
    sphere_format: GpuRtBvhFormat,
    triangle_format: GpuRtBvhFormat,
    tlas_format: GpuRtBvhFormat,
    blas_format: GpuRtBvhFormat,
    store_source_path: Bool,
    light_kind: PrimitiveKind = .UNKNOWN,
](
    camera_params: Pointer[Float32, ImmutAnyOrigin],
    source_path_ids: Pointer[UInt32, MutAnyOrigin],
    source_path_fields: Pointer[Float32, MutAnyOrigin],
    scene: GpuRtSceneView,
    queues: GpuRtTraceQueueView,
    width_i32: Int32,
    height_i32: Int32,
    samples_per_pixel_i32: Int32,
    sampling: SamplingConfig,
):
    var counters = queues.counters.unsafe_origin_cast[MutAnyOrigin]()
    var active_count = Int(counters[unsafe_offset=WAVE_COUNTER.ACTIVE])
    var idx = global_idx.x
    if idx >= active_count:
        return
    var capacity = Int(queues.capacity)
    var path = make_gpu_rt_primary_path(
        camera_params,
        queues.sample_base,
        idx,
        width_i32,
        height_i32,
        samples_per_pixel_i32,
        sampling,
    )
    clear_gpu_rt_sample(
        queues.sample_radiance.unsafe_origin_cast[MutAnyOrigin](),
        capacity,
        idx,
    )
    comptime if store_source_path:
        store_gpu_rt_path_values[integrator](
            path,
            source_path_ids,
            source_path_fields,
            capacity,
            idx,
        )
    _gpu_rt_scene_trace_path[
        integrator,
        scene_kind,
        sphere_format,
        triangle_format,
        tlas_format,
        blas_format,
        light_kind,
    ](idx, path, scene, queues, sampling, UInt32(0))


def gpu_rt_scene_trace_kernel[
    integrator: Integrator,
    scene_kind: GpuRtSceneKind,
    sphere_format: GpuRtBvhFormat,
    triangle_format: GpuRtBvhFormat,
    tlas_format: GpuRtBvhFormat,
    blas_format: GpuRtBvhFormat,
    light_kind: PrimitiveKind = .UNKNOWN,
](
    scene: GpuRtSceneView,
    queues: GpuRtTraceQueueView,
    sampling: SamplingConfig,
    bounce: UInt32,
):
    var counters = queues.counters.unsafe_origin_cast[MutAnyOrigin]()
    var active_count = Int(counters[unsafe_offset=WAVE_COUNTER.ACTIVE])
    var idx = global_idx.x
    var stride = Int(grid_dim.x * block_dim.x)
    while idx < active_count:
        var path_idx = idx
        comptime if Integrator.uses_direct_lighting[integrator]:
            if gpu_rt_spatial_order_enabled(Int(queues.capacity), bounce):
                path_idx = Int(
                    GpuRtQueueOrderScratch.from_queues(queues).indices[
                        unsafe_offset=idx
                    ]
                )
        _gpu_rt_scene_trace_one[
            integrator,
            scene_kind,
            sphere_format,
            triangle_format,
            tlas_format,
            blas_format,
            light_kind,
        ](path_idx, scene, queues, sampling, bounce)
        idx += stride


@always_inline
def _trace_scene_any[
    scene_kind: GpuRtSceneKind,
    sphere_format: GpuRtBvhFormat,
    triangle_format: GpuRtBvhFormat,
    tlas_format: GpuRtBvhFormat,
    blas_format: GpuRtBvhFormat,
](
    scene: GpuRtSceneView, ray: Rayf32[.WORLD], separate_traversal: Bool = False
) -> Bool:
    comptime if scene_kind.has_spheres():
        debug_assert["safe", _use_compiler_assume=True](Bool(scene.spheres))
        var spheres = scene.spheres.unsafe_value()
        var hit = trace_bounds_bvh[
            .WORLD,
            sphere_format.node_width,
            .ANY_HIT,
            _intersect_sphere_leaf[.WORLD, sphere_format.leaf_width, .ANY_HIT],
            sphere_format.node_width == 2,
        ](
            spheres.nodes.unsafe_origin_cast[ImmutAnyOrigin](),
            spheres.leaves.unsafe_origin_cast[ImmutAnyOrigin](),
            spheres.root,
            ray,
        )
        if hit.is_occluded():
            return True
    comptime if scene_kind.has_triangles():
        debug_assert["safe", _use_compiler_assume=True](Bool(scene.triangles))
        var triangles = scene.triangles.unsafe_value()
        var hit = trace_gpu_blas[
            .WORLD,
            triangle_format.node_width,
            .ANY_HIT,
            _intersect_triangle_leaf[
                .WORLD,
                triangle_format.leaf_width,
                .ANY_HIT,
                triangle_format.leaf_width > triangle_format.node_width
                or triangle_format.leaf_width == 8,
            ],
            triangle_format.layout,
            triangle_format.node_width == 4,
        ](
            triangles.nodes.unsafe_origin_cast[ImmutAnyOrigin](),
            triangles.leaves.unsafe_origin_cast[ImmutAnyOrigin](),
            triangles.root,
            ray,
            separate_traversal,
        )
        if hit.is_occluded():
            return True
    comptime if scene_kind.has_instances():
        debug_assert["safe", _use_compiler_assume=True](Bool(scene.instances))
        var instances = scene.instances.unsafe_value()
        var hit = _trace_tlas_ray[
            tlas_format.node_width,
            tlas_format.leaf_width,
            blas_format.node_width,
            blas_format.leaf_width,
            .ANY_HIT,
            _intersect_triangle_leaf[
                .LOCAL,
                blas_format.leaf_width,
                .ANY_HIT,
                blas_format.leaf_width > blas_format.node_width
                or blas_format.leaf_width == 8,
            ],
            blas_format.layout,
        ](
            instances.tlas_nodes.unsafe_origin_cast[ImmutAnyOrigin](),
            instances.tlas_leaf_instances.unsafe_origin_cast[ImmutAnyOrigin](),
            instances.inv_transforms.unsafe_origin_cast[ImmutAnyOrigin](),
            instances.blas_indices.unsafe_origin_cast[ImmutAnyOrigin](),
            instances.blas_descs.unsafe_origin_cast[ImmutAnyOrigin](),
            instances.blas_nodes.unsafe_origin_cast[ImmutAnyOrigin](),
            instances.blas_leaves.unsafe_origin_cast[ImmutAnyOrigin](),
            Int(instances.count),
            Int(instances.blas_count),
            instances.tlas_root,
            ray,
        )
        if hit.is_occluded():
            return True
    return False


@always_inline
def _gpu_rt_shadow_one[
    integrator: Integrator,
    scene_kind: GpuRtSceneKind,
    sphere_format: GpuRtBvhFormat,
    triangle_format: GpuRtBvhFormat,
    tlas_format: GpuRtBvhFormat,
    blas_format: GpuRtBvhFormat,
](idx: Int, scene: GpuRtSceneView, queues: GpuRtTraceQueueView):
    comptime assert Integrator.uses_visibility[integrator]
    var counters = queues.counters.unsafe_origin_cast[MutAnyOrigin]()
    var capacity = Int(queues.capacity)
    var path_ids = queues.shadow_path_ids.unsafe_mut_cast[
        False
    ]().unsafe_origin_cast[ImmutAnyOrigin]()
    var fields = queues.shadow_fields.unsafe_mut_cast[
        False
    ]().unsafe_origin_cast[ImmutAnyOrigin]()
    var work = load_gpu_rt_shadow[integrator != .AO](
        path_ids, fields, capacity, idx
    )
    var ray = Rayf32[.WORLD](
        Point3f32[.WORLD](work.ox, work.oy, work.oz),
        Vec3f32[.WORLD](work.dx, work.dy, work.dz),
        work.t_min,
        work.t_max,
    )
    var occluded = _trace_scene_any[
        scene_kind,
        sphere_format,
        triangle_format,
        tlas_format,
        blas_format,
    ](
        scene,
        ray,
        scene_kind == .TRIANGLES
        and integrator == .MIS
        and capacity >= GPU_RT_LARGE_QUEUE_MIN_CAPACITY,
    )
    var sample_radiance = queues.sample_radiance.unsafe_origin_cast[
        MutAnyOrigin
    ]()
    var sample_base = queues.sample_base
    comptime if integrator == .AO:
        var value = Color(0.08) if occluded else Color(1.0)
        _accumulate_sample(
            sample_radiance, capacity, sample_base, work.path_id, value
        )
    else:
        if not occluded:
            _accumulate_sample(
                sample_radiance,
                capacity,
                sample_base,
                work.path_id,
                Color(work.r, work.g, work.b),
            )


def gpu_rt_shadow_kernel[
    integrator: Integrator,
    scene_kind: GpuRtSceneKind,
    sphere_format: GpuRtBvhFormat,
    triangle_format: GpuRtBvhFormat,
    tlas_format: GpuRtBvhFormat,
    blas_format: GpuRtBvhFormat,
    ordered: Bool = False,
](scene: GpuRtSceneView, queues: GpuRtTraceQueueView):
    var counters = queues.counters.unsafe_origin_cast[MutAnyOrigin]()
    var count = Int(counters[unsafe_offset=WAVE_COUNTER.SHADOW])
    var slot = global_idx.x
    var stride = Int(grid_dim.x * block_dim.x)
    while slot < count:
        var idx = slot
        comptime if ordered:
            idx = Int(
                GpuRtQueueOrderScratch.from_queues(queues).indices[
                    unsafe_offset=slot
                ]
            )
        _gpu_rt_shadow_one[
            integrator,
            scene_kind,
            sphere_format,
            triangle_format,
            tlas_format,
            blas_format,
        ](idx, scene, queues)
        slot += stride


def enqueue_gpu_shadows[
    integrator: Integrator,
    scene_kind: GpuRtSceneKind,
    sphere_format: GpuRtBvhFormat,
    triangle_format: GpuRtBvhFormat,
    tlas_format: GpuRtBvhFormat,
    blas_format: GpuRtBvhFormat,
    MAX_BLOCKS: Int = GPU_RT_MAX_BLOCKS,
](
    ctx: DeviceContext,
    scene: GpuRtSceneView,
    queues: GpuRtTraceQueueView,
    capacity: Int,
    bounce: UInt32 = 0,
) raises:
    """Submit the compact visibility stage when the integrator needs it."""
    comptime if Integrator.uses_visibility[integrator]:
        var blocks = min(
            ceildiv(capacity, GPU_RT_SHADOW_BLOCK_SIZE), MAX_BLOCKS
        )
        blocks = min(
            blocks, gpu_rt_bounce_block_limit(bounce, GPU_RT_SHADOW_BLOCK_SIZE)
        )
        comptime if Integrator.uses_direct_lighting[integrator]:
            if capacity >= GPU_RT_LARGE_QUEUE_MIN_CAPACITY:
                var limit = GPU_RT_DIRECT_SHADOW_MAX_BLOCKS
                # Pure static geometry and uncompressed instance BLASes benefit
                # from a larger visibility grid than mixed/compressed scenes.
                comptime if not scene_kind.has_spheres():
                    comptime if not scene_kind.has_instances() or (
                        not scene_kind.has_triangles()
                        and not blas_format.layout.compressed
                    ):
                        limit = GPU_RT_DIRECT_WIDE_SHADOW_MAX_BLOCKS
                blocks = min(blocks, limit)

        @always_inline
        def enqueue_shadow[ordered: Bool]() raises {imm}:
            ctx.enqueue_function[
                gpu_rt_shadow_kernel[
                    integrator,
                    scene_kind,
                    sphere_format,
                    triangle_format,
                    tlas_format,
                    blas_format,
                    ordered,
                ]
            ](
                scene,
                queues,
                grid_dim=blocks,
                block_dim=GPU_RT_SHADOW_BLOCK_SIZE,
            )

        comptime if Integrator.uses_direct_lighting[integrator]:
            if (
                gpu_rt_queue_order_enabled(capacity, bounce)
                and scene.light_count > 0
                and scene.light_count < 64
            ):
                enqueue_gpu_rt_light_order(ctx, queues, bounce)
                enqueue_shadow[True]()
            else:
                enqueue_shadow[False]()
        else:
            enqueue_shadow[False]()
