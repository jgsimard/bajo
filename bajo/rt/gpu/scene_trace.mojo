"""Compile-time-specialized traversal kernel for all GPU RT scene shapes."""

from layout import TileTensor, Coord, Idx, row_major
from bajo.rt.denoise_guides import denoise_component
from max.gpu import block_dim, global_idx, grid_dim
from std.math import ceildiv, pi
from max.gpu.host import DeviceContext

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
from bajo.rt.lighting import _emissive_hit_light_pdf
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

from bajo.rt.gpu.config import (
    _GpuRtKernelFormats,
    GpuRtSceneKind,
)
from bajo.rt.wavefront_contract import (
    DeviceWavePath,
    load_device_wave_path,
    WavePathFloatAbi,
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
from bajo.rt.gpu.lighting import (
    _sample_direct_light_by_kind,
    _gpu_emissive_hit_weight,
)
from bajo.rt.gpu.queue_layout import gpu_rt_path_tile
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


comptime GPU_RT_SHADOW_BLOCK_SIZE = 128
# Below this size, retain the launch policy tuned for the 256K working set.
comptime GPU_RT_LARGE_QUEUE_MIN_CAPACITY = 1 << 19
comptime GPU_RT_DIRECT_SHADOW_MAX_BLOCKS = 2048
comptime GPU_RT_DIRECT_WIDE_SHADOW_MAX_BLOCKS = 8192


def _gpu_environment_radiance(
    scene: GpuRtSceneView,
    texture_descs: ImmPointer[UInt32, _],
    texture_pixels: ImmPointer[Float32, _],
    direction: Vec3f32[.WORLD],
) -> Color:
    var scale = Color(
        scene.environment_scale_x,
        scene.environment_scale_y,
        scene.environment_scale_z,
    )
    match scene.environment_kind:
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
    return scale * _GpuImageTextureView.load(
        scene.environment_texture_index, texture_descs, texture_pixels
    ).sample_environment(uv.u, uv.v)


@inline(.always)
def _gpu_rt_scene_trace_path[
    integrator: Integrator,
    scene_kind: GpuRtSceneKind,
    Formats: _GpuRtKernelFormats,
](
    idx: Int,
    var path: DeviceWavePath,
    scene: GpuRtSceneView,
    queues: GpuRtTraceQueueView,
    sampling: SamplingConfig,
    bounce: UInt32,
    uniform_sampling_kind: UInt32,
    use_mis: Bool,
    allow_scattering: Bool = True,
):
    comptime assert (
        scene_kind.has_spheres()
        or scene_kind.has_triangles()
        or scene_kind.has_instances()
    )
    var emissives = scene.emissives
    var lambertians = scene.lambertians
    var metals = scene.metals
    var coated_diffuses = scene.coated_diffuses
    var lambertian_texture_indices = scene.lambertian_texture_indices
    var coated_diffuse_texture_indices = scene.coated_diffuse_texture_indices
    var texture_descs = scene.texture_descs
    var texture_pixels = scene.texture_pixels
    var total_light_weight = scene.total_light_weight
    var dst_path_ids = queues.dst_path_ids
    var dst_path_fields = queues.dst_path_fields
    var shade_path_refs = queues.shade_path_refs
    var shade_surfaces = queues.shade_surfaces
    var shade_fields = queues.shade_fields
    var shadow_path_ids = queues.shadow_path_ids
    var shadow_fields = queues.shadow_fields
    var counters = queues.counters
    var sample_radiance = queues.sample_radiance
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
        var sphere_nodes = spheres.nodes
        var leaf_spheres = spheres.leaves
        var sphere_root = spheres.root
        var sphere_surfaces = spheres.surfaces
        var signed_radii = spheres.signed_radii
        var sphere_hit: Hit[.WORLD]
        if scene.opacity.count > 0:
            sphere_hit = trace_gpu_opacity_spheres[
                Formats.sphere, .CLOSEST_HIT
            ](scene, ray)
        else:
            sphere_hit = trace_bounds_bvh[
                .WORLD,
                Formats.sphere.node_width,
                .CLOSEST_HIT,
                _intersect_sphere_leaf[
                    .WORLD, Formats.sphere.leaf_width, .CLOSEST_HIT
                ],
                Formats.sphere.node_width == 2,
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
        var triangle_nodes = triangles.nodes
        var leaf_vertices = triangles.leaves
        var triangle_root = triangles.root
        var triangle_surfaces = triangles.surfaces
        var triangle_ray = Rayf32[.WORLD](ray.o, ray.d, ray.t_min, closest_t)
        # A separate GPU function avoids slow fused traversal/shading code
        # for large static MIS queues, without additional dispatch variants.
        var triangle_hit: Hit[.WORLD]
        if scene.opacity.count > 0:
            triangle_hit = trace_gpu_opacity_triangles[
                Formats.triangle, .CLOSEST_HIT
            ](scene, triangle_ray)
        else:
            triangle_hit = trace_gpu_blas[
                .WORLD,
                Formats.triangle.node_width,
                .CLOSEST_HIT,
                _intersect_triangle_leaf[
                    .WORLD,
                    Formats.triangle.leaf_width,
                    .CLOSEST_HIT,
                    Formats.triangle.leaf_width > Formats.triangle.node_width
                    or Formats.triangle.leaf_width == 8,
                ],
                Formats.triangle.layout,
                Formats.triangle.node_width == 4,
            ](
                triangle_nodes,
                leaf_vertices,
                triangle_root,
                triangle_ray,
                scene_kind == .TRIANGLES
                and Integrator.uses_direct_lighting[integrator]
                and use_mis
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
        var tlas_nodes = instances.tlas_nodes
        var tlas_leaf_instances = instances.tlas_leaf_instances
        var inst_inv_transform = instances.inv_transforms
        var inst_blas_indices = instances.blas_indices
        var blas_descs = instances.blas_descs
        var blas_nodes = instances.blas_nodes
        var blas_leaves = instances.blas_leaves
        var normal_offsets = instances.normal_offsets
        var vertex_normals = instances.normals
        var texcoord_offsets = instances.texcoord_offsets
        var vertex_texcoords = instances.texcoords
        var tlas_root = instances.tlas_root
        var instance_count_i32 = instances.count
        var instance_surfaces = instances.surfaces
        var instance_ray = Rayf32[.WORLD](ray.o, ray.d, ray.t_min, closest_t)
        var instance_hit: Hit[.WORLD]
        if scene.opacity.count > 0:
            instance_hit = trace_gpu_opacity_instances[
                Formats.tlas, Formats.blas, .CLOSEST_HIT
            ](scene, instance_ray)
        else:
            instance_hit = _trace_tlas_ray[
                Formats.tlas.node_width,
                Formats.tlas.leaf_width,
                Formats.blas.node_width,
                Formats.blas.leaf_width,
                .CLOSEST_HIT,
                _intersect_triangle_leaf[
                    .LOCAL,
                    Formats.blas.leaf_width,
                    .CLOSEST_HIT,
                    Formats.blas.leaf_width > Formats.blas.node_width
                    or Formats.blas.leaf_width == 8,
                ],
                Formats.blas.layout,
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
                var records = TileTensor(
                    ptr=vertex_texcoords.unsafe_offset(Int(texcoord_offset)),
                    layout=row_major(Coord(triangle_idx + 1, Idx[6])),
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
                var records = TileTensor(
                    ptr=vertex_normals.unsafe_offset(Int(normal_offset)),
                    layout=row_major(Coord(triangle_idx + 1, Idx[9])),
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

    if path.medium_id != 0:
        var medium = Int(path.medium_id)
        var absorption = TileTensor(
            ptr=scene.medium_absorption, layout=row_major(medium + 1, 3)
        )
        var sigma_a = Color(absorption.ptr_at_offset((medium, 0)))
        var attenuation = medium_transmittance(
            sigma_a,
            (closest_t if found else ray.t_max) * length(ray.d),
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
            comptime if Integrator.uses_direct_lighting[integrator]:
                if use_mis and bounce > 0 and not path.delta:
                    match scene.environment_kind:
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
            var environment_mis_weight = _gpu_emissive_hit_weight[integrator](
                bounce,
                path.delta,
                path.bsdf_pdf,
                environment_light_pdf,
                use_mis,
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
        elif integrator == .AO:
            _accumulate_sample(
                sample_radiance,
                capacity,
                sample_base,
                path.path_id,
                _gpu_environment_radiance(
                    scene, texture_descs, texture_pixels, ray.d
                ),
            )
        return

    if SurfaceId.kind_from_raw(surface_value) == .MIX:
        var material_rng = path_stage_rng(
            sampling, path.path_id, wavefront_rng_material_stage(UInt32(bounce))
        )
        var random = material_rng.f32()
        while SurfaceId.kind_from_raw(surface_value) == .MIX:
            var index = Int(SurfaceId.index_from_raw(surface_value))
            var scales = TileTensor(
                ptr=scene.mix_uv_scales,
                layout=row_major(Coord(index + 1, Idx[2])),
            )
            var children = TileTensor(
                ptr=scene.mix_surfaces,
                layout=row_major(Coord(index + 1, Idx[2])),
            )
            var amount = scene.mix_amounts[unsafe_offset=index]
            var texture_index = scene.mix_texture_indices[unsafe_offset=index]
            if texture_index != NO_TEXTURE:
                var color = _sample_gpu_image_texture(
                    texture_index,
                    uv_u * scales.load[invariant=False]((index, 0)),
                    uv_v * scales.load[invariant=False]((index, 1)),
                    texture_descs,
                    texture_pixels,
                )
                amount *= luminance(color)
            var branch = sample_mix_branch(amount.clamp(0.0, 1.0), random)
            surface_value = children.load[invariant=False]((index, branch))
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
        scene.normal_maps,
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
            scene.dielectrics,
            texture_descs,
            texture_pixels,
        )
    if (
        bounce == 0
        and Int(queues.denoise_components) != 0
        and Integrator.is_path_tracing[integrator]
    ):
        var primary_emitter = gpu_emitter(scene, primitive_value, surface_value)
        Pointer[UInt32, MutUntrackedOrigin](
            unsafe_from_address=queues.denoise_components
        )[unsafe_offset=Int(path.path_id)] = denoise_component(
            SurfaceId.kind_from_raw(surface_value),
            primary_emitter.radiance.max_component() > 0,
        ).value()
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

    comptime if Integrator.is_path_tracing[integrator]:
        var emitter = gpu_emitter(scene, primitive_value, surface_value)
        if (
            emitter.radiance.max_component() > 0.0
            and (geometric.front_face or emitter.two_sided)
            and opacity_accepts(
                gpu_opacity_value(scene.opacity, primitive_value, uv_u, uv_v),
                opacity_point_sample(ray.at(closest_t)),
            )
        ):
            var light_pdf = Float32(0.0)
            comptime if Integrator.uses_direct_lighting[integrator]:
                if use_mis and bounce > 0 and not path.delta:
                    light_pdf = _emissive_hit_light_pdf(
                        ray.d,
                        closest_t,
                        geometric.normal,
                        emitter.radiance,
                        scene.total_light_weight,
                        emitter.two_sided,
                    )
            var weight = _gpu_emissive_hit_weight[integrator](
                bounce, path.delta, path.bsdf_pdf, light_pdf, use_mis
            )
            _accumulate_sample(
                sample_radiance,
                capacity,
                sample_base,
                path.path_id,
                Color(path.tx, path.ty, path.tz) * emitter.radiance * weight,
            )

    if not allow_scattering:
        return

    comptime if Integrator.uses_direct_lighting[integrator]:
        var direct = _sample_direct_light_by_kind[integrator](
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
            uniform_sampling_kind,
            front_face=geometric.front_face,
            use_mis=use_mis,
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
        emission_visible=False,
    )


@inline(.always)
def _gpu_rt_scene_trace_one[
    integrator: Integrator,
    scene_kind: GpuRtSceneKind,
    Formats: _GpuRtKernelFormats,
](
    idx: Int,
    scene: GpuRtSceneView,
    queues: GpuRtTraceQueueView,
    sampling: SamplingConfig,
    bounce: UInt32,
    allow_scattering: Bool,
    uniform_sampling_kind: UInt32,
    use_mis: Bool,
):
    var src_path_ids = queues.src_path_ids
    var src_path_fields = queues.src_path_fields
    var path: DeviceWavePath
    comptime if Integrator.uses_direct_lighting[integrator]:
        # NEE does not initialize or consume the previous-BSDF PDF plane.
        path = load_device_wave_path[False, True, False, True, True](
            src_path_ids, src_path_fields, Int(queues.capacity), idx
        )
        if use_mis:
            path.bsdf_pdf = gpu_rt_path_tile(
                src_path_fields, Int(queues.capacity)
            )[(WavePathFloatAbi.BSDF_PDF, idx)]
    else:
        path = load_gpu_rt_path[integrator](
            src_path_ids, src_path_fields, Int(queues.capacity), idx
        )
    _gpu_rt_scene_trace_path[
        integrator,
        scene_kind,
        Formats,
    ](
        idx,
        path,
        scene,
        queues,
        sampling,
        bounce,
        uniform_sampling_kind,
        use_mis,
        allow_scattering,
    )


def gpu_rt_primary_scene_trace_kernel[
    integrator: Integrator,
    scene_kind: GpuRtSceneKind,
    Formats: _GpuRtKernelFormats,
    store_source_path: Bool,
](
    camera_params: Pointer[Float32, ImmUntrackedOrigin],
    source_path_ids: Pointer[UInt32, MutUntrackedOrigin],
    source_path_fields: Pointer[Float32, MutUntrackedOrigin],
    scene: GpuRtSceneView,
    queues: GpuRtTraceQueueView,
    width_i32: Int32,
    height_i32: Int32,
    samples_per_pixel_i32: Int32,
    sampling_arg: GpuRtSamplingConfig,
    uniform_sampling_kind: UInt32,
    use_mis_arg: UInt32,
):
    var sampling = sampling_arg.load()
    var counters = queues.counters
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
        queues.sample_radiance,
        capacity,
    )
    clear_gpu_rt_sample(
        queues.sample_radiance,
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
        Formats,
    ](
        idx,
        path,
        scene,
        queues,
        sampling,
        UInt32(0),
        uniform_sampling_kind,
        use_mis_arg != 0,
    )


def gpu_rt_scene_trace_kernel[
    integrator: Integrator,
    scene_kind: GpuRtSceneKind,
    Formats: _GpuRtKernelFormats,
](
    scene: GpuRtSceneView,
    queues: GpuRtTraceQueueView,
    sampling_arg: GpuRtSamplingConfig,
    bounce: UInt32,
    allow_scattering: UInt32,
    uniform_sampling_kind: UInt32,
    use_mis_arg: UInt32,
):
    var sampling = sampling_arg.load()
    var counters = queues.counters
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
            Formats,
        ](
            path_idx,
            scene,
            queues,
            sampling,
            bounce,
            allow_scattering != 0,
            uniform_sampling_kind,
            use_mis_arg != 0,
        )
        idx += stride


@inline(.always)
def _trace_scene_any[
    scene_kind: GpuRtSceneKind,
    Formats: _GpuRtKernelFormats,
](
    scene: GpuRtSceneView, ray: Rayf32[.WORLD], separate_traversal: Bool = False
) -> Bool:
    if scene.opacity.count > 0:
        comptime if scene_kind.has_spheres():
            if trace_gpu_opacity_spheres[Formats.sphere, .ANY_HIT](
                scene, ray
            ).is_occluded():
                return True
        comptime if scene_kind.has_triangles():
            if trace_gpu_opacity_triangles[Formats.triangle, .ANY_HIT](
                scene, ray
            ).is_occluded():
                return True
        comptime if scene_kind.has_instances():
            if trace_gpu_opacity_instances[
                Formats.tlas, Formats.blas, .ANY_HIT
            ](scene, ray).is_occluded():
                return True
        return False
    comptime if scene_kind.has_spheres():
        debug_assert["safe", _use_compiler_assume=True](Bool(scene.spheres))
        var spheres = scene.spheres.unsafe_value()
        var hit = trace_bounds_bvh[
            .WORLD,
            Formats.sphere.node_width,
            .ANY_HIT,
            _intersect_sphere_leaf[.WORLD, Formats.sphere.leaf_width, .ANY_HIT],
            Formats.sphere.node_width == 2,
        ](
            spheres.nodes,
            spheres.leaves,
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
            Formats.triangle.node_width,
            .ANY_HIT,
            _intersect_triangle_leaf[
                .WORLD,
                Formats.triangle.leaf_width,
                .ANY_HIT,
                Formats.triangle.leaf_width > Formats.triangle.node_width
                or Formats.triangle.leaf_width == 8,
            ],
            Formats.triangle.layout,
            Formats.triangle.node_width == 4,
        ](
            triangles.nodes,
            triangles.leaves,
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
            Formats.tlas.node_width,
            Formats.tlas.leaf_width,
            Formats.blas.node_width,
            Formats.blas.leaf_width,
            .ANY_HIT,
            _intersect_triangle_leaf[
                .LOCAL,
                Formats.blas.leaf_width,
                .ANY_HIT,
                Formats.blas.leaf_width > Formats.blas.node_width
                or Formats.blas.leaf_width == 8,
            ],
            Formats.blas.layout,
        ](
            instances.tlas_nodes,
            instances.tlas_leaf_instances,
            instances.inv_transforms,
            instances.blas_indices,
            instances.blas_descs,
            instances.blas_nodes,
            instances.blas_leaves,
            Int(instances.count),
            Int(instances.blas_count),
            instances.tlas_root,
            ray,
        )
        if hit.is_occluded():
            return True
    return False


@inline(.always)
def _gpu_rt_shadow_one[
    integrator: Integrator,
    scene_kind: GpuRtSceneKind,
    Formats: _GpuRtKernelFormats,
](idx: Int, scene: GpuRtSceneView, queues: GpuRtTraceQueueView):
    comptime assert Integrator.uses_visibility[integrator]
    var counters = queues.counters
    var capacity = Int(queues.capacity)
    var path_ids = queues.shadow_path_ids.unsafe_mut_cast[False]()
    var fields = queues.shadow_fields.unsafe_mut_cast[False]()
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
        Formats,
    ](
        scene,
        ray,
        scene_kind == .TRIANGLES
        and integrator == .MIS
        and capacity >= GPU_RT_LARGE_QUEUE_MIN_CAPACITY,
    )
    var sample_radiance = queues.sample_radiance
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
    Formats: _GpuRtKernelFormats,
    ordered: Bool = False,
](scene: GpuRtSceneView, queues: GpuRtTraceQueueView):
    var counters = queues.counters
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
            Formats,
        ](idx, scene, queues)
        slot += stride


def enqueue_gpu_shadows[
    integrator: Integrator,
    scene_kind: GpuRtSceneKind,
    Formats: _GpuRtKernelFormats,
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
                        and Formats.blas.layout != .CWBVH8
                    ):
                        limit = GPU_RT_DIRECT_WIDE_SHADOW_MAX_BLOCKS
                blocks = min(blocks, limit)

        def enqueue_shadow[ordered: Bool]() raises {imm}:
            ctx.enqueue_function[
                gpu_rt_shadow_kernel[
                    integrator,
                    scene_kind,
                    Formats,
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
