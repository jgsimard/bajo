"""Fixed deferred coated-BSDF stage shared by every GPU scene shape."""

from std.math import ceildiv
from max.gpu import block_dim, global_idx, grid_dim
from max.gpu.host import DeviceContext
from bajo.bvh.constants import PrimitiveKind
from bajo.core import Point3f32, Rayf32, Vec3f32
from bajo.rt.material_types import Integrator, SurfaceId
from bajo.rt.render_types import Color, SamplingConfig
from bajo.rt.rays import spawn_surface_ray
from bajo.rt.wavefront_contract import (
    DeviceWaveShadow,
    WAVE_COUNTER,
    load_device_wave_shade,
    wavefront_plane_index,
)
from bajo.rt.gpu.common_kernels import GPU_RT_MAX_BLOCKS
from bajo.rt.gpu.materials import GpuRtMaterials
from bajo.rt.gpu.lighting import _sample_direct_light_candidate
from bajo.rt.gpu.path_shading import (
    _append_shadow,
    _shade_coated_diffuse_inline,
    GPU_RT_SHADE_BLOCK_SIZE,
)
from bajo.rt.gpu.views import GpuRtSceneView, GpuRtTraceQueueView
from bajo.rt.gpu.wavefront_contract import (
    GPU_SHADE_U,
    GPU_SHADE_V,
    load_gpu_rt_path,
)


def gpu_rt_coated_shade_kernel[
    integrator: Integrator
](
    scene: GpuRtSceneView,
    queues: GpuRtTraceQueueView,
    sampling: SamplingConfig,
    bounce: UInt32,
):
    var capacity = Int(queues.capacity)
    var counters = queues.counters.unsafe_origin_cast[MutAnyOrigin]()
    var fields = queues.shade_fields.unsafe_mut_cast[
        False
    ]().unsafe_origin_cast[ImmutAnyOrigin]()
    var work_count = Int(counters[unsafe_offset=WAVE_COUNTER.SHADE])
    var idx = global_idx.x
    var stride = Int(grid_dim.x * block_dim.x)
    while idx < work_count:
        var surface_value = queues.shade_surfaces[unsafe_offset=idx]
        if SurfaceId.kind_from_raw(surface_value) == .COATED_DIFFUSE:
            var work = load_device_wave_shade(
                queues.shade_path_refs.unsafe_mut_cast[
                    False
                ]().unsafe_origin_cast[ImmutAnyOrigin](),
                queues.shade_surfaces.unsafe_mut_cast[
                    False
                ]().unsafe_origin_cast[ImmutAnyOrigin](),
                fields,
                capacity,
                idx,
            )
            var path = load_gpu_rt_path[integrator](
                queues.src_path_ids.unsafe_origin_cast[ImmutAnyOrigin](),
                queues.src_path_fields.unsafe_origin_cast[ImmutAnyOrigin](),
                capacity,
                Int(work.path_idx),
            )
            var normal = Vec3f32[.WORLD](work.nx, work.ny, work.nz)
            var ray = Rayf32[.WORLD](
                Point3f32[.WORLD](path.ox, path.oy, path.oz),
                Vec3f32[.WORLD](path.dx, path.dy, path.dz),
                path.t_min,
                path.t_max,
            )
            var uv_u = fields[
                unsafe_offset=wavefront_plane_index(GPU_SHADE_U, capacity, idx)
            ]
            var uv_v = fields[
                unsafe_offset=wavefront_plane_index(GPU_SHADE_V, capacity, idx)
            ]
            comptime if Integrator.uses_direct_lighting[integrator]:
                var direct = _sample_direct_light_candidate[
                    integrator, PrimitiveKind.UNKNOWN, True
                ](
                    path,
                    ray,
                    work.t,
                    normal,
                    uv_u,
                    uv_v,
                    surface_value,
                    scene.lambertians.unsafe_origin_cast[ImmutAnyOrigin](),
                    scene.lambertian_texture_indices.unsafe_origin_cast[
                        ImmutAnyOrigin
                    ](),
                    scene.texture_descs.unsafe_origin_cast[ImmutAnyOrigin](),
                    scene.texture_pixels.unsafe_origin_cast[ImmutAnyOrigin](),
                    scene.metals.unsafe_origin_cast[ImmutAnyOrigin](),
                    scene.coated_diffuses.unsafe_origin_cast[ImmutAnyOrigin](),
                    scene.coated_diffuse_texture_indices.unsafe_origin_cast[
                        ImmutAnyOrigin
                    ](),
                    scene.light_kinds.unsafe_origin_cast[ImmutAnyOrigin](),
                    scene.light_fields.unsafe_origin_cast[ImmutAnyOrigin](),
                    Int(scene.light_count),
                    scene.total_light_weight,
                    scene.environment_weight,
                    scene.environment_cdf_total,
                    scene.environment_cdf.unsafe_origin_cast[ImmutAnyOrigin](),
                    Int(scene.environment_cdf_count),
                    scene.environment_kind,
                    scene.environment_texture_index,
                    Color(
                        scene.environment_scale_x,
                        scene.environment_scale_y,
                        scene.environment_scale_z,
                    ),
                    scene.environment_light_to_world,
                    sampling,
                    bounce,
                )
                if direct.valid:
                    var shadow = spawn_surface_ray(
                        ray.at(work.t), direct.direction, direct.shadow_t_max
                    )
                    _append_shadow(
                        DeviceWaveShadow(
                            path.path_id,
                            shadow.o.x,
                            shadow.o.y,
                            shadow.o.z,
                            shadow.t_min,
                            shadow.d.x,
                            shadow.d.y,
                            shadow.d.z,
                            shadow.t_max,
                            direct.contribution.x,
                            direct.contribution.y,
                            direct.contribution.z,
                        ),
                        queues.shadow_path_ids.unsafe_origin_cast[
                            MutAnyOrigin
                        ](),
                        queues.shadow_fields.unsafe_origin_cast[MutAnyOrigin](),
                        counters,
                        capacity,
                    )
            _shade_coated_diffuse_inline[integrator](
                path,
                ray.d,
                normal,
                uv_u,
                uv_v,
                work.t,
                surface_value,
                scene.coated_diffuses.unsafe_origin_cast[ImmutAnyOrigin](),
                scene.coated_diffuse_texture_indices.unsafe_origin_cast[
                    ImmutAnyOrigin
                ](),
                scene.texture_descs.unsafe_origin_cast[ImmutAnyOrigin](),
                scene.texture_pixels.unsafe_origin_cast[ImmutAnyOrigin](),
                queues.dst_path_ids.unsafe_origin_cast[MutAnyOrigin](),
                queues.dst_path_fields.unsafe_origin_cast[MutAnyOrigin](),
                counters,
                capacity,
                sampling,
                bounce,
            )
        idx += stride


@always_inline
def enqueue_coated_shading[
    integrator: Integrator
](
    ctx: DeviceContext,
    materials: GpuRtMaterials,
    scene: GpuRtSceneView,
    queues: GpuRtTraceQueueView,
    sampling: SamplingConfig,
    bounce: UInt32,
) raises:
    if not materials.stages.coated_diffuse:
        return
    var blocks = min(
        ceildiv(Int(queues.capacity), GPU_RT_SHADE_BLOCK_SIZE),
        GPU_RT_MAX_BLOCKS,
    )
    ctx.enqueue_function[gpu_rt_coated_shade_kernel[integrator]](
        scene,
        queues,
        sampling,
        bounce,
        grid_dim=blocks,
        block_dim=GPU_RT_SHADE_BLOCK_SIZE,
    )
