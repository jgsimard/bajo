"""Compile-time-specialized traversal kernel for all GPU RT scene shapes."""

from std.gpu import block_dim, global_idx, grid_dim
from std.math import ceildiv
from max.gpu.host import DeviceContext

from bajo.bvh import Hit
from bajo.bvh.gpu import GpuBvhLayout
from bajo.bvh.gpu.sphere_bvh import _intersect_sphere_leaf
from bajo.bvh.gpu.tlas import _trace_tlas_ray
from bajo.bvh.gpu.trace import trace_bounds_bvh
from bajo.bvh.gpu.triangle_bvh import (
    _intersect_triangle_leaf,
    trace_cwbvh8_triangles,
)
from bajo.core import Frame, GeoKind, Point3f32, Rayf32, Vec3f32
from bajo.rt.common import sky_color
from bajo.rt.geometry import orient_surface_normal
from bajo.rt.types import Color, Integrator
from bajo.rt.wavefront_contract import (
    DeviceWaveShadow,
    WAVE_COUNTER,
)
from bajo.rt.gpu.wavefront_contract import (
    _reserve_slot,
    load_gpu_rt_path,
    load_gpu_rt_shadow,
)
from bajo.rt.gpu.common_kernels import GPU_RT_BLOCK_SIZE, GPU_RT_MAX_BLOCKS
from bajo.rt.gpu.path_shading import (
    _accumulate_sample,
    _append_shadow,
    _make_ao_ray,
    _route_surface_hit,
    _sample_direct_light_candidate,
)
from bajo.rt.gpu.views import GpuRtSceneView, GpuRtTraceQueueView


@always_inline
def _gpu_rt_scene_trace_one[
    ALGORITHM: Integrator,
    HAS_SPHERES: Bool,
    HAS_TRIANGLES: Bool,
    HAS_INSTANCES: Bool,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    triangle_node_width: SIMDLength,
    triangle_leaf_width: SIMDLength,
    tlas_node_width: SIMDLength,
    tlas_leaf_width: SIMDLength,
    blas_node_width: SIMDLength,
    blas_leaf_width: SIMDLength,
    triangle_layout: GpuBvhLayout,
    blas_layout: GpuBvhLayout,
](
    idx: Int,
    scene: GpuRtSceneView,
    queues: GpuRtTraceQueueView,
    rng_seed: UInt64,
    bounce: UInt32,
):
    comptime assert ALGORITHM.is_valid()
    comptime assert HAS_SPHERES or HAS_TRIANGLES or HAS_INSTANCES
    var emissives = scene.emissives.unsafe_origin_cast[ImmutAnyOrigin]()
    var lambertians = scene.lambertians.unsafe_origin_cast[ImmutAnyOrigin]()
    var metals = scene.metals.unsafe_origin_cast[ImmutAnyOrigin]()
    var dielectrics = scene.dielectrics.unsafe_origin_cast[ImmutAnyOrigin]()
    var light_kinds = scene.light_kinds.unsafe_origin_cast[ImmutAnyOrigin]()
    var light_fields = scene.light_fields.unsafe_origin_cast[ImmutAnyOrigin]()
    var light_count_i32 = scene.light_count
    var total_light_weight = scene.total_light_weight
    var src_path_ids = queues.src_path_ids.unsafe_origin_cast[ImmutAnyOrigin]()
    var src_path_fields = queues.src_path_fields.unsafe_origin_cast[
        ImmutAnyOrigin
    ]()
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
    var path = load_gpu_rt_path[ALGORITHM](
        src_path_ids, src_path_fields, capacity, idx
    )
    var ray = Rayf32[.WORLD](
        Point3f32[.WORLD](path.ox, path.oy, path.oz),
        Vec3f32[.WORLD](path.dx, path.dy, path.dz),
        path.t_min,
        path.t_max,
    )
    var found = False
    var closest_t = ray.t_max
    var outward = Vec3f32[.WORLD](0.0)
    var surface_value = UInt32(0)

    comptime if HAS_SPHERES:
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
            node_width,
            .CLOSEST_HIT,
            _intersect_sphere_leaf[.WORLD, leaf_width, .CLOSEST_HIT],
            node_width == 2,
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
            surface_value = sphere_surfaces[unsafe_offset=sphere_idx]

    comptime if HAS_TRIANGLES:
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
        var triangle_ray = Rayf32[.WORLD](
            ray.o, ray.d, ray.t_min, closest_t
        )
        var triangle_hit = Hit[.WORLD].miss(closest_t)
        comptime if triangle_layout.compressed:
            triangle_hit = trace_cwbvh8_triangles[
                .WORLD,
                .CLOSEST_HIT,
            ](triangle_nodes, leaf_vertices, triangle_root, triangle_ray)
        else:
            triangle_hit = trace_bounds_bvh[
                .WORLD,
                triangle_node_width,
                .CLOSEST_HIT,
                _intersect_triangle_leaf[
                    .WORLD,
                    triangle_leaf_width,
                    .CLOSEST_HIT,
                    triangle_leaf_width > triangle_node_width
                    or triangle_leaf_width == 8,
                ],
                triangle_node_width == 4,
            ](triangle_nodes, leaf_vertices, triangle_root, triangle_ray)
        if triangle_hit.is_hit() and triangle_hit.t < closest_t:
            found = True
            closest_t = triangle_hit.t
            outward = triangle_hit.normal.unsafe_convert[
                new_kind=GeoKind.VECTOR
            ]()
            surface_value = triangle_surfaces[
                unsafe_offset=Int(triangle_hit.prim)
            ]

    comptime if HAS_INSTANCES:
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
        var tlas_root = instances.tlas_root
        var instance_count_i32 = instances.count
        var instance_surfaces = instances.surfaces.unsafe_origin_cast[
            ImmutAnyOrigin
        ]()
        var instance_ray = Rayf32[.WORLD](
            ray.o, ray.d, ray.t_min, closest_t
        )
        var instance_hit = _trace_tlas_ray[
            tlas_node_width,
            tlas_leaf_width,
            blas_node_width,
            blas_leaf_width,
            .CLOSEST_HIT,
            _intersect_triangle_leaf[
                .LOCAL,
                blas_leaf_width,
                .CLOSEST_HIT,
                blas_leaf_width > blas_node_width or blas_leaf_width == 8,
            ],
            blas_layout,
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
            surface_value = instance_surfaces[
                unsafe_offset=Int(instance_hit.inst)
            ]

    if not found:
        comptime if ALGORITHM in (Integrator.PATH, Integrator.NEE, Integrator.MIS):
            _accumulate_sample(
                sample_radiance,
                capacity,
                sample_base,
                path.path_id,
                Color(path.tx, path.ty, path.tz) * sky_color(ray.d),
            )
        return

    var oriented = orient_surface_normal(ray.d, outward)
    var normal = oriented.normal
    comptime if ALGORITHM == .AO:
        var ao_ray = _make_ao_ray(rng_seed, path, ray, closest_t, normal)
        _append_shadow[False](
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

    comptime if ALGORITHM in (Integrator.NEE, Integrator.MIS):
        var direct = _sample_direct_light_candidate[ALGORITHM](
            path,
            ray,
            closest_t,
            normal,
            surface_value,
            lambertians,
            metals,
            dielectrics,
            light_kinds,
            light_fields,
            Int(light_count_i32),
            total_light_weight,
            rng_seed,
            bounce,
        )
        if direct.valid:
            var shadow_ray = Rayf32[.WORLD](
                ray.o + closest_t * ray.d,
                direct.direction,
                0.001,
                direct.shadow_t_max,
            )
            _append_shadow(
                DeviceWaveShadow(
                    path.path_id,
                    shadow_ray.o.x,
                    shadow_ray.o.y,
                    shadow_ray.o.z,
                    shadow_ray.t_min,
                    shadow_ray.d.x,
                    shadow_ray.d.y,
                    shadow_ray.d.z,
                    shadow_ray.t_max,
                    direct.contribution.x,
                    direct.contribution.y,
                    direct.contribution.z,
                ),
                shadow_path_ids,
                shadow_fields,
                counters,
                capacity,
            )

    _route_surface_hit[ALGORITHM](
        idx,
        path,
        ray.d,
        normal,
        oriented.front_face,
        closest_t,
        surface_value,
        bounce,
        total_light_weight,
        emissives,
        lambertians,
        dst_path_ids,
        dst_path_fields,
        shade_path_refs,
        shade_surfaces,
        shade_fields,
        counters,
        sample_radiance,
        capacity,
        sample_base,
        rng_seed,
    )


def gpu_rt_scene_trace_kernel[
    ALGORITHM: Integrator,
    HAS_SPHERES: Bool,
    HAS_TRIANGLES: Bool,
    HAS_INSTANCES: Bool,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    triangle_node_width: SIMDLength,
    triangle_leaf_width: SIMDLength,
    tlas_node_width: SIMDLength,
    tlas_leaf_width: SIMDLength,
    blas_node_width: SIMDLength,
    blas_leaf_width: SIMDLength,
    triangle_layout: GpuBvhLayout = .WIDE,
    blas_layout: GpuBvhLayout = .WIDE,
](
    scene: GpuRtSceneView,
    queues: GpuRtTraceQueueView,
    rng_seed: UInt64,
    bounce: UInt32,
):
    var counters = queues.counters.unsafe_origin_cast[MutAnyOrigin]()
    var active_count = Int(counters[unsafe_offset=WAVE_COUNTER.ACTIVE])
    var idx = global_idx.x
    var stride = Int(grid_dim.x * block_dim.x)
    while idx < active_count:
        _gpu_rt_scene_trace_one[
            ALGORITHM,
            HAS_SPHERES,
            HAS_TRIANGLES,
            HAS_INSTANCES,
            node_width,
            leaf_width,
            triangle_node_width,
            triangle_leaf_width,
            tlas_node_width,
            tlas_leaf_width,
            blas_node_width,
            blas_leaf_width,
            triangle_layout,
            blas_layout,
        ](idx, scene, queues, rng_seed, bounce)
        idx += stride


@always_inline
def _trace_scene_any[
    HAS_SPHERES: Bool,
    HAS_TRIANGLES: Bool,
    HAS_INSTANCES: Bool,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    triangle_node_width: SIMDLength,
    triangle_leaf_width: SIMDLength,
    tlas_node_width: SIMDLength,
    tlas_leaf_width: SIMDLength,
    blas_node_width: SIMDLength,
    blas_leaf_width: SIMDLength,
    triangle_layout: GpuBvhLayout,
    blas_layout: GpuBvhLayout,
](scene: GpuRtSceneView, ray: Rayf32[.WORLD]) -> Bool:
    comptime if HAS_SPHERES:
        debug_assert["safe", _use_compiler_assume=True](Bool(scene.spheres))
        var spheres = scene.spheres.unsafe_value()
        var hit = trace_bounds_bvh[
            .WORLD,
            node_width,
            .ANY_HIT,
            _intersect_sphere_leaf[.WORLD, leaf_width, .ANY_HIT],
            node_width == 2,
        ](
            spheres.nodes.unsafe_origin_cast[ImmutAnyOrigin](),
            spheres.leaves.unsafe_origin_cast[ImmutAnyOrigin](),
            spheres.root,
            ray,
        )
        if hit.is_occluded():
            return True
    comptime if HAS_TRIANGLES:
        debug_assert["safe", _use_compiler_assume=True](Bool(scene.triangles))
        var triangles = scene.triangles.unsafe_value()
        var hit = Hit[.WORLD].miss(ray.t_max)
        comptime if triangle_layout.compressed:
            hit = trace_cwbvh8_triangles[
                .WORLD,
                .ANY_HIT,
            ](
                triangles.nodes.unsafe_origin_cast[ImmutAnyOrigin](),
                triangles.leaves.unsafe_origin_cast[ImmutAnyOrigin](),
                triangles.root,
                ray,
            )
        else:
            hit = trace_bounds_bvh[
                .WORLD,
                triangle_node_width,
                .ANY_HIT,
                _intersect_triangle_leaf[
                    .WORLD,
                    triangle_leaf_width,
                    .ANY_HIT,
                    triangle_leaf_width > triangle_node_width
                    or triangle_leaf_width == 8,
                ],
                triangle_node_width == 4,
            ](
                triangles.nodes.unsafe_origin_cast[ImmutAnyOrigin](),
                triangles.leaves.unsafe_origin_cast[ImmutAnyOrigin](),
                triangles.root,
                ray,
            )
        if hit.is_occluded():
            return True
    comptime if HAS_INSTANCES:
        debug_assert["safe", _use_compiler_assume=True](Bool(scene.instances))
        var instances = scene.instances.unsafe_value()
        var hit = _trace_tlas_ray[
            tlas_node_width,
            tlas_leaf_width,
            blas_node_width,
            blas_leaf_width,
            .ANY_HIT,
            _intersect_triangle_leaf[
                .LOCAL,
                blas_leaf_width,
                .ANY_HIT,
                blas_leaf_width > blas_node_width or blas_leaf_width == 8,
            ],
            blas_layout,
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
    ALGORITHM: Integrator,
    HAS_SPHERES: Bool,
    HAS_TRIANGLES: Bool,
    HAS_INSTANCES: Bool,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    triangle_node_width: SIMDLength,
    triangle_leaf_width: SIMDLength,
    tlas_node_width: SIMDLength,
    tlas_leaf_width: SIMDLength,
    blas_node_width: SIMDLength,
    blas_leaf_width: SIMDLength,
    triangle_layout: GpuBvhLayout,
    blas_layout: GpuBvhLayout,
](idx: Int, scene: GpuRtSceneView, queues: GpuRtTraceQueueView):
    comptime assert ALGORITHM in (Integrator.AO, Integrator.NEE, Integrator.MIS)
    var counters = queues.counters.unsafe_origin_cast[MutAnyOrigin]()
    var capacity = Int(queues.capacity)
    var path_ids = queues.shadow_path_ids.unsafe_mut_cast[
        False
    ]().unsafe_origin_cast[ImmutAnyOrigin]()
    var fields = queues.shadow_fields.unsafe_mut_cast[
        False
    ]().unsafe_origin_cast[ImmutAnyOrigin]()
    var work = load_gpu_rt_shadow[ALGORITHM != .AO](
        path_ids, fields, capacity, idx
    )
    var ray = Rayf32[.WORLD](
        Point3f32[.WORLD](work.ox, work.oy, work.oz),
        Vec3f32[.WORLD](work.dx, work.dy, work.dz),
        work.t_min,
        work.t_max,
    )
    var occluded = _trace_scene_any[
        HAS_SPHERES,
        HAS_TRIANGLES,
        HAS_INSTANCES,
        node_width,
        leaf_width,
        triangle_node_width,
        triangle_leaf_width,
        tlas_node_width,
        tlas_leaf_width,
        blas_node_width,
        blas_leaf_width,
        triangle_layout,
        blas_layout,
    ](scene, ray)
    var sample_radiance = queues.sample_radiance.unsafe_origin_cast[
        MutAnyOrigin
    ]()
    var sample_base = queues.sample_base
    comptime if ALGORITHM == .AO:
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
    ALGORITHM: Integrator,
    HAS_SPHERES: Bool,
    HAS_TRIANGLES: Bool,
    HAS_INSTANCES: Bool,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    triangle_node_width: SIMDLength,
    triangle_leaf_width: SIMDLength,
    tlas_node_width: SIMDLength,
    tlas_leaf_width: SIMDLength,
    blas_node_width: SIMDLength,
    blas_leaf_width: SIMDLength,
    triangle_layout: GpuBvhLayout = .WIDE,
    blas_layout: GpuBvhLayout = .WIDE,
](scene: GpuRtSceneView, queues: GpuRtTraceQueueView):
    var counters = queues.counters.unsafe_origin_cast[MutAnyOrigin]()
    var count = Int(counters[unsafe_offset=WAVE_COUNTER.SHADOW])
    var idx = global_idx.x
    var stride = Int(grid_dim.x * block_dim.x)
    while idx < count:
        _gpu_rt_shadow_one[
            ALGORITHM,
            HAS_SPHERES,
            HAS_TRIANGLES,
            HAS_INSTANCES,
            node_width,
            leaf_width,
            triangle_node_width,
            triangle_leaf_width,
            tlas_node_width,
            tlas_leaf_width,
            blas_node_width,
            blas_leaf_width,
            triangle_layout,
            blas_layout,
        ](idx, scene, queues)
        idx += stride


def enqueue_gpu_shadows[
    ALGORITHM: Integrator,
    HAS_SPHERES: Bool,
    HAS_TRIANGLES: Bool,
    HAS_INSTANCES: Bool,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    triangle_node_width: SIMDLength,
    triangle_leaf_width: SIMDLength,
    tlas_node_width: SIMDLength,
    tlas_leaf_width: SIMDLength,
    blas_node_width: SIMDLength,
    blas_leaf_width: SIMDLength,
    MAX_BLOCKS: Int = GPU_RT_MAX_BLOCKS,
    triangle_layout: GpuBvhLayout = .WIDE,
    blas_layout: GpuBvhLayout = .WIDE,
](
    ctx: DeviceContext,
    scene: GpuRtSceneView,
    queues: GpuRtTraceQueueView,
    capacity: Int,
) raises:
    """Submit the compact visibility stage when the algorithm needs it."""
    comptime if ALGORITHM in (Integrator.AO, Integrator.NEE, Integrator.MIS):
        ctx.enqueue_function[
            gpu_rt_shadow_kernel[
                ALGORITHM,
                HAS_SPHERES,
                HAS_TRIANGLES,
                HAS_INSTANCES,
                node_width,
                leaf_width,
                triangle_node_width,
                triangle_leaf_width,
                tlas_node_width,
                tlas_leaf_width,
                blas_node_width,
                blas_leaf_width,
                triangle_layout,
                blas_layout,
            ]
        ](
            scene,
            queues,
            grid_dim=min(ceildiv(capacity, GPU_RT_BLOCK_SIZE), MAX_BLOCKS),
            block_dim=GPU_RT_BLOCK_SIZE,
        )
