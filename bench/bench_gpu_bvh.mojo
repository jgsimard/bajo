from std.benchmark import keep
from std.math import abs, round
from std.sys import has_accelerator
from std.time import perf_counter_ns
from std.gpu import DeviceContext, DeviceBuffer

from bajo.obj import read_obj, triangulated_indices
from bajo.core.morton import morton3
from bajo.core.vec import Vec3f32, vmin, vmax, cross, length, normalize
from bajo.sort.gpu.radix_sort import device_radix_sort_pairs, RadixSortWorkspace
from bajo.core.utils import ns_to_ms

from bajo.core.bvh import (
    generate_primary_rays,
    trace_bvh_primary,
    trace_bvh_shadow,
)
from bajo.core.bvh.cpu_bvh import BVH, Ray
from bajo.core.bvh import copy_list_to_device, compute_bounds
from bajo.core.bvh.gpu_bvh import (
    compute_centroid_bounds,
    generate_camera_params,
    ns_to_mrays_per_s,
    pack_obj_triangles,
    print_vec3_rounded,
    flatten_vertices,
    compute_morton_codes_kernel,
    init_lbvh_topology_kernel,
    init_lbvh_bounds_kernel,
    build_lbvh_topology_kernel,
    refit_lbvh_bounds_kernel,
    trace_lbvh_gpu_primary_camera_kernel,
    trace_lbvh_gpu_primary_camera_t_kernel,
    trace_lbvh_gpu_shadow_camera_kernel,
    reduce_hit_t_kernel,
    reduce_u32_flags_kernel,
    GPU_REDUCE_THREADS,
    LBVH_SENTINEL,
    LBVH_INDEX_MASK,
    LBVH_LEAF_FLAG,
)

# comptime DEFAULT_OBJ_PATH = "./assets/powerplant/powerplant.obj"
comptime DEFAULT_OBJ_PATH = "./assets/bunny/bunny.obj"
comptime GPU_BLOCK_SIZE = 128
comptime BENCH_REPEATS = 8
comptime PRIMARY_WIDTH = 640
comptime PRIMARY_HEIGHT = 360
comptime PRIMARY_VIEWS = 3


def run_gpu_lbvh_camera_traversal_benchmark(
    tri_vertices: List[Vec3f32],
    centroid_min: Vec3f32,
    centroid_max: Vec3f32,
    scene_min: Vec3f32,
    scene_max: Vec3f32,
    camera_params: List[Float32],
    ray_count: Int,
    reference_checksum: Float64,
    repeats: Int,
) raises -> Tuple[
    Int,
    Int,
    Int,
    Int,
    Int,
    Int,
    Int,
    Float64,
    Bool,
    Bool,
    Bool,
    Float64,
    UInt32,
    UInt64,
]:
    var tri_count = len(tri_vertices) // 3
    var internal_count = tri_count - 1
    var vertices = flatten_vertices(tri_vertices)

    var extent = centroid_max - centroid_min
    var inv_x = Float32(0.0)
    var inv_y = Float32(0.0)
    var inv_z = Float32(0.0)
    if extent.x() > 1.0e-20:
        inv_x = 1.0 / extent.x()
    if extent.y() > 1.0e-20:
        inv_y = 1.0 / extent.y()
    if extent.z() > 1.0e-20:
        inv_z = 1.0 / extent.z()

    with DeviceContext() as ctx:
        var static_t0 = perf_counter_ns()
        var d_vertices = copy_list_to_device(ctx, vertices)
        var d_camera_params = copy_list_to_device(ctx, camera_params)
        var d_keys = ctx.enqueue_create_buffer[DType.uint32](tri_count)
        var d_values = ctx.enqueue_create_buffer[DType.uint32](tri_count)
        var workspace = RadixSortWorkspace[DType.uint32, DType.uint32](
            ctx, tri_count
        )
        var d_node_meta = ctx.enqueue_create_buffer[DType.uint32](
            internal_count * 4
        )
        var d_leaf_parent = ctx.enqueue_create_buffer[DType.uint32](tri_count)
        var d_node_bounds = ctx.enqueue_create_buffer[DType.float32](
            internal_count * 12
        )
        var d_node_flags = ctx.enqueue_create_buffer[DType.uint32](
            internal_count
        )
        var d_hits_f32 = ctx.enqueue_create_buffer[DType.float32](ray_count * 3)
        var d_hits_u32 = ctx.enqueue_create_buffer[DType.uint32](ray_count)
        ctx.synchronize()
        var static_t1 = perf_counter_ns()

        var blocks_leaves = (tri_count + GPU_BLOCK_SIZE - 1) // GPU_BLOCK_SIZE
        var blocks_internal = (
            internal_count + GPU_BLOCK_SIZE - 1
        ) // GPU_BLOCK_SIZE
        var blocks_init = (
            max(tri_count, internal_count) + GPU_BLOCK_SIZE - 1
        ) // GPU_BLOCK_SIZE
        var blocks_rays = (ray_count + GPU_BLOCK_SIZE - 1) // GPU_BLOCK_SIZE

        # Build and refit the LBVH once, keeping all buffers resident for traversal.
        var b0 = perf_counter_ns()
        ctx.enqueue_function[
            compute_morton_codes_kernel, compute_morton_codes_kernel
        ](
            d_vertices.unsafe_ptr(),
            d_keys.unsafe_ptr(),
            d_values.unsafe_ptr(),
            tri_count,
            centroid_min.x(),
            centroid_min.y(),
            centroid_min.z(),
            inv_x,
            inv_y,
            inv_z,
            grid_dim=blocks_leaves,
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.synchronize()
        var m1 = perf_counter_ns()
        device_radix_sort_pairs[DType.uint32, DType.uint32](
            ctx, workspace, d_keys, d_values, tri_count
        )
        ctx.synchronize()
        var s1 = perf_counter_ns()
        ctx.enqueue_function[
            init_lbvh_topology_kernel, init_lbvh_topology_kernel
        ](
            d_node_meta.unsafe_ptr(),
            d_leaf_parent.unsafe_ptr(),
            internal_count,
            tri_count,
            grid_dim=blocks_init,
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.enqueue_function[
            build_lbvh_topology_kernel, build_lbvh_topology_kernel
        ](
            d_keys.unsafe_ptr(),
            d_node_meta.unsafe_ptr(),
            d_leaf_parent.unsafe_ptr(),
            tri_count,
            grid_dim=blocks_internal,
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        ctx.enqueue_function[init_lbvh_bounds_kernel, init_lbvh_bounds_kernel](
            d_node_bounds.unsafe_ptr(),
            d_node_flags.unsafe_ptr(),
            internal_count,
            grid_dim=blocks_internal,
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.enqueue_function[
            refit_lbvh_bounds_kernel, refit_lbvh_bounds_kernel
        ](
            d_vertices.unsafe_ptr(),
            d_values.unsafe_ptr(),
            d_node_meta.unsafe_ptr(),
            d_leaf_parent.unsafe_ptr(),
            d_node_bounds.unsafe_ptr(),
            d_node_flags.unsafe_ptr(),
            tri_count,
            grid_dim=blocks_leaves,
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.synchronize()
        var r1 = perf_counter_ns()

        var sorted_validation = validate_sorted_keys(
            d_keys, d_values, tri_count
        )
        var topo_validation = validate_topology(
            d_node_meta, d_leaf_parent, tri_count
        )
        var refit_validation = validate_refit_bounds(
            d_node_bounds,
            d_node_flags,
            d_node_meta,
            tri_count,
            scene_min,
            scene_max,
        )
        var root_idx = refit_validation[2]

        # Warmup traversal.
        ctx.enqueue_function[
            trace_lbvh_gpu_primary_camera_kernel,
            trace_lbvh_gpu_primary_camera_kernel,
        ](
            d_vertices.unsafe_ptr(),
            d_values.unsafe_ptr(),
            d_node_meta.unsafe_ptr(),
            d_node_bounds.unsafe_ptr(),
            d_camera_params.unsafe_ptr(),
            d_hits_f32.unsafe_ptr(),
            d_hits_u32.unsafe_ptr(),
            ray_count,
            PRIMARY_WIDTH,
            PRIMARY_HEIGHT,
            PRIMARY_VIEWS,
            root_idx,
            grid_dim=blocks_rays,
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.synchronize()

        var best_traversal_ns = Int(9223372036854775807)
        var best_download_ns = Int(9223372036854775807)
        var best_frame_ns = Int(9223372036854775807)
        var checksum = Float64(0.0)
        var hit_count = 0

        for _ in range(repeats):
            var frame0 = perf_counter_ns()

            var k0 = perf_counter_ns()
            ctx.enqueue_function[
                trace_lbvh_gpu_primary_camera_kernel,
                trace_lbvh_gpu_primary_camera_kernel,
            ](
                d_vertices.unsafe_ptr(),
                d_values.unsafe_ptr(),
                d_node_meta.unsafe_ptr(),
                d_node_bounds.unsafe_ptr(),
                d_camera_params.unsafe_ptr(),
                d_hits_f32.unsafe_ptr(),
                d_hits_u32.unsafe_ptr(),
                ray_count,
                PRIMARY_WIDTH,
                PRIMARY_HEIGHT,
                PRIMARY_VIEWS,
                root_idx,
                grid_dim=blocks_rays,
                block_dim=GPU_BLOCK_SIZE,
            )
            ctx.synchronize()
            var k1 = perf_counter_ns()
            var kernel_ns = Int(k1 - k0)
            if kernel_ns < best_traversal_ns:
                best_traversal_ns = kernel_ns

            var d0 = perf_counter_ns()
            checksum = Float64(0.0)
            hit_count = 0
            with d_hits_f32.map_to_host() as h:
                for i in range(ray_count):
                    var t = h[i * 3]
                    if t < 1.0e20:
                        checksum += Float64(t)
                        hit_count += 1
            ctx.synchronize()
            var d1 = perf_counter_ns()
            var download_ns = Int(d1 - d0)
            if download_ns < best_download_ns:
                best_download_ns = download_ns

            var frame1 = perf_counter_ns()
            var frame_ns = Int(frame1 - frame0)
            if frame_ns < best_frame_ns:
                best_frame_ns = frame_ns

        var diff = abs(checksum - reference_checksum)
        var combined_checksum = (
            sorted_validation[6]
            + topo_validation[3]
            + refit_validation[3]
            + UInt64(hit_count)
        )

        return (
            Int(static_t1 - static_t0),
            Int(m1 - b0),
            Int(s1 - m1),
            Int(t1 - s1),
            Int(r1 - t1),
            best_traversal_ns,
            best_download_ns,
            checksum,
            sorted_validation[0] and sorted_validation[1],
            topo_validation[0],
            refit_validation[0],
            diff,
            root_idx,
            combined_checksum,
        )


def run_gpu_lbvh_camera_reduce_and_shadow_benchmark(
    tri_vertices: List[Vec3f32],
    centroid_min: Vec3f32,
    centroid_max: Vec3f32,
    scene_min: Vec3f32,
    scene_max: Vec3f32,
    camera_params: List[Float32],
    ray_count: Int,
    reference_checksum: Float64,
    reference_occluded: Int,
    repeats: Int,
) raises -> Tuple[
    Int,
    Int,
    Int,
    Int,
    Int,
    Int,
    Int,
    Int,
    Float64,
    UInt32,
    Float64,
    Int,
    Int,
    Int,
    UInt32,
    Int,
    Bool,
    Bool,
    Bool,
    UInt32,
    UInt64,
]:
    var tri_count = len(tri_vertices) // 3
    var internal_count = tri_count - 1
    var vertices = flatten_vertices(tri_vertices)

    var extent = centroid_max - centroid_min
    var inv_x = Float32(0.0)
    var inv_y = Float32(0.0)
    var inv_z = Float32(0.0)
    if extent.x() > 1.0e-20:
        inv_x = 1.0 / extent.x()
    if extent.y() > 1.0e-20:
        inv_y = 1.0 / extent.y()
    if extent.z() > 1.0e-20:
        inv_z = 1.0 / extent.z()

    with DeviceContext() as ctx:
        var static_t0 = perf_counter_ns()
        var d_vertices = copy_list_to_device(ctx, vertices)
        var d_camera_params = copy_list_to_device(ctx, camera_params)
        var d_keys = ctx.enqueue_create_buffer[DType.uint32](tri_count)
        var d_values = ctx.enqueue_create_buffer[DType.uint32](tri_count)
        var workspace = RadixSortWorkspace[DType.uint32, DType.uint32](
            ctx, tri_count
        )
        var d_node_meta = ctx.enqueue_create_buffer[DType.uint32](
            internal_count * 4
        )
        var d_leaf_parent = ctx.enqueue_create_buffer[DType.uint32](tri_count)
        var d_node_bounds = ctx.enqueue_create_buffer[DType.float32](
            internal_count * 12
        )
        var d_node_flags = ctx.enqueue_create_buffer[DType.uint32](
            internal_count
        )
        var d_hit_t = ctx.enqueue_create_buffer[DType.float32](ray_count)
        var d_occluded = ctx.enqueue_create_buffer[DType.uint32](ray_count)
        var d_partial_sums = ctx.enqueue_create_buffer[DType.float64](
            GPU_REDUCE_THREADS
        )
        var d_partial_counts = ctx.enqueue_create_buffer[DType.uint32](
            GPU_REDUCE_THREADS
        )
        ctx.synchronize()
        var static_t1 = perf_counter_ns()

        var blocks_leaves = (tri_count + GPU_BLOCK_SIZE - 1) // GPU_BLOCK_SIZE
        var blocks_internal = (
            internal_count + GPU_BLOCK_SIZE - 1
        ) // GPU_BLOCK_SIZE
        var blocks_init = (
            max(tri_count, internal_count) + GPU_BLOCK_SIZE - 1
        ) // GPU_BLOCK_SIZE
        var blocks_rays = (ray_count + GPU_BLOCK_SIZE - 1) // GPU_BLOCK_SIZE
        var reduce_blocks = (
            GPU_REDUCE_THREADS + GPU_BLOCK_SIZE - 1
        ) // GPU_BLOCK_SIZE

        var b0 = perf_counter_ns()
        ctx.enqueue_function[
            compute_morton_codes_kernel, compute_morton_codes_kernel
        ](
            d_vertices.unsafe_ptr(),
            d_keys.unsafe_ptr(),
            d_values.unsafe_ptr(),
            tri_count,
            centroid_min.x(),
            centroid_min.y(),
            centroid_min.z(),
            inv_x,
            inv_y,
            inv_z,
            grid_dim=blocks_leaves,
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.synchronize()
        var m1 = perf_counter_ns()
        device_radix_sort_pairs[DType.uint32, DType.uint32](
            ctx, workspace, d_keys, d_values, tri_count
        )
        ctx.synchronize()
        var s1 = perf_counter_ns()
        ctx.enqueue_function[
            init_lbvh_topology_kernel, init_lbvh_topology_kernel
        ](
            d_node_meta.unsafe_ptr(),
            d_leaf_parent.unsafe_ptr(),
            internal_count,
            tri_count,
            grid_dim=blocks_init,
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.enqueue_function[
            build_lbvh_topology_kernel, build_lbvh_topology_kernel
        ](
            d_keys.unsafe_ptr(),
            d_node_meta.unsafe_ptr(),
            d_leaf_parent.unsafe_ptr(),
            tri_count,
            grid_dim=blocks_internal,
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        ctx.enqueue_function[init_lbvh_bounds_kernel, init_lbvh_bounds_kernel](
            d_node_bounds.unsafe_ptr(),
            d_node_flags.unsafe_ptr(),
            internal_count,
            grid_dim=blocks_internal,
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.enqueue_function[
            refit_lbvh_bounds_kernel, refit_lbvh_bounds_kernel
        ](
            d_vertices.unsafe_ptr(),
            d_values.unsafe_ptr(),
            d_node_meta.unsafe_ptr(),
            d_leaf_parent.unsafe_ptr(),
            d_node_bounds.unsafe_ptr(),
            d_node_flags.unsafe_ptr(),
            tri_count,
            grid_dim=blocks_leaves,
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.synchronize()
        var r1 = perf_counter_ns()

        var sorted_validation = validate_sorted_keys(
            d_keys, d_values, tri_count
        )
        var topo_validation = validate_topology(
            d_node_meta, d_leaf_parent, tri_count
        )
        var refit_validation = validate_refit_bounds(
            d_node_bounds,
            d_node_flags,
            d_node_meta,
            tri_count,
            scene_min,
            scene_max,
        )
        var root_idx = refit_validation[2]

        var best_primary_ns = Int(9223372036854775807)
        var best_primary_reduce_ns = Int(9223372036854775807)
        var best_primary_download_ns = Int(9223372036854775807)
        var primary_checksum = Float64(0.0)
        var primary_hits = UInt32(0)

        for _ in range(repeats):
            var k0 = perf_counter_ns()
            ctx.enqueue_function[
                trace_lbvh_gpu_primary_camera_t_kernel,
                trace_lbvh_gpu_primary_camera_t_kernel,
            ](
                d_vertices.unsafe_ptr(),
                d_values.unsafe_ptr(),
                d_node_meta.unsafe_ptr(),
                d_node_bounds.unsafe_ptr(),
                d_camera_params.unsafe_ptr(),
                d_hit_t.unsafe_ptr(),
                ray_count,
                PRIMARY_WIDTH,
                PRIMARY_HEIGHT,
                PRIMARY_VIEWS,
                root_idx,
                grid_dim=blocks_rays,
                block_dim=GPU_BLOCK_SIZE,
            )
            ctx.synchronize()
            var k1 = perf_counter_ns()
            var ns = Int(k1 - k0)
            if ns < best_primary_ns:
                best_primary_ns = ns

            var rr0 = perf_counter_ns()
            ctx.enqueue_function[reduce_hit_t_kernel, reduce_hit_t_kernel](
                d_hit_t.unsafe_ptr(),
                d_partial_sums.unsafe_ptr(),
                d_partial_counts.unsafe_ptr(),
                ray_count,
                GPU_REDUCE_THREADS,
                grid_dim=reduce_blocks,
                block_dim=GPU_BLOCK_SIZE,
            )
            ctx.synchronize()
            var rr1 = perf_counter_ns()
            ns = Int(rr1 - rr0)
            if ns < best_primary_reduce_ns:
                best_primary_reduce_ns = ns

            var d0 = perf_counter_ns()
            primary_checksum = Float64(0.0)
            primary_hits = UInt32(0)
            with d_partial_sums.map_to_host() as sums:
                for i in range(GPU_REDUCE_THREADS):
                    primary_checksum += sums[i]
            with d_partial_counts.map_to_host() as counts:
                for i in range(GPU_REDUCE_THREADS):
                    primary_hits += counts[i]
            ctx.synchronize()
            var d1 = perf_counter_ns()
            ns = Int(d1 - d0)
            if ns < best_primary_download_ns:
                best_primary_download_ns = ns

        var best_shadow_ns = Int(9223372036854775807)
        var best_shadow_reduce_ns = Int(9223372036854775807)
        var best_shadow_download_ns = Int(9223372036854775807)
        var shadow_occluded = UInt32(0)

        for _ in range(repeats):
            var k0 = perf_counter_ns()
            ctx.enqueue_function[
                trace_lbvh_gpu_shadow_camera_kernel,
                trace_lbvh_gpu_shadow_camera_kernel,
            ](
                d_vertices.unsafe_ptr(),
                d_values.unsafe_ptr(),
                d_node_meta.unsafe_ptr(),
                d_node_bounds.unsafe_ptr(),
                d_camera_params.unsafe_ptr(),
                d_occluded.unsafe_ptr(),
                ray_count,
                PRIMARY_WIDTH,
                PRIMARY_HEIGHT,
                PRIMARY_VIEWS,
                root_idx,
                grid_dim=blocks_rays,
                block_dim=GPU_BLOCK_SIZE,
            )
            ctx.synchronize()
            var k1 = perf_counter_ns()
            var ns = Int(k1 - k0)
            if ns < best_shadow_ns:
                best_shadow_ns = ns

            var rr0 = perf_counter_ns()
            ctx.enqueue_function[
                reduce_u32_flags_kernel, reduce_u32_flags_kernel
            ](
                d_occluded.unsafe_ptr(),
                d_partial_counts.unsafe_ptr(),
                ray_count,
                GPU_REDUCE_THREADS,
                grid_dim=reduce_blocks,
                block_dim=GPU_BLOCK_SIZE,
            )
            ctx.synchronize()
            var rr1 = perf_counter_ns()
            ns = Int(rr1 - rr0)
            if ns < best_shadow_reduce_ns:
                best_shadow_reduce_ns = ns

            var d0 = perf_counter_ns()
            shadow_occluded = UInt32(0)
            with d_partial_counts.map_to_host() as counts:
                for i in range(GPU_REDUCE_THREADS):
                    shadow_occluded += counts[i]
            ctx.synchronize()
            var d1 = perf_counter_ns()
            ns = Int(d1 - d0)
            if ns < best_shadow_download_ns:
                best_shadow_download_ns = ns

        var primary_diff = abs(primary_checksum - reference_checksum)
        var shadow_diff = Int(shadow_occluded) - reference_occluded
        if shadow_diff < 0:
            shadow_diff = -shadow_diff
        var guard = (
            sorted_validation[6]
            + topo_validation[3]
            + refit_validation[3]
            + UInt64(primary_hits)
            + UInt64(shadow_occluded)
        )

        return (
            Int(static_t1 - static_t0),
            Int(m1 - b0),
            Int(s1 - m1),
            Int(t1 - s1),
            Int(r1 - t1),
            best_primary_ns,
            best_primary_reduce_ns,
            best_primary_download_ns,
            primary_checksum,
            primary_hits,
            primary_diff,
            best_shadow_ns,
            best_shadow_reduce_ns,
            best_shadow_download_ns,
            shadow_occluded,
            shadow_diff,
            sorted_validation[0] and sorted_validation[1],
            topo_validation[0],
            refit_validation[0],
            root_idx,
            guard,
        )


def run_gpu_lbvh_refit_benchmark(
    tri_vertices: List[Vec3f32],
    centroid_min: Vec3f32,
    centroid_max: Vec3f32,
    scene_min: Vec3f32,
    scene_max: Vec3f32,
    repeats: Int,
) raises -> Tuple[
    Int,
    Int,
    Int,
    Int,
    Int,
    Bool,
    Bool,
    Bool,
    Int,
    UInt32,
    Bool,
    Float64,
    UInt32,
    UInt64,
]:
    var tri_count = len(tri_vertices) // 3
    var internal_count = tri_count - 1
    var vertices = flatten_vertices(tri_vertices)

    var extent = centroid_max - centroid_min
    var inv_x = Float32(0.0)
    var inv_y = Float32(0.0)
    var inv_z = Float32(0.0)
    if extent.x() > 1.0e-20:
        inv_x = 1.0 / extent.x()
    if extent.y() > 1.0e-20:
        inv_y = 1.0 / extent.y()
    if extent.z() > 1.0e-20:
        inv_z = 1.0 / extent.z()

    with DeviceContext() as ctx:
        var static_t0 = perf_counter_ns()
        var d_vertices = copy_list_to_device(ctx, vertices)
        var d_keys = ctx.enqueue_create_buffer[DType.uint32](tri_count)
        var d_values = ctx.enqueue_create_buffer[DType.uint32](tri_count)
        var workspace = RadixSortWorkspace[DType.uint32, DType.uint32](
            ctx, tri_count
        )
        var d_node_meta = ctx.enqueue_create_buffer[DType.uint32](
            internal_count * 4
        )
        var d_leaf_parent = ctx.enqueue_create_buffer[DType.uint32](tri_count)
        var d_node_bounds = ctx.enqueue_create_buffer[DType.float32](
            internal_count * 12
        )
        var d_node_flags = ctx.enqueue_create_buffer[DType.uint32](
            internal_count
        )
        ctx.synchronize()
        var static_t1 = perf_counter_ns()

        var blocks_leaves = (tri_count + GPU_BLOCK_SIZE - 1) // GPU_BLOCK_SIZE
        var blocks_internal = (
            internal_count + GPU_BLOCK_SIZE - 1
        ) // GPU_BLOCK_SIZE
        var blocks_init = (
            max(tri_count, internal_count) + GPU_BLOCK_SIZE - 1
        ) // GPU_BLOCK_SIZE

        var best_morton_ns = Int(9223372036854775807)
        var best_sort_ns = Int(9223372036854775807)
        var best_topology_ns = Int(9223372036854775807)
        var best_refit_ns = Int(9223372036854775807)

        for _ in range(repeats):
            var m0 = perf_counter_ns()
            ctx.enqueue_function[
                compute_morton_codes_kernel, compute_morton_codes_kernel
            ](
                d_vertices.unsafe_ptr(),
                d_keys.unsafe_ptr(),
                d_values.unsafe_ptr(),
                tri_count,
                centroid_min.x(),
                centroid_min.y(),
                centroid_min.z(),
                inv_x,
                inv_y,
                inv_z,
                grid_dim=blocks_leaves,
                block_dim=GPU_BLOCK_SIZE,
            )
            ctx.synchronize()
            var m1 = perf_counter_ns()
            var morton_ns = Int(m1 - m0)
            if morton_ns < best_morton_ns:
                best_morton_ns = morton_ns

            var s0 = perf_counter_ns()
            device_radix_sort_pairs[DType.uint32, DType.uint32](
                ctx, workspace, d_keys, d_values, tri_count
            )
            ctx.synchronize()
            var s1 = perf_counter_ns()
            var sort_ns = Int(s1 - s0)
            if sort_ns < best_sort_ns:
                best_sort_ns = sort_ns

            var t0 = perf_counter_ns()
            ctx.enqueue_function[
                init_lbvh_topology_kernel, init_lbvh_topology_kernel
            ](
                d_node_meta.unsafe_ptr(),
                d_leaf_parent.unsafe_ptr(),
                internal_count,
                tri_count,
                grid_dim=blocks_init,
                block_dim=GPU_BLOCK_SIZE,
            )
            ctx.enqueue_function[
                build_lbvh_topology_kernel, build_lbvh_topology_kernel
            ](
                d_keys.unsafe_ptr(),
                d_node_meta.unsafe_ptr(),
                d_leaf_parent.unsafe_ptr(),
                tri_count,
                grid_dim=blocks_internal,
                block_dim=GPU_BLOCK_SIZE,
            )
            ctx.synchronize()
            var t1 = perf_counter_ns()
            var topology_ns = Int(t1 - t0)
            if topology_ns < best_topology_ns:
                best_topology_ns = topology_ns

            var r0 = perf_counter_ns()
            ctx.enqueue_function[
                init_lbvh_bounds_kernel, init_lbvh_bounds_kernel
            ](
                d_node_bounds.unsafe_ptr(),
                d_node_flags.unsafe_ptr(),
                internal_count,
                grid_dim=blocks_internal,
                block_dim=GPU_BLOCK_SIZE,
            )
            ctx.enqueue_function[
                refit_lbvh_bounds_kernel, refit_lbvh_bounds_kernel
            ](
                d_vertices.unsafe_ptr(),
                d_values.unsafe_ptr(),
                d_node_meta.unsafe_ptr(),
                d_leaf_parent.unsafe_ptr(),
                d_node_bounds.unsafe_ptr(),
                d_node_flags.unsafe_ptr(),
                tri_count,
                grid_dim=blocks_leaves,
                block_dim=GPU_BLOCK_SIZE,
            )
            ctx.synchronize()
            var r1 = perf_counter_ns()
            var refit_ns = Int(r1 - r0)
            if refit_ns < best_refit_ns:
                best_refit_ns = refit_ns

        # Validation pass using the same buffers.
        ctx.enqueue_function[
            compute_morton_codes_kernel, compute_morton_codes_kernel
        ](
            d_vertices.unsafe_ptr(),
            d_keys.unsafe_ptr(),
            d_values.unsafe_ptr(),
            tri_count,
            centroid_min.x(),
            centroid_min.y(),
            centroid_min.z(),
            inv_x,
            inv_y,
            inv_z,
            grid_dim=blocks_leaves,
            block_dim=GPU_BLOCK_SIZE,
        )
        device_radix_sort_pairs[DType.uint32, DType.uint32](
            ctx, workspace, d_keys, d_values, tri_count
        )
        ctx.enqueue_function[
            init_lbvh_topology_kernel, init_lbvh_topology_kernel
        ](
            d_node_meta.unsafe_ptr(),
            d_leaf_parent.unsafe_ptr(),
            internal_count,
            tri_count,
            grid_dim=blocks_init,
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.enqueue_function[
            build_lbvh_topology_kernel, build_lbvh_topology_kernel
        ](
            d_keys.unsafe_ptr(),
            d_node_meta.unsafe_ptr(),
            d_leaf_parent.unsafe_ptr(),
            tri_count,
            grid_dim=blocks_internal,
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.enqueue_function[init_lbvh_bounds_kernel, init_lbvh_bounds_kernel](
            d_node_bounds.unsafe_ptr(),
            d_node_flags.unsafe_ptr(),
            internal_count,
            grid_dim=blocks_internal,
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.enqueue_function[
            refit_lbvh_bounds_kernel, refit_lbvh_bounds_kernel
        ](
            d_vertices.unsafe_ptr(),
            d_values.unsafe_ptr(),
            d_node_meta.unsafe_ptr(),
            d_leaf_parent.unsafe_ptr(),
            d_node_bounds.unsafe_ptr(),
            d_node_flags.unsafe_ptr(),
            tri_count,
            grid_dim=blocks_leaves,
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.synchronize()

        var sorted_validation = validate_sorted_keys(
            d_keys, d_values, tri_count
        )
        var topo_validation = validate_topology(
            d_node_meta, d_leaf_parent, tri_count
        )
        var refit_validation = validate_refit_bounds(
            d_node_bounds,
            d_node_flags,
            d_node_meta,
            tri_count,
            scene_min,
            scene_max,
        )
        var checksum = (
            sorted_validation[6] + topo_validation[3] + refit_validation[3]
        )

        return (
            Int(static_t1 - static_t0),
            best_morton_ns,
            best_sort_ns,
            best_topology_ns,
            best_refit_ns,
            sorted_validation[0],
            sorted_validation[1],
            topo_validation[0],
            topo_validation[1],
            topo_validation[2],
            refit_validation[0],
            refit_validation[1],
            refit_validation[2],
            checksum,
        )


def run_gpu_lbvh_topology_benchmark(
    tri_vertices: List[Vec3f32],
    centroid_min: Vec3f32,
    centroid_max: Vec3f32,
    repeats: Int,
) raises -> Tuple[
    Int,  # static upload ns
    Int,  # best morton ns
    Int,  # best sort ns
    Int,  # best topology ns
    Bool,  # keys sorted immediately after morton generation
    Bool,  # sampled values valid immediately after morton generation
    Int,  # first bad key index immediately after morton generation
    Int,  # first bad sampled value index immediately after morton generation
    UInt32,
    UInt32,
    UInt64,
    Bool,  # keys sorted immediately after sort
    Bool,  # sampled values valid immediately after sort
    Int,  # first bad key index immediately after sort
    Int,  # first bad sampled value index immediately after sort
    UInt32,
    UInt32,
    UInt64,
    Bool,  # keys sorted after topology
    Bool,  # sampled values valid after topology
    Int,  # first bad key index after topology
    Int,  # first bad sampled value index after topology
    UInt32,
    UInt32,
    UInt64,
    Bool,  # topology valid
    Int,
    UInt32,
    UInt64,
]:
    var tri_count = len(tri_vertices) // 3
    var internal_count = tri_count - 1
    var vertices = flatten_vertices(tri_vertices)

    var extent = centroid_max - centroid_min
    var inv_x = Float32(0.0)
    var inv_y = Float32(0.0)
    var inv_z = Float32(0.0)
    if extent.x() > 1.0e-20:
        inv_x = 1.0 / extent.x()
    if extent.y() > 1.0e-20:
        inv_y = 1.0 / extent.y()
    if extent.z() > 1.0e-20:
        inv_z = 1.0 / extent.z()

    with DeviceContext() as ctx:
        var static_t0 = perf_counter_ns()
        var d_vertices = copy_list_to_device(ctx, vertices)
        var d_keys = ctx.enqueue_create_buffer[DType.uint32](tri_count)
        var d_values = ctx.enqueue_create_buffer[DType.uint32](tri_count)
        var workspace = RadixSortWorkspace[DType.uint32, DType.uint32](
            ctx, tri_count
        )
        var d_node_meta = ctx.enqueue_create_buffer[DType.uint32](
            internal_count * 4
        )
        var d_leaf_parent = ctx.enqueue_create_buffer[DType.uint32](tri_count)
        ctx.synchronize()
        var static_t1 = perf_counter_ns()

        var blocks_leaves = (tri_count + GPU_BLOCK_SIZE - 1) // GPU_BLOCK_SIZE
        var blocks_internal = (
            internal_count + GPU_BLOCK_SIZE - 1
        ) // GPU_BLOCK_SIZE
        var blocks_init = (
            max(tri_count, internal_count) + GPU_BLOCK_SIZE - 1
        ) // GPU_BLOCK_SIZE

        # Warmup.
        ctx.enqueue_function[
            compute_morton_codes_kernel, compute_morton_codes_kernel
        ](
            d_vertices.unsafe_ptr(),
            d_keys.unsafe_ptr(),
            d_values.unsafe_ptr(),
            tri_count,
            centroid_min.x(),
            centroid_min.y(),
            centroid_min.z(),
            inv_x,
            inv_y,
            inv_z,
            grid_dim=blocks_leaves,
            block_dim=GPU_BLOCK_SIZE,
        )
        device_radix_sort_pairs[DType.uint32, DType.uint32](
            ctx, workspace, d_keys, d_values, tri_count
        )
        ctx.enqueue_function[
            init_lbvh_topology_kernel, init_lbvh_topology_kernel
        ](
            d_node_meta.unsafe_ptr(),
            d_leaf_parent.unsafe_ptr(),
            internal_count,
            tri_count,
            grid_dim=blocks_init,
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.enqueue_function[
            build_lbvh_topology_kernel, build_lbvh_topology_kernel
        ](
            d_keys.unsafe_ptr(),
            d_node_meta.unsafe_ptr(),
            d_leaf_parent.unsafe_ptr(),
            tri_count,
            grid_dim=blocks_internal,
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.synchronize()

        var best_morton_ns = Int(9223372036854775807)
        var best_sort_ns = Int(9223372036854775807)
        var best_topology_ns = Int(9223372036854775807)
        var best_total_ns = Int(9223372036854775807)

        for _ in range(repeats):
            var total_t0 = perf_counter_ns()

            var m0 = perf_counter_ns()
            ctx.enqueue_function[
                compute_morton_codes_kernel, compute_morton_codes_kernel
            ](
                d_vertices.unsafe_ptr(),
                d_keys.unsafe_ptr(),
                d_values.unsafe_ptr(),
                tri_count,
                centroid_min.x(),
                centroid_min.y(),
                centroid_min.z(),
                inv_x,
                inv_y,
                inv_z,
                grid_dim=blocks_leaves,
                block_dim=GPU_BLOCK_SIZE,
            )
            ctx.synchronize()
            var m1 = perf_counter_ns()
            var morton_ns = Int(m1 - m0)
            if morton_ns < best_morton_ns:
                best_morton_ns = morton_ns

            var s0 = perf_counter_ns()
            device_radix_sort_pairs[DType.uint32, DType.uint32](
                ctx, workspace, d_keys, d_values, tri_count
            )
            ctx.synchronize()
            var s1 = perf_counter_ns()
            var sort_ns = Int(s1 - s0)
            if sort_ns < best_sort_ns:
                best_sort_ns = sort_ns

            var t0 = perf_counter_ns()
            ctx.enqueue_function[
                init_lbvh_topology_kernel, init_lbvh_topology_kernel
            ](
                d_node_meta.unsafe_ptr(),
                d_leaf_parent.unsafe_ptr(),
                internal_count,
                tri_count,
                grid_dim=blocks_init,
                block_dim=GPU_BLOCK_SIZE,
            )
            ctx.enqueue_function[
                build_lbvh_topology_kernel, build_lbvh_topology_kernel
            ](
                d_keys.unsafe_ptr(),
                d_node_meta.unsafe_ptr(),
                d_leaf_parent.unsafe_ptr(),
                tri_count,
                grid_dim=blocks_internal,
                block_dim=GPU_BLOCK_SIZE,
            )
            ctx.synchronize()
            var t1 = perf_counter_ns()
            var topology_ns = Int(t1 - t0)
            if topology_ns < best_topology_ns:
                best_topology_ns = topology_ns

            var total_t1 = perf_counter_ns()
            var total_ns = Int(total_t1 - total_t0)
            if total_ns < best_total_ns:
                best_total_ns = total_ns

        # Validation pass: regenerate + sort, validate the sorted buffers,
        # then run topology and validate both topology and whether keys survived.
        ctx.enqueue_function[
            compute_morton_codes_kernel, compute_morton_codes_kernel
        ](
            d_vertices.unsafe_ptr(),
            d_keys.unsafe_ptr(),
            d_values.unsafe_ptr(),
            tri_count,
            centroid_min.x(),
            centroid_min.y(),
            centroid_min.z(),
            inv_x,
            inv_y,
            inv_z,
            grid_dim=blocks_leaves,
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.synchronize()
        var sorted_after_morton = validate_sorted_keys(
            d_keys, d_values, tri_count
        )
        if not sorted_after_morton[1]:
            print_values_window(
                "after morton", d_values, sorted_after_morton[3], tri_count
            )

        device_radix_sort_pairs[DType.uint32, DType.uint32](
            ctx, workspace, d_keys, d_values, tri_count
        )
        ctx.synchronize()
        var sorted_after_sort = validate_sorted_keys(
            d_keys, d_values, tri_count
        )
        if not sorted_after_sort[1]:
            print_values_window(
                "after sort", d_values, sorted_after_sort[3], tri_count
            )

        ctx.enqueue_function[
            init_lbvh_topology_kernel, init_lbvh_topology_kernel
        ](
            d_node_meta.unsafe_ptr(),
            d_leaf_parent.unsafe_ptr(),
            internal_count,
            tri_count,
            grid_dim=blocks_init,
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.enqueue_function[
            build_lbvh_topology_kernel, build_lbvh_topology_kernel
        ](
            d_keys.unsafe_ptr(),
            d_node_meta.unsafe_ptr(),
            d_leaf_parent.unsafe_ptr(),
            tri_count,
            grid_dim=blocks_internal,
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.synchronize()
        var sorted_after_topology = validate_sorted_keys(
            d_keys, d_values, tri_count
        )
        if not sorted_after_topology[1]:
            print_values_window(
                "after topology", d_values, sorted_after_topology[3], tri_count
            )
        var topo_validation = validate_topology(
            d_node_meta, d_leaf_parent, tri_count
        )

        return (
            Int(static_t1 - static_t0),
            best_morton_ns,
            best_sort_ns,
            best_topology_ns,
            sorted_after_morton[0],
            sorted_after_morton[1],
            sorted_after_morton[2],
            sorted_after_morton[3],
            sorted_after_morton[4],
            sorted_after_morton[5],
            sorted_after_morton[6],
            sorted_after_sort[0],
            sorted_after_sort[1],
            sorted_after_sort[2],
            sorted_after_sort[3],
            sorted_after_sort[4],
            sorted_after_sort[5],
            sorted_after_sort[6],
            sorted_after_topology[0],
            sorted_after_topology[1],
            sorted_after_topology[2],
            sorted_after_topology[3],
            sorted_after_topology[4],
            sorted_after_topology[5],
            sorted_after_topology[6],
            topo_validation[0],
            topo_validation[1],
            topo_validation[2],
            topo_validation[3],
        )


def validate_sorted_keys(
    keys: DeviceBuffer[DType.uint32],
    values: DeviceBuffer[DType.uint32],
    size: Int,
) raises -> Tuple[Bool, Bool, Int, Int, UInt32, UInt32, UInt64]:
    var keys_sorted = True
    var values_valid = True
    var first_bad_key = -1
    var first_bad_value = -1
    var first_code = UInt32(0)
    var last_code = UInt32(0)
    var checksum = UInt64(0)

    with keys.map_to_host() as k:
        if size > 0:
            first_code = k[0]
            last_code = k[size - 1]

        for i in range(1, size):
            if k[i - 1] > k[i]:
                keys_sorted = False
                if first_bad_key == -1:
                    first_bad_key = i

        # Lightweight checksum so the readback cannot be optimized away.
        for i in range(0, size, 1024):
            checksum += UInt64(k[i])

    with values.map_to_host() as v:
        for i in range(0, size, 1024):
            if v[i] >= UInt32(size):
                values_valid = False
                if first_bad_value == -1:
                    first_bad_value = i
            checksum += UInt64(v[i])

    return (
        keys_sorted,
        values_valid,
        first_bad_key,
        first_bad_value,
        first_code,
        last_code,
        checksum,
    )


def print_values_window(
    label: String,
    values: DeviceBuffer[DType.uint32],
    center: Int,
    size: Int,
) raises:
    if center < 0:
        return

    var lo = center - 4
    if lo < 0:
        lo = 0
    var hi = center + 5
    if hi > size:
        hi = size

    print(t"{label} value window around {center}:")
    with values.map_to_host() as v:
        for i in range(lo, hi):
            print(t"  v[{i}] = {v[i]}")


def validate_topology(
    node_meta: DeviceBuffer[DType.uint32],
    leaf_parent: DeviceBuffer[DType.uint32],
    leaf_count: Int,
) raises -> Tuple[Bool, Int, UInt32, UInt64]:
    var ok = True
    var root_count = 0
    var root_idx = UInt32(0xFFFFFFFF)
    var checksum = UInt64(0)
    var internal_count = leaf_count - 1

    with node_meta.map_to_host() as m:
        for i in range(internal_count):
            var base = i * 4
            var parent = UInt32(m[base + 0])
            var left = UInt32(m[base + 1])
            var right = UInt32(m[base + 2])
            var fence = UInt32(m[base + 3])

            checksum += UInt64(parent)
            checksum += UInt64(left)
            checksum += UInt64(right)
            checksum += UInt64(fence)

            if parent == LBVH_SENTINEL:
                root_count += 1
                root_idx = UInt32(i)
            elif parent >= UInt32(internal_count):
                ok = False

            var left_is_leaf = (left & LBVH_LEAF_FLAG) != 0
            var left_idx = left & LBVH_INDEX_MASK
            if left_is_leaf:
                if left_idx >= UInt32(leaf_count):
                    ok = False
            else:
                if left_idx >= UInt32(internal_count):
                    ok = False
                elif UInt32(m[Int(left_idx) * 4 + 0]) != UInt32(i):
                    ok = False

            var right_is_leaf = (right & LBVH_LEAF_FLAG) != 0
            var right_idx = right & LBVH_INDEX_MASK
            if right_is_leaf:
                if right_idx >= UInt32(leaf_count):
                    ok = False
            else:
                if right_idx >= UInt32(internal_count):
                    ok = False
                elif UInt32(m[Int(right_idx) * 4 + 0]) != UInt32(i):
                    ok = False

    with leaf_parent.map_to_host() as p:
        for i in range(leaf_count):
            var parent = UInt32(p[i])
            checksum += UInt64(parent)
            if parent == LBVH_SENTINEL or parent >= UInt32(internal_count):
                ok = False

    if root_count != 1:
        ok = False

    return (ok, root_count, root_idx, checksum)


def validate_refit_bounds(
    node_bounds: DeviceBuffer[DType.float32],
    node_flags: DeviceBuffer[DType.uint32],
    node_meta: DeviceBuffer[DType.uint32],
    leaf_count: Int,
    scene_min: Vec3f32,
    scene_max: Vec3f32,
) raises -> Tuple[Bool, Float64, UInt32, UInt64]:
    var ok = True
    var internal_count = leaf_count - 1
    var root_idx = UInt32(0xFFFFFFFF)
    var checksum = UInt64(0)

    with node_meta.map_to_host() as m:
        for i in range(internal_count):
            var parent = UInt32(m[i * 4 + 0])
            if parent == LBVH_SENTINEL:
                root_idx = UInt32(i)

    with node_flags.map_to_host() as f:
        for i in range(internal_count):
            var flag = UInt32(f[i])
            checksum += UInt64(flag)
            if flag != UInt32(2):
                ok = False

    var diff = Float64(1.0e30)
    if root_idx != UInt32(0xFFFFFFFF):
        with node_bounds.map_to_host() as b:
            var rb = Int(root_idx) * 12
            var mnx = min(Float32(b[rb + 0]), Float32(b[rb + 6]))
            var mny = min(Float32(b[rb + 1]), Float32(b[rb + 7]))
            var mnz = min(Float32(b[rb + 2]), Float32(b[rb + 8]))
            var mxx = max(Float32(b[rb + 3]), Float32(b[rb + 9]))
            var mxy = max(Float32(b[rb + 4]), Float32(b[rb + 10]))
            var mxz = max(Float32(b[rb + 5]), Float32(b[rb + 11]))

            checksum += UInt64(abs(Float64(mnx)) * 1000.0)
            checksum += UInt64(abs(Float64(mny)) * 1000.0)
            checksum += UInt64(abs(Float64(mnz)) * 1000.0)
            checksum += UInt64(abs(Float64(mxx)) * 1000.0)
            checksum += UInt64(abs(Float64(mxy)) * 1000.0)
            checksum += UInt64(abs(Float64(mxz)) * 1000.0)

            diff = max(
                max(
                    max(
                        abs(Float64(mnx - scene_min.x())),
                        abs(Float64(mny - scene_min.y())),
                    ),
                    max(
                        abs(Float64(mnz - scene_min.z())),
                        abs(Float64(mxx - scene_max.x())),
                    ),
                ),
                max(
                    abs(Float64(mxy - scene_max.y())),
                    abs(Float64(mxz - scene_max.z())),
                ),
            )
    else:
        ok = False

    if diff > 1.0e-4:
        ok = False

    return (ok, diff, root_idx, checksum)


def run_gpu_lbvh_direct_traversal_benchmark(
    tri_vertices: List[Vec3f32],
    centroid_min: Vec3f32,
    centroid_max: Vec3f32,
    scene_min: Vec3f32,
    scene_max: Vec3f32,
    rays: List[Ray],
    reference_checksum: Float64,
    repeats: Int,
) raises -> Tuple[
    Int,
    Int,
    Int,
    Int,
    Int,
    Int,
    Int,
    Int,
    Int,
    Float64,
    Bool,
    Bool,
    Bool,
    Float64,
    UInt32,
    UInt64,
]:
    var tri_count = len(tri_vertices) // 3
    var internal_count = tri_count - 1
    var vertices = flatten_vertices(tri_vertices)
    var rays_flat = flatten_rays(rays)

    var extent = centroid_max - centroid_min
    var inv_x = Float32(0.0)
    var inv_y = Float32(0.0)
    var inv_z = Float32(0.0)
    if extent.x() > 1.0e-20:
        inv_x = 1.0 / extent.x()
    if extent.y() > 1.0e-20:
        inv_y = 1.0 / extent.y()
    if extent.z() > 1.0e-20:
        inv_z = 1.0 / extent.z()

    with DeviceContext() as ctx:
        var static_t0 = perf_counter_ns()
        var d_vertices = copy_list_to_device(ctx, vertices)
        var d_keys = ctx.enqueue_create_buffer[DType.uint32](tri_count)
        var d_values = ctx.enqueue_create_buffer[DType.uint32](tri_count)
        var workspace = RadixSortWorkspace[DType.uint32, DType.uint32](
            ctx, tri_count
        )
        var d_node_meta = ctx.enqueue_create_buffer[DType.uint32](
            internal_count * 4
        )
        var d_leaf_parent = ctx.enqueue_create_buffer[DType.uint32](tri_count)
        var d_node_bounds = ctx.enqueue_create_buffer[DType.float32](
            internal_count * 12
        )
        var d_node_flags = ctx.enqueue_create_buffer[DType.uint32](
            internal_count
        )
        var d_rays = ctx.enqueue_create_buffer[DType.float32](len(rays_flat))
        var d_hits_f32 = ctx.enqueue_create_buffer[DType.float32](len(rays) * 3)
        var d_hits_u32 = ctx.enqueue_create_buffer[DType.uint32](len(rays))
        ctx.synchronize()
        var static_t1 = perf_counter_ns()

        var blocks_leaves = (tri_count + GPU_BLOCK_SIZE - 1) // GPU_BLOCK_SIZE
        var blocks_internal = (
            internal_count + GPU_BLOCK_SIZE - 1
        ) // GPU_BLOCK_SIZE
        var blocks_init = (
            max(tri_count, internal_count) + GPU_BLOCK_SIZE - 1
        ) // GPU_BLOCK_SIZE
        var blocks_rays = (len(rays) + GPU_BLOCK_SIZE - 1) // GPU_BLOCK_SIZE

        # Build and refit the LBVH once, keeping all buffers resident for traversal.
        var b0 = perf_counter_ns()
        ctx.enqueue_function[
            compute_morton_codes_kernel, compute_morton_codes_kernel
        ](
            d_vertices.unsafe_ptr(),
            d_keys.unsafe_ptr(),
            d_values.unsafe_ptr(),
            tri_count,
            centroid_min.x(),
            centroid_min.y(),
            centroid_min.z(),
            inv_x,
            inv_y,
            inv_z,
            grid_dim=blocks_leaves,
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.synchronize()
        var m1 = perf_counter_ns()
        device_radix_sort_pairs[DType.uint32, DType.uint32](
            ctx, workspace, d_keys, d_values, tri_count
        )
        ctx.synchronize()
        var s1 = perf_counter_ns()
        ctx.enqueue_function[
            init_lbvh_topology_kernel, init_lbvh_topology_kernel
        ](
            d_node_meta.unsafe_ptr(),
            d_leaf_parent.unsafe_ptr(),
            internal_count,
            tri_count,
            grid_dim=blocks_init,
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.enqueue_function[
            build_lbvh_topology_kernel, build_lbvh_topology_kernel
        ](
            d_keys.unsafe_ptr(),
            d_node_meta.unsafe_ptr(),
            d_leaf_parent.unsafe_ptr(),
            tri_count,
            grid_dim=blocks_internal,
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        ctx.enqueue_function[init_lbvh_bounds_kernel, init_lbvh_bounds_kernel](
            d_node_bounds.unsafe_ptr(),
            d_node_flags.unsafe_ptr(),
            internal_count,
            grid_dim=blocks_internal,
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.enqueue_function[
            refit_lbvh_bounds_kernel, refit_lbvh_bounds_kernel
        ](
            d_vertices.unsafe_ptr(),
            d_values.unsafe_ptr(),
            d_node_meta.unsafe_ptr(),
            d_leaf_parent.unsafe_ptr(),
            d_node_bounds.unsafe_ptr(),
            d_node_flags.unsafe_ptr(),
            tri_count,
            grid_dim=blocks_leaves,
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.synchronize()
        var r1 = perf_counter_ns()

        var sorted_validation = validate_sorted_keys(
            d_keys, d_values, tri_count
        )
        var topo_validation = validate_topology(
            d_node_meta, d_leaf_parent, tri_count
        )
        var refit_validation = validate_refit_bounds(
            d_node_bounds,
            d_node_flags,
            d_node_meta,
            tri_count,
            scene_min,
            scene_max,
        )
        var root_idx = refit_validation[2]

        # Warmup traversal.
        with d_rays.map_to_host() as h:
            for i in range(len(rays_flat)):
                h[i] = rays_flat[i]
        ctx.synchronize()
        ctx.enqueue_function[
            trace_lbvh_gpu_primary_kernel, trace_lbvh_gpu_primary_kernel
        ](
            d_vertices.unsafe_ptr(),
            d_values.unsafe_ptr(),
            d_node_meta.unsafe_ptr(),
            d_node_bounds.unsafe_ptr(),
            d_rays.unsafe_ptr(),
            d_hits_f32.unsafe_ptr(),
            d_hits_u32.unsafe_ptr(),
            len(rays),
            root_idx,
            grid_dim=blocks_rays,
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.synchronize()

        var best_ray_upload_ns = Int(9223372036854775807)
        var best_traversal_ns = Int(9223372036854775807)
        var best_download_ns = Int(9223372036854775807)
        var best_frame_ns = Int(9223372036854775807)
        var checksum = Float64(0.0)
        var hit_count = 0

        for _ in range(repeats):
            var frame0 = perf_counter_ns()

            var u0 = perf_counter_ns()
            with d_rays.map_to_host() as h:
                for i in range(len(rays_flat)):
                    h[i] = rays_flat[i]
            ctx.synchronize()
            var u1 = perf_counter_ns()
            var upload_ns = Int(u1 - u0)
            if upload_ns < best_ray_upload_ns:
                best_ray_upload_ns = upload_ns

            var k0 = perf_counter_ns()
            ctx.enqueue_function[
                trace_lbvh_gpu_primary_kernel, trace_lbvh_gpu_primary_kernel
            ](
                d_vertices.unsafe_ptr(),
                d_values.unsafe_ptr(),
                d_node_meta.unsafe_ptr(),
                d_node_bounds.unsafe_ptr(),
                d_rays.unsafe_ptr(),
                d_hits_f32.unsafe_ptr(),
                d_hits_u32.unsafe_ptr(),
                len(rays),
                root_idx,
                grid_dim=blocks_rays,
                block_dim=GPU_BLOCK_SIZE,
            )
            ctx.synchronize()
            var k1 = perf_counter_ns()
            var kernel_ns = Int(k1 - k0)
            if kernel_ns < best_traversal_ns:
                best_traversal_ns = kernel_ns

            var d0 = perf_counter_ns()
            checksum = Float64(0.0)
            hit_count = 0
            with d_hits_f32.map_to_host() as h:
                for i in range(len(rays)):
                    var t = h[i * 3]
                    if t < 1.0e20:
                        checksum += Float64(t)
                        hit_count += 1
            ctx.synchronize()
            var d1 = perf_counter_ns()
            var download_ns = Int(d1 - d0)
            if download_ns < best_download_ns:
                best_download_ns = download_ns

            var frame1 = perf_counter_ns()
            var frame_ns = Int(frame1 - frame0)
            if frame_ns < best_frame_ns:
                best_frame_ns = frame_ns

        var diff = abs(checksum - reference_checksum)
        var combined_checksum = (
            sorted_validation[6]
            + topo_validation[3]
            + refit_validation[3]
            + UInt64(hit_count)
        )

        return (
            Int(static_t1 - static_t0),
            Int(m1 - b0),
            Int(s1 - m1),
            Int(t1 - s1),
            Int(r1 - t1),
            best_ray_upload_ns,
            best_traversal_ns,
            best_download_ns,
            best_frame_ns,
            checksum,
            sorted_validation[0] and sorted_validation[1],
            topo_validation[0],
            refit_validation[0],
            diff,
            root_idx,
            combined_checksum,
        )


def main() raises:
    print("Best GPU LBVH benchmark")
    print(t"Path: {DEFAULT_OBJ_PATH}")
    print(t"Image rays: {PRIMARY_WIDTH} x {PRIMARY_HEIGHT} x {PRIMARY_VIEWS}")
    print(t"Repeats: {BENCH_REPEATS}")

    print("\nLoading + packing OBJ...")
    var load_t0 = perf_counter_ns()
    var tri_vertices = pack_obj_triangles(DEFAULT_OBJ_PATH)
    var load_t1 = perf_counter_ns()

    var tri_count = len(tri_vertices) // 3
    var internal_count = tri_count - 1
    var bounds = compute_bounds(tri_vertices)
    var bmin = bounds[0].copy()
    var bmax = bounds[1].copy()
    var cbounds = compute_centroid_bounds(tri_vertices)
    var cmin = cbounds[0].copy()
    var cmax = cbounds[1].copy()
    var load_ms = round(ns_to_ms(Int(load_t1 - load_t0)), 3)

    print(t"Packed vertices: {len(tri_vertices)}")
    print(t"Triangles: {tri_count}")
    print(t"Internal nodes: {internal_count}")
    print(t"Load+pack ms: {load_ms}")
    print_vec3_rounded("Bounds min:", bmin)
    print_vec3_rounded("Bounds max:", bmax)
    print_vec3_rounded("Centroid min:", cmin)
    print_vec3_rounded("Centroid max:", cmax)

    print("\nGenerating reference rays and camera parameters...")
    var rays = generate_primary_rays(
        bmin, bmax, PRIMARY_WIDTH, PRIMARY_HEIGHT, PRIMARY_VIEWS
    )
    var camera_params = generate_camera_params(bmin, bmax, PRIMARY_VIEWS)
    print(t"Rays: {len(rays)}")
    print(t"Camera params floats: {len(camera_params)}")

    print("\nCPU reference")
    print("-------------")
    var ref_build_t0 = perf_counter_ns()
    var ref_bvh = BVH(tri_vertices.unsafe_ptr(), UInt32(tri_count))
    ref_bvh.build["sah", True]()
    var ref_build_t1 = perf_counter_ns()

    var ref_trace_t0 = perf_counter_ns()
    var ref_checksum = trace_bvh_primary(ref_bvh, rays)
    var ref_occluded = trace_bvh_shadow(ref_bvh, rays)
    var ref_trace_t1 = perf_counter_ns()

    print(
        t"SAH MT build:       "
        t" {round(ns_to_ms(Int(ref_build_t1 - ref_build_t0)), 3)} ms"
    )
    print(
        t"reference queries:  "
        t" {round(ns_to_ms(Int(ref_trace_t1 - ref_trace_t0)), 3)} ms |"
        t" checksum: {round(ref_checksum, 3)} | occluded: {ref_occluded}"
    )

    print("\nBest GPU LBVH path")
    print("------------------")
    comptime if has_accelerator():
        var gpu_result = run_gpu_lbvh_camera_reduce_and_shadow_benchmark(
            tri_vertices,
            cmin,
            cmax,
            bmin,
            bmax,
            camera_params,
            len(rays),
            ref_checksum,
            ref_occluded,
            BENCH_REPEATS,
        )

        var static_upload_ns = gpu_result[0]
        var morton_ns = gpu_result[1]
        var sort_ns = gpu_result[2]
        var topology_ns = gpu_result[3]
        var refit_ns = gpu_result[4]

        var primary_kernel_ns = gpu_result[5]
        var primary_reduce_ns = gpu_result[6]
        var primary_download_ns = gpu_result[7]
        var primary_checksum = gpu_result[8]
        var primary_hits = gpu_result[9]
        var primary_diff = gpu_result[10]

        var shadow_kernel_ns = gpu_result[11]
        var shadow_reduce_ns = gpu_result[12]
        var shadow_download_ns = gpu_result[13]
        var shadow_occluded = gpu_result[14]
        var shadow_diff = gpu_result[15]

        var sorted_ok = gpu_result[16]
        var topology_ok = gpu_result[17]
        var bounds_ok = gpu_result[18]
        var root_idx = gpu_result[19]
        var guard = gpu_result[20]

        var build_ns = morton_ns + sort_ns + topology_ns + refit_ns
        var primary_frame_ns = (
            primary_kernel_ns + primary_reduce_ns + primary_download_ns
        )
        var shadow_frame_ns = (
            shadow_kernel_ns + shadow_reduce_ns + shadow_download_ns
        )
        var primary_total_ns = build_ns + primary_frame_ns
        var shadow_total_ns = build_ns + shadow_frame_ns

        print(t"static upload once:  {round(ns_to_ms(static_upload_ns), 3)} ms")
        print(
            t"build valid:         sorted={sorted_ok} |"
            t" topology={topology_ok} | bounds={bounds_ok} | root={root_idx}"
        )

        print("\nGPU LBVH build")
        print(t"morton generation: {round(ns_to_ms(morton_ns), 3)} ms")
        print(t"radix sort pairs:  {round(ns_to_ms(sort_ns), 3)} ms")
        print(t"topology build:    {round(ns_to_ms(topology_ns), 3)} ms")
        print(t"bounds refit:      {round(ns_to_ms(refit_ns), 3)} ms")
        print(t"build total:       {round(ns_to_ms(build_ns), 3)} ms")

        print("\nGenerated primary rays + GPU checksum")
        print(
            t"camera kernel:       {round(ns_to_ms(primary_kernel_ns), 3)} ms"
            t" | {round(ns_to_mrays_per_s(primary_kernel_ns, len(rays)), 3)} MRays/s"
        )
        print(
            t"checksum reduction:  {round(ns_to_ms(primary_reduce_ns), 3)} ms"
        )
        print(
            t"partial download:    {round(ns_to_ms(primary_download_ns), 3)} ms"
        )
        print(
            t"query total:         {round(ns_to_ms(primary_frame_ns), 3)} ms"
            t" | {round(ns_to_mrays_per_s(primary_frame_ns, len(rays)), 3)} MRays/s"
        )
        print(
            t"build + query total: {round(ns_to_ms(primary_total_ns), 3)} ms"
            t" | {round(ns_to_mrays_per_s(primary_total_ns, len(rays)), 3)} MRays/s"
        )
        print(
            t"validation diff:     {round(primary_diff, 3)}"
            t" | checksum: {round(primary_checksum, 3)} | hits: {primary_hits}"
        )

        print("\nGenerated shadow rays + GPU occlusion count")
        print(
            t"shadow kernel:       {round(ns_to_ms(shadow_kernel_ns), 3)} ms"
            t" | {round(ns_to_mrays_per_s(shadow_kernel_ns, len(rays)), 3)} MRays/s"
        )
        print(t"occlusion reduction: {round(ns_to_ms(shadow_reduce_ns), 3)} ms")
        print(
            t"partial download:    {round(ns_to_ms(shadow_download_ns), 3)} ms"
        )
        print(
            t"query total:         {round(ns_to_ms(shadow_frame_ns), 3)} ms"
            t" | {round(ns_to_mrays_per_s(shadow_frame_ns, len(rays)), 3)} MRays/s"
        )
        print(
            t"build + query total: {round(ns_to_ms(shadow_total_ns), 3)} ms"
            t" | {round(ns_to_mrays_per_s(shadow_total_ns, len(rays)), 3)} MRays/s"
        )
        print(
            t"validation diff:     {shadow_diff}"
            t" | occluded: {shadow_occluded} | ref: {ref_occluded}"
        )
        print(t"checksum guard:      {guard}")
    else:
        print("No compatible GPU found; skipped Mojo GPU benchmark.")
