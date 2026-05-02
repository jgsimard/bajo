from std.benchmark import keep
from std.math import abs, max
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true
from std.gpu import DeviceContext, DeviceBuffer

from bajo.sort.gpu.radix_sort import device_radix_sort_pairs, RadixSortWorkspace
from bajo.core.vec import Vec3f32, normalize
from bajo.core.bvh.gpu.validate import (
    validate_sorted_keys,
    validate_topology,
    validate_refit_bounds,
)
from bajo.core.bvh import (
    compute_bounds,
    copy_list_to_device,
    flatten_rays,
    flatten_vertices,
    generate_primary_rays,
    trace_bvh_primary,
    trace_bvh_shadow,
)
from bajo.core.bvh.cpu.binary_bvh import BinaryBvh, Ray
from bajo.core.bvh.gpu.kernels import (
    GPU_REDUCE_THREADS,
    TRACE_PRIMARY_FULL,
    TRACE_PRIMARY_T,
    TRACE_SHADOW,
    compute_centroid_bounds,
    compute_morton_codes_kernel,
    generate_camera_params,
    init_lbvh_topology_kernel,
    init_lbvh_bounds_kernel,
    build_lbvh_topology_kernel,
    refit_lbvh_bounds_kernel,
    trace_lbvh_gpu_primary_kernel,
    trace_lbvh_gpu_camera_kernel,
    reduce_hit_t_kernel,
    reduce_u32_flags_kernel,
)
from bajo.core.bvh.gpu.utils import _download_full_hit_checksum


comptime GPU_TEST_BLOCK_SIZE = 128
comptime GPU_TEST_WIDTH = 64
comptime GPU_TEST_HEIGHT = 48
comptime GPU_TEST_VIEWS = 3
comptime GPU_TEST_CHECKSUM_EPS = 0.05


@always_inline
def _blocks_for(n: Int) -> Int:
    return (n + GPU_TEST_BLOCK_SIZE - 1) // GPU_TEST_BLOCK_SIZE


@always_inline
def _abs64(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


def _append_tri(
    mut verts: List[Vec3f32],
    cx: Float32,
    cy: Float32,
    z: Float32,
):
    verts.append(Vec3f32(cx - 0.5, cy - 0.5, z))
    verts.append(Vec3f32(cx + 0.5, cy - 0.5, z))
    verts.append(Vec3f32(cx, cy + 0.5, z))


def _make_small_scene() -> List[Vec3f32]:
    var verts = List[Vec3f32](capacity=8 * 3)
    _append_tri(verts, -1.0, -1.0, 2.0)
    _append_tri(verts, 1.0, -1.0, 2.0)
    _append_tri(verts, -1.0, 1.0, 2.0)
    _append_tri(verts, 1.0, 1.0, 2.0)
    _append_tri(verts, -1.0, -1.0, 4.0)
    _append_tri(verts, 1.0, -1.0, 4.0)
    _append_tri(verts, -1.0, 1.0, 4.0)
    _append_tri(verts, 1.0, 1.0, 4.0)
    return verts^


def _make_duplicate_centroid_scene() -> List[Vec3f32]:
    # All triangles have identical bounds/centroids. This stresses duplicate
    # Morton-code tie-breaking in the topology builder.
    var verts = List[Vec3f32](capacity=8 * 3)
    for _ in range(8):
        verts.append(Vec3f32(-0.5, -0.5, 2.0))
        verts.append(Vec3f32(0.5, -0.5, 2.0))
        verts.append(Vec3f32(0.0, 0.5, 2.0))
    return verts^


def _make_degenerate_axis_scene() -> List[Vec3f32]:
    # Coplanar flat triangles give zero thickness on one axis and make sure the
    # Morton normalization/refit path handles zero extents safely.
    var verts = List[Vec3f32](capacity=6 * 3)
    for i in range(6):
        var cx = Float32(i * 2 - 5)
        _append_tri(verts, cx, 0.0, 2.0)
    return verts^


def _build_gpu_lbvh_in_place(
    ctx: DeviceContext,
    mut workspace: RadixSortWorkspace[DType.uint32, DType.uint32],
    d_vertices: DeviceBuffer[DType.float32],
    mut d_keys: DeviceBuffer[DType.uint32],
    mut d_values: DeviceBuffer[DType.uint32],
    d_node_meta: DeviceBuffer[DType.uint32],
    d_leaf_parent: DeviceBuffer[DType.uint32],
    d_node_bounds: DeviceBuffer[DType.float32],
    d_node_flags: DeviceBuffer[DType.uint32],
    tri_count: Int,
    centroid_min: Vec3f32,
    norm: Vec3f32,
) raises:
    var internal_count = tri_count - 1
    var blocks_leaves = _blocks_for(tri_count)
    var blocks_internal = _blocks_for(internal_count)
    var blocks_init = _blocks_for(max(tri_count, internal_count))

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
        norm.x(),
        norm.y(),
        norm.z(),
        grid_dim=blocks_leaves,
        block_dim=GPU_TEST_BLOCK_SIZE,
    )
    ctx.synchronize()

    device_radix_sort_pairs[DType.uint32, DType.uint32](
        ctx, workspace, d_keys, d_values, tri_count
    )
    ctx.synchronize()

    ctx.enqueue_function[init_lbvh_topology_kernel, init_lbvh_topology_kernel](
        d_node_meta.unsafe_ptr(),
        d_leaf_parent.unsafe_ptr(),
        internal_count,
        tri_count,
        grid_dim=blocks_init,
        block_dim=GPU_TEST_BLOCK_SIZE,
    )
    ctx.enqueue_function[
        build_lbvh_topology_kernel, build_lbvh_topology_kernel
    ](
        d_keys.unsafe_ptr(),
        d_node_meta.unsafe_ptr(),
        d_leaf_parent.unsafe_ptr(),
        tri_count,
        grid_dim=blocks_internal,
        block_dim=GPU_TEST_BLOCK_SIZE,
    )
    ctx.synchronize()

    ctx.enqueue_function[init_lbvh_bounds_kernel, init_lbvh_bounds_kernel](
        d_node_bounds.unsafe_ptr(),
        d_node_flags.unsafe_ptr(),
        internal_count,
        grid_dim=blocks_internal,
        block_dim=GPU_TEST_BLOCK_SIZE,
    )
    ctx.enqueue_function[refit_lbvh_bounds_kernel, refit_lbvh_bounds_kernel](
        d_vertices.unsafe_ptr(),
        d_values.unsafe_ptr(),
        d_node_meta.unsafe_ptr(),
        d_leaf_parent.unsafe_ptr(),
        d_node_bounds.unsafe_ptr(),
        d_node_flags.unsafe_ptr(),
        tri_count,
        grid_dim=blocks_leaves,
        block_dim=GPU_TEST_BLOCK_SIZE,
    )
    ctx.synchronize()


def _build_cpu_reference(
    mut verts: List[Vec3f32],
    rays: List[Ray],
) raises -> Tuple[Float64, Int]:
    var bvh = BinaryBvh(verts.unsafe_ptr(), UInt32(len(verts) // 3))
    bvh.build["sah", False]()
    var checksum = trace_bvh_primary(bvh, rays)
    var occluded = trace_bvh_shadow(bvh, rays)
    return (checksum, occluded)


def test_gpu_lbvh_bounds_helpers() raises:
    var verts = _make_small_scene()
    var bounds = compute_bounds(verts)
    var bmin = bounds[0].copy()
    var bmax = bounds[1].copy()

    assert_true(bmin.x() < -1.4, "scene min x")
    assert_true(bmin.y() < -1.4, "scene min y")
    assert_true(bmin.z() == 2.0, "scene min z")
    assert_true(bmax.x() > 1.4, "scene max x")
    assert_true(bmax.y() > 1.4, "scene max y")
    assert_true(bmax.z() == 4.0, "scene max z")

    var cbounds = compute_centroid_bounds(verts)
    var cmin = cbounds[0].copy()
    var cmax = cbounds[1].copy()
    assert_true(cmin.z() == 2.0, "centroid min z")
    assert_true(cmax.z() == 4.0, "centroid max z")


def test_gpu_lbvh_build_validate_small_scene() raises:
    comptime if has_accelerator():
        var verts = _make_small_scene()
        var tri_count = len(verts) // 3
        var internal_count = tri_count - 1
        var vertices = flatten_vertices(verts)
        var bounds = compute_bounds(verts)
        var bmin = bounds[0].copy()
        var bmax = bounds[1].copy()
        var cbounds = compute_centroid_bounds(verts)
        var cmin = cbounds[0].copy()
        var cmax = cbounds[1].copy()
        var norm = normalize(cmax - cmin)

        with DeviceContext() as ctx:
            var d_vertices = copy_list_to_device(ctx, vertices)
            var d_keys = ctx.enqueue_create_buffer[DType.uint32](tri_count)
            var d_values = ctx.enqueue_create_buffer[DType.uint32](tri_count)
            var workspace = RadixSortWorkspace[DType.uint32, DType.uint32](
                ctx, tri_count
            )
            var d_node_meta = ctx.enqueue_create_buffer[DType.uint32](
                internal_count * 4
            )
            var d_leaf_parent = ctx.enqueue_create_buffer[DType.uint32](
                tri_count
            )
            var d_node_bounds = ctx.enqueue_create_buffer[DType.float32](
                internal_count * 12
            )
            var d_node_flags = ctx.enqueue_create_buffer[DType.uint32](
                internal_count
            )

            _build_gpu_lbvh_in_place(
                ctx,
                workspace,
                d_vertices,
                d_keys,
                d_values,
                d_node_meta,
                d_leaf_parent,
                d_node_bounds,
                d_node_flags,
                tri_count,
                cmin,
                norm,
            )

            var sorted = validate_sorted_keys(d_keys, d_values, tri_count)
            var topo = validate_topology(d_node_meta, d_leaf_parent, tri_count)
            var refit = validate_refit_bounds(
                d_node_bounds,
                d_node_flags,
                d_node_meta,
                tri_count,
                bmin,
                bmax,
            )

            assert_true(sorted[0], "keys sorted")
            assert_true(sorted[1], "sorted primitive ids valid")
            assert_true(topo[0], "topology valid")
            assert_true(refit[0], "refit bounds valid")
            assert_true(refit[2] == UInt32(0), "root should be node 0")
            keep(sorted[6] + topo[3] + refit[3])
    else:
        assert_true(False, "No Accelerator found")


def test_gpu_lbvh_duplicate_morton_codes_validate() raises:
    comptime if has_accelerator():
        var verts = _make_duplicate_centroid_scene()
        var tri_count = len(verts) // 3
        var internal_count = tri_count - 1
        var vertices = flatten_vertices(verts)
        var bounds = compute_bounds(verts)
        var bmin = bounds[0].copy()
        var bmax = bounds[1].copy()
        var cbounds = compute_centroid_bounds(verts)
        var cmin = cbounds[0].copy()
        var cmax = cbounds[1].copy()
        var norm = normalize(cmax - cmin)

        with DeviceContext() as ctx:
            var d_vertices = copy_list_to_device(ctx, vertices)
            var d_keys = ctx.enqueue_create_buffer[DType.uint32](tri_count)
            var d_values = ctx.enqueue_create_buffer[DType.uint32](tri_count)
            var workspace = RadixSortWorkspace[DType.uint32, DType.uint32](
                ctx, tri_count
            )
            var d_node_meta = ctx.enqueue_create_buffer[DType.uint32](
                internal_count * 4
            )
            var d_leaf_parent = ctx.enqueue_create_buffer[DType.uint32](
                tri_count
            )
            var d_node_bounds = ctx.enqueue_create_buffer[DType.float32](
                internal_count * 12
            )
            var d_node_flags = ctx.enqueue_create_buffer[DType.uint32](
                internal_count
            )

            _build_gpu_lbvh_in_place(
                ctx,
                workspace,
                d_vertices,
                d_keys,
                d_values,
                d_node_meta,
                d_leaf_parent,
                d_node_bounds,
                d_node_flags,
                tri_count,
                cmin,
                norm,
            )

            var sorted = validate_sorted_keys(d_keys, d_values, tri_count)
            var topo = validate_topology(d_node_meta, d_leaf_parent, tri_count)
            var refit = validate_refit_bounds(
                d_node_bounds,
                d_node_flags,
                d_node_meta,
                tri_count,
                bmin,
                bmax,
            )

            assert_true(sorted[0], "duplicate-code keys sorted")
            assert_true(sorted[1], "duplicate-code primitive ids valid")
            assert_true(topo[0], "duplicate-code topology valid")
            assert_true(refit[0], "duplicate-code refit valid")
            keep(sorted[6] + topo[3] + refit[3])
    else:
        assert_true(False, "No Accelerator found")


def test_gpu_lbvh_zero_extent_axis_validate() raises:
    comptime if has_accelerator():
        var verts = _make_degenerate_axis_scene()
        var tri_count = len(verts) // 3
        var internal_count = tri_count - 1
        var vertices = flatten_vertices(verts)
        var bounds = compute_bounds(verts)
        var bmin = bounds[0].copy()
        var bmax = bounds[1].copy()
        var cbounds = compute_centroid_bounds(verts)
        var cmin = cbounds[0].copy()
        var cmax = cbounds[1].copy()
        var norm = normalize(cmax - cmin)

        with DeviceContext() as ctx:
            var d_vertices = copy_list_to_device(ctx, vertices)
            var d_keys = ctx.enqueue_create_buffer[DType.uint32](tri_count)
            var d_values = ctx.enqueue_create_buffer[DType.uint32](tri_count)
            var workspace = RadixSortWorkspace[DType.uint32, DType.uint32](
                ctx, tri_count
            )
            var d_node_meta = ctx.enqueue_create_buffer[DType.uint32](
                internal_count * 4
            )
            var d_leaf_parent = ctx.enqueue_create_buffer[DType.uint32](
                tri_count
            )
            var d_node_bounds = ctx.enqueue_create_buffer[DType.float32](
                internal_count * 12
            )
            var d_node_flags = ctx.enqueue_create_buffer[DType.uint32](
                internal_count
            )

            _build_gpu_lbvh_in_place(
                ctx,
                workspace,
                d_vertices,
                d_keys,
                d_values,
                d_node_meta,
                d_leaf_parent,
                d_node_bounds,
                d_node_flags,
                tri_count,
                cmin,
                norm,
            )

            var sorted = validate_sorted_keys(d_keys, d_values, tri_count)
            var topo = validate_topology(d_node_meta, d_leaf_parent, tri_count)
            var refit = validate_refit_bounds(
                d_node_bounds,
                d_node_flags,
                d_node_meta,
                tri_count,
                bmin,
                bmax,
            )

            assert_true(sorted[0], "zero-extent keys sorted")
            assert_true(sorted[1], "zero-extent primitive ids valid")
            assert_true(topo[0], "zero-extent topology valid")
            assert_true(refit[0], "zero-extent refit valid")
            keep(sorted[6] + topo[3] + refit[3])
    else:
        assert_true(False, "No Accelerator found")


def test_gpu_lbvh_uploaded_primary_matches_cpu() raises:
    comptime if has_accelerator():
        var verts = _make_small_scene()
        var tri_count = len(verts) // 3
        var internal_count = tri_count - 1
        var vertices = flatten_vertices(verts)
        var bounds = compute_bounds(verts)
        var bmin = bounds[0].copy()
        var bmax = bounds[1].copy()
        var cbounds = compute_centroid_bounds(verts)
        var cmin = cbounds[0].copy()
        var cmax = cbounds[1].copy()
        var norm = normalize(cmax - cmin)
        var rays = generate_primary_rays(
            bmin, bmax, GPU_TEST_WIDTH, GPU_TEST_HEIGHT, GPU_TEST_VIEWS
        )
        var rays_flat = flatten_rays(rays)
        var cpu_ref = _build_cpu_reference(verts, rays)
        var ref_checksum = cpu_ref[0]

        with DeviceContext() as ctx:
            var d_vertices = copy_list_to_device(ctx, vertices)
            var d_keys = ctx.enqueue_create_buffer[DType.uint32](tri_count)
            var d_values = ctx.enqueue_create_buffer[DType.uint32](tri_count)
            var workspace = RadixSortWorkspace[DType.uint32, DType.uint32](
                ctx, tri_count
            )
            var d_node_meta = ctx.enqueue_create_buffer[DType.uint32](
                internal_count * 4
            )
            var d_leaf_parent = ctx.enqueue_create_buffer[DType.uint32](
                tri_count
            )
            var d_node_bounds = ctx.enqueue_create_buffer[DType.float32](
                internal_count * 12
            )
            var d_node_flags = ctx.enqueue_create_buffer[DType.uint32](
                internal_count
            )
            var d_rays = ctx.enqueue_create_buffer[DType.float32](
                len(rays_flat)
            )
            var d_hits_f32 = ctx.enqueue_create_buffer[DType.float32](
                len(rays) * 3
            )
            var d_hits_u32 = ctx.enqueue_create_buffer[DType.uint32](len(rays))

            _build_gpu_lbvh_in_place(
                ctx,
                workspace,
                d_vertices,
                d_keys,
                d_values,
                d_node_meta,
                d_leaf_parent,
                d_node_bounds,
                d_node_flags,
                tri_count,
                cmin,
                norm,
            )
            var refit = validate_refit_bounds(
                d_node_bounds,
                d_node_flags,
                d_node_meta,
                tri_count,
                bmin,
                bmax,
            )
            var root_idx = UInt32(refit[2])

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
                grid_dim=_blocks_for(len(rays)),
                block_dim=GPU_TEST_BLOCK_SIZE,
            )
            ctx.synchronize()

            var gpu = _download_full_hit_checksum(d_hits_f32, len(rays))
            var diff = _abs64(gpu[0] - ref_checksum)
            assert_true(
                diff <= GPU_TEST_CHECKSUM_EPS, "uploaded primary checksum"
            )
            keep(gpu[1])
    else:
        assert_true(False, "No Accelerator found")


def test_gpu_lbvh_camera_full_matches_cpu() raises:
    comptime if has_accelerator():
        var verts = _make_small_scene()
        var tri_count = len(verts) // 3
        var internal_count = tri_count - 1
        var vertices = flatten_vertices(verts)
        var bounds = compute_bounds(verts)
        var bmin = bounds[0].copy()
        var bmax = bounds[1].copy()
        var cbounds = compute_centroid_bounds(verts)
        var cmin = cbounds[0].copy()
        var cmax = cbounds[1].copy()
        var norm = normalize(cmax - cmin)
        var rays = generate_primary_rays(
            bmin, bmax, GPU_TEST_WIDTH, GPU_TEST_HEIGHT, GPU_TEST_VIEWS
        )
        var camera_params = generate_camera_params(bmin, bmax, GPU_TEST_VIEWS)
        var cpu_ref = _build_cpu_reference(verts, rays)
        var ref_checksum = cpu_ref[0]

        with DeviceContext() as ctx:
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
            var d_leaf_parent = ctx.enqueue_create_buffer[DType.uint32](
                tri_count
            )
            var d_node_bounds = ctx.enqueue_create_buffer[DType.float32](
                internal_count * 12
            )
            var d_node_flags = ctx.enqueue_create_buffer[DType.uint32](
                internal_count
            )
            var d_hits_f32 = ctx.enqueue_create_buffer[DType.float32](
                len(rays) * 3
            )
            var d_hits_u32 = ctx.enqueue_create_buffer[DType.uint32](len(rays))

            _build_gpu_lbvh_in_place(
                ctx,
                workspace,
                d_vertices,
                d_keys,
                d_values,
                d_node_meta,
                d_leaf_parent,
                d_node_bounds,
                d_node_flags,
                tri_count,
                cmin,
                norm,
            )
            var refit = validate_refit_bounds(
                d_node_bounds,
                d_node_flags,
                d_node_meta,
                tri_count,
                bmin,
                bmax,
            )
            var root_idx = UInt32(refit[2])

            ctx.enqueue_function[
                trace_lbvh_gpu_camera_kernel[TRACE_PRIMARY_FULL],
                trace_lbvh_gpu_camera_kernel[TRACE_PRIMARY_FULL],
            ](
                d_vertices.unsafe_ptr(),
                d_values.unsafe_ptr(),
                d_node_meta.unsafe_ptr(),
                d_node_bounds.unsafe_ptr(),
                d_camera_params.unsafe_ptr(),
                d_hits_f32.unsafe_ptr(),
                d_hits_u32.unsafe_ptr(),
                len(rays),
                GPU_TEST_WIDTH,
                GPU_TEST_HEIGHT,
                GPU_TEST_VIEWS,
                root_idx,
                grid_dim=_blocks_for(len(rays)),
                block_dim=GPU_TEST_BLOCK_SIZE,
            )
            ctx.synchronize()

            var gpu = _download_full_hit_checksum(d_hits_f32, len(rays))
            var diff = _abs64(gpu[0] - ref_checksum)
            assert_true(diff <= GPU_TEST_CHECKSUM_EPS, "camera full checksum")
            keep(gpu[1])
    else:
        assert_true(False, "No Accelerator found")


def test_gpu_lbvh_camera_t_reduce_matches_cpu() raises:
    comptime if has_accelerator():
        var verts = _make_small_scene()
        var tri_count = len(verts) // 3
        var internal_count = tri_count - 1
        var vertices = flatten_vertices(verts)
        var bounds = compute_bounds(verts)
        var bmin = bounds[0].copy()
        var bmax = bounds[1].copy()
        var cbounds = compute_centroid_bounds(verts)
        var cmin = cbounds[0].copy()
        var cmax = cbounds[1].copy()
        var norm = normalize(cmax - cmin)
        var rays = generate_primary_rays(
            bmin, bmax, GPU_TEST_WIDTH, GPU_TEST_HEIGHT, GPU_TEST_VIEWS
        )
        var camera_params = generate_camera_params(bmin, bmax, GPU_TEST_VIEWS)
        var cpu_ref = _build_cpu_reference(verts, rays)
        var ref_checksum = cpu_ref[0]

        with DeviceContext() as ctx:
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
            var d_leaf_parent = ctx.enqueue_create_buffer[DType.uint32](
                tri_count
            )
            var d_node_bounds = ctx.enqueue_create_buffer[DType.float32](
                internal_count * 12
            )
            var d_node_flags = ctx.enqueue_create_buffer[DType.uint32](
                internal_count
            )
            var d_hit_t = ctx.enqueue_create_buffer[DType.float32](len(rays))
            var d_occluded = ctx.enqueue_create_buffer[DType.uint32](len(rays))
            var d_partial_sums = ctx.enqueue_create_buffer[DType.float64](
                GPU_REDUCE_THREADS
            )
            var d_partial_counts = ctx.enqueue_create_buffer[DType.uint32](
                GPU_REDUCE_THREADS
            )

            _build_gpu_lbvh_in_place(
                ctx,
                workspace,
                d_vertices,
                d_keys,
                d_values,
                d_node_meta,
                d_leaf_parent,
                d_node_bounds,
                d_node_flags,
                tri_count,
                cmin,
                norm,
            )
            var refit = validate_refit_bounds(
                d_node_bounds,
                d_node_flags,
                d_node_meta,
                tri_count,
                bmin,
                bmax,
            )
            var root_idx = UInt32(refit[2])

            ctx.enqueue_function[
                trace_lbvh_gpu_camera_kernel[TRACE_PRIMARY_T],
                trace_lbvh_gpu_camera_kernel[TRACE_PRIMARY_T],
            ](
                d_vertices.unsafe_ptr(),
                d_values.unsafe_ptr(),
                d_node_meta.unsafe_ptr(),
                d_node_bounds.unsafe_ptr(),
                d_camera_params.unsafe_ptr(),
                d_hit_t.unsafe_ptr(),
                d_occluded.unsafe_ptr(),
                len(rays),
                GPU_TEST_WIDTH,
                GPU_TEST_HEIGHT,
                GPU_TEST_VIEWS,
                root_idx,
                grid_dim=_blocks_for(len(rays)),
                block_dim=GPU_TEST_BLOCK_SIZE,
            )
            ctx.synchronize()

            ctx.enqueue_function[reduce_hit_t_kernel, reduce_hit_t_kernel](
                d_hit_t.unsafe_ptr(),
                d_partial_sums.unsafe_ptr(),
                d_partial_counts.unsafe_ptr(),
                len(rays),
                GPU_REDUCE_THREADS,
                grid_dim=_blocks_for(GPU_REDUCE_THREADS),
                block_dim=GPU_TEST_BLOCK_SIZE,
            )
            ctx.synchronize()

            var gpu = _download_reduced_hit_t(d_partial_sums, d_partial_counts)
            var diff = _abs64(gpu[0] - ref_checksum)
            assert_true(
                diff <= GPU_TEST_CHECKSUM_EPS, "camera t-reduce checksum"
            )
            keep(gpu[1])
    else:
        assert_true(False, "No Accelerator found")


def test_gpu_lbvh_camera_shadow_reduce_matches_cpu() raises:
    comptime if has_accelerator():
        var verts = _make_small_scene()
        var tri_count = len(verts) // 3
        var internal_count = tri_count - 1
        var vertices = flatten_vertices(verts)
        var bounds = compute_bounds(verts)
        var bmin = bounds[0].copy()
        var bmax = bounds[1].copy()
        var cbounds = compute_centroid_bounds(verts)
        var cmin = cbounds[0].copy()
        var cmax = cbounds[1].copy()
        var norm = normalize(cmax - cmin)
        var rays = generate_primary_rays(
            bmin, bmax, GPU_TEST_WIDTH, GPU_TEST_HEIGHT, GPU_TEST_VIEWS
        )
        var camera_params = generate_camera_params(bmin, bmax, GPU_TEST_VIEWS)
        var cpu_ref = _build_cpu_reference(verts, rays)
        var ref_occluded = cpu_ref[1]

        with DeviceContext() as ctx:
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
            var d_leaf_parent = ctx.enqueue_create_buffer[DType.uint32](
                tri_count
            )
            var d_node_bounds = ctx.enqueue_create_buffer[DType.float32](
                internal_count * 12
            )
            var d_node_flags = ctx.enqueue_create_buffer[DType.uint32](
                internal_count
            )
            var d_hit_t = ctx.enqueue_create_buffer[DType.float32](len(rays))
            var d_occluded = ctx.enqueue_create_buffer[DType.uint32](len(rays))
            var d_partial_counts = ctx.enqueue_create_buffer[DType.uint32](
                GPU_REDUCE_THREADS
            )

            _build_gpu_lbvh_in_place(
                ctx,
                workspace,
                d_vertices,
                d_keys,
                d_values,
                d_node_meta,
                d_leaf_parent,
                d_node_bounds,
                d_node_flags,
                tri_count,
                cmin,
                norm,
            )
            var refit = validate_refit_bounds(
                d_node_bounds,
                d_node_flags,
                d_node_meta,
                tri_count,
                bmin,
                bmax,
            )
            var root_idx = UInt32(refit[2])

            ctx.enqueue_function[
                trace_lbvh_gpu_camera_kernel[TRACE_SHADOW],
                trace_lbvh_gpu_camera_kernel[TRACE_SHADOW],
            ](
                d_vertices.unsafe_ptr(),
                d_values.unsafe_ptr(),
                d_node_meta.unsafe_ptr(),
                d_node_bounds.unsafe_ptr(),
                d_camera_params.unsafe_ptr(),
                d_hit_t.unsafe_ptr(),
                d_occluded.unsafe_ptr(),
                len(rays),
                GPU_TEST_WIDTH,
                GPU_TEST_HEIGHT,
                GPU_TEST_VIEWS,
                root_idx,
                grid_dim=_blocks_for(len(rays)),
                block_dim=GPU_TEST_BLOCK_SIZE,
            )
            ctx.synchronize()

            ctx.enqueue_function[
                reduce_u32_flags_kernel, reduce_u32_flags_kernel
            ](
                d_occluded.unsafe_ptr(),
                d_partial_counts.unsafe_ptr(),
                len(rays),
                GPU_REDUCE_THREADS,
                grid_dim=_blocks_for(GPU_REDUCE_THREADS),
                block_dim=GPU_TEST_BLOCK_SIZE,
            )
            ctx.synchronize()

            var gpu_occluded = _download_reduced_u32_count(d_partial_counts)
            assert_true(
                Int(gpu_occluded) == ref_occluded, "shadow occlusion count"
            )
            keep(gpu_occluded)
    else:
        assert_true(False, "No Accelerator found")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
