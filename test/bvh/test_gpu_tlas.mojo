from std.benchmark import keep
from std.math import abs
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true, assert_almost_equal
from std.gpu import DeviceContext, DeviceBuffer

from bajo.core.aabb import AABB
from bajo.core.vec import Vec3f32
from bajo.core.transform import Affine3f32
from bajo.bvh.constants import TRACE
from bajo.bvh.types import Ray, Sphere, Instance
from bajo.bvh.host_utils import (
    flatten_rays,
    generate_primary_rays,
    hit_t_for_checksum,
    compute_bounds,
)
from bajo.bvh.gpu.tlas import GpuTlas
from bajo.bvh.gpu.sphere_bvh import GpuSphereBvh
from bajo.bvh.gpu.triangle_bvh import GpuTriangleBvh
from bajo.bvh.gpu.utils import _upload_rays, _download_full_hit_checksum

from fixtures import _append_tri


comptime WIDTH = 64
comptime HEIGHT = 48
comptime VIEWS = 3
comptime EPS = 0.05


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


def _download_tlas_checksum(
    hits_f32: DeviceBuffer[DType.float32],
    hits_u32: DeviceBuffer[DType.uint32],
    ray_count: Int,
) raises -> Tuple[Float64, UInt32, UInt64]:
    var checksum = Float64(0.0)
    var hits = UInt32(0)
    var inst_checksum = UInt64(0)

    with hits_f32.map_to_host() as hf:
        for i in range(ray_count):
            var t = hf[i * 3]
            checksum += hit_t_for_checksum(t)
            if t < 1.0e20:
                hits += 1

    with hits_u32.map_to_host() as hu:
        for i in range(ray_count):
            var inst = UInt32(hu[i * 2 + 1])
            if inst != UInt32(0xFFFFFFFF):
                inst_checksum += UInt64(inst)

    return (checksum, hits, inst_checksum)


def _download_shadow_count(
    flags: DeviceBuffer[DType.uint32], ray_count: Int
) raises -> UInt32:
    var out = UInt32(0)
    with flags.map_to_host() as f:
        for i in range(ray_count):
            if UInt32(f[i]) != 0:
                out += 1
    return out


def _make_spheres() -> Tuple[List[Sphere], AABB]:
    var spheres = List[Sphere](capacity=4)
    var bounds = AABB.invalid()

    spheres.append(Sphere(Vec3f32(0.0, 0.0, 2.0), 1.0))
    spheres.append(Sphere(Vec3f32(4.0, 0.0, 4.0), 1.0))
    spheres.append(Sphere(Vec3f32(-4.0, 0.0, 6.0), 1.0))
    spheres.append(Sphere(Vec3f32(0.0, 4.0, 8.0), 1.0))

    for s in spheres:
        var r = s.radius
        bounds.grow(Vec3f32(s.center.x - r, s.center.y - r, s.center.z - r))
        bounds.grow(Vec3f32(s.center.x + r, s.center.y + r, s.center.z + r))

    return (spheres^, bounds)


def test_gpu_tlas_single_identity_matches_direct_blas() raises:
    var verts = _make_small_scene()
    var bounds = compute_bounds(verts)
    var rays = generate_primary_rays(
        bounds,
        WIDTH,
        HEIGHT,
        VIEWS,
    )
    var rays_flat = flatten_rays(rays)

    var instances = [
        Instance(
            Affine3f32.identity(), Affine3f32.identity(), UInt32(0), bounds
        )
    ]

    with DeviceContext() as ctx:
        var blas = GpuTriangleBvh[4](ctx, verts)
        var tlas = GpuTlas[4](ctx, instances)

        var d_rays = ctx.enqueue_create_buffer[DType.float32](len(rays_flat))
        var d_blas_f32 = ctx.enqueue_create_buffer[DType.float32](len(rays) * 3)
        var d_blas_u32 = ctx.enqueue_create_buffer[DType.uint32](len(rays))
        var d_tlas_f32 = ctx.enqueue_create_buffer[DType.float32](len(rays) * 3)
        var d_tlas_u32 = ctx.enqueue_create_buffer[DType.uint32](len(rays) * 2)
        var d_tlas_flags = ctx.enqueue_create_buffer[DType.uint32](len(rays))

        _upload_rays(ctx, d_rays, rays_flat)

        blas.launch_uploaded(ctx, d_rays, d_blas_f32, d_blas_u32, len(rays))
        tlas.launch_uploaded["triangle", TRACE.CLOSEST_HIT, 4](
            ctx,
            blas.tree.wide_bounds,
            blas.tree.wide_data,
            blas.tree.wide_counts,
            blas.leaf_vertices,
            blas.leaf_prims,
            blas.tree.root_idx,
            d_rays,
            d_tlas_f32,
            d_tlas_u32,
            d_tlas_flags,
            len(rays),
        )
        ctx.synchronize()

        var blas_res = _download_full_hit_checksum(ctx, d_blas_f32, len(rays))
        var tlas_res = _download_tlas_checksum(
            d_tlas_f32, d_tlas_u32, len(rays)
        )

        assert_true(abs(blas_res[0] - tlas_res[0]) <= EPS)
        assert_true(blas_res[1] == tlas_res[1])
        assert_true(tlas_res[2] == UInt64(0))
        keep(tlas.tree.leaf_block_count)


def test_gpu_tlas_translated_single_instance_hit() raises:
    var verts = List[Vec3f32](capacity=3)
    _append_tri(verts, 0.0, 0.0, 2.0)
    var bounds = compute_bounds(verts)

    t = Vec3f32(10, 0, 0)
    var instances = [
        Instance(
            Affine3f32.from_translation(t),
            Affine3f32.from_translation(-t),
            UInt32(0),
            bounds,
        )
    ]

    var rays = [Ray(t, Vec3f32(0.0, 0.0, 1.0))]
    var rays_flat = flatten_rays(rays)

    with DeviceContext() as ctx:
        var blas = GpuTriangleBvh[4](ctx, verts)
        var tlas = GpuTlas[4](ctx, instances)

        var d_rays = ctx.enqueue_create_buffer[DType.float32](len(rays_flat))
        var d_hits_f32 = ctx.enqueue_create_buffer[DType.float32](len(rays) * 3)
        var d_hits_u32 = ctx.enqueue_create_buffer[DType.uint32](len(rays) * 2)
        var d_flags = ctx.enqueue_create_buffer[DType.uint32](len(rays))

        _upload_rays(ctx, d_rays, rays_flat)
        tlas.launch_uploaded["triangle", TRACE.CLOSEST_HIT, 4](
            ctx,
            blas.tree.wide_bounds,
            blas.tree.wide_data,
            blas.tree.wide_counts,
            blas.leaf_vertices,
            blas.leaf_prims,
            blas.tree.root_idx,
            d_rays,
            d_hits_f32,
            d_hits_u32,
            d_flags,
            len(rays),
        )
        ctx.synchronize()

        with d_hits_f32.map_to_host() as hf:
            assert_almost_equal(hf[0], 2.0)

        with d_hits_u32.map_to_host() as hu:
            assert_true(hu[0] == UInt32(0))
            assert_true(hu[1] == UInt32(0))

        keep(tlas.tree.leaf_block_count)


def test_gpu_tlas_sphere_single_identity_matches_direct_blas() raises:
    var scene = _make_spheres()
    var spheres = scene[0].copy()
    var bounds = scene[1].copy()
    var rays = generate_primary_rays(
        bounds,
        WIDTH,
        HEIGHT,
        VIEWS,
    )
    var rays_flat = flatten_rays(rays)
    var instances = [
        Instance(
            Affine3f32.identity(), Affine3f32.identity(), UInt32(0), bounds
        )
    ]

    with DeviceContext() as ctx:
        var blas = GpuSphereBvh[4](ctx, spheres)
        var tlas = GpuTlas[4](ctx, instances)

        var d_rays = ctx.enqueue_create_buffer[DType.float32](len(rays_flat))
        var d_blas_f32 = ctx.enqueue_create_buffer[DType.float32](len(rays) * 3)
        var d_blas_u32 = ctx.enqueue_create_buffer[DType.uint32](len(rays))
        var d_tlas_f32 = ctx.enqueue_create_buffer[DType.float32](len(rays) * 3)
        var d_tlas_u32 = ctx.enqueue_create_buffer[DType.uint32](len(rays) * 2)
        var d_tlas_flags = ctx.enqueue_create_buffer[DType.uint32](len(rays))

        _upload_rays(ctx, d_rays, rays_flat)
        blas.launch_uploaded(ctx, d_rays, d_blas_f32, d_blas_u32, len(rays))
        tlas.launch_uploaded["sphere", TRACE.CLOSEST_HIT, 4](
            ctx,
            blas.tree.wide_bounds,
            blas.tree.wide_data,
            blas.tree.wide_counts,
            blas.leaf_spheres,
            blas.leaf_prims,
            blas.tree.root_idx,
            d_rays,
            d_tlas_f32,
            d_tlas_u32,
            d_tlas_flags,
            len(rays),
        )
        ctx.synchronize()

        var blas_res = _download_full_hit_checksum(ctx, d_blas_f32, len(rays))
        var tlas_res = _download_tlas_checksum(
            d_tlas_f32, d_tlas_u32, len(rays)
        )

        assert_true(abs(blas_res[0] - tlas_res[0]) <= EPS)
        assert_true(blas_res[1] == tlas_res[1])
        assert_true(tlas_res[2] == UInt64(0))
        keep(tlas.tree.leaf_block_count)


def test_gpu_tlas_sphere_translated_single_instance_hit() raises:
    var spheres = [Sphere(Vec3f32(0.0, 0.0, 2.0), 1.0)]
    var bounds = AABB(Vec3f32(-1.0, -1.0, 1.0), Vec3f32(1.0, 1.0, 3.0))
    t = Vec3f32(10, 0, 0)
    var instances = [
        Instance(
            Affine3f32.from_translation(t),
            Affine3f32.from_translation(-t),
            UInt32(0),
            bounds,
        )
    ]

    var rays = [Ray(t, Vec3f32(0.0, 0.0, 1.0))]
    var rays_flat = flatten_rays(rays)

    with DeviceContext() as ctx:
        var blas = GpuSphereBvh[4](ctx, spheres)
        var tlas = GpuTlas[4](ctx, instances)

        var d_rays = ctx.enqueue_create_buffer[DType.float32](len(rays_flat))
        var d_hits_f32 = ctx.enqueue_create_buffer[DType.float32](len(rays) * 3)
        var d_hits_u32 = ctx.enqueue_create_buffer[DType.uint32](len(rays) * 2)
        var d_flags = ctx.enqueue_create_buffer[DType.uint32](len(rays))

        _upload_rays(ctx, d_rays, rays_flat)
        tlas.launch_uploaded["sphere", TRACE.CLOSEST_HIT, 4](
            ctx,
            blas.tree.wide_bounds,
            blas.tree.wide_data,
            blas.tree.wide_counts,
            blas.leaf_spheres,
            blas.leaf_prims,
            blas.tree.root_idx,
            d_rays,
            d_hits_f32,
            d_hits_u32,
            d_flags,
            len(rays),
        )
        ctx.synchronize()

        with d_hits_f32.map_to_host() as hf:
            assert_almost_equal(hf[0], 1.0)

        with d_hits_u32.map_to_host() as hu:
            assert_true(hu[0] == UInt32(0))
            assert_true(hu[1] == UInt32(0))

        keep(tlas.tree.leaf_block_count)


def main() raises:
    comptime if not has_accelerator():
        raise "No Accelerator found"
    TestSuite.discover_tests[__functions_in_module()]().run()
