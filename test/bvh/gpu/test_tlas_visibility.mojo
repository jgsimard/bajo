"""TLAS2 visibility and closest hits agree with transformed primitive tests."""

from std.math import ceildiv
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_almost_equal,
)
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from bajo.core import (
    Affine3f32,
    Point3f32,
    Rayf32,
    Vec3f32,
    GeoKind,
    cross,
    normalize,
)
from bajo.core.intersect import intersect_ray_sphere, intersect_ray_tri
from bajo.bvh.constants import PrimitiveKind, TraceMode
from bajo.bvh.types import Hit, Instance
from bajo.core.geometry import Sphere, sphere_bounds
from bajo.core.aabb import compute_bounds

from bajo.bvh.gpu.ray_launch import _load_packed_ray, _store_packed_hit
from bajo.bvh.gpu.sphere_bvh import (
    build_gpu_sphere_blas_set,
    _intersect_sphere_leaf,
)
from bajo.bvh.gpu.triangle_bvh import build_gpu_triangle_blas_set
from bajo.bvh.gpu.triangle_trace import _intersect_triangle_leaf
from bajo.bvh.gpu.tlas import _trace_tlas_ray, build_gpu_tlas
from bajo.bvh.gpu.utils import upload_rays
from bajo.bvh.gpu import GpuBlasSet
from bajo.bvh.gpu.trace import GpuLeafFn


def _visibility_kernel[
    leaf_fn: GpuLeafFn[.LOCAL],
    mode: TraceMode = .ANY_HIT,
    closest_leaf_tasks: Int = 0,
](
    nodes: Pointer[Float32, ImmutAnyOrigin],
    leaf_instances: Pointer[UInt32, ImmutAnyOrigin],
    inverse_transforms: Pointer[Float32, ImmutAnyOrigin],
    blas_indices: Pointer[UInt32, ImmutAnyOrigin],
    descriptors: Pointer[UInt32, ImmutAnyOrigin],
    blas_nodes: Pointer[Float32, ImmutAnyOrigin],
    blas_leaves: Pointer[Float32, ImmutAnyOrigin],
    rays: Pointer[Float32, ImmutAnyOrigin],
    output: Pointer[UInt32, MutAnyOrigin],
    root: UInt32,
    instance_count: Int32,
    ray_count: Int32,
):
    var idx = global_idx.x
    if idx >= Int(ray_count):
        return
    var hit = _trace_tlas_ray[
        2, 1, 4, 4, mode, leaf_fn, closest_leaf_tasks=closest_leaf_tasks
    ](
        nodes,
        leaf_instances,
        inverse_transforms,
        blas_indices,
        descriptors,
        blas_nodes,
        blas_leaves,
        Int(instance_count),
        1,
        root,
        _load_packed_ray[.WORLD](rays, Int(ray_count), idx),
    )
    comptime if mode == .CLOSEST_HIT:
        _store_packed_hit(
            hit, output.unsafe_bitcast[Float32](), Int(ray_count), idx
        )
    else:
        output[unsafe_offset=idx] = UInt32(hit.is_occluded())


def _assert_visibility[
    kind: PrimitiveKind,
    mode: TraceMode = .ANY_HIT,
    closest_leaf_tasks: Int = 0,
]() raises:
    var vertices: List[Point3f32[.LOCAL]] = [
        Point3f32[.LOCAL](-0.45, -0.45, 0.0),
        Point3f32[.LOCAL](0.45, -0.45, 0.0),
        Point3f32[.LOCAL](0.45, 0.45, 0.0),
        Point3f32[.LOCAL](-0.45, -0.45, 0.0),
        Point3f32[.LOCAL](0.45, 0.45, 0.0),
        Point3f32[.LOCAL](-0.45, 0.45, 0.0),
    ]
    var spheres: List[Sphere[.LOCAL]] = [Sphere(Point3f32[.LOCAL](0.0), 0.45)]
    var bounds = compute_bounds(vertices)
    comptime if kind == .SPHERE:
        bounds = sphere_bounds(spheres)
    var instances = List[Instance]()
    for y in range(8):
        for x in range(8):
            var transform = Affine3f32[.LOCAL, .WORLD].from_scale(
                Vec3f32[.LOCAL](
                    Float32(0.8) if x % 2 else Float32(1.4),
                    Float32(1.3) if y % 2 else Float32(0.6),
                    Float32(0.7) if (x + y) % 2 else Float32(1.6),
                )
            )
            transform.tx = (Float32(x) - 3.5) * 1.2
            transform.ty = (Float32(y) - 3.5) * 1.1
            transform.tz = 4.0 + 1.2 * Float32((x + 3 * y) % 5)
            instances.append(Instance(transform, 0, bounds, kind))
    var rays = List[Rayf32[.WORLD]]()
    for y in range(17):
        for x in range(17):
            var ox = Float32(x) * 0.57 - 4.48
            var oy = Float32(y) * 0.53 - 4.35
            var t_max = Float32(4 + (x + 2 * y) % 9)
            rays.append(
                Rayf32[.WORLD](
                    Point3f32[.WORLD](ox, oy, 0.0),
                    Vec3f32[.WORLD](0.08, 0.01, 1.0),
                    0.001,
                    t_max,
                )
            )
            rays.append(
                Rayf32[.WORLD](
                    Point3f32[.WORLD](ox, oy, 15.0),
                    Vec3f32[.WORLD](0.03, -0.06, -1.0),
                    0.001,
                    t_max,
                )
            )
    # Parallel slabs, signed-zero directions, clipped intervals, and a ray
    # starting inside an instance exercise cases that images rarely isolate.
    for sign in [Float32(0.0), Float32(-0.0)]:
        rays.append(
            Rayf32[.WORLD](
                Point3f32[.WORLD](0.6, 0.55, 0.0),
                Vec3f32[.WORLD](sign, 0.0, 1.0),
                0.001,
                20.0,
            )
        )
        rays.append(
            Rayf32[.WORLD](
                Point3f32[.WORLD](0.6, 0.55, 0.0),
                Vec3f32[.WORLD](sign, 0.0, 1.0),
                10.0,
                20.0,
            )
        )
    rays.append(
        Rayf32[.WORLD](
            Point3f32[.WORLD](0.6, 0.55, 5.2),
            Vec3f32[.WORLD](0.0, 0.0, 1.0),
            0.001,
            20.0,
        )
    )
    var expected = List[UInt32]()
    var expected_hits = List[Hit[.WORLD]]()
    var blocked_count = 0
    for ray in rays:
        var blocked = False
        var closest = Hit[.WORLD].miss(ray.t_max)
        for instance_idx, instance in enumerate(instances):
            var local = instance.inv_transform.ray(ray, ray.t_max)
            comptime if kind == .SPHERE:
                var candidate = intersect_ray_sphere(
                    local.o,
                    local.d,
                    spheres[0].center,
                    spheres[0].radius,
                    local.t_max,
                    local.t_min,
                )
                blocked |= candidate.mask[0]
                comptime if mode == .CLOSEST_HIT:
                    if candidate.mask and candidate.t < closest.t:
                        closest.t = candidate.t
                        closest.prim = 0
                        closest.inst = UInt32(instance_idx)
                        var local_normal = normalize(
                            local.at(candidate.t) - spheres[0].center
                        ).unsafe_convert[new_kind=GeoKind.NORMAL]()
                        closest.normal = Affine3f32[
                            .LOCAL, .WORLD
                        ].normal_from_inverse(
                            local_normal, instance.inv_transform
                        )
            else:
                for tri in range(2):
                    var candidate = intersect_ray_tri(
                        local.o,
                        local.d,
                        vertices[3 * tri],
                        vertices[3 * tri + 1],
                        vertices[3 * tri + 2],
                        local.t_max,
                        local.t_min,
                    )
                    blocked |= candidate.mask[0]
                    comptime if mode == .CLOSEST_HIT:
                        if candidate.mask and candidate.t < closest.t:
                            closest.t = candidate.t
                            closest.u = candidate.u
                            closest.v = candidate.v
                            closest.prim = UInt32(tri)
                            closest.inst = UInt32(instance_idx)
                            var local_normal = normalize(
                                cross(
                                    vertices[3 * tri + 1] - vertices[3 * tri],
                                    vertices[3 * tri + 2] - vertices[3 * tri],
                                )
                            ).unsafe_convert[new_kind=GeoKind.NORMAL]()
                            closest.normal = Affine3f32[
                                .LOCAL, .WORLD
                            ].normal_from_inverse(
                                local_normal, instance.inv_transform
                            )
        expected.append(UInt32(blocked))
        expected_hits.append(closest)
        blocked_count += Int(blocked)
    assert_true(blocked_count > 0 and blocked_count < len(rays))
    with DeviceContext() as ctx:
        comptime if kind == .SPHERE:
            var blases = build_gpu_sphere_blas_set[4, 4](ctx, [spheres^])
            _assert_gpu_results[
                .SPHERE,
                _intersect_sphere_leaf[.LOCAL, 4, mode],
                mode,
                closest_leaf_tasks,
            ](ctx, blases, instances, rays, expected, expected_hits)
        else:
            var blases = build_gpu_triangle_blas_set[4, 4](ctx, [vertices^])
            _assert_gpu_results[
                .TRIANGLE,
                _intersect_triangle_leaf[.LOCAL, 4, mode],
                mode,
                closest_leaf_tasks,
            ](ctx, blases, instances, rays, expected, expected_hits)


def _assert_gpu_results[
    kind: PrimitiveKind,
    leaf_fn: GpuLeafFn[.LOCAL],
    mode: TraceMode,
    closest_leaf_tasks: Int = 0,
](
    mut ctx: DeviceContext,
    blases: GpuBlasSet[kind, .WIDE, 4, 4],
    instances: List[Instance],
    rays: List[Rayf32[.WORLD]],
    expected: List[UInt32],
    expected_hits: List[Hit[.WORLD]],
) raises:
    var tlas = build_gpu_tlas[kind, 2, 4, 1, 4](ctx, instances)
    var input = upload_rays(ctx, rays)
    comptime output_stride = Hit.STRIDE if mode == .CLOSEST_HIT else 1
    var output = ctx.enqueue_create_buffer[.uint32](len(rays) * output_stride)
    ctx.enqueue_function[_visibility_kernel[leaf_fn, mode, closest_leaf_tasks]](
        tlas._tree.wide_nodes,
        tlas._tree.leaf_block_indices,
        tlas._inst_inv_transform,
        tlas._inst_blas_indices,
        blases.descs,
        blases.nodes,
        blases.leaves,
        input,
        output,
        tlas._tree.root_idx,
        Int32(len(instances)),
        Int32(len(rays)),
        grid_dim=ceildiv(len(rays), 64),
        block_dim=64,
    )
    ctx.synchronize()
    with output.map_to_host() as host:
        comptime if mode == .CLOSEST_HIT:
            var hits = Span(
                unsafe_ptr=host.unsafe_ptr().unsafe_bitcast[Float32](),
                length=len(host),
            )
            for i, value in enumerate(expected_hits):
                var actual = Hit[.WORLD].load(hits, i)
                assert_equal(actual.prim, value.prim)
                assert_equal(actual.inst, value.inst)
                assert_almost_equal(actual.t, value.t, atol=2.0e-4)
                assert_almost_equal(actual.u, value.u, atol=2.0e-4)
                assert_almost_equal(actual.v, value.v, atol=2.0e-4)
                assert_almost_equal(
                    actual.normal.x, value.normal.x, atol=2.0e-4
                )
                assert_almost_equal(
                    actual.normal.y, value.normal.y, atol=2.0e-4
                )
                assert_almost_equal(
                    actual.normal.z, value.normal.z, atol=2.0e-4
                )
        else:
            for i, value in enumerate(expected):
                assert_equal(host[i], value)


def test_gpu_tlas2_triangle_visibility_matches_primitive_reference() raises:
    _assert_visibility[.TRIANGLE]()


def test_gpu_tlas2_sphere_visibility_matches_primitive_reference() raises:
    _assert_visibility[.SPHERE]()


def test_gpu_tlas2_triangle_closest_matches_primitive_reference() raises:
    _assert_visibility[.TRIANGLE, .CLOSEST_HIT]()


def test_gpu_tlas2_sphere_closest_matches_primitive_reference() raises:
    _assert_visibility[.SPHERE, .CLOSEST_HIT]()


def test_gpu_tlas2_ordered_triangle_closest_matches_reference() raises:
    _assert_visibility[.TRIANGLE, .CLOSEST_HIT, 1]()


def test_gpu_tlas2_ordered_sphere_closest_matches_reference() raises:
    _assert_visibility[.SPHERE, .CLOSEST_HIT, 1]()


def test_gpu_tlas2_separate_triangle_closest_matches_reference() raises:
    _assert_visibility[.TRIANGLE, .CLOSEST_HIT, 2]()


def test_gpu_tlas2_separate_sphere_closest_matches_reference() raises:
    _assert_visibility[.SPHERE, .CLOSEST_HIT, 2]()


def test_gpu_tlas2_distance_pruned_triangle_matches_reference() raises:
    _assert_visibility[.TRIANGLE, .CLOSEST_HIT, 3]()


def test_gpu_tlas2_distance_pruned_sphere_matches_reference() raises:
    _assert_visibility[.SPHERE, .CLOSEST_HIT, 3]()


def run_tests[_discovery: Bool = True]() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
