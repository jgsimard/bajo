"""Empty-segment coverage for packed GPU BLAS construction and traversal."""

from std.math import abs
from std.testing import TestSuite, assert_equal, assert_true
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh.camera import Camera
from bajo.bvh.constants import (
    Primitive,
    SPHERE_LEAF_PACKED_STRIDE,
    TRI_LEAF_PACKED_STRIDE,
    WideNode,
)
from bajo.bvh.gpu.builder import GpuBvhBuildMethod
from bajo.bvh.gpu.compressed_bounds_bvh import (
    CWBVH_NODE_WORDS,
    CWBVH_TRIANGLE_WORDS,
)
from bajo.bvh.gpu.sphere_bvh import build_sphere_blas_set
from bajo.bvh.gpu.tlas import build_sphere_tlas, build_triangle_tlas
from bajo.bvh.gpu.triangle_bvh import build_triangle_blas_set
from bajo.bvh.gpu.utils import upload_camera
from bajo.bvh.host_utils import compute_bounds, sphere_bounds
from bajo.bvh.gpu import GpuBlasSet, GpuBvhLayout
from bajo.bvh.types import BlasDescLayout, Hit, Instance, Sphere
from bajo.core import AABB, Affine3f32, Frame, Point3f32, Vec3f32
from test.bvh.fixtures import _make_camera_ray


def _triangle() -> List[Point3f32[Frame.LOCAL]]:
    return [
        Point3f32[Frame.LOCAL](-1.0, -1.0, 6.0),
        Point3f32[Frame.LOCAL](1.0, -1.0, 6.0),
        Point3f32[Frame.LOCAL](0.0, 1.0, 6.0),
    ]


def _triangles(count: Int, z: Float32) -> List[Point3f32[Frame.LOCAL]]:
    var vertices = List[Point3f32[Frame.LOCAL]](capacity=count * 3)
    for idx in range(count):
        var x = Float32(idx) * 3.0
        vertices.append(Point3f32[Frame.LOCAL](x - 1.0, -1.0, z))
        vertices.append(Point3f32[Frame.LOCAL](x + 1.0, -1.0, z))
        vertices.append(Point3f32[Frame.LOCAL](x, 1.0, z))
    return vertices^


def _assert_exact_storage[
    kind: Primitive,
    layout: GpuBvhLayout,
    node_width: SIMDLength,
    leaf_width: SIMDLength,
](
    blases: GpuBlasSet[kind, layout, node_width, leaf_width],
    node_stride: Int,
    leaf_stride: Int,
) raises:
    var node_count = 0
    var leaf_count = 0
    with blases.descs.map_to_host() as descs:
        for blas_idx in range(blases.blas_count):
            var base = BlasDescLayout.base(blas_idx)
            assert_equal(
                Int(descs[base + BlasDescLayout.NODE_F32_BASE]),
                node_count * node_stride,
            )
            assert_equal(
                Int(descs[base + BlasDescLayout.LEAF_F32_BASE]),
                leaf_count * leaf_stride,
            )
            node_count += Int(descs[base + BlasDescLayout.NODE_COUNT])
            leaf_count += Int(descs[base + BlasDescLayout.LEAF_BLOCK_COUNT])
    assert_equal(len(blases.nodes), node_count * node_stride)
    assert_equal(len(blases.leaves), leaf_count * leaf_stride)


def _instance(
    blas_idx: UInt32, bounds: AABB[Frame.LOCAL], kind: Primitive
) -> Instance:
    return Instance(
        Affine3f32[Frame.LOCAL, Frame.WORLD].identity(),
        blas_idx,
        bounds,
        kind,
    )


def _download_hit(
    hits: DeviceBuffer[DType.float32],
) raises -> Hit[Frame.WORLD]:
    with hits.map_to_host() as host:
        return Hit[Frame.WORLD].load(
            Span(unsafe_ptr=host.unsafe_ptr(), length=len(host)), 0
        )


def _camera() -> Camera:
    return _make_camera_ray(
        Point3f32[Frame.WORLD](0.0, 0.0, 0.0),
        Vec3f32[Frame.WORLD](0.0, 0.0, 1.0),
    )


def test_triangle_lbvh_empty_segments_trace_and_describe() raises:
    var empty = List[Point3f32[Frame.LOCAL]]()
    var vertices = _triangle()
    var bounds = compute_bounds(vertices)
    with DeviceContext() as ctx:
        var blases = build_triangle_blas_set[4, 4, GpuBvhBuildMethod.LBVH](
            ctx, [empty.copy(), vertices^, empty^]
        )
        var node_count: Int
        var leaf_count: Int
        with blases.descs.map_to_host() as descs:
            assert_equal(descs[BlasDescLayout.PRIM_COUNT], UInt32(0))
            assert_equal(
                descs[BlasDescLayout.base(1) + BlasDescLayout.PRIM_COUNT],
                UInt32(1),
            )
            assert_equal(
                descs[BlasDescLayout.base(2) + BlasDescLayout.PRIM_COUNT],
                UInt32(0),
            )
            node_count = Int(
                descs[BlasDescLayout.base(1) + BlasDescLayout.NODE_COUNT]
            )
            leaf_count = Int(
                descs[BlasDescLayout.base(1) + BlasDescLayout.LEAF_BLOCK_COUNT]
            )
        assert_equal(len(blases.nodes), node_count * 4 * WideNode.CHILD_STRIDE)
        assert_equal(
            len(blases.leaves), leaf_count * 4 * TRI_LEAF_PACKED_STRIDE
        )
        var instances: List = [
            _instance(0, bounds, Primitive.TRIANGLE),
            _instance(1, bounds, Primitive.TRIANGLE),
        ]
        var tlas = build_triangle_tlas[4, 4](ctx, instances)
        var hits = ctx.enqueue_create_buffer[DType.float32](Hit.STRIDE)
        tlas.launch_camera(
            ctx, blases, upload_camera(ctx, _camera()), hits, 1, 1, 1
        )
        ctx.synchronize()
        var hit = _download_hit(hits)
        assert_true(abs(hit.t - 6.0) < 1.0e-5)
        assert_equal(hit.inst, UInt32(1))


def test_triangle_hploc_cwbvh8_empty_segments_trace() raises:
    var empty = List[Point3f32[Frame.LOCAL]]()
    var vertices = _triangle()
    var bounds = compute_bounds(vertices)
    with DeviceContext() as ctx:
        var blases = build_triangle_blas_set[
            8, 4, GpuBvhBuildMethod.HPLOC, GpuBvhLayout.CWBVH8
        ](ctx, [empty.copy(), vertices^, empty^])
        with blases.descs.map_to_host() as descs:
            var node_count = Int(
                descs[BlasDescLayout.base(1) + BlasDescLayout.NODE_COUNT]
            )
            assert_equal(len(blases.nodes), node_count * CWBVH_NODE_WORDS)
        assert_equal(len(blases.leaves), CWBVH_TRIANGLE_WORDS)
        var instances: List = [
            _instance(0, bounds, Primitive.TRIANGLE),
            _instance(1, bounds, Primitive.TRIANGLE),
        ]
        var tlas = build_triangle_tlas[
            2, 8, 2, 4, GpuBvhBuildMethod.LBVH, GpuBvhLayout.CWBVH8
        ](ctx, instances)
        var hits = ctx.enqueue_create_buffer[DType.float32](Hit.STRIDE)
        tlas.launch_camera(
            ctx, blases, upload_camera(ctx, _camera()), hits, 1, 1, 1
        )
        ctx.synchronize()
        var hit = _download_hit(hits)
        assert_true(abs(hit.t - 6.0) < 1.0e-5)
        assert_equal(hit.inst, UInt32(1))


def test_sphere_hploc_empty_segments_trace() raises:
    var empty = List[Sphere[Frame.LOCAL]]()
    var spheres: List = [
        Sphere[Frame.LOCAL](Point3f32[Frame.LOCAL](0.0, 0.0, 6.0), 1.0)
    ]
    var bounds = sphere_bounds(spheres)
    with DeviceContext() as ctx:
        var blases = build_sphere_blas_set[4, 4, GpuBvhBuildMethod.HPLOC](
            ctx, [empty.copy(), spheres^, empty^]
        )
        with blases.descs.map_to_host() as descs:
            var node_count = Int(
                descs[BlasDescLayout.base(1) + BlasDescLayout.NODE_COUNT]
            )
            var leaf_count = Int(
                descs[BlasDescLayout.base(1) + BlasDescLayout.LEAF_BLOCK_COUNT]
            )
            assert_equal(
                len(blases.nodes), node_count * 4 * WideNode.CHILD_STRIDE
            )
            assert_equal(
                len(blases.leaves),
                leaf_count * 4 * SPHERE_LEAF_PACKED_STRIDE,
            )
        var instances: List = [
            _instance(0, bounds, Primitive.SPHERE),
            _instance(1, bounds, Primitive.SPHERE),
        ]
        var tlas = build_sphere_tlas[4, 4](ctx, instances)
        var hits = ctx.enqueue_create_buffer[DType.float32](Hit.STRIDE)
        tlas.launch_camera(
            ctx, blases, upload_camera(ctx, _camera()), hits, 1, 1, 1
        )
        ctx.synchronize()
        var hit = _download_hit(hits)
        assert_true(abs(hit.t - 5.0) < 1.0e-5)
        assert_equal(hit.inst, UInt32(1))


def test_all_empty_triangle_batch_has_zero_descriptors() raises:
    var empty = List[Point3f32[Frame.LOCAL]]()
    with DeviceContext() as ctx:
        var blases = build_triangle_blas_set[
            8, 4, GpuBvhBuildMethod.HPLOC, GpuBvhLayout.CWBVH8
        ](ctx, [empty.copy(), empty^])
        assert_equal(blases.blas_count, 2)
        assert_equal(len(blases.nodes), 1)
        assert_equal(len(blases.leaves), 1)
        with blases.descs.map_to_host() as descs:
            for idx in range(len(descs)):
                assert_equal(descs[idx], UInt32(0))


def test_nonempty_segments_use_exact_consecutive_storage() raises:
    var first = _triangles(5, 4.0)
    var second = _triangles(17, 8.0)
    with DeviceContext() as ctx:
        var wide = build_triangle_blas_set[4, 4, GpuBvhBuildMethod.HPLOC](
            ctx, [first.copy(), second.copy()]
        )
        _assert_exact_storage(
            wide,
            4 * WideNode.CHILD_STRIDE,
            4 * TRI_LEAF_PACKED_STRIDE,
        )

        var compressed = build_triangle_blas_set[
            8, 4, GpuBvhBuildMethod.HPLOC, GpuBvhLayout.CWBVH8
        ](ctx, [first^, second^])
        _assert_exact_storage(
            compressed, CWBVH_NODE_WORDS, CWBVH_TRIANGLE_WORDS
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
