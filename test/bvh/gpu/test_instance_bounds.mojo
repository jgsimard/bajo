"""Conservative instance bounds from GPU-built BLAS root frontiers."""

from std.sys import has_accelerator
from std.memory import bitcast
from std.testing import TestSuite, assert_equal, assert_true
from max.gpu.host import DeviceContext
from bajo.core import AABB, Affine3f32, Point3f32, Vec3f32
from bajo.core.aabb import compute_bounds
from bajo.bvh.types import Instance
from bajo.bvh.gpu import GpuBvhLayout, GpuBvhBuildMethod
from bajo.bvh.gpu.triangle_bvh import build_gpu_triangle_blas_set
from bajo.gpu.buffers import upload_list
from bajo.bvh.gpu.instance_bounds import (
    refine_gpu_instance_bounds,
)


def _mesh() -> List[Point3f32[.LOCAL]]:
    var vertices = List[Point3f32[.LOCAL]]()
    for i in range(80):
        var p = Float32(i % 20) * 0.3 - 2.8
        var z = Float32(i // 20) * 0.2
        vertices.extend(
            [
                Point3f32[.LOCAL](p - 0.12, p - 0.12, z),
                Point3f32[.LOCAL](p + 0.12, p - 0.12, z),
                Point3f32[.LOCAL](p, p + 0.12, z),
            ]
        )
    return vertices^


def _instances(bounds: AABB[.LOCAL]) -> List[Instance]:
    var instances = List[Instance]()
    for i in range(32):
        var transform = Affine3f32[.LOCAL, .WORLD].identity()
        transform.m00 = -0.70710677 if i % 2 else 0.70710677
        transform.m01 = -0.70710677
        transform.m10 = -0.70710677 if i % 2 else 0.70710677
        transform.m11 = 0.70710677
        transform.m02 = Float32(i % 3) * 0.1
        transform.m22 = Float32(0.7) if i % 3 else Float32(1.4)
        transform.tx = Float32(i % 4) * 2.0 - 3.0
        transform.ty = Float32((i // 4) % 4) * 2.0 - 3.0
        transform.tz = Float32(i // 16) * 2.0 + 4.0
        transform.tz += Float32(i) * 0.003
        # Descriptor 1 is deliberately empty; keep its caller-supplied bound.
        instances.append(Instance(transform, UInt32(i % 3), bounds, .TRIANGLE))
    return instances^


def _assert_bounds[
    width: SIMDLength,
    layout: GpuBvhLayout,
    method: GpuBvhBuildMethod = .HPLOC,
]() raises:
    var mesh = _mesh()
    var original = _instances(compute_bounds(mesh))
    with DeviceContext() as ctx:
        var meshes: List = [mesh.copy(), List[Point3f32[.LOCAL]](), mesh.copy()]
        var blases = build_gpu_triangle_blas_set[width, 4, method, layout](
            ctx, meshes
        )
        var refined = refine_gpu_instance_bounds(ctx, blases, original)
        var reduced = 0
        for i, instance in enumerate(refined):
            if instance.blas_idx == 1:
                assert_equal(
                    instance.bounds.surface_area(),
                    original[i].bounds.surface_area(),
                )
                continue
            for vertex in mesh:
                assert_true(
                    instance.bounds.contains_point(
                        instance.transform.point(vertex)
                    )[0]
                )
            assert_true(
                instance.bounds.surface_area()
                <= original[i].bounds.surface_area()
            )
            if (
                instance.bounds.surface_area()
                < original[i].bounds.surface_area() * 0.8
            ):
                reduced += 1
        assert_true(
            reduced > 0,
            "sparse rotated geometry should get materially tighter bounds",
        )


def test_refined_cwbvh_bounds_enclose_transformed_geometry() raises:
    _assert_bounds[8, .CWBVH8]()
    _assert_bounds[8, .CWBVH8, .LBVH]()


def test_refined_wide_bounds_enclose_transformed_geometry() raises:
    _assert_bounds[4, .WIDE]()
    _assert_bounds[2, .WIDE]()


def test_nonfinite_active_child_retains_the_entire_original_bound() raises:
    var mesh = _mesh()
    var instances = _instances(compute_bounds(mesh))
    with DeviceContext() as ctx:
        var meshes: List = [mesh.copy(), List[Point3f32[.LOCAL]](), mesh.copy()]
        var blases = build_gpu_triangle_blas_set[4, 4, .HPLOC, .WIDE](
            ctx, meshes
        )
        var nodes = List[Float32]()
        with blases.nodes.map_to_host() as mapped:
            for i in range(len(mapped)):
                nodes.append(mapped[i])
        # Root child zero is active. Invalidating its lower X endpoint must
        # not produce a bound formed from only the remaining valid children.
        nodes[0] = bitcast[.float32](UInt32(0x7FC00000))
        blases.nodes = upload_list(ctx, nodes)
        var result = refine_gpu_instance_bounds(ctx, blases, instances)
        for i, instance in enumerate(result):
            if instance.blas_idx == 0:
                for axis in range(3):
                    assert_equal(
                        instance.bounds._min[axis],
                        instances[i].bounds._min[axis],
                    )
                    assert_equal(
                        instance.bounds._max[axis],
                        instances[i].bounds._max[axis],
                    )


def test_axis_aligned_instances_and_empty_input_retain_caller_bounds() raises:
    var mesh = _mesh()
    var instances: List = [
        Instance(
            Affine3f32[.LOCAL, .WORLD].from_scale(
                Vec3f32[.LOCAL](-2.0, 3.0, 0.5)
            ),
            UInt32(0),
            compute_bounds(mesh),
            .TRIANGLE,
        )
    ]
    with DeviceContext() as ctx:
        var meshes: List = [mesh.copy()]
        var blases = build_gpu_triangle_blas_set[4](ctx, meshes)
        var result = refine_gpu_instance_bounds(ctx, blases, instances)
        for axis in range(3):
            assert_equal(
                result[0].bounds._min[axis], instances[0].bounds._min[axis]
            )
            assert_equal(
                result[0].bounds._max[axis], instances[0].bounds._max[axis]
            )
        var empty = List[Instance]()
        assert_equal(len(refine_gpu_instance_bounds(ctx, blases, empty)), 0)


def run_tests[unused: Bool = True]() raises:
    comptime if has_accelerator():
        TestSuite.discover_tests[__functions_in_module()]().run()
