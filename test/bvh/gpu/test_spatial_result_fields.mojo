from std.testing import TestSuite, assert_equal, assert_almost_equal
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from bajo.core import Point3f32, Vec3f32
from bajo.bvh.spatial import MeshQueryView, BoundsQueryView
from bajo.bvh.spatial.gpu_mesh import GpuSpatialMesh
from bajo.bvh.gpu.utils import _device_span
from test.bvh.spatial_fixtures import tetrahedron
from test.bvh.spatial_result_fields import FIELD_COUNT, mesh_result_fields


def _fields_kernel[
    count: Int
](
    points: ImmSpan[Float32, ImmutAnyOrigin],
    indices: ImmSpan[UInt32, ImmutAnyOrigin],
    velocities: ImmSpan[Float32, ImmutAnyOrigin],
    bounds: ImmSpan[Float32, ImmutAnyOrigin],
    nodes: ImmSpan[UInt32, ImmutAnyOrigin],
    output: MutSpan[Float64, MutAnyOrigin],
    root: UInt32,
):
    var lane = Int(global_idx.x)
    if lane >= count:
        return
    var view = MeshQueryView[ImmutAnyOrigin, .WORLD](
        points,
        indices,
        velocities,
        BoundsQueryView[ImmutAnyOrigin, .WORLD](bounds, nodes),
    )
    mesh_result_fields(
        view, lane, root, output[lane * FIELD_COUNT : (lane + 1) * FIELD_COUNT]
    )


def test_every_mesh_result_field_matches_cpu_across_groups_and_updates() raises:
    var cpu = tetrahedron()
    var velocities: List[Vec3f32[.WORLD]] = [
        Vec3f32[.WORLD](1.0, 2.0, 3.0),
        Vec3f32[.WORLD](2.0, 3.0, 4.0),
        Vec3f32[.WORLD](3.0, 4.0, 5.0),
        Vec3f32[.WORLD](4.0, 5.0, 6.0),
    ]
    cpu.set_velocities(velocities)
    with DeviceContext() as ctx:
        var gpu = GpuSpatialMesh(ctx, cpu.copy())
        var output = ctx.enqueue_create_buffer[.float64](105 * FIELD_COUNT)
        for phase in range(3):
            if phase > 0:
                var offset = Float32(phase) * 0.1
                var points: List[Point3f32[.WORLD]] = [
                    Point3f32[.WORLD](offset),
                    Point3f32[.WORLD](1.0 + offset, offset, offset),
                    Point3f32[.WORLD](offset, 1.0 + offset, offset),
                    Point3f32[.WORLD](offset, offset, 1.0 + offset),
                ]
                if phase == 1:
                    cpu.refit(points)
                    gpu.refit(ctx, points)
                else:
                    cpu.rebuild(points)
                    gpu.rebuild(ctx, points)
            var roots: List[UInt32] = [
                0,
                cpu.group_root(7),
                cpu.group_root(999),
            ]
            for root in roots:
                ctx.enqueue_function[_fields_kernel[105]](
                    _device_span[mut=False](gpu.points),
                    _device_span[mut=False](gpu.indices),
                    _device_span[mut=False](gpu.velocities),
                    _device_span[mut=False](gpu.bounds),
                    _device_span[mut=False](gpu.nodes),
                    _device_span[mut=True](output),
                    root,
                    grid_dim=1,
                    block_dim=128,
                )
                ctx.synchronize()
                with output.map_to_host() as actual:
                    for lane in range(105):
                        var expected = List[Float64](
                            length=FIELD_COUNT, fill=0.0
                        )
                        mesh_result_fields(cpu.view(), lane, root, expected)
                        for field in range(FIELD_COUNT):
                            if (
                                field == 0
                                or field == 1
                                or field == 9
                                or field == 10
                                or field == 18
                                or field == 19
                                or field == 26
                                or field == 42
                            ):
                                assert_equal(
                                    actual[lane * FIELD_COUNT + field],
                                    expected[field],
                                )
                            else:
                                assert_almost_equal(
                                    actual[lane * FIELD_COUNT + field],
                                    expected[field],
                                    atol=1e-6,
                                    rtol=1e-6,
                                )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
