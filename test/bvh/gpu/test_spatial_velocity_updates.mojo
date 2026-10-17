"""Transactional resident velocity publication, including unreferenced vertices."""
from std.testing import TestSuite, assert_equal, assert_true
from std.memory import bitcast
from max.gpu.host import DeviceContext
from bajo.core import Point3f32
from bajo.bvh.spatial import SpatialMesh
from bajo.bvh.spatial.gpu_mesh import GpuSpatialMesh
from bajo.bvh.gpu.utils import upload_list


def test_velocity_publication_rejection_alias_and_clear() raises:
    with DeviceContext() as ctx:
        var sizes = List[Int]([0, 3, 342, 21846])
        for count in sizes:
            var points = List[Point3f32[.WORLD]]()
            for vertex in range(count):
                points.append(
                    Point3f32[.WORLD](
                        Float32(vertex % 17),
                        Float32(vertex % 13),
                        Float32(vertex % 11),
                    )
                )
            var indices = List[UInt32]()
            if count >= 3:
                indices.extend([0, 1, 2])
            var cpu = SpatialMesh(points, indices)
            var gpu = GpuSpatialMesh(ctx, cpu.copy())
            var values = List[Float32]()
            for component in range(count * 3):
                values.append(Float32((component * 13) % 127 - 63) * 0.125)
            var input = upload_list(ctx, values)
            comptime for block_size in [0, 32]:
                gpu.enqueue_velocities[block_size](ctx, input)
                gpu.check_update(ctx)
                with gpu.velocities.map_to_host() as actual:
                    for component in range(len(values)):
                        assert_equal(actual[component], values[component])
                # In-place updates validate every component before publication.
                gpu.enqueue_velocities[block_size](ctx, gpu.velocities)
                gpu.check_update(ctx)
                if count > 0:
                    for probe, slot in enumerate(
                        [0, len(values) // 2, len(values) - 1]
                    ):
                        with input.map_to_host() as data:
                            data[slot] = bitcast[.float32](
                                UInt32(0x7FC00000) if probe % 2
                                == 0 else UInt32(0x7F800000)
                            )
                        gpu.enqueue_velocities[block_size](ctx, input)
                        var rejected = False
                        try:
                            gpu.check_update(ctx)
                        except:
                            rejected = True
                        assert_true(rejected)
                        with gpu.velocities.map_to_host() as actual:
                            for component in range(len(values)):
                                assert_equal(
                                    actual[component], values[component]
                                )
                        with input.map_to_host() as data:
                            data[slot] = values[slot]
                var empty = ctx.enqueue_create_buffer[.float32](0)
                gpu.enqueue_velocities[block_size](ctx, empty)
                gpu.check_update(ctx)
                with gpu.velocities.map_to_host() as actual:
                    for component in range(len(values)):
                        assert_equal(actual[component], Float32(0.0))
            with gpu.points.map_to_host() as actual:
                for vertex in range(count):
                    comptime for axis in range(3):
                        assert_equal(
                            actual[vertex * 3 + axis], points[vertex][axis]
                        )
            with gpu.bounds.map_to_host() as actual:
                for component in range(len(cpu.bvh.bounds)):
                    assert_equal(actual[component], cpu.bvh.bounds[component])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
