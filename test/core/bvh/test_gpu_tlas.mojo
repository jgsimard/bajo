from std.benchmark import keep
from std.gpu.host import DeviceContext
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true

from bajo.core.bvh.cpu.binary_bvh import BinaryBvh
from bajo.core.bvh.cpu.tlas import BvhInstance, Tlas
from bajo.core.bvh.gpu.tlas import (
    GpuTlasLayout,
    GPU_TLAS_INTERNAL_FLAG,
    GPU_TLAS_LEAF_FLAG,
    GPU_TLAS_NODE_META_STRIDE,
    GPU_TLAS_NODE_BOUNDS_STRIDE,
    GPU_TLAS_INSTANCE_META_STRIDE,
    GPU_TLAS_TRANSFORM_STRIDE,
    GPU_TLAS_INSTANCE_BOUNDS_STRIDE,
)
from bajo.core.mat import Mat44f32
from bajo.core.vec import Vec3f32


def _translation(tx: Float32, ty: Float32, tz: Float32) -> Mat44f32:
    return Mat44f32(
        1.0,
        0.0,
        0.0,
        tx,
        0.0,
        1.0,
        0.0,
        ty,
        0.0,
        0.0,
        1.0,
        tz,
        0.0,
        0.0,
        0.0,
        1.0,
    )


def _uniform_scale(s: Float32) -> Mat44f32:
    return Mat44f32(
        s,
        0.0,
        0.0,
        0.0,
        0.0,
        s,
        0.0,
        0.0,
        0.0,
        0.0,
        s,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )


def _append_tri(mut verts: List[Vec3f32], cx: Float32, z: Float32):
    verts.append(Vec3f32(cx - 1.0, -1.0, z))
    verts.append(Vec3f32(cx + 1.0, -1.0, z))
    verts.append(Vec3f32(cx, 1.0, z))


def _make_strip(count: Int, z: Float32 = 2.0) -> List[Vec3f32]:
    var verts = List[Vec3f32](capacity=count * 3)
    for i in range(count):
        _append_tri(verts, Float32(i) * 4.0, z)
    return verts^


def _make_tlas(mut blas: BinaryBvh, count: Int = 8) -> Tlas:
    var instances = List[BvhInstance](capacity=count)
    for i in range(count):
        var x = Float32((i % 4) - 2) * 6.0
        var y = Float32(i // 4) * 4.0
        instances.append(
            BvhInstance.from_blas(
                _translation(x, y, 0.0),
                _translation(-x, -y, 0.0),
                0,
                blas,
            )
        )

    var tlas = Tlas(instances)
    tlas.build()
    return tlas^


@always_inline
def _node_flag(tlas: Tlas, node_idx: Int) -> UInt32:
    if tlas.tlas_nodes[node_idx].is_leaf():
        return GPU_TLAS_LEAF_FLAG
    return GPU_TLAS_INTERNAL_FLAG


def test_gpu_tlas_upload_validates_against_cpu_tlas() raises:
    comptime if has_accelerator():
        var verts = _make_strip(4)
        var blas = BinaryBvh(verts.unsafe_ptr(), UInt32(len(verts) // 3))
        blas.build["median", False]()
        var tlas = _make_tlas(blas, 8)

        with DeviceContext() as ctx:
            var gpu_tlas = GpuTlasLayout(ctx, tlas)
            var validation = gpu_tlas.validate(tlas)

            assert_true(validation.ok, "uploaded TLAS layout validates")
            assert_true(validation.node_count == tlas.nodes_used)
            assert_true(validation.inst_count == tlas.inst_count)
            assert_true(validation.leaf_instance_sum == tlas.inst_count)
            assert_true(validation.checksum != 0)
            keep(validation.checksum)
    else:
        assert_true(False, "No Accelerator found")


def test_gpu_tlas_node_meta_and_bounds_match_cpu() raises:
    comptime if has_accelerator():
        var verts = _make_strip(4)
        var blas = BinaryBvh(verts.unsafe_ptr(), UInt32(len(verts) // 3))
        blas.build["median", False]()
        var tlas = _make_tlas(blas, 8)

        with DeviceContext() as ctx:
            var gpu_tlas = GpuTlasLayout(ctx, tlas)

            with gpu_tlas.node_meta.map_to_host() as meta:
                for i in range(Int(tlas.nodes_used)):
                    ref node = tlas.tlas_nodes[i]
                    var base = i * GPU_TLAS_NODE_META_STRIDE
                    assert_true(meta[base + 0] == node.left_first)
                    assert_true(meta[base + 1] == node.tri_count)
                    assert_true(meta[base + 3] == _node_flag(tlas, i))

            with gpu_tlas.node_bounds.map_to_host() as bounds:
                for i in range(Int(tlas.nodes_used)):
                    ref node = tlas.tlas_nodes[i]
                    var base = i * GPU_TLAS_NODE_BOUNDS_STRIDE
                    assert_true(bounds[base + 0] == node.aabb._min.x())
                    assert_true(bounds[base + 1] == node.aabb._min.y())
                    assert_true(bounds[base + 2] == node.aabb._min.z())
                    assert_true(bounds[base + 3] == node.aabb._max.x())
                    assert_true(bounds[base + 4] == node.aabb._max.y())
                    assert_true(bounds[base + 5] == node.aabb._max.z())
    else:
        assert_true(False, "No Accelerator found")


def test_gpu_tlas_leaf_ranges_preserve_instance_indices() raises:
    comptime if has_accelerator():
        var verts = _make_strip(4)
        var blas = BinaryBvh(verts.unsafe_ptr(), UInt32(len(verts) // 3))
        blas.build["median", False]()
        var tlas = _make_tlas(blas, 12)

        with DeviceContext() as ctx:
            var gpu_tlas = GpuTlasLayout(ctx, tlas)

            with gpu_tlas.inst_indices.map_to_host() as inst_indices:
                var seen = List[UInt32](capacity=Int(tlas.inst_count))
                for _ in range(Int(tlas.inst_count)):
                    seen.append(0)

                for i in range(Int(tlas.inst_count)):
                    var inst_idx = inst_indices[i]
                    assert_true(inst_idx < tlas.inst_count)
                    assert_true(inst_idx == tlas.inst_indices[i])
                    seen[Int(inst_idx)] = seen[Int(inst_idx)] + 1

                for i in range(Int(tlas.inst_count)):
                    assert_true(seen[i] == 1)

            with gpu_tlas.node_meta.map_to_host() as meta:
                var leaf_sum = UInt32(0)
                for i in range(Int(tlas.nodes_used)):
                    ref node = tlas.tlas_nodes[i]
                    var base = i * GPU_TLAS_NODE_META_STRIDE
                    if node.is_leaf():
                        assert_true(meta[base + 3] == GPU_TLAS_LEAF_FLAG)
                        assert_true(meta[base + 1] > 0)
                        assert_true(
                            meta[base + 0] + meta[base + 1] <= tlas.inst_count
                        )
                        leaf_sum += meta[base + 1]
                    else:
                        assert_true(meta[base + 3] == GPU_TLAS_INTERNAL_FLAG)
                        assert_true(meta[base + 1] == 0)

                assert_true(leaf_sum == tlas.inst_count)
    else:
        assert_true(False, "No Accelerator found")


def test_gpu_tlas_instance_meta_transforms_and_bounds_match_cpu() raises:
    comptime if has_accelerator():
        var verts = _make_strip(4)
        var blas = BinaryBvh(verts.unsafe_ptr(), UInt32(len(verts) // 3))
        blas.build["median", False]()
        var instances = List[BvhInstance](capacity=2)
        instances.append(
            BvhInstance.from_blas(
                _translation(10.0, 2.0, -3.0),
                _translation(-10.0, -2.0, 3.0),
                0,
                blas,
            )
        )
        instances.append(
            BvhInstance.from_blas(
                _uniform_scale(2.0),
                _uniform_scale(0.5),
                0,
                blas,
            )
        )

        var tlas = Tlas(instances)
        tlas.build()

        with DeviceContext() as ctx:
            var gpu_tlas = GpuTlasLayout(ctx, tlas)

            with gpu_tlas.inst_meta.map_to_host() as meta:
                for i in range(Int(tlas.inst_count)):
                    var base = i * GPU_TLAS_INSTANCE_META_STRIDE
                    assert_true(meta[base + 0] == tlas.instances[i].blas_idx)
                    assert_true(meta[base + 1] == UInt32(i))
                    assert_true(meta[base + 2] == 0)
                    assert_true(meta[base + 3] == 0)

            with gpu_tlas.inst_transform.map_to_host() as xform:
                for i in range(Int(tlas.inst_count)):
                    var base = i * GPU_TLAS_TRANSFORM_STRIDE
                    comptime for row in range(4):
                        comptime for col in range(4):
                            comptime j = row * 4 + col
                            assert_true(
                                xform[base + j]
                                == tlas.instances[i].transform[row][col]
                            )

            with gpu_tlas.inst_inv_transform.map_to_host() as inv_xform:
                for i in range(Int(tlas.inst_count)):
                    var base = i * GPU_TLAS_TRANSFORM_STRIDE
                    comptime for row in range(4):
                        comptime for col in range(4):
                            comptime j = row * 4 + col
                            assert_true(
                                inv_xform[base + j]
                                == tlas.instances[i].inv_transform[row][col]
                            )

            with gpu_tlas.inst_bounds.map_to_host() as bounds:
                for i in range(Int(tlas.inst_count)):
                    ref inst = tlas.instances[i]
                    var base = i * GPU_TLAS_INSTANCE_BOUNDS_STRIDE
                    assert_true(bounds[base + 0] == inst.bounds_min.x())
                    assert_true(bounds[base + 1] == inst.bounds_min.y())
                    assert_true(bounds[base + 2] == inst.bounds_min.z())
                    assert_true(bounds[base + 3] == inst.bounds_max.x())
                    assert_true(bounds[base + 4] == inst.bounds_max.y())
                    assert_true(bounds[base + 5] == inst.bounds_max.z())
    else:
        assert_true(False, "No Accelerator found")


# Keep this last so `run_tests.sh` can discover and run the file directly.
def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
