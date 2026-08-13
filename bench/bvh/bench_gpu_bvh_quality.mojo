from std.math import round
from max.gpu.host import DeviceBuffer, DeviceContext

from bajo.bvh.gpu.diagnostics import build_bounds_bvh_for_diagnostics
from bajo.bvh.gpu.wide_layout import GpuWideBoundsBvh
from bajo.bvh.gpu.quality import (
    GpuBvhQuality,
    measure_binary_bvh_quality,
    measure_wide_bvh_quality,
)
from bajo.bvh.gpu.utils import upload_list
from bajo.bvh.host_utils import triangle_bounds
from bajo.core import Frame
from bajo.obj.pack import pack_obj_triangles


comptime DEFAULT_OBJ_PATH = "./assets/dragon/dragon.obj"


def _print_quality_row(
    label: String,
    layout: String,
    quality: GpuBvhQuality,
):
    print(
        label.ascii_ljust(11),
        layout.ascii_ljust(7),
        String(round(quality.quality, 3)).ascii_rjust(9),
        String(round(quality.internal_area_ratio, 3)).ascii_rjust(10),
        String(round(quality.primitive_area_ratio, 3)).ascii_rjust(10),
        String(quality.internal_nodes).ascii_rjust(10),
        String(quality.leaf_references).ascii_rjust(10),
        String(quality.primitives).ascii_rjust(10),
    )


def _measure_configuration[
    node_width: SIMDLength,
    leaf_width: SIMDLength,
    max_leaf_size: Int = Int(leaf_width),
](
    mut ctx: DeviceContext,
    leaf_bounds: DeviceBuffer[DType.float32],
    payloads: DeviceBuffer[DType.uint32],
    label: String,
) raises:
    var triangle_count = len(payloads)

    var tree = GpuWideBoundsBvh[node_width, leaf_width, max_leaf_size](
        ctx, triangle_count
    )
    var binary = build_bounds_bvh_for_diagnostics(
        ctx, tree, leaf_bounds.copy(), payloads.copy()
    )
    ctx.synchronize()

    _print_quality_row(label, "binary", measure_binary_bvh_quality(binary))
    _print_quality_row(label, "wide", measure_wide_bvh_quality(tree))


def main() raises:
    var vertices = pack_obj_triangles[Frame.WORLD](DEFAULT_OBJ_PATH)
    var host_leaf_bounds = List[Float32](capacity=(len(vertices) / 3) * 6)
    var host_payloads = List[UInt32](capacity=len(vertices) / 3)
    for triangle_idx in range(len(vertices) / 3):
        var bounds = triangle_bounds(
            vertices[triangle_idx * 3 + 0],
            vertices[triangle_idx * 3 + 1],
            vertices[triangle_idx * 3 + 2],
        )
        host_leaf_bounds.append(bounds._min.x)
        host_leaf_bounds.append(bounds._min.y)
        host_leaf_bounds.append(bounds._min.z)
        host_leaf_bounds.append(bounds._max.x)
        host_leaf_bounds.append(bounds._max.y)
        host_leaf_bounds.append(bounds._max.z)
        host_payloads.append(UInt32(triangle_idx))

    print("GPU LBVH quality baseline")
    print(t"OBJ path: {DEFAULT_OBJ_PATH}")
    print(t"triangles: {len(vertices) / 3}\n")
    print(
        "config      layout    quality   internal  primitive      nodes "
        "      leaves primitives"
    )
    print(
        "----------- ------- --------- ---------- ---------- ---------- "
        "---------- ----------"
    )

    with DeviceContext() as ctx:
        var leaf_bounds = upload_list(ctx, host_leaf_bounds)
        var payloads = upload_list(ctx, host_payloads)
        _measure_configuration[2, 2](ctx, leaf_bounds, payloads, "n2/l2")
        _measure_configuration[2, 4](ctx, leaf_bounds, payloads, "n2/l4")
        _measure_configuration[4, 4](ctx, leaf_bounds, payloads, "n4/l4")
        _measure_configuration[8, 8](ctx, leaf_bounds, payloads, "n8/l8")
        _measure_configuration[8, 4, 3](ctx, leaf_bounds, payloads, "n8/l4/m3")
