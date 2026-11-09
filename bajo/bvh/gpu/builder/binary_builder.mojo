from max.gpu.host import DeviceContext

from bajo.bvh.gpu.builder.binary_layout import (
    GpuBinaryBoundsBvh,
    GpuBinaryBuildWorkspace,
)
from bajo.bvh.gpu.builder.hploc_binary import build_binary_bvh_with_hploc
from bajo.bvh.gpu.builder.lbvh import build_binary_bvh_with_lbvh
from bajo.bvh.gpu.utils import GpuBuildTimings


__enum GpuBvhBuildMethod(Equatable, ImplicitlyCopyable):
    """Compile-time GPU binary builder selector; LBVH remains the default."""

    case LBVH
    case HPLOC


def build_binary_bvh[
    method: GpuBvhBuildMethod = .LBVH,
](
    mut ctx: DeviceContext,
    mut binary: GpuBinaryBoundsBvh,
    mut workspace: GpuBinaryBuildWorkspace,
    measure_stages: Bool = False,
) raises -> GpuBuildTimings:
    """Select the binary topology builder at compile time; LBVH is default."""

    comptime match method:
        case .LBVH:
            return build_binary_bvh_with_lbvh(
                ctx, binary, workspace, measure_stages
            )
        case .HPLOC:
            return build_binary_bvh_with_hploc(
                ctx, binary, workspace, measure_stages
            )
