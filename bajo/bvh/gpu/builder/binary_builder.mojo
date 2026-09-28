from max.gpu.host import DeviceContext

from bajo.bvh.gpu.builder.binary_layout import (
    GpuBinaryBoundsBvh,
    GpuBinaryBuildWorkspace,
)
from bajo.bvh.gpu.builder.hploc_binary import build_binary_bvh_with_hploc
from bajo.bvh.gpu.builder.lbvh import build_binary_bvh_with_lbvh
from bajo.bvh.gpu.utils import GpuBuildTimings


@fieldwise_init
struct GpuBvhBuildMethod(EnumLike, Equatable):
    """Compile-time GPU binary builder selector; LBVH remains the default."""

    comptime LBVH = Self(0)
    comptime HPLOC = Self(1)
    comptime _enum_case_names = ParameterList.of[
        "LBVH".value, "HPLOC".value
    ].values
    comptime _enum_case_types = TypeList.of[
        Trait=AnyType, NoneType, NoneType
    ].values
    var value: Int

    def _get_enum_discriminant(self) -> Int:
        return self.value

    def _unsafe_get_enum_payload[
        id: Int
    ](ref self) -> ref[self] TypeList[Trait=AnyType, Self._enum_case_types]()[
        id
    ]:
        while True:
            pass


def build_binary_bvh[
    method: GpuBvhBuildMethod = .LBVH,
](
    mut ctx: DeviceContext,
    mut binary: GpuBinaryBoundsBvh,
    mut workspace: GpuBinaryBuildWorkspace,
    measure_stages: Bool = False,
) raises -> GpuBuildTimings:
    """Select the binary topology builder at compile time; LBVH is default."""

    comptime __match method:
    case .LBVH:
        return build_binary_bvh_with_lbvh(
            ctx, binary, workspace, measure_stages
        )
    case .HPLOC:
        return build_binary_bvh_with_hploc(
            ctx, binary, workspace, measure_stages
        )
