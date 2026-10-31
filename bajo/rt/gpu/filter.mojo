"""Owned GPU reconstruction-filter tables, retained across compatible renders."""

from max.gpu.host import DeviceBuffer, DeviceContext
from bajo.gpu.buffers import upload_list
from bajo.rt.filter import ReconstructionFilter, FilterSampler, FilterView


struct GpuFilterTables:
    var config: ReconstructionFilter
    var data: DeviceBuffer[.float32]
    var nx: Int32
    var ny: Int32

    def __init__(
        out self, mut ctx: DeviceContext, config: ReconstructionFilter
    ) raises:
        var host = FilterSampler(config)
        self.config = config
        self.nx = host.nx
        self.ny = host.ny
        self.data = upload_list(ctx, host.data)

    def update(
        mut self, ctx: DeviceContext, config: ReconstructionFilter
    ) raises:
        if self.config.matches(config):
            return
        var context = ctx.copy()
        self = Self(context, config)

    def view(self) -> FilterView:
        return FilterView(
            self.config,
            Optional(
                self.data.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin]()
            ),
            self.nx,
            self.ny,
        )
