"""Original spectral reconstruction tables uploaded without quantization."""

from max.gpu.host import DeviceBuffer, DeviceContext
from bajo.gpu.buffers import upload_list
from bajo.core.rgb_spectrum import RgbSpectrumTable, RgbSpectrumView
from bajo.core.spectrum_tables import (
    PiecewiseSpectrumTable,
    PiecewiseSpectrumView,
)
from bajo.rt.material_types import SurfaceStore, NO_TEXTURE
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder


@fieldwise_init
struct GpuMaterialSpectraView(Copyable, DevicePassable):
    var curves: PiecewiseSpectrumView
    var coated_eta_k: ImmPointer[UInt32, ImmUntrackedOrigin]
    var dielectric_eta: ImmPointer[UInt32, ImmUntrackedOrigin]

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "GpuMaterialSpectraView"


struct GpuMaterialSpectra:
    var samples: DeviceBuffer[.float32]
    var ranges: DeviceBuffer[.uint32]
    var coated_eta_k: DeviceBuffer[.uint32]
    var dielectric_eta: DeviceBuffer[.uint32]
    var count: UInt32

    def __init__(
        out self, mut ctx: DeviceContext, surfaces: SurfaceStore
    ) raises:
        var host = PiecewiseSpectrumTable(surfaces.optical_spectra)
        self.samples = upload_list(ctx, host.samples)
        self.ranges = upload_list(ctx, host.ranges)
        self.count = host.count
        var coated = List[UInt32]()
        for material in surfaces.coated_diffuses:
            coated.extend(
                [material.conductor_eta_spectrum, material.conductor_k_spectrum]
            )
        if not coated:
            coated.extend([NO_TEXTURE, NO_TEXTURE])
        self.coated_eta_k = upload_list(ctx, coated)
        var dielectric = List[UInt32]()
        for material in surfaces.dielectrics:
            dielectric.append(material.eta_spectrum)
        if not dielectric:
            dielectric.append(NO_TEXTURE)
        self.dielectric_eta = upload_list(ctx, dielectric)

    def view(self) -> GpuMaterialSpectraView:
        return GpuMaterialSpectraView(
            PiecewiseSpectrumView(
                self.samples.unsafe_ptr().unsafe_origin_cast[
                    ImmUntrackedOrigin
                ](),
                self.ranges.unsafe_ptr().unsafe_origin_cast[
                    ImmUntrackedOrigin
                ](),
                self.count,
            ),
            self.coated_eta_k.unsafe_ptr().unsafe_origin_cast[
                ImmUntrackedOrigin
            ](),
            self.dielectric_eta.unsafe_ptr().unsafe_origin_cast[
                ImmUntrackedOrigin
            ](),
        )


struct GpuRgbSpectrumTable:
    var data: DeviceBuffer[.float32]

    def __init__(
        out self, mut ctx: DeviceContext, host: RgbSpectrumTable
    ) raises:
        self.data = upload_list(ctx, host.values)

    def view(self) -> RgbSpectrumView:
        return RgbSpectrumView(
            self.data.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin]()
        )
