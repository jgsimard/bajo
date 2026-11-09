"""Lazy GPU spectral resources; RGB scenes retain only small host bindings."""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.collections import Array
from std.sys.info import size_of
from bajo.core.rgb_spectrum import RgbSpectrumTable, RgbSpectrumView
from bajo.core.spectrum_tables import (
    PiecewiseSpectrumTable,
    PiecewiseSpectrumView,
)
from bajo.rt.scene_data import SceneData
from bajo.rt.material_types import SurfaceId, MaterialKind
from bajo.rt.spectrum_sources import (
    SpectrumSource,
    PackedSpectrumSources,
    SpectrumSourcesView,
    NO_SPECTRUM,
)
from bajo.rt.gpu.upload import _upload_nonempty
from bajo.rt.gpu.spectra import GpuRgbSpectrumTable
from bajo.rt.spectral_light_distribution import (
    SpectralLightDistribution,
    SpectralLightDistributionView,
)


@fieldwise_init
struct GpuSpectralSceneView(DevicePassable, TrivialRegisterPassable):
    var sources: SpectrumSourcesView
    var lambertians: ImmPointer[UInt32, ImmUntrackedOrigin]
    var metals: ImmPointer[UInt32, ImmUntrackedOrigin]
    var coated: ImmPointer[UInt32, ImmUntrackedOrigin]
    var layers: ImmPointer[UInt32, ImmUntrackedOrigin]
    var reflectance: ImmPointer[UInt32, ImmUntrackedOrigin]
    var media: ImmPointer[UInt32, ImmUntrackedOrigin]
    var emitters: ImmPointer[UInt32, ImmUntrackedOrigin]
    var environment: UInt32
    var dielectrics: ImmPointer[UInt32, ImmUntrackedOrigin]
    var dielectric_dispersive: ImmPointer[UInt32, ImmUntrackedOrigin]
    var lighting: SpectralLightDistributionView

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "GpuSpectralSceneView"


struct _GpuSpectralBuffers:
    var nodes: DeviceBuffer[.uint32]
    var leaves: DeviceBuffer[.uint32]
    var ranges: DeviceBuffer[.uint32]
    var photometric: DeviceBuffer[.float32]
    var bindings: DeviceBuffer[.uint32]
    var rgb: GpuRgbSpectrumTable
    var light_weights: DeviceBuffer[.float32]
    var light_probabilities: DeviceBuffer[.float32]
    var light_aliases: DeviceBuffer[.uint32]
    var environment_probabilities: DeviceBuffer[.float32]
    var environment_aliases: DeviceBuffer[.uint32]
    var source_power: DeviceBuffer[.float32]
    var environment_factors: DeviceBuffer[.float32]

    def __init__(
        out self,
        mut ctx: DeviceContext,
        host: PackedSpectrumSources,
        bindings: ImmSpan[UInt32, _],
        lighting: SpectralLightDistribution,
        rgb: RgbSpectrumTable,
    ) raises:
        comptime words = size_of[SpectrumSource]() // size_of[UInt32]()
        self.nodes = _upload_nonempty(
            ctx,
            ImmSpan(
                unsafe_ptr=host.sources.unsafe_ptr().unsafe_bitcast[UInt32](),
                length=len(host.sources) * words,
            ),
        )
        self.leaves = _upload_nonempty(ctx, host.leaves)
        self.ranges = _upload_nonempty(ctx, host.ranges)
        self.photometric = _upload_nonempty(ctx, host.photometric)
        self.bindings = _upload_nonempty(ctx, bindings)
        self.rgb = GpuRgbSpectrumTable(ctx, rgb)
        self.light_weights = _upload_nonempty(ctx, lighting.weights)
        self.light_probabilities = _upload_nonempty(ctx, lighting.probabilities)
        self.light_aliases = _upload_nonempty(ctx, lighting.aliases)
        self.environment_probabilities = _upload_nonempty(
            ctx, lighting.environment_probabilities
        )
        self.environment_aliases = _upload_nonempty(
            ctx, lighting.environment_aliases
        )
        self.source_power = _upload_nonempty(ctx, lighting.source_power)
        self.environment_factors = _upload_nonempty(
            ctx, lighting.environment_factors
        )


struct GpuSpectralResources:
    var host: PackedSpectrumSources
    var lighting: SpectralLightDistribution
    var bindings: List[UInt32]
    var offsets: Array[Int, 9]
    var environment: UInt32
    var device: Optional[_GpuSpectralBuffers]
    var curves: PiecewiseSpectrumTable
    var rgb: Optional[RgbSpectrumTable]

    def __init__(out self, data: SceneData) raises:
        ref surfaces = data.surfaces()
        self.host = PackedSpectrumSources(surfaces.spectrum_sources)
        self.curves = PiecewiseSpectrumTable(surfaces.optical_spectra)
        self.rgb = Optional[RgbSpectrumTable]()
        # RGB/product normalizers use CIE_Y_INTEGRAL without accessing RGB
        # reconstruction coefficients. The table remains unloaded in RGB mode.
        self.host.prepare_photometric(self.curves.view())
        self.lighting = SpectralLightDistribution(data, self.host)
        self.bindings = List[UInt32]()
        self.offsets = Array[Int, 9](fill=0)
        self.environment = data.environment().spectrum
        self.device = Optional[_GpuSpectralBuffers]()
        comptime for group in range(3):
            self.offsets[group] = len(self.bindings)
            var count = (
                len(surfaces.lambertians) if group
                == 0 else len(surfaces.metals) if group
                == 1 else len(surfaces.coated_diffuses)
            )
            for index in range(count):
                var kind = (
                    MaterialKind.LAMBERTIAN if group
                    == 0 else MaterialKind.METAL if group
                    == 1 else MaterialKind.COATED_DIFFUSE
                )
                var surface = SurfaceId[1](kind, UInt32(index))
                self.bindings.append(
                    surfaces.albedo_spectra.get(surface.value[0]).or_else(
                        NO_SPECTRUM
                    )
                )
        self.offsets[3] = len(self.bindings)
        for index in range(len(surfaces.coated_diffuses)):
            var surface = SurfaceId[1](.COATED_DIFFUSE, UInt32(index))
            self.bindings.append(
                surfaces.layer_albedo_spectra.get(surface.value[0]).or_else(
                    NO_SPECTRUM
                )
            )
        self.offsets[4] = len(self.bindings)
        for material in surfaces.coated_diffuses:
            self.bindings.append(UInt32(material.conductor_reflectance))
        self.offsets[5] = len(self.bindings)
        self.bindings.extend(Span(data.media().spectra))
        self.offsets[6] = len(self.bindings)
        var keys = List[UInt32]()
        for entry in data.emission().records.items():
            keys.append(entry.key)
        sort(keys)
        for key in keys:
            self.bindings.append(data.emission().records[key].spectrum)

        self.offsets[7] = len(self.bindings)
        for index in range(len(surfaces.dielectrics)):
            var surface = SurfaceId[1](.DIELECTRIC, UInt32(index))
            self.bindings.append(
                surfaces.dielectric_eta_sources.get(surface.value[0]).or_else(
                    NO_SPECTRUM
                )
            )
        self.offsets[8] = len(self.bindings)
        for index in range(len(surfaces.dielectrics)):
            var source = self.bindings[self.offsets[7] + index]
            var dispersive = False
            if source != NO_SPECTRUM:
                ref descriptor = self.host.sources[Int(source)]
                dispersive = (
                    descriptor.kind == descriptor.CURVE
                    or descriptor.kind == descriptor.PRODUCT
                    or (
                        descriptor.kind == descriptor.RGB
                        and (
                            descriptor.value.x != descriptor.value.y
                            or descriptor.value.y != descriptor.value.z
                        )
                    )
                )
            self.bindings.append(UInt32(dispersive))

    def enable(mut self, mut ctx: DeviceContext) raises:
        if not self.device:
            self.rgb = Optional(RgbSpectrumTable.read_default())
            self.lighting.prepare(
                self.host.view(self.curves.view(), self.rgb.value().view())
            )
            self.device = Optional(
                _GpuSpectralBuffers(
                    ctx,
                    self.host,
                    self.bindings,
                    self.lighting,
                    self.rgb.value(),
                )
            )

    def view(
        self, curves: PiecewiseSpectrumView
    ) -> Optional[GpuSpectralSceneView]:
        if not self.device:
            return Optional[GpuSpectralSceneView]()
        ref device = self.device.value()
        var bindings = device.bindings.unsafe_ptr().unsafe_origin_cast[
            ImmUntrackedOrigin
        ]()
        return Optional(
            GpuSpectralSceneView(
                SpectrumSourcesView(
                    device.nodes.unsafe_ptr()
                    .unsafe_bitcast[SpectrumSource]()
                    .unsafe_origin_cast[ImmUntrackedOrigin](),
                    device.leaves.unsafe_ptr().unsafe_origin_cast[
                        ImmUntrackedOrigin
                    ](),
                    device.ranges.unsafe_ptr().unsafe_origin_cast[
                        ImmUntrackedOrigin
                    ](),
                    self.host.count,
                    curves,
                    device.rgb.view(),
                    device.photometric.unsafe_ptr().unsafe_origin_cast[
                        ImmUntrackedOrigin
                    ](),
                ),
                bindings.unsafe_offset(self.offsets[0]),
                bindings.unsafe_offset(self.offsets[1]),
                bindings.unsafe_offset(self.offsets[2]),
                bindings.unsafe_offset(self.offsets[3]),
                bindings.unsafe_offset(self.offsets[4]),
                bindings.unsafe_offset(self.offsets[5]),
                bindings.unsafe_offset(self.offsets[6]),
                self.environment,
                bindings.unsafe_offset(self.offsets[7]),
                bindings.unsafe_offset(self.offsets[8]),
                SpectralLightDistributionView(
                    device.light_weights.unsafe_ptr().unsafe_origin_cast[
                        ImmUntrackedOrigin
                    ](),
                    device.light_probabilities.unsafe_ptr().unsafe_origin_cast[
                        ImmUntrackedOrigin
                    ](),
                    device.light_aliases.unsafe_ptr().unsafe_origin_cast[
                        ImmUntrackedOrigin
                    ](),
                    device.environment_probabilities.unsafe_ptr().unsafe_origin_cast[
                        ImmUntrackedOrigin
                    ](),
                    device.environment_aliases.unsafe_ptr().unsafe_origin_cast[
                        ImmUntrackedOrigin
                    ](),
                    device.source_power.unsafe_ptr().unsafe_origin_cast[
                        ImmUntrackedOrigin
                    ](),
                    self.lighting.luminance,
                    self.lighting.environment_count,
                    self.lighting.environment_weight,
                    self.lighting.total_weight,
                    self.lighting.environment_floor,
                    device.environment_factors.unsafe_ptr().unsafe_origin_cast[
                        ImmUntrackedOrigin
                    ](),
                    device.rgb.view(),
                ),
            )
        )
