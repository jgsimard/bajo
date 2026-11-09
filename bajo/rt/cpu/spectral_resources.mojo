"""Host-owned spectral data retained by a CPU renderer workspace."""

from bajo.core.rgb_spectrum import RgbSpectrumTable
from bajo.core.spectrum_tables import PiecewiseSpectrumTable
from bajo.rt.spectrum_sources import PackedSpectrumSources, SpectrumSourcesView
from bajo.rt.material_types import SurfaceStore
from bajo.rt.scene_data import SceneData
from bajo.rt.spectral_light_distribution import SpectralLightDistribution


struct CpuSpectralResources:
    var rgb: RgbSpectrumTable
    var curves: PiecewiseSpectrumTable
    var sources: PackedSpectrumSources
    var lighting: Optional[SpectralLightDistribution]

    def __init__(out self, surfaces: SurfaceStore) raises:
        self.rgb = RgbSpectrumTable.read_default()
        self.lighting = Optional[SpectralLightDistribution]()
        self.curves = PiecewiseSpectrumTable(surfaces.optical_spectra)
        self.sources = PackedSpectrumSources(surfaces.spectrum_sources)
        self.sources.prepare_photometric(self.curves.view())

    def prepare_lighting(mut self, data: SceneData) raises:
        self.lighting = Optional(SpectralLightDistribution(data, self.sources))
        self.lighting.value().prepare(self.view())

    def prepare(mut self, surfaces: SurfaceStore) raises:
        self.lighting = Optional[SpectralLightDistribution]()
        self.curves = PiecewiseSpectrumTable(surfaces.optical_spectra)
        self.sources = PackedSpectrumSources(surfaces.spectrum_sources)
        self.sources.prepare_photometric(self.curves.view())

    def view(self) -> SpectrumSourcesView:
        return self.sources.view(self.curves.view(), self.rgb.view())
