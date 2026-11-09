"""Photometric light-selection distributions for spectral transport.

RGB sources use the output color space's luminance vector. Explicit spectra
use their retained photometric normalization. These are sampling densities;
actual emitted energy always comes from wavelength-domain evaluation.
"""

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.math import pi
from bajo.core import Vec3
from bajo.core.color import Color
from bajo.core.colorimetry import srgb_to_xyz_matrix
from bajo.core.rgb_spectrum import RgbSpectrumView
from bajo.core.sampled_spectrum import (
    SampledSpectrum,
    SampledWavelengths,
    cie_xyz,
    CIE_Y_INTEGRAL,
)
from bajo.core.sampling import build_alias_distribution
from bajo.rt.scene_data import SceneData
from bajo.rt.material_types import EnvironmentKind
from bajo.rt.spectrum_sources import (
    SpectrumSource,
    PackedSpectrumSources,
    SpectrumSourcesView,
    NO_SPECTRUM,
)
from bajo.rt.lighting_types import _light_importance

comptime ENVIRONMENT_SPECTRAL_BINS = 32


@always_inline
def _environment_importance[
    width: SIMDLength
](
    rgb: Vec3[.float32, .WORLD, width],
    table: RgbSpectrumView,
    factors: ImmPointer[Float32, ImmUntrackedOrigin],
) -> SIMD[.float32, width]:
    # Integrate the slowly varying RGB polynomial against bins of the exact
    # illuminant * observer product. Narrow authored emission retains support.
    var scale = 2 * rgb.max_component()
    var polynomial = table.coefficients(rgb / scale.gt(0).select(scale, 1))
    var result = SIMD[.float32, width](0)
    comptime for index in range(ENVIRONMENT_SPECTRAL_BINS):
        var nm = Float32(360) + (Float32(index) + 0.5) * (
            471.0 / ENVIRONMENT_SPECTRAL_BINS
        )
        result += (
            polynomial.evaluate(SIMD[.float32, width](nm))
            * factors[unsafe_offset=index]
        )
    return max(scale * result, 0)


def _validate_source(index: UInt32, count: UInt32) raises:
    if index != NO_SPECTRUM and index >= count:
        raise Error("scene spectrum source binding is out of range")


@fieldwise_init
struct SpectralLightDistributionView(DevicePassable, TrivialRegisterPassable):
    var weights: ImmPointer[Float32, ImmUntrackedOrigin]
    var probabilities: ImmPointer[Float32, ImmUntrackedOrigin]
    var aliases: ImmPointer[UInt32, ImmUntrackedOrigin]
    var environment_probabilities: ImmPointer[Float32, ImmUntrackedOrigin]
    var environment_aliases: ImmPointer[UInt32, ImmUntrackedOrigin]
    var source_power: ImmPointer[Float32, ImmUntrackedOrigin]
    var luminance: Color
    var environment_count: UInt32
    var environment_weight: Float32
    var total_weight: Float32
    var environment_floor: Float32
    var environment_factors: ImmPointer[Float32, ImmUntrackedOrigin]
    var rgb: RgbSpectrumView

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "SpectralLightDistributionView"

    @always_inline
    def importance[
        width: SIMDLength
    ](self, rgb: Vec3[.float32, .WORLD, width]) -> SIMD[.float32, width]:
        return max(
            rgb.x * self.luminance.x
            + rgb.y * self.luminance.y
            + rgb.z * self.luminance.z,
            0,
        )

    @always_inline
    def emission(self, source: UInt32, rgb: Color) -> Float32:
        return (
            self.importance(rgb) if source
            == NO_SPECTRUM else self.source_power[unsafe_offset=Int(source)]
        )

    @always_inline
    def environment_pdf[
        width: SIMDLength
    ](
        self, kind: EnvironmentKind, emission: Vec3[.float32, .WORLD, width]
    ) -> SIMD[.float32, width]:
        if self.total_weight <= 0 or self.environment_weight <= 0:
            return 0
        if kind == .IMAGE:
            return (
                max(
                    _environment_importance(
                        emission, self.rgb, self.environment_factors
                    ),
                    self.environment_floor,
                )
                / self.total_weight
            )
        return self.environment_weight / (Float32(4 * pi) * self.total_weight)


struct SpectralLightDistribution:
    var weights: List[Float32]
    var probabilities: List[Float32]
    var aliases: List[UInt32]
    var environment_probabilities: List[Float32]
    var environment_aliases: List[UInt32]
    var source_power: List[Float32]
    var luminance: Color
    var environment_count: UInt32
    var environment_weight: Float32
    var total_weight: Float32
    var environment_floor: Float32
    var environment_factors: List[Float32]
    var rgb: Optional[RgbSpectrumView]
    var light_sources: List[UInt32]
    var light_rgb: List[Color]
    var light_measure: List[Float32]
    var environment_texels: List[Color]
    var environment_kind: EnvironmentKind
    var environment_scale: Color
    var environment_source: UInt32

    def __init__(
        out self, data: SceneData, sources: PackedSpectrumSources
    ) raises:
        for entry in data.surfaces().albedo_spectra.items():
            _validate_source(entry.value, sources.count)
        for entry in data.surfaces().layer_albedo_spectra.items():
            _validate_source(entry.value, sources.count)
        for entry in data.surfaces().dielectric_eta_sources.items():
            _validate_source(entry.value, sources.count)
        for entry in data.emission().records.items():
            _validate_source(entry.value.spectrum, sources.count)
        for source in data.media().spectra:
            _validate_source(source, sources.count)
        _validate_source(data.environment().spectrum, sources.count)
        var matrix = srgb_to_xyz_matrix()
        self.environment_factors = List[Float32](
            length=ENVIRONMENT_SPECTRAL_BINS, fill=0
        )
        self.rgb = Optional[RgbSpectrumView]()
        self.light_sources = List[UInt32]()
        self.light_rgb = List[Color]()
        self.light_measure = List[Float32]()
        self.environment_texels = List[Color]()
        self.environment_kind = data.environment().kind
        self.environment_scale = data.environment().scale
        self.environment_source = data.environment().spectrum
        self.luminance = Color(
            Float32(matrix[1][0]), Float32(matrix[1][1]), Float32(matrix[1][2])
        )
        self.source_power = List[Float32](
            length=max(1, Int(sources.count)), fill=0
        )
        self.weights = List[Float32]()
        self.probabilities = List[Float32]()
        self.aliases = List[UInt32]()
        self.environment_probabilities = List[Float32]()
        self.environment_aliases = List[UInt32]()
        self.environment_count = 0
        self.environment_weight = 0
        self.environment_floor = 0
        self.total_weight = 0
        for light in data.lights().records:
            self.weights.append(0)
            self.light_sources.append(
                data.emitter(light.primitive.value, light.surface).spectrum
            )
            self.light_rgb.append(light.radiance)
            var old = _light_importance(light.radiance)
            self.light_measure.append(light.weight / old if old > 0 else 0)
        ref environment = data.environment()
        if environment.kind == .IMAGE:
            ref texture = data.surfaces().image_textures[
                Int(environment.texture_index)
            ]
            var count = texture.width * texture.height
            self.environment_texels.reserve(count)
            for index in range(count):
                self.environment_texels.append(
                    Color.load(texture.pixels, 3 * index)
                )
            self.environment_count = UInt32(count)
        elif environment.kind != .BLACK:
            var importance = self._rgb_importance(environment.scale)
            if environment.kind == .PROCEDURAL:
                importance = self._rgb_importance(Color(0.75, 0.85, 1))
            self.environment_weight = Float32(4 * pi) * importance
        if not self.weights:
            self.weights.append(0)
        if not self.probabilities:
            self.probabilities.append(1)
            self.aliases.append(0)
        if not self.environment_probabilities:
            self.environment_probabilities.append(1)
            self.environment_aliases.append(0)

    def _rgb_importance(self, rgb: Color) -> Float32:
        return max(
            rgb.x * self.luminance.x
            + rgb.y * self.luminance.y
            + rgb.z * self.luminance.z,
            0,
        )

    def prepare(mut self, spectra: SpectrumSourcesView):
        """Refine all distributions after the reconstruction table is enabled.
        """
        self.rgb = Optional(spectra.rgb)
        self.environment_factors.resize(ENVIRONMENT_SPECTRAL_BINS, fill=0)
        for index in range(ENVIRONMENT_SPECTRAL_BINS):
            self.environment_factors[index] = 0
        for source in range(Int(spectra.count)):
            var integral = Float64(0)
            for nm in range(360, 831):
                var wavelengths = SampledWavelengths[1](
                    SampledSpectrum[1](Float32(nm)), SampledSpectrum[1](1)
                )
                integral += Float64(
                    spectra.emission(UInt32(source), wavelengths).s0
                    * cie_xyz(Float32(nm)).y
                )
            self.source_power[source] = Float32(integral)
        var finite_weights = List[Float32]()
        var finite_total = Float64(0)
        for index in range(len(self.light_sources)):
            var source = self.light_sources[index]
            var importance = (
                self._rgb_importance(self.light_rgb[index]) if source
                == NO_SPECTRUM else self.source_power[Int(source)]
            )
            self.weights[index] = importance
            var weight = importance * self.light_measure[index]
            finite_weights.append(weight)
            finite_total += Float64(weight)
        self.probabilities.clear()
        self.aliases.clear()
        build_alias_distribution(
            Span(finite_weights), finite_total, self.probabilities, self.aliases
        )
        if self.environment_kind == .IMAGE:
            for nm in range(360, 831):
                var wavelengths = SampledWavelengths[1](
                    SampledSpectrum[1](Float32(nm)), SampledSpectrum[1](1)
                )
                var light = (
                    spectra.rgb.sample(
                        self.environment_scale,
                        wavelengths,
                        RgbSpectrumView.ILLUMINANT,
                    )
                    / CIE_Y_INTEGRAL
                )
                if self.environment_source != NO_SPECTRUM:
                    light = spectra.emission(
                        self.environment_source, wavelengths
                    )
                var bin = min(
                    (nm - 360) * ENVIRONMENT_SPECTRAL_BINS // 471,
                    ENVIRONMENT_SPECTRAL_BINS - 1,
                )
                self.environment_factors[bin] += (
                    light.s0 * cie_xyz(Float32(nm)).y
                )
            var weights = List[Float32]()
            var total = Float64(0)
            for texel in self.environment_texels:
                var weight = _environment_importance(
                    texel,
                    spectra.rgb,
                    self.environment_factors.unsafe_ptr()
                    .unsafe_mut_cast[False]()
                    .unsafe_origin_cast[ImmUntrackedOrigin](),
                )
                weights.append(weight)
                total += Float64(weight)
            self.environment_floor = max(
                Float32(total / Float64(len(weights))) * 1e-4, Float32(1e-8)
            )
            total = 0
            for index in range(len(weights)):
                weights[index] = max(weights[index], self.environment_floor)
                total += Float64(weights[index])
            self.environment_probabilities.clear()
            self.environment_aliases.clear()
            build_alias_distribution(
                Span(weights),
                total,
                self.environment_probabilities,
                self.environment_aliases,
            )
            self.environment_weight = Float32(
                4 * Float64(pi) * total / Float64(len(weights))
            )
        elif (
            self.environment_source != NO_SPECTRUM
            and self.environment_kind != .BLACK
        ):
            self.environment_weight = (
                Float32(4 * pi)
                * self.source_power[Int(self.environment_source)]
            )
        self.total_weight = Float32(finite_total) + self.environment_weight
        if not self.probabilities:
            self.probabilities.append(1)
            self.aliases.append(0)

    def view(self) -> SpectralLightDistributionView:
        return SpectralLightDistributionView(
            self.weights.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin](),
            self.probabilities.unsafe_ptr().unsafe_origin_cast[
                ImmUntrackedOrigin
            ](),
            self.aliases.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin](),
            self.environment_probabilities.unsafe_ptr().unsafe_origin_cast[
                ImmUntrackedOrigin
            ](),
            self.environment_aliases.unsafe_ptr().unsafe_origin_cast[
                ImmUntrackedOrigin
            ](),
            self.source_power.unsafe_ptr().unsafe_origin_cast[
                ImmUntrackedOrigin
            ](),
            self.luminance,
            self.environment_count,
            self.environment_weight,
            self.total_weight,
            self.environment_floor,
            self.environment_factors.unsafe_ptr().unsafe_origin_cast[
                ImmUntrackedOrigin
            ](),
            self.rgb.value(),
        )
