"""Render configuration, sampling, timing, and result value types."""

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder

from bajo.core import Vec3f32
from bajo.core.random import Sampler


comptime Color = Vec3f32[.WORLD]


struct RenderSettings(Copyable, Writable):
    var image_width: Int
    var image_height: Int
    var samples_per_pixel: Int
    var rng_seed: UInt64
    var max_depth: Int
    var sampler: Sampler
    var sample_offset: Int
    var sample_sequence_length: Int

    def __init__(
        out self,
        image_width: Int,
        image_height: Int,
        samples_per_pixel: Int,
        rng_seed: UInt64,
        max_depth: Int = 8,
        sampler: Sampler = .INDEPENDENT,
        sample_offset: Int = 0,
        sample_sequence_length: Int = 0,
    ):
        debug_assert["safe", _use_compiler_assume=True](
            image_width > 0, "image width must be positive"
        )
        debug_assert["safe", _use_compiler_assume=True](
            image_height > 0, "image height must be positive"
        )
        debug_assert["safe", _use_compiler_assume=True](
            samples_per_pixel > 0, "samples per pixel must be positive"
        )
        debug_assert["safe", _use_compiler_assume=True](
            max_depth >= 0, "max depth must be non-negative"
        )
        var sequence_length = sample_sequence_length
        if sequence_length == 0:
            sequence_length = samples_per_pixel
        debug_assert["safe", _use_compiler_assume=True](
            sampler.is_valid(), "unknown sampler"
        )
        debug_assert["safe", _use_compiler_assume=True](
            sample_offset >= 0
            and sample_offset + samples_per_pixel <= sequence_length,
            "sample batch is outside the sample sequence",
        )

        self.image_width = image_width
        self.image_height = image_height
        self.samples_per_pixel = samples_per_pixel
        self.rng_seed = rng_seed
        self.max_depth = max_depth
        self.sampler = sampler
        self.sample_offset = sample_offset
        self.sample_sequence_length = sequence_length

    def validate(self) raises:
        """Validate mutable render settings at a host API boundary."""
        if self.image_width <= 0:
            raise Error("image width must be positive")
        if self.image_height <= 0:
            raise Error("image height must be positive")
        if self.samples_per_pixel <= 0:
            raise Error("samples per pixel must be positive")
        if self.max_depth < 0:
            raise Error("max depth must be non-negative")
        if not self.sampler.is_valid():
            raise Error("unknown sampler")
        if self.sample_offset < 0:
            raise Error("sample offset must be non-negative")
        if self.sample_sequence_length <= 0:
            raise Error("sample sequence length must be positive")
        if (
            self.sample_offset > self.sample_sequence_length
            or self.samples_per_pixel
            > self.sample_sequence_length - self.sample_offset
        ):
            raise Error("sample batch is outside the sample sequence")

        var u32_max = UInt64(UInt32.MAX)
        if (
            UInt64(self.image_width) > u32_max
            or UInt64(self.image_height) > u32_max
            or UInt64(self.samples_per_pixel) > u32_max
            or UInt64(self.max_depth) > u32_max
            or UInt64(self.sample_offset) > u32_max
            or UInt64(self.sample_sequence_length) > u32_max
        ):
            raise Error("render settings exceed the 32-bit sampling contract")

        var pixel_count = UInt64(self.image_width) * UInt64(self.image_height)
        if pixel_count > u32_max / UInt64(self.samples_per_pixel):
            raise Error("render requires more than 2^32-1 sample paths")


@fieldwise_init
struct SamplingConfig(
    Copyable, DevicePassable, TrivialRegisterPassable, Writable
):
    """Compact CPU/GPU description of one batch in a pixel sample sequence."""

    var seed: UInt64
    var sampler_value: UInt32
    var samples_per_pixel: UInt32
    var sample_offset: UInt32
    var sequence_length: UInt32
    var image_width: UInt32

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "SamplingConfig"

    @staticmethod
    def from_settings(settings: RenderSettings) -> Self:
        return Self(
            settings.rng_seed,
            settings.sampler.value,
            UInt32(settings.samples_per_pixel),
            UInt32(settings.sample_offset),
            UInt32(settings.sample_sequence_length),
            UInt32(settings.image_width),
        )


@fieldwise_init
struct RenderTimings(Copyable, Writable):
    var total_ns: Int
    var init_ns: Int
    var render_ns: Int
    var pixel_count: Int
    var sample_count: Int
    var max_depth: Int


struct RenderResult:
    var pixels: List[Color]
    var timings: RenderTimings

    def __init__(
        out self,
        var pixels: List[Color],
        timings: RenderTimings,
    ):
        self.pixels = pixels^
        self.timings = timings.copy()
