"""Pixel addressing, RGB sampling and bulk color-image conversion."""

from max.algorithm import parallelize
from std.builtin.globals import global_constant
from std.math import ceildiv, floor
from std.sys import num_logical_cores, simd_width_of
from bajo.core import Vec3
from bajo.core.color import _SRGB_U8_VALUES, luminance
from bajo.core.tensor_layout import (
    RecordTile,
    gather_record_vec3,
    gather_record_field,
    record_tile,
)

comptime SRGB_SIMD_WIDTH = simd_width_of[Float32]()
comptime SRGB_CHANNELS_PER_TASK = 4 * 1024 * 1024


def linearize_srgb_u8[
    channels_per_task: Int = SRGB_CHANNELS_PER_TASK
](encoded: List[UInt8]) -> List[Float32]:
    """Convert exact u8 sRGB values with threaded native-width SIMD gathers."""
    ref table = global_constant[_SRGB_U8_VALUES]()
    var pixels = List[Float32](unsafe_uninit_length=len(encoded))
    var vector_end = len(encoded) - len(encoded) % SRGB_SIMD_WIDTH
    var vector_count = vector_end // SRGB_SIMD_WIDTH
    comptime assert channels_per_task >= SRGB_SIMD_WIDTH
    var vectors_per_task = channels_per_task // SRGB_SIMD_WIDTH
    var task_count = ceildiv(vector_count, vectors_per_task)

    def worker(task_idx: Int) {imm, mut pixels}:
        var vector_begin = task_idx * vectors_per_task
        var vector_stop = min(vector_begin + vectors_per_task, vector_count)
        var encoded_ptr = encoded.unsafe_ptr()
        var table_ptr = table.unsafe_ptr()
        var pixels_ptr = pixels.unsafe_ptr()
        for vector_idx in range(vector_begin, vector_stop):
            var base = vector_idx * SRGB_SIMD_WIDTH
            var indices = (
                encoded_ptr.unsafe_offset(base)
                .unsafe_load[width=SRGB_SIMD_WIDTH]()
                .cast[.int32]()
            )
            var linear = table_ptr.unsafe_gather(indices)
            pixels_ptr.unsafe_offset(base).unsafe_store[width=SRGB_SIMD_WIDTH](
                linear
            )

    if task_count > 1:
        parallelize(worker, task_count, min(num_logical_cores(), task_count))
    elif task_count == 1:
        worker(0)

    for index in range(vector_end, len(encoded)):
        pixels[index] = table[Int(encoded[index])]
    return pixels^


@always_inline
def image_pixel_index[
    length: SIMDLength = 1
](
    width: Int,
    height: Int,
    u: SIMD[.float32, length],
    v: SIMD[.float32, length],
) -> SIMD[.int64, length]:
    """Repeat UVs and map their lower-left origin to top-to-bottom RGB rows."""
    var wrapped_u = u - floor(u)
    var wrapped_v = v - floor(v)
    var x = min(
        (wrapped_u * Float32(width)).cast[.int64](),
        SIMD[.int64, length](width - 1),
    )
    var y = min(
        ((Float32(1.0) - wrapped_v) * Float32(height)).cast[.int64](),
        SIMD[.int64, length](height - 1),
    )
    return y * SIMD[.int64, length](width) + x


@always_inline
def sample_image_texture[
    length: SIMDLength,
](
    pixels: RecordTile[.float32, 3, _],
    width: Int,
    height: Int,
    u: SIMD[.float32, length],
    v: SIMD[.float32, length],
) -> Vec3[.float32, .WORLD, length]:
    return gather_record_vec3(pixels, image_pixel_index(width, height, u, v))


@always_inline
def load_environment_texel[
    length: SIMDLength,
](
    pixels: RecordTile[.float32, 3, _],
    index: SIMD[.int64, length],
) -> Vec3[
    .float32, .WORLD, length
]:
    var value = gather_record_vec3(pixels, index)
    return Vec3[.float32, .WORLD, length](
        max(value.x, 0.0), max(value.y, 0.0), max(value.z, 0.0)
    )


@always_inline
def sample_environment_texture[
    length: SIMDLength,
](
    pixels: RecordTile[.float32, 3, _],
    width: Int,
    height: Int,
    u: SIMD[.float32, length],
    v: SIMD[.float32, length],
) -> Vec3[.float32, .WORLD, length]:
    """Clamp equal-area UVs to the environment's top-to-bottom RGB rows."""
    var x = (
        (u * Float32(width))
        .cast[.int64]()
        .clamp(0, SIMD[.int64, length](width - 1))
    )
    var y = (
        (v * Float32(height))
        .cast[.int64]()
        .clamp(0, SIMD[.int64, length](height - 1))
    )
    return load_environment_texel(pixels, y * SIMD[.int64, length](width) + x)


@always_inline
def sample_image_texture_bilinear[
    length: SIMDLength,
](
    pixels: RecordTile[.float32, 3, _],
    width: Int,
    height: Int,
    u: SIMD[.float32, length],
    v: SIMD[.float32, length],
) -> Vec3[.float32, .WORLD, length]:
    var x = (u - floor(u)) * Float32(width) - 0.5
    var y = (1.0 - (v - floor(v))) * Float32(height) - 0.5
    var fx = floor(x)
    var fy = floor(y)
    var dx = x - fx
    var dy = y - fy
    var x0 = fx.cast[.int64]()
    var y0 = fy.cast[.int64]()
    var w = SIMD[.int64, length](width)
    var h = SIMD[.int64, length](height)
    var x1 = ((x0 + 1) % w + w) % w
    var y1 = ((y0 + 1) % h + h) % h
    x0 = (x0 % w + w) % w
    y0 = (y0 % h + h) % h
    var top = (
        gather_record_vec3(pixels, y0 * w + x0) * (1.0 - dx)
        + gather_record_vec3(pixels, y0 * w + x1) * dx
    )
    var bottom = (
        gather_record_vec3(pixels, y1 * w + x0) * (1.0 - dx)
        + gather_record_vec3(pixels, y1 * w + x1) * dx
    )
    return top * (1.0 - dy) + bottom * dy


@always_inline
def sample_scalar_texture_bilinear[
    length: SIMDLength,
](
    pixels: RecordTile[.float32, 1, _],
    width: Int,
    height: Int,
    u: SIMD[.float32, length],
    v: SIMD[.float32, length],
) -> SIMD[.float32, length]:
    var x = (u - floor(u)) * Float32(width) - 0.5
    var y = (1.0 - (v - floor(v))) * Float32(height) - 0.5
    var fx = floor(x)
    var fy = floor(y)
    var dx = x - fx
    var dy = y - fy
    var x0 = fx.cast[.int64]()
    var y0 = fy.cast[.int64]()
    var w = SIMD[.int64, length](width)
    var h = SIMD[.int64, length](height)
    var x1 = ((x0 + 1) % w + w) % w
    var y1 = ((y0 + 1) % h + h) % h
    x0 = (x0 % w + w) % w
    y0 = (y0 % h + h) % h
    var top = (
        gather_record_field[0](pixels, y0 * w + x0) * (1.0 - dx)
        + gather_record_field[0](pixels, y0 * w + x1) * dx
    )
    var bottom = (
        gather_record_field[0](pixels, y1 * w + x0) * (1.0 - dx)
        + gather_record_field[0](pixels, y1 * w + x1) * dx
    )
    return top * (1.0 - dy) + bottom * dy


struct ImageTexture:
    """Linear RGB image used by host-side material sampling."""

    var width: Int
    var height: Int
    var pixels: List[Float32]
    var alpha: List[Float32]
    var has_scalar_alpha: Bool

    def __init__(
        out self,
        width: Int,
        height: Int,
        var pixels: List[Float32],
        var alpha: List[Float32] = List[Float32](),
    ):
        self.width = width
        self.height = height
        self.pixels = pixels^
        self.alpha = alpha^
        self.has_scalar_alpha = False
        for value in self.alpha:
            if value != 1.0:
                self.has_scalar_alpha = True
                break

    @always_inline
    def sample[
        length: SIMDLength = 1
    ](self, u: SIMD[.float32, length], v: SIMD[.float32, length]) -> Vec3[
        .float32, .WORLD, length
    ]:
        return sample_image_texture(
            record_tile[.float32, 3](
                self.pixels.unsafe_ptr(), self.width * self.height
            ),
            self.width,
            self.height,
            u,
            v,
        )

    @always_inline
    def sample_luminance[
        length: SIMDLength = 1
    ](self, u: SIMD[.float32, length], v: SIMD[.float32, length]) -> SIMD[
        .float32, length
    ]:
        return luminance(self.sample(u, v))

    @always_inline
    def sample_bilinear[
        length: SIMDLength = 1
    ](self, u: SIMD[.float32, length], v: SIMD[.float32, length]) -> Vec3[
        .float32, .WORLD, length
    ]:
        return sample_image_texture_bilinear(
            record_tile[.float32, 3](
                self.pixels.unsafe_ptr(), self.width * self.height
            ),
            self.width,
            self.height,
            u,
            v,
        )

    @always_inline
    def sample_scalar_bilinear[
        length: SIMDLength = 1
    ](self, u: SIMD[.float32, length], v: SIMD[.float32, length]) -> SIMD[
        .float32, length
    ]:
        # PBRT scalar maps use meaningful alpha, otherwise average RGB.
        if self.has_scalar_alpha:
            return sample_scalar_texture_bilinear(
                record_tile[.float32, 1](
                    self.alpha.unsafe_ptr(), self.width * self.height
                ),
                self.width,
                self.height,
                u,
                v,
            )
        var rgb = self.sample_bilinear(u, v)
        return (rgb.x + rgb.y + rgb.z) / 3.0

    @always_inline
    def sample_environment[
        length: SIMDLength = 1
    ](self, u: SIMD[.float32, length], v: SIMD[.float32, length]) -> Vec3[
        .float32, .WORLD, length
    ]:
        """Nearest lookup in PBRT's top-to-bottom equal-area image domain."""
        return sample_environment_texture(
            record_tile[.float32, 3](
                self.pixels.unsafe_ptr(), self.width * self.height
            ),
            self.width,
            self.height,
            u,
            v,
        )
