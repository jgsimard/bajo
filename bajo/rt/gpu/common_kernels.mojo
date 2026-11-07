"""Geometry-independent GPU RT kernels."""

from max.gpu import global_idx

from bajo.core.camera import Camera
from bajo.rt.rays import make_camera_ray_from_samples
from bajo.rt.render_types import SamplingConfig
from bajo.rt.film import FilmResponse
from bajo.core.vec import Vec3
from bajo.core.color import Color
from bajo.rt.gpu.sampling import GpuRtSamplingConfig
from bajo.rt.material_types import Integrator
from bajo.rt.wavefront_contract import DeviceWavePath, WaveSampleFloatAbi
from bajo.rt.gpu.queue_layout import gpu_rt_radiance_tile
from bajo.rt.gpu.wavefront_contract import store_gpu_rt_path_values
from bajo.rt.camera_sampling import sample_camera


comptime GPU_RT_BLOCK_SIZE = 64
comptime GPU_RT_MAX_BLOCKS = 1 << 30


@always_inline
def gpu_rt_bounce_block_limit(bounce: UInt32, block_size: Int) -> Int:
    """Bound late-bounce grids while grid-stride loops retain every path."""
    if bounce >= 64:
        return max(1024 // block_size, 1)
    if bounce >= 32:
        return max(4096 // block_size, 1)
    if bounce >= 16:
        return max(16384 // block_size, 1)
    if bounce >= 8:
        return max(65536 // block_size, 1)
    return GPU_RT_MAX_BLOCKS


@always_inline
def make_gpu_rt_primary_path(
    camera_params: Pointer[Float32, ImmutAnyOrigin],
    sample_base: UInt32,
    idx: Int,
    width_i32: Int32,
    height_i32: Int32,
    samples_per_pixel_i32: Int32,
    sampling: SamplingConfig,
    sample_radiance: Pointer[Float32, MutAnyOrigin],
    capacity: Int,
) -> DeviceWavePath:
    var width = Int(width_i32)
    var height = Int(height_i32)
    var samples_per_pixel = Int(samples_per_pixel_i32)
    var path_id = sample_base + UInt32(idx)
    var pixel_idx = Int(path_id) / samples_per_pixel
    var px = pixel_idx % width
    var py = pixel_idx / width
    var samples = sample_camera(sampling, path_id)
    var radiance = gpu_rt_radiance_tile(sample_radiance, capacity)
    radiance[(WaveSampleFloatAbi.FILTER_WEIGHT, idx)] = samples.filter_weight
    var camera = Camera(Span(unsafe_ptr=camera_params, length=Camera.STRIDE))
    var ray = make_camera_ray_from_samples(
        camera,
        Float32(px),
        Float32(py),
        Float32(width),
        Float32(height),
        samples.pixel_u,
        samples.pixel_v,
        samples.lens_u,
        samples.lens_v,
    )
    return DeviceWavePath(
        path_id,
        ray.o.x,
        ray.o.y,
        ray.o.z,
        ray.t_min,
        ray.d.x,
        ray.d.y,
        ray.d.z,
        ray.t_max,
        1.0,
        1.0,
        1.0,
        0.0,
        True,
    )


@always_inline
def clear_gpu_rt_sample(
    sample_radiance: Pointer[Float32, MutAnyOrigin],
    capacity: Int,
    idx: Int,
):
    var radiance = gpu_rt_radiance_tile(sample_radiance, capacity)
    radiance[(WaveSampleFloatAbi.R, idx)] = 0.0
    radiance[(WaveSampleFloatAbi.G, idx)] = 0.0
    radiance[(WaveSampleFloatAbi.B, idx)] = 0.0


def gpu_rt_primary_kernel[
    integrator: Integrator,
](
    camera_params: Pointer[Float32, ImmutAnyOrigin],
    path_ids: Pointer[UInt32, MutAnyOrigin],
    path_fields: Pointer[Float32, MutAnyOrigin],
    sample_radiance: Pointer[Float32, MutAnyOrigin],
    capacity_i32: Int32,
    active_count_i32: Int32,
    sample_base: UInt32,
    width_i32: Int32,
    height_i32: Int32,
    samples_per_pixel_i32: Int32,
    sampling_arg: GpuRtSamplingConfig,
):
    var sampling = sampling_arg.load()
    var idx = global_idx.x
    var capacity = Int(capacity_i32)
    var active_count = Int(active_count_i32)
    if idx >= active_count:
        return

    store_gpu_rt_path_values[integrator](
        make_gpu_rt_primary_path(
            camera_params,
            sample_base,
            idx,
            width_i32,
            height_i32,
            samples_per_pixel_i32,
            sampling,
            sample_radiance,
            capacity,
        ),
        path_ids,
        path_fields,
        capacity,
        idx,
    )
    clear_gpu_rt_sample(sample_radiance, capacity, idx)


@always_inline
def _gpu_rt_resolve_pixel[
    apply_film: Bool
](
    sample_radiance: Pointer[Float32, ImmutAnyOrigin],
    pixels: Pointer[Float32, MutAnyOrigin],
    sample_capacity_i32: Int32,
    pixel_base_i32: Int32,
    pixel_count_i32: Int32,
    samples_per_pixel_i32: Int32,
    film: FilmResponse,
):
    var pixel_idx = global_idx.x
    var sample_capacity = Int(sample_capacity_i32)
    var pixel_base = Int(pixel_base_i32)
    var pixel_count = Int(pixel_count_i32)
    if pixel_idx >= pixel_count:
        return
    var samples_per_pixel = Int(samples_per_pixel_i32)
    var red = Float32(0.0)
    var green = Float32(0.0)
    var blue = Float32(0.0)
    var weight_sum = Float32(0.0)
    var sample_begin = pixel_idx * samples_per_pixel
    var radiance = gpu_rt_radiance_tile(sample_radiance, sample_capacity)
    if samples_per_pixel % 4 == 0 and sample_capacity % 4 == 0:
        var sample_idx = sample_begin
        while sample_idx < sample_begin + samples_per_pixel:
            var r = radiance.load[width=4, alignment=16](
                (WaveSampleFloatAbi.R, sample_idx)
            )
            var g = radiance.load[width=4, alignment=16](
                (WaveSampleFloatAbi.G, sample_idx)
            )
            var b = radiance.load[width=4, alignment=16](
                (WaveSampleFloatAbi.B, sample_idx)
            )
            var weights = radiance.load[width=4, alignment=16](
                (WaveSampleFloatAbi.FILTER_WEIGHT, sample_idx)
            )
            var samples = Vec3[.float32, .WORLD, 4](r, g, b)
            comptime if apply_film:
                samples = film.apply_sample(samples)
            samples = samples * weights
            # Retain the original summation order while grouping memory reads.
            comptime for lane in range(4):
                red += samples.x[lane]
                green += samples.y[lane]
                blue += samples.z[lane]
                weight_sum += weights[lane]
            sample_idx += 4
    else:
        for sample_idx in range(sample_begin, sample_begin + samples_per_pixel):
            var sample = Color(
                radiance[(WaveSampleFloatAbi.R, sample_idx)],
                radiance[(WaveSampleFloatAbi.G, sample_idx)],
                radiance[(WaveSampleFloatAbi.B, sample_idx)],
            )
            comptime if apply_film:
                sample = film.apply_sample(sample)
            var weight = radiance[
                (WaveSampleFloatAbi.FILTER_WEIGHT, sample_idx)
            ]
            red += sample.x * weight
            green += sample.y * weight
            blue += sample.z * weight
            weight_sum += weight
    var scale = Float32(1.0) / weight_sum if weight_sum != 0 else Float32(0.0)
    var output_idx = pixel_base + pixel_idx
    var color = Color(red * scale, green * scale, blue * scale)
    pixels[unsafe_offset=3 * output_idx + 0] = color.x
    pixels[unsafe_offset=3 * output_idx + 1] = color.y
    pixels[unsafe_offset=3 * output_idx + 2] = color.z


def gpu_rt_resolve_kernel(
    sample_radiance: Pointer[Float32, ImmutAnyOrigin],
    pixels: Pointer[Float32, MutAnyOrigin],
    sample_capacity_i32: Int32,
    pixel_base_i32: Int32,
    pixel_count_i32: Int32,
    samples_per_pixel_i32: Int32,
):
    _gpu_rt_resolve_pixel[False](
        sample_radiance,
        pixels,
        sample_capacity_i32,
        pixel_base_i32,
        pixel_count_i32,
        samples_per_pixel_i32,
        FilmResponse(),
    )


def gpu_rt_resolve_film_kernel(
    sample_radiance: Pointer[Float32, ImmutAnyOrigin],
    pixels: Pointer[Float32, MutAnyOrigin],
    sample_capacity_i32: Int32,
    pixel_base_i32: Int32,
    pixel_count_i32: Int32,
    samples_per_pixel_i32: Int32,
    film: FilmResponse,
):
    _gpu_rt_resolve_pixel[True](
        sample_radiance,
        pixels,
        sample_capacity_i32,
        pixel_base_i32,
        pixel_count_i32,
        samples_per_pixel_i32,
        film,
    )
