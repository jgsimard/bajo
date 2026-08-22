"""Geometry-independent GPU RT kernels."""

from std.gpu import global_idx

from bajo.bvh import Camera
from bajo.core import Frame
from bajo.core.random import random_in_unit_disk
from bajo.rt.common import path_stage_rng
from bajo.rt.types import RENDER
from bajo.rt.wavefront_contract import (
    DeviceWavePath,
    WaveSampleFloatAbi,
    wavefront_plane_index,
)
from bajo.rt.gpu.wavefront_contract import store_gpu_rt_path


comptime GPU_RT_BLOCK_SIZE = 64
comptime GPU_RT_MAX_BLOCKS = 1 << 30


def gpu_rt_primary_kernel[
    ALGORITHM: RENDER,
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
    rng_seed: UInt64,
):
    var idx = global_idx.x
    var capacity = Int(capacity_i32)
    var active_count = Int(active_count_i32)
    if idx >= active_count:
        return

    var width = Int(width_i32)
    var height = Int(height_i32)
    var samples_per_pixel = Int(samples_per_pixel_i32)
    var path_id = sample_base + UInt32(idx)
    var pixel_idx = Int(path_id) / samples_per_pixel
    var px = pixel_idx % width
    var py = pixel_idx / width
    var rng = path_stage_rng(rng_seed, path_id, UInt32(0))
    var lens = random_in_unit_disk[Frame.WORLD](rng)
    var camera = Camera(Span(unsafe_ptr=camera_params, length=Camera.STRIDE))
    var ray = camera.make_ray_sampled(
        Float32(px),
        Float32(py),
        Float32(width),
        Float32(height),
        rng.f32(),
        rng.f32(),
        lens.x,
        lens.y,
        0.001,
    )
    store_gpu_rt_path[ALGORITHM](
        DeviceWavePath(
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
        ),
        path_ids,
        path_fields,
        capacity,
        idx,
    )
    sample_radiance[
        unsafe_offset=wavefront_plane_index(WaveSampleFloatAbi.R, capacity, idx)
    ] = 0.0
    sample_radiance[
        unsafe_offset=wavefront_plane_index(WaveSampleFloatAbi.G, capacity, idx)
    ] = 0.0
    sample_radiance[
        unsafe_offset=wavefront_plane_index(WaveSampleFloatAbi.B, capacity, idx)
    ] = 0.0


def gpu_rt_resolve_kernel(
    sample_radiance: Pointer[Float32, ImmutAnyOrigin],
    pixels: Pointer[Float32, MutAnyOrigin],
    sample_capacity_i32: Int32,
    pixel_base_i32: Int32,
    pixel_count_i32: Int32,
    samples_per_pixel_i32: Int32,
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
    var sample_begin = pixel_idx * samples_per_pixel
    for sample_idx in range(sample_begin, sample_begin + samples_per_pixel):
        red += sample_radiance[
            unsafe_offset=wavefront_plane_index(
                WaveSampleFloatAbi.R, sample_capacity, sample_idx
            )
        ]
        green += sample_radiance[
            unsafe_offset=wavefront_plane_index(
                WaveSampleFloatAbi.G, sample_capacity, sample_idx
            )
        ]
        blue += sample_radiance[
            unsafe_offset=wavefront_plane_index(
                WaveSampleFloatAbi.B, sample_capacity, sample_idx
            )
        ]
    var scale = Float32(1.0) / Float32(samples_per_pixel)
    var output_idx = pixel_base + pixel_idx
    pixels[unsafe_offset=3 * output_idx + 0] = red * scale
    pixels[unsafe_offset=3 * output_idx + 1] = green * scale
    pixels[unsafe_offset=3 * output_idx + 2] = blue * scale
