"""Compatibility façade for the CPU ray-tracing implementation."""


from .bsdf import (
    bsdf_is_delta,
    evaluate_bsdf,
    pdf_bsdf,
    reflect,
    reflectance,
    refract,
    sample_bsdf,
    sample_dielectric,
    sample_lambertian,
    sample_metal,
)
from .common import (
    _path_stage_rng,
    _russian_roulette,
)
from .depth_first import (
    CPU_RENDER_TILE_HEIGHT,
    CPU_RENDER_TILE_WIDTH,
    color_to_byte,
    linear_to_gamma,
    render_depth_first,
    write_ppm_from_colors,
)
from .lighting import (
    emitted_radiance,
    emissive_triangle_area,
    light_pdf_for_emissive_hit,
    power_heuristic,
    sample_direct_lighting,
    sample_direct_lighting_mis,
)
from .wavefront import (
    CPU_WAVEFRONT_PARALLEL_CHUNK_PATHS,
    CPU_WAVEFRONT_SERIAL_CHUNK_PATHS,
    WAVE_PARALLEL_LOGICAL_CORES,
    WAVE_PARALLEL_RUNTIME_DEFAULT,
    WAVE_PARALLEL_TASK_PARTITIONS,
    render_wavefront,
)
from .wavefront.primary import _make_initial_path_packets_range
