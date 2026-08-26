"""Public CPU ray-tracing API."""


from bajo.rt.shading import reflect, reflectance, refract
from .bsdf import evaluate_bsdf, sample_bsdf
from .depth_first import render_depth_first, write_ppm_from_colors
from .scene import CPU_SCENE_DEFAULT_CONFIG, CpuScene, CpuSceneConfig
from .scheduler_mode import CpuSchedulerMode
from .wavefront import render_wavefront, render_wavefront_configured
