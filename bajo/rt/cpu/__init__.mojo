"""Public CPU ray-tracing API."""


from bajo.rt.shading import reflect, reflectance, refract
from .bsdf import evaluate_bsdf, sample_bsdf
from .depth_first import render_depth_first, write_ppm_from_colors
from .wavefront import render_wavefront
