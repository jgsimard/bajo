"""Register material parameters used by scalar GPU and SIMD CPU shading."""

from bajo.core import Vec3
from bajo.rt.material_types import CoatedDiffuse


@fieldwise_init
struct CoatedParameters[length: SIMDLength = 1](TrivialRegisterPassable):
    var roughness: SIMD[.float32, Self.length]
    var eta: SIMD[.float32, Self.length]
    var thickness: SIMD[.float32, Self.length]
    var layer_albedo: Vec3[.float32, .WORLD, Self.length]
    var g: SIMD[.float32, Self.length]
    var max_depth: SIMD[.float32, Self.length]
    var n_samples: SIMD[.float32, Self.length]
    var conductor_eta: Vec3[.float32, .WORLD, Self.length]
    var conductor_k: Vec3[.float32, .WORLD, Self.length]
    var conductor_roughness: SIMD[.float32, Self.length]

    def __init__(out self):
        self.roughness = 1.0
        self.eta = 1.5
        self.thickness = 0.01
        self.layer_albedo = Vec3[.float32, .WORLD, Self.length](0.0)
        self.g = 0.0
        self.max_depth = 10.0
        self.n_samples = 1.0
        self.conductor_eta = Vec3[.float32, .WORLD, Self.length](0.0)
        self.conductor_k = Vec3[.float32, .WORLD, Self.length](0.0)
        self.conductor_roughness = 0.0

    @staticmethod
    def from_material(material: CoatedDiffuse) -> CoatedParameters[1]:
        """Project authoring constants for scalar CPU shading."""
        return CoatedParameters[1](
            material.roughness,
            material.eta,
            material.thickness,
            material.layer_albedo,
            material.g,
            Float32(material.max_depth),
            Float32(material.n_samples),
            material.conductor_eta,
            material.conductor_k,
            material.conductor_roughness,
        )

    @always_inline
    def set_lane(mut self, lane: Int, value: CoatedParameters[1]):
        self.roughness[lane] = value.roughness
        self.eta[lane] = value.eta
        self.thickness[lane] = value.thickness
        self.layer_albedo.x[lane] = value.layer_albedo.x
        self.layer_albedo.y[lane] = value.layer_albedo.y
        self.layer_albedo.z[lane] = value.layer_albedo.z
        self.g[lane] = value.g
        self.max_depth[lane] = value.max_depth
        self.n_samples[lane] = value.n_samples
        self.conductor_eta.x[lane] = value.conductor_eta.x
        self.conductor_eta.y[lane] = value.conductor_eta.y
        self.conductor_eta.z[lane] = value.conductor_eta.z
        self.conductor_k.x[lane] = value.conductor_k.x
        self.conductor_k.y[lane] = value.conductor_k.y
        self.conductor_k.z[lane] = value.conductor_k.z
        self.conductor_roughness[lane] = value.conductor_roughness
