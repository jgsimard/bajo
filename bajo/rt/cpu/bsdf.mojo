"""CPU scalar and packet BSDF setup over shared host/device shading."""

from std.memory import Pointer
from std.sys.info import size_of
from bajo.core import Vec3, Vec3f32, Rayf32, dot
from bajo.core.tensor_layout import record_tile, gather_record_vec3
from bajo.core.random import Rng
from bajo.rt.material_parameters import CoatedParameters
from bajo.rt.material_storage import load_coated_parameters, CoatedDiffuseLayout
from bajo.rt.shading import (
    METAL_DELTA_FUZZ_THRESHOLD,
    dielectric_geometry,
    _evaluate_coated_diffuse,
    _evaluate_material,
    _sample_coated_diffuse,
    _sample_material,
)
from bajo.core.color import Color
from bajo.rt.material_types import (
    MaterialKind,
    Metal,
    Dielectric,
    NO_TEXTURE,
    SurfaceId,
    SurfaceStore,
    SURFACE_INDEX_BITS,
    SURFACE_INDEX_MASK,
)
from bajo.rt.shading_types import BsdfEvaluation, BsdfSample, ShadingPoint


struct _MaterialParameters[length: SIMDLength]:
    var albedo: Vec3[.float32, .WORLD, Self.length]
    # Fuzz/roughness, or the dielectric refraction index.
    var parameter: SIMD[.float32, Self.length]
    var coated: CoatedParameters[Self.length]

    def __init__(out self):
        self.albedo = Vec3[.float32, .WORLD, Self.length](0.0)
        self.parameter = SIMD[.float32, Self.length](1.0)
        self.coated = CoatedParameters[Self.length]()

    @always_inline
    def gather[
        kind: MaterialKind
    ](
        mut self,
        surfaces: SurfaceStore,
        indices: SIMD[.uint32, Self.length],
        active: SIMD[.bool, Self.length],
    ):
        comptime if kind == .METAL or kind == .DIELECTRIC:
            if not active.reduce_or():
                return
            # Borrow the actual field address and use the record's ABI stride;
            # no hard-coded offsets or second prepared copy of the materials.
            comptime if kind == .METAL:
                self._gather_parameter(
                    Pointer(to=surfaces.metals[0].fuzz),
                    size_of[Metal]() // size_of[Float32](),
                    indices,
                    active,
                )
            else:
                self._gather_parameter(
                    Pointer(to=surfaces.dielectrics[0].refraction_index),
                    size_of[Dielectric]() // size_of[Float32](),
                    indices,
                    active,
                )

    @always_inline
    def _gather_parameter[
        origin: ImmOrigin
    ](
        mut self,
        fields: Pointer[Float32, origin],
        stride: Int,
        indices: SIMD[.uint32, Self.length],
        active: SIMD[.bool, Self.length],
    ):
        var offsets = active.select(indices, 0).cast[.int]() * stride
        self.parameter = active.select(fields.unsafe_gather(offsets), 1.0)

    @always_inline
    def load[
        kind: MaterialKind, LOAD_ALBEDO: Bool = True
    ](
        mut self,
        surfaces: SurfaceStore,
        indices: SIMD[.uint32, Self.length],
        u: SIMD[.float32, Self.length],
        v: SIMD[.float32, Self.length],
        active: SIMD[.bool, Self.length],
        coated_fields: ImmSpan[Float32, _] = ImmSpan[Float32, ImmutAnyOrigin](),
    ):
        """Load one material group from authoring records or prepared tables."""
        if not active.reduce_or():
            return
        var surface = SurfaceId[Self.length](
            (SIMD[.uint32, Self.length](kind.value) << SURFACE_INDEX_BITS)
            | indices
        )
        var texture_albedo = True
        var texture_roughness = True
        var albedo = self.albedo
        comptime if kind == .COATED_DIFFUSE:
            if len(coated_fields) > 0:
                var first = 0
                while not active[first]:
                    first += 1
                var ids = active.select(indices, indices[first])
                var uniform = ids.eq(ids[first]).reduce_and()
                var records = record_tile[
                    .float32, CoatedDiffuseLayout.STRIDE, True
                ](coated_fields.unsafe_ptr(), len(surfaces.coated_diffuses))
                self.coated = load_coated_parameters(records, ids, uniform)
                comptime if LOAD_ALBEDO:
                    albedo = gather_record_vec3[
                        .WORLD, CoatedDiffuseLayout.ALBEDO
                    ](records, ids, uniform)
                if uniform:
                    ref source = surfaces.coated_diffuses[Int(ids[first])]
                    texture_albedo = source.texture_index != NO_TEXTURE
                    texture_roughness = (
                        source.roughness_texture_index != NO_TEXTURE
                    )
            else:
                for lane in range(Self.length):
                    if active[lane]:
                        self.coated.set_lane(
                            lane,
                            CoatedParameters[1].from_material(
                                surfaces.coated_diffuses[Int(indices[lane])]
                            ),
                        )
            if texture_roughness:
                self.coated.conductor_roughness = (
                    surfaces.sample_conductor_roughness(surface, u, v, active)
                )
        else:
            self.gather[kind](surfaces, indices, active)
        comptime if LOAD_ALBEDO and kind in [
            MaterialKind.LAMBERTIAN,
            MaterialKind.METAL,
            MaterialKind.COATED_DIFFUSE,
        ]:
            if texture_albedo:
                albedo = surfaces.sample_albedo(surface, u, v, active)
            self.albedo = Vec3.select(active, albedo, self.albedo)

    @always_inline
    def evaluate[
        kind: MaterialKind
    ](
        self,
        direction: Vec3[.float32, .WORLD, Self.length],
        normal: Vec3[.float32, .WORLD, Self.length],
        outgoing: Vec3[.float32, .WORLD, Self.length],
        estimator_seed: SIMD[.uint32, Self.length] = 0,
        active: SIMD[.bool, Self.length] = SIMD[.bool, Self.length](fill=True),
    ) -> BsdfEvaluation[Self.length]:
        comptime if kind == .COATED_DIFFUSE:
            return _evaluate_coated_diffuse(
                direction,
                normal,
                self.albedo,
                self.coated,
                outgoing,
                active=active,
                estimator_seed=estimator_seed,
            )
        else:
            return _evaluate_material[kind, Self.length](
                direction, normal, self.albedo, self.parameter, outgoing
            )

    @always_inline
    def sample[
        kind: MaterialKind
    ](
        self,
        direction: Vec3[.float32, .WORLD, Self.length],
        normal: Vec3[.float32, .WORLD, Self.length],
        front_face: SIMD[.bool, Self.length],
        mut rng: Rng[Self.length],
        estimator_seed: SIMD[.uint32, Self.length] = 0,
        active: SIMD[.bool, Self.length] = SIMD[.bool, Self.length](fill=True),
    ) -> BsdfSample[Self.length]:
        """Consume the same draws for scalar and packet material sampling."""
        comptime assert MaterialKind.has_bsdf[kind]
        var random_u = SIMD[.float32, Self.length](0.0)
        var random_v = SIMD[.float32, Self.length](0.0)
        var eligible = active
        comptime if kind == .DIELECTRIC:
            debug_assert["safe", _use_compiler_assume=True](
                (~active | self.parameter.gt(0.0)).reduce_and()
            )
            var ri = front_face.select(
                Float32(1.0) / self.parameter, self.parameter
            )
            var geometry = dielectric_geometry(direction, normal, ri)
            eligible &= geometry[2].le(1.0)
            if eligible.reduce_or():
                random_u = eligible.select(rng.f32(), 0.0)
        else:
            comptime if kind == .METAL:
                debug_assert["safe", _use_compiler_assume=True](
                    (
                        ~active
                        | (self.parameter.ge(0.0) & self.parameter.le(1.0))
                    ).reduce_and()
                )
                eligible &= self.parameter.gt(METAL_DELTA_FUZZ_THRESHOLD)
            if eligible.reduce_or():
                random_u = eligible.select(rng.f32(), 0.0)
                random_v = eligible.select(rng.f32(), 0.0)
        var sampled: BsdfSample[Self.length]
        comptime if kind == .COATED_DIFFUSE:
            sampled = _sample_coated_diffuse(
                direction,
                normal,
                self.albedo,
                self.coated,
                random_u,
                random_v,
                active=active,
                estimator_seed=estimator_seed,
            )
        else:
            sampled = _sample_material[kind, Self.length](
                direction,
                normal,
                self.albedo,
                self.parameter,
                front_face,
                random_u,
                random_v,
            )
        sampled.ok &= active
        return sampled^


@always_inline
def _evaluate_surface[
    kind: MaterialKind
](
    surface: SurfaceId[1],
    surfaces: SurfaceStore,
    ray: Rayf32[.WORLD],
    hit: ShadingPoint[1],
    outgoing: Vec3f32[.WORLD],
    estimator_seed: UInt32,
) -> BsdfEvaluation[1]:
    var material = _MaterialParameters[1]()
    comptime if kind not in [MaterialKind.DIELECTRIC, MaterialKind.EMISSIVE]:
        material.load[kind](
            surfaces,
            surface.value & SURFACE_INDEX_MASK,
            hit.uv_u,
            hit.uv_v,
            True,
        )
    return material.evaluate[kind](ray.d, hit.normal, outgoing, estimator_seed)


@always_inline
def _sample_surface[
    kind: MaterialKind
](
    surface: SurfaceId[1],
    surfaces: SurfaceStore,
    ray: Rayf32[.WORLD],
    hit: ShadingPoint[1],
    mut rng: Rng[1],
    estimator_seed: UInt32,
) -> BsdfSample[1]:
    var material = _MaterialParameters[1]()
    material.load[kind](
        surfaces, surface.value & SURFACE_INDEX_MASK, hit.uv_u, hit.uv_v, True
    )
    return material.sample[kind](
        ray.d, hit.normal, hit.front_face, rng, estimator_seed
    )


def evaluate_bsdf(
    surface: SurfaceId[1],
    surfaces: SurfaceStore,
    ray: Rayf32[.WORLD],
    hit: ShadingPoint[1],
    out_direction: Vec3f32[.WORLD],
    estimator_seed: UInt32 = 0,
) -> BsdfEvaluation[1]:
    match surface.kind():
        case .LAMBERTIAN:
            return _evaluate_surface[.LAMBERTIAN](
                surface, surfaces, ray, hit, out_direction, estimator_seed
            )
        case .METAL:
            return _evaluate_surface[.METAL](
                surface, surfaces, ray, hit, out_direction, estimator_seed
            )
        case .COATED_DIFFUSE:
            return _evaluate_surface[.COATED_DIFFUSE](
                surface, surfaces, ray, hit, out_direction, estimator_seed
            )
        case .DIELECTRIC:
            return _evaluate_surface[.DIELECTRIC](
                surface, surfaces, ray, hit, out_direction, estimator_seed
            )
        case .EMISSIVE:
            return _evaluate_surface[.EMISSIVE](
                surface, surfaces, ray, hit, out_direction, estimator_seed
            )
        case .MIX:
            var index = Int(surface.index())
            var first = evaluate_bsdf(
                SurfaceId.from_raw(surfaces.mix_surfaces[2 * index]),
                surfaces,
                ray,
                hit,
                out_direction,
                estimator_seed,
            )
            var second = evaluate_bsdf(
                SurfaceId.from_raw(surfaces.mix_surfaces[2 * index + 1]),
                surfaces,
                ray,
                hit,
                out_direction,
                estimator_seed,
            )
            var amount = surfaces.sample_mix_amount(
                index, hit.uv_u[0], hit.uv_v[0]
            )
            return BsdfEvaluation[1](
                first.value * (1.0 - amount) + second.value * amount,
                first.pdf * (1.0 - amount) + second.pdf * amount,
                first.delta & second.delta,
            )


def sample_bsdf(
    surface: SurfaceId[1],
    surfaces: SurfaceStore,
    ray: Rayf32[.WORLD],
    hit: ShadingPoint[1],
    mut rng: Rng[1],
    estimator_seed: UInt32 = 0,
) -> BsdfSample[1]:
    match surface.kind():
        case .LAMBERTIAN:
            return _sample_surface[.LAMBERTIAN](
                surface, surfaces, ray, hit, rng, estimator_seed
            )
        case .METAL:
            return _sample_surface[.METAL](
                surface, surfaces, ray, hit, rng, estimator_seed
            )
        case .COATED_DIFFUSE:
            return _sample_surface[.COATED_DIFFUSE](
                surface, surfaces, ray, hit, rng, estimator_seed
            )
        case .DIELECTRIC:
            return _sample_surface[.DIELECTRIC](
                surface, surfaces, ray, hit, rng, estimator_seed
            )
        case .EMISSIVE:
            return BsdfSample(
                hit.normal,
                Color(0.0),
                0.0,
                False,
                False,
            )
        case .MIX:
            var index = Int(surface.index())
            var amount = surfaces.sample_mix_amount(
                index, hit.uv_u[0], hit.uv_v[0]
            )
            var choose_second = rng.f32() < amount
            var selected = SurfaceId.from_raw(
                surfaces.mix_surfaces[2 * index + (1 if choose_second else 0)]
            )
            var sampled = sample_bsdf(
                selected, surfaces, ray, hit, rng, estimator_seed
            )
            if sampled.ok[0]:
                if sampled.delta[0]:
                    sampled.pdf *= amount if choose_second else (1.0 - amount)
                else:
                    var evaluated = evaluate_bsdf(
                        surface,
                        surfaces,
                        ray,
                        hit,
                        sampled.direction,
                        estimator_seed,
                    )
                    sampled.pdf = evaluated.pdf
                    sampled.weight = evaluated.value * (
                        max(dot(hit.normal, sampled.direction), 0.0)
                        / max(evaluated.pdf, Float32(1.0e-20))
                    )
            return sampled^
