"""Layered BSDF throughput for dense, half-active and one-active-lane packets.

Build with `mojo build -O3 -I . bench/rt/bench_cpu_coated_simd.mojo`.
Reports seven-trial medians after warmup; setup is outside the timed region.
Checksums retain the computed values and permit numerical A/B comparisons.
Materials: 0 = coated diffuse, 1 = coated conductor, 2 = scattering coat,
3 = mixed roughness, scattering, depths and estimator sample counts.
"""

from std.benchmark import keep
from std.math import floor
from std.time import perf_counter_ns
from bajo.core import Vec3
from bajo.rt.material_parameters import CoatedParameters
from bajo.rt.coated_diffuse import (
    _evaluate_coated_diffuse,
    _sample_coated_diffuse,
)


def _bench[width: SIMDLength, evaluate: Bool](kind: Int, mask: Int):
    comptime V = Vec3[.float32, .WORLD, width]
    var ray = V(0.1, -0.2, -1.0)
    var normal = V(0.0, 0.0, 1.0)
    var direction = V(-0.3, 0.1, 1.0)
    var albedo = V(0.7, 0.5, 0.25)
    var p = CoatedParameters[width]()
    p.roughness = 0.2
    var active = SIMD[.bool, width](fill=True)
    var seeds = SIMD[.uint32, width](0)
    var u_base = SIMD[.float32, width](0.0)
    var v_base = u_base
    var active_count = 0
    for lane in range(Int(width)):
        active[lane] = mask == 0 or (lane % 2 == 0 if mask == 1 else lane == 0)
        active_count += Int(active[lane])
        seeds[lane] = UInt32(lane) * 0x9E3779B9 + 73
        u_base[lane] = Float32(lane + 1) / Float32(Int(width) + 1)
        v_base[lane] = Float32((lane * 7 + 3) % 19) / 19.0
        if kind == 1:
            p.conductor_eta.x[lane] = 1.2
            p.conductor_eta.y[lane] = 0.95
            p.conductor_eta.z[lane] = 0.65
            p.conductor_k.x[lane] = 7.3
            p.conductor_k.y[lane] = 6.7
            p.conductor_k.z[lane] = 5.6
            p.conductor_roughness[lane] = 0.2
        elif kind == 2:
            p.layer_albedo.x[lane] = 0.7
            p.layer_albedo.y[lane] = 0.4
            p.layer_albedo.z[lane] = 0.2
            p.thickness[lane] = 0.4
            p.g[lane] = 0.5
        elif kind == 3:
            p.max_depth[lane] = Float32(1 + lane % 19)
            p.n_samples[lane] = Float32(1 + lane % 4)
            p.roughness[lane] = 0.0 if lane % 3 == 0 else 0.2
            p.layer_albedo.x[lane] = 0.7 if lane % 2 == 0 else 0.0
            p.thickness[lane] = 0.4
    var times = List[Float64](capacity=7)
    var checksum = Float64(0.0)
    for trial in range(8):
        var sum = SIMD[.float32, width](0.0)
        var start = perf_counter_ns()
        for packet in range(256):
            var u = u_base + Float32(packet % 61) / 61.0
            var v = v_base + Float32(packet % 71) / 71.0
            u -= floor(u)
            v -= floor(v)
            var seed = seeds + UInt32(packet) * 0x85EBCA6B
            comptime if evaluate:
                var result = _evaluate_coated_diffuse(
                    ray,
                    normal,
                    albedo,
                    p,
                    direction,
                    active=active,
                    estimator_seed=seed,
                )
                sum += (
                    result.value.x
                    + result.value.y
                    + result.value.z
                    + result.pdf
                )
            else:
                var result = _sample_coated_diffuse(
                    ray,
                    normal,
                    albedo,
                    p,
                    u,
                    v,
                    active=active,
                    estimator_seed=seed,
                )
                sum += (
                    result.weight.x
                    + result.weight.y
                    + result.weight.z
                    + result.pdf
                )
        var elapsed = Float64(perf_counter_ns() - start) / Float64(
            256 * active_count
        )
        keep(sum)
        checksum = Float64(sum.reduce_add())
        if trial > 0:
            times.append(elapsed)
    sort(times)
    print(
        Int(width),
        kind,
        active_count,
        "evaluate" if evaluate else "sample",
        times[3],
        checksum,
        sep=",",
    )


def main():
    print(
        "width,material,active_lanes,operation,median_ns_per_active_lane,checksum"
    )
    comptime for width in [4, 8, 16]:
        for kind in range(4):
            for mask in range(3):
                _bench[width, True](kind, mask)
                _bench[width, False](kind, mask)
