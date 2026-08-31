from __future__ import annotations

import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench.bvh.bench_bvh_cpu_report import (
    merge_build_modes,
    parse_benchmark_output,
)


def _mode_output(mode: str, cpus: int, build_ms: float) -> str:
    return f"""\
=== BVH build threads: {mode}; available CPUs: {cpus}; affinity: 0 ===
PrimitiveKind BoundsBvh benchmark
Primitives: 65536
Rays: 262144
prim split_method width build nodes prims primary MRay_s checksum
tri hploc 16 {build_ms} 273 65536 7.0 37.4 6443188224.0

CPU shared-stack packet BVH benchmark

Regular grid / hploc / BVH16 leaf16
  adaptive-16-8-scalar: 4.25 ms, 61.68 MRay/s, hits=196608, checksum=6443188224.0
  adaptive-16-8-4-scalar: 4.10 ms, 63.94 MRay/s, hits=196608, checksum=6443188224.0
"""


class CpuBvhReportTest(unittest.TestCase):
    def test_current_header_and_adaptive_packet_row(self) -> None:
        output = _mode_output("1", 1, 5.0) + _mode_output("all", 16, 2.0)

        parsed = parse_benchmark_output(output)
        merged = merge_build_modes(parsed)
        adaptive = merged.filter(merged["traversal"] == "adaptive-16-8-scalar")
        adaptive4 = merged.filter(
            merged["traversal"] == "adaptive-16-8-4-scalar"
        )

        self.assertEqual(adaptive.height, 1)
        self.assertEqual(adaptive4.height, 1)
        self.assertEqual(adaptive["ray_width"].item(), 16)
        self.assertEqual(adaptive4["ray_width"].item(), 16)
        self.assertEqual(adaptive["build_ms_1"].item(), 5.0)
        self.assertEqual(adaptive["build_ms_all"].item(), 2.0)

    def test_shuffled_closest_hit_keeps_scalar_baseline(self) -> None:
        output = """\
=== BVH build threads: 1; available CPUs: 1; affinity: 0 ===
CPU shared-stack packet BVH benchmark

Dragon shuffled closest-hit / sah / BVH16 leaf16
  scalar: 14.0 ms, 42.0 MRay/s, hits=71597, checksum=7943562615.175
"""
        parsed = parse_benchmark_output(output)

        self.assertEqual(parsed.height, 1)
        self.assertEqual(parsed["benchmark"].item(), "dragon_shuffled_closest")
        self.assertEqual(parsed["traversal"].item(), "scalar1")
        self.assertEqual(parsed["ray_width"].item(), 1)

    def test_instanced_section_metadata_and_packet_width(self) -> None:
        output = """\
=== BVH build threads: 1; available CPUs: 1; affinity: 0 ===
CPU instanced Dragon BVH benchmark

Instanced Dragon closest-hit benchmark
Triangles: 249882
Instances: 108
Rays: 147456
split_method bounds_width leaf_width traversal build_ms trace_ms MRay_s hits checksum
sah 16 1 packet16 22.0 2.0 73.0 8250 1022292856.0
"""
        parsed = parse_benchmark_output(output)

        self.assertEqual(parsed["benchmark"].item(), "dragon_instances_closest")
        self.assertEqual(parsed["instance_count"].item(), 108)
        self.assertEqual(parsed["ray_width"].item(), 16)

    def test_instanced_triangle_and_flattened_controls(self) -> None:
        output = """\
=== BVH build threads: 1; available CPUs: 1; affinity: 0 ===
CPU instanced closest-hit diagnostic benchmark

Instanced triangle closest-hit benchmark
Triangles: 1
Instances: 108
Rays: 147456
split_method bounds_width leaf_width traversal build_ms trace_ms MRay_s hits checksum
sah 16 1 packet8 0.02 2.1 70.0 20416 1177505.9

Flattened triangle grid closest-hit benchmark
Triangles: 108
Instances: 1
Rays: 147456
split_method bounds_width leaf_width traversal build_ms trace_ms MRay_s hits checksum
sah 16 16 packet16 0.03 1.0 147.0 20416 2281401.9
"""
        parsed = parse_benchmark_output(output)

        controls = {
            row["benchmark"]: (row["instance_count"], row["ray_width"])
            for row in parsed.to_dicts()
        }
        self.assertEqual(
            controls,
            {
                "triangle_instances_closest": (108, 8),
                "triangle_grid_closest": (1, 16),
            },
        )


if __name__ == "__main__":
    unittest.main()
