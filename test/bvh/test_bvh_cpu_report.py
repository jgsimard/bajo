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


if __name__ == "__main__":
    unittest.main()
