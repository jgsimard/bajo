from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import math
import os
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NEXUSBVH_DIR = ROOT / "external/nexusbvh"
DEFAULT_BUILD_DIR = Path("/tmp/bajo_nexusbvh_bench")
DEFAULT_REPORT = ROOT / "bench/results/bvh_gpu_nexus/comparison.md"
BENCH_REPEATS = 11


@dataclass(frozen=True)
class Result:
    implementation: str
    label: str
    builder: str
    layout: str
    node_width: int
    leaf_width: int
    max_leaf_size: int
    triangles: int
    build_median_ms: float
    build_minimum_ms: float
    build_maximum_ms: float
    rays: int
    trace_median_ms: float
    trace_minimum_ms: float
    trace_maximum_ms: float
    hits: int
    checksum: float

    @property
    def mrays_per_second(self) -> float:
        return self.rays / (self.trace_median_ms * 1000.0)

    @property
    def mean_hit_distance(self) -> float:
        return self.checksum / self.hits


@dataclass(frozen=True)
class Validation:
    label: str
    hit_delta: int
    mean_distance_delta: float


def run(command: list[str], *, capture: bool = False) -> str:
    process = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    output = process.stdout or ""
    if capture:
        print(output, end="" if output.endswith("\n") else "\n")
    process.check_returncode()
    return output


def probe(command: list[str], *, cwd: Path = ROOT) -> str:
    try:
        output = subprocess.check_output(
            command,
            cwd=cwd,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return " / ".join(line.strip() for line in output.splitlines() if line.strip())


def parse_results(output: str) -> list[Result]:
    results: list[Result] = []
    for line in output.splitlines():
        if not line.startswith("RESULT\t"):
            continue
        fields = line.split("\t")
        if len(fields) != 18:
            raise RuntimeError(f"Malformed benchmark result: {line!r}")
        results.append(
            Result(
                implementation=fields[1],
                label=fields[2],
                builder=fields[3],
                layout=fields[4],
                node_width=int(fields[5]),
                leaf_width=int(fields[6]),
                max_leaf_size=int(fields[7]),
                triangles=int(fields[8]),
                build_median_ms=float(fields[9]),
                build_minimum_ms=float(fields[10]),
                build_maximum_ms=float(fields[11]),
                rays=int(fields[12]),
                trace_median_ms=float(fields[13]),
                trace_minimum_ms=float(fields[14]),
                trace_maximum_ms=float(fields[15]),
                hits=int(fields[16]),
                checksum=float(fields[17]),
            )
        )
    if not results:
        raise RuntimeError("Benchmark did not emit a RESULT line")
    return results


def validate_results(
    bajo_results: list[Result], nexus: Result
) -> list[Validation]:
    validations: list[Validation] = []
    for result in bajo_results:
        if result.triangles != nexus.triangles:
            raise RuntimeError(
                f"Input mismatch for {result.label}: Bajo used "
                f"{result.triangles} triangles, NexusBVH used "
                f"{nexus.triangles}"
            )
        if result.rays != nexus.rays:
            raise RuntimeError(
                f"Ray-count mismatch for {result.label}: Bajo traced "
                f"{result.rays}, NexusBVH traced {nexus.rays}"
            )
        hit_delta = abs(result.hits - nexus.hits)
        if hit_delta > 1:
            raise RuntimeError(
                f"Traversal mismatch for {result.label}: Bajo found "
                f"{result.hits} hits, NexusBVH found {nexus.hits}"
            )
        if hit_delta == 0:
            checksum_tolerance = max(0.05, abs(result.checksum) * 1.0e-6)
            if not math.isclose(
                result.checksum,
                nexus.checksum,
                rel_tol=0.0,
                abs_tol=checksum_tolerance,
            ):
                raise RuntimeError(
                    f"Traversal checksum mismatch for {result.label}: "
                    f"Bajo={result.checksum}, NexusBVH={nexus.checksum}, "
                    f"tolerance={checksum_tolerance}"
                )
        mean_distance_delta = abs(
            result.mean_hit_distance - nexus.mean_hit_distance
        )
        if hit_delta != 0 and mean_distance_delta > 0.01:
            raise RuntimeError(
                f"Traversal mean-distance mismatch for {result.label}: "
                f"Bajo={result.mean_hit_distance}, "
                f"NexusBVH={nexus.mean_hit_distance}"
            )
        validations.append(
            Validation(result.label, hit_delta, mean_distance_delta)
        )
    return validations


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def write_report(
    report_path: Path,
    bajo_results: list[Result],
    nexus: Result,
    validations: list[Validation],
    nexusbvh_dir: Path,
) -> None:
    all_results = [nexus, *bajo_results]
    fastest_bajo_build = min(bajo_results, key=lambda item: item.build_median_ms)
    fastest_bajo_trace = min(bajo_results, key=lambda item: item.trace_median_ms)
    gpu = probe(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    )
    mojo_version = probe(["mojo", "--version"])
    nexus_revision = probe(
        ["git", "rev-parse", "--short=12", "HEAD"], cwd=nexusbvh_dir
    )
    generated = datetime.now().astimezone().isoformat(timespec="seconds")

    lines = [
        "# NexusBVH vs Bajo GPU BVH benchmark",
        "",
        f"Generated: `{markdown_escape(generated)}`  ",
        f"GPU: `{markdown_escape(gpu)}`  ",
        f"Mojo: `{markdown_escape(mojo_version)}`  ",
        f"Nexus checkout: `{markdown_escape(str(nexusbvh_dir))}`  ",
        f"Nexus revision: `{markdown_escape(nexus_revision)}`",
        "",
        "## Summary",
        "",
        f"- Scene: Dragon OBJ, {nexus.triangles:,} triangles.",
        f"- Traversal: 1024x576 camera, {nexus.rays:,} closest-hit rays.",
        f"- Timing: median of {BENCH_REPEATS} synchronized runs; ranges "
        "show minimum to maximum.",
        f"- Fastest Bajo build: `{fastest_bajo_build.label}` at "
        f"{fastest_bajo_build.build_median_ms:.3f} ms "
        f"({fastest_bajo_build.build_median_ms / nexus.build_median_ms:.3f}x "
        "Nexus build time).",
        f"- Fastest Bajo traversal: `{fastest_bajo_trace.label}` at "
        f"{fastest_bajo_trace.trace_median_ms:.3f} ms / "
        f"{fastest_bajo_trace.mrays_per_second:.1f} MRay/s "
        f"({fastest_bajo_trace.trace_median_ms / nexus.trace_median_ms:.3f}x "
        "Nexus traversal time).",
        "",
        "## Build results",
        "",
        "| Implementation | Configuration | Builder | Layout | Node width | "
        "Leaf width | Max leaf | Median ms | Min–max ms | Time / Nexus |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in sorted(all_results, key=lambda item: item.build_median_ms):
        lines.append(
            f"| {result.implementation} | `{result.label}` | "
            f"{result.builder} | {result.layout} | {result.node_width} | "
            f"{result.leaf_width} | {result.max_leaf_size} | "
            f"{result.build_median_ms:.3f} | "
            f"{result.build_minimum_ms:.3f}–{result.build_maximum_ms:.3f} | "
            f"{result.build_median_ms / nexus.build_median_ms:.3f}x |"
        )

    lines.extend(
        [
            "",
            "## Traversal results",
            "",
            "| Implementation | Configuration | Builder | Layout | Median ms | "
            "MRay/s | Min–max ms | Time / Nexus | Hits |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for result in sorted(all_results, key=lambda item: item.trace_median_ms):
        lines.append(
            f"| {result.implementation} | `{result.label}` | "
            f"{result.builder} | {result.layout} | "
            f"{result.trace_median_ms:.3f} | "
            f"{result.mrays_per_second:.1f} | "
            f"{result.trace_minimum_ms:.3f}–{result.trace_maximum_ms:.3f} | "
            f"{result.trace_median_ms / nexus.trace_median_ms:.3f}x | "
            f"{result.hits:,} |"
        )

    lines.extend(
        [
            "",
            "## Validation",
            "",
            "Every Bajo row is compared with NexusBVH. A one-hit difference "
            "is accepted for a ray exactly on a silhouette edge; in that case "
            "the mean hit-distance difference must be at most 0.01.",
            "",
            "| Bajo configuration | Hit-count delta | Mean-distance delta |",
            "|---|---:|---:|",
        ]
    )
    for validation in validations:
        lines.append(
            f"| `{validation.label}` | {validation.hit_delta} | "
            f"{validation.mean_distance_delta:.6g} |"
        )

    lines.extend(
        [
            "",
            "## Methodology",
            "",
            "Bajo covers LBVH and H-PLOC ordinary-wide combinations, an "
            "H-PLOC 8/1/1 row matching NexusBVH's one-triangle leaves, and "
            "LBVH/H-PLOC CWBVH8. Bajo CWBVH8 uses storage leaf width 4 and is "
            "measured with maximum encoded leaf sizes 3 and 1. NexusBVH uses "
            "its H-PLOC CWBVH8 builder and currently stores exactly one "
            "triangle per leaf. Both implementations trace the same generated "
            "camera rays with native packed CWBVH8 or ordinary-wide traversal.",
            "",
            "OBJ parsing, camera setup, and initial host-to-device upload are "
            "outside the timed regions. Build timing includes the complete GPU "
            "build and synchronization. Traversal timing includes kernel launch "
            "and synchronization. Different builder/layout rows do not imply "
            "equivalent hierarchy quality.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    nexusbvh_dir = Path(
        os.environ.get("NEXUSBVH_DIR", DEFAULT_NEXUSBVH_DIR)
    ).resolve()
    build_dir = Path(
        os.environ.get("NEXUSBVH_BUILD_DIR", DEFAULT_BUILD_DIR)
    ).resolve()
    report_path = Path(os.environ.get("NEXUSBVH_REPORT", DEFAULT_REPORT))
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    if not (nexusbvh_dir / "NexusBVH/include/NXB/BVHBuilder.h").is_file():
        raise SystemExit(
            f"NexusBVH not found at {nexusbvh_dir}.\n"
            "Clone https://github.com/StokastX/NexusBVH there, or set "
            "NEXUSBVH_DIR."
        )

    run(
        [
            "cmake",
            "-S",
            str(ROOT / "bench/bvh/nexusbvh"),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DNEXUSBVH_DIR={nexusbvh_dir}",
        ]
    )
    run(["cmake", "--build", str(build_dir), "--parallel"])

    print("\n=== Bajo parameter sweep ===")
    bajo_output = run(
        ["mojo", "-I", ".", "bench/bvh/bench_gpu_nexus_bajo.mojo"],
        capture=True,
    )
    print("\n=== NexusBVH ===")
    nexus_output = run(
        [
            str(build_dir / "bajo_bench_nexusbvh"),
            "assets/dragon/dragon.obj",
        ],
        capture=True,
    )

    bajo_results = parse_results(bajo_output)
    nexus_results = parse_results(nexus_output)
    if len(nexus_results) != 1:
        raise RuntimeError(
            f"Expected one NexusBVH result, got {len(nexus_results)}"
        )
    nexus = nexus_results[0]
    validations = validate_results(bajo_results, nexus)
    write_report(report_path, bajo_results, nexus, validations, nexusbvh_dir)

    print("\n=== Fastest results ===")
    fastest_build = min(bajo_results, key=lambda item: item.build_median_ms)
    fastest_trace = min(bajo_results, key=lambda item: item.trace_median_ms)
    print(
        f"Bajo build: {fastest_build.label} "
        f"{fastest_build.build_median_ms:.3f} ms"
    )
    print(
        f"Bajo traversal: {fastest_trace.label} "
        f"{fastest_trace.trace_median_ms:.3f} ms "
        f"({fastest_trace.mrays_per_second:.1f} MRay/s)"
    )
    print(
        f"NexusBVH: build {nexus.build_median_ms:.3f} ms; traversal "
        f"{nexus.trace_median_ms:.3f} ms "
        f"({nexus.mrays_per_second:.1f} MRay/s)"
    )
    print(f"Validated {len(validations)} Bajo configurations.")
    print(f"Wrote Markdown report: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
