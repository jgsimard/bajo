from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import json
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


@dataclass(frozen=True)
class TraversalProfile:
    implementation: str
    label: str
    node_visits: int
    leaf_groups: int
    primitive_tests: int
    max_stack: int


@dataclass(frozen=True)
class BuildProfile:
    implementation: str
    label: str
    morton_ns: int
    sort_ns: int
    topology_ns: int
    refit_ns: int
    collapse_ns: int
    bounds_pack_ns: int
    leaf_pack_ns: int
    total_ns: int


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


def parse_traversal_profiles(output: str) -> list[TraversalProfile]:
    profiles: list[TraversalProfile] = []
    for line in output.splitlines():
        if not line.startswith("PROFILE_TRAVERSAL\t"):
            continue
        fields = line.split("\t")
        if len(fields) != 7:
            raise RuntimeError(f"Malformed traversal profile: {line!r}")
        profiles.append(
            TraversalProfile(
                implementation=fields[1],
                label=fields[2],
                node_visits=int(fields[3]),
                leaf_groups=int(fields[4]),
                primitive_tests=int(fields[5]),
                max_stack=int(fields[6]),
            )
        )
    return profiles


def parse_build_profiles(output: str) -> list[BuildProfile]:
    profiles: list[BuildProfile] = []
    for line in output.splitlines():
        if not line.startswith("PROFILE_BUILD_STAGES\t"):
            continue
        fields = line.split("\t")
        if len(fields) != 11:
            raise RuntimeError(f"Malformed build profile: {line!r}")
        profiles.append(
            BuildProfile(
                implementation=fields[1],
                label=fields[2],
                morton_ns=int(fields[3]),
                sort_ns=int(fields[4]),
                topology_ns=int(fields[5]),
                refit_ns=int(fields[6]),
                collapse_ns=int(fields[7]),
                bounds_pack_ns=int(fields[8]),
                leaf_pack_ns=int(fields[9]),
                total_ns=int(fields[10]),
            )
        )
    return profiles


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


def parse_rebuild_timings(output: str) -> dict[str, tuple[float, float, float]]:
    timings = {}
    for line in output.splitlines():
        if not line.startswith("REBUILD_TIMING\t"):
            continue
        fields = line.split("\t")
        if len(fields) != 6:
            raise RuntimeError(f"Malformed rebuild timing: {line!r}")
        timings[fields[2]] = (float(fields[3]), float(fields[4]), float(fields[5]))
    return timings


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def write_report(
    report_path: Path,
    bajo_results: list[Result],
    nexus: Result,
    validations: list[Validation],
    traversal_profiles: list[TraversalProfile],
    build_profiles: list[BuildProfile],
    nexusbvh_dir: Path,
    rebuild_timings: dict[str, tuple[float, float, float]] | None = None,
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

    details_path = report_path.with_suffix(".json")
    details = {
        "generated": generated,
        "gpu": gpu,
        "mojo": mojo_version,
        "nexus_revision": nexus_revision,
        "nexus_checkout": str(nexusbvh_dir),
        "repeats": BENCH_REPEATS,
        "results": [asdict(item) for item in all_results],
        "validations": [asdict(item) for item in validations],
        "traversal_profiles": [asdict(item) for item in traversal_profiles],
        "build_profiles": [asdict(item) for item in build_profiles],
        "rebuild_timings": rebuild_timings or {},
    }
    lines = [
        "# NexusBVH vs Bajo GPU BVH benchmark",
        "",
        f"{generated} · {gpu} · `{mojo_version}` · Nexus `{nexus_revision}`.",
        "",
        f"Dragon: {nexus.triangles:,} triangles, {nexus.rays:,} camera closest-hit rays. "
        f"Median of {BENCH_REPEATS} synchronized runs.",
        "",
        f"Fastest Bajo build: `{fastest_bajo_build.label}` "
        f"({fastest_bajo_build.build_median_ms:.3f} ms). "
        f"Fastest traversal: `{fastest_bajo_trace.label}` "
        f"({fastest_bajo_trace.trace_median_ms:.3f} ms).",
        "",
        "| Configuration | Build ms | Trace ms | MRay/s |",
        "|---|---:|---:|---:|",
    ]
    for result in sorted(all_results, key=lambda item: item.trace_median_ms):
        lines.append(
            f"| `{result.label}` | {result.build_median_ms:.3f} | "
            f"{result.trace_median_ms:.3f} | {result.mrays_per_second:.1f} |"
        )
    lines.extend([
        "",
        "Fresh builds include allocation, bounds, topology, conversion and "
        "synchronization; parsing, camera setup and initial upload are excluded. "
        "Traversal includes launch and synchronization. Cached rebuilds and "
        "instrumented stage timings are separate in the raw data.",
        "",
        f"Validated {len(validations)} Bajo configurations. "
        "Tolerance: one silhouette-edge hit and mean-distance delta ≤ 0.01. "
        f"Maximum observed deltas: {max((v.hit_delta for v in validations), default=0)} "
        "hits, "
        f"{max((v.mean_distance_delta for v in validations), default=0):.6g} distance.",
        "",
        f"Full results, ranges, profiles and validation: [{details_path.name}]"
        f"({details_path.name}).",
        "",
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
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
        ["mojo", "run", "-I", ".", "bench/bvh/bench_gpu_nexus_bajo.mojo"],
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
    traversal_profiles = [
        *parse_traversal_profiles(nexus_output),
        *parse_traversal_profiles(bajo_output),
    ]
    build_profiles = parse_build_profiles(bajo_output)
    if len(nexus_results) != 1:
        raise RuntimeError(
            f"Expected one NexusBVH result, got {len(nexus_results)}"
        )
    nexus = nexus_results[0]
    validations = validate_results(bajo_results, nexus)
    write_report(
        report_path,
        bajo_results,
        nexus,
        validations,
        traversal_profiles,
        build_profiles,
        nexusbvh_dir,
        parse_rebuild_timings(bajo_output),
    )

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
