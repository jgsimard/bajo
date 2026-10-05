"""Run matched Bajo/Warp benchmarks and validate every ray and sorted pair.

Run with `pixi run python bench/warp/compare.py --warp-path /tmp/bajo_warp_deps`.
Warp 1.17.0 must be installed in the runner's Python environment or --warp-path.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def run(command, log=None):
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    if log:
        log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"{command}:\n{result.stdout}")
    return result.stdout


def samples(output, scene=None):
    rows = {}
    for line in output.splitlines():
        fields = line.split("\t")
        if fields[0] != "SAMPLE":
            continue
        if scene:
            _, label, phase, ns = fields
            key = label
            row = rows.setdefault(key, dict(implementation="bajo", scene=scene, label=label))
            row.setdefault(phase + "_ns", []).append(int(ns))
        else:
            _, label, size, pattern, ns = fields
            key = (label, int(size), int(pattern))
            row = rows.setdefault(key, dict(implementation="bajo", label=label,
                                           size=int(size), pattern=int(pattern), validated=True))
            row.setdefault("sort_ns", []).append(int(ns))
    if not rows:
        raise RuntimeError("No benchmark samples emitted")
    return list(rows.values())


def ms(row, phase):
    return statistics.median(row[phase + "_ns"]) / 1e6


def validate(meshes, work):
    results = []
    for scene in sorted({row["scene"] for row in meshes}):
        reference_row = next(row for row in meshes if row["scene"] == scene
                             and row["label"] == "cpu-sah-l4" and row["implementation"] == "warp")
        reference = np.fromfile(work / f"{scene}-warp-{reference_row['label']}.hits", dtype="<f4")
        reference_hits = reference < np.finfo(np.float32).max
        for row in (r for r in meshes if r["scene"] == scene):
            label = ("warp-" if row["implementation"] == "warp" else "") + row["label"]
            actual = np.fromfile(work / f"{scene}-{label}.hits", dtype="<f4")
            if len(actual) != len(reference):
                raise RuntimeError(f"Ray count mismatch: {scene}/{label}")
            hit_mask = actual < np.finfo(np.float32).max
            disagreements = int(np.count_nonzero(hit_mask != reference_hits))
            common = hit_mask & reference_hits
            delta = np.abs(actual[common] - reference[common])
            bad_distances = int(np.count_nonzero(delta > 1e-4 + 1e-4 * np.abs(reference[common])))
            if disagreements > 1 or bad_distances or not np.isfinite(actual).all():
                raise RuntimeError(f"Validation failed: {scene}/{label}: "
                                   f"{disagreements} hit differences, {bad_distances} distance differences")
            any_hits = np.fromfile(work / f"{scene}-{label}.any", dtype="<f4")
            if len(any_hits) != len(reference) or not np.isin(any_hits, [0.0, 1.0]).all():
                raise RuntimeError(f"Invalid any-hit output: {scene}/{label}")
            any_disagreements = int(np.count_nonzero(any_hits.astype(bool) != reference_hits))
            if any_disagreements > 1:
                raise RuntimeError(f"Any-hit validation failed: {scene}/{label}")
            results.append(dict(scene=scene, label=label, hits=int(hit_mask.sum()),
                                hit_disagreements=disagreements,
                                any_hit_disagreements=any_disagreements,
                                maximum_distance_delta=float(delta.max(initial=0)), validated=True))
    return results


def report(data, destination):
    cpu_model = next((line.split(":", 1)[1].strip() for line in data["cpu"].splitlines()
                      if line.startswith("Model name:")), "CPU recorded in JSON")
    lines = ["# Bajo vs NVIDIA Warp", "", f"{data['date'][:10]} · {cpu_model} · {data['gpu']} · Warp {data['warp_version']}", "",
             "Identical triangle soup and precomputed 512×288 camera rays; 11 warmed samples per case. "
             "Build includes fresh BVH allocation and bounds/packing; input upload and destruction are excluded. "
             "Queries include launch and synchronization, write distance or an any-hit flag, and exclude download. "
             "Meshes are centered and scaled to a largest extent of 2. CPU builds use library-default threading; "
             "CPU queries use one scalar ray per iteration (Bajo still uses SIMD within each BVH node).", "",
             "Bajo: CPU median/SAH/LBVH/H-PLOC with node16/leaf16; GPU H-PLOC/CWBVH8 with leaf1/3. "
             "Warp: CPU SAH/median/cuBQL; GPU LBVH/SAH/cuBQL, each with leaf1/4. "
             "Fastest measured configuration for each phase; build and query winners may differ.", "",
             "| Scene / backend | Bajo build ms | Warp build ms | Bajo closest ms | Warp closest ms | Bajo any ms | Warp any ms |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for scene in sorted({r["scene"] for r in data["meshes"]}):
        for backend in ("cpu", "cuda"):
            rows = [r for r in data["meshes"] if r["scene"] == scene and r["label"].startswith(backend)]
            bajo = [r for r in rows if r["implementation"] == "bajo"]
            warp = [r for r in rows if r["implementation"] == "warp"]
            lines.append(f"| {scene} / {backend} | {min(ms(r, 'build') for r in bajo):.3f} | "
                         f"{min(ms(r, 'build') for r in warp):.3f} | "
                         f"{min(ms(r, 'trace') for r in bajo):.3f} | {min(ms(r, 'trace') for r in warp):.3f} | "
                         f"{min(ms(r, 'any') for r in bajo):.3f} | {min(ms(r, 'any') for r in warp):.3f} |")
    lines += ["", "GPU stable key/value sorting: identical nonnegative 31-bit keys and 32-bit IDs, "
              "retained scratch, reset copies excluded. Patterns: mixed / duplicates / 24-bit.", "",
              "| Pairs | Pattern | Bajo radix ms | Bajo OneSweep ms | Warp radix ms |",
              "|---:|---|---:|---:|---:|"]
    for size in sorted({r["size"] for r in data["sorts"]}):
        for pattern in range(3):
            rows = [r for r in data["sorts"] if r["size"] == size and r["pattern"] == pattern]
            times = [ms(next(r for r in rows if r["implementation"] == impl and r["label"] == label), "sort")
                     for impl, label in (("bajo", "radix"), ("bajo", "onesweep"), ("warp", "radix"))]
            lines.append(f"| {size:,} | {('mixed', 'duplicates', '24-bit')[pattern]} | "
                         + " | ".join(f"{value:.3f}" for value in times) + " |")
    lines += ["", "Every ray is checked against Warp CPU SAH/leaf4: ≤1 hit-mask disagreement per case "
              "(edge tolerance), distance tolerance `1e-4 + 1e-4 × |t|`. Every sort checks order, "
              "key/value association and stability. "
              f"Observed maximum closest/any-hit mask disagreements: "
              f"{max(max(r['hit_disagreements'], r['any_hit_disagreements']) for r in data['validation'])}.", "",
              "Scope: two static meshes, coherent camera rays, two sort sizes, one machine; no rendering, "
              "instancing, incoherent rays or dynamic updates. Warp uses triangle-soup indices here; "
              "its indexed-mesh memory advantage is not measured. CPU queries are not an all-core throughput comparison.", "",
              "Full configurations, samples, input/source hashes, environment and validation: "
              "[comparison.json](comparison.json). Reproduce: "
              "`pixi run python bench/warp/compare.py --warp-path /tmp/bajo_warp_deps`."]
    destination.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warp-path", type=Path)
    parser.add_argument("--gpu-arch", default="sm_120")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/bajo_warp_comparison"))
    parser.add_argument("--report", type=Path, default=ROOT / "bench/results/warp/comparison.md")
    parser.add_argument("--report-only", action="store_true", help="Regenerate Markdown from existing raw results")
    args = parser.parse_args()
    if args.report_only:
        report(json.loads(args.report.with_suffix(".json").read_text()), args.report)
        return
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    data = dict(date=datetime.now(timezone.utc).isoformat(), meshes=[], sorts=[],
                compiler=run(["mojo", "--version"]).strip(),
                gpu=run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]).strip(),
                cpu=run(["lscpu"]).strip(), source_hashes={}, fixture_hashes={})
    for name in ("bvh", "sort"):
        print("Compiling Bajo", name, flush=True)
        run(["mojo", "build", "-I", str(ROOT), "--target-accelerator", args.gpu_arch,
             str(ROOT / f"bench/warp/bench_{name}.mojo"), "-o", str(work / name)], work / f"compile-{name}.log")
    fixtures = []
    for scene in ("bunny", "dragon"):
        prefix = work / scene
        fixtures.append(prefix)
        mesh = ROOT / f"assets/{scene}/{scene}.obj"
        run([str(work / "bvh"), "fixtures", str(mesh), str(prefix)])
        for suffix in ("vertices", "rays"):
            path = Path(str(prefix) + "." + suffix)
            data["fixture_hashes"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for backend in ("cpu", "cuda"):
            print("Benchmarking Bajo", scene, backend, flush=True)
            output = run([str(work / "bvh"), backend, str(mesh), str(prefix)], work / f"bajo-{scene}-{backend}.log")
            rows = samples(output, scene)
            for row in rows:
                row["triangles"] = Path(str(prefix) + ".vertices").stat().st_size // 36
                row["rays"] = Path(str(prefix) + ".rays").stat().st_size // 24
            data["meshes"].extend(rows)
    print("Benchmarking Bajo sorting", flush=True)
    data["sorts"].extend(samples(run([str(work / "sort")], work / "bajo-sort.log")))
    print("Benchmarking Warp", flush=True)
    command = [sys.executable, str(ROOT / "bench/warp/bench_warp.py"),
               "--output", str(work / "warp.json"), "--fixtures", *map(str, fixtures)]
    if args.warp_path:
        command += ["--warp-path", str(args.warp_path.resolve())]
    run(command, work / "warp.log")
    warp = json.loads((work / "warp.json").read_text())
    data["warp_version"] = warp["warp_version"]
    data["meshes"].extend(warp["meshes"])
    data["sorts"].extend(warp["sorts"])
    data["validation"] = validate(data["meshes"], work)
    for path in [*(ROOT / "bajo").rglob("*.mojo"), *(ROOT / "bench/warp").glob("*.mojo"),
                 *(ROOT / "bench/warp").glob("*.py")]:
        data["source_hashes"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    args.report.with_suffix(".json").write_text(json.dumps(data, indent=2) + "\n")
    report(data, args.report)
    print(args.report.read_text(), flush=True)


if __name__ == "__main__":
    main()
