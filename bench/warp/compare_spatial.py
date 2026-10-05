"""Matched forward bounds-query conformance and performance against Warp 1.17.

Run with pixi run python bench/warp/compare_spatial.py --warp-path /tmp/bajo_warp_deps.
No VJPs or differentiation are part of this benchmark.
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
import time

import numpy as np  # Import the runner's NumPy before the isolated Warp path.

ROOT = Path(__file__).resolve().parents[2]
ITEMS, QUERIES, REPEATS = 256, 512, 7
KINDS = ("aabb", "ray", "sphere", "capsule")


def run(command):
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(result.stdout)
    return result.stdout


def make_kernel(wp, kind, grouped):
    @wp.kernel
    def candidates(tree: wp.uint64, params: wp.array(dtype=wp.float32),
                   output: wp.array(dtype=wp.float32), item_count: int):
        q = wp.tid()
        root = int(-1)
        if wp.static(grouped):
            root = wp.bvh_get_group_root(tree, q % 4)
        p = wp.vec3(params[q * 7], params[q * 7 + 1], params[q * 7 + 2])
        d = wp.vec3(params[q * 7 + 3], params[q * 7 + 4], params[q * 7 + 5])
        radius = params[q * 7 + 6]
        for item in range(item_count):
            output[q * item_count + item] = float(0.0)
        cutoff = float(1.0)
        if wp.static(kind == 0):
            query = wp.bvh_query_aabb(tree, p - wp.vec3(radius), p + wp.vec3(radius), root)
        elif wp.static(kind == 1):
            query = wp.bvh_query_ray(tree, p, d, root)
            cutoff = float(5.0)
        elif wp.static(kind == 2):
            query = wp.bvh_query_sphere(tree, p, radius, root)
        else:
            query = wp.bvh_query_capsule(tree, p, d, radius, root)
        item = int(0)
        while wp.bvh_query_next(query, item, cutoff):
            output[q * item_count + item] += float(1.0)
    return candidates


def timed(wp, device, operation):
    samples = []
    for repeat in range(REPEATS + 2):
        wp.synchronize_device(device)
        start = time.perf_counter_ns()
        retained = operation()
        wp.synchronize_device(device)
        elapsed = time.perf_counter_ns() - start
        if repeat >= 2:
            samples.append(elapsed)
        del retained
    return samples


def exhaustive_sets(boxes, params, kind, grouped):
    rows = params.reshape(QUERIES, 7)
    p = rows[:, :3].astype(np.float64)[:, None, :]
    radius = rows[:, 6].astype(np.float64)[:, None, None]
    lower, upper = boxes[None, :, :3].astype(np.float64), boxes[None, :, 3:].astype(np.float64)
    if kind == 0:
        # Query box endpoints are constructed in Float32 in both libraries.
        qlo = (rows[:, :3] - rows[:, 6, None]).astype(np.float64)[:, None, :]
        qhi = (rows[:, :3] + rows[:, 6, None]).astype(np.float64)[:, None, :]
        result = ((qlo <= upper) & (qhi >= lower)).all(axis=2)
    elif kind == 2:
        delta = np.maximum(np.maximum(lower - p, p - upper), 0.0)
        result = (delta * delta).sum(axis=2) <= radius[:, :, 0] ** 2
    else:
        if kind == 3:
            lower, upper = lower - radius, upper + radius
        direction = rows[:, 3:6].astype(np.float64)[:, None, :]
        t0, t1 = (lower - p) / direction, (upper - p) / direction
        entry = np.maximum(np.minimum(t0, t1).max(axis=2), 0.0)
        exit = np.minimum(np.maximum(t0, t1).min(axis=2), 5.0 if kind == 1 else 1.0)
        result = entry <= exit
    if grouped:
        result &= (np.arange(ITEMS)[None, :] % 4 == np.arange(QUERIES)[:, None] % 4)
    return result


def refresh_matrix(data):
    """Refresh only measured bounds rows; preserve pending capability rows."""
    path = ROOT / "bajo/bvh/PERFORMANCE.md"
    mapping = {
        "General bounds BVH construction": "build",
        "Bounds AABB overlap iterator": "query-0-0",
        "Bounds ray candidate iterator": "query-1-0",
        "Bounds sphere overlap iterator": "query-2-0",
        "Bounds capsule sweep iterator": "query-3-0",
        "Bounds refit": "update-0",
        "Retained-buffer bounds rebuild": "update-1",
    }
    def ratio(phase, device):
        times = {r["implementation"]: statistics.median(r["ns"]) for r in data["samples"]
                 if r["phase"] == phase and r["device"] == device}
        return times["warp"] / times["bajo"]
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        parts = line.split("|")
        if len(parts) < 6:
            continue
        label = parts[1].strip()
        if label in mapping:
            for column, device in ((2, "cpu"), (3, "cuda")):
                parts[column] = f" {ratio(mapping[label], device):.2f} "
        elif label == "Group-restricted bounds queries":
            for column, device in ((2, "cpu"), (3, "cuda")):
                ratios = [ratio(f"query-{kind}-1", device) for kind in range(4)]
                parts[column] = f" {min(ratios):.2f}–{max(ratios):.2f} "
        else:
            continue
        lines[i] = "|".join(parts)
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warp-path", type=Path)
    parser.add_argument("--gpu-arch", default="sm_120")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/bajo_warp_spatial"))
    args = parser.parse_args()
    if args.warp_path:
        sys.path.insert(0, str(args.warp_path))
    import warp as wp
    wp.config.kernel_cache_dir = "/tmp/bajo_warp_spatial_cache"
    wp.init()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    executable = work / "spatial"
    run(["mojo", "build", "-I", str(ROOT), "--target-accelerator", args.gpu_arch,
         str(ROOT / "bench/warp/bench_spatial.mojo"), "-o", str(executable)])
    prefix = work / "fixture"
    run([str(executable), "fixtures", str(prefix)])
    boxes = np.fromfile(str(prefix) + ".boxes", dtype="<f4").reshape(ITEMS, 6)
    params = np.fromfile(str(prefix) + ".queries", dtype="<f4")
    groups = np.arange(ITEMS, dtype=np.int32) % 4
    data = dict(date=datetime.now(timezone.utc).isoformat(), warp_version=wp.__version__,
                items=ITEMS, queries=QUERIES, groups=4, repeats=REPEATS,
                compiler=run(["mojo", "--version"]).strip(),
                cpu=run(["lscpu"]).strip(),
                gpu=run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]).strip(),
                samples=[], validation=[], fixture_hashes={}, source_hashes={})
    for suffix in ("boxes", "queries"):
        path = Path(str(prefix) + "." + suffix)
        data["fixture_hashes"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for device in ("cpu", "cuda"):
        print("Bajo", device, flush=True)
        log = run([str(executable), device, str(prefix)])
        (work / f"bajo-{device}.log").write_text(log)
        bajo_rows = {}
        for line in log.splitlines():
            if line.startswith("SAMPLE\t"):
                _, backend, phase, value = line.split("\t")
                bajo_rows.setdefault(phase, []).append(int(value))
        for phase, values in bajo_rows.items():
            data["samples"].append(dict(implementation="bajo", device=device,
                                        phase=phase, ns=values))
        constructor = "sah" if device == "cpu" else "lbvh"
        lowers = wp.array(boxes[:, :3].copy(), dtype=wp.vec3, device=device)
        uppers = wp.array(boxes[:, 3:].copy(), dtype=wp.vec3, device=device)
        tags = wp.array(groups, dtype=wp.int32, device=device)
        if device == "cpu":
            factory = lambda: wp.Bvh(lowers, uppers, constructor=constructor, groups=tags, leaf_size=1)
        else:
            # Both GPU build timings start from host geometry and include all
            # uploads, ownership/scratch allocation and tree construction.
            def factory():
                lo = wp.array(boxes[:, :3].copy(), dtype=wp.vec3, device=device)
                hi = wp.array(boxes[:, 3:].copy(), dtype=wp.vec3, device=device)
                gr = wp.array(groups, dtype=wp.int32, device=device)
                return wp.Bvh(lo, hi, constructor=constructor, groups=gr, leaf_size=1)
        for phase, operation in (("build", factory),):
            data["samples"].append(dict(implementation="warp", device=device, phase=phase,
                                        ns=timed(wp, device, operation)))
        tree = wp.Bvh(lowers, uppers, constructor=constructor, groups=tags, leaf_size=1)
        for phase, operation in (("update-0", tree.refit), ("update-1", tree.rebuild)):
            data["samples"].append(dict(implementation="warp", device=device, phase=phase,
                                        ns=timed(wp, device, operation)))
        query_params = wp.array(params, dtype=wp.float32, device=device)
        output = wp.empty(QUERIES * ITEMS, dtype=wp.float32, device=device)
        for grouped in (False, True):
            for kind in range(4):
                phase = f"query-{kind}-{int(grouped)}"
                kernel = make_kernel(wp, kind, grouped)
                launch = lambda: wp.launch(kernel, dim=QUERIES,
                                            inputs=[tree.id, query_params, output, ITEMS], device=device)
                # Warm/compile before retaining timed samples.
                values = timed(wp, device, launch)
                actual = output.numpy().reshape(QUERIES, ITEMS)
                expected = np.fromfile(f"{prefix}-bajo-{device}-{phase}.sets", dtype="<f4").reshape(QUERIES, ITEMS)
                disagreements = int(np.count_nonzero(actual != expected))
                if disagreements or not np.isin(actual, [0.0, 1.0]).all():
                    raise RuntimeError(f"Candidate mismatch {device}/{phase}: {disagreements}")
                reference = exhaustive_sets(boxes, params, kind, grouped)
                reference_disagreements = int(np.count_nonzero(actual.astype(bool) != reference))
                if reference_disagreements:
                    raise RuntimeError(f"Exhaustive mismatch {device}/{phase}: {reference_disagreements}")
                data["validation"].append(dict(device=device, phase=phase,
                    disagreements=disagreements, exhaustive_disagreements=reference_disagreements, matches=int(actual.sum())))
                data["samples"].append(dict(implementation="warp", device=device, phase=phase, ns=values))
                print(device, KINDS[kind], "groups" if grouped else "global", "validated", flush=True)
    for path in [*(ROOT / "bajo").rglob("*.mojo"),
                 *(ROOT / "bench/warp").glob("*.mojo"), Path(__file__)]:
        data["source_hashes"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    destination = ROOT / "bench/results/warp/spatial.json"
    destination.write_text(json.dumps(data, indent=2) + "\n")
    lines = ["# Spatial bounds: Bajo vs Warp", "", f"{data['date'][:10]} · Warp {wp.__version__} · 256 boxes, four groups, 512 queries, seven warmed samples.", "",
             "Bajo median/preorder vs Warp CPU SAH / GPU LBVH, one item per leaf. Ratios are Warp/Bajo; above 1 favors Bajo.", "",
             "| Operation | CPU ratio | GPU ratio |", "|---|---:|---:|"]
    for phase in bajo_rows:
        ratios = []
        for device in ("cpu", "cuda"):
            times = {r['implementation']: statistics.median(r['ns']) for r in data['samples'] if r['device'] == device and r['phase'] == phase}
            ratios.append(times['warp'] / times['bajo'])
        if phase.startswith("query-"):
            _, kind, grouped = phase.split("-")
            label = KINDS[int(kind)].upper() + (" / group" if int(grouped) else " / global")
        else:
            label = {"build": "Fresh build", "update-0": "Refit", "update-1": "Retained rebuild"}[phase]
        lines.append(f"| {label} | {ratios[0]:.2f} | {ratios[1]:.2f} |")
    lines += ["", "Queries time full per-query candidate matrices, including zeroing/output writes, launch and synchronization; download is excluded. GPU construction starts from host bounds and includes uploads and retained scratch. Updates use unchanged warmed inputs; GPU inputs are already resident. Bajo GPU updates include validation; status download is excluded. CPU/GPU rebuild algorithms differ: CPU median versus SAH, GPU X-order baseline versus LBVH.", "",
              "All 16 query cases have identical complete candidate sets and match an independent exhaustive reference. Scope: one small bounds workload, not a library-wide speed ranking. [Raw samples and hashes](spatial.json).", "",
              "Reproduce: `pixi run python bench/warp/compare_spatial.py --warp-path /tmp/bajo_warp_deps`."]
    destination.with_suffix(".md").write_text("\n".join(lines) + "\n")
    refresh_matrix(data)
    print(destination.with_suffix(".md").read_text())


if __name__ == "__main__":
    main()
