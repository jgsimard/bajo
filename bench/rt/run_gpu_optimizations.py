"""Run rendering A/B cases serially and compare full linear-float images."""

import argparse
import array
import hashlib
import json
import pathlib
import statistics
import subprocess
import time

CASES = {
    "crown_quick": ("assets/crown/crown.pbrt", 16, 350, 490),
    "crown_quick_independent": ("assets/crown/crown.pbrt", 16, 350, 490),
    "crown16": ("assets/crown/crown.pbrt", 16, 0, 0),
    "crown64": ("assets/crown/crown.pbrt", 64, 250, 350),
    "bmw_quick": ("assets/pbrt/bmw-m6/bmw-m6.pbrt", 48, 640, 360),
    "bmw16": ("assets/pbrt/bmw-m6/bmw-m6.pbrt", 16, 320, 180),
    "crown_independent": ("assets/crown/crown.pbrt", 64, 250, 350),
    "crown512": ("assets/crown/crown.pbrt", 512, 0, 0),
}


def compare(reference, candidate):
    before, after = reference.read_bytes(), candidate.read_bytes()
    if len(before) != len(after):
        raise ValueError("image sizes differ")
    a, b = array.array("f"), array.array("f")
    a.frombytes(before)
    b.frombytes(after)
    return {
        "byte_identical": before == after,
        "different_channels": sum(x != y for x, y in zip(a, b)),
        "max_abs_error": max((abs(x - y) for x, y in zip(a, b)), default=0),
        "rmse": (sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)) ** 0.5,
        "reference_sha256": hashlib.sha256(before).hexdigest(),
        "candidate_sha256": hashlib.sha256(after).hexdigest(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("binary")
    p.add_argument("label")
    p.add_argument("--results", default="bench/results/render_optimizations_20261011")
    p.add_argument("--reference", default="baseline")
    p.add_argument(
        "--cases", nargs="+", choices=CASES,
        default=["crown_quick", "crown_quick_independent", "bmw16"],
    )
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()
    if args.repeats < 1:
        p.error("--repeats must be positive")
    root = pathlib.Path(args.results)
    root.mkdir(parents=True, exist_ok=True)
    summary = {"binary": args.binary, "label": args.label, "cases": {}}
    for case in args.cases:
        scene, spp, width, height = CASES[case]
        image = root / f"{args.label}_{case}.f32"
        cmd = [args.binary, scene, str(spp), str(width), str(height),
               str(args.repeats), str(image)]
        if case in ("crown_independent", "crown_quick_independent"):
            cmd.append("0")
        print("RUN", args.label, case, flush=True)
        started = time.time()
        result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        (root / f"{args.label}_{case}.log").write_text(result.stdout)
        if result.returncode:
            print(result.stdout, flush=True)
            raise subprocess.CalledProcessError(result.returncode, cmd)
        timings = [float(line.split()[3]) for line in result.stdout.splitlines()
                   if line.startswith("TRIAL ")]
        record = {"command": cmd, "trials_ms": timings,
                  "median_ms": statistics.median(timings),
                  "process_wall_s": time.time() - started}
        ref_image = root / f"{args.reference}_{case}.f32"
        if args.label != args.reference and ref_image.exists():
            record["image_comparison"] = compare(ref_image, image)
        ref_summary = root / f"{args.reference}.json"
        if args.label != args.reference and ref_summary.exists():
            ref = json.loads(ref_summary.read_text())["cases"].get(case)
            if ref:
                record["speedup"] = ref["median_ms"] / record["median_ms"]
        summary["cases"][case] = record
        (root / f"{args.label}.json").write_text(json.dumps(summary, indent=2)+"\n")
        print(case, record, flush=True)


if __name__ == "__main__":
    main()
