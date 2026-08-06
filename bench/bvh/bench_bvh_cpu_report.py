from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os
import platform
import re
import subprocess
import sys

import polars as pl


ROOT = Path(__file__).resolve().parents[2]
BENCH_SCRIPT = ROOT / "bench/bvh/bench_bvh_cpu_compare.sh"
RESULTS_DIR = ROOT / "bench/results/bvh_cpu"

ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def run_command(*args: str) -> str:
    try:
        return subprocess.check_output(
            args,
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def run_benchmarks() -> str:
    process = subprocess.Popen(
        ["bash", str(BENCH_SCRIPT)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if process.stdout is None:
        raise RuntimeError("Could not capture benchmark output")

    lines: list[str] = []

    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)

    return_code = process.wait()
    output = ANSI_ESCAPE.sub("", "".join(lines))

    if return_code != 0:
        raise RuntimeError(
            f"Benchmark command failed with exit code {return_code}"
        )

    return output


def width_from_layout(layout: str) -> int:
    match = re.search(r"(\d+)$", layout)

    if match is None:
        raise ValueError(f"Cannot extract SIMD width from {layout!r}")

    return int(match.group(1))


def parse_benchmark_output(output: str) -> pl.DataFrame:
    rows: list[dict[str, object]] = []

    implementation: str | None = None
    version: str | None = None
    benchmark: str | None = None
    primitive_count: int | None = None
    ray_count: int | None = None
    table_kind: str | None = None

    for raw_line in output.splitlines():
        line = raw_line.strip()

        if not line:
            table_kind = None
            continue

        # Implementation and benchmark sections.
        if line == "Primitive BoundsBvh benchmark":
            implementation = "bajo"
            version = None
            benchmark = "grid"
            primitive_count = None
            ray_count = None
            table_kind = None
            continue

        if (
            line.startswith("Embree ")
            and line.endswith(" CPU triangle benchmark")
        ):
            implementation = "embree"
            version = line.removesuffix(" CPU triangle benchmark")
            benchmark = None
            table_kind = None
            continue

        if (
            line.startswith("TinyBVH ")
            and line.endswith(" CPU triangle benchmark")
        ):
            implementation = "tinybvh"
            version = line.removesuffix(" CPU triangle benchmark")
            benchmark = None
            table_kind = None
            continue

        if line == "Regular-grid microbenchmark":
            benchmark = "grid"
            primitive_count = None
            ray_count = None
            table_kind = None
            continue

        if line == "Representative Dragon camera-ray benchmark":
            # The first implementation is Bajo and has no implementation title.
            if implementation is None:
                implementation = "bajo"

            benchmark = "dragon"
            primitive_count = None
            ray_count = None
            table_kind = None
            continue

        # Benchmark metadata.
        if line.startswith("Primitives:"):
            primitive_count = int(line.split(":", 1)[1].strip())
            continue

        if line.startswith("Triangles:"):
            primitive_count = int(line.split(":", 1)[1].strip())
            continue

        if line.startswith("Rays:"):
            ray_count = int(line.split(":", 1)[1].strip())
            continue

        # Table headers.
        if line.startswith("prim split_method width"):
            table_kind = "bajo_grid"
            continue

        if line.startswith("split_method") and "build_ms" in line:
            table_kind = "bajo_dragon"
            continue

        if line.startswith("quality") and "traversal" in line:
            table_kind = "embree"
            continue

        if line.startswith("quality") and "layout" in line:
            table_kind = "tinybvh"
            continue

        if line.startswith("-"):
            continue

        if table_kind is None:
            continue

        values = line.split()

        base: dict[str, object] = {
            "benchmark": benchmark,
            "implementation": implementation,
            "version": version,
            "primitive_count": primitive_count,
            "ray_count": ray_count,
            "nodes": None,
        }

        try:
            if table_kind == "bajo_grid":
                if len(values) != 9:
                    raise ValueError(
                        f"expected 9 columns, received {len(values)}"
                    )

                width = int(values[2])

                row = base | {
                    "build_method": values[1],
                    "layout": f"bvh{width}",
                    "width": width,
                    "build_ms": float(values[3]),
                    "nodes": int(values[4]),
                    "trace_ms": float(values[6]),
                    "mrays_s": float(values[7]),
                    "hits": None,
                    "checksum": float(values[8]),
                }

            elif table_kind == "bajo_dragon":
                if len(values) != 7:
                    raise ValueError(
                        f"expected 7 columns, received {len(values)}"
                    )

                width = int(values[1])

                row = base | {
                    "build_method": values[0],
                    "layout": f"bvh{width}",
                    "width": width,
                    "build_ms": float(values[2]),
                    "trace_ms": float(values[3]),
                    "mrays_s": float(values[4]),
                    "hits": int(values[5]),
                    "checksum": float(values[6]),
                }

            elif table_kind in ("embree", "tinybvh"):
                if len(values) != 7:
                    raise ValueError(
                        f"expected 7 columns, received {len(values)}"
                    )

                row = base | {
                    "build_method": values[0],
                    "layout": values[1],
                    "width": width_from_layout(values[1]),
                    "build_ms": float(values[2]),
                    "trace_ms": float(values[3]),
                    "mrays_s": float(values[4]),
                    "hits": int(values[5]),
                    "checksum": float(values[6]),
                }

            else:
                raise AssertionError(f"Unknown table kind: {table_kind}")

        except (ValueError, IndexError) as error:
            raise ValueError(
                f"Could not parse {table_kind} row:\n{raw_line}"
            ) from error

        rows.append(row)

    if not rows:
        raise ValueError("No benchmark result rows were found")

    return pl.DataFrame(
        rows,
        infer_schema_length=None,
    ).sort(
        [
            "benchmark",
            "implementation",
            "build_method",
            "width",
        ]
    )


def format_markdown_value(value: object) -> str:
    if value is None:
        return ""

    if isinstance(value, float):
        return f"{value:.3f}"

    return str(value).replace("|", r"\|")


def markdown_table(
    frame: pl.DataFrame,
    columns: list[str],
    labels: dict[str, str],
) -> str:
    selected = frame.select(columns)

    headers = [labels.get(column, column) for column in columns]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]

    for row in selected.iter_rows():
        cells = [format_markdown_value(value) for value in row]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def best_results(frame: pl.DataFrame) -> pl.DataFrame:
    selections: list[pl.DataFrame] = []

    benchmarks = frame.get_column("benchmark").unique().sort().to_list()

    for benchmark in benchmarks:
        benchmark_frame = frame.filter(
            pl.col("benchmark") == benchmark
        )

        implementations = (
            benchmark_frame
            .get_column("implementation")
            .unique()
            .sort()
            .to_list()
        )

        for implementation in implementations:
            best = (
                benchmark_frame
                .filter(pl.col("implementation") == implementation)
                .sort("mrays_s", descending=True)
                .head(1)
            )
            selections.append(best)

    best = pl.concat(selections)

    bajo_reference = (
        best
        .filter(pl.col("implementation") == "bajo")
        .select(
            "benchmark",
            pl.col("mrays_s").alias("bajo_mrays_s"),
        )
    )

    return (
        best
        .join(
            bajo_reference,
            on="benchmark",
            how="left",
            validate="m:1",
        )
        .with_columns(
            (
                (pl.col("mrays_s") / pl.col("bajo_mrays_s") - 1.0)
                * 100.0
            ).alias("vs_bajo_pct")
        )
        .drop("bajo_mrays_s")
        .sort(
            ["benchmark", "mrays_s"],
            descending=[False, True],
        )
    )


def cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")

    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()

    return platform.processor() or "unknown"


def update_latest_link(path: Path, link_name: str) -> None:
    link = path.parent / link_name

    if link.exists() or link.is_symlink():
        link.unlink()

    link.symlink_to(path.name)


def generate_report(
    frame: pl.DataFrame,
    timestamp: datetime,
    raw_filename: str,
    csv_filename: str,
) -> str:
    labels = {
        "benchmark": "Benchmark",
        "implementation": "Implementation",
        "build_method": "Build",
        "layout": "Layout",
        "width": "Width",
        "build_ms": "Build ms",
        "trace_ms": "Trace ms",
        "mrays_s": "MRay/s",
        "vs_bajo_pct": "vs Bajo (%)",
        "hits": "Hits",
        "nodes": "Nodes",
        "checksum": "Checksum",
    }

    result_columns = [
        "implementation",
        "build_method",
        "layout",
        "width",
        "build_ms",
        "trace_ms",
        "mrays_s",
        "hits",
        "nodes",
        "checksum",
    ]

    best_columns = [
        "benchmark",
        "implementation",
        "build_method",
        "layout",
        "build_ms",
        "trace_ms",
        "mrays_s",
        "vs_bajo_pct",
    ]

    mojo_version = run_command("mojo", "--version")

    grid = frame.filter(pl.col("benchmark") == "grid")
    dragon = frame.filter(pl.col("benchmark") == "dragon")
    best = best_results(frame)

    return "\n".join(
        [
            "# CPU BVH benchmark results",
            "",
            f"- **Date:** {timestamp.isoformat(timespec='seconds')}",
            f"- **CPU:** {cpu_name()}",
            f"- **System:** {platform.platform()}",
            f"- **Mojo:** `{mojo_version}`",
            "",
            "## Best traversal result per implementation",
            "",
            markdown_table(best, best_columns, labels),
            "",
            "## Regular-grid microbenchmark",
            "",
            markdown_table(grid, result_columns, labels),
            "",
            "## Dragon camera-ray benchmark",
            "",
            markdown_table(dragon, result_columns, labels),
            "",
        ]
    )


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().astimezone()
    stem = timestamp.strftime("bvh_cpu_%Y-%m-%d_%H-%M-%S")

    raw_path = RESULTS_DIR / f"{stem}.txt"
    csv_path = RESULTS_DIR / f"{stem}.csv"
    markdown_path = RESULTS_DIR / f"bvh_cpu.md"

    try:
        output = run_benchmarks()
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    raw_path.write_text(output, encoding="utf-8")

    try:
        frame = parse_benchmark_output(output)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        print(f"Raw output preserved at {raw_path}", file=sys.stderr)
        return 1

    frame.write_csv(
        csv_path,
        float_precision=3,
    )

    report = generate_report(
        frame,
        timestamp,
        raw_path.name,
        csv_path.name,
    )
    markdown_path.write_text(report, encoding="utf-8")

    update_latest_link(raw_path, "latest.txt")
    update_latest_link(csv_path, "latest.csv")
    update_latest_link(markdown_path, "latest.md")

    print()
    print(f"Raw output: {raw_path}")
    print(f"CSV:        {csv_path}")
    print(f"Markdown:   {markdown_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
