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
SINGLE_THREAD_MODE = "1"
ALL_THREAD_MODE = "all"

ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PACKET_SECTION = re.compile(
    r"^(Regular grid|Dragon camera rays) / ([^/]+) / BVH(\d+) leaf(\d+)$"
)
PACKET_TIMING = re.compile(
    r"^((?:unmasked-)?scalar|(?:coh-)?packet(\d+)): "
    r"([0-9.]+) ms, ([0-9.]+) MRay/s, "
    r"hits=(\d+), checksum=([-+0-9.eE]+)$"
)
BUILD_THREAD_MODE = re.compile(
    r"^=== BVH build threads: (1|all); available CPUs: (\d+); "
    r"affinity: (.+) ===$"
)


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
    packet_build_method: str | None = None
    packet_bounds_width: int | None = None
    packet_leaf_width: int | None = None
    build_threads = SINGLE_THREAD_MODE
    available_cpus = 1
    cpu_affinity = "unknown"

    for raw_line in output.splitlines():
        line = raw_line.strip()

        if not line:
            table_kind = None
            continue

        thread_mode = BUILD_THREAD_MODE.fullmatch(line)
        if thread_mode is not None:
            build_threads = thread_mode.group(1)
            available_cpus = int(thread_mode.group(2))
            cpu_affinity = thread_mode.group(3)
            table_kind = None
            continue

        # Mojo runtime diagnostics can be emitted between benchmark processes
        # without a separating blank line. They are not part of any table.
        if (
            (line.startswith("[") and ":ERROR " in line)
            or line.startswith("Failed to initialize Crashpad.")
        ):
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

        if line == "CPU shared-stack packet BVH benchmark":
            implementation = "bajo"
            version = None
            benchmark = None
            primitive_count = None
            ray_count = None
            table_kind = None
            continue

        packet_section = PACKET_SECTION.fullmatch(line)
        if packet_section is not None:
            benchmark = (
                "grid"
                if packet_section.group(1) == "Regular grid"
                else "dragon"
            )
            packet_build_method = packet_section.group(2).strip()
            packet_bounds_width = int(packet_section.group(3))
            packet_leaf_width = int(packet_section.group(4))
            table_kind = "bajo_packet"
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
            "build_threads": build_threads,
            "available_cpus": available_cpus,
            "cpu_affinity": cpu_affinity,
            "primitive_count": primitive_count,
            "ray_count": ray_count,
            "nodes": None,
            "leaf_width": None,
            "traversal": "scalar1",
            "ray_width": 1,
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
                    "leaf_width": width,
                    "traversal": "scalar1",
                    "ray_width": 1,
                    "build_ms": float(values[3]),
                    "nodes": int(values[4]),
                    "trace_ms": float(values[6]),
                    "mrays_s": float(values[7]),
                    "hits": None,
                    "checksum": float(values[8]),
                }

            elif table_kind == "bajo_dragon":
                if len(values) != 9:
                    raise ValueError(
                        f"expected 9 columns, received {len(values)}"
                    )

                bounds_width = int(values[1])
                leaf_width = int(values[2])

                row = base | {
                    "build_method": values[0],
                    "layout": f"bvh{bounds_width}",
                    "width": bounds_width,
                    "leaf_width": leaf_width,
                    "traversal": values[3],
                    "ray_width": 1,
                    "build_ms": float(values[4]),
                    "trace_ms": float(values[5]),
                    "mrays_s": float(values[6]),
                    "hits": int(values[7]),
                    "checksum": float(values[8]),
                }

            elif table_kind == "bajo_packet":
                timing = PACKET_TIMING.fullmatch(line)

                if timing is None:
                    raise ValueError("expected a packet timing row")

                # Scalar controls are already present in the primary Bajo
                # tables. Keep only the additional packet configurations.
                if timing.group(1) == "scalar":
                    continue

                if (
                    packet_build_method is None
                    or packet_bounds_width is None
                    or packet_leaf_width is None
                ):
                    raise ValueError("packet section metadata is incomplete")

                ray_width = (
                    1 if timing.group(2) is None else int(timing.group(2))
                )
                row = base | {
                    "build_method": packet_build_method,
                    "layout": f"bvh{packet_bounds_width}",
                    "width": packet_bounds_width,
                    "leaf_width": packet_leaf_width,
                    "traversal": timing.group(1),
                    "ray_width": ray_width,
                    "build_ms": None,
                    "trace_ms": float(timing.group(3)),
                    "mrays_s": float(timing.group(4)),
                    "hits": int(timing.group(5)),
                    "checksum": float(timing.group(6)),
                }

            elif table_kind in ("embree", "tinybvh"):
                if len(values) != 7:
                    raise ValueError(
                        f"expected 7 columns, received {len(values)}"
                    )

                traversal = values[1] if table_kind == "embree" else "scalar1"
                ray_width = (
                    width_from_layout(values[1])
                    if table_kind == "embree"
                    else 1
                )
                row = base | {
                    "build_method": values[0],
                    "layout": (
                        "native" if table_kind == "embree" else values[1]
                    ),
                    "width": (
                        None
                        if table_kind == "embree"
                        else width_from_layout(values[1])
                    ),
                    "leaf_width": None,
                    "traversal": traversal,
                    "ray_width": ray_width,
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

    # Auxiliary traversal rows reuse the exact same built hierarchy. Carry the
    # build time from the matching primary scalar configuration into the report.
    bajo_build_times: dict[tuple[object, ...], object] = {}
    for row in rows:
        if (
            row["implementation"] == "bajo"
            and row["traversal"] == "scalar1"
            and row["build_ms"] is not None
        ):
            key = (
                row["build_threads"],
                row["benchmark"],
                row["build_method"],
                row["width"],
                row["leaf_width"],
            )
            bajo_build_times[key] = row["build_ms"]

    for row in rows:
        if row["implementation"] != "bajo":
            continue

        if row["build_ms"] is not None:
            continue

        key = (
            row["build_threads"],
            row["benchmark"],
            row["build_method"],
            row["width"],
            row["leaf_width"],
        )
        row["build_ms"] = bajo_build_times.get(key)

    return pl.DataFrame(
        rows,
        infer_schema_length=None,
    ).sort(
        [
            "benchmark",
            "build_threads",
            "implementation",
            "build_method",
            "width",
            "leaf_width",
            "traversal",
            "ray_width",
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


def merge_build_modes(frame: pl.DataFrame) -> pl.DataFrame:
    """Return one canonical traversal row with both build measurements.

    Traversal always uses one calling thread. Use the threads=1 traversal as
    the canonical sample, while preserving the all-thread build time from the
    matching configuration. If the same traversal is emitted by more than one
    benchmark binary, retain its faster canonical sample.
    """
    identity = [
        "benchmark",
        "implementation",
        "build_method",
        "layout",
        "width",
        "leaf_width",
        "traversal",
        "ray_width",
    ]

    single = (
        frame
        .filter(pl.col("build_threads") == SINGLE_THREAD_MODE)
        .sort("mrays_s", descending=True)
        .unique(subset=identity, keep="first", maintain_order=True)
        .rename({"build_ms": "build_ms_1"})
    )
    all_builds = (
        frame
        .filter(pl.col("build_threads") == ALL_THREAD_MODE)
        .sort("mrays_s", descending=True)
        .unique(subset=identity, keep="first", maintain_order=True)
        .select(identity + [pl.col("build_ms").alias("build_ms_all")])
    )

    merged = single.join(
        all_builds,
        on=identity,
        how="left",
        validate="1:1",
        nulls_equal=True,
    )
    if merged.get_column("build_ms_all").null_count() != 0:
        raise ValueError("Every traversal row requires an all-thread build time")

    return (
        merged
        .drop("build_threads", "available_cpus", "cpu_affinity")
        .sort(
            [
                "benchmark",
                "implementation",
                "build_method",
                "width",
                "leaf_width",
                "traversal",
                "ray_width",
            ]
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
) -> str:
    labels = {
        "benchmark": "Benchmark",
        "implementation": "Implementation",
        "build_threads": "Build threads",
        "build_method": "Build",
        "layout": "Layout",
        "width": "Width",
        "leaf_width": "Leaf width",
        "traversal": "Traversal",
        "ray_width": "Ray width",
        "build_ms": "Build ms",
        "trace_ms": "Trace ms",
        "mrays_s": "MRay/s",
        "vs_bajo_pct": "vs Bajo (%)",
        "hits": "Hits",
        "nodes": "Nodes",
        "checksum": "Checksum",
        "build_ms_1": "Build ms (1)",
        "build_ms_all": "Build ms (all)",
    }

    result_columns = [
        "implementation",
        "build_method",
        "layout",
        "width",
        "leaf_width",
        "traversal",
        "ray_width",
        "build_ms_1",
        "build_ms_all",
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
        "traversal",
        "ray_width",
        "build_ms_1",
        "build_ms_all",
        "trace_ms",
        "mrays_s",
        "vs_bajo_pct",
    ]

    mojo_version = run_command("mojo", "--version")

    single_threaded = frame.filter(
        pl.col("build_threads") == SINGLE_THREAD_MODE
    )
    multithreaded = frame.filter(
        pl.col("build_threads") == ALL_THREAD_MODE
    )
    if single_threaded.is_empty() or multithreaded.is_empty():
        raise ValueError("Report requires both single- and all-thread results")
    merged = merge_build_modes(frame)
    grid = merged.filter(pl.col("benchmark") == "grid")
    dragon = merged.filter(pl.col("benchmark") == "dragon")
    best = best_results(merged)
    all_available_cpus = multithreaded.get_column("available_cpus").item(0)
    all_cpu_affinity = multithreaded.get_column("cpu_affinity").item(0)

    return "\n".join(
        [
            "# CPU BVH benchmark results",
            "",
            f"- **Date:** {timestamp.isoformat(timespec='seconds')}",
            f"- **CPU:** {cpu_name()}",
            f"- **System:** {platform.platform()}",
            f"- **Mojo:** `{mojo_version}`",
            "- **Build thread modes:** `1` and `all`",
            f"- **All-thread affinity:** `{all_cpu_affinity}` "
            f"({all_available_cpus} logical CPUs)",
            "- **Traversal:** one calling thread; timings use the `threads=1` "
            "run",
            "- **Raw data:** CSV/TXT retain both build-thread runs",
            "- **Build timing:** one sample per configuration; descriptive, "
            "not a regression gate",
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
