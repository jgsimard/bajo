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
PACKET_SECTION = re.compile(r"^(.+) / ([^/]+) / BVH(\d+) leaf(\d+)$")
PACKET_TIMING = re.compile(
    r"^(?P<traversal>"
    r"(?:unmasked-)?scalar|"
    r"(?:coh-)?packet(?P<packet_width>\d+)|"
    r"adaptive-(?P<adaptive_widths>\d+(?:-\d+)*)-scalar"
    r"): "
    r"(?P<trace_ms>[0-9.]+) ms, "
    r"(?P<mrays_s>[0-9.]+) MRay/s, "
    r"hits=(?P<hits>\d+), "
    r"checksum=(?P<checksum>[-+0-9.eE]+)$"
)
BUILD_THREAD_MODE = re.compile(
    r"^=== BVH build threads: (1|all); available CPUs: (\d+); "
    r"affinity: (.+) ===$"
)

BENCHMARK_META: dict[str, dict[str, str]] = {
    "grid_closest": {
        "title": "Regular-grid closest-hit",
        "geometry": "grid",
        "query": "closest",
        "ray_order": "structured",
    },
    "grid_any": {
        "title": "Regular-grid any-hit",
        "geometry": "grid",
        "query": "any",
        "ray_order": "structured",
    },
    "dragon_camera_closest": {
        "title": "Dragon camera closest-hit",
        "geometry": "dragon",
        "query": "closest",
        "ray_order": "camera",
    },
    "dragon_camera_any": {
        "title": "Dragon camera any-hit",
        "geometry": "dragon",
        "query": "any",
        "ray_order": "camera",
    },
    "dragon_shuffled_closest": {
        "title": "Dragon shuffled closest-hit",
        "geometry": "dragon",
        "query": "closest",
        "ray_order": "shuffled-camera",
    },
    "dragon_shuffled_any": {
        "title": "Dragon shuffled any-hit",
        "geometry": "dragon",
        "query": "any",
        "ray_order": "shuffled-camera",
    },
    "dragon_instances_closest": {
        "title": "Instanced Dragon closest-hit",
        "geometry": "dragon-instances",
        "query": "closest",
        "ray_order": "camera",
    },
    "dragon_instances_any": {
        "title": "Instanced Dragon any-hit",
        "geometry": "dragon-instances",
        "query": "any",
        "ray_order": "camera",
    },
    "triangle_instances_closest": {
        "title": "Instanced triangle closest-hit",
        "geometry": "triangle-instances",
        "query": "closest",
        "ray_order": "camera",
    },
    "triangle_instances_any": {
        "title": "Instanced triangle any-hit",
        "geometry": "triangle-instances",
        "query": "any",
        "ray_order": "camera",
    },
    "triangle_grid_closest": {
        "title": "Flattened triangle grid closest-hit",
        "geometry": "triangle-grid",
        "query": "closest",
        "ray_order": "camera",
    },
    "triangle_grid_any": {
        "title": "Flattened triangle grid any-hit",
        "geometry": "triangle-grid",
        "query": "any",
        "ray_order": "camera",
    },
}

SECTION_BENCHMARKS = {
    "Regular-grid microbenchmark": "grid_closest",
    "Regular-grid closest-hit benchmark": "grid_closest",
    "Regular-grid any-hit benchmark": "grid_any",
    "Representative Dragon camera-ray benchmark": "dragon_camera_closest",
    "Dragon camera closest-hit benchmark": "dragon_camera_closest",
    "Dragon camera any-hit benchmark": "dragon_camera_any",
    "Dragon shuffled closest-hit benchmark": "dragon_shuffled_closest",
    "Dragon shuffled any-hit benchmark": "dragon_shuffled_any",
    "Instanced Dragon closest-hit benchmark": "dragon_instances_closest",
    "Instanced Dragon any-hit benchmark": "dragon_instances_any",
    "Instanced triangle closest-hit benchmark": (
        "triangle_instances_closest"
    ),
    "Instanced triangle any-hit benchmark": "triangle_instances_any",
    "Flattened triangle grid closest-hit benchmark": "triangle_grid_closest",
    "Flattened triangle grid any-hit benchmark": "triangle_grid_any",
}

PACKET_BENCHMARKS = {
    "Regular grid": "grid_closest",
    "Regular grid closest-hit": "grid_closest",
    "Regular grid any-hit": "grid_any",
    "Dragon camera rays": "dragon_camera_closest",
    "Dragon camera closest-hit": "dragon_camera_closest",
    "Dragon camera any-hit": "dragon_camera_any",
    "Dragon shuffled closest-hit": "dragon_shuffled_closest",
    "Dragon shuffled any-hit": "dragon_shuffled_any",
    "Instanced Dragon closest-hit": "dragon_instances_closest",
    "Instanced Dragon any-hit": "dragon_instances_any",
    "Instanced triangle closest-hit": "triangle_instances_closest",
    "Instanced triangle any-hit": "triangle_instances_any",
    "Flattened triangle grid closest-hit": "triangle_grid_closest",
    "Flattened triangle grid any-hit": "triangle_grid_any",
}


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
    instance_count = 1
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
        if (line.startswith("[") and ":ERROR " in line) or line.startswith(
            "Failed to initialize Crashpad."
        ):
            continue

        # Implementation and benchmark sections.
        if line in (
            "Primitive BoundsBvh benchmark",
            "PrimitiveKind BoundsBvh benchmark",
        ):
            implementation = "bajo"
            version = None
            benchmark = "grid_closest"
            primitive_count = None
            instance_count = 1
            ray_count = None
            table_kind = None
            continue

        if line.startswith("Embree ") and line.endswith(
            " CPU triangle benchmark"
        ):
            implementation = "embree"
            version = line.removesuffix(" CPU triangle benchmark")
            benchmark = None
            instance_count = 1
            table_kind = None
            continue

        if line.startswith("TinyBVH ") and line.endswith(
            " CPU triangle benchmark"
        ):
            implementation = "tinybvh"
            version = line.removesuffix(" CPU triangle benchmark")
            benchmark = None
            instance_count = 1
            table_kind = None
            continue

        if line == "CPU shared-stack packet BVH benchmark":
            implementation = "bajo"
            version = None
            benchmark = None
            primitive_count = None
            instance_count = 1
            ray_count = None
            table_kind = None
            continue

        if line in (
            "CPU instanced Dragon BVH benchmark",
            "CPU instanced closest-hit diagnostic benchmark",
        ):
            implementation = "bajo"
            version = None
            benchmark = None
            primitive_count = None
            instance_count = 1
            ray_count = None
            table_kind = None
            continue

        packet_section = PACKET_SECTION.fullmatch(line)
        if packet_section is not None:
            section_name = packet_section.group(1).strip()
            benchmark = PACKET_BENCHMARKS.get(section_name)
            if benchmark is None:
                raise ValueError(
                    f"Unknown Bajo packet benchmark section: {section_name!r}"
                )
            packet_build_method = packet_section.group(2).strip()
            packet_bounds_width = int(packet_section.group(3))
            packet_leaf_width = int(packet_section.group(4))
            table_kind = "bajo_packet"
            continue

        if line in SECTION_BENCHMARKS:
            # The first implementation is Bajo and has no implementation title.
            if (
                line == "Representative Dragon camera-ray benchmark"
                and implementation is None
            ):
                implementation = "bajo"

            benchmark = SECTION_BENCHMARKS[line]
            primitive_count = None
            instance_count = 1
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

        if line.startswith("Instances:"):
            instance_count = int(line.split(":", 1)[1].strip())
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

        if benchmark is None or implementation is None:
            raise ValueError(
                f"Result row has incomplete section metadata: {raw_line!r}"
            )
        benchmark_meta = BENCHMARK_META[benchmark]
        base: dict[str, object] = {
            "benchmark": benchmark,
            "geometry": benchmark_meta["geometry"],
            "query": benchmark_meta["query"],
            "ray_order": benchmark_meta["ray_order"],
            "implementation": implementation,
            "version": version,
            "build_threads": build_threads,
            "available_cpus": available_cpus,
            "cpu_affinity": cpu_affinity,
            "primitive_count": primitive_count,
            "instance_count": instance_count,
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
                    "ray_width": width_from_layout(values[3]),
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
                # closest-hit tables. Any-hit has no other scalar baseline.
                traversal = timing.group("traversal")
                if traversal == "scalar":
                    if benchmark in (
                        "grid_closest",
                        "dragon_camera_closest",
                    ):
                        continue
                    traversal = "scalar1"

                if (
                    packet_build_method is None
                    or packet_bounds_width is None
                    or packet_leaf_width is None
                ):
                    raise ValueError("packet section metadata is incomplete")

                packet_width = timing.group("packet_width")
                adaptive_widths = timing.group("adaptive_widths")
                if packet_width is not None:
                    ray_width = int(packet_width)
                elif adaptive_widths is not None:
                    ray_width = int(adaptive_widths.split("-", 1)[0])
                else:
                    ray_width = 1
                row = base | {
                    "build_method": packet_build_method,
                    "layout": f"bvh{packet_bounds_width}",
                    "width": packet_bounds_width,
                    "leaf_width": packet_leaf_width,
                    "traversal": traversal,
                    "ray_width": ray_width,
                    "build_ms": None,
                    "trace_ms": float(timing.group("trace_ms")),
                    "mrays_s": float(timing.group("mrays_s")),
                    "hits": int(timing.group("hits")),
                    "checksum": float(timing.group("checksum")),
                }

            elif table_kind in ("embree", "tinybvh"):
                if len(values) != 7:
                    raise ValueError(
                        f"expected 7 columns, received {len(values)}"
                    )

                traversal = values[1] if table_kind == "embree" else "scalar1"
                ray_width = (
                    width_from_layout(values[1]) if table_kind
                    == "embree" else 1
                )
                row = base | {
                    "build_method": values[0],
                    "layout": (
                        "native" if table_kind == "embree" else values[1]
                    ),
                    "width": (
                        None if table_kind
                        == "embree" else width_from_layout(values[1])
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
                row["geometry"],
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
            row["geometry"],
            row["build_method"],
            row["width"],
            row["leaf_width"],
        )
        row["build_ms"] = bajo_build_times.get(key)

    return pl.DataFrame(rows, infer_schema_length=None,).sort(
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
        benchmark_frame = frame.filter(pl.col("benchmark") == benchmark)

        implementations = (
            benchmark_frame.get_column("implementation")
            .unique()
            .sort()
            .to_list()
        )

        for implementation in implementations:
            best = (
                benchmark_frame.filter(
                    pl.col("implementation") == implementation
                )
                .sort("mrays_s", descending=True)
                .head(1)
            )
            selections.append(best)

    best = pl.concat(selections)

    bajo_reference = best.filter(pl.col("implementation") == "bajo").select(
        "benchmark",
        pl.col("mrays_s").alias("bajo_mrays_s"),
    )

    return (
        best.join(
            bajo_reference,
            on="benchmark",
            how="left",
            validate="m:1",
        )
        .with_columns(
            ((pl.col("mrays_s") / pl.col("bajo_mrays_s") - 1.0) * 100.0).alias(
                "vs_bajo_pct"
            )
        )
        .drop("bajo_mrays_s")
        .sort(
            ["benchmark", "mrays_s"],
            descending=[False, True],
        )
    )


def add_benchmark_titles(frame: pl.DataFrame) -> pl.DataFrame:
    titles = pl.DataFrame(
        {
            "benchmark": list(BENCHMARK_META),
            "benchmark_title": [
                metadata["title"] for metadata in BENCHMARK_META.values()
            ],
        }
    )
    return frame.join(titles, on="benchmark", how="left", validate="m:1")


def validate_benchmark_results(frame: pl.DataFrame) -> None:
    """Reject accidentally incomparable inputs and correctness disagreements."""
    for benchmark in frame.get_column("benchmark").unique().to_list():
        subset = frame.filter(pl.col("benchmark") == benchmark)
        for column in ("primitive_count", "instance_count", "ray_count"):
            values = subset.get_column(column).drop_nulls().unique()
            if len(values) != 1:
                raise ValueError(
                    f"{benchmark}: inconsistent {column}: {values.to_list()}"
                )

        hit_counts = subset.get_column("hits").drop_nulls().unique()
        if len(hit_counts) > 1:
            ray_count = subset.get_column("ray_count").drop_nulls()[0]
            tolerance = max(2, int(ray_count * 5.0e-5))
            hit_span = int(hit_counts.max()) - int(hit_counts.min())
        else:
            tolerance = 0
            hit_span = 0
        if hit_span > tolerance:
            raise ValueError(
                f"{benchmark}: traversal hit counts disagree: "
                f"{hit_counts.to_list()} (tolerance {tolerance})"
            )


def competitor_gaps(best: pl.DataFrame) -> pl.DataFrame:
    bajo = best.filter(pl.col("implementation") == "bajo").select(
        "benchmark",
        pl.col("mrays_s").alias("bajo_mrays_s"),
        pl.col("traversal").alias("bajo_traversal"),
    )
    competitor = (
        best.filter(pl.col("implementation") != "bajo")
        .sort("mrays_s", descending=True)
        .unique(subset=["benchmark"], keep="first", maintain_order=True)
        .select(
            "benchmark",
            pl.col("implementation").alias("fastest_competitor"),
            pl.col("mrays_s").alias("competitor_mrays_s"),
            pl.col("traversal").alias("competitor_traversal"),
        )
    )
    return add_benchmark_titles(
        bajo.join(competitor, on="benchmark", how="inner", validate="1:1")
        .with_columns(
            (
                (pl.col("bajo_mrays_s") / pl.col("competitor_mrays_s") - 1.0)
                * 100.0
            ).alias("bajo_vs_competitor_pct")
        )
        .sort("bajo_vs_competitor_pct")
    )


def best_build_results(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.filter(pl.col("traversal") == "scalar1")
        .sort("build_ms_all")
        .unique(
            subset=["geometry", "implementation"],
            keep="first",
            maintain_order=True,
        )
        .select(
            "geometry",
            "implementation",
            "build_method",
            "layout",
            "build_ms_1",
            "build_ms_all",
        )
        .sort(["geometry", "build_ms_all"])
    )


def build_competitor_gaps(best_build: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for geometry in best_build.get_column("geometry").unique().sort().to_list():
        geometry_frame = best_build.filter(pl.col("geometry") == geometry)
        bajo = geometry_frame.filter(pl.col("implementation") == "bajo")
        competitors = geometry_frame.filter(pl.col("implementation") != "bajo")
        for mode, column in (("1", "build_ms_1"), ("all", "build_ms_all")):
            fastest = competitors.sort(column).head(1)
            bajo_ms = float(bajo.get_column(column).item())
            competitor_ms = float(fastest.get_column(column).item())
            rows.append(
                {
                    "geometry": geometry,
                    "build_threads": mode,
                    "bajo_build_ms": bajo_ms,
                    "fastest_competitor": fastest.get_column(
                        "implementation"
                    ).item(),
                    "competitor_build_ms": competitor_ms,
                    "bajo_vs_competitor_pct": (
                        competitor_ms / bajo_ms - 1.0
                    )
                    * 100.0,
                }
            )
    return pl.DataFrame(rows).sort("bajo_vs_competitor_pct")


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
        frame.filter(pl.col("build_threads") == SINGLE_THREAD_MODE)
        .sort("mrays_s", descending=True)
        .unique(subset=identity, keep="first", maintain_order=True)
        .rename({"build_ms": "build_ms_1"})
    )
    all_builds = (
        frame.filter(pl.col("build_threads") == ALL_THREAD_MODE)
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
        raise ValueError(
            "Every traversal row requires an all-thread build time"
        )

    return merged.drop("build_threads", "available_cpus", "cpu_affinity").sort(
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
        "benchmark_title": "Workload",
        "geometry": "Geometry",
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
        "bajo_mrays_s": "Bajo MRay/s",
        "bajo_traversal": "Bajo traversal",
        "fastest_competitor": "Fastest competitor",
        "competitor_mrays_s": "Competitor MRay/s",
        "competitor_traversal": "Competitor traversal",
        "bajo_vs_competitor_pct": "Bajo vs competitor (%)",
        "bajo_build_ms": "Bajo build ms",
        "competitor_build_ms": "Competitor build ms",
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
        "benchmark_title",
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

    gap_columns = [
        "benchmark_title",
        "bajo_traversal",
        "bajo_mrays_s",
        "fastest_competitor",
        "competitor_traversal",
        "competitor_mrays_s",
        "bajo_vs_competitor_pct",
    ]

    build_columns = [
        "geometry",
        "implementation",
        "build_method",
        "layout",
        "build_ms_1",
        "build_ms_all",
    ]

    build_gap_columns = [
        "geometry",
        "build_threads",
        "bajo_build_ms",
        "fastest_competitor",
        "competitor_build_ms",
        "bajo_vs_competitor_pct",
    ]

    mojo_version = run_command("mojo", "--version")
    gxx_version = run_command("g++", "--version").splitlines()[0]
    clangxx_version = run_command("clang++", "--version").splitlines()[0]

    single_threaded = frame.filter(
        pl.col("build_threads") == SINGLE_THREAD_MODE
    )
    multithreaded = frame.filter(pl.col("build_threads") == ALL_THREAD_MODE)
    if single_threaded.is_empty() or multithreaded.is_empty():
        raise ValueError("Report requires both single- and all-thread results")
    validate_benchmark_results(frame)
    merged = merge_build_modes(frame)
    best = add_benchmark_titles(best_results(merged))
    scalar_best = add_benchmark_titles(
        best_results(
            merged.filter(
                (pl.col("traversal") == "scalar1")
                & (pl.col("ray_width") == 1)
            )
        )
    )
    gaps = competitor_gaps(best)
    build_best = best_build_results(merged)
    build_gaps = build_competitor_gaps(build_best)
    implementation_versions = {
        implementation: ", ".join(
            str(version)
            for version in frame.filter(
                pl.col("implementation") == implementation
            )
            .get_column("version")
            .drop_nulls()
            .unique()
            .sort()
            .to_list()
        )
        for implementation in ("embree", "tinybvh")
    }
    all_available_cpus = multithreaded.get_column("available_cpus").item(0)
    all_cpu_affinity = multithreaded.get_column("cpu_affinity").item(0)

    sections: list[str] = []
    for benchmark, metadata in BENCHMARK_META.items():
        results = merged.filter(pl.col("benchmark") == benchmark)
        if results.is_empty():
            continue
        primitive_count = results.get_column("primitive_count").drop_nulls()[0]
        instance_count = results.get_column("instance_count").drop_nulls()[0]
        ray_count = results.get_column("ray_count").drop_nulls()[0]
        sections.extend(
            [
                f"## {metadata['title']}",
                "",
                f"Triangles per BLAS: {primitive_count}; instances: "
                f"{instance_count}; rays: {ray_count}; query: "
                f"{metadata['query']}; ray order: {metadata['ray_order']}.",
                "",
                markdown_table(results, result_columns, labels),
                "",
            ]
        )

    return "\n".join(
        [
            "# CPU BVH benchmark results",
            "",
            f"- **Date:** {timestamp.isoformat(timespec='seconds')}",
            f"- **CPU:** {cpu_name()}",
            f"- **System:** {platform.platform()}",
            f"- **Mojo:** `{mojo_version}`",
            f"- **Embree:** `{implementation_versions['embree']}`; compiler: "
            f"`{gxx_version}`",
            f"- **TinyBVH:** `{implementation_versions['tinybvh']}`; compiler: "
            f"`{clangxx_version}`",
            "- **C++ flags:** Embree harness `-O3 -DNDEBUG -march=native`; "
            "TinyBVH harness additionally `-ffast-math -mavx2 -mfma`",
            "- **Build thread modes:** `1` and `all`",
            f"- **All-thread affinity:** `{all_cpu_affinity}` "
            f"({all_available_cpus} logical CPUs)",
            "- **Traversal:** one calling thread; timings use the `threads=1` "
            "run",
            "- **Raw data:** CSV/TXT retain both build-thread runs",
            "- **Build timing:** median of five builds after one untimed "
            "warm-up per configuration",
            "- **Correctness gate:** triangle/instance/ray counts must match; hit "
            "counts must agree within 50 ppm (minimum two boundary rays)",
            "- **Interpretation:** negative `Bajo vs competitor` means Bajo is "
            "slower; positive means faster",
            "",
            "## Where Bajo still needs work",
            "",
            "Traversal deficits larger than 2%:",
            "",
            markdown_table(
                gaps.filter(pl.col("bajo_vs_competitor_pct") < -2.0),
                gap_columns,
                labels,
            ),
            "",
            "Build deficits larger than 2% (lower time is better):",
            "",
            markdown_table(
                build_gaps.filter(pl.col("bajo_vs_competitor_pct") < -2.0),
                build_gap_columns,
                labels,
            ),
            "",
            "These are the optimization queue: the most negative rows are the "
            "largest measured deficits on this machine. Differences within "
            "2% are treated as parity.",
            "",
            "## Best traversal per implementation",
            "",
            markdown_table(best, best_columns, labels),
            "",
            "## Best scalar traversal per implementation",
            "",
            "This removes packet-width advantages and is the fairest direct "
            "comparison with TinyBVH's scalar API.",
            "",
            markdown_table(scalar_best, best_columns, labels),
            "",
            "## Fastest build per geometry and implementation",
            "",
            markdown_table(build_best, build_columns, labels),
            "",
            "## Coverage",
            "",
            "The matrix covers synthetic and real mesh geometry, closest-hit "
            "and early-exit any-hit, coherent camera ordering and the same rays "
            "shuffled to remove neighboring-ray coherence, plus an instance-heavy "
            "BLAS/TLAS scene (one reused BLAS, a 12x9 translated-instance grid). "
            "A one-triangle BLAS and its flattened 108-triangle equivalent isolate "
            "instance continuation from BLAS complexity. "
            "Traversal is single-calling-thread; build is measured with one CPU "
            "and all available CPUs. The core traversal suites report the best "
            "of eight timed repetitions after one warmup. Instance diagnostics "
            "report the median of eight repetitions, each averaged across eight "
            "timed traversal batches, to resolve small performance differences.",
            "",
            *sections,
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
