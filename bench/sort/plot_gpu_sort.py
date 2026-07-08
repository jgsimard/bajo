import os
import sys

import polars as pl
import matplotlib.pyplot as plt


OUTPUT_FILENAME = "gpu_sort_benchmark_plot.png"

DEFAULT_CSV_FILES = [
    "gpu_sort_benchmark_results.csv",
    "cub_sort_benchmark_results.csv",
]

# Ordered list. Lines are plotted only if present in the CSV.
PLOT_ALGORITHMS = [
    # Main comparison
    "Radix_Pairs",
    "Radix_Keys",
    "Onesweep_Pairs",
    "Onesweep_Keys",
    "CUB_Radix_Pairs",
    "CUB_Radix_Keys",

    # Bajo/Mojo baselines
    "Basic_Bitonic_Sort_Pairs",
    "Shared_Memory_Bitonic_Sort_Pairs",
    "SMEM bitonic + Onesweep_Pairs",
]

STYLES = {
    # Bajo / Mojo radix
    "Radix_Pairs": {
        "color": "#2ca02c",
        "marker": "^",
        "ls": "-",
        "alpha": 0.85,
        "lw": 2.0,
        "label": "Mojo Radix pairs",
    },
    "Radix_Keys": {
        "color": "#2ca02c",
        "marker": "^",
        "ls": "--",
        "alpha": 0.85,
        "lw": 2.0,
        "label": "Mojo Radix keys",
    },

    # Bajo / Mojo OneSweep
    "Onesweep_Pairs": {
        "color": "#ff7f0e",
        "marker": "D",
        "ls": "-",
        "alpha": 0.95,
        "lw": 2.5,
        "label": "Mojo OneSweep pairs",
    },
    "Onesweep_Keys": {
        "color": "#ff7f0e",
        "marker": "D",
        "ls": "--",
        "alpha": 0.95,
        "lw": 2.5,
        "label": "Mojo OneSweep keys",
    },

    # CUB
    "CUB_Radix_Pairs": {
        "color": "#1f77b4",
        "marker": "s",
        "ls": "-",
        "alpha": 0.95,
        "lw": 2.0,
        "label": "CUB pairs",
    },
    "CUB_Radix_Keys": {
        "color": "#1f77b4",
        "marker": "s",
        "ls": "--",
        "alpha": 0.95,
        "lw": 2.0,
        "label": "CUB keys",
    },

    # Bajo / Mojo bitonic baselines
    "Basic_Bitonic_Sort_Pairs": {
        "color": "#8c564b",
        "marker": "o",
        "ls": ":",
        "alpha": 0.70,
        "lw": 1.6,
        "label": "Mojo basic bitonic pairs",
    },
    "Shared_Memory_Bitonic_Sort_Pairs": {
        "color": "#9467bd",
        "marker": "o",
        "ls": "-.",
        "alpha": 0.75,
        "lw": 1.8,
        "label": "Mojo shared-memory bitonic pairs",
    },
    "SMEM bitonic + Onesweep_Pairs": {
        "color": "#d62728",
        "marker": "o",
        "ls": "-",
        "alpha": 0.80,
        "lw": 2.0,
        "label": "Mojo hybrid pairs",
    },
}


def format_size(n):
    try:
        n = int(n)
        if n >= 1_000_000:
            return f"{int(n / 1_000_000)}M"
        if n >= 1_000:
            return f"{int(n / 1_000)}k"
        return str(n)
    except Exception:
        return str(n)


def load_results(csv_files):
    frames = []

    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"Warning: {csv_file} not found; skipping.")
            continue

        frames.append(pl.read_csv(csv_file))

    if not frames:
        raise FileNotFoundError(
            "No benchmark CSV found. Expected at least one input CSV."
        )

    return pl.concat(frames, how="vertical")


def filter_for_plot(df):
    available = set(df["Algorithm"].unique().to_list())

    plotted = [algo for algo in PLOT_ALGORITHMS if algo in available]

    if not plotted:
        raise ValueError(
            "No plottable algorithms found. Check Algorithm names in the CSV files."
        )

    ignored = sorted(available - set(PLOT_ALGORITHMS))
    if ignored:
        print("Info: ignoring unstyled algorithms:")
        for algo in ignored:
            print(f"  - {algo}")

    return df.filter(pl.col("Algorithm").is_in(plotted)), plotted


def plot_results(csv_files=None):
    if csv_files is None:
        csv_files = DEFAULT_CSV_FILES

    df = load_results(csv_files)
    df, plotted_algorithms = filter_for_plot(df)

    plt.figure(figsize=(12, 7), facecolor="#f8f9fa")
    plt.xscale("log", base=2)
    plt.grid(True, which="both", ls="-", alpha=0.12, color="black")

    for algo in plotted_algorithms:
        subset = df.filter(pl.col("Algorithm") == algo).sort("N")
        if subset.is_empty():
            continue

        plt.plot(
            subset["N"].to_list(),
            subset["Throughput_GKs"].to_list(),
            **STYLES[algo],
        )

    plt.title(
        "GPU Sort Throughput: Bajo/Mojo vs CUB on RTX 5060 Ti",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Input size (N)", fontsize=12)
    plt.ylabel("Throughput (GK/s)", fontsize=12)

    unique_n = df["N"].unique().sort().to_list()
    plt.xticks(unique_n, [format_size(n) for n in unique_n])

    max_y = df["Throughput_GKs"].max()
    plt.ylim(0, max_y * 1.15)

    # Clean external legend. More columns if bitonic baselines are present.
    ncol = 3 if len(plotted_algorithms) <= 6 else 4

    plt.legend(
        loc="upper left",
        frameon=True,
        fancybox=True,
        shadow=False,
        fontsize=9,
        handlelength=2.4,
        columnspacing=1.4,
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_FILENAME, dpi=300)
    print(f"Graph successfully saved as {OUTPUT_FILENAME}")
    plt.show()


if __name__ == "__main__":
    csv_files = sys.argv[1:] if len(sys.argv) > 1 else None
    plot_results(csv_files)