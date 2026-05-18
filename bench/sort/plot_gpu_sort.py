import polars as pl
import matplotlib.pyplot as plt
import os

def format_size(n):
    """Formats large integers into k/M human-readable strings."""
    try:
        n = int(n)
        if n >= 1_000_000:
            return f"{int(n/1_000_000)}M"
        if n >= 1_000:
            return f"{int(n/1_000)}k"
        return str(n)
    except:
        return str(n)

def plot_results(csv_file="gpu_sort_benchmark_results.csv"):
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Run the Mojo benchmark first.")
        return

    # Load data
    df = pl.read_csv(csv_file)

    
    # Define visual styles for the labels used in your Mojo code
    styles = {
        'Radix_Pairs':    {'color': '#2ca02c', 'marker': '^', 'ls': '-',  'alpha': 0.8, 'label': 'Radix Sort (Pairs)'},
        'Radix_Keys':     {'color': '#2ca02c', 'marker': '^', 'ls': '--', 'alpha': 0.7, 'label': 'Radix Sort (Keys)'},
        'Onesweep_Pairs': {'color': '#ff7f0e', 'marker': 'D', 'ls': '-',  'alpha': 0.9, 'lw': 2.5, 'label': 'OneSweep (Pairs)'},
        'Onesweep_Keys': {'color': '#ff7f0e', 'marker': 'D', 'ls': '--',  'alpha': 0.9, 'lw': 2.5, 'label': 'OneSweep (Keys)'},
    }

    plt.figure(figsize=(12, 7), facecolor='#f8f9fa')
    plt.xscale('log', base=2)
    plt.grid(True, which="both", ls="-", alpha=0.1, color='black')

    # Get unique labels
    algorithms = df["Algorithm"].unique().to_list()

    for algo in algorithms:
        subset = df.filter(pl.col("Algorithm") == algo).sort("N")
        style = styles.get(algo, {'marker': 'o', 'label': algo})
        
        plt.plot(
            subset["N"].to_list(), 
            subset["Throughput_GKs"].to_list(), 
            **style
        )

    # Hardware Specifics: RTX 5060 Ti L2 Cache (32MB)
    # Pairs (uint32 keys + uint32 vals) = 16 bytes/element total (Double Buffered)
    # Keys (uint32 keys) = 8 bytes/element total (Double Buffered)
    l2_limit_pairs = 2_000_000 
    l2_limit_keys = 4_000_000  

    plt.axvline(x=l2_limit_pairs, color='navy', linestyle=':', alpha=0.4)
    plt.text(l2_limit_pairs * 0.85, 0.5, 'L2 Limit (Pairs)', rotation=90, color='navy', fontsize=9, alpha=0.6)
    
    plt.axvline(x=l2_limit_keys, color='darkred', linestyle=':', alpha=0.4)
    plt.text(l2_limit_keys * 1.1, 0.5, 'L2 Limit (Keys)', rotation=90, color='darkred', fontsize=9, alpha=0.6)

    # Formatting
    plt.title('Mojo GPU Sort: OneSweep vs Radix (RTX 5060 Ti)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Input Size (N)', fontsize=12)
    plt.ylabel('Throughput (GK/s)', fontsize=12)
    
    unique_n = df["N"].unique().sort().to_list()
    plt.xticks(unique_n, [format_size(n) for n in unique_n])
    
    plt.legend(loc='upper left', frameon=True, shadow=True)

    # 10 GK/s Breakthrough Line
    plt.axhline(y=10.0, color='gold', linestyle='--', alpha=0.5, lw=1)
    plt.text(unique_n[0], 10.2, '10 GK/s Barrier', color='#997a00', fontweight='bold')

    plt.ylim(0, df["Throughput_GKs"].max() * 1.15)
    plt.tight_layout()
    
    output_filename = "gpu_sort_benchmark_plot.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Graph successfully saved as {output_filename}")
    plt.show()

if __name__ == "__main__":
    plot_results()