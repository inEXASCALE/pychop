import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

files = {
    'MATLAB chop': 'results/chop_runtimes_avg.csv',
    'NumPy LightChop': 'results/chop_runtimes_avg_np.csv',
    'PyTorch LightChop': 'results/chop_runtimes_avg_th.csv',
    'NumPy Chop': 'results/chop_runtimes_avg_np2.csv',
    'PyTorch Chop': 'results/chop_runtimes_avg_th2.csv'
}

rounding_modes = [
    'Nearest (even)', 'Up', 'Down', 'Zero', 
    'Stochastic (prop)', 'Stochastic (uniform)'
]

dataframes = {}
for impl, file in files.items():
    if os.path.exists(file):
        df = pd.read_csv(file)
        dataframes[impl] = df.set_index('Size')
    else:
        print(f"Warning: File {file} not found. Skipping {impl}.")

# Function to compare speeds across rounding modes for a given implementation
def compare_rounding_speeds(df, implementation):
    print(f"\n=== {implementation} ===")
    for size in df.index:
        times = df.loc[size, rounding_modes]
        fastest_mode = times.idxmin()
        slowest_mode = times.idxmax()
        fastest_time = times.min()
        slowest_time = times.max()
        print(f"Size {size}:")
        print(f"  Fastest: {fastest_mode} ({fastest_time:.6f}s)")
        print(f"  Slowest: {slowest_mode} ({slowest_time:.6f}s)")
        print(f"  Difference: {(slowest_time - fastest_time):.6f}s "
              f"({(slowest_time / fastest_time - 1) * 100:.2f}% slower)")

# Function to plot runtimes for a specific size across all implementations
def plot_runtimes_by_size(dfs, size):
    plt.figure(figsize=(12, 6))
    for impl, df in dfs.items():
        if size in df.index:
            times = df.loc[size, rounding_modes]
            plt.plot(rounding_modes, times, marker='o', label=impl)
    plt.title(f'Runtimes for Size {size}x{size}')
    plt.xlabel('Rounding Mode')
    plt.ylabel('Average Runtime (seconds)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_runtimes_by_mode(dfs, mode):
    plt.figure(figsize=(12, 6))
    for impl, df in dfs.items():
        times = df[mode]
        plt.plot(df.index, times, marker='o', label=impl)
    plt.title(f'Runtimes for {mode} Rounding Mode')
    plt.xlabel('Matrix Size')
    plt.ylabel('Average Runtime (seconds)')
    plt.xscale('log', base=2)  # Log scale for sizes (64, 256, 1024, etc.)
    plt.xticks(df.index, df.index)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

for impl, df in dataframes.items():
    compare_rounding_speeds(df, impl)

plot_size = 1024  # Adjust as needed (e.g., 64, 256, 1024, 4096, 16384)
plot_runtimes_by_size(dataframes, plot_size)

plot_mode = 'Nearest (even)'  # Adjust as needed
plot_runtimes_by_mode(dataframes, plot_mode)

# Aggregate statistics across all sizes
print("\n=== Aggregate Statistics Across All Sizes ===")
for impl, df in dataframes.items():
    avg_times = df[rounding_modes].mean()
    print(f"\n{impl} Average Runtimes:")
    for mode, time in avg_times.items():
        print(f"  {mode}: {time:.6f}s")
    fastest_mode = avg_times.idxmin()
    slowest_mode = avg_times.idxmax()
    print(f"  Fastest overall: {fastest_mode} ({avg_times[fastest_mode]:.6f}s)")
    print(f"  Slowest overall: {slowest_mode} ({avg_times[slowest_mode]:.6f}s)")