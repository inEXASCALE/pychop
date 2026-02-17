import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

# Create output directory if it does not exist
os.makedirs("figures", exist_ok=True)

# Load previously saved results (make sure the filename matches your saved file)
data = np.load("quantize_results.npz", allow_pickle=True)
results = data["results"].item()  # loaded as a dictionary

# Extract matrix sizes (sorted for consistent plotting)
sizes = sorted(results.keys())  # e.g. [2000, 4000, 6000, 8000, 10000]

# Define formats (you can also extract them dynamically if preferred)
formats = ["binary16", "bfloat16"]

# Prepare time dictionaries: format → backend → list of times across sizes
times_pychop = {fmt: {} for fmt in formats}
times_gfloat = {fmt: {} for fmt in formats}

for fmt in formats:
    for size in sizes:
        res = results[size][fmt]
        
        # We use numpy backend for the fairest comparison
        times_pychop[fmt].setdefault("numpy", []).append(res["pychop_numpy"]["time"])
        times_gfloat[fmt].setdefault("numpy", []).append(res["gfloat_numpy"]["time"])

        # Optional: collect other backends if you want to plot them later
        # times_pychop[fmt].setdefault("jax", []).append(res["pychop_jax"]["time"])
        # times_gfloat[fmt].setdefault("jax", []).append(res["gfloat_jax"]["time"])

# ────────────────────────────────────────────────
# Time comparison plots — one figure per format (bf16 and fp16 separately)
# ────────────────────────────────────────────────

fontsize = 14
linewidth = 2
markers = ['o', 's', '^']
linestyles = ['-', '--']

# Which backends to include in the plot (currently only numpy)
backends_to_plot = ["numpy"]

for fmt in ["bfloat16", "binary16"]:
    label_map = {"bfloat16": "bf16", "binary16": "fp16"}
    short_name = label_map[fmt]
    
    plt.figure(figsize=(8, 6))
    
    for backend in backends_to_plot:
        # Plot PyChop
        plt.plot(sizes, times_pychop[fmt][backend],
                 marker='o', linestyle='-',
                 linewidth=linewidth, markersize=8,
                 label=f"PyChop ({short_name})")
        
        # Plot gfloat
        plt.plot(sizes, times_gfloat[fmt][backend],
                 marker='s', linestyle='--',
                 linewidth=linewidth, markersize=8, alpha=0.85,
                 label=f"gfloat ({short_name})")
    
    plt.xlabel("Matrix Size", fontsize=fontsize)
    plt.ylabel("Average Time (seconds)", fontsize=fontsize)
    plt.title(f"PyChop vs gfloat — {short_name}", fontsize=fontsize + 2)
    plt.xticks(sizes, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=fontsize, loc="upper left")
    plt.tight_layout()
    
    # Save separate figures for each format
    safe_name = short_name.replace(" ", "_")
    plt.savefig(f"figures/time_comparison_{safe_name}.png", dpi=150)
    plt.close()

# ────────────────────────────────────────────────
# Consistency heatmap (both formats together)
# ────────────────────────────────────────────────

match_matrix = np.zeros((len(sizes), len(formats)))

# Note: If you did not save a "match" field, this will remain all zeros.
# You can add your own consistency check logic here if needed.
for i, size in enumerate(sizes):
    for j, fmt in enumerate(formats):
        # Example: match_matrix[i, j] = int(results[size][fmt].get("match", 0))
        pass

# Only generate the plot if there is any non-zero data
if np.any(match_matrix):
    plt.figure(figsize=(5, 6))
    sns.heatmap(match_matrix,
                annot=True,
                xticklabels=[label_map.get(f, f) for f in formats],
                yticklabels=sizes,
                cmap="Greens",
                cbar=False,
                fmt=".0f",
                linewidths=1)
    plt.xlabel("Format", fontsize=fontsize)
    plt.ylabel("Matrix Size", fontsize=fontsize)
    plt.title("Consistency (1 = match)", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize, rotation=0)
    plt.tight_layout()
    plt.savefig("figures/consistency_heatmap.png", dpi=150)
    plt.close()

# ────────────────────────────────────────────────
# Relative error heatmap (both formats together)
# ────────────────────────────────────────────────

err_matrix = np.zeros((len(sizes), len(formats)))

for i, size in enumerate(sizes):
    for j, fmt in enumerate(formats):
        # Retrieve the relative error saved earlier
        err_matrix[i, j] = results[size][fmt].get("rel_error_numpy", 0.0)

plt.figure(figsize=(5, 6))
sns.heatmap(err_matrix,
            annot=True,
            fmt=".2e",
            xticklabels=[label_map.get(f, f) for f in formats],
            yticklabels=sizes,
            cmap="magma",
            linewidths=1)

plt.xlabel("Format", fontsize=fontsize)
plt.ylabel("Matrix Size", fontsize=fontsize)
plt.title("Relative Error (PyChop vs gfloat)", fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize, rotation=0)
plt.tight_layout()
plt.savefig("figures/relative_error_heatmap.png", dpi=150)
plt.close()

# ────────────────────────────────────────────────
# Print summary of saved figures
# ────────────────────────────────────────────────

print("Figures saved in ./figures/")
print("  - time_comparison_bf16.png")
print("  - time_comparison_fp16.png")
print("  - consistency_heatmap.png      (if consistency data exists)")
print("  - relative_error_heatmap.png")