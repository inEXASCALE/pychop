import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

os.makedirs("figures", exist_ok=True)

data = np.load("results.npz", allow_pickle=True)

sizes = data["sizes"]
formats = data["formats"]
times_pychop = data["times_pychop"].item()
times_gfloat = data["times_gfloat"].item()
results_match = data["results_match"].item()
rel_errors = data["rel_errors"].item()

fontsize = 14
linewidth = 2
markers = ['o', 's', '^', 'D', 'x', '*', 'v']
linestyles = ['-', '--', '-.', ':']

plt.figure(figsize=(8,6))
label_map = {
    "bfloat16": "bf16",
    "binary16": "fp16"
}

for i, fmt in enumerate(formats):
    marker = markers[i % len(markers)]
    linestyle = linestyles[i % len(linestyles)]
    # Pychop
    plt.plot(sizes, times_pychop[fmt],
             marker=marker, linestyle=linestyle,
             linewidth=linewidth,
             markersize=8,
             label=f"Pychop ({label_map.get(fmt, fmt)})")
    
    # gfloat
    plt.plot(sizes, times_gfloat[fmt],
             marker=marker, linestyle=linestyle,
             alpha=0.6,
             linewidth=linewidth,
             markersize=8,
             label=f"gfloat ({label_map.get(fmt, fmt)})")

plt.xlabel("Matrix Size", fontsize=fontsize)
plt.ylabel("Average Time (seconds)", fontsize=fontsize)
plt.title("Pychop vs gfloat", fontsize=fontsize)
plt.xticks(sizes, fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.grid(True)
plt.legend(fontsize=fontsize)
plt.tight_layout()
plt.savefig("figures/time_comparison.png")
plt.close()

match_matrix = np.zeros((len(sizes), len(formats)))

for i in range(len(sizes)):
    for j, fmt in enumerate(formats):
        match_matrix[i, j] = int(results_match[fmt][i])

plt.figure(figsize=(6,6))
sns.heatmap(match_matrix,
            annot=True,
            xticklabels=formats,
            yticklabels=sizes,
            cmap="Greens",
            cbar=False,
            fmt=".0f",
            linewidths=1)

plt.xlabel("Format", fontsize=fontsize)
plt.ylabel("Matrix Size", fontsize=fontsize)
plt.title("Consistency (1=True)", fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize, rotation=0)
plt.tight_layout()
plt.savefig("figures/consistency_heatmap.png")
plt.close()

err_matrix = np.zeros((len(sizes), len(formats)))

for i in range(len(sizes)):
    for j, fmt in enumerate(formats):
        err_matrix[i, j] = rel_errors[fmt][i]

plt.figure(figsize=(6,6))
sns.heatmap(err_matrix,
            annot=True,
            xticklabels=formats,
            yticklabels=sizes,
            cmap="magma",
            linewidths=1)

plt.xlabel("Format", fontsize=fontsize)
plt.ylabel("Matrix Size", fontsize=fontsize)
plt.title("Relative Error", fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize, rotation=0)
plt.tight_layout()
plt.savefig("figures/relative_error_heatmap.png")
plt.close()

print("Figures saved in ./figures/")
