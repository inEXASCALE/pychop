import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

# Create output directory if needed
os.makedirs("figures", exist_ok=True)

# Load results
data = np.load("quantize_results.npz", allow_pickle=True)
results = data["results"].item()

# Extract sizes and formats
sizes = sorted(results.keys())
formats = ["binary16", "bfloat16"]

# Prepare time data: format → library → backend → list of times
times = {
    "pychop":   {fmt: {} for fmt in formats},
    "gfloat":   {fmt: {} for fmt in formats},
}

for fmt in formats:
    for size in sizes:
        res = results[size][fmt]
        for lib in ["pychop", "gfloat"]:
            for backend in ["numpy", "jax", "torch"]:
                key = f"{lib}_{backend}"
                if key in res:
                    times[lib][fmt].setdefault(backend, []).append(res[key]["time"])

# ────────────────────────────────────────────────
# Time comparison: one figure per format, all backends shown
# ────────────────────────────────────────────────

fontsize = 15
linewidth = 1.8
markers = ['o', 's', '^', 'D', 'v', 'P']
linestyles = ['-', '--', '-.']

backend_labels = {
    "numpy": "NumPy",
    "jax":   "JAX",
    "torch": "PyTorch"
}

for fmt in formats:
    short_name = "fp16" if fmt == "binary16" else "bf16"
    
    plt.figure(figsize=(10, 6))
    
    lines = []
    labels = []
    
    for i, backend in enumerate(["numpy", "jax", "torch"]):
        # PyChop
        if backend in times["pychop"][fmt]:
            line, = plt.plot(
                sizes, times["pychop"][fmt][backend],
                marker=markers[i], linestyle=linestyles[0],
                linewidth=linewidth, markersize=7,
                label=f"PyChop {backend_labels[backend]}"
            )
            lines.append(line)
            labels.append(f"PyChop {backend_labels[backend]}")
        
        # gfloat
        if backend in times["gfloat"][fmt]:
            line, = plt.plot(
                sizes, times["gfloat"][fmt][backend],
                marker=markers[i+3], linestyle=linestyles[1],
                linewidth=linewidth, markersize=7, alpha=0.9,
                label=f"gfloat {backend_labels[backend]}"
            )
            lines.append(line)
            labels.append(f"gfloat {backend_labels[backend]}")
    
    plt.xlabel("Matrix Size", fontsize=fontsize)
    plt.ylabel("Average Quantization Time (s)", fontsize=fontsize)
    #plt.title(f"Quantization Performance — {short_name}", fontsize=fontsize + 2)
    plt.xticks(sizes, fontsize=fontsize-1)
    plt.yticks(fontsize=fontsize-1)
    plt.grid(True, alpha=0.25, linestyle='--')
    
    # Put legend outside to avoid crowding
    plt.legend(lines, labels,
               loc='upper left', bbox_to_anchor=(1.02, 1),
               fontsize=fontsize-1, frameon=True)
    
    plt.tight_layout()
    plt.savefig(f"figures/time_comparison_{short_name}.png", dpi=160, bbox_inches='tight')
    plt.close()

# ────────────────────────────────────────────────
# Relative error heatmap (PyChop numpy vs gfloat numpy)
# ────────────────────────────────────────────────

label_map = {"binary16": "fp16", "bfloat16": "bf16"}

err_matrix = np.zeros((len(sizes), len(formats)))

for i, size in enumerate(sizes):
    for j, fmt in enumerate(formats):
        err_matrix[i, j] = results[size][fmt].get("rel_error_numpy", np.nan)

plt.figure(figsize=(5, 6))
sns.heatmap(err_matrix,
            annot=True,
            fmt=".2e",
            xticklabels=[label_map[f] for f in formats],
            yticklabels=sizes,
            cmap="magma_r",
            linewidths=0.8,
            cbar_kws={'label': 'Relative Error'})

plt.xlabel("Format", fontsize=fontsize)
plt.ylabel("Matrix Size", fontsize=fontsize)
plt.title("Relative Error\n(PyChop numpy vs gfloat numpy)", fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize, rotation=0)
plt.tight_layout()
plt.savefig("figures/relative_error_heatmap.png", dpi=160)
plt.close()

print("Saved figures:")
print("  figures/time_comparison_fp16.png")
print("  figures/time_comparison_bf16.png")
print("  figures/relative_error_heatmap.png")
