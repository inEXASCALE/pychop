import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

# Create output directory if needed
os.makedirs("figures", exist_ok=True)

# Load results
data = np.load("quantize_results.npz", allow_pickle=True)
results = data["results"].item()

# Extract sizes and formats (bf16 first, then fp16)
sizes = sorted(results.keys())
formats = ["bfloat16", "binary16"]   # order: bf16 first, then fp16

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
# Combined time comparison: one figure with both precisions (bf16 top, fp16 bottom)
# shared legend, semilogy scale
# ────────────────────────────────────────────────

fontsize = 15
linewidth = 1.9
markers = ['o', 's', '^', 'D', 'v', 'P']
linestyles = ['-', '--', '-.']
backend_labels = {
    "numpy": "NumPy",
    "jax":   "JAX",
    "torch": "PyTorch"
}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10), sharex=True)

# ── Plot 1: bfloat16 (top)
fmt = "bfloat16"
short_name = "bf16"
for i, backend in enumerate(["numpy", "jax", "torch"]):
    # PyChop
    if backend in times["pychop"][fmt]:
        ax1.semilogy(
            sizes, times["pychop"][fmt][backend],
            marker=markers[i], linestyle=linestyles[0],
            linewidth=linewidth, markersize=8,
            label=f"PyChop {backend_labels[backend]}"
        )
    # gfloat
    if backend in times["gfloat"][fmt]:
        ax1.semilogy(
            sizes, times["gfloat"][fmt][backend],
            marker=markers[i+3], linestyle=linestyles[1],
            linewidth=linewidth, markersize=8, alpha=0.92,
            label=f"gfloat {backend_labels[backend]}"
        )

ax1.set_title(f"Quantization Performance — {short_name} (log scale)", fontsize=fontsize + 1)
ax1.set_ylabel("Average Time (seconds)", fontsize=fontsize)
ax1.tick_params(axis='both', labelsize=fontsize-1)
ax1.grid(True, which="both", alpha=0.3, linestyle='--')

# ── Plot 2: binary16 / fp16 (bottom)
fmt = "binary16"
short_name = "fp16"
for i, backend in enumerate(["numpy", "jax", "torch"]):
    # PyChop
    if backend in times["pychop"][fmt]:
        ax2.semilogy(
            sizes, times["pychop"][fmt][backend],
            marker=markers[i], linestyle=linestyles[0],
            linewidth=linewidth, markersize=8,
            label=f"PyChop {backend_labels[backend]}"
        )
    # gfloat
    if backend in times["gfloat"][fmt]:
        ax2.semilogy(
            sizes, times["gfloat"][fmt][backend],
            marker=markers[i+3], linestyle=linestyles[1],
            linewidth=linewidth, markersize=8, alpha=0.92,
            label=f"gfloat {backend_labels[backend]}"
        )

ax2.set_title(f"Quantization Performance — {short_name} (log scale)", fontsize=fontsize + 1)
ax2.set_xlabel("Matrix Size (N×N)", fontsize=fontsize)
ax2.set_ylabel("Average Time (seconds)", fontsize=fontsize)
ax2.tick_params(axis='both', labelsize=fontsize-1)
ax2.grid(True, which="both", alpha=0.3, linestyle='--')
ax2.set_xticks(sizes)

# Shared legend (placed outside, at the top-right)
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels,
           loc='upper right',
           bbox_to_anchor=(0.98, 0.98),
           fontsize=fontsize-1,
           frameon=True,
           ncol=2)

plt.tight_layout(rect=[0, 0, 0.98, 0.96])  # leave space for legend
plt.savefig("figures/time_comparison_combined_semilogy.png", dpi=160, bbox_inches='tight')
plt.close()

# ────────────────────────────────────────────────
# Relative error heatmap (unchanged)
# ────────────────────────────────────────────────

label_map = {"binary16": "fp16", "bfloat16": "bf16"}
err_matrix = np.zeros((len(sizes), len(formats)))

for i, size in enumerate(sizes):
    for j, fmt in enumerate(formats):
        err_matrix[i, j] = results[size][fmt].get("rel_error_numpy", np.nan)

plt.figure(figsize=(6, 6))
sns.heatmap(
    err_matrix,
    annot=True,
    fmt=".2e",
    xticklabels=[label_map[f] for f in formats],
    yticklabels=sizes,
    cmap="magma_r",
    linewidths=0.8,
    cbar_kws={'label': 'Relative Error'}
)

plt.xlabel("Format", fontsize=fontsize)
plt.ylabel("Matrix Size", fontsize=fontsize)
plt.title("Relative Error\n(PyChop numpy vs gfloat numpy)", fontsize=fontsize + 1)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize, rotation=0)
plt.tight_layout()
plt.savefig("figures/relative_error_heatmap.png", dpi=160)
plt.close()

print("Saved figures:")
print("  figures/time_comparison_combined_semilogy.png   ← combined bf16 + fp16")
print("  figures/relative_error_heatmap.png")
