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
formats = ["bfloat16", "binary16"]   # order: bf16 left, fp16 right

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
# Combined time comparison: side-by-side (left: bf16, right: fp16)
# shared legend placed at the bottom center
# ────────────────────────────────────────────────

fontsize = 14
linewidth = 1.9
markers = ['o', 's', '^', 'D', 'v', 'P']
linestyles = ['-', '--', '-.']
backend_labels = {
    "numpy": "NumPy",
    "jax":   "JAX",
    "torch": "PyTorch"
}

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# ── Left: bfloat16
fmt = "bfloat16"
short_name = "bf16"
for i, backend in enumerate(["numpy", "jax", "torch"]):
    # PyChop
    if backend in times["pychop"][fmt]:
        ax_left.semilogy(
            sizes, times["pychop"][fmt][backend],
            marker=markers[i], linestyle=linestyles[0],
            linewidth=linewidth, markersize=8,
            label=f"PyChop {backend_labels[backend]}"
        )
    # gfloat
    if backend in times["gfloat"][fmt]:
        ax_left.semilogy(
            sizes, times["gfloat"][fmt][backend],
            marker=markers[i+3], linestyle=linestyles[1],
            linewidth=linewidth, markersize=8, alpha=0.92,
            label=f"gfloat {backend_labels[backend]}"
        )

ax_left.set_title(f"Quantization Performance — {short_name}", fontsize=fontsize + 1)
ax_left.set_xlabel("Matrix Size", fontsize=fontsize)
ax_left.set_ylabel("Average Time (seconds)", fontsize=fontsize)
ax_left.tick_params(axis='both', labelsize=fontsize-1)
ax_left.grid(True, which="both", alpha=0.3, linestyle='--')
ax_left.set_xticks(sizes)

# ── Right: binary16 / fp16
fmt = "binary16"
short_name = "fp16"
for i, backend in enumerate(["numpy", "jax", "torch"]):
    # PyChop
    if backend in times["pychop"][fmt]:
        ax_right.semilogy(
            sizes, times["pychop"][fmt][backend],
            marker=markers[i], linestyle=linestyles[0],
            linewidth=linewidth, markersize=8,
            label=f"PyChop {backend_labels[backend]}"
        )
    # gfloat
    if backend in times["gfloat"][fmt]:
        ax_right.semilogy(
            sizes, times["gfloat"][fmt][backend],
            marker=markers[i+3], linestyle=linestyles[1],
            linewidth=linewidth, markersize=8, alpha=0.92,
            label=f"gfloat {backend_labels[backend]}"
        )

ax_right.set_title(f"Quantization Performance — {short_name}", fontsize=fontsize + 1)
ax_right.set_xlabel("Matrix Size", fontsize=fontsize)
# No ylabel on right since sharey=True
ax_right.tick_params(axis='both', labelsize=fontsize-1)
ax_right.grid(True, which="both", alpha=0.3, linestyle='--')
ax_right.set_xticks(sizes)

# ── Shared legend at bottom center
handles, labels = ax_left.get_legend_handles_labels()  # or ax_right, same content
fig.legend(handles, labels,
           loc='lower center',
           bbox_to_anchor=(0.5, -0.02),   # slightly below the figure
           ncol=3,                        # 3 columns to make it compact
           fontsize=fontsize-1,
           frameon=True)

# Adjust layout to make room for the bottom legend
plt.tight_layout(rect=[0, 0.08, 1, 0.98])  # leave space at bottom
plt.savefig("figures/time_comparison_side_by_side_semilogy.png", dpi=160, bbox_inches='tight')
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
print("  figures/time_comparison_side_by_side_semilogy.png   ← bf16 (left) + fp16 (right)")
print("  figures/relative_error_heatmap.png")
