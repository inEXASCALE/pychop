import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", font_scale=1.2)
os.makedirs("results", exist_ok=True)

arr_sizes = [2000, 4000, 6000, 8000, 10000]
rounding_modes = ["Nearest (even)", "Up", "Down", "Zero", "Stochastic (prop)", "Stochastic (uniform)"]
rounding_mode_names = [
    'Round to nearest', 
    'Round up', 
    'Round down', 
    'Round toward zero', 
    'Stochastic (prob.)', 
    'Stochastic (uniform)'
]

operations = ["quantize_only"]
metrics = ["time", "throughput"]
metric_names = {"time": "Runtime (seconds)", "throughput": "Throughput (G elements/s)"}
y_scales = {"time": "log", "throughput": "log"}

backend_types = ["cpu", "gpu"]
backend_labels = {
    "numpy": "NumPy",
    "torch_cpu": "GPU, PyTorch",
    "jax_eager": "GPU, JAX",
    "jax_jit": "GPU, JAX (JIT)",
    "torch_gpu": "GPU, PyTorch"
}

backend_styles = {
    "numpy": {"marker": "o", "linestyle": "-", "linewidth": 2},
    "torch_cpu": {"marker": "s", "linestyle": "--", "linewidth": 2},
    "jax_eager": {"marker": "^", "linestyle": "-.", "linewidth": 2},
    "jax_jit": {"marker": "v", "linestyle": ":", "linewidth": 2},
    "torch_gpu": {"marker": "D", "linestyle": "-", "linewidth": 2}
}

backend_colors = {
    "numpy": "tab:blue",
    "torch_cpu": "tab:orange",
    "jax_eager": "tab:green",
    "jax_jit": "tab:red",
    "torch_gpu": "tab:purple"
}

fontsize = 15

for op in operations:
    for i, mode in enumerate(rounding_modes):
        safe_mode = mode.replace(" ", "_").replace("(", "").replace(")", "")
        readable_mode = rounding_mode_names[i]

        for metric in metrics:
            plt.figure(figsize=(8, 6))
            files_exist = False  

            for btype in backend_types:
                csv_file = f"results/{op}_{safe_mode}_{btype}_{metric}.csv"
                if not os.path.exists(csv_file):
                    print(f"Warning: did not find {csv_file}, skipping")
                    continue

                df = pd.read_csv(csv_file, index_col=0)
                df.index = arr_sizes
                files_exist = True

                for col in df.columns:
                    label = backend_labels.get(col, col)
                    style = backend_styles.get(col, {"marker": "o", "linestyle": "-", "linewidth": 3})
                    plt.plot(
                        df.index,
                        df[col],
                        marker=style["marker"],
                        linestyle=style["linestyle"],
                        linewidth=style["linewidth"],
                        color=backend_colors.get(col, "black"),
                        label=label,
                        markersize=8
                    )

            if not files_exist:
                print(f"Warning: did not find any {metric} files, skipping visualization")
                plt.close()
                continue

            plt.title(f"{readable_mode}", fontsize=fontsize)
            plt.xlabel("Matrix Size", fontsize=fontsize)
            plt.ylabel(metric_names[metric], fontsize=fontsize)
            plt.yscale(y_scales[metric])

            plt.legend(loc='upper left', framealpha=0.0, fontsize=fontsize)

            plt.xticks(arr_sizes, arr_sizes, fontsize=fontsize)
            plt.yticks(fontsize=fontsize)

            plt.grid(True, which="both", ls="--")
            plt.tight_layout()

            plt.savefig(f"figures/{op}_{safe_mode}_{metric}.png")
            plt.close()

print("\nCompleted! Plots are saved in the 'results/' folder.")
