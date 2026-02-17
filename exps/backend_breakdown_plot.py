import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", font_scale=1.2)
os.makedirs("results", exist_ok=True)

arr_sizes = [2000, 4000, 6000, 8000, 10000]
rounding_modes = ["Nearest (even)","Up","Down","Zero","Stochastic (prop)","Stochastic (uniform)"]
operations = ["quantize_only"]
metrics = ["time", "throughput"]
metric_names = {"time": "Runtime (seconds)", "throughput": "Throughput (G elements/s)"}
y_scales = {"time": "log", "throughput": "linear"}

backend_types = ["cpu", "gpu"]
backend_labels = {
    "numpy": "NumPy",
    "torch_cpu": "GPU, PyTorch",
    "jax_eager": "GPU, JAX",
    "jax_jit": "GPU, JAX (JIT)",
    "torch_gpu": "GPU, PyTorch"
}


for op in operations:
    for mode in rounding_modes:
        safe_mode = mode.replace(" ", "_").replace("(", "").replace(")", "")
        for metric in metrics:
            plt.figure(figsize=(8, 6))
            files_exist = False  

            for btype in backend_types:
                csv_file = f"results/{op}_{safe_mode}_{btype}_{metric}.csv"
                if not os.path.exists(csv_file):
                    print(f"警告：文件 {csv_file} 不存在，跳过")
                    continue

                df = pd.read_csv(csv_file, index_col=0)
                df.index = arr_sizes
                files_exist = True

                for col in df.columns:
                    label = backend_labels.get(col, col)
                    plt.plot(df.index, df[col], marker='o', label=label)

            if not files_exist:
                print(f"警告：没有找到任何 {metric} 文件，跳过绘图")
                plt.close()
                continue

            plt.title(f"{op.replace('_', ' ').capitalize()} - {mode.capitalize()} - {metric_names[metric]}")
            plt.xlabel("Matrix Size (N x N)")
            plt.ylabel(metric_names[metric])
            plt.yscale(y_scales[metric])
            plt.legend(loc='upper left', framealpha=0.0)
            plt.grid(True, which="both", ls="--")
            plt.tight_layout()

            # 保存图表
            plt.savefig(f"results/{op}_{safe_mode}_{metric}.png")
            plt.close()

print("\n可视化完成！图表已保存到 results/ 目录。")
