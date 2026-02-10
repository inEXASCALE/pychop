import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置风格
sns.set(style="whitegrid", font_scale=1.2)
os.makedirs("results", exist_ok=True)

# 参数（与 benchmark 一致）
arr_sizes = [2000, 4000, 6000, 8000, 10000]
rounding_modes = ["nearest", "stoc_prop"]
operations = ["quantize_only", "quantize_matmul"]
metrics = ["time", "throughput"]
metric_names = {"time": "Runtime (seconds)", "throughput": "Throughput (G elements/s)"}
y_scales = {"time": "log", "throughput": "linear"}

# 加载并画图
for op in operations:
    for mode in rounding_modes:
        for metric in metrics:
            csv_file = f"results/{op}_{mode}_{metric}.csv"
            if not os.path.exists(csv_file):
                print(f"警告：文件 {csv_file} 不存在，跳过")
                continue
            
            df = pd.read_csv(csv_file, index_col=0)
            df.index = arr_sizes
            
            plt.figure(figsize=(10, 6))
            for col in df.columns:
                plt.plot(df.index, df[col], marker='o', label=col)
            
            plt.title(f"{op.replace('_', ' ').capitalize()} - {mode.capitalize()} - {metric_names[metric]}")
            plt.xlabel("Matrix Size (N x N)")
            plt.ylabel(metric_names[metric])
            plt.yscale(y_scales[metric])
            plt.legend(title="Backend")
            plt.grid(True, which="both", ls="--")
            
            plt.tight_layout()
            plt.savefig(f"results/{op}_{mode}_{metric}.png")
            plt.close()

print("\n可视化完成！图表已保存到 results/ 目录。")