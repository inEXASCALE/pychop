import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_files = {
    'chop_runtimes_avg_b.csv': 'chop (MATLAB)',
    'chop_runtimes_avg_np_b.csv': 'Chop (MATLAB, NumPy backend)',
    'chop_runtimes_avg_np2_b.csv': 'FaultChop (MATLAB, NumPy backend)',
    'chop_runtimes_avg_th_b.csv': 'Chop (MATLAB, PyTorch backend)',
    'chop_runtimes_avg_th2_b.csv': 'FaultChop (MATLAB, PyTorch backend)',
    'chop_runtimes_avg_th_gpu_b.csv': 'Chop (MATLAB, GPU, PyTorch backend)',
    'chop_runtimes_avg_th2_gpu_b.csv': 'FaultChop (MATLAB, GPU, PyTorch backend)',
    'pychop_runtimes_avg_np_b.csv': 'Chop (Python, NumPy backend)',
    'pychop_runtimes_avg_np2_b.csv': 'FaultChop (Python, NumPy backend)',
    'pychop_runtimes_avg_th_b.csv': 'Chop (Python, PyTorch backend)',
    'pychop_runtimes_avg_th2_b.csv': 'FaultChop (Python, PyTorch backend)',
    'pychop_runtimes_avg_th_gpu_b.csv': 'Chop (Python, GPU, PyTorch backend)',
    'pychop_runtimes_avg_th2_gpu_b.csv': 'FaultChop (Python, GPU, PyTorch backend)',
}

base_path = 'results_backup/'

dataframes = {method: pd.read_csv(base_path + file) for file, method in csv_files.items()}

sizes = [2000, 4000, 6000, 8000, 10000]

# Indices for equal spacing (0, 1, 2, 3, 4)
x_indices = range(len(sizes))

# Rounding modes (column names excluding 'Size')
rounding_modes = dataframes['chop (MATLAB)'].columns[1:].tolist()  # ['Nearest (even)', 'Up', ...]
rounding_mode_names = ['Round to nearest', 'Round up', 'Round down', 
                      'Round toward zero', 'Stochastic (prob.)', 'Stochastic (uniform)']

reference_method = 'chop (MATLAB)'

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
          '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78']
markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'h', 'x', '+', 'd', '<']

fig, axes = plt.subplots(2, 3, figsize=(21, 12), sharex=True, sharey=True)
axes = axes.flatten()  # Flatten the 2D array of axes for easier iteration

for idx, mode in enumerate(rounding_modes):
    ax = axes[idx]
    
    # Get the reference runtime for this rounding mode
    reference_runtimes = dataframes[reference_method][mode].values
    
    # Plot each method's runtime ratio with custom styles
    for i, (method, df) in enumerate(dataframes.items()):
        if method != reference_method:  # Skip the reference method itself
            # Compute ratios, avoiding division by zero or invalid values
            runtime_ratios = np.where(df[mode].values != 0, 
                                    reference_runtimes / df[mode].values, 
                                    np.nan)  # Use NaN for invalid cases
            # Use dashed line for methods with "MATLAB" in the name, solid otherwise
            linestyle = '--' if 'MATLAB' in method else '-'
            ax.semilogy(x_indices, runtime_ratios, 
                        linestyle=linestyle, 
                        marker=markers[i % len(markers)], 
                        color=colors[i % len(colors)], 
                        linewidth=2.5, 
                        markersize=8, 
                        label=method)
    
    ax.set_title(f'{rounding_mode_names[idx]}', fontsize=19)
    ax.set_xlabel('Matrix Size', fontsize=19)
    ax.set_ylabel(f'Runtime Ratio', fontsize=19)
    
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f'{size}' for size in sizes], fontsize=19)
    ax.tick_params(axis='both', labelsize=20)
    
    ax.grid(True, which="both", ls="--")

handles, labels = axes[0].get_legend_handles_labels()  
fig.legend(handles, labels, fontsize=20, title_fontsize=20, 
           loc='center', bbox_to_anchor=(0.5, -0.1), 
           framealpha=0, ncol=2)

plt.tight_layout()

plt.savefig('results/runtime_ratio_comparison_subplots_b.pdf', 
            format='pdf', 
            bbox_inches='tight')
plt.savefig('results/runtime_ratio_comparison_subplots_b.jpg', 
            format='jpg', 
            bbox_inches='tight')
# Close the figure to free memory
plt.close()