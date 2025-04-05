import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_files = {
    'chop_runtimes_avg.csv': 'chop (MATLAB)',
    'chop_runtimes_avg_np.csv': 'LightChop (MATLAB, NumPy backend)',
    'chop_runtimes_avg_np2.csv': 'Chop (MATLAB, NumPy backend)',
    'chop_runtimes_avg_th.csv': 'LightChop (MATLAB, PyTorch backend)',
    'chop_runtimes_avg_th2.csv': 'Chop (MATLAB, PyTorch backend)',
    'chop_runtimes_avg_th_gpu.csv': 'LightChop (MATLAB, GPU, PyTorch backend)',
    'chop_runtimes_avg_th2_gpu.csv': 'Chop (MATLAB, GPU, PyTorch backend)',
    'pychop_runtimes_avg_np.csv': 'LightChop (Python, NumPy backend)',
    'pychop_runtimes_avg_np2.csv': 'Chop (Python, NumPy backend)',
    'pychop_runtimes_avg_th.csv': 'LightChop (Python, PyTorch backend)',
    'pychop_runtimes_avg_th2.csv': 'Chop (Python, PyTorch backend)',
    'pychop_runtimes_avg_th_gpu.csv': 'LightChop (Python, GPU, PyTorch backend)',
    'pychop_runtimes_avg_th2_gpu.csv': 'Chop (Python, GPU, PyTorch backend)',
}

# Base path to CSV files (adjust if needed)
base_path = 'results/'

# Load all CSV files into a dictionary of DataFrames
dataframes = {method: pd.read_csv(base_path + file) for file, method in csv_files.items()}

# Define sizes explicitly as powers of 2: 2^[8, 9, 10, 11, 12, 13]
sizes = [256, 512, 1024, 2048, 4096, 8192]
exponents = [8, 9, 10, 11, 12, 13]  # Corresponding exponents for 2^n

# Indices for equal spacing (0, 1, 2, 3, 4, 5)
x_indices = range(len(sizes))

# Rounding modes (column names excluding 'Size')
rounding_modes = dataframes['chop (MATLAB)'].columns[1:].tolist()  # ['Nearest (even)', 'Up', ...]
rounding_mode_names = ['Round to nearest', 'Round up', 'Round down', 
                  'Round toward zero', 'Stochastic (prob.)', 'Stochastic (uniform)']
# print(rounding_modes)

# Reference method for ratio
reference_method = 'chop (MATLAB)'

# Define distinct colors and markers for clarity
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
          '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78']
markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'h', 'x', '+', 'd', '<']

# Create a single figure with 6 subplots (2 rows, 3 columns)
fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharex=True, sharey=True)
axes = axes.flatten()  # Flatten the 2D array of axes for easier iteration

# Plot each rounding mode in its own subplot
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
    
    # Customize each subplot
    ax.set_title(f'{rounding_mode_names[idx]}', fontsize=18)
    ax.set_xlabel('Matrix Size', fontsize=18)
    ax.set_ylabel(f'Runtime Ratio', fontsize=18)
    
    # Set x-ticks with labels in the form of 2^exponent
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f'$2^{{{exp}}}$' for exp in exponents], fontsize=18)
    ax.tick_params(axis='both', labelsize=18)
    
    # Enable grid for both major and minor ticks on the y-axis
    ax.grid(True, which="both", ls="--")

# Add a single legend outside the subplots at the bottom
handles, labels = axes[0].get_legend_handles_labels()  # Get handles and labels from first subplot
fig.legend(handles, labels, fontsize=17, title_fontsize=17, 
           loc='center', bbox_to_anchor=(0.5, -0.1), 
           framealpha=0, ncol=2)

# fig.suptitle('Runtime Ratio vs Matrix Size (half precision)\nReference: chop (MATLAB)', fontsize=18, y=1.05)

plt.tight_layout()

plt.savefig('results/runtime_ratio_comparison_subplots.pdf', 
            format='pdf', 
            bbox_inches='tight')

# Close the figure to free memory
plt.close()
