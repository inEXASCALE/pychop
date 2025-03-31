import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# List of CSV files and their corresponding method names
csv_files = {
    'chop_runtimes_avg_b.csv': 'chop (MATLAB)',
    'chop_runtimes_avg_np_b.csv': 'LightChop (MATLAB, NumPy backend)',
    'chop_runtimes_avg_np2_b.csv': 'Chop (MATLAB, NumPy backend)',
    'chop_runtimes_avg_th_b.csv': 'LightChop (MATLAB, PyTorch backend)',
    'chop_runtimes_avg_th2_b.csv': 'Chop (MATLAB, PyTorch backend)',
    'chop_runtimes_avg_th_gpu_b.csv': 'LightChop (MATLAB, GPU, PyTorch backend)',
    'chop_runtimes_avg_th2_gpu_b.csv': 'Chop (MATLAB, GPU, PyTorch backend)',
    'pychop_runtimes_avg_np_b.csv': 'LightChop (Python, NumPy backend)',
    'pychop_runtimes_avg_np2_b.csv': 'Chop (Python, NumPy backend)',
    'pychop_runtimes_avg_th_b.csv': 'LightChop (Python, PyTorch backend)',
    'pychop_runtimes_avg_th2_b.csv': 'Chop (Python, PyTorch backend)',
    'pychop_runtimes_avg_th_gpu_b.csv': 'LightChop (Python, GPU, PyTorch backend)',
    'pychop_runtimes_avg_th2_gpu_b.csv': 'Chop (Python, GPU, PyTorch backend)',
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

# Reference method for ratio
reference_method = 'chop (MATLAB)'

# Define distinct colors and markers for clarity
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
          '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78']
markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'h', 'x', '+', 'd', '<']

# Create a separate plot for each rounding mode
for mode in rounding_modes:
    plt.figure(figsize=(10, 9))
    
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
            plt.semilogy(x_indices, runtime_ratios, 
                        linestyle=linestyle, 
                        marker=markers[i % len(markers)], 
                        color=colors[i % len(colors)], 
                        linewidth=2.5, 
                        markersize=8, 
                        label=method)
    
    # Customize the plot
    plt.title(f'Runtime Ratio vs Matrix Size (bfloat16 precision)\nRounding Mode: {mode}', fontsize=18)
    plt.xlabel('Matrix Size', fontsize=16)
    plt.ylabel(f'Runtime Ratio ({reference_method} / Pychop Methods)', fontsize=16)
    
    # Set x-ticks with labels in the form of 2^exponent and increase tick size
    plt.xticks(x_indices, [f'$2^{{{exp}}}$' for exp in exponents], fontsize=14)
    plt.yticks(fontsize=14)
    
    # Enable grid for both major and minor ticks on the y-axis
    plt.grid(True, which="both", ls="--")
    
    # Add legend with transparency, centered outside, and larger font size
    plt.legend(fontsize=14, title_fontsize=16, 
              loc='center', bbox_to_anchor=(0.5, -0.4), 
              framealpha=0, ncol=2)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot as PDF with tight bounding box
    plt.savefig(f'results/runtime_ratio_comparison_{mode.replace(" ", "_")}_b.pdf', 
                format='pdf', 
                bbox_inches='tight')
    
    # Close the figure to free memory
    plt.close()

print("Semilogy PDF plots with runtime ratios (MATLAB chop / Method) generated for all rounding modes with base-2 exponent x-labels, distinct styles, and tight bounding box.")