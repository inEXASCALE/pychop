import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# List of CSV files and their corresponding method names
csv_files = {
    'chop_runtimes_avg.csv': 'MATLAB chop',
    'chop_runtimes_avg_np.csv': 'NumPy LightChop',
    'chop_runtimes_avg_np2.csv': 'NumPy Chop',
    'chop_runtimes_avg_th.csv': 'PyTorch LightChop',
    'chop_runtimes_avg_th2.csv': 'PyTorch Chop',
    'chop_runtimes_avg_th_gpu.csv': 'PyTorch LightChop (GPU)',
    'chop_runtimes_avg_th2_gpu.csv': 'PyTorch Chop (GPU)',

    'pychop_runtimes_avg_np.csv': 'NumPy LightChop',
    'pychop_runtimes_avg_np2.csv': 'NumPy Chop',
    'pychop_runtimes_avg_th.csv': 'PyTorch LightChop',
    'pychop_runtimes_avg_th2.csv': 'PyTorch Chop',
    'pychop_runtimes_avg_th_gpu.csv': 'PyTorch LightChop (GPU)',
    'pychop_runtimes_avg_th2_gpu.csv': 'PyTorch Chop (GPU)',
}


# Base path to CSV files (adjust if needed)
base_path = 'results/'

# Load all CSV files into a dictionary of DataFrames
dataframes = {method: pd.read_csv(base_path + file) for file, method in csv_files.items()}

# Define sizes explicitly as powers of 2: 2^[8, 9, 10, 11, 12, 13, 14]
sizes = [256, 512, 1024, 2048, 4096, 8192]
exponents = [8, 9, 10, 11, 12, 13]  # Corresponding exponents for 2^n

# Indices for equal spacing (0, 1, 2, 3, 4)
x_indices = range(len(sizes))

# Rounding modes (column names excluding 'Size')
rounding_modes = dataframes['MATLAB chop'].columns[1:].tolist()  # ['Nearest (even)', 'Up', ...]

# Reference method for ratio (MATLAB chop)
reference_method = 'MATLAB chop'

# Create a separate plot for each rounding mode
for mode in rounding_modes:
    plt.figure(figsize=(10, 8))
    
    # Get the reference runtime (MATLAB chop) for this rounding mode
    reference_runtimes = dataframes[reference_method][mode].values
    
    # Plot each method's runtime ratio for this rounding mode with semilogy
    for method, df in dataframes.items():
        if method != reference_method:  # Skip the reference method itself
            # Compute ratios as reference / method, avoiding division by zero or invalid values
            runtime_ratios = np.where(df[mode].values != 0, 
                                    reference_runtimes / df[mode].values, 
                                    np.nan)  # Use NaN for invalid cases
            plt.semilogy(x_indices, runtime_ratios, marker='o', label=method)
    
    # Customize the plot
    plt.title(f'Runtime Ratio vs Matrix Size\nRounding Mode: {mode}', fontsize=14)
    plt.xlabel('Matrix Size (n x n)', fontsize=15)
    plt.ylabel(f'Runtime Ratio ({reference_method} / Method)', fontsize=15)
    
    # Set x-ticks with labels in the form of 2^exponent
    plt.xticks(x_indices, [f'$2^{{{exp}}}$' for exp in exponents])
    
    # Enable grid for both major and minor ticks on the y-axis
    plt.grid(True, which="both", ls="--")
    plt.legend(title='Method', fontsize=15)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot as PDF with tight bounding box
    plt.savefig(f'results/runtime_ratio_comparison_{mode.replace(" ", "_")}_b.pdf', 
                format='pdf', 
                bbox_inches='tight')
    
    # Close the figure to free memory
    plt.close()

print("Semilogy PDF plots with runtime ratios (MATLAB chop / Method) generated for all rounding modes with base-2 exponent x-labels and tight bounding box.")