import pandas as pd
import matplotlib.pyplot as plt

# List of CSV files and their corresponding method names
csv_files = {
    'chop_runtimes_avg.csv': 'MATLAB chop',
    'chop_runtimes_avg_np.csv': 'PyTorch LightChop',
    'chop_runtimes_avg_th.csv': 'NumPy LightChop',
    'chop_runtimes_avg_np2.csv': 'PyTorch Chop',
    'chop_runtimes_avg_th2.csv': 'NumPy Chop'
}

# Base path to CSV files (adjust if needed)
base_path = 'results/'

# Load all CSV files into a dictionary of DataFrames
dataframes = {method: pd.read_csv(base_path + file) for file, method in csv_files.items()}

# Define sizes explicitly as per 2.^[6, 8, 10, 12, 14]
sizes = [64, 256, 1024, 4096, 16384]

# Indices for equal spacing (0, 1, 2, 3, 4)
x_indices = range(len(sizes))

# Rounding modes (column names excluding 'Size')
rounding_modes = dataframes['MATLAB chop'].columns[1:].tolist()  # ['Nearest (even)', 'Up', ...]

# Create a separate plot for each rounding mode
for mode in rounding_modes:
    plt.figure(figsize=(10, 6))
    
    # Plot each method's runtime for this rounding mode
    for method, df in dataframes.items():
        plt.plot(x_indices, df[mode], marker='o', label=method)
    
    # Customize the plot
    plt.title(f'Average Runtime vs Matrix Size\nRounding Mode: {mode}', fontsize=14)
    plt.xlabel('Matrix Size (n x n)', fontsize=12)
    plt.ylabel('Average Runtime (seconds)', fontsize=12)
    
    # Set x-ticks to be equally spaced with custom labels
    plt.xticks(x_indices, sizes)
    
    plt.grid(True, which="both", ls="--")
    plt.legend(title='Method', fontsize=10)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot as PDF with tight bounding box
    plt.savefig(f'results/runtime_comparison_{mode.replace(" ", "_")}.pdf', 
                format='pdf', 
                bbox_inches='tight')
    
    # Close the figure to free memory (optional, since we're not displaying)
    plt.close()

print("PDF plots generated for all rounding modes with tight bounding box.")