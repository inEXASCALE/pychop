import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import itertools

# Results file from the previous script
CSV_FILE = "quantization_results.csv"
OUTPUT_DIR = "analysis_plots"

# Global font size scaling factor (recommended range: 0.8–1.8)
# 1.0 = original size, 1.2 = +20%, 0.9 = -10%, etc.
FONT_SIZE_FACTOR = 1.0


def setup_style():
    """Set up a clean, professional academic plotting style"""
    sns.set_theme(style="whitegrid")
    
    base_sizes = {
        'font.size':           14,     # global default font size
        'axes.titlesize':      14,     # figure title
        'axes.labelsize':      14,     # x/y axis labels
        'xtick.labelsize':     14,     # x-axis tick labels
        'ytick.labelsize':     14,     # y-axis tick labels
        'legend.fontsize':     14,     # legend text
        'legend.title_fontsize': 14,   # legend title (if present)
    }
    
    # Apply uniform scaling factor
    scaled_sizes = {k: v * FONT_SIZE_FACTOR for k, v in base_sizes.items()}
    
    plt.rcParams.update({
        **scaled_sizes,
        'figure.figsize': (10, 6),
        # Optional: make lines a bit thicker for better visibility
        'lines.linewidth': 1.8,
    })
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_accuracy_by_format(df):
    """
    Bar plot: Accuracy for each quantization Format, grouped by Dataset.
    Good for seeing which format performs best overall.
    """
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Dataset", y="Accuracy", hue="Format",
                errorbar="sd", palette="viridis")
   
    plt.title("Impact of Quantization Format on Model Accuracy")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Dataset")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Format")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/accuracy_by_format.png", dpi=300)
    plt.close()
    print("Generated: accuracy_by_format.png")


def plot_sqnr_vs_accuracy(df):
    """
    Scatter plot: Activation SQNR vs final Accuracy.
    Directly addresses reviewer question: "Does numerical error correlate with performance?"
    """
    plt.figure(figsize=(10, 6))
   
    ax = sns.scatterplot(
        data=df,
        x="Activation_SQNR_dB",
        y="Accuracy",
        hue="Dataset",
        style="Format",
        s=100,          # marker size
        alpha=0.9
    )
   
    plt.title("Correlation: Numerical Fidelity (SQNR) vs Task Performance (Accuracy)")
    plt.xlabel("Activation Signal-to-Quantization-Noise Ratio (dB) \n(Higher = less error)")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.7)
    legend = ax.legend()
    legend.get_frame().set_alpha(0.1)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/sqnr_vs_accuracy_correlation.png", dpi=300)
    plt.close()
    print("Generated: sqnr_vs_accuracy_correlation.png")


def plot_rounding_mode_impact(df):
    """
    Line plots: Effect of Rounding Mode on Weight MSE — one figure per dataset.
    Each 'Format' gets a unique combination of color + linestyle + marker.
    """
    datasets = sorted(df['Dataset'].unique())  # consistent order
    
    # Define style cycles — enough combinations for 6–8 formats
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']
    linestyles = ['-', '--', '-.', ':', (0, (3,1,1,1)), (0, (5,2))]
    
    # Cycle through combinations
    style_combinations = list(itertools.product(markers, linestyles))
    
    for i, dataset_name in enumerate(datasets, 1):
        subset = df[df['Dataset'] == dataset_name].copy()
        
        # Get unique formats in this subset
        formats = sorted(subset['Format'].unique())
        n_formats = len(formats)
        
        plt.figure(figsize=(9, 6))
        
        # Use a good categorical palette (adjust name if needed)
        palette = sns.color_palette("tab10", n_colors=max(10, n_formats))
        
        for j, fmt in enumerate(formats):
            data_fmt = subset[subset['Format'] == fmt]
            
            # Get style for this format
            marker, ls = style_combinations[j % len(style_combinations)]
            
            sns.lineplot(
                data=data_fmt,
                x="Rounding_Mode",
                y="Weight_MSE",
                color=palette[j],
                linestyle=ls,
                marker=marker,
                markersize=8,
                label=fmt,
                linewidth=2.0,
                alpha=0.95
            )
        
        plt.title(f"Effect of Rounding Mode on Weight Error\n(Dataset: {dataset_name})")
        plt.ylabel("Weight Mean Squared Error (MSE)")
        plt.xlabel("Rounding Mode Index")
        plt.xticks(sorted(df['Rounding_Mode'].unique()))
        
        # Place legend outside if many formats
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Format",
                   frameon=True, fontsize=13)
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/rounding_impact_weight_mse{i}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated: rounding_impact_weight_mse{i}.png")


def plot_rounding_mode_impact_combined(df):
    """
    Combined figure: Effect of Rounding Mode on Weight MSE across all datasets.
    - One subplot per dataset, sharing the same y-axis scale for fair comparison.
    - Single shared legend placed at the bottom center of the entire figure.
    - Each 'Format' uses a unique combination of color + linestyle + marker.
    """
    import itertools
    
    datasets = sorted(df['Dataset'].unique())
    n_datasets = len(datasets)
    if n_datasets == 0:
        print("No datasets found.")
        return

    # Style combinations
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']
    linestyles = ['-', '--', '-.', ':', (0, (3,1,1,1)), (0, (5,2))]
    style_combinations = list(itertools.product(markers, linestyles))
    
    # Layout
    ncols = min(2, n_datasets) if n_datasets <= 4 else min(3, n_datasets)
    nrows = (n_datasets + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols,4*nrows), sharex=True, sharey=True)
    
    # Make axes iterable regardless of shape
    if n_datasets == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten()
    
    all_formats = sorted(df['Format'].unique())
    palette = sns.color_palette("tab10", n_colors=max(10,len(all_formats)))
    
    handles, labels = [], []
    datasets = ['MNIST', 'FashionMNIST', 'Caltech101', 'OxfordIIITPet']
    for idx, dataset_name in enumerate(datasets):
        ax = axes_flat[idx]
        subset = df[df['Dataset']==dataset_name]
        
        for j, fmt in enumerate(all_formats):
            data_fmt = subset[subset['Format']==fmt]
            if data_fmt.empty:
                continue
            marker, ls = style_combinations[j % len(style_combinations)]
            color = palette[j % len(palette)]
            
            line, = ax.plot(
                data_fmt["Rounding_Mode"],
                data_fmt["Weight_MSE"],
                color=color,
                linestyle=ls,
                marker=marker,
                markersize=7,
                linewidth=1.8,
                alpha=0.95,
                label=fmt
            )
            
            if fmt not in labels:
                handles.append(line)
                labels.append(fmt)
        
        ax.set_title(dataset_name, fontsize=13, pad=8)
        ax.set_xlabel("Rounding Mode Index")
        ax.set_ylabel("Weight MSE")
        ax.grid(True, linestyle="--", alpha=0.6)
        unique_rm = sorted(subset['Rounding_Mode'].unique())
        ax.set_xticks(unique_rm)
        ax.tick_params(axis='both', which='major', labelsize=11)
    
    # Hide empty subplots
    for ax in axes_flat[len(datasets):]:
        ax.set_visible(False)
    
    # Shared legend with transparency
    leg = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5,-0.1),
              ncol=max(4,len(all_formats)//2), title="Quantization Format", frameon=True)
    leg.get_frame().set_alpha(0.3)
    
    # Adjust layout
    fig.subplots_adjust(bottom=0.15, top=0.95, left=0.08, right=0.98, hspace=0.3, wspace=0.25)
    
    save_path = f"{OUTPUT_DIR}/rounding_impact_weight_mse_combined.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated combined plot: rounding_impact_weight_mse_combined.png")


def plot_tradeoff_efficiency(df):
    """
    Box plot: Accuracy vs theoretical bit cost (Bits = Exp + Sig + sign bit).
    Shows bit-width efficiency across formats and datasets.
    """
    df = df.copy()
    df['Total_Bits'] = df['Exp_Bits'] + df['Sig_Bits'] + 1
   
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="Total_Bits", y="Accuracy", hue="Dataset")
   
    plt.title("Bit-width Efficiency: Bits vs Accuracy")
    plt.xlabel("Total Bits per Value")
    plt.ylabel("Accuracy (%)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/bitwidth_efficiency.png", dpi=300)
    plt.close()
    print("Generated: bitwidth_efficiency.png")


if __name__ == "__main__":
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found. Please run the experiment script first.")
        exit()
       
    print("Loading results...")
    df = pd.read_csv(CSV_FILE)
   
    # Ensure numeric columns
    numeric_cols = ["Accuracy", "Weight_MSE", "Activation_SQNR_dB", "Exp_Bits", "Sig_Bits", "Rounding_Mode"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    setup_style()
   
    print("Generating plots...")
    plot_accuracy_by_format(df)
    plot_sqnr_vs_accuracy(df)
    plot_rounding_mode_impact(df)
    plot_rounding_mode_impact_combined(df)
    plot_tradeoff_efficiency(df)
   
    print(f"\nAnalysis complete. Check the '{OUTPUT_DIR}' folder.")