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
   
    sns.scatterplot(
        data=df,
        x="Activation_SQNR_dB",
        y="Accuracy",
        hue="Dataset",
        style="Format",
        s=100,          # marker size
        alpha=0.8
    )
   
    plt.title("Correlation: Numerical Fidelity (SQNR) vs Task Performance (Accuracy)")
    plt.xlabel("Activation Signal-to-Quantization-Noise Ratio (dB) \n(Higher = less error)")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.7)
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
    plot_tradeoff_efficiency(df)
   
    print(f"\nAnalysis complete. Check the '{OUTPUT_DIR}' folder.")