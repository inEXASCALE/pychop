import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Results file from the previous script
CSV_FILE = "quantization_results.csv"
OUTPUT_DIR = "analysis_plots"

def setup_style():
    # Set a professional academic style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (10, 6)
    })
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_accuracy_by_format(df):
    """
    Bar plot showing Accuracy for each Format, grouped by Dataset.
    Useful to see which format performs best overall.
    """
    plt.figure(figsize=(12, 6))
    # Taking the mean accuracy across rounding modes for a high-level view
    sns.barplot(data=df, x="Dataset", y="Accuracy", hue="Format", errorbar="sd", palette="viridis")
    
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
    Scatter plot: Activation SQNR vs Accuracy.
    This directly answers the reviewer: "Does numerical error correlate with performance?"
    """
    plt.figure(figsize=(10, 6))
    
    # We use different markers/colors for datasets to distinguish them
    sns.scatterplot(
        data=df, 
        x="Activation_SQNR_dB", 
        y="Accuracy", 
        hue="Dataset", 
        style="Format", 
        s=100, # dot size
        alpha=0.8
    )
    
    plt.title("Correlation: Numerical Fidelity (SQNR) vs Task Performance (Accuracy)")
    plt.xlabel("Activation Signal-to-Quantization-Noise Ratio (dB) \n(Higher is less error)")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/sqnr_vs_accuracy_correlation.png", dpi=300)
    plt.close()
    print("Generated: sqnr_vs_accuracy_correlation.png")

def plot_rounding_mode_impact(df):
    """
    Line plot to analyze how Rounding Mode affects Weight MSE.
    Specific to the reviewer's comment about analyzing weight error.
    """
    # Filter for one dataset to keep plot clean (assuming behavior is similar across datasets)
    subset = df[df['Dataset'] == df['Dataset'].unique()[0]]
    dataset_name = df['Dataset'].unique()[0]
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=subset, 
        x="Rounding_Mode", 
        y="Weight_MSE", 
        hue="Format", 
        marker="o",
        palette="magma"
    )
    
    plt.title(f"Effect of Rounding Mode on Weight Error (Dataset: {dataset_name})")
    plt.ylabel("Weight Mean Squared Error (MSE)")
    plt.xlabel("Rounding Mode Index")
    plt.xticks(sorted(df['Rounding_Mode'].unique()))
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/rounding_impact_weight_mse.png", dpi=300)
    plt.close()
    print("Generated: rounding_impact_weight_mse.png")

def plot_tradeoff_efficiency(df):
    """
    Advanced: Plotting Accuracy vs Theoretical Bit Cost.
    (Approximation: Total Bits = Exp + Sig + 1 sign bit)
    """
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
    
    # Ensure numeric types
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